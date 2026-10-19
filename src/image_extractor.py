"""
Image Extractor — extracts figures/diagrams from PDF pages and saves them as PNGs.

Why render instead of extracting embedded images?
  NCERT-style PDFs store (a) full-page 1-bit images with soft masks (raw extraction
  gives black/blank pictures) and (b) figures as VECTOR drawings (no image object at
  all). So we locate figure regions (vector-drawing clusters + genuine raster images)
  and RENDER each region exactly as a reader sees it (masks, fonts and labels included).
"""

import io
import os
import re
import pymupdf as fitz
from PIL import Image
import config

# Initialize Cloudinary if credentials are configured
if config.USE_CLOUDINARY:
    import cloudinary
    import cloudinary.uploader
    cloudinary.config(
        cloud_name=config.CLOUDINARY_CLOUD_NAME,
        api_key=config.CLOUDINARY_API_KEY,
        api_secret=config.CLOUDINARY_API_SECRET,
        secure=True
    )

RENDER_DPI = 170
CLUSTER_GAP = 48          # pt: drawings closer than this belong to one figure
PAD = 2                   # pt: padding around the rendered region
MIN_FIG_W, MIN_FIG_H = 50, 40   # pt: smaller regions are decoration / underlines
MAX_PAGE_FRACTION = 0.80  # regions covering more of the page are backgrounds
MIN_RASTER_PT = 90        # pt: smaller raster images are icons / logos / bullets

_CAPTION_RE = re.compile(r'^\s*(Fig(?:ure|\.)?\s*\d+(?:\.\d+)*)\s*[:\-.]?\s*(.*)$', re.IGNORECASE)


def _compress_pixmap(pix: fitz.Pixmap, max_dim: int = 1200, quality: int = 82) -> tuple[bytes, int, int]:
    """
    Compress a raw PyMuPDF Pixmap in RAM to WebP format.
    Reduces 10-30 MB raw PNGs down to ~150-300 KB without visible loss.
    Returns (compressed_bytes, final_width, final_height).
    """
    # Open PyMuPDF pixmap buffer into PIL Image
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    
    # Resize if exceeding max_dim while preserving aspect ratio
    orig_w, orig_h = img.size
    if orig_w > max_dim or orig_h > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    
    # Convert RGBA/P to RGB if needed for clean compression
    if img.mode in ("RGBA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            background.paste(img, mask=img.split()[3])
        else:
            background.paste(img)
        img = background

    out_io = io.BytesIO()
    # Save as WebP with high-efficiency lossy compression
    img.save(out_io, format="WEBP", quality=quality, method=4)
    compressed_bytes = out_io.getvalue()
    return compressed_bytes, img.width, img.height


def _image_output_dir(standard: int, subject: str) -> str:
    """Return the output directory path for extracted images."""
    out_dir = os.path.join(config.PUBLIC_ASSETS_DIR, f"standard_{standard}", subject)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


MAX_DRAWING_PARTS = 400   # pages with more vector parts are formula-heavy, skip drawing-based detection
MAX_MERGE_PASSES  = 20    # safety cap on iterative merge passes


def _merge_rects(rects: list[fitz.Rect], gap: float,
                 barriers: list[fitz.Rect] | None = None) -> list[fitz.Rect]:
    """Iteratively merge rectangles that overlap or lie within `gap` of each other.
    A merge is refused if the merged box would swallow a barrier (a figure caption),
    so two separate figures never fuse across a 'Fig. X' line."""
    barriers = barriers or []

    def _blocked(a: fitz.Rect, b: fitz.Rect) -> bool:
        u = a | b
        for bar in barriers:
            if u.contains(bar) and not a.contains(bar) and not b.contains(bar):
                return True
        return False

    rects = [fitz.Rect(r) for r in rects]
    merged = True
    passes = 0
    while merged and passes < MAX_MERGE_PASSES:
        passes += 1
        merged = False
        out: list[fitz.Rect] = []
        while rects:
            cur = rects.pop()
            grown = fitz.Rect(cur.x0 - gap, cur.y0 - gap, cur.x1 + gap, cur.y1 + gap)
            i = 0
            while i < len(rects):
                if grown.intersects(rects[i]) and not _blocked(cur, rects[i]):
                    cur |= rects[i]
                    grown = fitz.Rect(cur.x0 - gap, cur.y0 - gap, cur.x1 + gap, cur.y1 + gap)
                    rects.pop(i)
                    merged = True
                else:
                    i += 1
            out.append(cur)
        rects = out
    return rects


def _decorative_image_keys(doc: fitz.Document) -> set[tuple]:
    """
    Identify images that are NOT content: page-scan backgrounds, watermarks, logos.
      - 1-bit images with a soft mask (scan layer / watermark overlay)
      - any image (by size+colourspace) that repeats on 3+ pages
    Returns a set of (width, height, colourspace) keys to ignore.
    """
    seen_pages: dict[tuple, set[int]] = {}
    decor: set[tuple] = set()
    for pno in range(len(doc)):
        for img in doc[pno].get_images(full=True):
            xref, smask, w, h, bpc, cs = img[0], img[1], img[2], img[3], img[4], img[5]
            key = (w, h, cs)
            if smask and bpc == 1:
                decor.add(key)
            seen_pages.setdefault(key, set()).add(pno)
    for key, pages in seen_pages.items():
        if len(pages) >= 3:
            decor.add(key)
    return decor


def _candidate_rects(page: fitz.Page, decor: set[tuple],
                     decor_rects: list[fitz.Rect] | None = None) -> list[tuple[fitz.Rect, int]]:
    """Return (region, item_count) for every plausible CONTENT figure on the page."""
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height
    margin_top, margin_bottom = 45, 40   # running header / footer zones

    parts: list[fitz.Rect] = []
    glyphs: list[fitz.Rect] = []

    # Vector drawings (lines, curves, arcs) — ignore thin header/footer rules
    all_drawings = page.get_drawings()
    skip_drawings = len(all_drawings) > MAX_DRAWING_PARTS  # formula-heavy page

    if not skip_drawings:
        for d in all_drawings:
            r = fitz.Rect(d["rect"])
            if r.is_empty and r.width == 0 and r.height == 0:
                continue
            if r.y1 < margin_top or r.y0 > page_rect.height - margin_bottom:
                continue
            if r.width > page_rect.width * 0.9 and r.height < 3:   # full-width rules
                continue
            if (r.height < 3 and r.width > 150) or (r.width < 3 and r.height > 150):
                continue                                           # heading / column rules
            if d.get("fill") and not d.get("color") and r.width < 14 and r.height < 14:
                glyphs.append(r)                                   # vector text glyphs (labels)
                continue
            if any(abs(r.x0 - dr.x0) < 4 and abs(r.y0 - dr.y0) < 4 and
                   abs(r.x1 - dr.x1) < 4 and abs(r.y1 - dr.y1) < 4 for dr in (decor_rects or [])):
                continue                                           # backing rect of watermark layer
            if r.get_area() > page_area * MAX_PAGE_FRACTION:       # page border / background
                continue
            parts.append(r)

    # Genuine raster images (skip backgrounds, watermarks, logos, tiny icons)
    raster_rects: list[fitz.Rect] = []
    for img in page.get_images(full=True):
        if (img[2], img[3], img[5]) in decor:
            continue
        for r in page.get_image_rects(img[0]):
            if r.get_area() > page_area * MAX_PAGE_FRACTION:
                continue
            if r.width < MIN_RASTER_PT or r.height < MIN_RASTER_PT:
                continue
            parts.append(fitz.Rect(r))
            raster_rects.append(fitz.Rect(r))

    if not parts:
        return []

    captions = [fitz.Rect(b[:4]) for b in page.get_text("blocks")
                if _CAPTION_RE.match(b[4].strip().split("\n")[0] if b[4].strip() else "")]
    clusters = _merge_rects(parts, CLUSTER_GAP, captions)
    words = page.get_text("words")
    results: list[tuple[fitz.Rect, int]] = []
    for c in clusters:
        c &= page_rect
        if c.width < MIN_FIG_W or c.height < MIN_FIG_H:
            continue
        if c.get_area() > page_area * MAX_PAGE_FRACTION:
            continue
        has_raster = any(c.intersects(rr) for rr in raster_rects)
        count = sum(1 for p in parts if c.contains(p) or c.intersects(p))
        if count < 3 and not has_raster:   # a lone box/underline is not a figure
            continue
        # Shaded prose boxes ("A Note to the Reader", summaries): lots of words, few strokes
        if not has_raster:
            n_words = sum(1 for w in words if c.contains(fitz.Rect(w[:4])))
            if n_words > 25 and count < 10:
                continue
        # attach vector-glyph labels (A, 5.0 cm, 70°) lying just outside the strokes
        near = fitz.Rect(c.x0 - 20, c.y0 - 10, c.x1 + 20, c.y1 + 10)
        for g in glyphs:
            if g.intersects(near):
                c |= g
        results.append((c, count))

    results.sort(key=lambda t: (round(t[0].y0), t[0].x0))
    return results


def _find_caption(page: fitz.Page, region: fitz.Rect, fallback: str) -> tuple[str, fitz.Rect | None]:
    """Find a 'Fig. 6.1' / 'Figure 6.1' text block closest to the region.
    Returns (caption, caption_rect_or_None)."""
    best, best_dist, best_rect = None, 1e9, None
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        first = text.strip().split("\n")[0]
        m = _CAPTION_RE.match(first)
        if not m:
            continue
        block = fitz.Rect(x0, y0, x1, y1)
        # distance from region (caption normally sits just below / inside it)
        dist = 0 if block.intersects(region) else min(abs(y0 - region.y1), abs(region.y0 - y1))
        if dist < best_dist and dist < 45 and block.width < 260 and block.height < 40:
            label = m.group(1).strip()
            rest = m.group(2).strip()
            best, best_dist, best_rect = (f"{label}: {rest}" if rest else label), dist, block
    return (best or fallback), best_rect


def extract_images_from_page(
    doc: fitz.Document,
    page: fitz.Page,
    page_num: int,           # 0-indexed PDF page number
    chapter_num: int,
    standard: int,
    subject: str,
    page_offset: int = 0,
    decor: set[tuple] | None = None,
) -> list[dict]:
    """
    Render every figure region on a page to PNG.

    Returns a list of diagram dicts: { caption, image_url, width, height }
    """
    diagrams: list[dict] = []
    out_dir = _image_output_dir(standard, subject)
    real_page = page_num + 1 + (page_offset - 1 if page_offset else 0)

    # 1) Detect figures FIRST (delete_image leaves a placeholder that would look like an image)
    decor_rects: list[fitz.Rect] = []
    decor_xrefs: list[int] = []
    for img in page.get_images(full=True):
        if (img[2], img[3], img[5]) in (decor or set()):
            decor_rects.extend(page.get_image_rects(img[0]))
            decor_xrefs.append(img[0])
    candidates = _candidate_rects(page, decor or set(), decor_rects)

    # 2) Hide watermark / page-scan layers (in memory only — the PDF file is untouched)
    for xref in decor_xrefs:
        try:
            page.delete_image(xref)
        except Exception:
            pass

    for fig_idx, (region, _count) in enumerate(candidates):
        caption, cap_rect = _find_caption(page, region, f"Figure on page {real_page}")
        full = fitz.Rect(region)
        # Include short labels (A, B, 5.0 cm, 70°) sitting just outside the strokes
        near = fitz.Rect(region.x0 - 18, region.y0 - 14, region.x1 + 18, region.y1 + 14)
        for wx0, wy0, wx1, wy1, word, *_ in page.get_text("words"):
            w = fitz.Rect(wx0, wy0, wx1, wy1)
            if len(word) <= 4 and w.intersects(near) and not w.intersects(region):
                full |= w
        if cap_rect is not None:
            full |= cap_rect            # keep "Fig. 6.1" under the figure
        clip = fitz.Rect(full.x0 - PAD, full.y0 - PAD, full.x1 + PAD, full.y1 + PAD) & page.rect
        try:
            pix = page.get_pixmap(clip=clip, dpi=RENDER_DPI, alpha=False)
        except Exception as e:
            print(f"    [WARN] Could not render figure on page {real_page}: {e}")
            continue

        if pix.width < config.MIN_IMAGE_SIZE or pix.height < config.MIN_IMAGE_SIZE // 2:
            continue

        # In-memory compression from 10-30MB raw PNG to ~150-250KB WebP
        compressed_bytes, comp_width, comp_height = _compress_pixmap(
            pix,
            max_dim=config.MAX_IMAGE_DIM,
            quality=config.IMAGE_QUALITY
        )

        base_id = f"ch{chapter_num:02d}_p{real_page:04d}_fig{fig_idx + 1:02d}"
        image_url = None

        # 1. Cloudinary upload if configured
        if config.USE_CLOUDINARY:
            try:
                folder_path = f"school_textbooks/standard_{standard}/{subject}"
                upload_res = cloudinary.uploader.upload(
                    compressed_bytes,
                    folder=folder_path,
                    public_id=base_id,
                    format="webp",
                    resource_type="image",
                    overwrite=True
                )
                image_url = upload_res.get("secure_url") or upload_res.get("url")
            except Exception as e:
                print(f"    [WARN] Cloudinary upload failed for {base_id}: {e}. Falling back to local storage.")

        # 2. Local fallback / cache
        if not image_url:
            img_filename = f"{base_id}.webp"
            with open(os.path.join(out_dir, img_filename), "wb") as f:
                f.write(compressed_bytes)
            image_url = f"/book-assets/standard_{standard}/{subject}/{img_filename}"

        diagrams.append({
            "caption":   caption,
            "image_url": image_url,
            "width":     comp_width,
            "height":    comp_height,
        })

    return diagrams


def extract_all_images_from_pdf(
    pdf_path: str,
    chapter_num: int,
    standard: int,
    subject: str,
    page_offset: int = 0,
) -> dict[int, list[dict]]:
    """
    Process all pages of a PDF and return a dict mapping
    page_number (0-indexed) → list of diagram dicts.
    """
    page_diagrams: dict[int, list[dict]] = {}
    doc = fitz.open(pdf_path)
    decor = _decorative_image_keys(doc)

    for page_num in range(len(doc)):
        page = doc[page_num]
        diagrams = extract_images_from_page(
            doc, page, page_num, chapter_num, standard, subject, page_offset, decor
        )
        if diagrams:
            page_diagrams[page_num] = diagrams

    doc.close()
    return page_diagrams
