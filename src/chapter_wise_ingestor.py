"""
Chapter-Wise Ingestor (PRIMARY MODE)
Handles subjects that have separate chapter PDFs (ch01_*.pdf, ch02_*.pdf, ...).
This is the preferred and most accurate ingestion mode.
"""

import os
import fitz   # PyMuPDF

import config
import src.metadata_builder as mb
import src.image_extractor as ie
import src.chunker as chunker
import src.embedder as embedder
import src.qdrant_uploader as qdrant_uploader
from src.pdf_mode_detector import SubjectFolder


def _extract_full_text(pdf_path: str) -> tuple[str, int, list[str]]:
    """
    Extract all text from a PDF using PyMuPDF.
    Returns (full_text, total_pages, pages_text).
    Works for English, Odia, Hindi, and other Unicode-encoded languages.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text("text"))
    doc.close()
    return "\n\n".join(pages_text), total_pages, pages_text


def ingest_chapter(
    *,
    chapter_pdf_filename: str,
    subject_folder: SubjectFolder,
    chapters_meta: dict,
    client,
    dry_run: bool = False,
) -> dict:
    """
    Ingest a single chapter PDF:
      1. Extract text
      2. Extract images
      3. Chunk adaptively
      4. Build payloads
      5. Embed
      6. Upload to Qdrant collection: school_textbooks_class{N}

    Returns a summary dict with chunk count and image count.
    """
    # Resolve the per-class Qdrant collection name
    collection_name = config.collection_name_for_class(subject_folder.standard)
    pdf_path = os.path.join(subject_folder.folder_path, chapter_pdf_filename)

    # ── Resolve chapter metadata ──────────────────────────────────────────────
    ch_info      = mb.resolve_chapter_info(chapter_pdf_filename, chapters_meta)
    chapter_num  = ch_info["chapter_number"]
    chapter_name = ch_info["chapter_name"]
    page_offset  = ch_info["page_offset"]
    board        = mb.build_board(chapters_meta)
    book_title   = mb.build_book_title(
        subject_folder.standard, subject_folder.subject, board, chapters_meta
    )

    print(f"\n  📖 Chapter {chapter_num:02d}: {chapter_name}")
    print(f"     File: {chapter_pdf_filename} | Page offset: {page_offset}")

    # ── Step 1: Extract Text ──────────────────────────────────────────────────
    full_text, total_pages, pages_text = _extract_full_text(pdf_path)
    if not full_text.strip():
        print(f"     [WARN] No extractable text found — PDF may be scanned. Skipping.")
        return {"chunks": 0, "images": 0}

    # ── Step 2: Extract Images ────────────────────────────────────────────────
    page_diagrams: dict[int, list[dict]] = {}
    if not dry_run:
        page_diagrams = ie.extract_all_images_from_pdf(
            pdf_path=pdf_path,
            chapter_num=chapter_num,
            standard=subject_folder.standard,
            subject=subject_folder.subject,
            page_offset=page_offset,
        )
    total_images = sum(len(d) for d in page_diagrams.values())
    print(f"     Images extracted: {total_images}")

    # ── Step 3: Chunk Text ────────────────────────────────────────────────────
    raw_chunks = chunker.chunk_page_text(
        full_text=full_text,
        subject=subject_folder.subject,
        page_diagrams=page_diagrams,
        total_pages=total_pages,
        chapter_num=chapter_num,
        page_texts=pages_text,
    )
    print(f"     Text chunks: {len(raw_chunks)}")

    # ── Step 4: Build Payloads ────────────────────────────────────────────────
    payloads: list[dict] = []
    for chunk_idx, raw in enumerate(raw_chunks):
        payload = mb.build_payload(
            standard=subject_folder.standard,
            subject=subject_folder.subject,
            pdf_mode="chapter_wise",
            source_pdf=chapter_pdf_filename,
            chapter_num=chapter_num,
            chapter_name=chapter_name,
            page_offset=page_offset,
            book_title=book_title,
            board=board,
            topic_name=chapter_name,         # chapter-level topic; can be refined later
            content_type=raw["content_type"],
            page_start=raw["page_start"] + 1,   # convert 0-indexed to 1-indexed
            page_end=raw["page_end"] + 1,
            has_diagram=raw["has_diagram"],
            has_math_formula=raw["has_math_formula"],
            diagrams=raw["diagrams"],
            content=raw["text"],
            chunk_index=chunk_idx,
        )
        payloads.append(payload)

    if dry_run:
        print(f"     [DRY RUN] Would upload {len(payloads)} chunks (skipping embed + Qdrant).")
        return {"chunks": len(payloads), "images": total_images}

    # ── Resume support: skip chapters already in Qdrant ──────────────────────
    if config.SKIP_EXISTING:
        existing = qdrant_uploader.count_chapter_chunks(
            client, collection_name, subject_folder.standard, subject_folder.subject, chapter_num
        )
        if existing > 0:
            print(f"     ⏭  Already ingested ({existing} chunks) — skipping (SKIP_EXISTING=1).")
            return {"chunks": existing, "images": total_images}

    # ── Step 5: Delete existing chunks for this chapter (safe re-ingest) ─────
    deleted = qdrant_uploader.delete_chapter_chunks(
        client,
        collection_name,
        subject_folder.standard,
        subject_folder.subject,
        chapter_num,
    )
    if deleted:
        print(f"     Deleted {deleted} existing chunks for re-ingestion.")

    # ── Step 6: Embed + Upload ────────────────────────────────────────────────
    chunks_with_vectors: list[tuple[dict, list[float]]] = []
    for payload in payloads:
        enriched_text = mb.enrich_chunk_with_banner(payload)
        vector = embedder.embed_document(enriched_text)
        chunks_with_vectors.append((payload, vector))

    uploaded = qdrant_uploader.upload_chunks(client, collection_name, chunks_with_vectors)
    print(f"     ✅ Uploaded {uploaded} chunks to '{collection_name}' in Qdrant.")

    return {"chunks": uploaded, "images": total_images}


def ingest_subject_chapter_wise(
    subject_folder: SubjectFolder,
    chapters_meta: dict,
    client,
    dry_run: bool = False,
    on_chapter_progress = None,
) -> dict:
    """
    Ingest all chapter PDFs in a subject folder (chapter_wise mode).
    Returns a summary dict.
    """
    total_chunks = 0
    total_images = 0

    for idx, chapter_pdf in enumerate(subject_folder.chapter_pdfs):
        ch_info = mb.resolve_chapter_info(chapter_pdf, chapters_meta)
        ch_num = ch_info.get("chapter_number")
        ch_title = ch_info.get("chapter_name", "")

        if on_chapter_progress:
            on_chapter_progress({
                "status": "processing",
                "filename": chapter_pdf,
                "chapter_number": ch_num,
                "chapter_title": ch_title,
                "current_index": idx + 1,
                "total_files": len(subject_folder.chapter_pdfs)
            })

        result = ingest_chapter(
            chapter_pdf_filename=chapter_pdf,
            subject_folder=subject_folder,
            chapters_meta=chapters_meta,
            client=client,
            dry_run=dry_run,
        )
        total_chunks += result["chunks"]
        total_images += result["images"]

        if on_chapter_progress:
            on_chapter_progress({
                "status": "completed",
                "filename": chapter_pdf,
                "chapter_number": ch_num,
                "chapter_title": ch_title,
                "chunks": result["chunks"],
                "images": result["images"],
                "current_index": idx + 1,
                "total_files": len(subject_folder.chapter_pdfs)
            })

    return {"chunks": total_chunks, "images": total_images}
