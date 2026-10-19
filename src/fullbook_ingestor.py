"""
Full-Book Ingestor (FALLBACK MODE)
Handles subjects that have a single full-book PDF.
Uses regex-based chapter boundary detection to assign chapter metadata.
Less precise than chapter_wise mode — use only when chapter PDFs are unavailable.
"""

import os
import re
import fitz   # PyMuPDF

import config
import src.metadata_builder as mb
import src.image_extractor as ie
import src.chunker as chunker
import src.embedder as embedder
import src.qdrant_uploader as qdrant_uploader
from src.pdf_mode_detector import SubjectFolder


# Regex to detect chapter headings in full-book PDF text
_CHAPTER_HEADING_PATTERNS = [
    r'^\s*Chapter\s+(\d+)\s*[:\-–—]?\s*(.+)',
    r'^\s*CHAPTER\s+(\d+)\s*[:\-–—]?\s*(.+)',
    r'^\s*(\d+)\.\s+([A-Z][A-Za-z\s]{5,60})',
]


def _detect_chapters_from_text(full_text: str) -> list[dict]:
    """
    Detect chapter boundaries in the full book text.
    Returns a list of { chapter_num, chapter_name, char_offset } dicts, sorted by offset.
    Falls back to a single virtual chapter if detection fails.
    """
    chapters: list[dict] = []
    lines = full_text.split('\n')
    char_offset = 0

    for line in lines:
        for pattern in _CHAPTER_HEADING_PATTERNS:
            match = re.match(pattern, line)
            if match:
                try:
                    ch_num  = int(match.group(1))
                    ch_name = match.group(2).strip().title()
                    chapters.append({
                        "chapter_num":  ch_num,
                        "chapter_name": ch_name,
                        "char_offset":  char_offset,
                    })
                except (IndexError, ValueError):
                    pass
                break
        char_offset += len(line) + 1  # +1 for the \n

    # Deduplicate by chapter number (keep first occurrence)
    seen = set()
    unique = []
    for ch in chapters:
        if ch["chapter_num"] not in seen:
            seen.add(ch["chapter_num"])
            unique.append(ch)

    if not unique:
        # No chapters detected — treat the entire book as Chapter 0
        unique = [{"chapter_num": 0, "chapter_name": "Full Book", "char_offset": 0}]

    return unique


def _split_text_by_chapters(full_text: str, chapters: list[dict]) -> list[dict]:
    """
    Split the full book text into sections per detected chapter.
    Returns a list of { chapter_num, chapter_name, text } dicts.
    """
    sections = []
    for i, ch in enumerate(chapters):
        start = ch["char_offset"]
        end   = chapters[i + 1]["char_offset"] if i + 1 < len(chapters) else len(full_text)
        text  = full_text[start:end].strip()
        if text:
            sections.append({
                "chapter_num":  ch["chapter_num"],
                "chapter_name": ch["chapter_name"],
                "text":         text,
            })
    return sections


def ingest_subject_full_book(
    subject_folder: SubjectFolder,
    chapters_meta: dict,
    client,
    dry_run: bool = False,
) -> dict:
    """
    Ingest a single full-book PDF (fallback mode).
    Returns a summary dict with chunk count and image count.
    """
    pdf_path = os.path.join(subject_folder.folder_path, subject_folder.full_book_pdf)
    board      = mb.build_board(chapters_meta)
    book_title = mb.build_book_title(
        subject_folder.standard, subject_folder.subject, board, chapters_meta
    )
    # Resolve the per-class Qdrant collection name
    collection_name = config.collection_name_for_class(subject_folder.standard)

    print(f"\n  📚 Full-Book Mode: {subject_folder.full_book_pdf}")
    print(f"     ⚠️  Chapter-wise PDFs preferred for better accuracy.")

    # ── Step 1: Extract full text ─────────────────────────────────────────────
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    full_text = "\n\n".join(page.get_text("text") for page in doc)
    doc.close()

    if not full_text.strip():
        print(f"     [WARN] No extractable text — PDF may be scanned. Skipping.")
        return {"chunks": 0, "images": 0}

    # ── Step 2: Detect chapter boundaries ────────────────────────────────────
    chapters = _detect_chapters_from_text(full_text)
    print(f"     Detected {len(chapters)} chapters in full-book PDF.")
    sections  = _split_text_by_chapters(full_text, chapters)

    # ── Step 3: Extract all images at book level ──────────────────────────────
    all_page_diagrams: dict[int, list[dict]] = {}
    if not dry_run:
        all_page_diagrams = ie.extract_all_images_from_pdf(
            pdf_path=pdf_path,
            chapter_num=0,
            standard=subject_folder.standard,
            subject=subject_folder.subject,
            page_offset=0,
        )
    total_images = sum(len(d) for d in all_page_diagrams.values())
    print(f"     Images extracted: {total_images}")

    # ── Step 4: Chunk each section and upload ─────────────────────────────────
    total_chunks = 0

    for section in sections:
        ch_num  = section["chapter_num"]
        ch_name = section["chapter_name"]
        text    = section["text"]

        raw_chunks = chunker.chunk_page_text(
            full_text=text,
            subject=subject_folder.subject,
            page_diagrams=all_page_diagrams,
            total_pages=total_pages,
            chapter_num=ch_num,
        )

        payloads: list[dict] = []
        for chunk_idx, raw in enumerate(raw_chunks):
            payload = mb.build_payload(
                standard=subject_folder.standard,
                subject=subject_folder.subject,
                pdf_mode="full_book",
                source_pdf=subject_folder.full_book_pdf,
                chapter_num=ch_num,
                chapter_name=ch_name,
                page_offset=0,
                book_title=book_title,
                board=board,
                topic_name=ch_name,
                content_type=raw["content_type"],
                page_start=raw["page_start"] + 1,
                page_end=raw["page_end"] + 1,
                has_diagram=raw["has_diagram"],
                has_math_formula=raw["has_math_formula"],
                diagrams=raw["diagrams"],
                content=raw["text"],
                chunk_index=chunk_idx,
            )
            payloads.append(payload)

        if dry_run:
            print(f"     [DRY RUN] Chapter {ch_num}: {len(payloads)} chunks (skip upload).")
            total_chunks += len(payloads)
            continue

        # Delete existing for re-ingest safety
        qdrant_uploader.delete_chapter_chunks(
            client, collection_name, subject_folder.standard, subject_folder.subject, ch_num
        )

        chunks_with_vectors: list[tuple[dict, list[float]]] = []
        for payload in payloads:
            enriched_text = mb.enrich_chunk_with_banner(payload)
            vector = embedder.embed_document(enriched_text)
            chunks_with_vectors.append((payload, vector))

        uploaded = qdrant_uploader.upload_chunks(client, collection_name, chunks_with_vectors)
        print(f"     ✅ Chapter {ch_num} ({ch_name}): {uploaded} chunks → '{collection_name}'.")
        total_chunks += uploaded

    return {"chunks": total_chunks, "images": total_images}
