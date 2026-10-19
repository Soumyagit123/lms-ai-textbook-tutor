"""
Metadata Builder — constructs the Qdrant payload dict for each chunk,
combining folder-path-derived fields, chapters.json data, and content analysis.
"""

import json
import os
import re
import uuid
import config


def load_chapters_json(json_path: str | None) -> dict:
    """Load chapters.json if it exists, else return empty dict."""
    if not json_path or not os.path.isfile(json_path):
        return {}
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def get_subject_category(subject: str) -> str:
    """Map subject name to category (STEM / HUMANITIES / LANGUAGE_LIT)."""
    return config.SUBJECT_CATEGORY_MAP.get(subject.lower(), "STEM")


def parse_chapter_from_filename(filename: str) -> tuple[int, str]:
    """
    Parse chapter number and raw name from a chapter PDF filename.

    Supports:
    'ch01_real_numbers.pdf' -> (1, 'Real Numbers')
    'ch11_human_eye_colourful_world.pdf' -> (11, 'Human Eye Colourful World')
    'chapter_20_statistics.pdf' -> (20, 'Statistics')
    'iesc101.pdf' -> (1, 'Chapter 1') (NCERT standard convention: code + 100 + chapter)
    'iesc113.pdf' -> (13, 'Chapter 13')
    """
    base = os.path.splitext(filename)[0]  # remove .pdf

    # 1. Match NCERT pattern like iesc101 -> 1, lemh201 -> 1, lemh213 -> 13
    ncert_match = re.search(r'^[a-zA-Z]+\d(\d{2})$', base)
    if ncert_match:
        ch_num = int(ncert_match.group(1))
        return ch_num, f"Chapter {ch_num}"

    # 2. Match standard ch/chapter patterns: ch01_name, ch1_name, chapter_12_name, ch-15-name, ch15
    match = re.match(r'^(?:ch|chapter)[\s_-]*(\d+)(?:[\s_-]+(.*))?$', base, re.IGNORECASE)
    if not match:
        # Fallback: search for ch or chapter anywhere
        sub_match = re.search(r'(?:ch|chapter)[\s_-]*(\d+)', base, re.IGNORECASE)
        if sub_match:
            ch_num = int(sub_match.group(1))
        else:
            # Fallback: any trailing or isolated digits in filename
            num_match = re.search(r'(\d+)', base)
            ch_num = int(num_match.group(1)) if num_match else 0

        raw_name = base.replace('_', ' ').replace('-', ' ').strip().title()
        return (ch_num, raw_name or f"Chapter {ch_num}")
    
    chapter_num = int(match.group(1))
    raw_title = match.group(2)
    if raw_title:
        chapter_name = raw_title.replace('_', ' ').replace('-', ' ').strip().title()
    else:
        chapter_name = f"Chapter {chapter_num}"
    return chapter_num, chapter_name


def build_standard_label(standard: int) -> str:
    return f"Class {standard}"


def build_book_title(standard: int, subject: str, board: str, chapters_meta: dict) -> str:
    """Resolve book title from chapters.json _meta or construct a default."""
    meta = chapters_meta.get("_meta", {})
    if meta.get("book_title"):
        return meta["book_title"]
    subject_title = subject.replace('_', ' ').title()
    return f"{board} {subject_title} Class {standard}"


def build_board(chapters_meta: dict) -> str:
    meta = chapters_meta.get("_meta", {})
    return meta.get("board", config.DEFAULT_BOARD)


def build_chunk_id(standard: int, subject: str, chapter_num: int,
                   page_start: int, chunk_index: int) -> str:
    """
    Unique, reproducible chunk ID.
    Format: std{N}_{subject}_ch{NN}_p{PPP}_c{CC}
    """
    return f"std{standard}_{subject}_ch{chapter_num:02d}_p{page_start:04d}_c{chunk_index:03d}"


def resolve_chapter_info(pdf_filename: str, chapters_meta: dict) -> dict:
    """
    Return resolved chapter_number, chapter_name, and page_offset
    by combining filename parsing with chapters.json data.
    Supports both file-based keys ("ch01_xxx.pdf") and chapter-number-based keys ("1", "10", "20").
    """
    chapter_num, name_from_file = parse_chapter_from_filename(pdf_filename)

    # 1. Direct filename match
    file_entry = chapters_meta.get(pdf_filename, {})

    # 2. Keyed under 'chapters' dictionary (as sent by UI: { chapters: { "1": { title, page_offset } } })
    if not file_entry and "chapters" in chapters_meta and isinstance(chapters_meta["chapters"], dict):
        chapters_dict = chapters_meta["chapters"]
        file_entry = (
            chapters_dict.get(str(chapter_num))
            or chapters_dict.get(f"{chapter_num:02d}")
            or chapters_dict.get(pdf_filename, {})
        )

    # 3. Direct chapter number match at root level ("1": { title, ... })
    if not file_entry:
        file_entry = (
            chapters_meta.get(str(chapter_num))
            or chapters_meta.get(f"{chapter_num:02d}", {})
        )

    chapter_name = file_entry.get("chapter_name") or file_entry.get("title") or name_from_file
    page_offset  = file_entry.get("page_offset", 0)

    # If chapters_meta provided an explicit chapter number
    if "chapter_number" in file_entry and file_entry["chapter_number"]:
        try:
            chapter_num = int(file_entry["chapter_number"])
        except (ValueError, TypeError):
            pass

    return {
        "chapter_number": chapter_num,
        "chapter_name":   chapter_name,
        "page_offset":    page_offset,
    }


def build_payload(
    *,
    standard:      int,
    subject:       str,
    pdf_mode:      str,           # "chapter_wise" | "full_book"
    source_pdf:    str,           # filename of source PDF
    chapter_num:   int,
    chapter_name:  str,
    page_offset:   int,
    book_title:    str,
    board:         str,
    topic_name:    str,
    content_type:  str,           # e.g. "concept_with_diagram" | "concept" | "exercise"
    page_start:    int,           # PDF-relative page start
    page_end:      int,           # PDF-relative page end
    has_diagram:   bool,
    has_math_formula: bool,
    diagrams:      list[dict],    # list of {caption, image_url, width, height}
    content:       str,
    chunk_index:   int,
) -> dict:
    """
    Build the complete Qdrant payload dict for a single chunk.
    page_start/end are stored as original-book page numbers (PDF page + page_offset).
    """
    shift = page_offset - 1 if page_offset else 0  # offset = printed page of PDF page 1
    real_page_start = page_start + shift
    real_page_end   = page_end   + shift

    return {
        "chunk_id":         build_chunk_id(standard, subject, chapter_num, real_page_start, chunk_index),
        "pdf_mode":         pdf_mode,
        "standard":         standard,
        "standard_label":   build_standard_label(standard),
        "subject":          subject,
        "subject_category": get_subject_category(subject),
        "board":            board,
        "book_title":       book_title,
        "chapter_number":   chapter_num,
        "chapter_name":     chapter_name,
        "source_pdf":       source_pdf,
        "page_offset":      page_offset,
        "topic_name":       topic_name,
        "content_type":     content_type,
        "page_start":       real_page_start,
        "page_end":         real_page_end,
        "has_math_formula": has_math_formula,
        "has_diagram":      has_diagram,
        "diagrams":         diagrams,
        "content":          content,
    }


def build_context_banner(payload: dict) -> str:
    """
    Build the context banner prepended to each chunk before embedding.
    This helps the embedding model understand context at retrieval time.
    """
    lines = [
        "=" * 80,
        f"[CLASS/STANDARD: {payload['standard']} | SUBJECT: {payload['subject'].upper()} | CATEGORY: {payload['subject_category']}]",
        f"[BOOK: {payload['book_title']} | BOARD: {payload['board']}]",
        f"[CHAPTER {payload['chapter_number']}: {payload['chapter_name']}]",
        f"[TOPIC: {payload['topic_name']} | PAGE: {payload['page_start']}]",
        f"[PDF MODE: {payload['pdf_mode']} | SOURCE: {payload['source_pdf']}]",
    ]

    for diagram in payload.get("diagrams", []):
        lines.append(
            f"[DIAGRAM AVAILABLE: \"{diagram['caption']}\" -> Image URL: {diagram['image_url']}]"
        )

    lines.append("=" * 80)
    return "\n".join(lines)


def enrich_chunk_with_banner(payload: dict) -> str:
    """Return the bannered text to embed into the vector store."""
    banner = build_context_banner(payload)
    return f"{banner}\n{payload['content']}"
