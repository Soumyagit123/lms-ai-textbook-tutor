"""
Subject-Adaptive Chunker — splits extracted PDF text into chunks
using the correct strategy for each subject category (STEM / HUMANITIES / LANGUAGE_LIT).
"""

import re
import config
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _get_splitter(subject: str) -> RecursiveCharacterTextSplitter:
    """Return a text splitter configured for the subject's category."""
    category = config.SUBJECT_CATEGORY_MAP.get(subject.lower(), "STEM")
    params = config.CHUNK_PARAMS[category]

    # Custom separators — tried in order until a clean split is found.
    # Prioritise paragraph and sentence breaks over mid-sentence splits.
    separators = [
        "\n\n\n",       # triple newline (chapter/section break)
        "\n\n",         # double newline (paragraph break)
        "\n",           # single newline
        ". ",           # sentence end
        "। ",           # Devanagari / Odia danda sentence end
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
        "",
    ]

    return RecursiveCharacterTextSplitter(
        chunk_size=params["chunk_size"],
        chunk_overlap=params["chunk_overlap"],
        separators=separators,
        length_function=len,
    )


def _detect_has_math(text: str) -> bool:
    """Heuristic: detect if a chunk likely contains math formulas."""
    math_indicators = [
        r'\$[^$]+\$',           # inline LaTeX $...$
        r'\$\$[^$]+\$\$',       # block LaTeX $$...$$
        r'\\frac', r'\\sqrt', r'\\sum', r'\\int',
        r'=\s*\d',              # equation like "x = 5"
        r'\d+\s*[\+\-\×\÷\*\/]\s*\d+',  # arithmetic
    ]
    return any(re.search(p, text) for p in math_indicators)


def _classify_content_type(text: str, has_diagram: bool, has_math: bool) -> str:
    """
    Classify chunk content into a content_type label for Qdrant payload.
    """
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["exercise", "q.", "question", "solve", "find the", "calculate"]):
        return "exercise"
    if has_diagram and has_math:
        return "concept_with_diagram_and_formula"
    if has_diagram:
        return "concept_with_diagram"
    if has_math:
        return "concept_with_formula"
    if any(kw in text_lower for kw in ["example", "solution:", "step 1", "proof:"]):
        return "solved_example"
    return "concept"


def chunk_page_text(
    full_text: str,
    subject: str,
    page_diagrams: dict[int, list[dict]],
    total_pages: int,
    chapter_num: int,
    page_texts: list[str] | None = None,
) -> list[dict]:
    """
    Split a full chapter's text into chunks with metadata annotations.

    Args:
        full_text:    Complete extracted text from the PDF (all pages concatenated).
        subject:      Subject name (used to select chunking strategy).
        page_diagrams: Dict of page_num → list of diagram dicts (from image extractor).
        total_pages:  Total number of pages in the PDF.
        chapter_num:  Chapter number (for diagram association).

    Returns:
        List of chunk dicts with keys:
          text, page_start, page_end, has_diagram, has_math_formula,
          content_type, diagrams
    """
    splitter = _get_splitter(subject)
    raw_chunks = splitter.split_text(full_text)

    # Collect all diagrams from all pages into a flat list for association
    all_diagrams: list[dict] = []
    for page_num in sorted(page_diagrams.keys()):
        all_diagrams.extend(page_diagrams[page_num])

    # We distribute diagrams across chunks roughly:
    # — if a diagram exists on the page, attach it to the chunk that contains that content.
    # For simplicity, we round-robin assign diagrams to early chunks that mention figures.
    diagram_iter = iter(all_diagrams)

    chunks: list[dict] = []
    chars_per_page = max(len(full_text) / max(total_pages, 1), 1)

    # Exact page spans (char ranges) when per-page text is supplied.
    # full_text MUST equal "\n\n".join(page_texts) for the spans to line up.
    page_ends: list[int] = []
    if page_texts:
        pos = 0
        for t in page_texts:
            pos += len(t)
            page_ends.append(pos)
            pos += 2  # the "\n\n" joiner

    def _page_of(char_idx: int) -> int:
        for p, end in enumerate(page_ends):
            if char_idx <= end:
                return p
        return len(page_ends) - 1

    cursor = 0
    for chunk_idx, chunk_text in enumerate(raw_chunks):
        if page_ends:
            start_char = full_text.find(chunk_text, cursor)
            if start_char < 0:
                start_char = full_text.find(chunk_text)
            start_char = max(start_char, 0)
            cursor = start_char + 1  # starts are monotonic; overlap handled by +1
            page_start = _page_of(start_char)
            page_end = _page_of(start_char + max(len(chunk_text) - 1, 0))
        else:
            char_offset = full_text.find(chunk_text[:50]) if chunk_text else 0
            page_start = page_end = min(int(char_offset / chars_per_page), total_pages - 1)

        # Diagrams on every page the chunk spans
        chunk_diagrams: list[dict] = []
        for p in range(page_start, page_end + 1):
            chunk_diagrams.extend(page_diagrams.get(p, []))

        has_diagram    = len(chunk_diagrams) > 0
        has_math       = _detect_has_math(chunk_text)
        content_type   = _classify_content_type(chunk_text, has_diagram, has_math)

        chunks.append({
            "text":             chunk_text,
            "page_start":       page_start,       # PDF-relative (0-indexed internally, caller adds +1)
            "page_end":         page_end,
            "has_diagram":      has_diagram,
            "has_math_formula": has_math,
            "content_type":     content_type,
            "diagrams":         chunk_diagrams,
        })

    return chunks
