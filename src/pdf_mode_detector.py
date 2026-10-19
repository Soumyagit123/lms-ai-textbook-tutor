"""
PDF Mode Detector — determines whether a subject folder uses
Chapter-Wise mode (multiple chapter PDFs) or Full-Book mode (single PDF).
"""

import os
import re
from dataclasses import dataclass


# Pattern that matches chapter filenames: ch01, ch1, chapter1, iesc101, etc.
_CHAPTER_PATTERN = re.compile(r'^(?:(?:ch|chapter)[\s_-]*\d+|[a-zA-Z]+\d+).*\.pdf$', re.IGNORECASE)


@dataclass
class SubjectFolder:
    standard: int               # e.g. 10
    subject: str                # e.g. "science"
    folder_path: str            # absolute path to subject folder
    mode: str                   # "chapter_wise" | "full_book"
    chapter_pdfs: list[str]     # sorted list of chapter PDF filenames (chapter_wise only)
    full_book_pdf: str | None   # filename of full PDF (full_book only)
    chapters_json_path: str | None  # path to chapters.json if present


def detect_mode(subject_folder_path: str, standard: int, subject: str) -> SubjectFolder:
    """
    Scan a subject folder and return a SubjectFolder describing its mode.

    Rules:
      - If 2+ PDF files exist → chapter_wise mode (sort them naturally)
      - If exactly 1 PDF file exists → full_book
      - If 0 PDFs → raises FileNotFoundError
    """
    entries = os.listdir(subject_folder_path)
    pdf_files = [f for f in entries if f.lower().endswith('.pdf')]

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {subject_folder_path}"
        )

    chapters_json = os.path.join(subject_folder_path, "chapters.json")
    chapters_json_path = chapters_json if os.path.isfile(chapters_json) else None

    # Determine mode
    if len(pdf_files) >= 2:
        mode = "chapter_wise"
        full_book_pdf = None
        # Natural sort filenames so iesc101, iesc102 ... iesc113 or ch1 ... ch13 are properly ordered
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        chapter_pdfs = sorted(pdf_files, key=natural_sort_key)
    else:
        mode = "full_book"
        full_book_pdf = pdf_files[0]
        chapter_pdfs = []

    return SubjectFolder(
        standard=standard,
        subject=subject,
        folder_path=subject_folder_path,
        mode=mode,
        chapter_pdfs=chapter_pdfs,
        full_book_pdf=full_book_pdf,
        chapters_json_path=chapters_json_path,
    )


def scan_books_directory(books_dir: str) -> list[SubjectFolder]:
    """
    Walk the entire books/ directory tree and return a list of SubjectFolder
    objects for every standard+subject combination found.

    Expected structure:
      books/standard_{N}/{subject}/
    """
    results: list[SubjectFolder] = []

    if not os.path.isdir(books_dir):
        raise FileNotFoundError(f"books/ directory not found: {books_dir}")

    for standard_folder in sorted(os.listdir(books_dir)):
        standard_path = os.path.join(books_dir, standard_folder)
        if not os.path.isdir(standard_path):
            continue

        # Parse standard number from folder name "standard_10" → 10
        match = re.match(r'^standard_(\d+)$', standard_folder, re.IGNORECASE)
        if not match:
            print(f"  [SKIP] Unexpected folder name (expected standard_N): {standard_folder}")
            continue
        standard = int(match.group(1))

        for subject_folder in sorted(os.listdir(standard_path)):
            subject_path = os.path.join(standard_path, subject_folder)
            if not os.path.isdir(subject_path):
                continue
            subject = subject_folder.lower()

            try:
                sf = detect_mode(subject_path, standard, subject)
                results.append(sf)
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")

    return results
