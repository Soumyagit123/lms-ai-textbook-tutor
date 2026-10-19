"""
╔═══════════════════════════════════════════════════════════════════╗
║         LMS Book Ingestion — Master CLI Entry Point              ║
║   Textbook PDF → Qdrant Vector DB Ingestion Pipeline             ║
╚═══════════════════════════════════════════════════════════════════╝

Usage:
    python ingest_books.py                                    # ingest ALL books
    python ingest_books.py --standard 10                      # only Class 10
    python ingest_books.py --standard 10 --subject science    # only Class 10 Science
    python ingest_books.py --dry-run                          # preview chunks, no upload
    python ingest_books.py --info                             # show Qdrant collection stats
"""

import argparse
import sys
import time

import config
from src.pdf_mode_detector import scan_books_directory, SubjectFolder
from src.metadata_builder import load_chapters_json
from src.chapter_wise_ingestor import ingest_subject_chapter_wise
from src.fullbook_ingestor import ingest_subject_full_book
import src.qdrant_uploader as qdrant_uploader


def print_banner():
    print()
    print("=" * 65)
    print("  📚 LMS Book Ingestion Pipeline")
    print("  Textbook PDF → Qdrant Vector DB")
    print("=" * 65)
    print()


def print_scan_summary(subjects: list[SubjectFolder]):
    """Print a summary table of detected subjects and their modes."""
    print(f"  Found {len(subjects)} subject folder(s):\n")
    print(f"  {'Standard':<12} {'Subject':<20} {'Mode':<15} {'PDFs':<6}")
    print(f"  {'─' * 12} {'─' * 20} {'─' * 15} {'─' * 6}")
    for sf in subjects:
        pdf_count = len(sf.chapter_pdfs) if sf.mode == "chapter_wise" else 1
        mode_label = "📄 Chapter-Wise" if sf.mode == "chapter_wise" else "📕 Full-Book"
        print(f"  Class {sf.standard:<5} {sf.subject:<20} {mode_label:<15} {pdf_count}")
    print()


def run_info(client):
    """Display Qdrant collection stats for all detected class collections."""
    try:
        all_collections = [c.name for c in client.get_collections().collections]
        class_collections = sorted(
            [c for c in all_collections if c.startswith(config.QDRANT_COLLECTION + "_class")]
        )
        if not class_collections:
            print(f"\n  No class collections found (prefix: '{config.QDRANT_COLLECTION}_class*')")
            print(f"  Run ingestion first to create them.\n")
            return
        print()
        for cname in class_collections:
            info = qdrant_uploader.get_collection_info(client, cname)
            print(f"  Collection: {info['name']}")
            print(f"    Status:        {info['status']}")
            print(f"    Total vectors: {info['total_points']}")
            print(f"    Vector size:   {info['vector_size']}")
            print()
    except Exception as e:
        print(f"\n  [ERROR] Cannot reach Qdrant: {e}")
        print(f"  Make sure Qdrant is running at: {config.QDRANT_URL}\n")


def main():
    parser = argparse.ArgumentParser(
        description="LMS Book Ingestion — Textbook PDF to Qdrant Vector DB"
    )
    parser.add_argument(
        "--standard", type=int, default=None,
        help="Only ingest books for this class/standard (e.g. 10)"
    )
    parser.add_argument(
        "--subject", type=str, default=None,
        help="Only ingest this subject (e.g. 'science', 'mathematics')"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview chunks without embedding or uploading to Qdrant"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Show Qdrant collection stats and exit"
    )
    args = parser.parse_args()

    print_banner()

    # ── Validate config ──────────────────────────────────────────────────────
    try:
        if not args.dry_run:
            config.validate()
        else:
            # For dry run, only check that books dir exists
            import os
            if not os.path.isdir(config.BOOKS_DIR):
                raise FileNotFoundError(f"'books/' not found at: {config.BOOKS_DIR}")
    except (EnvironmentError, FileNotFoundError) as e:
        print(f"  ❌ Config Error: {e}")
        sys.exit(1)

    # ── Connect to Qdrant (skip in dry-run) ───────────────────────────────────
    client = None
    if not args.dry_run:
        try:
            client = qdrant_uploader.get_client()
            # Collections are created per-class inside the ingestion loop.
            # Just verify connectivity.
            client.get_collections()
        except Exception as e:
            print(f"\n  ❌ Cannot connect to Qdrant at {config.QDRANT_URL}")
            print(f"     Error: {e}")
            print(f"\n  💡 To run Qdrant locally with Docker:")
            print(f"     docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant\n")
            sys.exit(1)

    # ── --info mode ───────────────────────────────────────────────────────────
    if args.info:
        if client is None:
            client = qdrant_uploader.get_client()
        run_info(client)
        return

    # ── Scan books/ directory ─────────────────────────────────────────────────
    print(f"  Scanning: {config.BOOKS_DIR}\n")
    subjects = scan_books_directory(config.BOOKS_DIR)

    if not subjects:
        print("  ❌ No subject folders found in books/ directory.")
        print("  See README.md for the expected folder structure.\n")
        sys.exit(1)

    # ── Apply CLI filters ─────────────────────────────────────────────────────
    if args.standard is not None:
        subjects = [s for s in subjects if s.standard == args.standard]
    if args.subject is not None:
        subjects = [s for s in subjects if s.subject == args.subject.lower()]

    if not subjects:
        print(f"  ❌ No matching subjects found for filters: "
              f"standard={args.standard}, subject={args.subject}\n")
        sys.exit(1)

    print_scan_summary(subjects)

    # ── Ingest each subject ───────────────────────────────────────────────────
    overall_start = time.time()
    grand_total_chunks = 0
    grand_total_images = 0
    results: list[dict] = []

    for sf in subjects:
        print(f"\n{'─' * 65}")
        collection_name = config.collection_name_for_class(sf.standard)
        print(f"  🎯 Class {sf.standard} — {sf.subject.upper()} [{sf.mode}]")
        print(f"  🗂  Collection: {collection_name}")
        print(f"{'─' * 65}")

        # Ensure this class's collection exists before ingesting into it
        if client:
            qdrant_uploader.ensure_collection(client, collection_name)

        chapters_meta = load_chapters_json(sf.chapters_json_path)

        if sf.mode == "chapter_wise":
            result = ingest_subject_chapter_wise(sf, chapters_meta, client, dry_run=args.dry_run)
        else:
            result = ingest_subject_full_book(sf, chapters_meta, client, dry_run=args.dry_run)

        grand_total_chunks += result["chunks"]
        grand_total_images += result["images"]
        results.append({
            "standard": sf.standard,
            "subject":  sf.subject,
            "mode":     sf.mode,
            **result,
        })

    # ── Final Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - overall_start
    print(f"\n{'═' * 65}")
    print(f"  ✅ INGESTION COMPLETE")
    print(f"{'═' * 65}")
    print(f"  Subjects processed:  {len(results)}")
    print(f"  Total chunks:        {grand_total_chunks}")
    print(f"  Total images:        {grand_total_images}")
    print(f"  Time elapsed:        {elapsed:.1f}s")
    if not args.dry_run and client:
        print(f"  Collections created: (one per class below)")
    print(f"{'═' * 65}\n")

    # Per-subject breakdown
    print(f"  {'Standard':<12} {'Subject':<20} {'Mode':<15} {'Chunks':<10} {'Images':<8} {'Collection'}")
    print(f"  {'─' * 12} {'─' * 20} {'─' * 15} {'─' * 10} {'─' * 8} {'─' * 30}")
    for r in results:
        mode_label = "chapter_wise" if r["mode"] == "chapter_wise" else "full_book"
        coll = config.collection_name_for_class(r['standard'])
        print(f"  Class {r['standard']:<5} {r['subject']:<20} {mode_label:<15} {r['chunks']:<10} {r['images']:<8} {coll}")
    print()


if __name__ == "__main__":
    main()
