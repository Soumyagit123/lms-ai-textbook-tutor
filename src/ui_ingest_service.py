"""
UI Ingestion Service Adapter
Executes textbook ingestion asynchronously in background worker threads,
tracking step-by-step progress, chunks, image extraction, and Qdrant upload status.
"""

import os
import shutil
import time
import uuid
import structlog
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import config
from src.pdf_mode_detector import SubjectFolder, detect_mode
from src.metadata_builder import load_chapters_json
from src.chapter_wise_ingestor import ingest_subject_chapter_wise
from src.fullbook_ingestor import ingest_subject_full_book
import src.qdrant_uploader as qdrant_uploader

logger = structlog.get_logger()

# In-memory job state store
INGESTION_JOBS: Dict[str, Dict[str, Any]] = {}

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    return INGESTION_JOBS.get(job_id)

def list_all_jobs() -> List[Dict[str, Any]]:
    return list(INGESTION_JOBS.values())

def get_qdrant_collections_info() -> Dict[str, Any]:
    """Retrieve all class-level collections and stats from Qdrant."""
    client = qdrant_uploader.get_client()
    try:
        all_collections = [c.name for c in client.get_collections().collections]
        class_collections = sorted(
            [c for c in all_collections if c.startswith(config.QDRANT_COLLECTION + "_class")]
        )
        stats = []
        for cname in class_collections:
            info = qdrant_uploader.get_collection_info(client, cname)
            stats.append(info)
        return {
            "success": True,
            "qdrant_url": config.QDRANT_URL,
            "collections": stats
        }
    except Exception as e:
        logger.error("error_fetching_qdrant_collections", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "collections": []
        }

def process_ingestion_job(
    job_id: str,
    standard: int,
    subject: str,
    target_folder: str,
    mode: str,
    board: str = "CBSE",
    uploaded_files: Optional[List[str]] = None
):
    """
    Worker function executed in background thread.
    Updates INGESTION_JOBS dictionary in real-time.
    """
    job = INGESTION_JOBS[job_id]
    job["status"] = "PROCESSING"
    job["progress_message"] = f"Initializing ingestion for Class {standard} - {subject.upper()}..."
    job["started_at"] = time.time()

    client = None
    try:
        client = qdrant_uploader.get_client()
        collection_name = config.collection_name_for_class(standard)
        job["collection_name"] = collection_name
        job["progress_message"] = f"Ensuring Qdrant collection '{collection_name}' exists..."
        
        qdrant_uploader.ensure_collection(client, collection_name)
        
        # Detect mode & subject folder structure
        job["progress_message"] = f"Scanning subject files and mode in {target_folder}..."
        subject_folder = detect_mode(target_folder, standard, subject)
        
        # If user explicitly uploaded a subset of files in this job, only process those files
        if uploaded_files and subject_folder.mode == "chapter_wise":
            filtered_pdfs = [f for f in subject_folder.chapter_pdfs if f in uploaded_files]
            if filtered_pdfs:
                subject_folder.chapter_pdfs = filtered_pdfs

        chapters_meta = load_chapters_json(subject_folder.chapters_json_path)
        if board:
            chapters_meta["board"] = board

        job["chapter_progress"] = {}

        def handle_chapter_progress(info: dict):
            fname = info.get("filename")
            status = info.get("status")
            ch_num = info.get("chapter_number")
            cur_idx = info.get("current_index", 0)
            total = info.get("total_files", 1)
            
            job["chapter_progress"][fname] = info
            if ch_num:
                job["chapter_progress"][str(ch_num)] = info
            
            pct = int((cur_idx / max(total, 1)) * 90)
            job["progress_percentage"] = pct
            if status == "processing":
                job["progress_message"] = f"Ingesting Chapter {ch_num} ({cur_idx}/{total}): {info.get('chapter_title', '')}..."
            elif status == "completed":
                job["progress_message"] = f"Finished Chapter {ch_num} ({cur_idx}/{total}). Extracted {info.get('chunks', 0)} chunks, {info.get('images', 0)} diagrams."
                job["chunks"] = (job.get("chunks") or 0) + info.get("chunks", 0)
                job["images"] = (job.get("images") or 0) + info.get("images", 0)

        job["progress_message"] = f"Ingesting Class {standard} {subject} using mode: {subject_folder.mode}..."
        
        if subject_folder.mode == "chapter_wise":
            result = ingest_subject_chapter_wise(
                subject_folder,
                chapters_meta,
                client,
                dry_run=False,
                on_chapter_progress=handle_chapter_progress
            )
        else:
            result = ingest_subject_full_book(subject_folder, chapters_meta, client, dry_run=False)

        job["status"] = "COMPLETED"
        job["progress_percentage"] = 100
        job["progress_message"] = f"Successfully ingested {result['chunks']} chunks and {result['images']} diagrams into {collection_name}."
        job["chunks"] = result.get("chunks", 0)
        job["images"] = result.get("images", 0)
        job["completed_at"] = time.time()
        job["elapsed_seconds"] = round(job["completed_at"] - job["started_at"], 1)

        logger.info(
            "job_completed_successfully",
            job_id=job_id,
            standard=standard,
            subject=subject,
            chunks=job["chunks"],
            images=job["images"]
        )

    except Exception as e:
        logger.error("job_failed", job_id=job_id, error=str(e))
        job["status"] = "FAILED"
        job["error"] = str(e)
        job["progress_message"] = f"Failed: {str(e)}"
        job["completed_at"] = time.time()
