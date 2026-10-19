import os
import json
import time
import uuid
import structlog
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from rag_graph.workflow import student_rag_graph, retrieval_rag_graph
from rag_services.llm import student_llm_service
from src.ui_ingest_service import (
    process_ingestion_job,
    get_job_status,
    list_all_jobs,
    get_qdrant_collections_info,
    INGESTION_JOBS
)

logger = structlog.get_logger()

app = FastAPI(
    title="LMS Student Textbook LangGraph RAG Agent",
    description="Multi-class, multi-subject student academic chatbot with dynamic collection routing and robust error handling.",
    version="2.0.0"
)

# Enable CORS for frontend or local analysis agent
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve extracted textbook images statically at /book-assets
os.makedirs(config.PUBLIC_ASSETS_DIR, exist_ok=True)
app.mount("/book-assets", StaticFiles(directory=config.PUBLIC_ASSETS_DIR), name="book-assets")

class StudentChatRequest(BaseModel):
    query: str = Field(..., description="The student's question", min_length=2)
    student_id: Optional[str] = Field(None, description="Optional student ID from UI auth")
    student_name: Optional[str] = Field(None, description="Student display name")
    standard: int = Field(10, description="Grade/Standard e.g. 9 or 10")
    subject: Optional[str] = Field(None, description="Optional subject e.g. mathematics, science")
    chapter_number: Optional[int] = Field(None, description="Optional chapter number")
    top_k: int = Field(12, description="Candidate retrieval count")
    top_n: int = Field(4, description="Top reranked documents to feed LLM context")

class Citation(BaseModel):
    book_title: str = "Textbook"
    chapter_number: Optional[int] = None
    chapter_name: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    topic_name: Optional[str] = None

class StudentChatResponse(BaseModel):
    student_name: Optional[str] = None
    standard: int
    subject_detected: Optional[str] = None
    collection_used: Optional[str] = None
    collection_exists: bool = True
    answer: str
    is_safe: bool = True
    has_error: bool = False
    error_message: Optional[str] = None
    citations: List[Citation] = []
    diagrams: List[Dict[str, Any]] = []

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "lms-langgraph-rag"}

@app.post("/api/rag/chat", response_model=StudentChatResponse)
async def chat_with_agent(request: StudentChatRequest):
    """
    Standard synchronous endpoint using the LangGraph orchestrator:
    1. Input Guardrails check
    2. Student Context & Subject sensing
    3. Qdrant Collection validation (checks if class collection exists)
    4. Dense Retrieval & Reranking
    5. Gemini Student Generation with LaTeX & Diagrams
    6. Non-crashing error handling
    """
    logger.info(
        "received_chat_request",
        student=request.student_name,
        standard=request.standard,
        subject=request.subject,
        query=request.query
    )

    initial_state = {
        "query": request.query,
        "student_id": request.student_id,
        "student_name": request.student_name,
        "standard": request.standard,
        "subject": request.subject,
        "chapter_number": request.chapter_number,
        "top_k": request.top_k,
        "top_n": request.top_n,
        "errors": []
    }

    try:
        final_state = await student_rag_graph.ainvoke(initial_state)

        return StudentChatResponse(
            student_name=final_state.get("student_name"),
            standard=final_state.get("standard", request.standard),
            subject_detected=final_state.get("detected_subject"),
            collection_used=final_state.get("collection_name"),
            collection_exists=final_state.get("collection_exists", True),
            answer=final_state.get("final_answer") or "No response could be generated.",
            is_safe=final_state.get("is_safe", True),
            has_error=final_state.get("has_error", False),
            error_message=final_state.get("error_message"),
            citations=[Citation(**c) for c in final_state.get("citations", [])],
            diagrams=final_state.get("diagrams", [])
        )

    except Exception as e:
        logger.error("graph_invocation_unhandled_exception", error=str(e))
        return StudentChatResponse(
            student_name=request.student_name,
            standard=request.standard,
            subject_detected=request.subject,
            collection_used=f"school_textbooks_class{request.standard}",
            collection_exists=False,
            answer=f"⚠️ System notice: An unexpected issue occurred while processing your question: {str(e)[:120]}",
            is_safe=True,
            has_error=True,
            error_message=str(e)
        )

@app.post("/api/rag/chat/stream")
async def chat_with_agent_stream(request: StudentChatRequest):
    """
    Streaming SSE endpoint:
    1. Runs guardrails, context resolver, validator, and Qdrant retrieval via retrieval_rag_graph
    2. Sends metadata (citations, diagrams, collection info) immediately
    3. Streams tokens in real-time as they are produced by the LLM
    """
    initial_state = {
        "query": request.query,
        "student_id": request.student_id,
        "student_name": request.student_name,
        "standard": request.standard,
        "subject": request.subject,
        "chapter_number": request.chapter_number,
        "top_k": request.top_k,
        "top_n": request.top_n,
        "errors": []
    }

    async def event_generator():
        try:
            # 1. Execute retrieval pipeline (fast, ~1-2 seconds)
            state = await retrieval_rag_graph.ainvoke(initial_state)

            is_safe = state.get("is_safe", True)
            collection_exists = state.get("collection_exists", True)

            # Send metadata immediately to client
            meta_payload = {
                "type": "metadata",
                "student_name": state.get("student_name"),
                "standard": state.get("standard"),
                "subject": state.get("detected_subject"),
                "collection": state.get("collection_name"),
                "collection_exists": collection_exists,
                "citations": state.get("citations", []),
                "diagrams": state.get("diagrams", []),
                "is_safe": is_safe
            }
            yield f"data: {json.dumps(meta_payload)}\n\n"

            # If blocked by safety or missing collection, yield notification and exit
            if not is_safe:
                blocked_msg = state.get("final_answer") or "⚠️ Request was blocked by safety policy."
                yield f"data: {json.dumps({'chunk': blocked_msg, 'done': True, 'is_safe': False})}\n\n"
                return

            if not collection_exists:
                notice = f"⚠️ Notice: Textbook collection '{state.get('collection_name')}' is not indexed yet in Qdrant."
                yield f"data: {json.dumps({'chunk': notice, 'done': True, 'is_safe': True})}\n\n"
                return

            # 2. Check if relevant docs were found
            docs = state.get("retrieved_docs", [])
            query = state.get("query", "")
            standard = state.get("standard", 10)
            student_name = state.get("student_name")
            subject = state.get("detected_subject")

            if not docs:
                subj_str = f" in {subject.title()}" if subject else ""
                no_info_msg = (
                    f"Hello{f' {student_name}' if student_name else ''}! "
                    f"I could not find matching information in your Class {standard} textbook{subj_str} for this specific question. "
                    "This topic is not covered in the currently indexed chapters. Please try asking a question from your enrolled chapters."
                )
                yield f"data: {json.dumps({'chunk': no_info_msg, 'done': True, 'is_safe': True})}\n\n"
                return

            # Stream tokens directly from student_llm_service
            async for token in student_llm_service.generate_response_stream(query, docs, standard):
                yield f"data: {json.dumps({'chunk': token, 'done': False, 'is_safe': True})}\n\n"

            # Final done event
            yield f"data: {json.dumps({'chunk': '', 'done': True, 'is_safe': True})}\n\n"

        except Exception as e:
            logger.error("stream_error", error=str(e))
            yield f"data: {json.dumps({'chunk': f'⚠️ Error: {str(e)}', 'done': True, 'is_safe': False})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
# ==============================================================================
# UI BOOK INGESTION API ENDPOINTS (For Admin UI & Qdrant Management)
# ==============================================================================

@app.get("/api/ingest/collections")
async def get_collections():
    """Returns all class-segregated Qdrant collections and vector statistics."""
    return get_qdrant_collections_info()

@app.get("/api/ingest/jobs")
async def get_ingestion_jobs():
    """Lists all active and completed ingestion tasks."""
    return {"jobs": list_all_jobs()}

@app.get("/api/ingest/status/{job_id}")
async def get_single_job_status(job_id: str):
    """Poll the real-time progress of a specific ingestion task."""
    status = get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return status

@app.post("/api/ingest/upload")
async def upload_and_ingest(
    background_tasks: BackgroundTasks,
    standard: int = Form(..., ge=1, le=12, description="Class / Grade (1 to 12)"),
    subject: str = Form(..., description="Subject name e.g. Mathematics, Science"),
    board: str = Form("CBSE", description="Educational board e.g. CBSE, NCERT"),
    mode: str = Form("chapter_wise", description="chapter_wise or full_book"),
    chapters_json: Optional[str] = Form(None, description="Optional custom chapters JSON mapping"),
    files: List[UploadFile] = File(..., description="One or more PDF chapter files")
):
    """
    Accepts uploaded textbook PDFs from the Admin UI, places them into the
    corresponding standard/subject books folder, and starts background ingestion.
    """
    clean_subject = subject.lower().strip().replace(" ", "_")
    target_dir = os.path.join(config.BOOKS_DIR, f"standard_{standard}", clean_subject)
    os.makedirs(target_dir, exist_ok=True)

    saved_filenames = []
    for uploaded in files:
        if not uploaded.filename.lower().endswith(".pdf"):
            continue
        dest_path = os.path.join(target_dir, uploaded.filename)
        with open(dest_path, "wb") as f:
            content = await uploaded.read()
            f.write(content)
        saved_filenames.append(uploaded.filename)

    if not saved_filenames:
        raise HTTPException(status_code=400, detail="No valid PDF files were uploaded.")

    # Save chapters.json if provided
    if chapters_json:
        try:
            parsed = json.loads(chapters_json)
            with open(os.path.join(target_dir, "chapters.json"), "w", encoding="utf-8") as jf:
                json.dump(parsed, jf, indent=2)
        except Exception as e:
            logger.warn("failed_parsing_chapters_json", error=str(e))

    # Initialize job in state store
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    INGESTION_JOBS[job_id] = {
        "job_id": job_id,
        "standard": standard,
        "subject": clean_subject,
        "board": board,
        "mode": mode,
        "files_count": len(saved_filenames),
        "files": saved_filenames,
        "status": "QUEUED",
        "progress_percentage": 0,
        "progress_message": "Queued for processing...",
        "created_at": time.time(),
        "chunks": 0,
        "images": 0,
        "error": None
    }

    # Dispatch to background task worker
    background_tasks.add_task(
        process_ingestion_job,
        job_id=job_id,
        standard=standard,
        subject=clean_subject,
        target_folder=target_dir,
        mode=mode,
        board=board,
        uploaded_files=saved_filenames
    )

    return {
        "success": True,
        "message": f"Ingestion job started for Class {standard} {subject}",
        "job_id": job_id,
        "files_uploaded": len(saved_filenames)
    }

if __name__ == "__main__":
    import uvicorn
    import config
    uvicorn.run("rag_server:app", host=config.RAG_HOST, port=config.RAG_PORT, reload=True)
