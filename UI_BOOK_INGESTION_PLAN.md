# Implementation Plan: UI-Based Textbook Ingestion to Qdrant Vector DB

## 1. Executive Summary
Currently, `lms-book-ingestion` runs offline CLI scripts (`python ingest_books.py`) against static folders (`books/standard_{N}/{subject}/`) to chunk, extract images, embed, and store textbook vectors in class-specific collections (`school_textbooks_class{N}`).

This plan details the end-to-end architecture and implementation for an **Admin UI Page** in the Angular LMS application (`d:\signal-angular\lms`), supported by dedicated API endpoints in the Python ingestion service (`d:\signal-angular\lms-book-ingestion`), allowing administrators to upload, configure, and ingest textbooks into class-segregated Qdrant vector collections with live progress tracking.

---

## 2. Architecture & Data Flow

```
[ Angular Admin UI (/admin/book-ingestion) ]
    │
    │  1. Select Class/Standard, Subject, Board, Mode (Chapter-Wise / Full-Book)
    │  2. Upload chapter PDFs (or single book PDF) & configure chapter metadata
    │  3. Click "Ingest to Qdrant" (POST /api/ingest/upload)
    ▼
[ FastAPI Ingestion Service (rag_server.py:8001) ]
    │
    │  4. Saves uploaded PDFs & generates temporary SubjectFolder + chapters.json
    │  5. Launches async background ingestion job (Job ID returned)
    ▼
[ Ingestion Pipeline Worker ]
    │
    ├─ PyMuPDF (fitz): Extracts text & extracts diagrams/figures
    │                   (Saves images to public/book-assets/standard_{N}/{subject}/)
    ├─ Chunker: Adaptive token & heading chunker (500–800 tokens)
    ├─ Embedder: Matryoshka 768-dim embeddings via Google Generative AI
    └─ Qdrant Uploader:
           - Verifies/creates class-isolated collection: `school_textbooks_class{N}`
           - Sets up payload indexes (`standard`, `subject`, `chapter_number`)
           - Upserts vectors & payloads
    │
    ▼ (Server-Sent Events / SSE or Polling on /api/ingest/status/{jobId})
[ Admin UI Progress Modal ]
    - Live step progress: [Text Extraction ➔ Image Extraction ➔ Chunking ➔ Embedding ➔ Qdrant Upload]
    - Success summary: Vectors uploaded, images extracted, collection name
```

---

## 3. Detailed Component Plan

### A. Python Backend (`lms-book-ingestion`)

1. **FastAPI Endpoints in `rag_server.py`**:
   - `GET /api/ingest/collections`:
     - Queries Qdrant to list all class collections (`school_textbooks_class*`), returning vector counts, status, and indexed subjects.
   - `POST /api/ingest/upload`:
     - Accepts `multipart/form-data`:
       - `standard`: `int` (e.g. `9`, `10`)
       - `subject`: `str` (e.g. `mathematics`, `science`)
       - `board`: `str` (e.g. `CBSE`, `NCERT`, `State Board`)
       - `mode`: `"chapter_wise"` or `"full_book"`
       - `chapters_meta`: JSON string containing chapter mapping (`chapter_number`, `chapter_name`, `page_offset`)
       - `files`: List of uploaded PDF files
     - Saves files into temporary working directory or directly into `books/standard_{standard}/{subject}/`.
     - Spawns background ingestion task with a unique `job_id`.
   - `GET /api/ingest/status/{job_id}`:
     - Returns live job state: `status` (`QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`), current chapter, total chunks, images saved, percentage, and error log.
   - `DELETE /api/ingest/collection/{standard}/{subject}`:
     - Allows clearing points for a specific subject within a class collection or resetting an entire class collection.

2. **Ingestion Service Adapter (`src/ui_ingest_service.py`)**:
   - Reusable programmatic wrapper around `ingest_subject_chapter_wise` and `ingest_subject_full_book` that emits status callbacks instead of printing only to stdout.

---

### B. Angular Admin UI (`lms`)

1. **Route & Navigation**:
   - **Route**: `/admin/book-ingestion`
   - **Sidebar Link**: Add under **AI Platform & Materials** or **Academic Infrastructure** in [admin-layout.component.html](file:///d:/signal-angular/lms/src/app/shared/layouts/admin-layout/admin-layout.component.html):
     ```html
     <a routerLink="/admin/book-ingestion" routerLinkActive="active" class="sidebar-link">
       <i class="bi bi-database-fill-up"></i>Book Ingestion (Qdrant)
     </a>
     ```

2. **UI Component Architecture (`BookIngestionComponent`)**:
   - **Tab 1: Ingest New Textbook**:
     - **Academic Scope Selector**:
       - Standard / Class (Class 6 through 12 dropdown)
       - Subject (Mathematics, Science, Social Science, English, etc.)
       - Board (CBSE, NCERT, ICSE, State Board)
     - **Mode Selection Toggle**:
       - *Chapter-Wise Mode* (Recommended for NCERT): Multi-file PDF uploader (`ch01.pdf`, `ch02.pdf`, etc.)
       - *Full-Book Mode*: Single PDF uploader
     - **Visual Chapter Metadata Editor**:
       - Auto-detects chapter numbers from filenames (`ch01_real_numbers.pdf` → Chapter 1).
       - Editable table: Chapter Number, Chapter Title, Page Offset in PDF.
     - **Primary Action Button**:
       - `<button class="btn btn-primary btn-lg"><i class="bi bi-cloud-arrow-up-fill me-2"></i>Ingest Books to Qdrant</button>`
   - **Tab 2: Qdrant Vector Collections Dashboard**:
     - Visual cards for each active class collection (`school_textbooks_class9`, `school_textbooks_class10`).
     - Vector count, status badge (`GREEN / HEALTHY`), vector dimension (`768-dim Matryoshka`).
     - "Test Retrieval" inline quick-query tool to test chatbot citations immediately after ingestion.

3. **Real-time Ingestion Modal**:
   - Step indicator:
     1. Text & Unicode Extraction
     2. Diagram & Image Extraction
     3. Chunking & Association
     4. Google AI 768-dim Vector Embedding
     5. Qdrant Upsert to `school_textbooks_class{N}`
   - Live stream of logs and progress bar.

---

## 4. Class-Specific Collection Isolation (Verified)

| Grade / Class | Target Qdrant Collection | Isolated Retrieval Scope |
| :--- | :--- | :--- |
| **Class 9** | `school_textbooks_class9` | Only Class 9 students & chatbot queries |
| **Class 10** | `school_textbooks_class10` | Only Class 10 students & chatbot queries |
| **Class N** | `school_textbooks_class{N}` | Automated dynamic collection provisioning |

---

## 5. Implementation Phases & Milestones

### Phase 1: Ingestion API Service (Python FastAPI)
- [ ] Add `src/ui_ingest_service.py` to support programmatic, non-blocking ingestion.
- [ ] Implement `POST /api/ingest/upload`, `GET /api/ingest/status/{job_id}`, and `GET /api/ingest/collections` in `rag_server.py`.
- [ ] Test endpoints using `curl` / Postman with Class 9 sample PDF chapters.

### Phase 2: Angular Service & State (`lms`)
- [ ] Create `BookIngestionService` in `src/app/core/services/book-ingestion.service.ts` connecting to `http://localhost:8001/api/ingest`.
- [ ] Define TypeScript interfaces for `IngestJobStatus`, `QdrantCollectionInfo`, `ChapterMeta`.

### Phase 3: Angular Admin Ingestion Page (`lms`)
- [ ] Create `BookIngestionComponent` in `src/app/features/admin/book-ingestion/`.
- [ ] Implement reactive forms, drag-and-drop file upload, chapter metadata table, and live progress modal.
- [ ] Register route `/admin/book-ingestion` in `app.routes.ts` and add nav link in `admin-layout.component.html`.

### Phase 4: Verification & End-to-End Testing
- [ ] Test uploading a Class 9 textbook chapter through the new UI.
- [ ] Confirm Qdrant automatically creates and populates `school_textbooks_class9`.
- [ ] Log in as `student9@lms.com` and verify the AI Chat Drawer retrieves citations and diagrams from the newly ingested book.
