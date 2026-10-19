# PROMPT: Integrating Python LangGraph RAG Agent with Angular LMS Frontend

> **How to use this file:**  
> When you resume your session or start a new chat with the assistant, simply copy and paste the prompt in **Section 1** below. It contains all the context, API specifications, and architectural details needed to perform the integration seamlessly.

---

## 1. Copy-Paste Prompt for Tomorrow's Session

```markdown
Hello! I want to integrate our Python LangGraph Textbook RAG backend with our Angular LMS Frontend.

Here is the exact status and context of what has already been built:

1. BACKEND SERVICE LOCATION & SPECS:
   - Location: `d:\signal-angular\lms-book-ingestion`
   - Server file: `rag_server.py`
   - Orchestrator: `rag_graph/workflow.py` (Compiled LangGraph StateGraph)
   - Config file: `config.py` (reads `PORT` from `.env`, default is 8001: `http://localhost:8001`)
   - Qdrant Vector DB: Ingested textbook collections named `school_textbooks_class{standard}` (e.g. `school_textbooks_class10`).

2. ENDPOINTS AVAILABLE ON BACKEND (Port 8001):
   - POST `/api/rag/chat`: Synchronous JSON response
   - POST `/api/rag/chat/stream`: SSE Streaming response (Server-Sent Events)
   - GET `/health`: Health status

3. PAYLOAD CONTRACT:
   Request:
   {
     "query": "string (student question)",
     "student_name": "string (e.g. 'Sneha' or 'Aman')",
     "standard": 10,             // integer class/grade (e.g. 9 or 10)
     "subject": "mathematics",   // optional: if null, agent auto-detects subject
     "chapter_number": null      // optional: integer chapter number
   }

   Response:
   {
     "student_name": "Sneha",
     "standard": 10,
     "subject_detected": "mathematics",
     "collection_used": "school_textbooks_class10",
     "collection_exists": true,
     "answer": "string (contains LaTeX math like $ax^2 + bx + c = 0$)",
     "is_safe": true,
     "has_error": false,
     "error_message": null,
     "citations": [
       {
         "book_title": "Mathematics - Class 10",
         "chapter_number": 4,
         "chapter_name": "Quadratic Equations",
         "page_start": 70,
         "page_end": 73
       }
     ],
     "diagrams": [
       {
         "caption": "Figure title",
         "image_url": "path_or_url"
       }
     ]
   }

4. FRONTEND LOCATION IN WORKSPACE:
   - Angular Project: `d:\signal-angular\lms`
   - Existing AI feature: `src/app/features/ai-platform/ai-chat-drawer/`
   - Existing service: `src/app/core/services/ai-chat.service.ts`

5. OBJECTIVE:
   Please guide me and implement the frontend integration:
   - Update or create the Angular service to point to `http://localhost:8001/api/rag/chat` (and `/stream`).
   - Pass logged-in student details (`student_name`, `standard` from current student profile/auth state).
   - Render the AI response with:
     a) LaTeX math equations support (KaTeX or MathJax)
     b) Textbook citations pill/card (Book, Chapter, Page numbers)
     c) Render textbook diagrams/images if present in payload
     d) Clean error/notification badge if the class textbook is not yet available.

Please review the frontend files and execute the integration step-by-step.
```

---

## 2. Integration Architecture & Data Flow

```
┌────────────────────────────────────────────────────────┐
│             Angular LMS Frontend (Port 4200)           │
│                                                        │
│  [Student Profile: Class 10, Name: Sneha]              │
│  [Student asks: "What are the roots of quadratic eq?"] │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ HTTP POST (SSE stream or JSON)
                            │ URL: http://localhost:8001/api/rag/chat/stream
                            ▼
┌────────────────────────────────────────────────────────┐
│      Python LangGraph RAG Agent (Port 8001)            │
│                                                        │
│  1. Guardrails Node (Blocks prompt injections)         │
│  2. Context Resolver Node (Senses subject: math)       │
│  3. Collection Validator (Verifies class10 in Qdrant)  │
│  4. Qdrant Dense Retrieval (768-dim embeddings)        │
│  5. Cohere / Score Reranker                            │
│  6. Gemini LLM Generator (LaTeX math + citations)      │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ SSE Stream / JSON Response
                            ▼
┌────────────────────────────────────────────────────────┐
│             Angular Chat Drawer Component              │
│                                                        │
│  • Displays typing animation / tokens                  │
│  • Renders LaTeX formula: ax² + bx + c = 0             │
│  • Shows textbook diagram image                        │
│  • Displays Citations: [Class 10 Math, Ch 4, Pg 70-73] │
└────────────────────────────────────────────────────────┘
```

---

## 3. Specific Angular Files to Update Tomorrow

1. **`d:\signal-angular\lms\src\environments\environment.ts`**:
   - Add `ragApiUrl: 'http://localhost:8001'`

2. **`d:\signal-angular\lms\src\app\core\services\ai-chat.service.ts`**:
   - Add `streamStudentRagChat(query: string, studentName: string, standard: number, subject?: string)`
   - Add `askStudentRagChat(...)` for synchronous calls.

3. **`d:\signal-angular\lms\src\app\features\ai-platform\ai-chat-drawer\ai-chat-drawer.component.ts`**:
   - Bind current student user profile from `AuthService` or `StudentStateService`.
   - Add Markdown + KaTeX rendering for math formulas ($...$ and $$...$$).
   - Add citation pill list and diagram rendering in the chat bubble UI.
