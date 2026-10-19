# LMS Student Textbook RAG API Guide

A fast, intelligent, and secure academic Q&A backend orchestrated using **LangGraph**, **FastAPI**, **Qdrant**, and **Google Gemini**.

---

## 1. How to Start the Server

From the `lms-book-ingestion` directory:

```powershell
# Using the dedicated virtual environment:
.\.venv\Scripts\python.exe rag_server.py
```

* The server runs on **Port 8001**: `http://localhost:8001`
* Interactive Swagger Docs: **`http://localhost:8001/docs`**

---

## 2. API Endpoints

### A. Standard Chat (Synchronous JSON)
Use this for regular REST request/response.

* **URL Path:** `http://localhost:8001/api/rag/chat`
* **Method:** `POST`
* **Headers:** `Content-Type: application/json`

#### Request Body
```json
{
  "query": "What is a quadratic equation?",
  "student_name": "Sneha",
  "standard": 10,
  "subject": "mathematics",
  "chapter_number": 4
}
```

> **Note on Fields:**
> * `query` *(required)*: The student's question.
> * `standard` *(default: 10)*: Student's grade/class (e.g. `9` or `10`). Automatically routes to the matching collection (e.g. `school_textbooks_class10`).
> * `student_name` *(optional)*: Used for personalized answers.
> * `subject` *(optional)*: If omitted, the agent auto-detects whether the question belongs to `mathematics`, `science`, `social_science`, etc.
> * `chapter_number` *(optional)*: Scopes retrieval to a specific chapter.

#### Response Example
```json
{
  "student_name": "Sneha",
  "standard": 10,
  "subject_detected": "mathematics",
  "collection_used": "school_textbooks_class10",
  "collection_exists": true,
  "answer": "Hello Sneha! In mathematics, a quadratic equation in the variable $x$ is of the form $ax^2 + bx + c = 0$...",
  "is_safe": true,
  "has_error": false,
  "error_message": null,
  "citations": [
    {
      "book_title": "Mathematics - Class 10",
      "chapter_number": 4,
      "chapter_name": "Quadratic Equations",
      "page_start": 70,
      "page_end": 73,
      "topic_name": "Quadratic Equations"
    }
  ],
  "diagrams": []
}
```

---

### B. Streaming Chat (Server-Sent Events - SSE)
Use this for real-time typing effect in your chat UI.

* **URL Path:** `http://localhost:8001/api/rag/chat/stream`
* **Method:** `POST`
* **Headers:** `Content-Type: application/json`
* **Request Body:** Same JSON as standard chat.

#### Stream Format
1. First event emits `metadata` (citations, diagrams, detected subject, collection):
   ```
   data: {"type": "metadata", "student_name": "Sneha", "citations": [...], "diagrams": [...]}
   ```
2. Subsequent events emit the answer tokens:
   ```
   data: {"chunk": "Hello Sneha!...", "done": true, "is_safe": true}
   ```

---

### C. Health Check
* **URL Path:** `http://localhost:8001/health`
* **Method:** `GET`
* **Response:** `{"status": "ok", "service": "lms-langgraph-rag"}`

---

## 3. Quick cURL & Fetch Examples

### cURL
```bash
curl -X POST http://localhost:8001/api/rag/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain photosynthesis and plant cell structure",
    "student_name": "Aman",
    "standard": 10
  }'
```

### JavaScript / TypeScript (Frontend Fetch)
```javascript
const response = await fetch("http://localhost:8001/api/rag/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "Explain what is a quadratic equation?",
    student_name: "Sneha",
    standard: 10,
    subject: "mathematics"
  })
});

const data = await response.json();
console.log(data.answer);
console.log(data.citations);
```

---

## 4. Automatic Features & Built-in Safety

1. **Class Isolation:** Class 9 queries go to `school_textbooks_class9`; Class 10 queries go to `school_textbooks_class10`.
2. **Missing Collection Protection:** If a class textbook is not yet in Qdrant, returns a friendly notice without crashing.
3. **Guardrails:** Automatically blocks prompt injections (e.g. `DROP TABLE`, `IGNORE INSTRUCTIONS`).
4. **LaTeX & Diagrams:** LaTeX formulas ($...$ and $$...$$) and textbook diagrams are automatically preserved and linked.
