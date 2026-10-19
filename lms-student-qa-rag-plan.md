# Universal Multi-Class, Multi-Subject Textbook Ingestion & Q&A RAG Architecture
## (With LaTeX Math, Exact Topic Grounding & Diagram / Image Support)

> **Purpose:** A complete, production-grade technical specification and implementation plan for a School Student Q&A AI Agent.
> **Scope:** Multi-Standard (Grade 6 to 12+), Multi-Subject (STEM / Math with LaTeX formulas & solved proofs, Science / Biology with diagrams and apparatus illustrations, Humanities / Social Science with chronological narratives & dates, and Languages/Literature), with zero-hallucination, strict topic-level retrieval and authentic textbook citations.

---

## 1. Core Principles & School Textbook Challenges

School textbooks differ fundamentally by subject and grade:

1. **Class Isolation:** A Class 9 student must **never** receive explanations or formulas meant for Class 10 or Class 11, even for overlapping topics (e.g., "Motion" or "Linear Equations").
2. **Subject Paradigm Differences:**
   - **Mathematics & STEM:** Contains formulas, equations ($x^2 + y^2 = r^2$), theorem proofs, and solved examples with step-by-step calculations. Splitting mid-calculation breaks the solution.
   - **Science (Biology, Physics, Chemistry):** Often accompanied by crucial diagrams (e.g., *Structure of Human Eye*, *Electric Circuits*, *Plant Cell*, *Ray Optics*, *Chemical Reaction Apparatus*). The agent must be capable of embedding the actual textbook diagram in its answer.
   - **Humanities (History, Geography, Civics, Economics):** Contains long continuous paragraphs, cause-and-effect sequences, timelines, maps, and dates. Chopping paragraphs prematurely loses historical context.
   - **Languages & Literature (English, Grammar):** Contains poems with stanzas, short stories, and grammar rules with exercises. Must preserve entire stanzas or rule-example pairs.
3. **No Hallucination / Exact Textbook Citation:** Answers must strictly match the book's approved syllabus, including explicit citations of Book Title, Chapter, Topic, and Page Number.

---

## 2. Subject Taxonomy & Adaptive Chunking Strategy

Rather than using one fixed chunking size for all books, the ingestion engine applies **Subject-Adaptive Chunking**:

```
                               ┌─────────────────────────────────────────────────────────────┐
                               │                 SCHOOL SUBJECT TAXONOMY                     │
                               └─────────────────────────────────────────────────────────────┘
                                                              │
         ┌────────────────────────────────────────────────────┼────────────────────────────────────────────────────┐
         │                                                    │                                                    │
         ▼                                                    ▼                                                    ▼
┌─────────────────────────────────┐          ┌─────────────────────────────────┐          ┌─────────────────────────────────┐
│       TYPE A: STEM / FORMULA    │          │     TYPE B: NARRATIVE HUMANITIES│          │       TYPE C: LANGUAGE & LIT    │
│  (Math, Physics, Chemistry)     │          │ (History, Geography, Civics,    │          │  (English, Grammar, Regional)   │
│                                 │          │  Social Science, Biology)       │          │                                 │
├─────────────────────────────────┼──────────┼─────────────────────────────────┼──────────┼─────────────────────────────────┤
│ • Formulae, proofs, equations   │          │ • Long chronological narratives │          │ • Poems, stories, essays        │
│ • Step-by-step solved examples  │          │ • Timelines, events, causes     │          │ • Grammar rules & exercises     │
│ • Exercise problem sets         │          │ • Maps & geographic features    │          │ • Vocabulary & comprehension    │
│ • Science / Circuit diagrams    │          │ • Broad context paragraphs      │          │                                 │
├─────────────────────────────────┼──────────┼─────────────────────────────────┼──────────┼─────────────────────────────────┤
│ 🎯 CHUNKING STRATEGY:           │          │ 🎯 CHUNKING STRATEGY:           │          │ 🎯 CHUNKING STRATEGY:           │
│ Atomic Unit Chunking            │          │ Semantic Paragraph Windowing    │          │ Passage & Rule Grouping         │
│ (Keep full Example + Solution + │          │ (Larger window with overlap     │          │ (Keep poems/stories together or │
│  associated Diagram links)      │          │  to preserve historical causes) │          │  by logical stanza/act)         │
│ Target: 450 - 750 tokens        │          │ Target: 900 - 1300 tokens       │          │ Target: 500 - 900 tokens        │
│ Overlap: 100 tokens             │          │ Overlap: 200 tokens             │          │ Overlap: 120 tokens             │
└─────────────────────────────────┘          └─────────────────────────────────┘          └─────────────────────────────────┘
```

> **Note on Chapter-Wise PDFs:** When separate chapter PDFs are provided (primary mode), chunking is
> strictly scoped to one chapter, eliminating any risk of cross-chapter chunk bleeding — especially
> critical for STEM subjects where each chapter introduces completely new concepts and formulas.

---

## 3. Textbook Diagram & Image Extraction Pipeline

To answer science questions like *"Describe the parts of the human eye"* or *"Explain the electric motor"* with actual visual diagrams:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              1. INGESTION PHASE (Python)                               │
│                                                                                        │
│  [Science PDF Book] (Page 48: "Human Eye Diagram")                                     │
│         │                                                                              │
│         ├── Step A: PyMuPDF extracts images & filters out tiny icons (< 120x120px).    │
│         │           Saves image to: `public/book-assets/std10/science/ch11_fig1.png`   │
│         │                                                                              │
│         ├── Step B: Detects Figure Caption in text (e.g. "Figure 11.1: The Human Eye") │
│         │                                                                              │
│         └── Step C: Associates diagram metadata with the text chunk in Qdrant:         │
│                     `"diagrams": [{ "caption": "Figure 11.1: The Human Eye",           │
│                                     "image_url": "/book-assets/.../ch11_fig1.png" }]`  │
└────────────────────────────────────────────────────────────────────────────────────────┘

                                           │
                                           ▼ (Query Time)

┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              2. QUERY PHASE (Live Agent)                               │
│                                                                                        │
│  [Student asks]: "Explain the parts of the human eye and its structure"               │
│         │                                                                              │
│         ├── Step A: Qdrant retrieves the chunk + the "diagrams" array in payload       │
│         │                                                                              │
│         ├── Step B: Prompt informs Gemini that a diagram is available.                 │
│         │           Gemini generates text and embeds:                                  │
│         │           `![Figure 11.1: The Human Eye](/book-assets/.../ch11_fig1.png)`   │
│         │                                                                              │
│         └── Step C: Angular Frontend renders text + high-res textbook diagram!         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Hybrid PDF Input Mode (Chapter-Wise PRIMARY + Full-Book FALLBACK)

The ingestion script supports **two input modes** and **auto-detects** which to use per subject folder.

### Mode Priority Rule
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  AUTO-DETECTION LOGIC (per subject folder)                                      │
│                                                                                 │
│  IF  subject folder contains multiple PDF files (e.g. ch01_*.pdf, ch02_*.pdf)  │
│      → USE  MODE 1: Chapter-Wise (PRIMARY) ✅ Preferred                        │
│                                                                                 │
│  ELSE IF  subject folder contains a single full PDF file                        │
│      → USE  MODE 2: Full-Book (FALLBACK) ⚠️ Less precise                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### MODE 1: Chapter-Wise PDFs (Primary — Used Most Often)

**Folder Structure:**
```
books/
└── standard_10/
    ├── mathematics/
    │   ├── ch01_real_numbers.pdf
    │   ├── ch02_polynomials.pdf
    │   ├── ch03_pair_of_linear_equations.pdf
    │   ├── ch04_quadratic_equations.pdf
    │   └── chapters.json                   ← optional: stores original page offsets
    │
    ├── science/
    │   ├── ch01_chemical_reactions.pdf
    │   ├── ch11_human_eye.pdf
    │   ├── ch12_electricity.pdf
    │   └── chapters.json
    │
    └── history/
        ├── ch01_rise_of_nationalism.pdf
        ├── ch02_nationalism_in_india.pdf
        └── chapters.json
```

**Chapter Filename Convention:**
```
ch{NUMBER}_{snake_case_chapter_name}.pdf

Examples:
  ch01_real_numbers.pdf
  ch11_human_eye_colourful_world.pdf
  ch03_water_resources.pdf
```

**`chapters.json` — Page Offset Config (Optional but Recommended):**
```json
{
  "ch01_real_numbers.pdf":               { "page_offset": 1,  "chapter_name": "Real Numbers" },
  "ch02_polynomials.pdf":                { "page_offset": 21, "chapter_name": "Polynomials" },
  "ch03_pair_of_linear_equations.pdf":   { "page_offset": 39, "chapter_name": "Pair of Linear Equations in Two Variables" },
  "ch11_human_eye_colourful_world.pdf":  { "page_offset": 179, "chapter_name": "The Human Eye and the Colourful World" }
}
```

> **Why page_offset?** Chapter PDFs start at page 1 internally, but Chapter 11 might start at
> page 179 in the original printed book. The offset restores the correct textbook page citation.
> If `chapters.json` is absent, the script uses PDF-relative page numbers.

**What the Ingestion Script Auto-Extracts from Chapter PDFs:**

| Source | Field Extracted |
|---|---|
| Folder `standard_10/` | `standard: 10`, `standard_label: "Class 10"` |
| Folder `mathematics/` | `subject: "mathematics"`, `subject_category: "STEM"` |
| Filename `ch03_pair_of_linear_equations.pdf` | `chapter_number: 3`, `chapter_name: "Pair Of Linear Equations"` |
| `chapters.json` (if present) | `page_offset: 39`, refined `chapter_name` |
| PDF content itself | `content`, `diagrams[]`, `has_math_formula`, `has_diagram` |

**Advantages of Chapter-Wise Mode:**

| Benefit | Detail |
|---|---|
| ✅ Zero cross-chapter chunk bleeding | Chunker operates inside one chapter boundary only |
| ✅ Precise metadata auto-discovery | Chapter number & name from filename — no content parsing |
| ✅ Cheap re-ingestion | Update only the changed chapter's PDF |
| ✅ Low memory usage | PyMuPDF processes 15–30 pages instead of 300+ |
| ✅ Better STEM atomic chunking | Math examples don't spill across chapter concepts |

---

### MODE 2: Full-Book PDF (Fallback — When Chapter PDFs Unavailable)

**Folder Structure:**
```
books/
└── standard_9/
    ├── mathematics/
    │   └── ncert_math_class9.pdf          ← single full book PDF
    │
    └── history/
        └── india_contemporary_world.pdf
```

**What the ingestion script does differently in Full-Book Mode:**
- Parses chapter headings from the PDF's text (e.g., lines matching `Chapter \d+:` patterns)
- Injects a soft chapter boundary marker before chunking
- Falls back to `chapter_number: null`, `chapter_name: "Unknown"` if detection fails
- Page numbers are direct PDF page numbers (accurate, since it's the full book)

> ⚠️ **Full-Book Mode Limitation:** Cross-chapter chunks can occur at chapter boundary pages.
> Chunk quality is lower than chapter-wise mode, especially for STEM subjects.

---

## 5. Updated Input Directory Layout (Both Modes Together)

A real deployment can mix both modes across subjects:

```
Lms-analysis-agent/
└── ingestion/
    ├── requirements.txt
    ├── ingest_books.py                   # Master ingestion script (handles both modes)
    ├── config.py                         # Qdrant URL, Gemini API keys
    │
    ├── books/
    │   │
    │   ├── standard_6/
    │   │   ├── mathematics/              ← MODE 2: single full PDF available
    │   │   │   └── ncert_math_class6.pdf
    │   │   └── science/                  ← MODE 2: single full PDF available
    │   │       └── ncert_science_class6.pdf
    │   │
    │   ├── standard_9/
    │   │   ├── mathematics/              ← MODE 1: chapter-wise PDFs ✅
    │   │   │   ├── ch01_number_systems.pdf
    │   │   │   ├── ch02_polynomials.pdf
    │   │   │   ├── ch03_coordinate_geometry.pdf
    │   │   │   ├── ch04_linear_equations.pdf
    │   │   │   └── chapters.json
    │   │   ├── science/                  ← MODE 1: chapter-wise PDFs ✅
    │   │   │   ├── ch01_matter_in_surroundings.pdf
    │   │   │   ├── ch08_motion.pdf
    │   │   │   ├── ch09_force_and_laws.pdf
    │   │   │   └── chapters.json
    │   │   └── history/                  ← MODE 2: only full PDF available
    │   │       └── india_contemporary_world_class9.pdf
    │   │
    │   └── standard_10/
    │       ├── mathematics/              ← MODE 1: chapter-wise PDFs ✅
    │       │   ├── ch01_real_numbers.pdf
    │       │   ├── ch02_polynomials.pdf
    │       │   └── chapters.json
    │       ├── science/                  ← MODE 1: chapter-wise PDFs ✅
    │       │   ├── ch11_human_eye.pdf
    │       │   ├── ch12_electricity.pdf
    │       │   └── chapters.json
    │       └── social_science/           ← MODE 2: full PDF
    │           └── ncert_sst_class10.pdf
    │
    └── public/book-assets/               # Extracted images (both modes output here)
        ├── standard_9/
        │   └── science/
        │       └── ch08_p102_fig1_motion_graph.png
        └── standard_10/
            └── science/
                ├── ch11_p48_fig1_human_eye.png
                └── ch12_p55_fig2_circuit_diagram.png
```

---

## 6. Comprehensive Metadata Payload Schema in Qdrant

Every chunk ingested into Qdrant stores the following structured payload:

```json
{
  "chunk_id": "std10_science_ch11_topic01_p48_c01",
  "pdf_mode": "chapter_wise",
  "standard": 10,
  "standard_label": "Class 10",
  "subject": "science",
  "subject_category": "STEM",
  "board": "CBSE",
  "book_title": "NCERT Science Class 10",
  "chapter_number": 11,
  "chapter_name": "The Human Eye and the Colourful World",
  "source_pdf": "ch11_human_eye.pdf",
  "page_offset": 179,
  "topic_number": "11.1",
  "topic_name": "The Human Eye",
  "content_type": "concept_with_diagram",
  "page_start": 48,
  "page_end": 48,
  "has_math_formula": false,
  "has_diagram": true,
  "diagrams": [
    {
      "caption": "Figure 11.1: The Human Eye",
      "image_url": "/book-assets/standard_10/science/ch11_p48_fig1_human_eye.png",
      "width": 640,
      "height": 420
    }
  ],
  "content": "..."
}
```

> **New Fields vs. Original Plan:**
> - `pdf_mode`: `"chapter_wise"` or `"full_book"` — tracks which mode produced this chunk
> - `source_pdf`: original PDF filename for traceability
> - `page_offset`: offset added to PDF-internal page numbers to get real book page citation

### Indexed Payload Fields (Indexed in Qdrant for <5ms Pre-filtering):
1. `standard` (Integer / Keyword)
2. `subject` (Keyword)
3. `chapter_number` (Integer)
4. `topic_name` (Keyword)
5. `has_diagram` (Boolean)
6. `content_type` (Keyword)
7. `pdf_mode` (Keyword) ← **new**

---

## 7. Chunk Enrichment with Context & Diagram Banners

Before vectors are computed, every chunk is prepended with a **Standard Context Banner**:

```text
================================================================================
[CLASS/STANDARD: 10 | SUBJECT: Science | CATEGORY: STEM]
[BOOK: NCERT Science Class 10 | BOARD: CBSE]
[CHAPTER 11: The Human Eye and the Colourful World]
[TOPIC: 11.1 The Human Eye | PAGE: 48]
[PDF MODE: chapter_wise | SOURCE: ch11_human_eye.pdf]
[DIAGRAM AVAILABLE: "Figure 11.1: The Human Eye" -> Image URL: /book-assets/standard_10/science/ch11_p48_fig1_human_eye.png]
================================================================================
The human eye is one of the most valuable and sensitive sense organs. It enables us
to see the wonderful world and the colours around us.
The human eye is like a camera. Its lens system forms an image on a light-sensitive
screen called the retina. Light enters the eye through a thin membrane called the cornea...
```

---

## 8. Subject-Adaptive System Prompts (Zero Hallucination + Diagrams)

### For Science (with Diagrams) & Mathematics (with LaTeX):
```text
You are the official textbook teaching assistant for Class {standard} {subject}.
Your task is to answer the student's question strictly using the provided textbook context.

RULES:
1. Provide the complete step-by-step explanation or calculation matching the textbook's methodology.
2. For Mathematics / Physics calculations:
   - Render all equations in LaTeX: inline ($formula$) or block ($$formula$$).
   - Do not skip intermediate calculation steps.
3. For Diagrams and Illustrations:
   - If the context lists an available diagram under [DIAGRAM AVAILABLE: "Caption" -> Image URL: "..."], and it is relevant to the student's question, you MUST embed it in the answer using Markdown:
     ![Caption](Image URL)
4. If the question cannot be answered using the provided excerpts, state:
   "This question is not covered in your Class {standard} {subject} textbook."
5. Always end with:
   📘 **Reference**: [Book Title], Chapter: [Chapter Name], Topic: [Topic Name], Page: [Page Number]
```

### For History, Geography & Social Sciences:
```text
You are the official textbook teaching assistant for Class {standard} {subject}.
Your task is to answer the student's question strictly using the provided textbook context.

RULES:
1. Answer using only the dates, events, causes, consequences, and definitions in the excerpts.
2. Do not summarize or import outside historical assumptions or extraneous facts.
3. If an illustrative map or timeline image is available in the context, embed it using Markdown: ![Caption](Image URL).
4. Quote key terms exactly as defined in the textbook.
5. If the question cannot be answered using the provided excerpts, state:
   "This topic is not covered in your Class {standard} {subject} textbook."
6. Always end with:
   📘 **Reference**: [Book Title], Chapter: [Chapter Name], Topic: [Topic Name], Page: [Page Number]
```

---

## 9. Frontend Integration (Angular)

To display text, formulas, and diagrams properly to students:
1. **Markdown & Image Rendering**: Use `ngx-markdown` or `markdown-it`. Configured to allow image tags and point to the static asset server.
2. **LaTeX Math Rendering**: `katex` for high-speed client-side math rendering.
3. **Diagram Zoom / Modal**: Clicking any rendered textbook diagram opens a high-resolution lightbox view for easy study.

---

## 10. Step-by-Step Implementation Roadmap

| Milestone | Module | Description |
| :--- | :--- | :--- |
| **Step 1** | Python Environment Setup | Create `Lms-analysis-agent/ingestion/` with `requirements.txt` (`pymupdf4llm`, `PyMuPDF`, `langchain-text-splitters`, `qdrant-client`, `google-genai`). |
| **Step 2** | PDF Mode Detector | In `ingest_books.py`: scan each subject folder — if multiple PDFs found → `chapter_wise` mode; if single PDF → `full_book` mode. Log detected mode per subject at startup. |
| **Step 3** | Chapter-Wise Ingestion (Primary) | For chapter PDFs: parse `ch{N}_{name}.pdf` filename → extract `chapter_number` + `chapter_name`. Load `chapters.json` if present for `page_offset` and refined names. Chunk strictly within chapter boundary. |
| **Step 4** | Full-Book Ingestion (Fallback) | For single full PDFs: detect chapter headings via regex, insert soft boundary markers, chunk with larger overlap. Flag chunks as `pdf_mode: "full_book"` in payload. |
| **Step 5** | Image Extraction & Ingestion Script | Auto-discovers PDFs in `standard_{X}/{subject}`, extracts page diagrams (>120x120px) to `public/book-assets/`, chunks adaptively, adds Context + Diagram banners, and uploads vectors to Qdrant. Works in both PDF modes. |
| **Step 6** | Qdrant Setup & Collection Indexing | Create `school_textbooks` collection with payload indexes on `standard`, `subject`, `chapter_number`, `topic_name`, `has_diagram`, and `pdf_mode`. |
| **Step 7** | Static Asset Serving | Serve `public/book-assets/` from Express in `Lms-analysis-agent` or LMS API so Angular can render image URLs directly. |
| **Step 8** | Live TypeScript Service | Implement Qdrant search helper and LangChain/Gemini Flash Q&A chain inside `Lms-analysis-agent/src/services/studentQa.service.ts` with diagram injection rules. |
| **Step 9** | REST API Route | Expose `POST /api/v1/student-qa/ask` in `Lms-analysis-agent/src/api/server.ts` with student grade validation. |
| **Step 10** | Angular UI Integration | Connect chat dialog with Markdown rendering, KaTeX math formulas, and interactive image previews. |

---

## 11. PDF Mode Decision Quick Reference

```
When do I use Chapter-Wise PDFs?  ← USE THIS MOST OF THE TIME
  ✅ You have separate chapter PDFs (ch01_*.pdf, ch02_*.pdf, ...)
  ✅ STEM subjects (Math, Physics, Chemistry, Biology)
  ✅ When you need precise citations & cheap re-ingestion
  ✅ When you want the best chunk quality

When do I fall back to Full-Book PDF?
  ⚠️  You only have one combined PDF for the entire subject
  ⚠️  Humanities subjects where chapter boundaries matter less
  ⚠️  Temporary — replace with chapter PDFs when available
```

---

*File location: `d:\signal-angular\Lms-analysis-agent\lms-student-qa-rag-plan.md`*
*Last updated: Chapter-Wise Primary + Full-Book Fallback Hybrid Mode*
