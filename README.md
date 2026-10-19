# LMS Book Ingestion — Python Project
## Textbook PDF → Qdrant Vector DB Ingestion Pipeline

This is a standalone Python project that ingests school textbook PDFs into a Qdrant vector database
for use with the LMS Student Q&A AI Agent.

---

## 📁 PDF Folder Structure (MANDATORY)

Place all PDF files inside the `books/` directory, organized strictly as:

```
books/
└── standard_{N}/                        ← Class number (6, 7, 8, 9, 10, 11, 12)
    └── {subject}/                       ← Subject name in lowercase (see list below)
        ├── chapters.json                ← OPTIONAL: page offsets & book metadata
        └── ch{NN}_{chapter_name}.pdf   ← Chapter PDFs (PRIMARY mode)
        OR
        └── {full_book_name}.pdf        ← Single full book PDF (FALLBACK mode)
```

---

## 📂 Supported Subject Folder Names

| Folder Name       | Category      | Chunking Strategy       |
|-------------------|---------------|-------------------------|
| `mathematics`     | STEM          | Atomic (450–750 tokens) |
| `physics`         | STEM          | Atomic (450–750 tokens) |
| `chemistry`       | STEM          | Atomic (450–750 tokens) |
| `science`         | STEM          | Atomic (450–750 tokens) |
| `biology`         | STEM          | Atomic (450–750 tokens) |
| `history`         | HUMANITIES    | Paragraph (900–1300)    |
| `geography`       | HUMANITIES    | Paragraph (900–1300)    |
| `civics`          | HUMANITIES    | Paragraph (900–1300)    |
| `economics`       | HUMANITIES    | Paragraph (900–1300)    |
| `social_science`  | HUMANITIES    | Paragraph (900–1300)    |
| `english`         | LANGUAGE_LIT  | Passage (500–900 tokens)|
| `odia`            | LANGUAGE_LIT  | Passage (500–900 tokens)|
| `hindi`           | LANGUAGE_LIT  | Passage (500–900 tokens)|
| `sanskrit`        | LANGUAGE_LIT  | Passage (500–900 tokens)|

---

## 📄 Chapter PDF Naming Convention (PRIMARY Mode)

```
ch{NUMBER}_{snake_case_chapter_name}.pdf
```

**Rules:**
- Chapter number must be **2 digits** with leading zero: `ch01`, `ch02`, ... `ch15`
- Chapter name in **lowercase snake_case** (spaces replaced with `_`)
- No special characters except underscores

**Examples:**
```
ch01_real_numbers.pdf
ch02_polynomials.pdf
ch11_human_eye_colourful_world.pdf
ch03_water_resources.pdf
ch01_rise_of_nationalism_in_europe.pdf
```

**Auto-Mode Detection:**
- Folder with **multiple PDFs** → Chapter-Wise Mode (PRIMARY ✅)
- Folder with **single PDF** → Full-Book Mode (FALLBACK ⚠️)

---

## 📋 chapters.json — Page Offset Config (Recommended)

Place a `chapters.json` in the subject folder to map original textbook page numbers.

```json
{
  "_meta": {
    "book_title": "NCERT Science Class 10",
    "board": "CBSE"
  },
  "ch01_chemical_reactions.pdf":        { "page_offset": 1,   "chapter_name": "Chemical Reactions and Equations" },
  "ch11_human_eye.pdf":                 { "page_offset": 179, "chapter_name": "The Human Eye and the Colourful World" },
  "ch12_electricity.pdf":               { "page_offset": 200, "chapter_name": "Electricity" }
}
```

**Fields:**
- `_meta.book_title` — Official book name (used in Qdrant citations)
- `_meta.board` — `"CBSE"`, `"ICSE"`, `"SCERT"`, etc.
- `page_offset` — Page number in the original printed book where this chapter starts
- `chapter_name` — Full official chapter name (overrides filename-derived name)

> If `chapters.json` is missing, page numbers are PDF-relative and chapter names are derived from filenames.

---

## 🗂️ Full Example Structure

```
books/
├── standard_9/
│   ├── mathematics/
│   │   ├── chapters.json
│   │   ├── ch01_number_systems.pdf
│   │   ├── ch02_polynomials.pdf
│   │   ├── ch03_coordinate_geometry.pdf
│   │   └── ch04_linear_equations_two_variables.pdf
│   │
│   ├── science/
│   │   ├── chapters.json
│   │   ├── ch01_matter_in_surroundings.pdf
│   │   ├── ch08_motion.pdf
│   │   └── ch09_force_and_laws_of_motion.pdf
│   │
│   └── history/
│       └── india_contemporary_world_class9.pdf    ← full-book fallback
│
└── standard_10/
    ├── mathematics/
    │   ├── chapters.json
    │   ├── ch01_real_numbers.pdf
    │   └── ch02_polynomials.pdf
    │
    ├── science/
    │   ├── chapters.json
    │   ├── ch11_human_eye.pdf
    │   └── ch12_electricity.pdf
    │
    └── odia/
        ├── chapters.json
        ├── ch01_poem_satyabadi.pdf
        └── ch02_story_dandi.pdf
```

---

## ⚙️ Setup & Run

### 1. Install Python (3.10+ required)

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env with your Qdrant URL and Google API Key
```

### 5. Add your PDFs
Place PDFs in `books/` following the structure above.

### 6. Run ingestion
```bash
# Ingest all books
python ingest_books.py

# Ingest specific standard only
python ingest_books.py --standard 10

# Ingest specific standard + subject
python ingest_books.py --standard 10 --subject science

# Dry run (no Qdrant upload, just preview chunks)
python ingest_books.py --dry-run
```

---

## 📦 Output

- **Qdrant collection:** `school_textbooks`
- **Extracted images:** `public/book-assets/standard_{N}/{subject}/`
- **Logs:** Console output with progress bars

---

## ⚠️ Notes

- PDFs must be **digital (text-selectable)**, not scanned images.
- For scanned PDFs, run OCR first (Tesseract / Google Vision).
- Extracted images are saved and must be **served statically** by the Express server in `Lms-analysis-agent`.
- Re-running ingestion for a chapter will **delete existing chunks** for that chapter before re-inserting (safe re-ingestion).
