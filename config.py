"""
Central configuration for the LMS Book Ingestion pipeline.
All values are read from .env — see .env.example for reference.
"""

import os
from dotenv import load_dotenv

# Explicitly load .env file from this project directory and override OS environment variables
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)  # .env wins over stale OS-level env vars

# ─── Server Configuration ───────────────────────────────────────────────────
RAG_PORT               = int(os.getenv("PORT", os.getenv("RAG_PORT", "8001")))
RAG_HOST               = os.getenv("HOST", os.getenv("RAG_HOST", "0.0.0.0"))

# ─── Qdrant ─────────────────────────────────────────────────────────────────
QDRANT_URL             = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY         = os.getenv("QDRANT_API_KEY") or None   # None = local, key = cloud
QDRANT_COLLECTION      = os.getenv("QDRANT_COLLECTION", "school_textbooks")  # base prefix
VECTOR_SIZE            = 768  # gemini-embedding-001 truncated to 768 dims (Matryoshka)


def collection_name_for_class(standard: int) -> str:
    """
    Returns the Qdrant collection name for a given class/standard.
    Example: standard=10  →  'school_textbooks_class10'
    The base prefix comes from QDRANT_COLLECTION env var so it's configurable.
    """
    return f"{QDRANT_COLLECTION}_class{standard}"

# ─── Google AI & LLM ──────────────────────────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")
LLM_MODEL       = os.getenv("LLM_MODEL", "")
COHERE_API_KEY  = os.getenv("COHERE_API_KEY", "")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
BOOKS_DIR        = os.path.join(BASE_DIR, "books")
PUBLIC_ASSETS_DIR = os.path.join(BASE_DIR, "public", "book-assets")

# ─── Cloudinary ─────────────────────────────────────────────────────────────
CLOUDINARY_CLOUD_NAME  = os.getenv("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_API_KEY     = os.getenv("CLOUDINARY_API_KEY", "")
CLOUDINARY_API_SECRET  = os.getenv("CLOUDINARY_API_SECRET", "")
USE_CLOUDINARY         = bool(CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET)

# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_BOARD    = os.getenv("DEFAULT_BOARD", "CBSE")
MIN_IMAGE_SIZE   = int(os.getenv("MIN_IMAGE_SIZE_PX", "120"))  # px (width & height)
MAX_IMAGE_DIM    = int(os.getenv("MAX_IMAGE_DIM_PX", "1200"))  # px (max width or height before upload)
IMAGE_QUALITY    = int(os.getenv("IMAGE_QUALITY", "82"))        # WebP / JPEG quality (80-85 is visually lossless)
SKIP_EXISTING    = os.getenv("SKIP_EXISTING", "0") == "1"  # resume: skip chapters already in Qdrant

# ─── Subject → Category Mapping ──────────────────────────────────────────────
# Determines which chunking strategy is applied
SUBJECT_CATEGORY_MAP: dict[str, str] = {
    "mathematics":    "STEM",
    "math":           "STEM",
    "physics":        "STEM",
    "chemistry":      "STEM",
    "science":        "STEM",
    "biology":        "STEM",
    "history":        "HUMANITIES",
    "geography":      "HUMANITIES",
    "civics":         "HUMANITIES",
    "economics":      "HUMANITIES",
    "social_science": "HUMANITIES",
    "sst":            "HUMANITIES",
    "english":        "LANGUAGE_LIT",
    "odia":           "LANGUAGE_LIT",
    "hindi":          "LANGUAGE_LIT",
    "sanskrit":       "LANGUAGE_LIT",
    "regional":       "LANGUAGE_LIT",
}

# ─── Chunking Parameters per Category ────────────────────────────────────────
# chunk_size and chunk_overlap are in characters (approx 4 chars/token)
# STEM:          450–750 tokens  → ~1800–3000 chars
# HUMANITIES:    900–1300 tokens → ~3600–5200 chars
# LANGUAGE_LIT:  500–900 tokens  → ~2000–3600 chars
CHUNK_PARAMS: dict[str, dict] = {
    "STEM":         {"chunk_size": 2400, "chunk_overlap": 400},
    "HUMANITIES":   {"chunk_size": 4400, "chunk_overlap": 800},
    "LANGUAGE_LIT": {"chunk_size": 2800, "chunk_overlap": 480},
}

# ─── Validation ──────────────────────────────────────────────────────────────
def validate():
    """Raise early if critical config is missing."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_api_key_here":
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    if not EMBEDDING_MODEL:
        raise EnvironmentError("EMBEDDING_MODEL is not set in .env")
    if not LLM_MODEL:
        raise EnvironmentError("LLM_MODEL is not set in .env")
    if not os.path.isdir(BOOKS_DIR):
        raise FileNotFoundError(
            f"'books/' directory not found at: {BOOKS_DIR}\n"
            "Create it and add subject PDFs following the README structure."
        )
