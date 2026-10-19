"""
Embedder — converts text to vectors using Google gemini-embedding-001
(768 dimensions via Matryoshka output_dimensionality).
Supports multilingual text including Odia, Hindi, English, etc.

NOTE: Diagrams are NOT embedded. They are extracted to disk and linked via
payload['diagrams']; retrieval is driven by the chunk text (which includes the
figure captions in the banner), so a text embedding model is the right choice.
"""

import time
from google import genai
from google.genai import types
import config

_client = None


def _get_client() -> genai.Client:
    """Create the Google GenAI client once."""
    global _client
    if _client is None:
        if not config.GOOGLE_API_KEY:
            raise EnvironmentError("GOOGLE_API_KEY is not set in .env")
        _client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _client


def _embed(text: str, task_type: str) -> list[float]:
    result = _get_client().models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=config.VECTOR_SIZE,
        ),
    )
    return list(result.embeddings[0].values)


def embed_document(text: str, retries: int = 8) -> list[float]:
    """
    Embed a single document text for storage in Qdrant.
    Uses task_type='RETRIEVAL_DOCUMENT' for ingestion.

    Returns a VECTOR_SIZE-dimensional float vector.
    Retries with backoff; on 429 (free-tier per-minute quota) waits ~65s so the
    quota window resets. Throttles requests to stay under ~60 req/min.
    """
    for attempt in range(retries):
        try:
            vec = _embed(text, "RETRIEVAL_DOCUMENT")
            time.sleep(1.0)  # throttle
            return vec
        except Exception as e:
            if attempt < retries - 1:
                is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                wait = 65 if is_rate_limit else 2 ** attempt
                print(f"    [WARN] Embedding failed (attempt {attempt + 1}): {str(e)[:90]}... Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Embedding failed after {retries} attempts: {e}") from e


def embed_query(query: str) -> list[float]:
    """
    Embed a user query for Qdrant search.
    Uses task_type='RETRIEVAL_QUERY' — must use the same model/dimension as documents.
    """
    return _embed(query, "RETRIEVAL_QUERY")


def embed_batch(texts: list[str], delay_ms: int = 100) -> list[list[float]]:
    """
    Embed a list of texts sequentially with a small delay to avoid rate limits.
    """
    vectors = []
    for i, text in enumerate(texts):
        vectors.append(embed_document(text))
        if delay_ms > 0 and i < len(texts) - 1:
            time.sleep(delay_ms / 1000)
    return vectors
