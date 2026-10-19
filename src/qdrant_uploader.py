"""
Qdrant Uploader — connects to Qdrant, creates the collection with correct
indexes if it doesn't exist, and uploads chunk payloads with their vectors.

Each class/standard gets its own collection: school_textbooks_class{N}
Use config.collection_name_for_class(standard) to get the right name.
"""

import uuid
import config
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
    Filter,
    FieldCondition,
    MatchValue,
)


def get_client() -> QdrantClient:
    """Create and return a Qdrant client (local or cloud)."""
    if config.QDRANT_API_KEY:
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    return QdrantClient(url=config.QDRANT_URL)


def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    """
    Create the given collection if it doesn't exist,
    and set up all payload indexes for fast pre-filtering.

    Args:
        collection_name: e.g. 'school_textbooks_class10'
    """
    existing = [c.name for c in client.get_collections().collections]

    if collection_name not in existing:
        print(f"  [Qdrant] Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=config.VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"  [Qdrant] Collection created with vector size {config.VECTOR_SIZE} (Cosine).")
    else:
        print(f"  [Qdrant] Collection '{collection_name}' already exists. ✓")

    # Create payload indexes for <5ms pre-filtering during search
    _create_indexes(client, collection_name)


def _create_indexes(client: QdrantClient, collection_name: str) -> None:
    """Create payload indexes for frequently filtered fields."""
    indexes = [
        ("standard",       PayloadSchemaType.INTEGER),
        ("subject",        PayloadSchemaType.KEYWORD),
        ("chapter_number", PayloadSchemaType.INTEGER),
        ("topic_name",     PayloadSchemaType.KEYWORD),
        ("has_diagram",    PayloadSchemaType.BOOL),
        ("content_type",   PayloadSchemaType.KEYWORD),
        ("pdf_mode",       PayloadSchemaType.KEYWORD),
        ("board",          PayloadSchemaType.KEYWORD),
    ]

    for field_name, schema_type in indexes:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )
        except Exception:
            pass  # Index already exists — safe to ignore


def count_chapter_chunks(
    client: QdrantClient,
    collection_name: str,
    standard: int,
    subject: str,
    chapter_num: int,
) -> int:
    """Return how many points already exist for a chapter (used to resume runs)."""
    return client.count(
        collection_name=collection_name,
        count_filter=Filter(
            must=[
                FieldCondition(key="standard",       match=MatchValue(value=standard)),
                FieldCondition(key="subject",        match=MatchValue(value=subject)),
                FieldCondition(key="chapter_number", match=MatchValue(value=chapter_num)),
            ]
        ),
        exact=True,
    ).count


def delete_chapter_chunks(
    client: QdrantClient,
    collection_name: str,
    standard: int,
    subject: str,
    chapter_num: int,
) -> int:
    """
    Delete all existing chunks for a specific chapter before re-ingesting.
    This ensures safe re-ingestion without duplicates.
    Returns the number of deleted points.
    """
    result = client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(key="standard",       match=MatchValue(value=standard)),
                FieldCondition(key="subject",        match=MatchValue(value=subject)),
                FieldCondition(key="chapter_number", match=MatchValue(value=chapter_num)),
            ]
        ),
    )
    return getattr(result, 'deleted', 0)


def upload_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks_with_vectors: list[tuple[dict, list[float]]],
    batch_size: int = 50,
) -> int:
    """
    Upload chunk payloads and their vectors to Qdrant in batches.

    Args:
        collection_name:      Target Qdrant collection (e.g. 'school_textbooks_class10').
        chunks_with_vectors:  List of (payload_dict, vector) tuples.
        batch_size:           Number of points per upload batch.

    Returns:
        Total number of points uploaded.
    """
    total = 0
    for i in range(0, len(chunks_with_vectors), batch_size):
        batch = chunks_with_vectors[i:i + batch_size]
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
            for payload, vector in batch
        ]
        client.upsert(
            collection_name=collection_name,
            points=points,
        )
        total += len(points)

    return total


def get_collection_info(client: QdrantClient, collection_name: str) -> dict:
    """Return basic stats about a collection."""
    info = client.get_collection(collection_name)
    return {
        "name":         collection_name,
        "total_points": info.points_count,
        "vector_size":  config.VECTOR_SIZE,
        "status":       str(info.status),
    }
