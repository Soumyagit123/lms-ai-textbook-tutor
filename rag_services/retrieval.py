import structlog
from typing import List, Dict, Any, Optional
import config
from src.embedder import embed_query
from src.qdrant_uploader import get_client
from qdrant_client.models import Filter, FieldCondition, MatchValue
from .reranker import reranker_service

logger = structlog.get_logger()

class QdrantRetrievalService:
    """
    Retrieval Service connecting to Qdrant.
    Uses exact same embedder (src/embedder.py with 768-dim Matryoshka gemini-embedding-001)
    and queries class-specific collections (school_textbooks_class{standard}).
    """
    def __init__(self):
        self.client = get_client()

    async def retrieve(
        self,
        query: str,
        standard: int,
        subject: Optional[str] = None,
        chapter_number: Optional[int] = None,
        top_k: int = 15,
        top_n: int = 4
    ) -> List[Dict[str, Any]]:
        """
        1. Embeds query using task_type='RETRIEVAL_QUERY'
        2. Queries Qdrant collection school_textbooks_class{standard}
        3. Pre-filters by subject and/or chapter if specified
        4. Reranks candidates using reranker_service to top_n
        """
        collection_name = config.collection_name_for_class(standard)
        logger.info("starting_qdrant_retrieval", collection=collection_name, query=query, subject=subject)

        # 1. Embed query
        query_vector = embed_query(query)

        # 2. Build metadata filter conditions
        must_conditions = []
        if subject:
            # Normalize subject name (e.g., 'mathematics', 'science')
            must_conditions.append(
                FieldCondition(key="subject", match=MatchValue(value=subject.lower().strip()))
            )
        if chapter_number is not None:
            must_conditions.append(
                FieldCondition(key="chapter_number", match=MatchValue(value=chapter_number))
            )

        qdrant_filter = Filter(must=must_conditions) if must_conditions else None

        # 3. Query Qdrant with score_threshold to filter out unrelated results
        # Matryoshka cosine similarity typically needs at least ~0.45-0.50 for topical relevance
        score_threshold = 0.45
        try:
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    query_filter=qdrant_filter,
                    score_threshold=score_threshold,
                    limit=top_k,
                    with_payload=True
                )
                results = response.points
            else:
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    query_filter=qdrant_filter,
                    score_threshold=score_threshold,
                    limit=top_k,
                    with_payload=True
                )
        except Exception as e:
            logger.error("qdrant_search_failed", error=str(e), collection=collection_name)
            return []

        # 4. Format retrieved candidates
        candidates = []
        for hit in results:
            payload = hit.payload or {}
            candidates.append({
                "id": hit.id,
                "score": hit.score,
                "content": payload.get("content", ""),
                "standard": payload.get("standard"),
                "subject": payload.get("subject"),
                "book_title": payload.get("book_title", ""),
                "chapter_number": payload.get("chapter_number"),
                "chapter_name": payload.get("chapter_name", ""),
                "topic_name": payload.get("topic_name", ""),
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
                "has_math_formula": payload.get("has_math_formula", False),
                "has_diagram": payload.get("has_diagram", False),
                "diagrams": payload.get("diagrams", [])
            })

        logger.info("qdrant_candidates_retrieved", count=len(candidates))

        # 5. Rerank down to top_n
        reranked_docs = await reranker_service.rerank_results(query, candidates, top_n=top_n)
        return reranked_docs

retrieval_service = QdrantRetrievalService()
