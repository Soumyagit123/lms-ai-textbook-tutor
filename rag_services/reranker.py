import os
import structlog
import cohere
from typing import List, Dict, Any
import config

logger = structlog.get_logger()
COHERE_API_KEY = config.COHERE_API_KEY

class RerankerService:
    """
    Reranks vector search candidates using Cohere rerank models.
    Provides graceful fallback to original similarity order if COHERE_API_KEY is not set.
    """
    def __init__(self):
        if COHERE_API_KEY and COHERE_API_KEY != "your_cohere_api_key_here":
            self.co = cohere.AsyncClient(COHERE_API_KEY)
            logger.info("cohere_reranker_initialized")
        else:
            self.co = None
            logger.warning("cohere_not_configured", message="COHERE_API_KEY not found. Operating with native Qdrant similarity ranking.")

    async def rerank_results(self, query: str, documents: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Takes candidate documents from retrieval, sends them to Cohere,
        and returns the top_n most relevant documents with rerank scores.
        """
        if not documents:
            return []

        # Fallback if Cohere is not configured
        if not self.co:
            logger.info("using_qdrant_native_ranking", count=min(len(documents), top_n))
            return documents[:top_n]

        logger.info("reranking_documents_with_cohere", count=len(documents))

        # Build list of text snippets for Cohere
        doc_texts = []
        for doc in documents:
            content = doc.get("content") or doc.get("payload", {}).get("content", "")
            doc_texts.append(content)

        try:
            # We use rerank-multilingual-v3.0 to support English, Hindi, Odia, and other textbook languages
            response = await self.co.rerank(
                query=query,
                documents=doc_texts,
                top_n=min(top_n, len(documents)),
                model='rerank-multilingual-v3.0'
            )

            reranked_docs = []
            # Minimum relevance threshold (0.0 to 1.0) for Cohere to consider doc relevant
            MIN_RELEVANCE_THRESHOLD = 0.25
            for result in response.results:
                if result.relevance_score >= MIN_RELEVANCE_THRESHOLD:
                    original_doc = documents[result.index]
                    original_doc["rerank_score"] = result.relevance_score
                    reranked_docs.append(original_doc)

            logger.info("reranking_complete", top_n_returned=len(reranked_docs), original_count=len(documents))
            return reranked_docs

        except Exception as e:
            logger.error("cohere_rerank_failed_fallback_to_vector_order", error=str(e))
            return documents[:top_n]

reranker_service = RerankerService()
