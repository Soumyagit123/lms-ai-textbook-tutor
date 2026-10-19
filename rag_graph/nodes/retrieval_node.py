import structlog
from typing import Dict, Any
from rag_services.retrieval import retrieval_service
try:
    from ..state import StudentRAGState
except (ImportError, ValueError):
    from rag_graph.state import StudentRAGState

logger = structlog.get_logger()

async def retrieval_node(state: StudentRAGState) -> Dict[str, Any]:
    """
    Executes dense vector search against the specific class collection in Qdrant,
    applying detected subject filters and reranking.
    Catches any embedding or database errors safely.
    """
    query = state.get("query", "")
    standard = state.get("standard", 10)
    subject = state.get("detected_subject")
    chapter_number = state.get("chapter_number")
    top_k = state.get("top_k", 12)
    top_n = state.get("top_n", 4)

    try:
        docs = await retrieval_service.retrieve(
            query=query,
            standard=standard,
            subject=subject,
            chapter_number=chapter_number,
            top_k=top_k,
            top_n=top_n
        )

        if not docs and subject:
            # Fallback attempt: if subject was too restrictive, search without subject filter
            logger.info("retrying_retrieval_without_subject_filter", subject=subject)
            docs = await retrieval_service.retrieve(
                query=query,
                standard=standard,
                subject=None,
                chapter_number=chapter_number,
                top_k=top_k,
                top_n=top_n
            )

        citations = []
        diagrams = []
        seen_diagram_urls = set()
        q_lower = query.lower()

        for doc in docs:
            citations.append({
                "book_title": doc.get("book_title", "Textbook"),
                "chapter_number": doc.get("chapter_number"),
                "chapter_name": doc.get("chapter_name", ""),
                "page_start": doc.get("page_start"),
                "page_end": doc.get("page_end"),
                "topic_name": doc.get("topic_name", "")
            })
            for d in doc.get("diagrams", []):
                url = d.get("image_url")
                if url and url not in seen_diagram_urls:
                    seen_diagram_urls.add(url)
                    diagrams.append(d)

        # Prioritize diagrams that mention question keywords in their caption
        query_words = [w for w in q_lower.split() if len(w) > 3]
        if query_words and len(diagrams) > 2:
            def diagram_relevance(diag):
                cap = (diag.get("caption") or "").lower()
                return sum(1 for w in query_words if w in cap)
            # Sort relevant captions first and take top 2 to avoid overwhelming client
            diagrams.sort(key=diagram_relevance, reverse=True)
            diagrams = diagrams[:2]

        return {
            "retrieved_docs": docs,
            "citations": citations,
            "diagrams": diagrams,
            "has_error": False
        }

    except Exception as e:
        logger.error("retrieval_node_error", error=str(e), standard=standard, subject=subject)
        return {
            "retrieved_docs": [],
            "citations": [],
            "diagrams": [],
            "has_error": True,
            "error_message": f"Textbook retrieval error: {str(e)}",
            "errors": [f"Retrieval failure: {str(e)}"]
        }
