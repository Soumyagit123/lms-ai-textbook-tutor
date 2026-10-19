import structlog
from typing import Dict, Any
from src.qdrant_uploader import get_client
from ..state import StudentRAGState

logger = structlog.get_logger()

async def collection_validator_node(state: StudentRAGState) -> Dict[str, Any]:
    """
    Validates whether the student's standard collection exists in Qdrant.
    If a collection (e.g. school_textbooks_class9) is not yet ingested,
    gracefully informs the student with available classes rather than throwing a 500 error.
    """
    collection_name = state.get("collection_name", "")
    standard = state.get("standard", 10)
    student_name = state.get("student_name") or "Student"

    try:
        client = get_client()
        collections_resp = client.get_collections()
        existing_names = [c.name for c in collections_resp.collections]

        if collection_name in existing_names:
            logger.info("collection_verified_ready", collection=collection_name)
            return {
                "collection_exists": True,
                "has_error": False
            }
        else:
            # Extract available classes from existing collections
            available_classes = []
            for name in existing_names:
                if "class" in name:
                    class_part = name.split("class")[-1]
                    available_classes.append(f"Class {class_part}")

            avail_str = ", ".join(available_classes) if available_classes else "None currently"
            message = (
                f"Hello {student_name}! The textbook library for Class {standard} is not yet available or is currently "
                f"being ingested into the system. Available textbook classes at present: {avail_str}."
            )
            logger.warning("collection_not_found", requested=collection_name, available=existing_names)

            return {
                "collection_exists": False,
                "has_error": True,
                "error_message": f"Class {standard} textbooks not ingested yet.",
                "final_answer": message,
                "retrieved_docs": [],
                "citations": []
            }

    except Exception as e:
        logger.error("collection_validation_error", error=str(e))
        # Network/connection error to Qdrant
        return {
            "collection_exists": False,
            "has_error": True,
            "error_message": f"Unable to reach textbook database: {str(e)}",
            "final_answer": "⚠️ Unable to connect to the textbook vector database. Please ensure Qdrant service is running.",
            "retrieved_docs": [],
            "citations": []
        }
