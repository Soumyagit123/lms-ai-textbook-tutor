import structlog
from typing import Dict, Any
from rag_services.guardrails import guardrails_service
from ..state import StudentRAGState

logger = structlog.get_logger()

async def guardrails_node(state: StudentRAGState) -> Dict[str, Any]:
    """
    Checks the student prompt against prompt injection and malicious keywords.
    Provides robust exception handling so that any unexpected check error won't crash the pipeline.
    """
    query = state.get("query", "")
    try:
        safety_result = await guardrails_service.check_prompt_safety(query)
        
        if not safety_result.get("is_safe", True):
            reason = safety_result.get("reason", "Query blocked by academic safety policy.")
            logger.warning("query_blocked_by_guardrails", reason=reason)
            return {
                "is_safe": False,
                "safety_reason": reason,
                "final_answer": f"⚠️ Request Blocked: {reason}",
                "has_error": True,
                "error_message": reason,
                "retrieved_docs": [],
                "citations": []
            }

        return {
            "is_safe": True,
            "safety_reason": None,
            "has_error": False
        }

    except Exception as e:
        logger.error("guardrails_check_error", error=str(e))
        # If guardrails check encounters an internal error, fail open with warning or safe fallback
        return {
            "is_safe": True,
            "errors": [f"Guardrails non-fatal warning: {str(e)}"],
            "has_error": False
        }
