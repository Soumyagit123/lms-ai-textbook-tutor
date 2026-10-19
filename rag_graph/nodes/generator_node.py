import structlog
from typing import Dict, Any
from rag_services.llm import student_llm_service
from ..state import StudentRAGState

logger = structlog.get_logger()

async def generator_node(state: StudentRAGState) -> Dict[str, Any]:
    """
    Generates student explanation using Gemini with LaTeX math and diagram formatting.
    Robustly catches quota limits (429), API key issues, or generation errors,
    formatting a clear message for the student.
    """
    query = state.get("query", "")
    docs = state.get("retrieved_docs", [])
    standard = state.get("standard", 10)
    student_name = state.get("student_name")
    subject = state.get("detected_subject")

    # If no documents were retrieved
    if not docs:
        subj_str = f" in {subject}" if subject else ""
        no_info_msg = (
            f"Hello{f' {student_name}' if student_name else ''}! "
            f"I could not find matching information in your Class {standard} textbook{subj_str} for this specific question. "
            "Please try rephrasing your question or checking your chapter name."
        )
        return {
            "final_answer": no_info_msg,
            "has_error": False
        }

    try:
        answer = await student_llm_service.generate_response(
            query=query,
            context_docs=docs,
            standard=standard
        )

        # Prepend personalized greeting if student name was provided and not already in answer
        if student_name and not answer.startswith("Hello") and not answer.startswith(f"Hi {student_name}"):
            answer = f"Hello {student_name}!\n\n{answer}"

        return {
            "final_answer": answer,
            "has_error": False
        }

    except Exception as e:
        err_msg = str(e)
        logger.error("generator_node_error", error=err_msg)

        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            user_error = "⚠️ The AI tutor is experiencing high traffic (rate limit reached). Please try again in 30 seconds."
        elif "API_KEY" in err_msg or "403" in err_msg:
            user_error = "⚠️ AI service configuration error: Google API key is invalid or quota expired."
        else:
            user_error = f"⚠️ An unexpected error occurred while generating your textbook explanation: {err_msg[:120]}"

        return {
            "final_answer": user_error,
            "has_error": True,
            "error_message": err_msg,
            "errors": [f"LLM Generation failure: {err_msg}"]
        }
