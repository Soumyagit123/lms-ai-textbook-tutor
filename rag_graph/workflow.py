import structlog
from langgraph.graph import StateGraph, START, END
from typing import Literal

from .state import StudentRAGState
from .nodes.guardrails_node import guardrails_node
from .nodes.resolver_node import context_resolver_node
from .nodes.validator_node import collection_validator_node
from .nodes.retrieval_node import retrieval_node
from .nodes.generator_node import generator_node

logger = structlog.get_logger()

# ─── Conditional Edge Functions ───────────────────────────────────────────────

def route_after_guardrails(state: StudentRAGState) -> Literal["context_resolver", "__end__"]:
    """If input is unsafe or injection detected, exit immediately."""
    if not state.get("is_safe", True):
        logger.info("routing_to_end_unsafe_query")
        return END
    return "context_resolver"

def route_after_validator(state: StudentRAGState) -> Literal["retrieval", "__end__"]:
    """If the standard textbook collection does not exist in Qdrant, exit with clear notification."""
    if not state.get("collection_exists", False):
        logger.info("routing_to_end_missing_collection", collection=state.get("collection_name"))
        return END
    return "retrieval"

def build_student_rag_graph() -> StateGraph:
    workflow = StateGraph(StudentRAGState)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("context_resolver", context_resolver_node)
    workflow.add_node("collection_validator", collection_validator_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("generator", generator_node)

    workflow.add_edge(START, "guardrails")
    workflow.add_conditional_edges("guardrails", route_after_guardrails)
    workflow.add_edge("context_resolver", "collection_validator")
    workflow.add_conditional_edges("collection_validator", route_after_validator)
    workflow.add_edge("retrieval", "generator")
    workflow.add_edge("generator", END)

    return workflow

def build_retrieval_subgraph() -> StateGraph:
    """Subgraph that runs guardrails -> context_resolver -> validator -> retrieval without generator."""
    workflow = StateGraph(StudentRAGState)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("context_resolver", context_resolver_node)
    workflow.add_node("collection_validator", collection_validator_node)
    workflow.add_node("retrieval", retrieval_node)

    workflow.add_edge(START, "guardrails")
    workflow.add_conditional_edges("guardrails", route_after_guardrails)
    workflow.add_edge("context_resolver", "collection_validator")
    workflow.add_conditional_edges("collection_validator", route_after_validator)
    workflow.add_edge("retrieval", END)

    return workflow

# Compile executable Pregel graphs
student_rag_graph = build_student_rag_graph().compile()
retrieval_rag_graph = build_retrieval_subgraph().compile()
