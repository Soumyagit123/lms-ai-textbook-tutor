from typing import TypedDict, Optional, List, Dict, Any

class StudentRAGState(TypedDict, total=False):
    # Student Profile & Input
    student_id: Optional[str]
    student_name: Optional[str]
    standard: int
    subject: Optional[str]
    query: str
    chapter_number: Optional[int]
    top_k: int
    top_n: int

    # Routing & Discovery
    collection_name: str
    collection_exists: bool
    detected_subject: Optional[str]
    
    # Safety & Status
    is_safe: bool
    safety_reason: Optional[str]

    # Documents & Generation
    retrieved_docs: List[Dict[str, Any]]
    final_answer: Optional[str]
    citations: List[Dict[str, Any]]
    diagrams: List[Dict[str, str]]

    # Error handling & Diagnostics
    errors: List[str]
    has_error: bool
    error_message: Optional[str]
