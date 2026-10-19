import re
import structlog
from typing import Dict, Any, Optional
import config
from ..state import StudentRAGState

logger = structlog.get_logger()

# Subject keyword map for auto-detection if subject is not provided by UI
SUBJECT_KEYWORDS = {
    "mathematics": [
        "equation", "quadratic", "polynomial", "arithmetic", "progression", "ap",
        "triangles", "coordinate", "geometry", "trigonometry", "circle", "area",
        "surface", "volume", "statistics", "probability", "theorem", "solve for x",
        "hypotenuse", "tangent", "secant", "mean", "median", "mode", "lcm", "hcf",
        "factorisation", "formula", "roots"
    ],
    "science": [
        "chemical", "reaction", "acid", "base", "salt", "metal", "non-metal", "carbon",
        "compound", "life processes", "photosynthesis", "respiration", "heart", "circulation",
        "excretion", "control", "coordination", "neuron", "brain", "reproduction",
        "heredity", "light", "reflection", "refraction", "human eye", "electricity",
        "ohm's law", "resistor", "magnetic", "solenoid", "motor", "generator", "ozone",
        "ecosystem", "environment"
    ],
    "social_science": [
        "history", "nationalism", "europe", "india", "industrialisation", "print culture",
        "resources", "development", "forest", "wildlife", "water resources", "agriculture",
        "minerals", "energy", "manufacturing", "power sharing", "federalism", "gender",
        "religion", "political parties", "democracy", "development", "sectors", "economy",
        "money", "credit", "globalisation", "consumer rights"
    ],
    "english": [
        "poem", "poet", "stanza", "character sketch", "summary", "theme", "grammar",
        "tense", "active passive", "direct indirect", "preposition", "conjunction"
    ]
}

def detect_subject_from_query(query: str) -> Optional[str]:
    """Senses subject from query keywords if not provided by UI."""
    q_lower = query.lower()
    
    # Check for direct math formulas
    if re.search(r'[\d\w]\^2|[=+\-*/]{2,}|sin|cos|tan|\bax\^2\b|\bsqrt\b', q_lower):
        return "mathematics"

    scores = {}
    for subj, keywords in SUBJECT_KEYWORDS.items():
        score = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', q_lower))
        scores[subj] = score

    best_subj, best_score = max(scores.items(), key=lambda x: x[1])
    return best_subj if best_score > 0 else None

async def context_resolver_node(state: StudentRAGState) -> Dict[str, Any]:
    """
    Resolves student class/standard and senses the subject from query if missing.
    Constructs the target Qdrant collection name: school_textbooks_class{standard}.
    """
    try:
        raw_std = state.get("standard", 10)
        # Parse standard safely (handles integers or strings like "10th", "10", "class 10")
        if isinstance(raw_std, str):
            digits = re.findall(r'\d+', raw_std)
            standard = int(digits[0]) if digits else 10
        else:
            standard = int(raw_std)

        # Subject sensing
        subject = state.get("subject")
        if subject and str(subject).strip().lower() not in ["", "all", "none"]:
            detected_subject = str(subject).strip().lower()
        else:
            detected_subject = detect_subject_from_query(state.get("query", ""))

        collection_name = config.collection_name_for_class(standard)

        logger.info(
            "context_resolved",
            student=state.get("student_name"),
            standard=standard,
            collection=collection_name,
            subject=detected_subject
        )

        return {
            "standard": standard,
            "detected_subject": detected_subject,
            "collection_name": collection_name,
            "has_error": False
        }

    except Exception as e:
        logger.error("context_resolver_failed", error=str(e))
        return {
            "standard": 10,
            "collection_name": config.collection_name_for_class(10),
            "detected_subject": None,
            "errors": [f"Context resolution warning: {str(e)}"],
            "has_error": False
        }
