import os
import structlog
from typing import List, Dict, Any, AsyncGenerator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import config

logger = structlog.get_logger()

class StudentLLMService:
    """
    LLM service tailored for school student Q&A:
    - Strictly follows authentic textbook syllabus
    - Retains LaTeX formatting for equations and math symbols ($...$ and $$...$$)
    - Emits markdown diagram tags ![caption](image_url) when diagrams are provided
    - Explicit textbook citations (Book, Chapter, Page range)
    """
    def __init__(self):
        # Read API key and model from config
        api_key = config.GOOGLE_API_KEY
        # Use LLM model from config (.env)
        model_name = config.LLM_MODEL
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            max_output_tokens=2048,
            google_api_key=api_key,
            max_retries=3,
            timeout=60.0
        )

    def _build_context_and_diagrams(self, context_docs: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, str]]]:
        context_parts = []
        available_diagrams = []

        for idx, doc in enumerate(context_docs, 1):
            source_info = f"Book: {doc.get('book_title', 'Textbook')}, Chapter {doc.get('chapter_number', 'N/A')}: {doc.get('chapter_name', '')}, Pages: {doc.get('page_start', '?')}-{doc.get('page_end', '?')}"
            topic = doc.get("topic_name") or "General"
            content = doc.get("content", "")

            # Collect diagrams if any
            diagrams = doc.get("diagrams") or []
            for d in diagrams:
                caption = d.get("caption") or "Textbook Illustration"
                url = d.get("image_url") or ""
                if url:
                    available_diagrams.append({"caption": caption, "url": url})

            context_parts.append(
                f"--- [SOURCE EXCERPT {idx}] ---\n"
                f"CITATION: {source_info}\n"
                f"TOPIC: {topic}\n\n"
                f"{content}\n"
            )

        context_str = "\n\n".join(context_parts)
        return context_str, available_diagrams

    def _build_system_prompt(self, context_str: str, available_diagrams: List[Dict[str, str]], standard: int) -> str:
        diagram_instructions = ""
        if available_diagrams:
            diagram_list = "\n".join([f"- Caption: {d['caption']} -> URL: {d['url']}" for d in available_diagrams])
            diagram_instructions = (
                "\nAVAILABLE TEXTBOOK DIAGRAMS:\n"
                f"{diagram_list}\n"
                "If relevant to explaining the answer, include the diagram in your answer using exact markdown format: "
                "![Caption](URL)\n"
            )

        system_prompt = (
            f"You are an expert school textbook academic tutor for Class {standard} students.\n"
            "Your task is to provide accurate, easy-to-understand explanations strictly based on the approved textbook excerpts below.\n\n"
            "RULES:\n"
            "1. Grounding: Answer ONLY based on the provided textbook excerpts. If the student's question cannot be answered from the provided excerpts (for instance, if it asks about a completely different topic or subject not present in the excerpts), state politely and clearly that this specific topic is not covered in the currently available chapter of their textbook.\n"
            "2. Mathematical Symbols & Formatting:\n"
            "   - Wrap standalone multi-term equations or multi-line steps in double dollar signs $$...$$ (e.g. $$p(x) = ax^3 + bx^2 + cx + d$$ and $$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$) so they render centered and clearly spaced.\n"
            "   - Wrap inline variables and small expressions in single dollar signs $...$ (e.g. $\\alpha$, $\\beta$, $\\gamma$, $x = 3$, $p(3) = 0$).\n"
            "   - Write fractions clearly using $\\frac{numerator}{denominator}$ (e.g. $\\frac{-1}{3}$ or $\\frac{c}{a}$).\n"
            "   - Use proper LaTeX superscripts for powers ($x^2$, $x^3$) and subscripts ($a_1$, $x_0$). Never leave unformatted symbols or raw broken slashes.\n"
            "3. Explanations: Explain step-by-step with clear, friendly, and pedagogical language suitable for this class level. Use bolding and bullet steps for clarity.\n"
            "4. Citations & References: Do NOT write a manual 'Textbook Citations:' or citation list at the end of your text response. The student interface automatically displays verified syllabus citation badges separately below your answer.\n"
            f"{diagram_instructions}\n"
            f"TEXTBOOK EXCERPTS:\n{context_str}\n"
        )
        return system_prompt


    async def generate_response(self, query: str, context_docs: List[Dict[str, Any]], standard: int) -> str:
        """
        Generates standard complete text answer.
        """
        logger.info("generating_student_rag_response", standard=standard)
        context_str, available_diagrams = self._build_context_and_diagrams(context_docs)

        system_prompt = self._build_system_prompt(context_str, available_diagrams, standard)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]

        try:
            response = await self.llm.ainvoke(messages)
            content = response.content
            if isinstance(content, list):
                # Extract text chunks if returned as list of dicts/parts
                text_parts = [
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                ]
                return "".join(text_parts)
            return str(content)
        except Exception as e:
            logger.error("llm_generation_failed", error=str(e))
            raise

    async def generate_response_stream(
        self, query: str, context_docs: List[Dict[str, Any]], standard: int
    ) -> AsyncGenerator[str, None]:
        """
        Yields tokens for real-time SSE streaming.
        """
        logger.info("generating_student_rag_response_stream", standard=standard)
        context_str, available_diagrams = self._build_context_and_diagrams(context_docs)

        system_prompt = self._build_system_prompt(context_str, available_diagrams, standard)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]

        # Execute streaming with retry on transient 503 high demand
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                async for chunk in self.llm.astream(messages):
                    content = chunk.content
                    if isinstance(content, list):
                        text = "".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in content])
                        if text:
                            yield text
                    elif content:
                        yield str(content)
                return
            except Exception as e:
                err_str = str(e)
                if ("503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str) and attempt < max_attempts - 1:
                    logger.warning("gemini_503_retry", attempt=attempt+1, message="Retrying after transient 503...")
                    import asyncio
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                logger.error("llm_stream_failed", error=err_str)
                raise

student_llm_service = StudentLLMService()
