from typing import List, Optional, Tuple
from .schemas import ChunkResponse
from .config import DEFAULT_MODEL_ID
from .llm_client import generate
from .provider import Generation
import logging

logger = logging.getLogger(__name__)

NO_SOURCES_ANSWER = "Nie znaleziono w notatkach informacji odpowiadających na Twoje pytanie."
FALLBACK_PREFIX = "LLM generation failed."


def build_prompt(query: str, sources: List[ChunkResponse]) -> Tuple[str, str]:
    """Builds the grounding prompt. Returns (prompt, context_block)."""
    # Construct context from sources
    context_parts = []
    for i, source in enumerate(sources, 1):
        file_name = source.metadata.file_name
        context_parts.append(f"--- Source {i} [{file_name}] ---")
        context_parts.append(source.text.strip())
        context_parts.append("")
        
    context_block = "\n".join(context_parts)
    
    prompt = f"""You are an assistant answering questions based only on the provided context notes.
Note that the context notes are written in Polish, and the question is in English.

Follow these rules strictly:
1. Answer the question in the same language as the question.
2. Answer using ONLY facts directly mentioned in the context. Translate technical terms appropriately across languages.
3. Rely on the context even if it is partial, as long as it contains the core answer.
4. If you are answering the question, you MUST include at least one bracketed source citation (e.g. "[Source 1]", "[Source 2]") at the end of your sentence or paragraph.
5. If the context does not contain any information relevant to the question at all, say: "I don't have enough information to answer that based on the notes." Do NOT append this refusal message if you have already answered the question.

Context:
{context_block}

Question:
{query}

Answer:"""
    return prompt, context_block


def generate_answer_with_meta(
    query: str, sources: List[ChunkResponse], model_id: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Generation:
    """Generates a grounded answer with the selected model; keeps token usage for cost tracking.
    Raises ValueError for a model id outside the allowlist or a temperature out of range."""
    model_id = model_id or DEFAULT_MODEL_ID
    if not sources:
        return Generation(text=NO_SOURCES_ANSWER, model_id=model_id)

    prompt, context_block = build_prompt(query, sources)
    try:
        return generate(prompt, model_id, temperature)
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to generate answer from LLM ({model_id}): {e}")
        # Fallback to the Stage 1 extractive behavior if LLM fails
        return Generation(
            text=f"{FALLBACK_PREFIX} Reason: {e}\nReturning raw context:\n\n" + context_block,
            model_id=model_id,
            error=str(e),
        )


def generate_answer(query: str, sources: List[ChunkResponse], model_id: Optional[str] = None) -> str:
    """Generates a grounded answer based solely on the sources."""
    return generate_answer_with_meta(query, sources, model_id).text
