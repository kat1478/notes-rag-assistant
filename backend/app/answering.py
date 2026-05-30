from typing import List
from .schemas import ChunkResponse
from .llm_client import generate_from_ollama
import logging

logger = logging.getLogger(__name__)

def generate_answer(query: str, sources: List[ChunkResponse]) -> str:
    """Generates a grounded answer based solely on the sources using local LLM."""
    if not sources:
        return "Nie znaleziono w notatkach informacji odpowiadających na Twoje pytanie."
        
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

    try:
        llm_response = generate_from_ollama(prompt)
        return llm_response
    except Exception as e:
        logger.error(f"Failed to generate answer from LLM: {e}")
        # Fallback to the Stage 1 extractive behavior if LLM fails
        return "LLM generation failed. Returning raw context:\n\n" + context_block
