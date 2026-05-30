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
1. Answer the question using ONLY the facts directly mentioned in the context. Translate the terms appropriately.
2. Rely on the context even if it is partial, as long as it contains the core answer to the question.
3. The context may contain some irrelevant chunks. Ignore any irrelevant information and do NOT refuse to answer just because some parts of the context are unrelated.
4. Cite the sources you used in your answer using bracketed notations like "[Source 1]", "[Source 2]", etc.
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
