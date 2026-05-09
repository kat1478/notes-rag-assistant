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
    
    prompt = f"""You are a helpful assistant answering questions based only on the provided context.
Follow these rules strictly:
1. Answer ONLY from the provided context.
2. If the context is insufficient to answer the question, say "I don't have enough information to answer that based on the notes."
3. Do not fabricate or invent facts.
4. Include source references (e.g., "[Source 1]") in your final output.

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
