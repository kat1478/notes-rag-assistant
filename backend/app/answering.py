from typing import List
from .schemas import ChunkResponse

def generate_answer(query: str, sources: List[ChunkResponse]) -> str:
    """Generates a grounded answer based solely on the sources."""
    if not sources:
        return "Nie znaleziono w notatkach informacji odpowiadających na Twoje pytanie."
        
    # Extractive summarization for Stage 1
    answer_parts = ["Znaleziono następujące fragmenty w Twoich notatkach:\n"]
    
    for i, source in enumerate(sources, 1):
        file_name = source.metadata.file_name
        score = f"{source.score:.2f}" if source.score is not None else "N/A"
        answer_parts.append(f"--- Źródło {i} [{file_name}, trafność: {score}] ---")
        answer_parts.append(source.text.strip())
        answer_parts.append("")
        
    return "\n".join(answer_parts)
