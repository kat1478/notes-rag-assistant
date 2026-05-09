from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from .schemas import ChunkResponse
from .storage import load_chunks
from .config import TOP_K
from .vector_store import search_chunks
import logging

logger = logging.getLogger(__name__)

def perform_search_tfidf(query: str, top_k: int = TOP_K) -> List[ChunkResponse]:
    """Uses TF-IDF and Cosine Similarity to find relevant chunks (Stage 1 fallback)."""
    chunks = load_chunks()
    if not chunks:
        return []
        
    texts = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words=None)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        query_vec = vectorizer.transform([query])
    except ValueError:
        return []
        
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score > 0.0:
            chunk = chunks[idx]
            chunk_copy = ChunkResponse(**chunk.model_dump())
            chunk_copy.score = float(score)
            results.append(chunk_copy)
            
    return results

def perform_search(query: str, top_k: int = TOP_K, use_semantic: bool = True) -> List[ChunkResponse]:
    """Primary retrieval function, using Chroma by default."""
    if use_semantic:
        try:
            results = search_chunks(query, top_k)
            if results:
                return results
            logger.info("Semantic search returned no results, falling back to TF-IDF.")
        except Exception as e:
            logger.error(f"Semantic search exception: {e}. Falling back to TF-IDF.")
            
    # Fallback or explicit TF-IDF
    return perform_search_tfidf(query, top_k)
