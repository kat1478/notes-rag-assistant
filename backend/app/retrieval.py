from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from .schemas import ChunkResponse
from .storage import load_chunks
from .config import TOP_K

def perform_search(query: str, top_k: int = TOP_K) -> List[ChunkResponse]:
    """Uses TF-IDF and Cosine Similarity to find relevant chunks."""
    chunks = load_chunks()
    if not chunks:
        return []
        
    texts = [chunk.text for chunk in chunks]
    
    # Initialize and fit TF-IDF
    vectorizer = TfidfVectorizer(stop_words=None)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        query_vec = vectorizer.transform([query])
    except ValueError:
        # Happens if vocab is empty or all words are ignored
        return []
        
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top_k indices
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score > 0.0:  # Only return chunks with some similarity
            chunk = chunks[idx]
            # Create a copy so we don't mutate the loaded cache if it was persistent
            chunk_copy = ChunkResponse(**chunk.model_dump())
            chunk_copy.score = float(score)
            results.append(chunk_copy)
            
    return results
