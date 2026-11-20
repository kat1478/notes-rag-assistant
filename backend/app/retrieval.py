from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, NamedTuple
from .schemas import ChunkResponse, ChunkMetadata
from .config import TOP_K
from .vector_store import search_chunks, get_collection
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

def expand_chunks_with_neighbors(chunks: List[ChunkResponse]) -> List[ChunkResponse]:
    """
    Expands retrieved chunks with their immediate neighbors (index-1 and index+1) from Chroma
    to resolve context fragmentation while deduplicating overlaps.
    """
    if not chunks:
        return []
        
    try:
        collection = get_collection()
    except Exception as e:
        logger.error(f"Cannot access Chroma collection for context expansion: {e}")
        return chunks
        
    # Group retrieved chunks by file_name to minimize Chroma queries
    files_to_query = {chunk.metadata.file_name for chunk in chunks}
    
    all_chunks_by_file = {}
    for file_name in files_to_query:
        try:
            res = collection.get(where={"file_name": file_name}, include=["documents", "metadatas"])
            if res and res["documents"]:
                file_chunks = []
                for i in range(len(res["documents"])):
                    doc = res["documents"][i]
                    meta = res["metadatas"][i]
                    metadata = ChunkMetadata(**meta)
                    file_chunks.append(ChunkResponse(
                        text=doc,
                        metadata=metadata,
                        score=None
                    ))
                all_chunks_by_file[file_name] = {c.metadata.chunk_index: c for c in file_chunks}
        except Exception as e:
            logger.error(f"Error fetching neighbor chunks for {file_name}: {e}")
            
    expanded_chunks_dict = {}
    
    for chunk in chunks:
        file_name = chunk.metadata.file_name
        curr_idx = chunk.metadata.chunk_index
        
        # Original chunk retains its retrieval score
        expanded_chunks_dict[chunk.metadata.chunk_id] = chunk
        
        # Fetch adjacent neighbors (chunk_index - 1 and chunk_index + 1)
        if file_name in all_chunks_by_file:
            file_map = all_chunks_by_file[file_name]
            for neighbor_idx in (curr_idx - 1, curr_idx + 1):
                if neighbor_idx in file_map:
                    neighbor_chunk = file_map[neighbor_idx]
                    chunk_id = neighbor_chunk.metadata.chunk_id
                    if chunk_id not in expanded_chunks_dict:
                        expanded_chunks_dict[chunk_id] = neighbor_chunk
                        
    # Sort final list by file_name and chunk_index for sequential reading by LLM
    sorted_expanded = sorted(
        expanded_chunks_dict.values(),
        key=lambda c: (c.metadata.file_name, c.metadata.chunk_index)
    )
    
    return sorted_expanded

class Retrieval(NamedTuple):
    results: List[ChunkResponse]
    mode: str  # "semantic", "tfidf_fallback" or "none"


def retrieve(query: str, top_k: int = TOP_K, use_semantic: bool = True, expand_context: bool = True) -> Retrieval:
    """Chroma semantic search with TF-IDF fallback and context expansion; also reports which path produced the results."""
    if use_semantic:
        try:
            results = search_chunks(query, top_k)
            if results:
                return Retrieval(expand_chunks_with_neighbors(results) if expand_context else results, "semantic")
            logger.info("Semantic search returned no results, falling back to TF-IDF.")
        except Exception as e:
            logger.error(f"Semantic search exception: {e}. Falling back to TF-IDF.")

    # Fallback or explicit TF-IDF
    tfidf_results = perform_search_tfidf(query, top_k)
    if not tfidf_results:
        return Retrieval([], "none")
    return Retrieval(expand_chunks_with_neighbors(tfidf_results) if expand_context else tfidf_results, "tfidf_fallback")


def perform_search(
    query: str,
    top_k: int = TOP_K,
    use_semantic: bool = True,
    expand_context: bool = True
) -> List[ChunkResponse]:
    """Primary retrieval function (results only); see `retrieve` for the path that ran."""
    return retrieve(query, top_k, use_semantic, expand_context).results

# Inline import to avoid circular dependency
from .storage import load_chunks
