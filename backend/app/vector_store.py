import logging
import chromadb
from typing import List
from .config import CHROMA_DIR
from .schemas import ChunkResponse, ChunkMetadata
from .embeddings import embed_text, embed_texts

logger = logging.getLogger(__name__)

# Global Chroma client
_client = None
COLLECTION_NAME = "notes_chunks"

def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        try:
            logger.info(f"Initializing Chroma client at {CHROMA_DIR}")
            _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        except Exception as e:
            logger.error(f"Failed to initialize Chroma client: {e}")
            raise RuntimeError(f"Chroma init failure: {e}")
    return _client

def get_collection():
    client = get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)

def upsert_chunks(chunks: List[ChunkResponse], file_name: str):
    """
    Upserts chunks to Chroma.
    Implements a re-ingestion strategy by first deleting any existing chunks
    for the given file_name to avoid duplication.
    """
    if not chunks:
        return

    collection = get_collection()
    
    # Delete existing chunks for this file to prevent duplicates
    try:
        collection.delete(where={"file_name": file_name})
        logger.info(f"Deleted old chunks for {file_name} from vector store.")
    except Exception as e:
        logger.warning(f"Could not delete old chunks for {file_name}: {e}")

    ids = [chunk.metadata.chunk_id for chunk in chunks]
    texts = [chunk.text for chunk in chunks]
    metadatas = [chunk.metadata.model_dump() for chunk in chunks]
    
    try:
        embeddings = embed_texts(texts)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        logger.info(f"Added {len(chunks)} chunks for {file_name} to Chroma.")
    except Exception as e:
        logger.error(f"Failed to upsert chunks to Chroma: {e}")
        raise RuntimeError(f"Chroma upsert failure: {e}")

def search_chunks(query: str, top_k: int) -> List[ChunkResponse]:
    """
    Performs semantic search using Chroma and the embedding model.
    """
    collection = get_collection()
    
    if collection.count() == 0:
        return []

    try:
        query_embedding = embed_text(query)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
            
        chunk_responses = []
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i] if 'distances' in results and results['distances'] else None
            
            metadata = ChunkMetadata(**meta)
            chunk = ChunkResponse(
                text=doc,
                metadata=metadata,
                score=float(dist) if dist is not None else None
            )
            chunk_responses.append(chunk)
            
        return chunk_responses
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []
