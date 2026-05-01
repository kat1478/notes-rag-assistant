from typing import List
from .config import CHUNK_SIZE, CHUNK_OVERLAP
from .schemas import ChunkResponse, ChunkMetadata
import uuid

def chunk_text(text: str, file_name: str) -> List[ChunkResponse]:
    """Splits text into chunks of CHUNK_SIZE with CHUNK_OVERLAP."""
    chunks = []
    if not text:
        return chunks
        
    start = 0
    text_length = len(text)
    chunk_index = 0
    
    while start < text_length:
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]
        
        chunk_id = str(uuid.uuid4())
        metadata = ChunkMetadata(
            file_name=file_name,
            chunk_index=chunk_index,
            chunk_id=chunk_id
        )
        
        chunks.append(ChunkResponse(text=chunk_text, metadata=metadata))
        
        chunk_index += 1
        # Move forward, accounting for overlap
        start += (CHUNK_SIZE - CHUNK_OVERLAP)
        
        # Prevent infinite loops if overlap >= chunk_size
        if CHUNK_SIZE <= CHUNK_OVERLAP:
            break
            
    return chunks
