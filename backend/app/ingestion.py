from fastapi import UploadFile
from typing import List
from .storage import save_raw_file, save_chunks
from .chunking import chunk_text
from .schemas import ChunkResponse
from .vector_store import upsert_chunks
import logging

logger = logging.getLogger(__name__)

async def process_upload(file: UploadFile) -> List[ChunkResponse]:
    """Reads file, saves it, chunks it, and saves chunks."""
    content = await file.read()
    
    # Save raw file
    save_raw_file(file.filename, content)
    
    # Decode text
    text = content.decode("utf-8", errors="replace")
    
    # Chunk text
    chunks = chunk_text(text, file.filename)
    
    # Save chunks to persistent JSON store (fallback)
    save_chunks(chunks)
    
    # Upsert chunks to Chroma vector store
    try:
        upsert_chunks(chunks, file.filename)
    except Exception as e:
        logger.error(f"Failed to upsert chunks to Chroma during ingestion: {e}")
        # Note: We still return chunks even if vector store fails, 
        # so user knows what was processed.
        # Could also raise an HTTPException if strict failure is preferred.
    
    return chunks
