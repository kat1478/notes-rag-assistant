from fastapi import UploadFile
from typing import List
from .storage import save_raw_file, save_chunks
from .chunking import chunk_text
from .schemas import ChunkResponse

async def process_upload(file: UploadFile) -> List[ChunkResponse]:
    """Reads file, saves it, chunks it, and saves chunks."""
    content = await file.read()
    
    # Save raw file
    save_raw_file(file.filename, content)
    
    # Decode text
    text = content.decode("utf-8", errors="replace")
    
    # Chunk text
    chunks = chunk_text(text, file.filename)
    
    # Save chunks
    save_chunks(chunks)
    
    return chunks
