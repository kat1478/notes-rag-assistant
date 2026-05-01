import json
import os
from typing import List
from pathlib import Path
from .schemas import ChunkResponse
from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR

CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks.json"

def save_raw_file(file_name: str, content: bytes) -> str:
    """Saves raw uploaded file to disk and returns the path."""
    file_path = RAW_DATA_DIR / file_name
    with open(file_path, "wb") as f:
        f.write(content)
    return str(file_path)

def save_chunks(new_chunks: List[ChunkResponse]):
    """Appends new chunks to the existing chunks file."""
    existing_chunks = load_chunks()
    existing_chunks.extend(new_chunks)
    
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        # Convert models to dicts
        json.dump([chunk.model_dump() for chunk in existing_chunks], f, ensure_ascii=False, indent=2)

def load_chunks() -> List[ChunkResponse]:
    """Loads all chunks from the processed file."""
    if not CHUNKS_FILE.exists():
        return []
    
    try:
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [ChunkResponse(**item) for item in data]
    except Exception:
        return []
