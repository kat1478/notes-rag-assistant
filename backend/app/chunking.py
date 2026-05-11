from typing import List
from .config import CHUNK_SIZE, CHUNK_OVERLAP
from .schemas import ChunkResponse, ChunkMetadata
import uuid
import re

# Sentence-ending pattern: period, exclamation, question mark, or double newline
_SENTENCE_END = re.compile(r'[.!?]\s|\n\n|\n')

def _find_break_point(text: str, target: int, search_range: int = 100) -> int:
    """
    Finds the nearest sentence boundary close to `target` position.
    Searches backwards from target within search_range characters.
    Falls back to a word boundary, then to the raw target position.
    """
    if target >= len(text):
        return len(text)

    # Search backwards for a sentence-ending marker
    search_start = max(0, target - search_range)
    region = text[search_start:target]

    for match in reversed(list(_SENTENCE_END.finditer(region))):
        return search_start + match.end()

    # Fallback: find last whitespace to avoid splitting a word
    last_space = region.rfind(' ')
    if last_space != -1:
        return search_start + last_space + 1

    return target

def chunk_text(text: str, file_name: str) -> List[ChunkResponse]:
    """Splits text into chunks of roughly CHUNK_SIZE, snapping to sentence boundaries."""
    chunks = []
    if not text:
        return chunks

    text_length = len(text)
    start = 0
    chunk_index = 0

    while start < text_length:
        raw_end = start + CHUNK_SIZE
        end = _find_break_point(text, raw_end)

        chunk_content = text[start:end].strip()

        if chunk_content:
            chunk_id = str(uuid.uuid4())
            metadata = ChunkMetadata(
                file_name=file_name,
                chunk_index=chunk_index,
                chunk_id=chunk_id
            )
            chunks.append(ChunkResponse(text=chunk_content, metadata=metadata))
            chunk_index += 1

        # Advance with overlap, also snapped to a boundary
        raw_next = end - CHUNK_OVERLAP
        next_start = _find_break_point(text, raw_next) if raw_next > start else end
        # Safety: always move forward
        start = max(next_start, start + 1)

        if CHUNK_SIZE <= CHUNK_OVERLAP:
            break

    return chunks

