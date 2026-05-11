from typing import List
from .config import CHUNK_SIZE, CHUNK_OVERLAP
from .schemas import ChunkResponse, ChunkMetadata
import uuid


def _split_paragraphs(text: str) -> List[str]:
    """Splits text on blank lines, discarding empty segments."""
    raw = text.split("\n\n")
    return [p.strip() for p in raw if p.strip()]


def _split_long_paragraph(paragraph: str, max_size: int) -> List[str]:
    """
    Character-level fallback for a single paragraph that exceeds max_size.
    Tries to break on sentence-ending punctuation or whitespace.
    """
    pieces = []
    start = 0
    while start < len(paragraph):
        end = start + max_size
        if end >= len(paragraph):
            pieces.append(paragraph[start:])
            break

        # Try to find a sentence boundary (. ! ?) within the last 120 chars
        search_start = max(start, end - 120)
        best = -1
        for i in range(end - 1, search_start - 1, -1):
            if paragraph[i] in ".!?" and i + 1 < len(paragraph) and paragraph[i + 1] in " \n\t":
                best = i + 1
                break

        # Fallback to last whitespace
        if best == -1:
            space_pos = paragraph.rfind(" ", search_start, end)
            best = space_pos + 1 if space_pos != -1 else end

        pieces.append(paragraph[start:best].strip())
        start = best

    return [p for p in pieces if p]


def chunk_text(text: str, file_name: str) -> List[ChunkResponse]:
    """
    Splits text into chunks using paragraph boundaries.
    Assembles consecutive paragraphs up to CHUNK_SIZE.
    Falls back to character splitting for oversized paragraphs.
    """
    if not text or not text.strip():
        return []

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    # Expand any oversized paragraphs into smaller pieces
    segments: List[str] = []
    for para in paragraphs:
        if len(para) <= CHUNK_SIZE:
            segments.append(para)
        else:
            segments.extend(_split_long_paragraph(para, CHUNK_SIZE))

    # Assemble segments into chunks
    chunks: List[ChunkResponse] = []
    chunk_index = 0
    current_parts: List[str] = []
    current_len = 0

    for seg in segments:
        seg_len = len(seg)
        separator_len = 2 if current_parts else 0  # "\n\n" between paragraphs

        if current_parts and current_len + separator_len + seg_len > CHUNK_SIZE:
            # Flush current chunk
            chunk_content = "\n\n".join(current_parts)
            chunks.append(_make_chunk(chunk_content, file_name, chunk_index))
            chunk_index += 1

            # Overlap: carry last paragraph(s) into next chunk
            overlap_parts: List[str] = []
            overlap_len = 0
            for part in reversed(current_parts):
                if overlap_len + len(part) > CHUNK_OVERLAP:
                    break
                overlap_parts.insert(0, part)
                overlap_len += len(part)

            current_parts = overlap_parts
            current_len = sum(len(p) for p in current_parts) + max(0, (len(current_parts) - 1) * 2)

        current_parts.append(seg)
        current_len += separator_len + seg_len

    # Flush remaining content
    if current_parts:
        chunk_content = "\n\n".join(current_parts)
        chunks.append(_make_chunk(chunk_content, file_name, chunk_index))

    return chunks


def _make_chunk(text: str, file_name: str, chunk_index: int) -> ChunkResponse:
    """Creates a ChunkResponse with generated metadata."""
    return ChunkResponse(
        text=text,
        metadata=ChunkMetadata(
            file_name=file_name,
            chunk_index=chunk_index,
            chunk_id=str(uuid.uuid4())
        )
    )
