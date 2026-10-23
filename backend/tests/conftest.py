"""Shared fixtures. Every test runs offline: no network, no model downloads, no paid API calls."""
import pytest

from app.schemas import ChunkMetadata, ChunkResponse


def chunk(text: str, file_name: str = "note.txt", index: int = 0, score=None) -> ChunkResponse:
    return ChunkResponse(text=text, score=score,
                         metadata=ChunkMetadata(file_name=file_name, chunk_index=index, chunk_id=f"{file_name}-{index}"))


@pytest.fixture
def sources():
    return [chunk("Notatka testowa o algorytmie Dijkstry.", "a.txt", 0),
            chunk("Second note about ReLU.", "b.txt", 1)]
