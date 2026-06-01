from pydantic import BaseModel
from typing import List, Optional

class ChunkMetadata(BaseModel):
    file_name: str
    chunk_index: int
    chunk_id: str

class ChunkResponse(BaseModel):
    text: str
    metadata: ChunkMetadata
    score: Optional[float] = None

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None

class SearchResponse(BaseModel):
    query: str
    results: List[ChunkResponse]

class AskRequest(BaseModel):
    query: str
    top_k: Optional[int] = None

class AskResponseMetadata(BaseModel):
    top_k_used: int
    retrieval_mode: str
    generation_mode: str
    response_time_ms: float

class AskResponse(BaseModel):
    answer: str
    sources: List[ChunkResponse]
    metadata: Optional[AskResponseMetadata] = None

class FileIngestDetail(BaseModel):
    file_name: str
    chunks_created: int
    status: str

class IngestResponse(BaseModel):
    files_ingested: int
    chunks_created: int
    file_names: List[str]
    file_details: List[FileIngestDetail] = []
    status: str

