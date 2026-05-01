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

class AskResponse(BaseModel):
    answer: str
    sources: List[ChunkResponse]
    
class IngestResponse(BaseModel):
    files_ingested: int
    chunks_created: int
    file_names: List[str]
    status: str
