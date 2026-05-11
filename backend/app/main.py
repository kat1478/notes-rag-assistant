from fastapi import FastAPI, UploadFile, File
from typing import List
from .schemas import SearchRequest, SearchResponse, AskRequest, AskResponse, IngestResponse
from .ingestion import process_upload
from .retrieval import perform_search
from .answering import generate_answer
from .config import TOP_K

app = FastAPI(
    title="Notes RAG Assistant API",
    description="Backend API for Stage 2 of Notes RAG Assistant",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {
        "project": "Notes RAG Assistant",
        "status": "running",
        "endpoints": ["/ingest", "/search", "/ask"]
    }

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".txt", ".md")):
        return IngestResponse(
            files_ingested=0,
            chunks_created=0,
            file_names=[],
            status="unsupported file type"
        )
        
    chunks = await process_upload(file)
    
    return IngestResponse(
        files_ingested=1,
        chunks_created=len(chunks),
        file_names=[file.filename],
        status="success"
    )

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    top_k = request.top_k if request.top_k is not None else TOP_K
    results = perform_search(request.query, top_k)
    return SearchResponse(
        query=request.query,
        results=results
    )

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    results = perform_search(request.query, TOP_K)
    answer = generate_answer(request.query, results)
    return AskResponse(
        answer=answer,
        sources=results
    )
