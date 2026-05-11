from fastapi import FastAPI, UploadFile, File
from typing import List
from .schemas import (
    SearchRequest, SearchResponse,
    AskRequest, AskResponse,
    IngestResponse, FileIngestDetail,
)
from .ingestion import process_upload
from .retrieval import perform_search
from .answering import generate_answer
from .config import TOP_K

app = FastAPI(
    title="Notes RAG Assistant API",
    description="Backend API for Stage 2 of Notes RAG Assistant",
    version="2.0.0"
)

@app.get("/")
def read_root():
    return {
        "project": "Notes RAG Assistant",
        "status": "running",
        "endpoints": ["/ingest", "/search", "/ask"]
    }

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    files: List[UploadFile] = File(..., description="One or more .txt/.md files to ingest")
):
    total_chunks = 0
    file_names = []
    file_details = []

    for file in files:
        if not file.filename.endswith((".txt", ".md")):
            file_details.append(FileIngestDetail(
                file_name=file.filename,
                chunks_created=0,
                status="skipped: unsupported file type"
            ))
            continue

        chunks = await process_upload(file)
        total_chunks += len(chunks)
        file_names.append(file.filename)
        file_details.append(FileIngestDetail(
            file_name=file.filename,
            chunks_created=len(chunks),
            status="success"
        ))

    status = "success" if file_names else "no valid files uploaded"

    return IngestResponse(
        files_ingested=len(file_names),
        chunks_created=total_chunks,
        file_names=file_names,
        file_details=file_details,
        status=status
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

