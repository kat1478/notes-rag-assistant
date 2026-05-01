from fastapi import FastAPI, UploadFile, File
from typing import List
from .schemas import SearchRequest, SearchResponse, AskRequest, AskResponse, IngestResponse
from .ingestion import process_upload

app = FastAPI(
    title="Notes RAG Assistant API",
    description="Backend API for Stage 1 of Notes RAG Assistant",
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
async def ingest_documents(files: List[UploadFile] = File(...)):
    total_chunks = 0
    file_names = []
    
    for file in files:
        if not file.filename.endswith((".txt", ".md")):
            continue
            
        chunks = await process_upload(file)
        total_chunks += len(chunks)
        file_names.append(file.filename)
        
    status = "success" if file_names else "no valid files uploaded"
    
    return IngestResponse(
        files_ingested=len(file_names),
        chunks_created=total_chunks,
        file_names=file_names,
        status=status
    )

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    # Placeholder for Stage 1 Step 4
    return SearchResponse(
        query=request.query,
        results=[]
    )

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    # Placeholder for Stage 1 Step 4
    return AskResponse(
        answer="I don't know yet.",
        sources=[]
    )
