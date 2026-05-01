from fastapi import FastAPI
from .schemas import SearchRequest, SearchResponse, AskRequest, AskResponse, IngestResponse

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
def ingest_documents():
    # Placeholder for Stage 1 Step 3
    return IngestResponse(
        files_ingested=0,
        chunks_created=0,
        file_names=[],
        status="not implemented yet"
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
