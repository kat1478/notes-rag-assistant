from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
from pathlib import Path
import json
import time
from .schemas import (
    SearchRequest, SearchResponse,
    AskRequest, AskResponse, AskResponseMetadata,
    IngestResponse, FileIngestDetail, ModelInfo,
)
from .ingestion import process_upload
from .retrieval import perform_search
from .answering import generate_answer_with_meta
from .config import (
    TOP_K, OLLAMA_MODEL, DEFAULT_MODEL_ID, DATA_DIR, TEMPERATURE, DEMO_MODE, ASK_RATE_LIMIT, ASK_RATE_WINDOW_S,
)
from .provider import list_models
from .rate_limit import RateLimiter, client_id

app = FastAPI(
    title="Notes RAG Assistant API",
    description="Backend API for Stage 2 of Notes RAG Assistant",
    version="2.0.0"
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static frontend files
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

ask_limiter = RateLimiter(ASK_RATE_LIMIT, ASK_RATE_WINDOW_S)


def require_uploads_enabled():
    if DEMO_MODE:
        raise HTTPException(status_code=403, detail="Uploads are disabled in the public demo. Run the app locally to add notes.")


def enforce_ask_rate_limit(request: Request):
    peer = request.client.host if request.client else None
    if not ask_limiter.allow(client_id(request.headers.get("x-forwarded-for"), peer)):
        raise HTTPException(status_code=429, detail="Too many questions from this address — try again later.")

@app.get("/")
def read_root():
    if (frontend_dir / "index.html").exists():
        return RedirectResponse(url="/frontend/")
    return {
        "project": "Notes RAG Assistant",
        "status": "running",
        "endpoints": ["/ingest", "/search", "/ask", "/health"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Notes RAG Assistant API",
        "version": "2.0.0",
        "provider": DEFAULT_MODEL_ID.split(":", 1)[0],
        "model": DEFAULT_MODEL_ID.split(":", 1)[1],
        "default_model_id": DEFAULT_MODEL_ID,
        "ollama_model": OLLAMA_MODEL,  # kept for older clients / smoke test
        "top_k_default": TOP_K,
        "temperature_default": TEMPERATURE,
        "demo_mode": DEMO_MODE,
    }

@app.get("/models", response_model=List[ModelInfo])
def models():
    """Allowlisted models selectable per request; hosted ones need OPENROUTER_API_KEY."""
    return list_models()

BENCHMARK_SUMMARIES = {
    "v2": DATA_DIR / "eval" / "v2" / "models" / "summary.json",
    "v1": DATA_DIR / "eval" / "models" / "summary.json",
}

@app.get("/benchmark")
def benchmark(corpus: Optional[str] = None):
    """Model comparison results produced by `python -m app.run_model_benchmark` (default: newest corpus with results)."""
    if corpus is not None and corpus not in BENCHMARK_SUMMARIES:
        raise HTTPException(status_code=400, detail=f"Unknown corpus '{corpus}'. Use one of {list(BENCHMARK_SUMMARIES)}")
    for name in [corpus] if corpus else list(BENCHMARK_SUMMARIES):
        path = BENCHMARK_SUMMARIES[name]
        if path.exists():
            return {"corpus": name, **json.loads(path.read_text(encoding="utf-8"))}
    raise HTTPException(status_code=404, detail="No benchmark yet. Run: python -m app.run_model_benchmark --corpus v2")

@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_uploads_enabled)])
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a single .txt or .md file. Use /ingest/batch for multiple files."""
    if not file.filename.endswith((".txt", ".md")):
        return IngestResponse(
            files_ingested=0,
            chunks_created=0,
            file_names=[],
            file_details=[FileIngestDetail(
                file_name=file.filename, chunks_created=0, status="unsupported file type"
            )],
            status="unsupported file type"
        )

    chunks = await process_upload(file)
    return IngestResponse(
        files_ingested=1,
        chunks_created=len(chunks),
        file_names=[file.filename],
        file_details=[FileIngestDetail(
            file_name=file.filename, chunks_created=len(chunks), status="success"
        )],
        status="success"
    )

@app.post("/ingest/batch", response_model=IngestResponse, dependencies=[Depends(require_uploads_enabled)])
async def ingest_documents_batch(files: List[UploadFile] = File(...)):
    """Ingest multiple .txt/.md files at once."""
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

@app.post("/ask", response_model=AskResponse, dependencies=[Depends(enforce_ask_rate_limit)])
def ask(request: AskRequest):
    start_time = time.time()
    top_k = request.top_k if request.top_k is not None else TOP_K
    results = perform_search(request.query, top_k)
    try:
        generation = generate_answer_with_meta(request.query, results, request.model, request.temperature)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    elapsed_ms = (time.time() - start_time) * 1000
    provider, model = generation.model_id.split(":", 1)

    return AskResponse(
        answer=generation.text,
        sources=results,
        metadata=AskResponseMetadata(
            top_k_used=top_k,
            retrieval_mode="hybrid (Chroma semantic + TF-IDF fallback)",
            generation_mode="extractive fallback" if generation.error else f"{provider} ({model})",
            response_time_ms=round(elapsed_ms, 2),
            model=generation.model_id,
            provider=provider,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            generation_error=generation.error,
            temperature=generation.temperature,
        )
    )

