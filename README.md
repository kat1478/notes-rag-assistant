# Notes RAG Assistant

A local-first, grounded notes assistant that answers questions using ONLY your local indexed notes. There is no external knowledge fallback, ensuring that answers are verified and directly grounded in the sources.

## Prerequisites

1. **Python Environment**:
   - Create or activate the Conda environment using the configuration in `backend/environment.yml`:
     ```bash
     conda activate notesrag
     ```
2. **Ollama**:
   - Ensure Ollama is running locally and has the `llama3.2:3b` model installed:
     ```bash
     ollama run llama3.2:3b
     ```

## How to Run

### 1. Start the Backend Server
Navigate to the `backend` folder and run the FastAPI server using `uvicorn`:
```bash
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 2. Access the Frontend App
Once the backend server starts up:
- Open your browser and navigate to: **[http://localhost:8000/](http://localhost:8000/)**
- The server will automatically redirect the root path to the static frontend at `/frontend/index.html`.
- Alternatively, you can open the local `frontend/index.html` file directly in your browser. The app detects this and communicates with the backend API running at `http://127.0.0.1:8000/ask`.

---

## Example Questions & Verification

### In-Scope Questions
- **Query**: `"What is ReLU?"` or `"Jak działa algorytm Dijkstry?"`
- **Behavior**: The assistant retrieves matching notes, generates a concise grounded response with numbered citations (e.g., `[Source 1]`), and displays retrieved file sections as reference cards. Click on the citations in the text to scroll directly to the source chunk.

### Out-of-Scope Questions
- **Query**: `"What is the capital of France?"`
- **Behavior**: The assistant detects that the required facts are missing from the notes and gracefully refuses to answer (e.g., `"Nie ma informacji w notatkach..."` or `"I don't have enough information..."`), preventing hallucinated general knowledge responses.

---

## Known Limitations (from Stage 3 Evaluation)

- **Local LLM Latency**: Generating answers using local models (Llama 3.2) on CPU-only hardware can take from 5 to 30 seconds. Cold-starts (first query) may take longer while loading embedding weights.
- **Cross-Lingual Quality**: Retrieval quality is weaker when Polish notes are queried in English.
- **Context Fragmentation**: Chunks are expanded by immediately adjacent neighbors (chunk-1, chunk+1), but wide-ranging topics might require deeper context traversal.

