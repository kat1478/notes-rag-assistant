# Stage 2 Instructions — Notes RAG Assistant

This document defines the implementation scope, constraints, and expected deliverables for **Stage 2** of the `notes-rag-assistant` project.

## Important continuity rule

All core rules, engineering assumptions, and quality expectations from **Stage 1** still apply unless explicitly overridden in this document.

That means in particular:

- keep the modular architecture,
- preserve the API-first approach,
- keep the project inside a single repository,
- keep endpoints testable through FastAPI / Swagger,
- keep source grounding visible in outputs,
- keep commits small and meaningful,
- avoid monolithic files and avoid mixing unrelated concerns,
- prefer clear folder structure and predictable naming,
- preserve the ability to run the backend locally in the existing environment.

## Additional coding language rule

Do **not** use Polish in code comments, docstrings, or developer-facing inline explanations.

Use:

- **English comments only**
- comments in a **golden middle** style:
  - present where they improve readability,
  - absent where code is already self-explanatory,
  - no over-commenting,
  - no trivial comments like `# increment i`

Comments should explain:

- intent,
- non-obvious logic,
- architecture decisions,
- assumptions,
- edge cases when relevant.

## Stage 2 objective

Stage 1 delivered a functional lexical RAG backend based on chunking + TF-IDF retrieval.

Stage 2 upgrades the system into a **semantic, fully local, free-to-run RAG stack**, using:

- **local embeddings**
- **local vector database**
- **local LLM**
- preserved source grounding

The selected stack for Stage 2 is:

- **Embeddings:** `sentence-transformers`
- **Vector store:** `Chroma`
- **LLM runtime:** `Ollama`
- **LLM model:** start with a lightweight free local model such as `llama3.2:3b` or `mistral`

The goal is to keep the project:

- free,
- local-first,
- educational,
- modular,
- ready for future improvement.

---

# 1. Stage 2 target architecture

## Existing Stage 1 modules

The Stage 1 backend already contains a pipeline roughly equivalent to:

- ingestion
- chunking
- retrieval
- answering
- FastAPI endpoints

## Stage 2 architecture update

Stage 2 should extend that pipeline into:

1. **Ingestion**
   - load `.txt` and `.md` notes
   - keep current ingestion behavior unless refactoring is needed

2. **Chunking**
   - keep current chunking logic and overlap
   - allow configuration if practical
   - preserve metadata per chunk

3. **Embedding generation**
   - generate dense semantic embeddings for each chunk using `sentence-transformers`
   - generate embeddings for user queries with the same model

4. **Vector storage**
   - persist chunks + metadata + embeddings in Chroma
   - keep a stable collection name
   - support re-ingestion without corrupting the store

5. **Retrieval**
   - perform semantic top-k search through Chroma
   - return the most relevant chunks with metadata and scores if available

6. **Answer generation**
   - build a grounded prompt from retrieved chunks
   - pass the prompt to a local LLM through Ollama
   - return:
     - final answer,
     - retrieved sources,
     - enough provenance for user trust

7. **API**
   - expose a clean FastAPI interface for ingestion, search, and ask flows

---

# 2. Core Stage 2 requirements

## 2.1 Semantic retrieval instead of TF-IDF

Replace or supplement lexical retrieval with embedding-based retrieval.

Expected result:

- queries can match chunks by **meaning**, not only by exact wording,
- retrieval becomes robust to paraphrases,
- the project becomes a true semantic RAG system.

## 2.2 Local-first and free-first

The implementation must remain usable without paid APIs.

That means:

- no OpenAI dependency is required for the default flow,
- the default path should work fully offline except for initial model downloads,
- all main features should be runnable on a local machine through WSL.

## 2.3 Source-grounded answers remain mandatory

Stage 1 already established source grounding as an important rule.

Stage 2 must preserve that.

Every answer returned by `/ask` should include:

- the generated answer,
- the retrieved chunks or source references,
- file names and chunk identifiers where practical,
- enough context for the user to verify the answer.

## 2.4 Backward engineering discipline remains mandatory

Keep the same engineering discipline as in Stage 1:

- modular code,
- meaningful file separation,
- no oversized files without justification,
- reusable services where possible,
- readable interfaces between layers.

---

# 3. Proposed dependency additions

Add the following dependencies to the backend environment as needed:

- `sentence-transformers`
- `chromadb`
- `ollama` or a simple HTTP client for the Ollama local API
- optionally `requests` if needed for Ollama API calls

Do not remove Stage 1 dependencies unless they are no longer needed and the removal is intentional.

If dependency updates are required, update the environment configuration cleanly.

---

# 4. Expected backend module evolution

This is a recommended structure, not a rigid requirement, but separation of concerns should remain strong.

Example backend structure:

```text
backend/
  app/
    main.py
    ingestion.py
    chunking.py
    retrieval.py
    answering.py
    embeddings.py
    vector_store.py
    llm_client.py
    config.py
```

## Recommended responsibilities

### `embeddings.py`
Responsible for:

- loading the sentence-transformer model,
- embedding chunks,
- embedding queries,
- hiding model-specific details from the rest of the app.

### `vector_store.py`
Responsible for:

- Chroma client initialization,
- collection creation / retrieval,
- upserting chunk embeddings and metadata,
- top-k semantic search.

### `llm_client.py`
Responsible for:

- communication with Ollama,
- model selection,
- prompt submission,
- response parsing,
- handling model/runtime errors.

### `retrieval.py`
Responsible for:

- orchestration of semantic retrieval,
- optional compatibility layer if lexical retrieval is kept temporarily.

### `answering.py`
Responsible for:

- prompt construction,
- source formatting,
- calling the LLM client,
- shaping the final response payload.

### `config.py`
Responsible for:

- local configuration defaults,
- model names,
- collection names,
- chunk defaults,
- top-k defaults,
- paths if needed.

---

# 5. Functional deliverables for Stage 2

Stage 2 should result in the following working capabilities.

## Deliverable A — Embedding pipeline

After ingestion:

- notes are chunked,
- each chunk receives a semantic embedding,
- embeddings are stored in Chroma with metadata.

## Deliverable B — Semantic search endpoint

The backend should expose a search endpoint that:

- accepts a natural-language query,
- runs semantic retrieval via embeddings + Chroma,
- returns top-k relevant chunks,
- includes metadata and relevance information when available.

## Deliverable C — Grounded `/ask` endpoint with local LLM

The backend should expose an ask endpoint that:

1. embeds the user question,
2. retrieves top-k chunks from Chroma,
3. constructs a prompt with explicit grounding instructions,
4. calls the local Ollama model,
5. returns:
   - generated answer,
   - sources,
   - raw retrieved context or a structured source block.

## Deliverable D — Local run instructions remain valid

The backend must still be runnable locally through the existing environment and local server startup flow.

Swagger testing should still work.

---

# 6. Prompting requirements for the local LLM

The answer-generation prompt should be grounded and conservative.

The LLM should be instructed to:

- answer only from the provided context,
- avoid inventing facts,
- state uncertainty when the context is insufficient,
- cite or reference the provided sources,
- keep the answer useful and readable.

A suitable prompt style is:

- concise system instruction,
- explicit context block,
- explicit user question,
- explicit rule to admit insufficient evidence.

The model must not be encouraged to answer from general world knowledge when the retrieved context does not support it.

---

# 7. Data and metadata requirements

Chunk metadata should remain useful.

At minimum, each chunk should preserve:

- source file name,
- chunk id,
- text content,
- optionally ingestion timestamp or internal id.

If Chroma metadata requires a specific format, adapt metadata cleanly and consistently.

Avoid metadata design that makes source display harder later.

---

# 8. Persistence expectations

Prefer a persistent local Chroma store rather than a purely in-memory store.

Why:

- faster repeated runs,
- easier debugging,
- closer to a realistic engineering workflow.

If persistent storage is added, document:

- where it lives,
- how to clear/reset it,
- how re-ingestion behaves.

---

# 9. Error handling expectations

Stage 2 introduces more moving parts than Stage 1, so robust error handling matters.

Handle at least these cases cleanly:

- empty collection / no notes ingested,
- embedding model load failure,
- Chroma initialization failure,
- Ollama not running,
- selected Ollama model not installed,
- retrieval returns no relevant chunks,
- malformed uploaded file or unsupported file type.

Errors should be informative and actionable.

---

# 10. Testing expectations

Full automated test coverage is not required immediately, but implementation should remain testable.

At minimum, manual verification should be possible for:

1. ingest a note,
2. confirm chunk persistence,
3. run semantic search,
4. ask a grounded question,
5. inspect returned sources.

If lightweight tests are added, prefer focused tests over broad unstable ones.

---

# 11. Suggested implementation order

Use this implementation sequence unless there is a better technical reason to change it.

## Step 1 — Add dependencies and configuration
- update environment / requirements
- introduce configuration module if needed

## Step 2 — Add embedding module
- load sentence-transformer model
- support chunk and query embeddings

## Step 3 — Add Chroma integration
- initialize persistent Chroma store
- create or load collection
- upsert chunk documents + metadata

## Step 4 — Refactor ingestion flow
- after chunking, compute embeddings
- store chunks in Chroma
- preserve metadata

## Step 5 — Implement semantic retrieval
- retrieve top-k relevant chunks for a query
- return structured results

## Step 6 — Add Ollama LLM client
- connect to local Ollama runtime
- support a configurable model name
- parse generation response safely

## Step 7 — Upgrade `/ask`
- retrieve semantic context
- build grounded prompt
- call Ollama
- return answer + sources

## Step 8 — Manual validation
- test via Swagger UI
- verify semantic matching quality
- verify failure modes

---

# 12. Non-goals for Stage 2

The following are explicitly **not required** unless implementation remains simple:

- frontend UI
- authentication
- multi-user support
- advanced reranking
- hybrid retrieval
- PDF parsing
- evaluation framework
- conversation memory
- production deployment

These can be future stages.

---

# 13. Definition of done for Stage 2

Stage 2 is complete when all of the following are true:

- notes can be ingested locally,
- chunks are embedded locally,
- embeddings are stored in Chroma,
- semantic retrieval works,
- `/ask` uses retrieved context plus a local Ollama LLM,
- answers remain source-grounded,
- the backend runs locally through FastAPI,
- code remains modular and readable,
- comments are in English only and used with moderation.

---

# 14. Style and implementation notes for the coding agent

Use these implementation preferences:

- prefer small, composable functions,
- avoid premature abstraction,
- keep interfaces explicit,
- preserve readability over cleverness,
- keep comments useful but not excessive,
- keep logs practical,
- do not silently swallow exceptions,
- document any necessary manual setup steps.

When making code changes:

- keep commits logically grouped if working incrementally,
- do not rewrite stable Stage 1 code without a clear reason,
- preserve backwards clarity in filenames and structure.

---

# 15. Manual setup assumptions

Assume the project is run in WSL with the existing Mamba-based environment.

Assume the user may need to install Ollama and pull a model separately.

If Stage 2 code depends on Ollama, clearly document the expected setup, for example:

- install Ollama
- start Ollama
- pull a small model such as `llama3.2:3b`

Do not hardcode assumptions that only work on one machine layout.

---

# 16. Final instruction to the coding agent

Implement Stage 2 as a clean continuation of Stage 1.

Do not discard the existing architectural discipline.
Do not use Polish in comments or docstrings.
Keep the system local-first, free-first, and educational.
Preserve source grounding as a central feature, not an optional extra.
