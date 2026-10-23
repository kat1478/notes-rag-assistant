#!/bin/sh
set -e
# Index the bundled notes on first start; later starts keep the existing vector store (and uploaded notes).
python -m app.ingest_corpus --corpus v1 --if-empty
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
