#!/bin/sh
set -e
# Chroma opens its SQLite file read-write and Lambda only allows writes in /tmp, so work on a copy of the baked
# index. A re-initialised sandbox may still have it from the previous start.
[ -d "$CHROMA_DIR" ] || cp -r /app/backend/data/chroma "$CHROMA_DIR"
exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
