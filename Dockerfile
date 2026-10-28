# notes-rag-assistant: FastAPI backend + static frontend in one image.
FROM python:3.12-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

# Unprivileged user created up front so files are written with the right owner — a late `chown -R` would copy
# every file (incl. the ~470 MB model) into a new layer.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /opt/hf /app/backend \
    && chown app:app /opt/hf /app /app/backend
WORKDIR /app/backend

# CPU-only PyTorch first: otherwise sentence-transformers pulls ~2 GB of CUDA libraries a CPU server never uses.
RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

USER app
# Bake the embedding model into the image (~470 MB) so containers start fast and work offline.
ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"
ENV HF_HUB_OFFLINE=1

COPY --chown=app:app backend/app ./app
COPY --chown=app:app backend/data/notes ./data/notes
COPY --chown=app:app backend/data/eval ./data/eval
COPY --chown=app:app backend/smoke_test.py ./
COPY --chown=app:app frontend /app/frontend
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN mkdir -p data/chroma data/processed data/raw

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
ENTRYPOINT ["docker-entrypoint.sh"]

# Lambda variant (docker build --target lambda .): same app behind the Lambda Web Adapter, which turns Function URL
# events into plain HTTP requests, so FastAPI runs unchanged.
FROM app AS lambda
# Index built into the image: Lambda can only write to /tmp, and embedding the notes on every cold start would be slow.
RUN python -m app.ingest_corpus --corpus v1
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter
COPY --chmod=755 lambda-entrypoint.sh /usr/local/bin/lambda-entrypoint.sh
# Lambda runs the image as an arbitrary uid with no passwd entry and a read-only root: torch calls getpass.getuser()
# at import time unless its cache dir is set, and libraries fall back to $HOME for caches.
# Startup (~20 s: torch + sentence-transformers imports) exceeds the 10 s init phase; async init lets the first
# request wait for readiness instead of Lambda failing and re-running init.
ENV CHROMA_DIR=/tmp/chroma \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    USER=lambda \
    HOME=/tmp \
    DEMO_MODE=true \
    ASK_RATE_LIMIT=10 \
    AWS_LWA_PORT=8000 \
    AWS_LWA_READINESS_CHECK_PATH=/health \
    AWS_LWA_ASYNC_INIT=true \
    ANONYMIZED_TELEMETRY=False
ENTRYPOINT ["lambda-entrypoint.sh"]

# Plain image stays the default target: Compose and CI build without --target.
FROM app
