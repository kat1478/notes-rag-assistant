import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_env_file(path: Path) -> None:
    """Minimal KEY=VALUE loader for backend/.env (avoids a python-dotenv dependency).
    Real environment variables always win over the file."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(BASE_DIR / ".env")

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval settings
TOP_K = 3

# Chroma settings
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
# Evaluation corpora: name -> (notes directory, Chroma collection). v1 is the app default; v2 is the frozen
# benchmark corpus (v1 notes + synthetic near-topic notes). The app searches CHROMA_COLLECTION.
CORPORA = {
    "v1": (DATA_DIR / "notes", "notes_chunks"),
    "v2": (DATA_DIR / "notes_v2", "notes_v2"),
}
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "notes_chunks")

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Generation settings. The default 0.0 (deterministic) keeps eval runs comparable; a request or a benchmark
# run may override it within [TEMPERATURE_MIN, TEMPERATURE_MAX] — 1.0 is the highest value every provider accepts.
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
TEMPERATURE_MIN, TEMPERATURE_MAX = 0.0, 1.0
# Hidden reasoning for models that support it: "off" (default — grounded answering doesn't need it, and it
# dominates cost/latency) or "default" (whatever the provider does by default).
REASONING = os.getenv("REASONING", "off")

# Ollama settings (local, no API key). Off by default since 2026-10-18: on a laptop CPU it takes 5–110 s per
# answer, and the same model hosted is in the benchmark. Kept for offline use and reproducing the Stage 1–4 evals.
OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# OpenRouter settings (optional hosted provider)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Allowlist: only these hosted models can be selected per request (protects a public demo from
# arbitrary, expensive model ids). Comma-separated OpenRouter model ids.
OPENROUTER_MODELS = [
    m.strip()
    for m in os.getenv(
        "OPENROUTER_MODELS",
        "meta-llama/llama-3.2-3b-instruct,"
        "qwen/qwen3-30b-a3b-instruct-2507,"
        "google/gemma-4-31b-it,"
        "anthropic/claude-haiku-5.5,"
        "anthropic/claude-sonnet-5.5,"
        "openai/gpt-5.6-sol",
    ).split(",")
    if m.strip()
]

# LLM judge (internal, not selectable per request): grades saved benchmark answers for correctness and
# faithfulness. Chosen 2026-10-22 as a cheap judge from a different family than the default model (Haiku).
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-5.6-luna")

# Model ids are "<provider>:<model>", e.g. "ollama:llama3.2:3b" or "openrouter:qwen/qwen3-30b-a3b-instruct-2507".
# Default: Claude Haiku 5.5 — the most cost-effective model in the v2 benchmark (see README "Model benchmark").
DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", "openrouter:anthropic/claude-haiku-5.5")
