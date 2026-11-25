"""Per-request telemetry for /ask as CloudWatch Embedded Metric Format (EMF).

One JSON line per request on stdout. On Lambda, CloudWatch Logs extracts the declared fields as metrics — no
client library, no IAM permission, no API call on the request path. Locally it is just a log line.

Privacy: the demo is public, so the question and answer text are never logged; a short hash lets repeated
questions be grouped without storing them.
"""
import hashlib
import json
import sys
import time
from typing import Dict, Optional

NAMESPACE = "NotesRag"

# Metric name -> CloudWatch unit. No dimensions: every dimension value is a separate billed custom metric,
# and these 8 stay within the 10 free ones. Model and provider are logged as plain properties instead.
METRICS = {
    "LatencyMs": "Milliseconds",
    "RetrievalMs": "Milliseconds",
    "GenerationMs": "Milliseconds",
    "PromptTokens": "Count",
    "CompletionTokens": "Count",
    "CostUsd": "None",
    "Refused": "Count",
    "GenerationError": "Count",
}


def question_hash(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()[:12]


def ask_event(question: str, model_id: str, retrieval_ms: float, generation_ms: float, n_sources: int,
              refused: bool, generation_error: bool, prompt_tokens: Optional[int],
              completion_tokens: Optional[int], cost_usd: Optional[float], now_ms: Optional[int] = None,
              fallback_from: Optional[str] = None) -> Dict:
    values = {
        "LatencyMs": round(retrieval_ms + generation_ms, 1),
        "RetrievalMs": round(retrieval_ms, 1),
        "GenerationMs": round(generation_ms, 1),
        "PromptTokens": prompt_tokens,
        "CompletionTokens": completion_tokens,
        "CostUsd": cost_usd,
        "Refused": int(refused),
        "GenerationError": int(generation_error),
    }
    # EMF rejects null metric values, so metrics without a value (e.g. no cost for Ollama) are left out.
    present = {k: v for k, v in values.items() if v is not None}
    return {
        "_aws": {
            "Timestamp": now_ms if now_ms is not None else int(time.time() * 1000),
            "CloudWatchMetrics": [{
                "Namespace": NAMESPACE,
                "Dimensions": [[]],
                "Metrics": [{"Name": k, "Unit": METRICS[k]} for k in present],
            }],
        },
        "event": "ask",
        "model": model_id,
        "question_hash": question_hash(question),
        "question_chars": len(question),
        "sources": n_sources,
        **({"fallback_from": fallback_from} if fallback_from else {}),
        **present,
    }


def emit(event: Dict) -> None:
    """Writes straight to stdout: a logging prefix would make the line invalid JSON and CloudWatch would skip it."""
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()
