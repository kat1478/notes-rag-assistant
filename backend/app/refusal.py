"""Refusal detection for evaluation: does an answer refuse, answer partly, or answer?

The phrase-list regex in evaluation.py misses soft refusals ("Bellman-Ford nie jest opisany w kontekście") and
flags answers that mention a gap as refusals. An LLM classifier fixed both on 60 human labels
(data/eval/v2/refusal_calibration/). Labels are cached, so re-scoring saved answers is free and deterministic.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from .config import DATA_DIR, OPENROUTER_API_KEY, REFUSAL_DETECTOR, REFUSAL_MODEL
from .evaluation import is_refusal_response
from .provider import OpenRouterProvider

logger = logging.getLogger(__name__)

LABELS = ("refusal", "partial", "answer")

# Calibrated wording: changing it invalidates the cache (prompt hash in the key) and the human calibration.
PROMPT = """A study-notes assistant answered a question. Classify what the ANSWER does (not whether it is true):
- "refusal": gives no substantive answer to the question; says the notes/context do not cover it (any wording,
  any language). Restating what the notes say about a neighbouring topic without answering is still a refusal.
- "partial": answers part of the question AND explicitly says another part is not in the notes.
- "answer": answers the question without saying anything is missing, even if the answer is wrong or invented.

QUESTION: {question}

ANSWER: {answer}
"""
SCHEMA = {"type": "json_schema", "json_schema": {"name": "refusal_label", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["label"],
    "properties": {"label": {"type": "string", "enum": list(LABELS)}}}}}
PROMPT_HASH = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()[:12]

# REFUSAL_CACHE lets the CI quality gate use a throwaway cache instead of the committed one.
CACHE = Path(os.getenv("REFUSAL_CACHE", str(DATA_DIR / "eval" / "v2" / "refusal_cache.jsonl")))
_cache: Optional[Dict[str, str]] = None


def normalize(text: str) -> str:
    """Saved CSVs flatten newlines, so live answers are flattened too: the same answer must hit the same key."""
    return text.replace("\n", " ").replace("\r", " ").strip()


def cache_key(question: str, answer: str) -> str:
    raw = f"{REFUSAL_MODEL}|{PROMPT_HASH}|{normalize(question)}|{normalize(answer)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _load() -> Dict[str, str]:
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE.exists():
            for line in CACHE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    _cache[e["key"]] = e["label"]
    return _cache


def _store(key: str, label: str) -> None:
    _load()[key] = label
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "label": label}) + "\n")


def detector() -> str:
    """The detector actually used: "llm" falls back to "regex" without a key, so offline runs still score."""
    if REFUSAL_DETECTOR == "llm" and not OPENROUTER_API_KEY:
        logger.warning("REFUSAL_DETECTOR=llm but OPENROUTER_API_KEY is not set — falling back to the regex detector")
        return "regex"
    return REFUSAL_DETECTOR


def classify(question: str, answer: str, mode: Optional[str] = None) -> Tuple[str, Optional[int], Optional[int]]:
    """Returns (label, prompt_tokens, completion_tokens); tokens are None for cache hits and the regex detector."""
    mode = mode or detector()
    if mode == "regex":
        return ("refusal" if is_refusal_response(answer) else "answer"), None, None
    key = cache_key(question, answer)
    if key in _load():
        return _load()[key], None, None
    gen = OpenRouterProvider(REFUSAL_MODEL).generate(
        PROMPT.format(question=normalize(question), answer=normalize(answer)), 0.0, response_format=SCHEMA)
    try:
        label = json.loads(gen.text)["label"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise RuntimeError(f"Refusal classifier returned unparseable output ({e}): {gen.text[:200]!r}")
    if label not in LABELS:
        raise RuntimeError(f"Refusal classifier returned an invalid label: {label!r}")
    _store(key, label)
    return label, gen.prompt_tokens, gen.completion_tokens
