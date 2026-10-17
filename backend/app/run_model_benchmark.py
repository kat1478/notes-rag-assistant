# Run from the backend directory:
#   python -m app.run_model_benchmark                       # every available model
#   python -m app.run_model_benchmark --models ollama:llama3.2:3b openrouter:anthropic/claude-haiku-5.5
#   python -m app.run_model_benchmark --limit 3             # quick smoke run
#
# Retrieval runs once per question; every model then answers from the *same* context,
# so differences come from generation only. Results: data/eval/models/<model>.csv + summary.json.

import argparse
import csv
import json
import logging
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from .answering import FALLBACK_PREFIX, generate_answer_with_meta
from .config import DATA_DIR, OPENROUTER_BASE_URL
from .evaluation import is_refusal_response, load_evaluation_dataset
from .provider import get_provider, list_models
from .retrieval import perform_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = DATA_DIR / "eval" / "models"
CITATION_RE = re.compile(r"\[Sources?\s*((?:\d+\s*(?:,|and|&)?\s*)+)\]", re.IGNORECASE)


def parse_citations(text: str, n_sources: int) -> Dict[str, int]:
    """Counts [Source N] citations that point to a chunk in the context vs. ones that don't."""
    valid = invalid = 0
    for group in CITATION_RE.findall(text):
        for n in map(int, re.findall(r"\d+", group)):
            if 1 <= n <= n_sources:
                valid += 1
            else:
                invalid += 1
    return {"valid": valid, "invalid": invalid}


def fetch_openrouter_prices() -> Dict[str, Dict[str, float]]:
    """USD per token for each OpenRouter model (public endpoint, no key needed)."""
    try:
        data = httpx.get(f"{OPENROUTER_BASE_URL}/models", timeout=30).json()["data"]
    except Exception as e:
        logger.warning(f"Could not fetch OpenRouter prices, cost will be empty: {e}")
        return {}
    return {
        m["id"]: {"prompt": float(m["pricing"]["prompt"]), "completion": float(m["pricing"]["completion"])}
        for m in data
        if m.get("pricing")
    }


def query_cost(model_id: str, prompt_tokens: Optional[int], completion_tokens: Optional[int],
               prices: Dict[str, Dict[str, float]]) -> Optional[float]:
    provider, model = model_id.split(":", 1)
    if provider == "ollama":
        return 0.0  # local inference; hardware/energy cost not counted
    price = prices.get(model)
    if price is None or prompt_tokens is None or completion_tokens is None:
        return None
    return prompt_tokens * price["prompt"] + completion_tokens * price["completion"]


def slug(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9.-]+", "_", model_id)


def warm_up(model_id: str) -> None:
    """Loads a local model into memory so the first timed question doesn't include the cold start."""
    if model_id.startswith("ollama:"):
        logger.info(f"Warming up {model_id}...")
        try:
            get_provider(model_id).generate("Reply with OK.")
        except RuntimeError as e:
            logger.warning(f"Warm-up failed: {e}")


def run_model(model_id: str, items: List[dict], contexts: Dict[str, list], prices) -> List[dict]:
    warm_up(model_id)
    rows = []
    for i, item in enumerate(items, 1):
        sources = contexts[item["id"]]
        t0 = time.perf_counter()
        gen = generate_answer_with_meta(item["question"], sources, model_id)
        latency_ms = (time.perf_counter() - t0) * 1000

        failed = gen.error is not None or gen.text.startswith(FALLBACK_PREFIX)
        refused = is_refusal_response(gen.text)
        cites = parse_citations(gen.text, len(sources))
        should_answer = bool(item.get("should_answer", True))
        rows.append({
            "question_id": item["id"],
            "category": item.get("category", ""),
            "should_answer": should_answer,
            "answered": (not failed) and (not refused) and bool(gen.text.strip()),
            "refused": refused,
            "failed": failed,
            "valid_citations": cites["valid"],
            "invalid_citations": cites["invalid"],
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": gen.prompt_tokens,
            "completion_tokens": gen.completion_tokens,
            "cost_usd": query_cost(model_id, gen.prompt_tokens, gen.completion_tokens, prices),
            "answer_text": gen.text.replace("\n", " ").replace("\r", " "),
            "error": gen.error or "",
        })
        logger.info(f"[{model_id}] {i}/{len(items)} {item['id']}: "
                    f"{'FAIL' if failed else 'refused' if refused else 'answered'}, {latency_ms / 1000:.1f}s")
    return rows


def summarize(model_id: str, rows: List[dict]) -> dict:
    in_scope = [r for r in rows if r["should_answer"]]
    oos = [r for r in rows if not r["should_answer"]]
    answered = [r for r in rows if r["answered"]]
    ok = [r for r in rows if not r["failed"]]
    latencies = [r["latency_ms"] / 1000 for r in ok]
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    provider, model = model_id.split(":", 1)

    def rate(num, den):
        return round(num / den, 4) if den else None

    return {
        "model_id": model_id,
        "provider": provider,
        "model": model,
        "n_questions": len(rows),
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(oos),
        "answer_rate_in_scope": rate(sum(r["answered"] for r in in_scope), len(in_scope)),
        "refusal_rate_out_of_scope": rate(sum(r["refused"] for r in oos), len(oos)),
        "valid_citation_rate": rate(sum(r["valid_citations"] > 0 for r in answered), len(answered)),
        "invalid_citations": sum(r["invalid_citations"] for r in rows),
        "failures": sum(r["failed"] for r in rows),
        "latency_mean_s": round(statistics.mean(latencies), 2) if latencies else None,
        "latency_p50_s": round(statistics.median(latencies), 2) if latencies else None,
        "prompt_tokens_mean": round(statistics.mean([r["prompt_tokens"] for r in rows if r["prompt_tokens"]]), 1)
        if any(r["prompt_tokens"] for r in rows) else None,
        "completion_tokens_mean": round(statistics.mean([r["completion_tokens"] for r in rows if r["completion_tokens"]]), 1)
        if any(r["completion_tokens"] for r in rows) else None,
        "cost_per_1k_queries_usd": round(1000 * sum(costs) / len(costs), 4) if costs and len(costs) == len(rows) else None,
    }


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Compare LLMs on the answer-eval dataset with identical retrieval.")
    parser.add_argument("--models", nargs="+", help="Model ids (default: all available, see GET /models).")
    parser.add_argument("--dataset-path", default=str(DATA_DIR / "eval" / "evaluation_dataset.json"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Only the first N questions (quick check; not saved to summary).")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    available = {m["id"]: m for m in list_models()}
    model_ids = args.models or [mid for mid, m in available.items() if m["available"]]
    unknown = [m for m in model_ids if m not in available]
    if unknown:
        sys.exit(f"Not in the allowlist: {unknown}. Allowed: {sorted(available)}")
    unavailable = [m for m in model_ids if not available[m]["available"]]
    if unavailable:
        sys.exit(f"Unavailable (set OPENROUTER_API_KEY in backend/.env): {unavailable}")

    items = load_evaluation_dataset(Path(args.dataset_path))
    if args.limit:
        items = items[: args.limit]
    logger.info(f"Retrieving context once for {len(items)} questions (top_k={args.top_k})...")
    contexts = {item["id"]: perform_search(item["question"], top_k=args.top_k) for item in items}
    prices = fetch_openrouter_prices() if any(m.startswith("openrouter:") for m in model_ids) else {}

    out_dir = Path(args.out_dir)
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"runs": {}}

    for model_id in model_ids:
        logger.info(f"=== {model_id} ===")
        rows = run_model(model_id, items, contexts, prices)
        result = summarize(model_id, rows)
        print(json.dumps(result, indent=2))
        if args.limit:
            continue  # partial runs are not comparable — keep them out of the summary
        write_csv(out_dir / f"{slug(model_id)}.csv", rows)
        result.update({
            "dataset": Path(args.dataset_path).name,
            "top_k": args.top_k,
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
        })
        summary["runs"][model_id] = result
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logger.info(f"Saved {model_id} → {summary_path}")


if __name__ == "__main__":
    main()
