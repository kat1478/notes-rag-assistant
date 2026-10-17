# Run from the backend directory:
#   python -m app.run_model_benchmark                       # every available model
#   python -m app.run_model_benchmark --models ollama:llama3.2:3b openrouter:anthropic/claude-haiku-5.5
#   python -m app.run_model_benchmark --limit 3             # quick smoke run
#   python -m app.run_model_benchmark --temperature 0.7     # saved as a separate run "<model>@t0.7"
#   python -m app.run_model_benchmark --corpus v2 --retrieval-only   # free: retrieval hit rates, no LLM calls
#   python -m app.run_model_benchmark --corpus v2 --max-cost-usd 5   # abort if the run would cost more
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
from .config import CORPORA, DATA_DIR, OPENROUTER_BASE_URL, REASONING, TEMPERATURE
from .evaluation import is_refusal_response, load_evaluation_dataset
from .provider import get_provider, list_models
from .retrieval import perform_search
from .vector_store import get_collection, use_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = DATA_DIR / "eval" / "models"  # corpus v1; v2 results go to data/eval/v2/models
DEFAULTS = {
    "v1": (DATA_DIR / "eval" / "evaluation_dataset.json", OUT_DIR),
    "v2": (DATA_DIR / "eval" / "v2" / "questions.json", DATA_DIR / "eval" / "v2" / "models"),
}


class BudgetExceeded(RuntimeError):
    pass
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


def run_key(model_id: str, temperature: float) -> str:
    """Summary key: the plain model id for the default temperature, '<model>@t<T>' otherwise."""
    return model_id if temperature == TEMPERATURE else f"{model_id}@t{temperature:g}"


def run_model(model_id: str, items: List[dict], contexts: Dict[str, list], prices,
              temperature: float, budget: Dict[str, float], hits: Dict[str, Optional[bool]]) -> List[dict]:
    warm_up(model_id)
    rows = []
    for i, item in enumerate(items, 1):
        sources = contexts[item["id"]]
        t0 = time.perf_counter()
        gen = generate_answer_with_meta(item["question"], sources, model_id, temperature)
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
            "reasoning_tokens": gen.reasoning_tokens,
            "reasoning": gen.reasoning or "",
            "retrieval_hit": hits[item["id"]],
            "temperature": gen.temperature,
            "answer_text": gen.text.replace("\n", " ").replace("\r", " "),
            "error": gen.error or "",
        })
        logger.info(f"[{model_id}] {i}/{len(items)} {item['id']}: "
                    f"{'FAIL' if failed else 'refused' if refused else 'answered'}, {latency_ms / 1000:.1f}s")
        budget["spent"] += rows[-1]["cost_usd"] or 0.0
        if budget["spent"] > budget["max"]:
            write_csv(budget["partial_dir"] / f"{slug(model_id)}.partial.csv", rows)
            raise BudgetExceeded(f"spent ${budget['spent']:.4f} > --max-cost-usd {budget['max']} "
                                 f"after {i} questions of {model_id}; partial CSV saved, summary not updated")
    return rows


def summarize(model_id: str, rows: List[dict], temperature_requested: float = TEMPERATURE) -> dict:
    in_scope = [r for r in rows if r["should_answer"]]
    oos = [r for r in rows if not r["should_answer"]]
    supported = [r for r in in_scope if r["retrieval_hit"] is not False]   # context contains the expected note
    unsupported = [r for r in in_scope if r["retrieval_hit"] is False]     # retrieval missed it
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
        # applied = what the model actually received; None when it ignores temperature (reasoning models)
        "temperature_requested": temperature_requested,
        "temperature_applied": next((r["temperature"] for r in rows if r.get("temperature") is not None), None),
        "n_questions": len(rows),
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(oos),
        "answer_rate_in_scope": rate(sum(r["answered"] for r in in_scope), len(in_scope)),
        # Grounding split: if retrieval missed the note, refusing is the correct grounded behaviour.
        "n_supported": len(supported),
        "answer_rate_supported": rate(sum(r["answered"] for r in supported), len(supported)),
        "n_unsupported": len(unsupported),
        "refusal_rate_unsupported": rate(sum(r["refused"] for r in unsupported), len(unsupported)),
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
        "reasoning": next((r["reasoning"] for r in rows if r.get("reasoning")), None),
        "reasoning_tokens_mean": round(statistics.mean([r["reasoning_tokens"] or 0 for r in rows]), 1)
        if any(r.get("reasoning_tokens") is not None for r in rows) else None,
        "retrieval_hit_rate": rate(sum(bool(r["retrieval_hit"]) for r in in_scope if r["retrieval_hit"] is not None),
                                   sum(1 for r in in_scope if r["retrieval_hit"] is not None)),
        # in scope → share answered; out of scope → share refused
        "by_category": {
            cat: {"n": len(group),
                  "rate": rate(sum(r["answered"] if r["should_answer"] else r["refused"] for r in group), len(group)),
                  "metric": "answered" if group[0]["should_answer"] else "refused"}
            for cat, group in _group(rows, "category").items()
        },
    }


def _group(rows: List[dict], key: str) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    return groups


def retrieval_hits(items: List[dict], contexts: Dict[str, list]) -> Dict[str, Optional[bool]]:
    """True if any expected file is among the context chunks; None for questions without an expected file."""
    hits = {}
    for item in items:
        expected = item.get("expected_files") or ([item["expected_file"]] if item.get("expected_file") else [])
        files = {c.metadata.file_name for c in contexts[item["id"]]}
        hits[item["id"]] = bool(files & set(expected)) if expected else None
    return hits


def rescore(out_dir: Path) -> None:
    """Re-derives refused/answered from the saved answer texts and rewrites CSVs + summary metrics."""
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    as_bool = lambda v: {"True": True, "False": False}.get(v)
    as_num = lambda v, f: f(v) if v not in ("", "None", None) else None
    for key, old in summary["runs"].items():
        path = out_dir / f"{slug(key)}.csv"
        rows = []
        for r in csv.DictReader(open(path, encoding="utf-8")):
            r.update(should_answer=as_bool(r["should_answer"]), failed=as_bool(r["failed"]),
                     retrieval_hit=as_bool(r["retrieval_hit"]), valid_citations=int(r["valid_citations"]),
                     invalid_citations=int(r["invalid_citations"]), latency_ms=float(r["latency_ms"]),
                     prompt_tokens=as_num(r["prompt_tokens"], int), completion_tokens=as_num(r["completion_tokens"], int),
                     reasoning_tokens=as_num(r.get("reasoning_tokens"), int), cost_usd=as_num(r["cost_usd"], float),
                     temperature=as_num(r.get("temperature"), float))
            r["refused"] = is_refusal_response(r["answer_text"])
            r["answered"] = (not r["failed"]) and (not r["refused"]) and bool(r["answer_text"].strip())
            rows.append(r)
        write_csv(path, rows)
        new = summarize(old["model_id"], rows, old.get("temperature_requested", TEMPERATURE))
        changed = {k: (old.get(k), v) for k, v in new.items() if old.get(k) != v and k != "by_category"}
        summary["runs"][key] = {**old, **new, "rescored_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        logger.info(f"rescored {key}: {changed or 'no change'}")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
    parser.add_argument("--corpus", choices=sorted(CORPORA), default="v1",
                        help="Notes corpus / Chroma collection (v2: ingest first with python -m app.ingest_corpus --corpus v2).")
    parser.add_argument("--dataset-path", help="Questions JSON (default depends on --corpus).")
    parser.add_argument("--retrieval-only", action="store_true", help="Report retrieval hit rates and exit (no LLM calls).")
    parser.add_argument("--rescore", action="store_true",
                        help="Recompute metrics from saved CSVs (e.g. after a refusal-detector fix); no LLM calls.")
    parser.add_argument("--max-cost-usd", type=float, default=5.0, help="Abort when the run's estimated spend exceeds this.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Only the first N questions (quick check; not saved to summary).")
    parser.add_argument("--out-dir", help="Results directory (default depends on --corpus).")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Sampling temperature 0–1 (default {TEMPERATURE}). Models without temperature support ignore it.")
    args = parser.parse_args()

    available = {m["id"]: m for m in list_models()}
    model_ids = args.models or [mid for mid, m in available.items() if m["available"]]
    unknown = [m for m in model_ids if m not in available]
    if unknown:
        sys.exit(f"Not in the allowlist: {unknown}. Allowed: {sorted(available)}")
    unavailable = [m for m in model_ids if not available[m]["available"]]
    if unavailable:
        sys.exit(f"Unavailable (set OPENROUTER_API_KEY in backend/.env): {unavailable}")

    default_dataset, default_out = DEFAULTS[args.corpus]
    if args.rescore:
        rescore(Path(args.out_dir or default_out))
        return
    dataset_path = Path(args.dataset_path or default_dataset)
    notes_dir, collection = CORPORA[args.corpus]
    use_collection(collection)
    corpus_files = {p.name for p in notes_dir.glob("*.txt")}
    if get_collection().count() == 0:
        sys.exit(f"Collection '{collection}' is empty. Run: python -m app.ingest_corpus --corpus {args.corpus}")

    items = load_evaluation_dataset(dataset_path)
    if args.limit:
        items = items[: args.limit]
    logger.info(f"Retrieving context once for {len(items)} questions (corpus {args.corpus}, top_k={args.top_k})...")
    contexts = {item["id"]: perform_search(item["question"], top_k=args.top_k) for item in items}
    # perform_search silently falls back to TF-IDF over data/processed/chunks.json, which is a different corpus.
    foreign = sorted({c.metadata.file_name for ctx in contexts.values() for c in ctx} - corpus_files)
    if foreign:
        sys.exit(f"Context contains files outside corpus {args.corpus} (TF-IDF fallback?): {foreign}")
    hits = retrieval_hits(items, contexts)

    if args.retrieval_only:
        in_scope = [i for i in items if hits[i["id"]] is not None]
        print(f"Retrieval hit rate (any expected file in context), corpus {args.corpus}, top_k={args.top_k}: "
              f"{sum(hits[i['id']] for i in in_scope)}/{len(in_scope)}")
        for cat, group in sorted(_group([{**i, "hit": hits[i["id"]]} for i in in_scope], "category").items()):
            print(f"  {cat:<24} {sum(g['hit'] for g in group)}/{len(group)}")
        for i in in_scope:
            if not hits[i["id"]]:
                got = sorted({c.metadata.file_name for c in contexts[i["id"]]})
                print(f"  MISS {i['id']}: expected {i.get('expected_files') or i['expected_file']} got {got}")
        return

    prices = fetch_openrouter_prices() if any(m.startswith("openrouter:") for m in model_ids) else {}
    out_dir = Path(args.out_dir or default_out)
    budget = {"spent": 0.0, "max": args.max_cost_usd, "partial_dir": out_dir}
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"runs": {}}

    for model_id in model_ids:
        logger.info(f"=== {model_id} ===")
        try:
            rows = run_model(model_id, items, contexts, prices, args.temperature, budget, hits)
        except BudgetExceeded as e:
            sys.exit(f"Budget guard: {e}")
        result = summarize(model_id, rows, args.temperature)
        key = run_key(model_id, args.temperature)
        print(json.dumps(result, indent=2))
        if args.limit:
            continue  # partial runs are not comparable — keep them out of the summary
        if result["failures"] > len(rows) / 2:
            logger.error(f"{key}: {result['failures']}/{len(rows)} generations failed — not saved to the summary")
            continue
        write_csv(out_dir / f"{slug(key)}.csv", rows)
        result.update({
            "dataset": dataset_path.name,
            "corpus": args.corpus,
            "reasoning_mode": REASONING,
            "top_k": args.top_k,
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
        })
        summary["runs"][key] = result
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logger.info(f"Saved {key} → {summary_path} (run spend so far ${budget['spent']:.4f})")


if __name__ == "__main__":
    main()
