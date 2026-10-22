# Run from the backend directory:
#   python -m app.run_judge --limit 5                 # dry run: first 5 judgeable answers per model, not saved
#   python -m app.run_judge --max-cost-usd 2          # judge every saved v2 answer (cached: re-runs are free)
#
# Grades the saved v2 benchmark answers (data/eval/v2/models/*.csv) with an LLM judge (config.JUDGE_MODEL):
# key-point correctness + faithfulness to the retrieved context. No answers are regenerated; the context is
# rebuilt with the same deterministic retrieval. Results: data/eval/v2/judge/<model>.csv + summary.json, and
# judge_* metrics merged into data/eval/v2/models/summary.json for the charts.

import argparse
import csv
import hashlib
import json
import logging
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .config import CORPORA, DATA_DIR, JUDGE_MODEL
from .evaluation import load_evaluation_dataset
from .judge import JUDGE_VERSION, judge_answer
from .retrieval import perform_search
from .run_model_benchmark import fetch_openrouter_prices, slug, write_csv
from .vector_store import use_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

V2 = DATA_DIR / "eval" / "v2"
JUDGE_DIR = V2 / "judge"
CACHE = JUDGE_DIR / "cache.jsonl"


def cache_key(model_id: str, qid: str, answer: str) -> str:
    return hashlib.sha256(f"{JUDGE_MODEL}|{JUDGE_VERSION}|{model_id}|{qid}|{answer}".encode("utf-8")).hexdigest()[:24]


def load_cache() -> Dict[str, dict]:
    if not CACHE.exists():
        return {}
    return {e["key"]: e["grade"] for e in map(json.loads, CACHE.read_text(encoding="utf-8").splitlines()) if e}


def append_cache(key: str, grade: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "grade": grade}, ensure_ascii=False) + "\n")


def summarize(rows: List[dict]) -> dict:
    answered = [r for r in rows if r["judged"]]
    supported = [r for r in rows if r["retrieval_hit"] == "True"]
    missed_answered = [r for r in answered if r["retrieval_hit"] == "False"]

    def rate(num, den):
        return round(num / den, 4) if den else None

    return {
        "n_in_scope": len(rows),
        "n_answered": len(answered),
        # Headline: of the questions whose note was retrieved, how many got a fully correct answer
        # (a refusal there counts as not correct).
        "correct_rate_supported": rate(sum(r["verdict"] == "correct" for r in supported), len(supported)),
        "correct_rate_answered": rate(sum(r["verdict"] == "correct" for r in answered), len(answered)),
        "partial_rate_answered": rate(sum(r["verdict"] == "partially_correct" for r in answered), len(answered)),
        "incorrect_rate_answered": rate(sum(r["verdict"] == "incorrect" for r in answered), len(answered)),
        "key_point_recall_mean": round(statistics.mean(float(r["key_point_recall"]) for r in answered), 4) if answered else None,
        "faithful_rate_answered": rate(sum(r["faithful"] is True for r in answered), len(answered)),
        # Answers given although retrieval missed the note: anything correct here came from outside the context.
        "n_answered_without_note": len(missed_answered),
        "faithful_rate_without_note": rate(sum(r["faithful"] is True for r in missed_answered), len(missed_answered)),
    }


def main():
    parser = argparse.ArgumentParser(description="Grade saved v2 benchmark answers with an LLM judge.")
    parser.add_argument("--models", nargs="+", help="Run keys from data/eval/v2/models/summary.json (default: all).")
    parser.add_argument("--limit", type=int, help="Dry run: first N judgeable answers per model; nothing saved.")
    parser.add_argument("--max-cost-usd", type=float, default=2.0)
    parser.add_argument("--results-dir", default=str(V2 / "models"),
                        help="Benchmark results to grade (summary.json + CSVs); grades go to <results-dir>/judge/ "
                             "unless it is the main v2 directory.")
    parser.add_argument("--dataset-path", default=str(V2 / "questions.json"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    main_run = results_dir.resolve() == (V2 / "models").resolve()
    judge_dir = JUDGE_DIR if main_run else results_dir / "judge"
    summary_path = results_dir / "summary.json"
    runs = json.loads(summary_path.read_text(encoding="utf-8"))["runs"]
    keys = args.models or list(runs)
    questions = {q["id"]: q for q in load_evaluation_dataset(Path(args.dataset_path))}
    use_collection(CORPORA["v2"][1])
    price = fetch_openrouter_prices().get(JUDGE_MODEL)
    if not price:
        sys.exit(f"No price for judge {JUDGE_MODEL} — refusing to run without a cost estimate.")

    cache = load_cache()
    contexts: Dict[str, list] = {}
    spent = 0.0
    judge_summary = {"judge_model": JUDGE_MODEL, "judge_version": JUDGE_VERSION, "runs": {}}

    for key in keys:
        rows = []
        judged_now = 0
        for r in csv.DictReader(open(results_dir / f"{slug(key)}.csv", encoding="utf-8")):
            q = questions[r["question_id"]]
            if not q["should_answer"]:
                continue
            row = {"question_id": q["id"], "category": q["category"], "retrieval_hit": r["retrieval_hit"],
                   "judged": False, "verdict": "", "key_point_recall": "", "faithful": "",
                   "unsupported_claims": "", "reason": "", "answer_text": r["answer_text"]}
            if r["refused"] == "True" or r["failed"] == "True":
                row["verdict"] = "refused" if r["refused"] == "True" else "failed"
                rows.append(row)
                continue
            if args.limit and judged_now >= args.limit:
                continue
            ck = cache_key(key, q["id"], r["answer_text"])
            grade = cache.get(ck)
            if grade is None:
                if q["id"] not in contexts:
                    contexts[q["id"]] = perform_search(q["question"], top_k=3)
                grade = judge_answer(q["question"], q["expected_key_points"], contexts[q["id"]], r["answer_text"])
                spent += (grade["prompt_tokens"] or 0) * price["prompt"] + (grade["completion_tokens"] or 0) * price["completion"]
                if not args.limit:
                    append_cache(ck, grade)
                    cache[ck] = grade
                if spent > args.max_cost_usd:
                    sys.exit(f"Budget guard: judge spend ${spent:.4f} > {args.max_cost_usd}; cached grades are kept.")
            judged_now += 1
            row.update(judged=True, verdict=grade["verdict"], key_point_recall=grade["key_point_recall"],
                       faithful=grade["faithful"], unsupported_claims=" | ".join(grade["unsupported_claims"]),
                       reason=grade["reason"])
            rows.append(row)
            logger.info(f"[{key}] {q['id']}: {grade['verdict']}, recall {grade['key_point_recall']}, "
                        f"{'faithful' if grade['faithful'] else 'UNFAITHFUL'}")

        if args.limit:
            for row in [x for x in rows if x["judged"]]:
                print(f"{key} {row['question_id']}: {row['verdict']} | recall {row['key_point_recall']} | "
                      f"unsupported: {row['unsupported_claims'] or '-'} | {row['reason']}")
            continue
        write_csv(judge_dir / f"{slug(key)}.csv", rows)
        judge_summary["runs"][key] = summarize(rows)
        logger.info(f"{key}: {judge_summary['runs'][key]}")

    logger.info(f"Judge spend this run: ${spent:.4f} ({JUDGE_MODEL})")
    if args.limit:
        return

    judge_summary["run_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    judge_dir.mkdir(parents=True, exist_ok=True)
    (judge_dir / "summary.json").write_text(json.dumps(judge_summary, indent=2) + "\n", encoding="utf-8")
    # Merge headline judge metrics into the model summary so the charts and the Models page can show them.
    models_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key, m in judge_summary["runs"].items():
        models_summary["runs"][key].update({
            "judge_model": JUDGE_MODEL,
            "judge_correct_rate_supported": m["correct_rate_supported"],
            "judge_key_point_recall": m["key_point_recall_mean"],
            "judge_faithful_rate": m["faithful_rate_answered"],
        })
    summary_path.write_text(json.dumps(models_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(f"Saved {judge_dir / 'summary.json'} and merged judge metrics into {summary_path}")


if __name__ == "__main__":
    main()
