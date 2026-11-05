# Run from the backend directory:
#   python -m app.compare_refusal_detectors                 # cache only: free, fails on an uncached answer
#   python -m app.compare_refusal_detectors --allow-calls   # classify uncached answers (~$0.00005 each)
#
# Regex vs LLM refusal detector: agreement with the human calibration labels, and what each detector does to the
# refusal metrics of every saved v2 run. Writes data/eval/v2/refusal_detectors.json; saved runs are not modified.

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

from . import refusal
from .config import DATA_DIR, REFUSAL_MODEL
from .evaluation import is_refusal_response

logger = logging.getLogger(__name__)

V2 = DATA_DIR / "eval" / "v2"
CAL = V2 / "refusal_calibration"
OUT = V2 / "refusal_detectors.json"


def llm_label(question: str, answer: str, allow_calls: bool) -> str:
    if not allow_calls and refusal.cache_key(question, answer) not in refusal._load():
        raise SystemExit(f"Uncached answer (pass --allow-calls to classify it): {answer[:80]!r}")
    return refusal.classify(question, answer, "llm")[0]


def binary_scores(pred: List[bool], gold: List[bool], weights: List[float]) -> Dict:
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum(g and not p for p, g in zip(pred, gold))
    wrong = sum(w for w, p, g in zip(weights, pred, gold) if p != g)
    return {"precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "accuracy": round(sum(p == g for p, g in zip(pred, gold)) / len(gold), 3),
            "false_positives": fp, "missed_refusals": fn,
            # the sample is stratified: each item stands for bucket_size / items-in-bucket answers of the pool
            "pool_weighted_error": round(wrong / sum(weights), 4)}


def cohen_kappa(a: List[str], b: List[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def calibration(allow_calls: bool) -> Dict:
    items = json.loads((CAL / "items.json").read_text(encoding="utf-8"))
    key = json.loads((CAL / "key.json").read_text(encoding="utf-8"))
    per_bucket = Counter(k["bucket"] for k in key)
    weights = [k["bucket_size"] / per_bucket[k["bucket"]] for k in key]
    regex = [is_refusal_response(it["answer"]) for it in items]
    llm = [llm_label(it["question"], it["answer"], allow_calls) for it in items]
    out = {}
    for name in ("labels", "labels_revised"):
        gold3 = [lab["label"] for lab in json.loads((CAL / f"{name}.json").read_text(encoding="utf-8"))["labels"]]
        gold = [g == "refusal" for g in gold3]
        out[name] = {
            "n": len(gold3), "human": dict(Counter(gold3)),
            "regex": binary_scores(regex, gold, weights),
            "llm": {**binary_scores([lab == "refusal" for lab in llm], gold, weights),
                    "three_class_agreement": sum(x == y for x, y in zip(llm, gold3)), "kappa": cohen_kappa(llm, gold3)},
        }
    return out


def questions() -> Dict[str, str]:
    return {q["id"]: q["question"] for f in ("questions.json", "questions_pl_probe.json")
            for q in json.loads((V2 / f).read_text(encoding="utf-8"))}


def run_metrics(path: Path, qs: Dict[str, str], allow_calls: bool) -> Dict:
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["failed"] != "True" and r["answer_text"].strip()]
    for r in rows:
        r["regex"] = is_refusal_response(r["answer_text"])
        r["llm"] = llm_label(qs[r["question_id"]], r["answer_text"], allow_calls)

    def rate(sel, hit) -> Dict:
        group = [r for r in rows if sel(r)]
        if not group:
            return {"n": 0, "regex": None, "llm": None}
        return {"n": len(group), "regex": round(sum(hit(r, "regex") for r in group) / len(group), 4),
                "llm": round(sum(hit(r, "llm") for r in group) / len(group), 4)}

    refused = lambda r, d: r["regex"] if d == "regex" else r["llm"] == "refusal"
    oos = lambda r: r["should_answer"] == "False"
    unsupported = lambda r: r["should_answer"] == "True" and r["retrieval_hit"] == "False"
    supported = lambda r: r["should_answer"] == "True" and r["retrieval_hit"] != "False"
    in_scope = lambda r: r["should_answer"] == "True"
    return {
        "refusal_rate_out_of_scope": rate(oos, refused),
        "refusal_rate_unsupported": rate(unsupported, refused),
        "answer_rate_supported": rate(supported, lambda r, d: not refused(r, d)),
        "partial_rate_in_scope": rate(in_scope, lambda r, d: d == "llm" and r["llm"] == "partial"),
        "disagreements": sum(r["regex"] != (r["llm"] == "refusal") for r in rows),
        "n_answers": len(rows),
    }


def saved_runs() -> List[Path]:
    """Every saved answer CSV under v2 except judge grades (those hold verdicts, not answers)."""
    return sorted(p for p in V2.rglob("*.csv") if "judge" not in p.parts and not p.name.endswith(".partial.csv"))


def main():
    parser = argparse.ArgumentParser(description="Regex vs LLM refusal detector on calibration labels and saved runs.")
    parser.add_argument("--allow-calls", action="store_true", help="Classify answers missing from the cache (paid).")
    args = parser.parse_args()
    qs = questions()
    result = {"classifier_model": REFUSAL_MODEL, "prompt_hash": refusal.PROMPT_HASH,
              "calibration": calibration(args.allow_calls),
              "runs": {str(p.relative_to(V2)): run_metrics(p, qs, args.allow_calls) for p in saved_runs()}}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for name, c in result["calibration"].items():
        print(f"{name}: human {c['human']}")
        for d in ("regex", "llm"):
            s = c[d]
            print(f"  {d:<6} precision {s['precision']}  recall {s['recall']}  accuracy {s['accuracy']}  "
                  f"missed {s['missed_refusals']}  pool error {s['pool_weighted_error']:.1%}"
                  + (f"  3-class {s['three_class_agreement']}/{c['n']} kappa {s['kappa']}" if d == "llm" else ""))
    print(f"\n{'run':<70}{'OOS refused':>16}{'unsupported refused':>22}{'partial':>9}")
    for run, m in result["runs"].items():
        fmt = lambda x: "  -" if x is None else f"{x:.0%}"
        o, u = m["refusal_rate_out_of_scope"], m["refusal_rate_unsupported"]
        print(f"{run:<70}{fmt(o['regex']):>7} → {fmt(o['llm']):<6}{fmt(u['regex']):>11} → {fmt(u['llm']):<8}"
              f"{fmt(m['partial_rate_in_scope']['llm']):>7}")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
