# Run from the backend directory (needs OPENROUTER_API_KEY; ~$0.07 per run):
#   python -m app.quality_gate                 # compare with data/eval/v2/quality_baseline.json
#   python -m app.quality_gate --write-baseline
#
# End-to-end quality gate used in CI on pull requests to master: the default model answers the v2 questions
# and the Polish probe with the app's current prompt and embeddings, the LLM judge grades the answers, and each
# metric must stay within a tolerance of the committed baseline. LLM runs are not bit-deterministic, so the
# tolerances are sized to a few questions, not to zero.

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import DATA_DIR, DEFAULT_MODEL_ID, EMBEDDING_MODEL, JUDGE_MODEL, PROMPT_VERSION
from .run_model_benchmark import slug

V2 = DATA_DIR / "eval" / "v2"
BASELINE = V2 / "quality_baseline.json"
SETS = {"main": V2 / "questions.json", "pl_probe": V2 / "questions_pl_probe.json"}
# Allowed drop below the baseline, as a share (0.05 = 5 pp). Sized to the set: 0.05 of 77 questions ≈ 4,
# 0.10 of 20 ≈ 2, 0.12 of 17 out-of-scope ≈ 2.
TOLERANCE = {"main_correct_share": 0.05, "pl_probe_correct_share": 0.10,
             "main_refusal_out_of_scope": 0.12, "main_language_match": 0.05, "pl_probe_language_match": 0.05}


def run(cmd: list, work: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    # Fresh judge cache: grades must come from this run, and the committed cache must stay untouched.
    subprocess.run([sys.executable, "-m", *cmd], check=True, env={**os.environ, "JUDGE_CACHE": str(work / "judge_cache.jsonl")})


def measure(work: Path, max_cost: float) -> dict:
    metrics = {}
    for name, questions in SETS.items():
        out = work / name
        run(["app.run_model_benchmark", "--corpus", "v2", "--models", DEFAULT_MODEL_ID, "--prompt", PROMPT_VERSION,
             "--dataset-path", str(questions), "--out-dir", str(out), "--max-cost-usd", str(max_cost)], work)
        run(["app.run_judge", "--results-dir", str(out), "--dataset-path", str(questions), "--max-cost-usd", str(max_cost)], work)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))["runs"][DEFAULT_MODEL_ID]
        graded = list(csv.DictReader(open(out / "judge" / f"{slug(DEFAULT_MODEL_ID)}.csv", encoding="utf-8")))
        # Share of ALL in-scope questions answered correctly (a refusal or a retrieval miss counts as not correct).
        metrics[f"{name}_correct_share"] = round(sum(r["verdict"] == "correct" for r in graded) / len(graded), 4)
        metrics[f"{name}_language_match"] = summary.get("language_match_rate")
        if summary.get("refusal_rate_out_of_scope") is not None and name == "main":
            metrics["main_refusal_out_of_scope"] = summary["refusal_rate_out_of_scope"]
    return metrics


def report(metrics: dict, baseline: dict) -> bool:
    rows, ok = [], True
    for key, tol in TOLERANCE.items():
        base, now = baseline["metrics"].get(key), metrics.get(key)
        passed = base is None or (now is not None and now >= base - tol)
        ok &= passed
        fmt = lambda v: "n/a" if v is None else f"{v:.0%}"
        rows.append(f"| {key} | {fmt(base)} | {fmt(now)} | ≥ {fmt(None if base is None else base - tol)} | {'✅' if passed else '❌'} |")
    text = "\n".join([
        f"### Quality gate {'passed ✅' if ok else 'FAILED ❌'}",
        f"Model `{DEFAULT_MODEL_ID}` · prompt `{PROMPT_VERSION}` · embeddings `{EMBEDDING_MODEL}` · judge `{JUDGE_MODEL}`",
        "",
        "| metric | baseline | this run | required | |",
        "| :--- | ---: | ---: | ---: | :---: |",
        *rows,
    ])
    print(text)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text + "\n")
    return ok


def main():
    parser = argparse.ArgumentParser(description="End-to-end LLM quality gate against a committed baseline.")
    parser.add_argument("--write-baseline", action="store_true", help="Measure and save as the new baseline.")
    parser.add_argument("--max-cost-usd", type=float, default=0.5, help="Budget guard per benchmark/judge step.")
    args = parser.parse_args()
    if not os.getenv("OPENROUTER_API_KEY"):
        from .config import OPENROUTER_API_KEY  # also read from backend/.env
        if not OPENROUTER_API_KEY:
            sys.exit("OPENROUTER_API_KEY is not set (in CI: add the OPENROUTER_API_KEY_CI repository secret).")

    with tempfile.TemporaryDirectory() as tmp:
        metrics = measure(Path(tmp), args.max_cost_usd)
    if args.write_baseline:
        BASELINE.write_text(json.dumps({"model": DEFAULT_MODEL_ID, "prompt": PROMPT_VERSION, "embedding_model": EMBEDDING_MODEL,
                                        "judge": JUDGE_MODEL, "metrics": metrics}, indent=2) + "\n", encoding="utf-8")
        print(f"Baseline written to {BASELINE}: {metrics}")
        return
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if not report(metrics, baseline):
        sys.exit(1)


if __name__ == "__main__":
    main()
