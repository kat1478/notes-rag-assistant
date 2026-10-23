# Run from the backend directory:  python -m app.compare_prompts
"""Compares grounding prompt versions (v1/v2/v3) on the main v2 questions and the Polish probe set."""
import csv
import json

from .language import detect_language

from .config import DATA_DIR

V2 = DATA_DIR / "eval" / "v2"
E = V2 / "prompt_experiment"
MODELS = ["openrouter:anthropic/claude-haiku-5.5", "openrouter:meta-llama/llama-3.2-3b-instruct",
          "openrouter:qwen/qwen3-30b-a3b-instruct-2507"]
CONDITIONS = [("main", "v1", V2 / "models", V2 / "questions.json"),
              ("main", "v2", E / "main_v2", V2 / "questions.json"),
              ("main", "v3", E / "main_v3", V2 / "questions.json"),
              ("pl-probe", "v1", E / "pl_v1", V2 / "questions_pl_probe.json"),
              ("pl-probe", "v2", E / "pl_v2", V2 / "questions_pl_probe.json"),
              ("pl-probe", "v3", E / "pl_v3", V2 / "questions_pl_probe.json")]


from .run_model_benchmark import slug  # noqa: E402


out = []
for dataset, prompt, d, qpath in CONDITIONS:
    qs = {q["id"]: q for q in json.load(open(qpath, encoding="utf-8"))}
    summ = json.load(open(d / "summary.json"))["runs"]
    for m in MODELS:
        rows = list(csv.DictReader(open(d / f"{slug(m)}.csv", encoding="utf-8")))
        lm = [detect_language(qs[r["question_id"]]["question"]) == detect_language(r["answer_text"])
              for r in rows if r["failed"] == "False"
              and "unknown" not in (detect_language(qs[r["question_id"]]["question"]), detect_language(r["answer_text"]))]
        s = summ[m]
        out.append({"dataset": dataset, "prompt": prompt, "model": m.split("/")[-1],
                    "language_match": round(sum(lm) / len(lm), 3) if lm else None,
                    "correct_supported": s.get("judge_correct_rate_supported"),
                    "faithful": s.get("judge_faithful_rate"),
                    "refuse_missed": s.get("refusal_rate_unsupported"),
                    "refuse_oos": s.get("refusal_rate_out_of_scope"),
                    "answered_supported": s.get("answer_rate_supported")})

fmt = lambda v: "  -  " if v is None else f"{v * 100:4.0f}%"
print(f"{'dataset':<9}{'prompt':<7}{'model':<30}{'lang':>6}{'correct':>8}{'faithf':>7}{'ref-miss':>9}{'ref-oos':>8}{'answered':>9}")
for r in sorted(out, key=lambda r: (r["dataset"], r["model"], r["prompt"])):
    print(f"{r['dataset']:<9}{r['prompt']:<7}{r['model']:<30}{fmt(r['language_match']):>6}{fmt(r['correct_supported']):>8}"
          f"{fmt(r['faithful']):>7}{fmt(r['refuse_missed']):>9}{fmt(r['refuse_oos']):>8}{fmt(r['answered_supported']):>9}")
(E / "comparison.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
