# Run from the backend directory:
#   python -m app.judge_calibration build [--size 25]          # sample judged answers → label.html (blind)
#   python -m app.judge_calibration score <labels.json> [--key key.json]  # human vs judge agreement → results.json
#   python -m app.judge_calibration rejudge                  # re-grade the sample with the current judge → key_<version>.json
#
# Measures how far to trust the LLM judge: a person labels a stratified sample of judged answers without
# seeing the model or the judge's grade; we then compare verdicts and faithfulness.

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from .config import CORPORA, DATA_DIR
from .evaluation import load_evaluation_dataset

V2 = DATA_DIR / "eval" / "v2"
CAL_DIR = V2 / "judge" / "calibration"
VERDICTS = ["correct", "partially_correct", "incorrect"]
SEED = 42


def judged_rows():
    summary = json.loads((V2 / "judge" / "summary.json").read_text(encoding="utf-8"))
    from .run_model_benchmark import slug
    for key in summary["runs"]:
        for r in csv.DictReader(open(V2 / "judge" / f"{slug(key)}.csv", encoding="utf-8")):
            if r["judged"] == "True":
                yield key, r


def sample(n: int):
    """Stratified: round-robin over models (equal share), each model's queue starting with up to 2 answers the
    judge found non-correct or unfaithful (they are rare), then the rest — so both kinds get checked."""
    rng = random.Random(SEED)
    by_model = {}
    for key, r in judged_rows():
        by_model.setdefault(key, []).append((key, r))
    queues = []
    for key in sorted(by_model):
        rows = by_model[key]
        rng.shuffle(rows)
        flagged = [x for x in rows if x[1]["verdict"] != "correct" or x[1]["faithful"] == "False"]
        rest = [x for x in rows if x not in flagged[:2]]
        queues.append(flagged[:2] + rest)
    picked = []
    while len(picked) < n and any(queues):
        for q in queues:
            if q and len(picked) < n:
                picked.append(q.pop(0))
    rng.shuffle(picked)
    return picked


def build(n: int) -> None:
    from .retrieval import perform_search
    from .vector_store import use_collection
    use_collection(CORPORA["v2"][1])
    questions = {q["id"]: q for q in load_evaluation_dataset(V2 / "questions.json")}
    items, key_file = [], []
    for i, (key, r) in enumerate(sample(n), 1):
        q = questions[r["question_id"]]
        ctx = perform_search(q["question"], top_k=3)
        items.append({
            "item": i,
            "question": q["question"],
            "key_points": q["expected_key_points"],
            "context": [{"n": j, "file": c.metadata.file_name, "text": c.text} for j, c in enumerate(ctx, 1)],
            "answer": r["answer_text"],
        })
        key_file.append({"item": i, "model": key, "question_id": q["id"], "judge_verdict": r["verdict"],
                         "judge_faithful": r["faithful"] == "True"})
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    (CAL_DIR / "items.json").write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # The key (model + judge grade) stays out of the page so labelling is blind.
    (CAL_DIR / "key.json").write_text(json.dumps(key_file, indent=2) + "\n", encoding="utf-8")
    page = (Path(__file__).parent / "calibration_template.html").read_text(encoding="utf-8")
    page = page.replace("/*ITEMS*/[]", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
    (CAL_DIR / "label.html").write_text(page, encoding="utf-8")
    print(f"{len(items)} items → {CAL_DIR / 'label.html'}")
    print(f"verdict mix in sample (judge): {Counter(k['judge_verdict'] for k in key_file)}; "
          f"unfaithful: {sum(not k['judge_faithful'] for k in key_file)}")


def cohen_kappa(a, b, labels) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def rejudge() -> None:
    """Re-grades the calibration items with the current judge (prompt iteration without new human labels)."""
    from .judge import JUDGE_VERSION, judge_answer
    from .retrieval import perform_search
    from .vector_store import use_collection
    use_collection(CORPORA["v2"][1])
    questions = {q["id"]: q for q in load_evaluation_dataset(V2 / "questions.json")}
    items = {i["item"]: i for i in json.loads((CAL_DIR / "items.json").read_text(encoding="utf-8"))}
    out, verified = [], []
    for k in json.loads((CAL_DIR / "key.json").read_text(encoding="utf-8")):
        q = questions[k["question_id"]]
        g = judge_answer(q["question"], q["expected_key_points"], perform_search(q["question"], top_k=3),
                         items[k["item"]]["answer"])
        out.append({**k, "judge_verdict": g["verdict"], "judge_faithful": g["faithful"],
                    "unsupported_claims": g["unsupported_claims"]})
        if g.get("evidence_verified_rate") is not None:
            verified.append(g["evidence_verified_rate"])
    path = CAL_DIR / f"key_judge_{JUDGE_VERSION}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"→ {path}; evidence quotes found verbatim in context: {sum(verified) / len(verified):.0%} (mean per item)")


def score(labels_path: Path, key_path: Path = None) -> None:
    key = {k["item"]: k for k in json.loads((key_path or CAL_DIR / "key.json").read_text(encoding="utf-8"))}
    labels = {l["item"]: l for l in json.loads(Path(labels_path).read_text(encoding="utf-8"))["labels"]}
    both = [i for i in key if i in labels and labels[i].get("verdict") and labels[i].get("faithful") is not None]
    hv = [labels[i]["verdict"] for i in both]
    jv = [key[i]["judge_verdict"] for i in both]
    hf = [labels[i]["faithful"] for i in both]
    jf = [key[i]["judge_faithful"] for i in both]
    # Lenient: correct vs not-correct (the distinction the headline metric uses)
    hb = [v == "correct" for v in hv]
    jb = [v == "correct" for v in jv]
    confusion = {h: {j: sum(1 for x, y in zip(hv, jv) if x == h and y == j) for j in VERDICTS} for h in VERDICTS}
    disagreements = [{"item": i, "model": key[i]["model"], "question_id": key[i]["question_id"],
                      "human": labels[i]["verdict"], "judge": key[i]["judge_verdict"],
                      "human_faithful": labels[i]["faithful"], "judge_faithful": key[i]["judge_faithful"],
                      "note": labels[i].get("note", "")}
                     for i in both if labels[i]["verdict"] != key[i]["judge_verdict"] or labels[i]["faithful"] != key[i]["judge_faithful"]]
    result = {
        "n": len(both),
        "verdict_agreement": round(sum(x == y for x, y in zip(hv, jv)) / len(both), 3),
        "verdict_kappa": cohen_kappa(hv, jv, VERDICTS),
        "correct_vs_not_agreement": round(sum(x == y for x, y in zip(hb, jb)) / len(both), 3),
        "faithful_agreement": round(sum(x == y for x, y in zip(hf, jf)) / len(both), 3),
        "confusion_human_rows_judge_cols": confusion,
        "disagreements": disagreements,
    }
    out_name = "results.json" if key_path is None else f"results_{Path(key_path).stem.replace('key_', '')}.json"
    (CAL_DIR / out_name).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "disagreements"}, indent=2))
    for d in disagreements:
        print(f"  item {d['item']} ({d['model'].split('/')[-1]} {d['question_id']}): human {d['human']}/"
              f"{'faithful' if d['human_faithful'] else 'unfaithful'} vs judge {d['judge']}/"
              f"{'faithful' if d['judge_faithful'] else 'unfaithful'} {('— ' + d['note']) if d['note'] else ''}")


def main():
    parser = argparse.ArgumentParser(description="Human calibration of the LLM judge.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--size", type=int, default=25)
    s = sub.add_parser("score"); s.add_argument("labels"); s.add_argument("--key")
    sub.add_parser("rejudge")
    args = parser.parse_args()
    if args.cmd == "build":
        build(args.size)
    elif args.cmd == "rejudge":
        rejudge()
    else:
        score(Path(args.labels), Path(args.key) if args.key else None)


if __name__ == "__main__":
    main()
