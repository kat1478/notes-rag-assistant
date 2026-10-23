# Run from the backend directory:  python -m app.compare_embeddings [--models <hf-id> ...]
# Compares embedding models on retrieval only (free, local): the v2 chunks, the in-scope questions of v2 + the
# Polish probe, cosine top-k in numpy and a file-level hit exactly like the benchmark (checked: the legacy model
# reproduces the benchmark's misses). Split by same-language vs cross-language (question vs note language).
# Results: data/eval/v2/embeddings_comparison.json (read by render_benchmark_charts).

import argparse
import json
import re
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import CORPORA, DATA_DIR
from .language import detect_language
from .vector_store import get_collection, use_collection

V2 = DATA_DIR / "eval" / "v2"
OUT = V2 / "embeddings_comparison.json"
PL_CHARS = re.compile(r"[ąćęłńóśźż]", re.I)

MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {},
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {},
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {},
    "intfloat/multilingual-e5-small": {"query": "query: ", "passage": "passage: "},
    "intfloat/multilingual-e5-base": {"query": "query: ", "passage": "passage: "},
    "BAAI/bge-m3": {},
}
KS = (1, 2, 3, 5, 8)


def load_chunks():
    # Chunk boundaries don't depend on the embedding model, so any v2 collection gives the same texts.
    use_collection(CORPORA["v2"][1])
    data = get_collection().get(include=["documents", "metadatas"])
    order = sorted(range(len(data["ids"])), key=lambda i: (data["metadatas"][i]["file_name"], data["metadatas"][i]["chunk_index"]))
    return [data["documents"][i] for i in order], [data["metadatas"][i]["file_name"] for i in order]


def load_questions():
    qs = json.load(open(V2 / "questions.json", encoding="utf-8")) + json.load(open(V2 / "questions_pl_probe.json", encoding="utf-8"))
    return [q for q in qs if q.get("expected_files")]


def note_languages(files):
    notes_dir = CORPORA["v2"][0]
    return {f: ("pl" if len(PL_CHARS.findall((notes_dir / f).read_text(encoding="utf-8"))) > 5 else "en") for f in set(files)}


def evaluate(name, prefixes, texts, files, questions, note_lang):
    t0 = time.perf_counter()
    model = SentenceTransformer(name, device="cpu")
    load_s = time.perf_counter() - t0
    doc = model.encode([prefixes.get("passage", "") + t for t in texts], normalize_embeddings=True, batch_size=16)
    qtexts = [prefixes.get("query", "") + q["question"] for q in questions]
    for t in qtexts[:5]:  # warm-up: the first calls include lazy initialisation
        model.encode([t], normalize_embeddings=True)
    times = []
    for t in qtexts[:30]:  # single-query latency, like one /ask; median is robust to laptop CPU noise
        t0 = time.perf_counter()
        model.encode([t], normalize_embeddings=True)
        times.append((time.perf_counter() - t0) * 1000)
    query_ms = float(np.median(times))
    qv = model.encode(qtexts, normalize_embeddings=True, batch_size=16)
    sims = qv @ doc.T

    per_q = []
    for q, s in zip(questions, sims):
        ranked_files = [files[i] for i in np.argsort(-s)]
        expected = set(q["expected_files"])
        first = next((r for r, f in enumerate(ranked_files, 1) if f in expected), None)
        qlang = detect_language(q["question"])
        cross = all(note_lang[f] != qlang for f in expected)
        per_q.append({"id": q["id"], "cross": cross, "first_rank": first,
                      })
        for k in KS:
            per_q[-1][f"hit@{k}"] = bool(expected & set(ranked_files[:k]))

    def agg(rows):
        if not rows:
            return None
        return {"n": len(rows), **{f"hit@{k}": round(sum(r[f"hit@{k}"] for r in rows) / len(rows), 3) for k in KS},
                "mrr": round(sum(1 / r["first_rank"] for r in rows if r["first_rank"]) / len(rows), 3)}

    params = sum(p.numel() for p in model.parameters()) / 1e6
    return {"model": name, "params_m": round(params), "load_s": round(load_s, 1), "query_ms": round(query_ms, 1),
            "all": agg(per_q), "same_language": agg([r for r in per_q if not r["cross"]]),
            "cross_language": agg([r for r in per_q if r["cross"]]),
            "misses_at_3": [r["id"] for r in per_q if not r["hit@3"]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()
    texts, files = load_chunks()
    questions = load_questions()
    note_lang = note_languages(files)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in args.models:
        print(f"=== {name}", flush=True)
        r = evaluate(name, MODELS[name], texts, files, questions, note_lang)
        results[name] = r
        OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in r.items() if k != "misses_at_3"}), flush=True)


if __name__ == "__main__":
    main()
