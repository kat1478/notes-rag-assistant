# Run from the backend directory:  python -m app.render_benchmark_charts
# Renders the README charts (light + dark SVG) from benchmark results, and a retrieval top_k sweep (free:
# embeddings + Chroma only, no LLM calls). Output: docs/benchmark/*.svg and data/eval/v2/retrieval_topk.json.
#
# Chart forms follow the app's Models page: small multiples of horizontal bars (one metric each, same model
# order, emphasis on the most cost-effective model), a dot plot for repeated temperature runs, and a line chart
# for the top_k sweep. Colours are validated for both GitHub themes (light #ffffff, dark #0d1117).

import json
import re
import statistics
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import CORPORA, DATA_DIR
from .evaluation import load_evaluation_dataset

REPO_ROOT = DATA_DIR.parent.parent
OUT_DIR = REPO_ROOT / "docs" / "benchmark"
V2 = DATA_DIR / "eval" / "v2"

THEMES = {
    "light": dict(bg="#ffffff", text="#1f2328", text2="#59636e", muted="#818b98", grid="#e7e9ec", base="#c8ccd1",
                  accent="#d97706", gray="#8b8b94", s1="#2a78d6", s2="#eb6834", border="#d1d9e0"),
    "dark": dict(bg="#0d1117", text="#e6edf3", text2="#9198a1", muted="#7d8590", grid="#21262d", base="#3d444d",
                 accent="#c98500", gray="#6b6b74", s1="#3987e5", s2="#d95926", border="#30363d"),
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
W = 900
PAD = 24

REFUSALS = ["refusal_rate_unsupported", "refusal_rate_out_of_scope"]
CORRECTNESS = "judge_correct_rate_supported"
CORRECTNESS_MARGIN = 0.03  # models within 3 pp of the best judge-verified correctness count as equally good


# ── helpers ────────────────────────────────────────────────────────────────

def short_name(run: dict) -> str:
    name = run["model"].split("/")[-1]
    name = re.sub(r"-(instruct|it)\b|-2507$", "", name).replace("-a3b", "")
    return f"{name} · local" if run["provider"] == "ollama" else name


def pct(v: float) -> str:
    return f"{round(v * 100)}%"


def usd(v: float) -> str:
    return "$0 (local)" if v == 0 else f"${v:.2f}" if v >= 1 else f"${v:.3f}"


def secs(v: float) -> str:
    return f"{v:.1f} s" if v < 10 else f"{round(v)} s"


def nice_ticks(top: float) -> List[float]:
    if top <= 0:
        return [0, 1]
    raw = top / 3
    mag = 10 ** __import__("math").floor(__import__("math").log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    n = int(-(-top // step))
    return [round(i * step, 6) for i in range(n + 1)]


def bar_path(x: float, y: float, w: float, h: float, r: float = 3) -> str:
    """Bar with a rounded data end (right) and a square baseline end (left)."""
    w = max(w, 2)
    r = min(r, w, h / 2)
    return (f"M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z")


def text(x, y, s, cls="", anchor="start", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" {extra}>{escape(str(s))}</text>'


def svg(height: int, body: str, theme: str, title: str, desc: str) -> str:
    t = THEMES[theme]
    style = f"""
    text {{ font-family: {FONT}; fill: {t['text']}; }}
    .h1 {{ font-size: 16px; font-weight: 600; }}
    .sub {{ font-size: 12px; fill: {t['text2']}; }}
    .ptitle {{ font-size: 13px; font-weight: 600; }}
    .dir {{ font-size: 11px; fill: {t['muted']}; }}
    .label {{ font-size: 12px; fill: {t['text2']}; }}
    .label-em {{ font-size: 12px; font-weight: 600; fill: {t['text']}; }}
    .val {{ font-size: 12px; font-variant-numeric: tabular-nums; }}
    .tick {{ font-size: 10.5px; fill: {t['muted']}; font-variant-numeric: tabular-nums; }}
    .legend {{ font-size: 12px; fill: {t['text2']}; }}
    """
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" '
            f'role="img" aria-labelledby="t d">\n<title id="t">{escape(title)}</title>\n<desc id="d">{escape(desc)}</desc>\n'
            f'<style>{style}</style>\n'
            f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" rx="6" fill="{t["bg"]}" stroke="{t["border"]}"/>\n'
            f'{body}\n</svg>\n')


# ── chart 1: model comparison (small multiples) ────────────────────────────

PANELS = [
    ("judge_correct_rate_supported", "Correct when the note was retrieved (judge)", "higher is better", pct, 1.0),
    ("refusal_rate_unsupported", "Refuses when retrieval missed the note", "higher is better", pct, 1.0),
    ("refusal_rate_out_of_scope", "Refuses out-of-scope questions", "higher is better", pct, 1.0),
    ("judge_faithful_rate", "Answers faithful to the context (judge)", "higher is better", pct, 1.0),
    ("latency_p50_s", "Median latency", "lower is better", secs, None),
    ("cost_per_1k_queries_usd", "Cost per 1,000 questions", "lower is better", usd, None),
]


def pick_recommended(runs: List[dict]) -> Optional[str]:
    """Cheapest model within CORRECTNESS_MARGIN of the best judge-verified correctness that also refuses
    >= 95% of the time both when retrieval missed and out of scope."""
    scored = [r for r in runs if r.get(CORRECTNESS) is not None and r.get("cost_per_1k_queries_usd") is not None]
    if not scored:
        return None
    best = max(r[CORRECTNESS] for r in scored)
    ok = [r for r in scored if r[CORRECTNESS] >= best - CORRECTNESS_MARGIN and all((r.get(m) or 0) >= 0.95 for m in REFUSALS)]
    return min(ok, key=lambda r: r["cost_per_1k_queries_usd"])["model_id"] if ok else None


def models_chart(runs: List[dict], theme: str, meta: dict) -> str:
    t = THEMES[theme]
    rec = pick_recommended(runs)
    col_w = (W - 2 * PAD - 24) / 2
    label_w, value_w, row_h, bar_h = 150, 74, 22, 12
    track_w = col_w - label_w - value_w
    panel_h = 30 + len(runs) * row_h + 24
    top = 100
    out = [
        text(PAD, 32, "Grounded RAG: which model is worth paying for?", "h1"),
        text(PAD, 52, f"{meta['n']} questions · identical retrieved context for every model · corpus v2 · top_k 3 · "
                      f"temperature 0 · hidden reasoning off where the model allows it", "sub"),
    ]
    lx = PAD
    for color, label in ((t["accent"], "most cost-effective: cheapest within 3 pp of the best correctness, refusing ≥ 95%"),
                         (t["gray"], "other models")):
        out.append(f'<rect x="{lx}" y="66" width="12" height="8" rx="2" fill="{color}"/>')
        out.append(text(lx + 18, 74, label, "legend"))
        lx += 18 + len(label) * 6.3 + 22

    for i, (key, title, direction, fmt, fixed_max) in enumerate(PANELS):
        px = PAD + (i % 2) * (col_w + 24)
        py = top + (i // 2) * (panel_h + 16)
        vals = [r.get(key) for r in runs]
        vmax = fixed_max or max(v for v in vals if v is not None)
        ticks = nice_ticks(vmax) if fixed_max is None else [0, 0.5, 1.0]
        tmax = ticks[-1]
        tx = px + label_w
        out.append(text(px, py + 14, title, "ptitle"))
        out.append(text(px + col_w, py + 14, direction, "dir", "end"))
        gy0, gy1 = py + 24, py + 30 + len(runs) * row_h
        for tk in ticks:
            x = tx + track_w * tk / tmax
            out.append(f'<line x1="{x:.1f}" y1="{gy0}" x2="{x:.1f}" y2="{gy1}" stroke="{t["base"] if tk == 0 else t["grid"]}" stroke-width="1"/>')
            tick_label = pct(tk) if fixed_max else (f"${tk:g}" if key.startswith("cost") else f"{tk:g} s")
            out.append(text(x, gy1 + 14, tick_label, "tick", "middle" if 0 < tk < tmax else ("start" if tk == 0 else "end")))
        for j, r in enumerate(runs):
            y = py + 30 + j * row_h
            em = r["model_id"] == rec
            out.append(text(px, y + 12, short_name(r), "label-em" if em else "label"))
            v = r.get(key)
            if v is None:
                out.append(text(tx + track_w + 8, y + 12, "n/a", "tick"))
                continue
            out.append(f'<path d="{bar_path(tx, y + 2, track_w * v / tmax, bar_h)}" fill="{t["accent"] if em else t["gray"]}">'
                       f'<title>{escape(r["model_id"])}: {escape(fmt(v))}</title></path>')
            out.append(text(tx + track_w + 8, y + 12, fmt(v), "val"))
    height = int(top + 3 * (panel_h + 16) + 8)
    desc = "; ".join(f"{r['model_id']}: " + ", ".join(f"{k}={r.get(k)}" for k, *_ in PANELS) for r in runs)
    return svg(height, "\n".join(out), theme, "Model comparison on the notes-rag benchmark (corpus v2)", desc)


# ── chart 2: temperature (dot plot of repeated runs) ───────────────────────

TEMP_PANELS = [
    ("refusal_rate_unsupported", "Refuses: retrieval missed"),
    ("refusal_rate_out_of_scope", "Refuses: out of scope"),
    ("answer_rate_supported", "Answers: context has it"),
]
TEMP_GAP = 40


def temperature_chart(runs: Dict[str, dict], theme: str) -> Optional[str]:
    t = THEMES[theme]
    groups: Dict[Tuple[str, float], List[dict]] = {}
    for r in runs.values():
        groups.setdefault((short_name(r), r.get("temperature_requested", 0.0)), []).append(r)
    if not groups:
        return None
    rows = sorted(groups, key=lambda k: (k[0], k[1]))
    label_w, row_h = 150, 26
    col_w = (W - 2 * PAD - label_w - 2 * TEMP_GAP) / 3
    top = 96
    n_reps = max(len(v) for v in groups.values())
    out = [
        text(PAD, 32, "Does sampling temperature change grounding?", "h1"),
        text(PAD, 52, f"Each dot = one full run of the benchmark ({n_reps} runs per setting); tick = mean. "
                      "Models that accept temperature only.", "sub"),
    ]
    lx = PAD
    for color, label in ((t["s1"], "temperature 0"), (t["s2"], "temperature 0.7")):
        out.append(f'<circle cx="{lx + 5}" cy="70" r="4.5" fill="{color}"/>')
        out.append(text(lx + 16, 74, label, "legend"))
        lx += 16 + len(label) * 6.5 + 24
    all_vals = [r[k] for v in groups.values() for r in v for k, _ in TEMP_PANELS if r.get(k) is not None]
    lo = max(0.0, (min(all_vals) * 10 // 1) / 10 - 0.1) if all_vals else 0.0
    ticks = [round(lo + i * (1 - lo) / 2, 3) for i in range(3)]
    for c, (key, title) in enumerate(TEMP_PANELS):
        x0 = PAD + label_w + c * (col_w + TEMP_GAP)
        out.append(text(x0, top + 4, title, "ptitle"))
        gy0, gy1 = top + 14, top + 18 + len(rows) * row_h
        for tk in ticks:
            x = x0 + col_w * (tk - lo) / (1 - lo)
            out.append(f'<line x1="{x:.1f}" y1="{gy0}" x2="{x:.1f}" y2="{gy1}" stroke="{t["grid"]}" stroke-width="1"/>')
            anchor = "start" if tk == ticks[0] else "end" if tk == ticks[-1] else "middle"
            out.append(text(x, gy1 + 14, pct(tk), "tick", anchor))
        for j, (name, temp) in enumerate(rows):
            y = top + 18 + j * row_h + row_h / 2
            if c == 0:
                out.append(text(PAD, y + 4, f"{name}  T={temp:g}", "label"))
            color = t["s1"] if temp == 0 else t["s2"]
            vals = [r[key] for r in groups[(name, temp)] if r.get(key) is not None]
            for v in sorted(set(vals)):
                x = x0 + col_w * (v - lo) / (1 - lo)
                n_same = vals.count(v)
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" stroke="{t["bg"]}" stroke-width="2">'
                           f'<title>{escape(name)} T={temp:g}: {pct(v)} ({n_same} run{"s" if n_same > 1 else ""})</title></circle>')
                if n_same > 1:  # identical runs overlap — say so instead of hiding them
                    lx = x - 9 if x > x0 + col_w - 24 else x + 9
                    out.append(text(lx, y + 4, f"×{n_same}", "tick", "end" if lx < x else "start"))
            if vals:
                m = statistics.mean(vals)
                xm = x0 + col_w * (m - lo) / (1 - lo)
                out.append(f'<line x1="{xm:.1f}" y1="{y - 8:.1f}" x2="{xm:.1f}" y2="{y + 8:.1f}" stroke="{t["text"]}" stroke-width="1.5"/>')
    height = int(top + 18 + len(rows) * row_h + 34)
    desc = "; ".join(f"{n} T={tt:g}: " + ", ".join(f"{k}={[r.get(k) for r in groups[(n, tt)]]}" for k, _ in TEMP_PANELS)
                     for n, tt in rows)
    return svg(height, "\n".join(out), theme, "Temperature 0 vs 0.7 across repeated runs", desc)


# ── chart 3: retrieval top_k sweep (free) ──────────────────────────────────

PL_CHARS = re.compile(r"[ąćęłńóśźż]", re.I)
PL_WORDS = re.compile(r"\b(jak|jakie|czym|czy|dlaczego|co)\b", re.I)


def is_polish(s: str) -> bool:
    return bool(PL_CHARS.search(s) or PL_WORDS.search(s))


def retrieval_sweep(top_ks=(1, 2, 3, 5, 8)) -> dict:
    from .retrieval import perform_search
    from .vector_store import use_collection
    notes_dir, collection = CORPORA["v2"]
    use_collection(collection)
    note_pl = {p.name: len(PL_CHARS.findall(p.read_text(encoding="utf-8"))) > 5 for p in notes_dir.glob("*.txt")}
    items = [q for q in load_evaluation_dataset(V2 / "questions.json") if q.get("expected_files")]
    result = {"n": len(items), "top_k": list(top_ks), "series": {"same language": [], "cross-language": []}, "counts": {}}
    for k in top_ks:
        hits = {"same language": [], "cross-language": []}
        for q in items:
            files = {c.metadata.file_name for c in perform_search(q["question"], top_k=k)}
            q_pl = is_polish(q["question"])
            cross = all(note_pl.get(f, False) != q_pl for f in q["expected_files"])
            hits["cross-language" if cross else "same language"].append(bool(files & set(q["expected_files"])))
        for s, v in hits.items():
            result["series"][s].append(round(sum(v) / len(v), 4))
            result["counts"][s] = len(v)
    return result


def topk_chart(data: dict, theme: str) -> str:
    t = THEMES[theme]
    ks = data["top_k"]
    x0, x1, y0, y1 = PAD + 40, W - PAD - 140, 92, 300
    xs = {k: x0 + (x1 - x0) * i / (len(ks) - 1) for i, k in enumerate(ks)}
    yv = lambda v: y1 - (y1 - y0) * v
    out = [
        text(PAD, 32, "Retrieval: how often is the right note in the context?", "h1"),
        text(PAD, 52, f"{data['n']} in-scope questions on corpus v2 (English-centric embeddings, all-MiniLM-L6-v2). "
                      "Cross-language = question and note in different languages (EN↔PL).", "sub"),
    ]
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{x0}" y1="{yv(v):.1f}" x2="{x1}" y2="{yv(v):.1f}" stroke="{t["base"] if v == 0 else t["grid"]}" stroke-width="1"/>')
        out.append(text(x0 - 8, yv(v) + 4, pct(v), "tick", "end"))
    for k in ks:
        out.append(text(xs[k], y1 + 18, f"top_k {k}", "tick", "middle"))
    xk = xs.get(3)
    if xk:
        out.append(f'<line x1="{xk:.1f}" y1="{y0}" x2="{xk:.1f}" y2="{y1}" stroke="{t["muted"]}" stroke-width="1"/>')
        out.append(text(xk + 6, y1 - 8, "used in the model benchmark", "dir"))
    for (series, vals), color in zip(data["series"].items(), (t["s1"], t["s2"])):
        pts = " ".join(f"{xs[k]:.1f},{yv(v):.1f}" for k, v in zip(ks, vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for k, v in zip(ks, vals):
            out.append(f'<circle cx="{xs[k]:.1f}" cy="{yv(v):.1f}" r="4.5" fill="{color}" stroke="{t["bg"]}" stroke-width="2">'
                       f'<title>{series}, top_k {k}: {pct(v)}</title></circle>')
        out.append(f'<line x1="{x1 + 14}" y1="{yv(vals[-1]):.1f}" x2="{x1 + 26}" y2="{yv(vals[-1]):.1f}" stroke="{color}" stroke-width="2"/>')
        out.append(text(x1 + 32, yv(vals[-1]) + 4, f"{series} (n={data['counts'][series]})", "label"))
        out.append(text(xs[3], yv(vals[ks.index(3)]) - 10, pct(vals[ks.index(3)]), "val", "middle"))
    desc = "; ".join(f"{s}: " + ", ".join(f"top_k {k}={pct(v)}" for k, v in zip(ks, vals)) for s, vals in data["series"].items())
    return svg(y1 + 40, "\n".join(out), theme, "Retrieval hit rate by top_k, same-language vs cross-language", desc)


# ── static interactive page for GitHub Pages ──────────────────────────────

def build_pages() -> None:
    """Copies the app's Models page into docs/benchmark/ as a static page (no backend): it reads summary.json
    next to it. Served by GitHub Pages from /docs at <user>.github.io/notes-rag-assistant/benchmark/."""
    frontend = REPO_ROOT / "frontend"
    html = (frontend / "models.html").read_text(encoding="utf-8")
    html = html.replace("<body>", "<body data-static>", 1)
    html = re.sub(r'\s*<a class="tab" href="index.html">Ask</a>', "", html)
    html = re.sub(r'\s*<a href="/docs"[^>]*>API</a>', "", html)
    html = html.replace("Click a model to use it in <a href=\"index.html\">Ask</a>.",
                        "Click a model to highlight it across all charts. Static snapshot of the app's Models view.")
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")
    for name in ("models.js", "styles.css"):
        (OUT_DIR / name).write_text((frontend / name).read_text(encoding="utf-8"), encoding="utf-8")
    summary = json.loads((V2 / "models" / "summary.json").read_text(encoding="utf-8"))
    (OUT_DIR / "summary.json").write_text(json.dumps({"corpus": "v2", **summary}, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")


# ── main ───────────────────────────────────────────────────────────────────

def load_runs(path: Path) -> Dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))["runs"] if path.exists() else {}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    order = ["ollama:", "openrouter:meta-llama/", "openrouter:qwen/", "openrouter:google/",
             "openrouter:anthropic/claude-haiku", "openrouter:anthropic/claude-sonnet", "openrouter:openai/"]
    rank = lambda r: next((i for i, p in enumerate(order) if r["model_id"].startswith(p)), len(order))
    runs = sorted(load_runs(V2 / "models" / "summary.json").values(), key=rank)
    temp_runs = load_runs(V2 / "temperature" / "summary.json")
    sweep_path = V2 / "retrieval_topk.json"
    sweep = retrieval_sweep()
    sweep_path.write_text(json.dumps(sweep, indent=2) + "\n", encoding="utf-8")

    for theme in THEMES:
        (OUT_DIR / f"models-{theme}.svg").write_text(models_chart(runs, theme, {"n": runs[0]["n_questions"]}), encoding="utf-8")
        tc = temperature_chart(temp_runs, theme)
        if tc:
            (OUT_DIR / f"temperature-{theme}.svg").write_text(tc, encoding="utf-8")
        (OUT_DIR / f"retrieval-topk-{theme}.svg").write_text(topk_chart(sweep, theme), encoding="utf-8")
    build_pages()
    print(f"recommended: {pick_recommended(runs)}")
    print(f"charts → {OUT_DIR}")
    print(json.dumps(sweep["series"]))


if __name__ == "__main__":
    main()
