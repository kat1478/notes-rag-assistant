// Model comparison view — reads GET /benchmark (written by `python -m app.run_model_benchmark`).
// Form: small multiples of horizontal bars, one metric each, same model order everywhere;
// emphasis (accent) on the model selected for Ask, gray for the rest.

const API_BASE = window.location.protocol === 'file:' ? 'http://127.0.0.1:8000' : '';
const MODEL_KEY = 'notes-rag.model';

const store = {
    get() { try { return localStorage.getItem(MODEL_KEY); } catch { return null; } },
    set(v) { try { localStorage.setItem(MODEL_KEY, v); } catch { /* private mode: selection lasts this page only */ } },
};

const pct = (v) => `${Math.round(v * 100)}%`;
const secs = (v) => `${v < 10 ? v.toFixed(1) : Math.round(v)} s`;
const usd = (v) => (v === 0 ? '$0' : v < 0.01 ? `$${v.toFixed(4)}` : v < 1 ? `$${v.toFixed(3)}` : `$${v.toFixed(2)}`);

const CHARTS = [
    {
        key: 'answer_rate_in_scope', title: 'Answered in-scope questions', dir: 'higher is better',
        fmt: pct, max: () => 1,
        detail: (r) => `${Math.round(r.answer_rate_in_scope * r.n_in_scope)} of ${r.n_in_scope} questions`,
    },
    {
        key: 'valid_citation_rate', title: 'Answers citing a real chunk', dir: 'higher is better',
        fmt: pct, max: () => 1,
        detail: (r) => `${r.invalid_citations} citation(s) to chunks not in context`,
    },
    {
        key: 'latency_p50_s', title: 'Median latency', dir: 'lower is better',
        fmt: secs, tick: (t) => `${t} s`, max: (rows) => Math.max(...rows.map((r) => r.latency_p50_s ?? 0)),
        detail: (r) => `mean ${secs(r.latency_mean_s ?? 0)}${r.provider === 'ollama' ? ' · local CPU' : ' · hosted'}`,
    },
    {
        key: 'cost_per_1k_queries_usd', title: 'Cost per 1,000 questions', dir: 'lower is better',
        fmt: usd, tick: (t) => `$${t}`, max: (rows) => Math.max(...rows.map((r) => r.cost_per_1k_queries_usd ?? 0)),
        detail: (r) => (r.provider === 'ollama' ? 'local inference, hardware not counted'
            : `~${r.prompt_tokens_mean ?? '?'} in / ${r.completion_tokens_mean ?? '?'} out tokens per question`),
    },
];

const state = { rows: [], selected: store.get() };

function shortName(r) {
    const name = r.model.includes('/') ? r.model.split('/').pop() : r.model;
    const temp = r.temperature_requested ? ` · t=${r.temperature_requested}` : '';
    return `${name}${r.provider === 'ollama' ? ' · local' : ''}${temp}`;
}

// 0 + "nice" steps (1, 2, 2.5, 5 × 10^k) covering max, 3–5 ticks.
function niceTicks(max) {
    if (!(max > 0)) return [0, 1];
    const raw = max / 3;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const top = Math.ceil(max / step) * step;
    const ticks = [];
    for (let t = 0; t <= top + step / 2; t += step) ticks.push(Number(t.toPrecision(6)));
    return ticks;
}

function node(tag, className, text) {
    const n = document.createElement(tag);
    if (className) n.className = className;
    if (text !== undefined) n.textContent = text;
    return n;
}

// ── Tooltip ────────────────────────────────────────────────

const tip = document.getElementById('tooltip');

function showTip(row, chart, x, y) {
    const v = row[chart.key];
    tip.replaceChildren(
        node('strong', '', v === null || v === undefined ? 'n/a' : chart.fmt(v)),
        node('div', 'tt-model', row.model_id),
        node('div', 'tt-detail', chart.detail(row)),
    );
    tip.hidden = false;
    const { width, height } = tip.getBoundingClientRect();
    tip.style.left = `${Math.min(x + 14, window.innerWidth - width - 8)}px`;
    tip.style.top = `${Math.max(8, y - height - 10)}px`;
}

function hideTip() { tip.hidden = true; }

// ── Rendering ──────────────────────────────────────────────

function renderChart(chart) {
    const rows = state.rows;
    const ticks = niceTicks(chart.max(rows));
    const top = ticks[ticks.length - 1];

    const card = node('section', 'chart');
    const head = node('header', 'chart-head');
    head.append(node('h2', '', chart.title), node('span', 'chart-dir', chart.dir));

    const plot = node('div', 'plot');
    const grid = node('div', 'plot-grid');
    ticks.forEach((t) => {
        const line = node('i');
        line.style.left = `${(t / top) * 100}%`;
        grid.append(line);
    });
    plot.append(grid);

    rows.forEach((r) => {
        const v = r[chart.key];
        const btn = node('button', 'bar-row');
        btn.type = 'button';
        btn.dataset.model = r.model_id;
        btn.setAttribute('aria-pressed', String(r.model_id === state.selected));
        btn.setAttribute('aria-label', `${r.model_id}: ${v === null || v === undefined ? 'n/a' : chart.fmt(v)}. Select for Ask.`);

        const track = node('span', 'bar-track');
        if (v !== null && v !== undefined) {
            const bar = node('span', 'bar');
            bar.style.width = `${(v / top) * 100}%`;
            track.append(bar);
        }
        const value = node('span', 'bar-value', v === null || v === undefined ? 'n/a' : chart.fmt(v));
        if (v === null || v === undefined) value.classList.add('is-na');
        btn.append(node('span', 'bar-label', shortName(r)), track, value);

        btn.addEventListener('pointermove', (e) => showTip(r, chart, e.clientX, e.clientY));
        btn.addEventListener('pointerleave', hideTip);
        btn.addEventListener('focus', () => {
            const rect = btn.getBoundingClientRect();
            showTip(r, chart, rect.left + rect.width / 2, rect.top);
        });
        btn.addEventListener('blur', hideTip);
        btn.addEventListener('click', () => select(r.model_id));
        plot.append(btn);
    });

    const axis = node('div', 'axis');
    const tickRow = node('div', 'axis-ticks');
    ticks.forEach((t) => {
        const label = node('span', '', (chart.tick || chart.fmt)(t));
        label.style.left = `${(t / top) * 100}%`;
        tickRow.append(label);
    });
    axis.append(node('span'), tickRow, node('span'));
    plot.append(axis);

    card.append(head, plot);
    return card;
}

function renderTable() {
    const body = document.querySelector('#metrics-table tbody');
    body.replaceChildren(...state.rows.map((r) => {
        const tr = node('tr');
        const na = (v, f) => (v === null || v === undefined ? '—' : f(v));
        const cells = [
            r.model_id,
            na(r.answer_rate_in_scope, (v) => `${pct(v)} (n=${r.n_in_scope})`),
            na(r.valid_citation_rate, pct),
            r.invalid_citations,
            na(r.refusal_rate_out_of_scope, (v) => `${pct(v)} (n=${r.n_out_of_scope})`),
            r.failures,
            na(r.latency_p50_s, secs),
            na(r.latency_mean_s, secs),
            r.prompt_tokens_mean ? `${Math.round(r.prompt_tokens_mean)} / ${Math.round(r.completion_tokens_mean ?? 0)}` : '—',
            na(r.cost_per_1k_queries_usd, usd),
        ];
        cells.forEach((c, i) => tr.append(node(i === 0 ? 'th' : 'td', '', String(c))));
        tr.firstChild.scope = 'row';
        tr.dataset.model = r.model_id;
        return tr;
    }));
}

function applySelection() {
    document.querySelectorAll('[data-model]').forEach((el) => {
        const on = el.dataset.model === state.selected;
        el.classList.toggle('is-selected', on);
        if (el.tagName === 'BUTTON') el.setAttribute('aria-pressed', String(on));
    });
}

function select(modelId) {
    state.selected = modelId;
    store.set(modelId);
    applySelection();
}

function renderMeta(runs) {
    const uniq = (k) => [...new Set(runs.map((r) => r[k]).filter(Boolean))].join(', ');
    const dates = runs.map((r) => r.run_at).filter(Boolean).sort();
    const range = dates.length ? `${dates[0].slice(0, 10)}${dates.at(-1).slice(0, 10) !== dates[0].slice(0, 10) ? ` → ${dates.at(-1).slice(0, 10)}` : ''}` : '';
    document.getElementById('run-meta').textContent =
        `${uniq('dataset')} · ${uniq('n_questions')} questions · top_k ${uniq('top_k')} · run ${range} · commit ${uniq('git_commit')}`;
}

async function fetchJson(path) {
    const res = await fetch(`${API_BASE}${path}`);
    if (!res.ok) throw Object.assign(new Error(`HTTP ${res.status}`), { status: res.status });
    return res.json();
}

async function init() {
    let summary;
    try {
        summary = await fetchJson('/benchmark');
    } catch (err) {
        const notice = document.getElementById('notice');
        notice.hidden = false;
        notice.textContent = err.status === 404
            ? 'No benchmark results yet. From backend/ run: python -m app.run_model_benchmark'
            : `Backend not reachable (${err.message}). Start it from backend/: python -m uvicorn app.main:app`;
        return;
    }

    // Stable order = the backend's model list (local first), so a model keeps its row in every chart.
    let order = [];
    try { order = (await fetchJson('/models')).map((m) => m.id); } catch { /* fall back to run order */ }
    const runs = Object.values(summary.runs || {});
    const rank = (id) => (order.includes(id) ? order.indexOf(id) : order.length);
    state.rows = runs.sort((a, b) => rank(a.model_id) - rank(b.model_id));

    if (!state.selected) {
        try { state.selected = (await fetchJson('/health')).default_model_id; } catch { /* no emphasis */ }
    }

    renderMeta(runs);
    document.getElementById('charts').replaceChildren(...CHARTS.map(renderChart));
    renderTable();
    applySelection();
}

window.addEventListener('scroll', hideTip, { passive: true });
init();
