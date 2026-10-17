// notes-rag frontend — vanilla JS, talks to the FastAPI backend (/health, /ask).

// Refusal sentence fixed by the grounding prompt in backend/app/answering.py.
const REFUSAL_RE = /don['’]t have enough information/i;
const TOP_K_MIN = 1;
const TOP_K_MAX = 10;

// Served by FastAPI → same origin; opened as a file → assume the default local server.
const API_BASE = window.location.protocol === 'file:' ? 'http://127.0.0.1:8000' : '';

const MODEL_KEY = 'notes-rag.model';  // shared with models.html

const store = {
    get() { try { return localStorage.getItem(MODEL_KEY); } catch { return null; } },
    set(v) { try { localStorage.setItem(MODEL_KEY, v); } catch { /* private mode: selection lasts this page only */ } },
};

const $ = (id) => document.getElementById(id);

const el = {
    form: $('ask-form'),
    input: $('question-input'),
    submit: $('submit-btn'),
    topk: $('topk-value'),
    model: $('model-select'),
    loadingNote: $('loading-note'),
    status: $('backend-status'),
    statusLabel: document.querySelector('.backend-label'),
    answerStatus: $('answer-status'),
    answerTime: $('answer-time'),
    views: {
        empty: $('view-empty'),
        loading: $('view-loading'),
        error: $('view-error'),
        answer: $('view-answer'),
    },
    elapsed: $('elapsed'),
    errorDetail: $('error-detail'),
    question: $('answer-question'),
    answer: $('answer-text'),
    sources: $('sources-list'),
    contextSummary: $('context-summary'),
    stats: {
        retrieval: $('stat-retrieval'),
        generation: $('stat-generation'),
        topk: $('stat-topk'),
        chunks: $('stat-chunks'),
        cited: $('stat-cited'),
        tokens: $('stat-tokens'),
        latency: $('stat-latency'),
    },
};

const state = {
    topK: 3,
    lastQuestion: '',
    models: [],
    defaultModel: null,
    pinned: null,   // source number kept highlighted after a click
    timer: null,
};

// ── Backend status ─────────────────────────────────────────

function setBackend(stateName, label) {
    el.status.dataset.state = stateName;
    el.statusLabel.textContent = label;
}

async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        state.defaultModel = data.default_model_id || `ollama:${data.ollama_model}`;
        setBackend('up', 'backend ready');
        if (Number.isInteger(data.top_k_default) && !state.lastQuestion) setTopK(data.top_k_default);
        return true;
    } catch {
        setBackend('down', 'backend offline');
        return false;
    }
}

// ── top_k stepper ──────────────────────────────────────────

function setTopK(value) {
    state.topK = Math.min(TOP_K_MAX, Math.max(TOP_K_MIN, value));
    el.topk.textContent = state.topK;
}

document.querySelectorAll('.topk-btn').forEach((btn) => {
    btn.addEventListener('click', () => setTopK(state.topK + Number(btn.dataset.step)));
});

// ── View switching ─────────────────────────────────────────

function show(viewName) {
    Object.entries(el.views).forEach(([name, node]) => { node.hidden = name !== viewName; });
}

function startTimer() {
    const t0 = performance.now();
    el.elapsed.textContent = '0.0 s';
    state.timer = setInterval(() => {
        el.elapsed.textContent = `${((performance.now() - t0) / 1000).toFixed(1)} s`;
    }, 100);
}

function stopTimer() {
    clearInterval(state.timer);
    state.timer = null;
}

function resetResult() {
    el.answerStatus.hidden = true;
    el.answerTime.textContent = '';
    el.sources.innerHTML = '<li class="sources-empty">Waiting for retrieval…</li>';
    el.contextSummary.textContent = 'chunks sent to the LLM, in prompt order';
    Object.values(el.stats).forEach((node) => { node.textContent = '—'; });
    state.pinned = null;
}

// ── Ask ────────────────────────────────────────────────────

async function ask(question) {
    question = question.trim();
    if (!question) return;
    state.lastQuestion = question;

    resetResult();
    el.loadingNote.textContent = isLocal(el.model.value)
        ? 'Local CPU inference takes ~30–45 s. The first request after startup is slower while the model loads.'
        : 'Hosted model via OpenRouter — usually a few seconds.';
    show('loading');
    el.submit.disabled = true;
    startTimer();

    try {
        const res = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: question, top_k: state.topK, model: el.model.value || undefined }),
        });
        if (!res.ok) {
            const detail = await res.text().catch(() => '');
            throw new Error(`HTTP ${res.status}${detail ? ` — ${detail.slice(0, 200)}` : ''}`);
        }
        const data = await res.json();
        renderResult(question, data);
        show('answer');
        checkHealth();
    } catch (err) {
        console.error('Ask failed:', err);
        const up = await checkHealth();
        el.errorDetail.textContent = up
            ? `The backend is up but the request failed: ${err.message}`
            : `No response from ${API_BASE || window.location.origin}. (${err.message})`;
        el.sources.innerHTML = '<li class="sources-empty">No context — the request did not complete.</li>';
        show('error');
    } finally {
        stopTimer();
        el.submit.disabled = false;
    }
}

// ── Rendering ──────────────────────────────────────────────

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Matches "[Source 2]", "[Source 1, 3]", "[Sources 1 and 2]".
const CITATION_RE = /\[Sources?\s*((?:\d+\s*(?:,|and|&)?\s*)+)\]/gi;

function renderAnswer(text, sourceCount) {
    const cited = new Set();
    let missing = 0;

    const html = escapeHtml(text).replace(CITATION_RE, (_, nums) => {
        return nums.match(/\d+/g).map((n) => {
            const num = Number(n);
            if (num < 1 || num > sourceCount) {
                missing += 1;
                return `<span class="cite is-missing" title="Cites source ${num}, which was not in the context">${num}</span>`;
            }
            cited.add(num);
            return `<button type="button" class="cite" data-source="${num}" aria-label="Show source ${num}">${num}</button>`;
        }).join('');
    });

    el.answer.innerHTML = html;
    return { cited, missing };
}

function classify(text, citedCount, error) {
    if (error) return { kind: 'failed', label: 'generation failed', hint: error };
    if (citedCount > 0) return { kind: 'grounded', label: 'grounded' };
    if (REFUSAL_RE.test(text)) return { kind: 'refused', label: 'refused · out of scope' };
    return {
        kind: 'uncited',
        label: 'no citation',
        hint: 'No [Source N] citation and not the standard refusal sentence — unverifiable answer or a non-standard refusal.',
    };
}

function renderSources(sources, cited) {
    if (!sources.length) {
        el.sources.innerHTML = '<li class="sources-empty">No chunks were retrieved for this question.</li>';
        return;
    }

    el.sources.innerHTML = sources.map((src, i) => {
        const n = i + 1;
        const meta = src.metadata || {};
        const retrieved = src.score !== null && src.score !== undefined;
        const isCited = cited.has(n);
        const classes = ['source', isCited && 'is-cited', isCited && 'is-open'].filter(Boolean).join(' ');
        return `
            <li class="${classes}" id="source-${n}" data-source="${n}">
                <button type="button" class="source-row" aria-expanded="${isCited}" aria-controls="source-text-${n}">
                    <span class="src-n">${isCited ? '●' : ''}${n}</span>
                    <span class="src-id">${escapeHtml(meta.file_name || 'unknown')}<span class="chunk">:${meta.chunk_index ?? '?'}</span></span>
                    <span class="src-kind" data-kind="${retrieved ? 'retrieved' : 'neighbor'}">${retrieved ? 'retrieved' : '±1 neighbor'}</span>
                    <span class="src-score" title="${retrieved ? 'Raw retrieval score: Chroma distance (lower is closer); TF-IDF fallback reports similarity (higher is closer)' : 'Neighbor chunks have no score'}">${retrieved ? src.score.toFixed(3) : '—'}</span>
                    <span class="src-caret" aria-hidden="true">›</span>
                </button>
                <div class="source-text" id="source-text-${n}" ${isCited ? '' : 'hidden'}>${escapeHtml(src.text)}</div>
            </li>`;
    }).join('');
}

function renderStats(meta, sources, cited, missing) {
    const retrievedCount = sources.filter((s) => s.score !== null && s.score !== undefined).length;
    const neighbors = sources.length - retrievedCount;
    const ms = meta?.response_time_ms;

    el.stats.retrieval.textContent = meta?.retrieval_mode ?? '—';
    el.stats.generation.textContent = meta?.generation_mode ?? '—';
    el.stats.topk.textContent = meta?.top_k_used ?? state.topK;
    el.stats.chunks.textContent = `${sources.length} (${retrievedCount} retrieved + ${neighbors} neighbors)`;
    el.stats.cited.textContent = `${cited.size} / ${sources.length}${missing ? ` · ${missing} invalid` : ''}`;
    el.stats.tokens.textContent = meta?.prompt_tokens ? `${meta.prompt_tokens} in · ${meta.completion_tokens ?? '?'} out` : '—';
    el.stats.latency.textContent = ms !== undefined ? `${(ms / 1000).toFixed(1)} s` : '—';
    el.answerTime.textContent = ms !== undefined ? `${(ms / 1000).toFixed(1)} s` : '';
}

function renderResult(question, data) {
    const sources = data.sources || [];
    const answer = data.answer || '';

    el.question.textContent = question;
    const { cited, missing } = renderAnswer(answer, sources.length);
    renderSources(sources, cited);
    renderStats(data.metadata, sources, cited, missing);

    const status = classify(answer, cited.size, data.metadata?.generation_error);
    el.answerStatus.dataset.kind = status.kind;
    el.answerStatus.textContent = status.label;
    el.answerStatus.title = status.hint || '';
    el.answerStatus.hidden = false;
    el.contextSummary.textContent = `${sources.length} chunks · ${cited.size} cited`;
}

// ── Citation ↔ source linking ──────────────────────────────

function highlight(n) {
    document.querySelectorAll('.is-active').forEach((node) => node.classList.remove('is-active'));
    if (n === null) return;
    document.querySelectorAll(`[data-source="${n}"]`).forEach((node) => node.classList.add('is-active'));
}

function openSource(n, open = true) {
    const item = $(`source-${n}`);
    if (!item) return;
    item.classList.toggle('is-open', open);
    item.querySelector('.source-row').setAttribute('aria-expanded', String(open));
    item.querySelector('.source-text').hidden = !open;
}

el.answer.addEventListener('mouseover', (e) => {
    const cite = e.target.closest('.cite[data-source]');
    if (cite) highlight(cite.dataset.source);
});
el.answer.addEventListener('mouseout', (e) => {
    if (e.target.closest('.cite[data-source]')) highlight(state.pinned);
});
el.answer.addEventListener('click', (e) => {
    const cite = e.target.closest('.cite[data-source]');
    if (!cite) return;
    const n = cite.dataset.source;
    state.pinned = state.pinned === n ? null : n;
    highlight(state.pinned);
    if (state.pinned) {
        openSource(n, true);
        $(`source-${n}`).scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});

el.sources.addEventListener('click', (e) => {
    const row = e.target.closest('.source-row');
    if (!row) return;
    const item = row.parentElement;
    openSource(item.dataset.source, !item.classList.contains('is-open'));
});
el.sources.addEventListener('mouseover', (e) => {
    const item = e.target.closest('.source');
    if (item && item.classList.contains('is-cited')) highlight(item.dataset.source);
});
el.sources.addEventListener('mouseleave', () => highlight(state.pinned));

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { state.pinned = null; highlight(null); }
    if (e.key === '/' && document.activeElement !== el.input) { e.preventDefault(); el.input.focus(); }
});

// ── Model picker ───────────────────────────────────────────

const isLocal = (modelId) => (modelId || state.defaultModel || '').startsWith('ollama:');

function modelLabel(m) {
    const name = m.model.includes('/') ? m.model.split('/').pop() : m.model;
    return `${name}${m.local ? ' · local' : ''}${m.available ? '' : ' (needs API key)'}`;
}

async function loadModels() {
    try {
        const res = await fetch(`${API_BASE}/models`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        state.models = await res.json();
    } catch {
        return;  // keep the picker empty → backend default model is used
    }
    el.model.replaceChildren(...state.models.map((m) => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = modelLabel(m);
        opt.disabled = !m.available;
        return opt;
    }));
    const saved = store.get();
    const usable = (id) => state.models.some((m) => m.id === id && m.available);
    el.model.value = usable(saved) ? saved : (usable(state.defaultModel) ? state.defaultModel : state.models[0]?.id);
}

el.model.addEventListener('change', () => store.set(el.model.value));

// ── Wiring ─────────────────────────────────────────────────

el.form.addEventListener('submit', (e) => {
    e.preventDefault();
    ask(el.input.value);
});

document.querySelectorAll('.chip[data-question]').forEach((chip) => {
    chip.addEventListener('click', () => {
        el.input.value = chip.dataset.question;
        ask(chip.dataset.question);
    });
});

$('retry-btn').addEventListener('click', () => {
    if (state.lastQuestion) ask(state.lastQuestion);
    else checkHealth();
});

el.sources.innerHTML = '<li class="sources-empty">No query yet. Retrieved chunks will appear here.</li>';
checkHealth().then(loadModels);
