// notes-rag frontend — vanilla JS, talks to the FastAPI backend (/health, /ask).

// Refusal sentence fixed by the grounding prompt in backend/app/answering.py.
const REFUSAL_RE = /don['’]t have enough information|nie mam wystarczających informacji/i;
const TOP_K_MIN = 1;
const TOP_K_MAX = 10;
const TEMP_MAX = 1.0;   // matches backend TEMPERATURE_MAX
const TEMP_STEP = 0.1;

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
    temp: $('temp-value'),
    tempBox: $('temp-box'),
    loadingNote: $('loading-note'),
    status: $('backend-status'),
    statusLabel: document.querySelector('.backend-label'),
    answerStatus: $('answer-status'),
    answerTime: $('answer-time'),
    views: {
        loading: $('view-loading'),
        error: $('view-error'),
        answer: $('view-answer'),
    },
    elapsed: $('elapsed'),
    errorDetail: $('error-detail'),
    errorTitle: $('error-title'),
    errorHelp: $('error-help'),
    useDefault: $('use-default-btn'),
    hero: $('hero'),
    workspace: $('workspace'),
    runStats: $('run-stats'),
    notice: $('answer-notice'),
    question: $('answer-question'),
    answer: $('answer-text'),
    sources: $('sources-list'),
    contextSummary: $('context-summary'),
    stats: {
        retrieval: $('stat-retrieval'),
        generation: $('stat-generation'),
        topk: $('stat-topk'),
        temp: $('stat-temp'),
        chunks: $('stat-chunks'),
        cited: $('stat-cited'),
        tokens: $('stat-tokens'),
        latency: $('stat-latency'),
    },
};

const state = {
    topK: 3,
    temperature: 0,
    lastQuestion: '',
    models: [],
    defaultModel: null,
    pinned: null,   // source number kept highlighted after a click
    timer: null,
    busy: false,    // one question at a time (chips and Retry bypass the disabled Ask button)
    demoMode: false,   // public demo on AWS Lambda (from /health): cold starts after idle, rate-limited /ask
};

// The serverless demo needs ~12 s to start after a quiet period, plus ~3 s to load the embedding model.
const SLOW_HEALTH_MS = 3000;
const COLD_START_NOTE_MS = 5000;
const COLD_START_NOTE = 'The public demo runs on AWS Lambda and scales to zero when idle — the first question '
    + 'after a quiet period takes ~15 s while it starts. Later questions take a few seconds.';

// ── Backend status ─────────────────────────────────────────

function setBackend(stateName, label) {
    el.status.dataset.state = stateName;
    el.statusLabel.textContent = label;
}

async function checkHealth() {
    // A cold start makes the first health check slow; say so instead of looking stuck on "connecting…".
    const slow = setTimeout(() => setBackend('waking', 'waking up…'), SLOW_HEALTH_MS);
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        state.defaultModel = data.default_model_id || `ollama:${data.ollama_model}`;
        state.demoMode = data.demo_mode === true;
        setBackend('up', state.demoMode ? 'demo ready' : 'backend ready');
        if (Number.isInteger(data.top_k_default) && !state.lastQuestion) setTopK(data.top_k_default);
        if (typeof data.temperature_default === 'number' && !state.lastQuestion) setTemperature(data.temperature_default);
        return true;
    } catch {
        setBackend('down', 'backend offline');
        return false;
    } finally {
        clearTimeout(slow);
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

// ── temperature stepper ────────────────────────────────────

function setTemperature(value) {
    // Work in tenths to avoid 0.30000000000000004.
    const tenths = Math.round(value / TEMP_STEP);
    state.temperature = Math.min(TEMP_MAX / TEMP_STEP, Math.max(0, tenths)) / 10;
    el.temp.textContent = state.temperature.toFixed(1);
}

document.querySelectorAll('.temp-btn').forEach((btn) => {
    btn.addEventListener('click', () => setTemperature(state.temperature + Number(btn.dataset.step) * TEMP_STEP));
});

// ── View switching ─────────────────────────────────────────

function show(viewName) {
    el.hero.hidden = true;
    el.workspace.hidden = false;
    document.body.classList.add('has-result');
    el.runStats.hidden = viewName !== 'answer';
    Object.entries(el.views).forEach(([name, node]) => { node.hidden = name !== viewName; });
}

function startTimer() {
    const t0 = performance.now();
    el.elapsed.textContent = '0.0 s';
    state.timer = setInterval(() => {
        const ms = performance.now() - t0;
        el.elapsed.textContent = `${(ms / 1000).toFixed(1)} s`;
        if (state.demoMode && ms > COLD_START_NOTE_MS && el.loadingNote.textContent !== COLD_START_NOTE) {
            el.loadingNote.textContent = COLD_START_NOTE;
        }
    }, 100);
}

function stopTimer() {
    clearInterval(state.timer);
    state.timer = null;
}

function resetResult() {
    el.answerStatus.hidden = true;
    el.answerTime.textContent = '';
    el.sources.innerHTML = '<li class="sources-empty">Searching the notes…</li>';
    el.contextSummary.textContent = 'sent to the model, in this order';
    Object.values(el.stats).forEach((node) => { node.textContent = '—'; });
    state.pinned = null;
}

// ── Ask ────────────────────────────────────────────────────

async function ask(question) {
    question = question.trim();
    if (!question || state.busy) return;
    state.busy = true;
    state.lastQuestion = question;

    resetResult();
    el.loadingNote.textContent = isLocal(el.model.value)
        ? 'Local CPU inference takes ~30–45 s. The first request after startup is slower while the model loads.'
        : 'Hosted model via OpenRouter — usually a few seconds.';
    show('loading');
    el.submit.disabled = true;
    el.submit.textContent = 'Asking…';
    startTimer();

    try {
        const res = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: question,
                top_k: state.topK,
                temperature: state.temperature,
                model: el.model.value || undefined,
            }),
        });
        if (res.status === 429) {
            showError('Question limit reached',
                'The public demo allows 10 questions per hour from one address, to keep its API costs bounded. '
                + 'Try again later, or run the app locally (see the README).', false);
            return;
        }
        if (!res.ok) {
            const detail = await res.text().catch(() => '');
            throw new Error(`HTTP ${res.status}${detail ? ` — ${detail.slice(0, 200)}` : ''}`);
        }
        const data = await res.json();
        show(renderResult(question, data));
        checkHealth();
    } catch (err) {
        console.error('Ask failed:', err);
        const up = await checkHealth();
        showError(up ? 'Request failed' : 'Backend not reachable', up
            ? `The backend is up but the request failed: ${err.message}`
            : `No response from ${API_BASE || window.location.origin}. (${err.message})`, !up && !state.demoMode);
    } finally {
        stopTimer();
        el.submit.disabled = false;
        el.submit.textContent = 'Ask';
        state.busy = false;
    }
}

function showError(title, detail, showStartHelp) {
    el.errorTitle.textContent = title;
    el.errorDetail.textContent = detail;
    el.useDefault.hidden = true;
    el.errorHelp.hidden = !showStartHelp;   // "start uvicorn" only helps when running locally
    el.sources.innerHTML = '<li class="sources-empty">No context — the request did not complete.</li>';
    show('error');
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

    // Hosted models answer in light Markdown; render **bold** (on already-escaped text, so no HTML injection).
    const html = escapeHtml(text).replace(/\*\*([^\n]+?)\*\*/g, '<strong>$1</strong>').replace(CITATION_RE, (_, nums) => {
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
    if (error) return { kind: 'failed', label: 'no answer', hint: error };
    // "cited", not "grounded": the UI only knows the answer points at chunks that were in the context, not that
    // those chunks support each claim (that is what the offline judge measures).
    if (citedCount > 0) return {
        kind: 'cited',
        label: 'cited',
        hint: 'Cites sources that were in the context. Whether they support every claim is checked offline by the judge, not here.',
    };
    // "not in context", not "out of scope": a refusal can also mean retrieval missed a note that does hold the answer.
    if (REFUSAL_RE.test(text)) return {
        kind: 'refused',
        label: 'refused · not in context',
        hint: 'The retrieved chunks do not answer this. Either the notes do not cover it or retrieval missed the right note.',
    };
    return {
        kind: 'uncited',
        label: 'no citation',
        hint: 'No [Source N] citation and not the standard refusal sentence — unverifiable answer or a non-standard refusal.',
    };
}

function renderSources(sources, cited) {
    if (!sources.length) {
        el.sources.innerHTML = '<li class="sources-empty">No passages were retrieved for this question.</li>';
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
                    <span class="src-n">${n}</span>
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
    el.stats.temp.textContent = typeof meta?.temperature === 'number' ? meta.temperature.toFixed(1) : 'n/a';
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
    const error = data.metadata?.generation_error;
    // A failed generation carries the raw context as its text; the Sources panel already shows those passages.
    const { cited, missing } = renderAnswer(error ? '' : answer, sources.length);
    renderSources(sources, cited);
    renderStats(data.metadata, sources, cited, missing);

    const status = classify(answer, cited.size, error);
    el.answerStatus.dataset.kind = status.kind;
    el.answerStatus.textContent = status.label;
    el.answerStatus.title = status.hint || '';
    el.answerStatus.hidden = false;
    el.contextSummary.textContent = `${sources.length} passages · ${cited.size} cited`;

    const fallbackFrom = data.metadata?.fallback_from;
    el.notice.hidden = !fallbackFrom;
    if (fallbackFrom) {
        el.notice.textContent = `${shortModel(fallbackFrom)} is unavailable right now, so ${shortModel(data.metadata.model)} answered instead.`;
    }
    if (error) {
        const failed = data.metadata.model;
        el.errorTitle.textContent = `${shortModel(failed)} could not answer`;
        el.errorDetail.textContent = /error 429/.test(error)
            ? 'The model provider is limiting requests at the moment. The passages found for your question are listed under Sources.'
            : `${error}. The passages found for your question are listed under Sources.`;
        el.errorHelp.hidden = true;
        const canSwitch = state.defaultModel && failed !== state.defaultModel;
        el.useDefault.hidden = !canSwitch;
        if (canSwitch) el.useDefault.textContent = `Ask ${shortModel(state.defaultModel)}`;
        return 'error';
    }
    return 'answer';
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

const shortModel = (id) => (id || 'The model').split(/[:/]/).pop();

function modelLabel(m) {
    const name = m.model.includes('/') ? m.model.split('/').pop() : m.model;
    return `${name}${m.local ? ' · local' : ''}${m.id === state.defaultModel ? ' (recommended)' : ''}${m.available ? '' : ' (needs API key)'}`;
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
    syncTemperatureSupport();
}

// Reasoning models (e.g. Claude 4.5+, GPT-5.x) don't accept temperature — hide the control.
function syncTemperatureSupport() {
    const m = state.models.find((x) => x.id === el.model.value);
    const off = m?.supports_temperature === false;
    el.tempBox.hidden = off;
}

el.model.addEventListener('change', () => {
    store.set(el.model.value);
    syncTemperatureSupport();
});

// ── Wiring ─────────────────────────────────────────────────

el.form.addEventListener('submit', (e) => {
    e.preventDefault();
    ask(el.input.value);
});

document.querySelectorAll('.chip[data-question]').forEach((chip) => {
    chip.addEventListener('click', () => {
        if (state.busy) return;
        el.input.value = chip.dataset.question;
        ask(chip.dataset.question);
    });
});

$('retry-btn').addEventListener('click', () => {
    if (state.lastQuestion) ask(state.lastQuestion);
    else checkHealth();
});

el.useDefault.addEventListener('click', () => {
    el.model.value = state.defaultModel;
    store.set(state.defaultModel);
    syncTemperatureSupport();
    ask(state.lastQuestion);
});

checkHealth().then(loadModels);
