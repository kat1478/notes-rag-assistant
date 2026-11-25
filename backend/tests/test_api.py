"""API contract with retrieval and generation mocked (no Chroma query, no model, no network)."""
import pytest
from fastapi.testclient import TestClient

from app import main, provider
from app.provider import Generation
from app.retrieval import Retrieval

from app.config import MAX_QUERY_CHARS, MAX_TOP_K

from .conftest import chunk


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(provider, "_openrouter_supported_params", lambda: {})
    monkeypatch.setattr(main, "retrieve", lambda query, top_k: Retrieval([chunk("Dijkstra relaxes edges.", "d.txt", 0, 0.4)], "semantic"))
    return TestClient(main.app)


def test_health_reports_the_default_model(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["default_model_id"] == f"{body['provider']}:{body['model']}"


def test_models_lists_only_allowlisted_ids(client):
    ids = [m["id"] for m in client.get("/models").json()]
    assert ids and all(i.startswith(("openrouter:", "ollama:")) for i in ids)


def test_ask_returns_answer_sources_and_metadata(client, monkeypatch):
    seen = {}

    def fake_generate(query, sources, model_id, temperature):
        seen.update(query=query, n_sources=len(sources), model_id=model_id)
        return Generation(text="Relaxation updates distances [Source 1].", model_id="openrouter:anthropic/claude-haiku-5.5",
                          prompt_tokens=100, completion_tokens=12, temperature=None)

    monkeypatch.setattr(main, "generate_answer_with_meta", fake_generate)
    r = client.post("/ask", json={"query": "What is relaxation?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].endswith("[Source 1].")
    assert body["sources"][0]["metadata"]["file_name"] == "d.txt"
    assert body["metadata"]["provider"] == "openrouter"
    assert body["metadata"]["prompt_tokens"] == 100
    assert body["metadata"]["retrieval_mode"] == "semantic"
    assert seen == {"query": "What is relaxation?", "n_sources": 1, "model_id": None}


def test_ask_rejects_a_model_outside_the_allowlist(client):
    r = client.post("/ask", json={"query": "q", "model": "openrouter:some/expensive-model"})
    assert r.status_code == 400


def test_ask_validates_temperature_range(client):
    assert client.post("/ask", json={"query": "q", "temperature": 3}).status_code == 422


@pytest.mark.parametrize("payload", [
    {"query": ""},
    {"query": "x" * (MAX_QUERY_CHARS + 1)},
    {"query": "q", "top_k": 0},
    {"query": "q", "top_k": MAX_TOP_K + 1},
])
def test_ask_and_search_reject_out_of_bounds_requests(client, payload):
    # Bounded before any retrieval or paid call: top_k is prompt size, so it is cost.
    assert client.post("/ask", json=payload).status_code == 422
    assert client.post("/search", json=payload).status_code == 422


def test_search_accepts_the_bounds(client):
    assert client.post("/search", json={"query": "x" * MAX_QUERY_CHARS, "top_k": MAX_TOP_K}).status_code == 200


LLAMA = "openrouter:meta-llama/llama-3.2-3b-instruct"


def _failing(error, calls, default_error=None):
    def fake(query, sources, model_id, temperature):
        calls.append(model_id)
        err = default_error if model_id == main.DEFAULT_MODEL_ID else error
        return Generation(text="LLM generation failed." if err else "ok [Source 1].", model_id=model_id, error=err)
    return fake


def test_unavailable_model_falls_back_to_the_default_and_says_so(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "generate_answer_with_meta", _failing("OpenRouter error 429: rate limited", calls))
    meta = client.post("/ask", json={"query": "q", "model": LLAMA}).json()["metadata"]
    assert calls == [LLAMA, main.DEFAULT_MODEL_ID]
    assert meta["model"] == main.DEFAULT_MODEL_ID and meta["fallback_from"] == LLAMA and meta["generation_error"] is None


@pytest.mark.parametrize("model,error,default_error", [
    (LLAMA, "OpenRouter error 401: invalid OPENROUTER_API_KEY", None),  # no other model would work either
    (LLAMA, "OpenRouter error 503: down", "OpenRouter error 503: down"),
    (main.DEFAULT_MODEL_ID, None, "OpenRouter error 429: rate limited"),
])
def test_errors_without_a_working_alternative_are_reported(client, monkeypatch, model, error, default_error):
    calls = []
    monkeypatch.setattr(main, "generate_answer_with_meta", _failing(error, calls, default_error))
    meta = client.post("/ask", json={"query": "q", "model": model}).json()["metadata"]
    assert meta["model"] == model and meta["fallback_from"] is None and meta["generation_error"]
    assert len(calls) <= 2
