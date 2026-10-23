"""API contract with retrieval and generation mocked (no Chroma query, no model, no network)."""
import pytest
from fastapi.testclient import TestClient

from app import main, provider
from app.provider import Generation

from .conftest import chunk


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(provider, "_openrouter_supported_params", lambda: {})
    monkeypatch.setattr(main, "perform_search", lambda query, top_k: [chunk("Dijkstra relaxes edges.", "d.txt", 0, 0.4)])
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
    assert seen == {"query": "What is relaxation?", "n_sources": 1, "model_id": None}


def test_ask_rejects_a_model_outside_the_allowlist(client):
    r = client.post("/ask", json={"query": "q", "model": "openrouter:some/expensive-model"})
    assert r.status_code == 400


def test_ask_validates_temperature_range(client):
    assert client.post("/ask", json={"query": "q", "temperature": 3}).status_code == 422
