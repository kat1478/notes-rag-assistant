"""Public-demo guards: uploads off and a per-client cap on /ask (no network, generation mocked)."""
import pytest
from fastapi.testclient import TestClient

from app import main, provider
from app.provider import Generation
from app.rate_limit import RateLimiter, client_id

from .conftest import chunk


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(provider, "_openrouter_supported_params", lambda: {})
    monkeypatch.setattr(main, "perform_search", lambda query, top_k: [chunk("Dijkstra relaxes edges.", "d.txt", 0, 0.4)])
    monkeypatch.setattr(main, "generate_answer_with_meta", lambda *a: Generation(
        text="ok [Source 1].", model_id="openrouter:anthropic/claude-haiku-5.5", temperature=None))
    return TestClient(main.app)


def test_uploads_are_rejected_in_demo_mode(client, monkeypatch):
    monkeypatch.setattr(main, "DEMO_MODE", True)
    files = {"file": ("n.md", b"# note", "text/markdown")}
    assert client.post("/ingest", files=files).status_code == 403
    assert client.post("/ingest/batch", files=[("files", ("n.md", b"# note", "text/markdown"))]).status_code == 403
    assert client.get("/health").json()["demo_mode"] is True


def test_ask_returns_429_after_the_limit(client, monkeypatch):
    monkeypatch.setattr(main, "ask_limiter", RateLimiter(limit=2, window_s=3600))
    headers = {"x-forwarded-for": "203.0.113.7"}
    codes = [client.post("/ask", json={"query": "q"}, headers=headers).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    # another client is counted separately
    assert client.post("/ask", json={"query": "q"}, headers={"x-forwarded-for": "198.51.100.1"}).status_code == 200


def test_window_expires_and_zero_means_unlimited():
    limiter = RateLimiter(limit=1, window_s=10)
    assert limiter.allow("a", now=0) and not limiter.allow("a", now=5)
    assert limiter.allow("a", now=10.1)
    assert all(RateLimiter(limit=0, window_s=10).allow("a") for _ in range(100))


def test_client_id_uses_the_last_forwarded_entry():
    assert client_id("1.1.1.1, 203.0.113.7", "127.0.0.1") == "203.0.113.7"
    assert client_id(None, "10.0.0.2") == "10.0.0.2"
