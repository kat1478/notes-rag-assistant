"""EMF telemetry for /ask: valid metric format, no raw text, one event per request (no network)."""
import json

from app import main, telemetry
from app.provider import Generation

from .test_api import client  # noqa: F401 — fixture

QUESTION = "Jak działa algorytm Dijkstry?"


def _event(**overrides):
    args = dict(question=QUESTION, model_id="openrouter:anthropic/claude-haiku-5.5", retrieval_ms=40.0,
                generation_ms=900.0, n_sources=3, refused=False, generation_error=False, prompt_tokens=800,
                completion_tokens=60, cost_usd=0.0011, now_ms=1)
    return telemetry.ask_event(**{**args, **overrides})


def test_event_is_valid_emf():
    e = _event()
    spec = e["_aws"]["CloudWatchMetrics"][0]
    assert spec["Namespace"] == "NotesRag" and spec["Dimensions"] == [[]]
    names = [m["Name"] for m in spec["Metrics"]]
    # every declared metric must be a numeric top-level field, or CloudWatch drops the whole line
    assert names and all(isinstance(e[n], (int, float)) for n in names)
    assert len(telemetry.METRICS) <= 10  # free custom-metric allowance
    assert e["LatencyMs"] == 940.0 and e["Refused"] == 0


def test_missing_values_are_left_out_not_null():
    e = _event(prompt_tokens=None, completion_tokens=None, cost_usd=None)
    names = {m["Name"] for m in e["_aws"]["CloudWatchMetrics"][0]["Metrics"]}
    assert not names & {"PromptTokens", "CompletionTokens", "CostUsd"}
    assert "CostUsd" not in e


def test_question_text_is_not_logged():
    line = json.dumps(_event(), ensure_ascii=False)
    assert QUESTION not in line and "Dijkstr" not in line
    assert _event()["question_hash"] == telemetry.question_hash("  jak działa algorytm dijkstry? ")


def test_ask_emits_one_event_and_returns_cost(client, monkeypatch):  # noqa: F811
    events = []
    monkeypatch.setattr(telemetry, "emit", events.append)
    monkeypatch.setattr(main, "generate_answer_with_meta", lambda q, s, m, t: Generation(
        text="I don't have enough information to answer that based on the notes.",
        model_id="openrouter:anthropic/claude-haiku-5.5", prompt_tokens=100, completion_tokens=12, cost_usd=0.0002))
    body = client.post("/ask", json={"query": QUESTION}).json()
    assert body["metadata"]["cost_usd"] == 0.0002
    assert len(events) == 1
    e = events[0]
    assert e["Refused"] == 1 and e["GenerationError"] == 0 and e["CostUsd"] == 0.0002 and e["sources"] == 1
