"""LLM refusal classifier: cache, parsing, regex fallback (provider mocked — no network)."""
import pytest

from app import refusal
from app.provider import Generation


@pytest.fixture
def calls(monkeypatch, tmp_path):
    """Empty throwaway cache and a fake provider that records prompts and answers with `reply`."""
    monkeypatch.setattr(refusal, "CACHE", tmp_path / "cache.jsonl")
    monkeypatch.setattr(refusal, "_cache", None)
    monkeypatch.setattr(refusal, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(refusal, "REFUSAL_DETECTOR", "llm")
    seen = {"prompts": [], "reply": '{"label": "refusal"}'}

    class FakeProvider:
        def __init__(self, model):
            seen["model"] = model

        def generate(self, prompt, temperature, response_format=None):
            seen["prompts"].append(prompt)
            assert temperature == 0.0 and response_format == refusal.SCHEMA
            return Generation(text=seen["reply"], model_id=f"openrouter:{seen['model']}", prompt_tokens=200, completion_tokens=5)

    monkeypatch.setattr(refusal, "OpenRouterProvider", FakeProvider)
    return seen


def test_classifies_and_caches(calls):
    q, a = "Who invented A*?", "The notes do not state who invented A*.\nThey describe how it works."
    assert refusal.classify(q, a) == ("refusal", 200, 5)
    # same answer again, with the newline flattened as in saved CSVs → cache hit, no second call, no tokens
    assert refusal.classify(q, a.replace("\n", " ")) == ("refusal", None, None)
    assert len(calls["prompts"]) == 1
    assert "\n" not in calls["prompts"][0].split("ANSWER: ")[1].strip()
    # a fresh process reads the label back from the file
    refusal._cache = None
    assert refusal.classify(q, a)[0] == "refusal" and len(calls["prompts"]) == 1


def test_cache_key_depends_on_prompt(monkeypatch):
    k = refusal.cache_key("q", "a")
    monkeypatch.setattr(refusal, "PROMPT_HASH", "changed")
    assert refusal.cache_key("q", "a") != k


@pytest.mark.parametrize("reply", ["not json", '{"label": "maybe"}', '{"other": 1}'])
def test_bad_output_raises_and_is_not_cached(calls, reply):
    calls["reply"] = reply
    with pytest.raises(RuntimeError):
        refusal.classify("q", "a")
    assert not refusal.CACHE.exists()


def test_falls_back_to_regex_without_a_key(calls, monkeypatch):
    monkeypatch.setattr(refusal, "OPENROUTER_API_KEY", "")
    assert refusal.detector() == "regex"
    assert refusal.classify("q", "I don't have enough information to answer that based on the notes.")[0] == "refusal"
    # the soft refusal the regex misses — the reason for the LLM detector
    assert refusal.classify("q", "Bellman-Ford nie jest opisany w kontekście.")[0] == "answer"
    assert calls["prompts"] == []


def _row(qid, should_answer, refusal_class, hit=True):
    refused = refusal_class == "refusal"
    return {"question_id": qid, "category": "c", "should_answer": should_answer, "retrieval_hit": hit,
            "answered": not refused, "refused": refused, "refusal_class": refusal_class, "failed": False,
            "suspended": False, "latency_ms": 1000.0, "cost_usd": 0.0, "prompt_tokens": 10, "completion_tokens": 5,
            "reasoning": "", "reasoning_tokens": None, "language_match": True, "valid_citations": 1,
            "invalid_citations": 0}


def test_partial_counts_as_answered_and_is_reported():
    from app.run_model_benchmark import summarize
    rows = [_row("a", True, "answer"), _row("b", True, "partial"), _row("c", True, "refusal", hit=False),
            _row("d", False, "refusal")]
    s = summarize("openrouter:x/y", rows)
    assert s["partial_rate_in_scope"] == round(1 / 3, 4)
    assert s["answer_rate_supported"] == 1.0          # answer + partial, both with the note retrieved
    assert s["refusal_rate_unsupported"] == 1.0
    assert s["refusal_rate_out_of_scope"] == 1.0


def test_rescore_refuses_to_switch_detector_in_place(tmp_path, monkeypatch):
    import json
    from app import run_model_benchmark as bench
    (tmp_path / "summary.json").write_text(json.dumps({"runs": {"openrouter:x/y": {"model_id": "openrouter:x/y"}}}))
    monkeypatch.setattr(refusal, "detector", lambda: "llm")
    with pytest.raises(SystemExit, match="another refusal detector"):
        bench.rescore(tmp_path, tmp_path / "questions.json")
