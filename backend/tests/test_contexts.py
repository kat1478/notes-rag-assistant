"""Stored contexts: the judge grades the chunks the generator saw, not a later re-retrieval (offline, judge mocked)."""
import csv
import json
import sys

import pytest

from app import run_judge
from app.contexts import context_hash, load_store, save_contexts

from .conftest import chunk


def test_hash_is_stable_and_order_sensitive():
    a, b = chunk("first", "a.txt", 0), chunk("second", "b.txt", 1)
    assert context_hash([a, b]) == context_hash([chunk("first", "a.txt", 0), chunk("second", "b.txt", 1)])
    # [Source N] numbers follow the order, so a reordered context is a different context.
    assert context_hash([a, b]) != context_hash([b, a])


def test_store_round_trip_and_dedup(tmp_path):
    contexts = {"q1": [chunk("x", "a.txt", 0)], "q2": [chunk("x", "a.txt", 0)], "q3": [chunk("y", "b.txt", 2, 0.5)]}
    hashes = save_contexts(tmp_path, contexts)
    save_contexts(tmp_path, contexts)  # a second run with the same contexts adds nothing
    assert len((tmp_path / "contexts.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert hashes["q1"] == hashes["q2"]
    assert load_store(tmp_path)[hashes["q3"]] == contexts["q3"]


def test_cache_key_includes_the_context_but_legacy_keys_are_unchanged():
    args = ("openrouter:x/y", "q1", "answer [Source 1].", ["point"])
    assert run_judge.cache_key(*args) == run_judge.cache_key(*args, None)
    assert run_judge.cache_key(*args, "aaaa") != run_judge.cache_key(*args, "bbbb")


def _results_dir(tmp_path, row_extra, run_extra):
    model = "openrouter:x/y"
    (tmp_path / "summary.json").write_text(json.dumps({"runs": {model: {"model_id": model, "top_k": 5, **run_extra}}}))
    row = {"question_id": "q1", "refused": "False", "failed": "False", "retrieval_hit": "True",
           "answer_text": "Dijkstra relaxes edges [Source 1].", **row_extra}
    with open(tmp_path / "openrouter_x_y.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    dataset = tmp_path / "questions.json"
    dataset.write_text(json.dumps([{"id": "q1", "question": "What does Dijkstra do?", "category": "c",
                                    "should_answer": True, "expected_key_points": ["relaxes edges"]}]))
    return dataset


@pytest.fixture
def judge_env(tmp_path, monkeypatch):
    seen = {}

    def fake_judge(question, key_points, sources, answer):
        seen["sources"] = sources
        return {"verdict": "correct", "key_point_recall": 1.0, "faithful": True, "unsupported_claims": [],
                "reason": "", "prompt_tokens": 10, "completion_tokens": 5}

    monkeypatch.setattr(run_judge, "CACHE", tmp_path / "cache.jsonl")
    monkeypatch.setattr(run_judge, "judge_answer", fake_judge)
    monkeypatch.setattr(run_judge, "fetch_openrouter_prices", lambda: {run_judge.JUDGE_MODEL: {"prompt": 0, "completion": 0}})
    monkeypatch.setattr(run_judge, "use_collection", lambda name: None)
    monkeypatch.setattr(run_judge, "slug", lambda key: key.replace(":", "_").replace("/", "_"))
    return seen


def _run(monkeypatch, results_dir, dataset):
    monkeypatch.setattr(sys, "argv", ["run_judge", "--results-dir", str(results_dir), "--dataset-path", str(dataset)])
    run_judge.main()


def test_judge_uses_the_stored_context_not_a_re_retrieval(tmp_path, monkeypatch, judge_env):
    stored = [chunk("Dijkstra relaxes edges.", "d.txt", 0, 0.4)]
    h = save_contexts(tmp_path, {"q1": stored})["q1"]
    dataset = _results_dir(tmp_path, {"context_hash": h}, {})
    monkeypatch.setattr(run_judge, "perform_search", lambda *a, **k: pytest.fail("must not re-retrieve"))
    _run(monkeypatch, tmp_path, dataset)
    assert judge_env["sources"] == stored


def test_legacy_rows_re_retrieve_with_the_run_top_k_under_its_embeddings(tmp_path, monkeypatch, judge_env):
    calls = []
    monkeypatch.setattr(run_judge, "perform_search", lambda q, top_k: calls.append(top_k) or [chunk("c", "d.txt")])
    dataset = _results_dir(tmp_path, {}, {"embedding_model": run_judge.EMBEDDING_MODEL})
    _run(monkeypatch, tmp_path, dataset)
    assert calls == [5]


def test_legacy_rows_under_other_embeddings_are_refused(tmp_path, monkeypatch, judge_env):
    monkeypatch.setattr(run_judge, "perform_search", lambda *a, **k: pytest.fail("must not re-retrieve"))
    dataset = _results_dir(tmp_path, {}, {"embedding_model": "some/other-model"})
    with pytest.raises(SystemExit, match="no stored contexts"):
        _run(monkeypatch, tmp_path, dataset)
