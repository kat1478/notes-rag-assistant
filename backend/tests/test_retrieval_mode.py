"""`retrieve` reports which path produced the context (no Chroma, no model: both searches mocked)."""
from app import retrieval

from .conftest import chunk


def _patch(monkeypatch, semantic, tfidf):
    def fake_semantic(query, top_k):
        if isinstance(semantic, Exception):
            raise semantic
        return semantic

    monkeypatch.setattr(retrieval, "search_chunks", fake_semantic)
    monkeypatch.setattr(retrieval, "perform_search_tfidf", lambda query, top_k: tfidf)
    monkeypatch.setattr(retrieval, "expand_chunks_with_neighbors", lambda chunks: chunks)


def test_semantic_results_are_reported_as_semantic(monkeypatch):
    _patch(monkeypatch, [chunk("a")], [chunk("b")])
    results, mode = retrieval.retrieve("q")
    assert mode == "semantic" and results[0].text == "a"


def test_empty_or_failing_semantic_search_falls_back_to_tfidf(monkeypatch):
    for semantic in ([], RuntimeError("chroma down")):
        _patch(monkeypatch, semantic, [chunk("b")])
        results, mode = retrieval.retrieve("q")
        assert mode == "tfidf_fallback" and results[0].text == "b"


def test_nothing_found_is_reported_as_none(monkeypatch):
    _patch(monkeypatch, [], [])
    assert retrieval.retrieve("q") == retrieval.Retrieval([], "none")
    assert retrieval.perform_search("q") == []
