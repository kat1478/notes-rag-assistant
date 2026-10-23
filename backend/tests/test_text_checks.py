"""Deterministic text checks used by the evaluation: language, refusals, citations, judge output parsing."""
import pytest

from app.evaluation import is_refusal_response
from app.judge import CITATION_ONLY, _parse
from app.language import detect_language
from app.run_model_benchmark import parse_citations


@pytest.mark.parametrize("text, lang", [
    ("Jak działa algorytm Dijkstry?", "pl"),
    ("Co oznacza, że heurystyka w A* jest dopuszczalna?", "pl"),
    ("What is the dying ReLU problem?", "en"),
    ("Who wrote “Pan Tadeusz”?", "en"),
    ("Algorytm Dijkstry jest podstawą [Source 1].", "pl"),
    ("Adam Mickiewicz. [Source 2]", "unknown"),
])
def test_detect_language(text, lang):
    assert detect_language(text) == lang


@pytest.mark.parametrize("text, refused", [
    ("I don't have enough information to answer that based on the notes.", True),
    ("Nie mam wystarczających informacji, aby odpowiedzieć na podstawie notatek.", True),
    ("Nie ma w tym tekście informacji o cenach biletów kinowych.", True),
    ("Brak w notatkach informacji na ten temat.", True),
    ("Algorytm nie ma heurystyki, więc sprawdza wszystko [Source 1].", False),
    ("Nie ma znaczenia, jaka jest pogoda; informacja o gradiencie jest w Source 2.", False),
])
def test_refusal_detection(text, refused):
    assert is_refusal_response(text) is refused


def test_citations_are_split_into_valid_and_invalid():
    text = "Fact one [Source 1]. Fact two [Source 2, 3]. Made up [Source 9]. Both [Sources 1 and 2]."
    assert parse_citations(text, n_sources=3) == {"valid": 5, "invalid": 1}


def test_citation_only_claims_are_recognised():
    assert CITATION_ONLY.match("[Source 2]")
    assert CITATION_ONLY.match("[Source 1] [Source 3].")
    assert not CITATION_ONLY.match("Bellman-Ford computes shortest paths [Source 1]")


def test_judge_output_parsing_tolerates_prose_around_json():
    assert _parse('{"verdict": "correct"}') == {"verdict": "correct"}
    assert _parse('Here is the grade:\n{"verdict": "incorrect"}\nThanks') == {"verdict": "incorrect"}
