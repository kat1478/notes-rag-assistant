import hashlib

import pytest

from app.answering import REFUSALS, build_prompt

# v1 produced every published benchmark result; it must stay byte-identical (CLAUDE.md boundary).
PROMPT_V1_SHA256 = "44c1792e6e29b7509d467cd40f6df8618b4a37005a7fddcc9eb1fbbe825248ed"


def test_prompt_v1_is_frozen(sources):
    prompt, _ = build_prompt("What is relaxation?", sources, "v1")
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == PROMPT_V1_SHA256


def test_context_block_numbers_sources_in_order(sources):
    _, context = build_prompt("q", sources, "v3")
    assert context.index("--- Source 1 [a.txt] ---") < context.index("--- Source 2 [b.txt] ---")


@pytest.mark.parametrize("question, language, refusal", [
    ("Jak działa algorytm Dijkstry?", "Polish", REFUSALS["pl"]),
    ("What is edge relaxation?", "English", REFUSALS["en"]),
    ("Adam Mickiewicz?", "English", REFUSALS["en"]),  # undetectable → English
])
def test_prompt_v3_states_the_question_language(sources, question, language, refusal):
    prompt, _ = build_prompt(question, sources, "v3")
    assert f"The question is in {language}." in prompt
    assert f"Write the whole answer in {language}" in prompt
    assert refusal in prompt
    assert "the question is in English." not in prompt.replace("The question is in English.", "")


def test_unknown_prompt_version_is_rejected(sources):
    with pytest.raises(ValueError):
        build_prompt("q", sources, "v9")
