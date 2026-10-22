"""LLM-as-judge: grades one answer for correctness (expected key points) and faithfulness (no claims
outside the retrieved context). Used by run_judge.py on saved benchmark answers."""
import json
import logging
import re
from typing import Dict, List

from .answering import build_prompt
from .config import JUDGE_MODEL
from .provider import Generation, OpenRouterProvider
from .schemas import ChunkResponse

logger = logging.getLogger(__name__)

VERDICTS = ("correct", "partially_correct", "incorrect")
COVERAGE = {"yes": 1.0, "partial": 0.5, "no": 0.0}

JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_answer_grade",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["key_points", "unsupported_claims", "verdict", "reason"],
            "properties": {
                "key_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["point", "covered"],
                        "properties": {
                            "point": {"type": "string"},
                            "covered": {"type": "string", "enum": list(COVERAGE)},
                        },
                    },
                },
                "unsupported_claims": {"type": "array", "items": {"type": "string"}},
                "verdict": {"type": "string", "enum": list(VERDICTS)},
                "reason": {"type": "string"},
            },
        },
    },
}

JUDGE_PROMPT = """You are grading an answer produced by a retrieval-augmented assistant over study notes.
Use ONLY the material below. The notes are in Polish and English; answers may be in either language —
paraphrases and translations count as conveying the same content.

1. Key points: for EACH expected key point, decide whether the answer conveys it: "yes", "partial" (only part
   of it, or vaguely), or "no".
2. Unsupported claims: list every factual claim in the answer that the CONTEXT does not support. A claim that
   is true in general but absent from the context IS unsupported. Ignore citation markers like [Source 1],
   restating the question, and statements that the notes lack information.
3. Verdict: "correct" if every key point is "yes"; "incorrect" if no key point is at least "partial" or the
   answer contradicts a key point; otherwise "partially_correct".
4. Reason: one short sentence.

QUESTION:
{question}

EXPECTED KEY POINTS:
{key_points}

CONTEXT GIVEN TO THE ASSISTANT:
{context}

ANSWER TO GRADE:
{answer}
"""


def _parse(text: str) -> Dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)  # tolerate prose around the JSON
        if not match:
            raise
        return json.loads(match.group(0))


def judge_answer(question: str, key_points: List[str], sources: List[ChunkResponse], answer: str) -> Dict:
    """Returns the judge's grade plus derived metrics and token usage. Raises RuntimeError on API/parse failure."""
    _, context = build_prompt(question, sources) if sources else (None, "(no context was retrieved)")
    prompt = JUDGE_PROMPT.format(
        question=question,
        key_points="\n".join(f"- {kp}" for kp in key_points),
        context=context,
        answer=answer,
    )
    gen: Generation = OpenRouterProvider(JUDGE_MODEL).generate(prompt, 0.0, response_format=JUDGE_SCHEMA)
    try:
        grade = _parse(gen.text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Judge returned non-JSON output ({e}): {gen.text[:200]!r}")
    if grade.get("verdict") not in VERDICTS:
        raise RuntimeError(f"Judge returned an invalid verdict: {grade.get('verdict')!r}")

    covered = [COVERAGE.get(kp.get("covered"), 0.0) for kp in grade.get("key_points", [])]
    grade["key_point_recall"] = round(sum(covered) / len(covered), 4) if covered else None
    grade["faithful"] = not grade.get("unsupported_claims")
    grade["prompt_tokens"] = gen.prompt_tokens
    grade["completion_tokens"] = gen.completion_tokens
    return grade
