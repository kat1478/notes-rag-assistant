from typing import List, Optional, Tuple
from .schemas import ChunkResponse
from .config import DEFAULT_MODEL_ID, PROMPT_VERSION
from .language import detect_language
from .llm_client import generate
from .provider import Generation
import logging

logger = logging.getLogger(__name__)

NO_SOURCES_ANSWER = "Nie znaleziono w notatkach informacji odpowiadających na Twoje pytanie."

# Grounding prompt versions. v1 (Stages 1–5A) is kept byte-identical so earlier results stay reproducible.
# v2 (2026-10-22) drops v1's false claim that the question is always in English (Haiku 5.5 obeyed it and answered
# Polish questions in English), asks for the question's language explicitly and fixes the refusal sentence in
# both languages (models were improvising Polish refusals the scorer could not recognise).
PROMPT_V2 = """You are an assistant answering questions based only on the provided context notes.
The context notes may be written in Polish or English, and the question may be in either language.

Follow these rules strictly:
1. Write the whole answer in the language of the question: an English question gets an English answer, a Polish question gets a Polish answer, even when the notes are in the other language.
2. Answer using ONLY facts directly mentioned in the context. Translate technical terms appropriately across languages.
3. Rely on the context even if it is partial, as long as it contains the core answer.
4. If you are answering the question, you MUST include at least one bracketed source citation (e.g. "[Source 1]", "[Source 2]") at the end of your sentence or paragraph.
5. If the context does not contain any information relevant to the question at all, reply with exactly one sentence in the language of the question. English: "I don't have enough information to answer that based on the notes." Polish: "Nie mam wystarczających informacji, aby odpowiedzieć na podstawie notatek." Do NOT append this refusal message if you have already answered the question.

Context:
{context_block}

Question:
{query}

Answer:"""
# v3 (2026-10-22): v2 fixed Polish questions but pulled Haiku into Polish on 5 English questions (the bilingual
# wording primes the notes' language). v3 detects the question language in code and states it — the real fix for
# v1's hard-coded "the question is in English" — with a single refusal sentence in that language.
PROMPT_V3 = """You are an assistant answering questions based only on the provided context notes.
The context notes may be written in Polish or English. The question is in {language}.

Follow these rules strictly:
1. Write the whole answer in {language}, even when the notes are in another language.
2. Answer using ONLY facts directly mentioned in the context. Translate technical terms appropriately across languages.
3. Rely on the context even if it is partial, as long as it contains the core answer.
4. If you are answering the question, you MUST include at least one bracketed source citation (e.g. "[Source 1]", "[Source 2]") at the end of your sentence or paragraph.
5. If the context does not contain any information relevant to the question at all, reply with exactly: "{refusal}" Do NOT append this refusal message if you have already answered the question.

Context:
{context_block}

Question:
{query}

Answer:"""
LANGUAGES = {"en": "English", "pl": "Polish"}
REFUSALS = {"en": "I don't have enough information to answer that based on the notes.",
            "pl": "Nie mam wystarczających informacji, aby odpowiedzieć na podstawie notatek."}
PROMPT_VERSIONS = ("v1", "v2", "v3")
FALLBACK_PREFIX = "LLM generation failed."


def build_prompt(query: str, sources: List[ChunkResponse], version: Optional[str] = None) -> Tuple[str, str]:
    """Builds the grounding prompt (version: v1 | v2, default PROMPT_VERSION). Returns (prompt, context_block)."""
    # Construct context from sources
    context_parts = []
    for i, source in enumerate(sources, 1):
        file_name = source.metadata.file_name
        context_parts.append(f"--- Source {i} [{file_name}] ---")
        context_parts.append(source.text.strip())
        context_parts.append("")
        
    context_block = "\n".join(context_parts)
    
    prompt = f"""You are an assistant answering questions based only on the provided context notes.
Note that the context notes are written in Polish, and the question is in English.

Follow these rules strictly:
1. Answer the question in the same language as the question.
2. Answer using ONLY facts directly mentioned in the context. Translate technical terms appropriately across languages.
3. Rely on the context even if it is partial, as long as it contains the core answer.
4. If you are answering the question, you MUST include at least one bracketed source citation (e.g. "[Source 1]", "[Source 2]") at the end of your sentence or paragraph.
5. If the context does not contain any information relevant to the question at all, say: "I don't have enough information to answer that based on the notes." Do NOT append this refusal message if you have already answered the question.

Context:
{context_block}

Question:
{query}

Answer:"""
    version = version or PROMPT_VERSION
    if version == "v2":
        prompt = PROMPT_V2.format(context_block=context_block, query=query)
    elif version == "v3":
        lang = detect_language(query)
        lang = lang if lang in LANGUAGES else "en"  # undetectable (e.g. a bare name) → English
        prompt = PROMPT_V3.format(context_block=context_block, query=query,
                                  language=LANGUAGES[lang], refusal=REFUSALS[lang])
    elif version != "v1":
        raise ValueError(f"Unknown prompt version {version!r}; use one of {PROMPT_VERSIONS}")
    return prompt, context_block


def generate_answer_with_meta(
    query: str, sources: List[ChunkResponse], model_id: Optional[str] = None,
    temperature: Optional[float] = None, prompt_version: Optional[str] = None,
) -> Generation:
    """Generates a grounded answer with the selected model; keeps token usage for cost tracking.
    Raises ValueError for a model id outside the allowlist or a temperature out of range."""
    model_id = model_id or DEFAULT_MODEL_ID
    if not sources:
        return Generation(text=NO_SOURCES_ANSWER, model_id=model_id)

    prompt, context_block = build_prompt(query, sources, prompt_version)
    try:
        return generate(prompt, model_id, temperature)
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to generate answer from LLM ({model_id}): {e}")
        # Fallback to the Stage 1 extractive behavior if LLM fails
        return Generation(
            text=f"{FALLBACK_PREFIX} Reason: {e}\nReturning raw context:\n\n" + context_block,
            model_id=model_id,
            error=str(e),
        )


def generate_answer(query: str, sources: List[ChunkResponse], model_id: Optional[str] = None) -> str:
    """Generates a grounded answer based solely on the sources."""
    return generate_answer_with_meta(query, sources, model_id).text
