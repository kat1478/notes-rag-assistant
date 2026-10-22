"""Tiny deterministic PL/EN language detector for evaluation (questions are short, answers mixed with citations).

Signals: Polish diacritics and function words vs. English function words. Good enough to tell which of the two
languages an answer is written in; not a general-purpose language identifier."""
import re

PL_CHARS = re.compile(r"[ąćęłńóśźż]", re.I)
PL_WORDS = re.compile(r"\b(i|w|z|na|nie|jest|się|to|że|do|od|jak|czy|co|dla|przez|oraz|który|która|które|jego|są|jako|tylko|gdy|czym|dlaczego|jakie)\b", re.I)
EN_WORDS = re.compile(r"\b(the|is|are|of|and|to|in|it|that|this|with|for|on|by|as|be|an|which|what|how|why|when|does|not|who|where|do|can|should|wrote|use)\b", re.I)
CITATION = re.compile(r"\[Sources?[^\]]*\]", re.I)


def detect_language(text: str) -> str:
    """Returns 'pl', 'en' or 'unknown'."""
    text = CITATION.sub(" ", text or "")
    pl = len(PL_CHARS.findall(text)) * 2 + len(PL_WORDS.findall(text))
    en = len(EN_WORDS.findall(text))
    if pl == 0 and en == 0:
        return "unknown"
    return "pl" if pl > en else "en"
