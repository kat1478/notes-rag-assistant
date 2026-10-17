"""Thin facade over the provider layer (see provider.py)."""
from typing import Optional

from .provider import Generation, get_provider


def generate(prompt: str, model_id: Optional[str] = None) -> Generation:
    """Generates a completion with the selected (or default) allowlisted model."""
    return get_provider(model_id).generate(prompt)


def generate_from_ollama(prompt: str) -> str:
    """Backward-compatible entry point: default model, text only."""
    return generate(prompt).text
