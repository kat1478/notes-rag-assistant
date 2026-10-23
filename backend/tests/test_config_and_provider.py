import subprocess
import sys

import pytest

from app import provider


def _config_value(env: dict, expr: str) -> str:
    """Evaluates a config expression in a fresh interpreter (config is read once at import time)."""
    out = subprocess.run([sys.executable, "-c", f"from app import config; print({expr})"],
                         env={**__import__("os").environ, **env}, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_each_embedding_model_gets_its_own_collections():
    assert _config_value({"EMBEDDING_MODEL_KEY": "minilm-l6"}, "config.CORPORA['v1'][1]") == "notes_chunks"
    assert (_config_value({"EMBEDDING_MODEL_KEY": "multilingual-minilm-l12"}, "config.CORPORA['v2'][1]")
            == "notes_v2__multilingual-minilm-l12")


def test_unknown_embedding_model_key_fails_fast():
    with pytest.raises(subprocess.CalledProcessError):
        _config_value({"EMBEDDING_MODEL_KEY": "nope"}, "config.EMBEDDING_MODEL")


def test_model_outside_the_allowlist_is_rejected():
    with pytest.raises(ValueError):
        provider.get_provider("openrouter:some/expensive-model")


def test_temperature_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        provider.OllamaProvider("llama3.2:3b").resolve_temperature(1.5)


def test_models_that_ignore_temperature_get_none(monkeypatch):
    monkeypatch.setattr(provider, "_openrouter_supported_params",
                        lambda: {"anthropic/claude-haiku-5.5": ["reasoning"], "google/gemma-4-31b-it": ["temperature"]})
    assert provider.OpenRouterProvider("anthropic/claude-haiku-5.5").resolve_temperature(0.7) is None
    assert provider.OpenRouterProvider("google/gemma-4-31b-it").resolve_temperature(0.7) == 0.7


def test_reasoning_off_payload_per_vendor(monkeypatch):
    monkeypatch.setattr(provider, "_openrouter_supported_params",
                        lambda: {"openai/gpt-5.6-sol": ["reasoning"], "anthropic/claude-haiku-5.5": ["reasoning"],
                                 "google/gemma-4-31b-it": ["temperature"]})
    assert provider.OpenRouterProvider("openai/gpt-5.6-sol").reasoning_off_payload() == {"effort": "none"}
    assert provider.OpenRouterProvider("anthropic/claude-haiku-5.5").reasoning_off_payload() == {"enabled": False}
    assert provider.OpenRouterProvider("google/gemma-4-31b-it").reasoning_off_payload() is None
