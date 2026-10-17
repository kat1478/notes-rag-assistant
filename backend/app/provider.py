"""LLM providers behind one interface.

Model ids are "<provider>:<model>" (split on the first colon), e.g. "ollama:llama3.2:3b" or
"openrouter:anthropic/claude-haiku-5.5". Only allowlisted models can be resolved.
"""
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

import httpx

from .config import (
    DEFAULT_MODEL_ID,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODELS,
    REASONING,
    TEMPERATURE,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
)

logger = logging.getLogger(__name__)

# Local models on CPU can be slow with long prompts
OLLAMA_TIMEOUT = 180.0
OPENROUTER_TIMEOUT = 90.0
PROJECT_URL = "https://github.com/kat1478/notes-rag-assistant"


@dataclass
class Generation:
    text: str
    model_id: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    error: Optional[str] = None   # set when generation failed and the extractive fallback was returned
    temperature: Optional[float] = None  # value actually sent; None = the model does not accept temperature
    reasoning_tokens: Optional[int] = None  # hidden reasoning tokens (billed as output), when reported
    reasoning: Optional[str] = None  # reasoning setting actually sent, e.g. 'effort=none'


class LLMProvider:
    name = "base"

    def __init__(self, model: str):
        self.model = model

    @property
    def model_id(self) -> str:
        return f"{self.name}:{self.model}"

    @property
    def supports_temperature(self) -> Optional[bool]:
        return True

    def resolve_temperature(self, temperature: Optional[float]) -> Optional[float]:
        """Requested (or default) temperature, or None when the model would ignore it anyway."""
        t = TEMPERATURE if temperature is None else float(temperature)
        if not TEMPERATURE_MIN <= t <= TEMPERATURE_MAX:
            raise ValueError(f"temperature must be between {TEMPERATURE_MIN} and {TEMPERATURE_MAX}, got {t}")
        return None if self.supports_temperature is False else t

    def generate(self, prompt: str, temperature: Optional[float] = None) -> Generation:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    name = "ollama"

    def generate(self, prompt: str, temperature: Optional[float] = None) -> Generation:
        temperature = self.resolve_temperature(temperature)
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        logger.info(f"Sending request to Ollama ({self.model}) at {url}, timeout={OLLAMA_TIMEOUT}s")

        try:
            with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError:
            logger.error(f"Failed to connect to Ollama at {OLLAMA_BASE_URL}. Is 'ollama serve' running?")
            raise RuntimeError(
                f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                "Make sure Ollama is running ('ollama serve' in a separate terminal)."
            )
        except httpx.ReadTimeout:
            logger.error(f"Ollama request timed out after {OLLAMA_TIMEOUT}s")
            raise RuntimeError(
                f"Ollama took longer than {OLLAMA_TIMEOUT}s to respond. "
                "The model may be too slow on this hardware, or the prompt may be too long."
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error {e.response.status_code}: {e.response.text}")
            if e.response.status_code == 404:
                raise RuntimeError(
                    f"Ollama model '{self.model}' not found. Run 'ollama pull {self.model}' first."
                )
            raise RuntimeError(f"Ollama API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Unexpected Ollama error ({type(e).__name__}): {e}")
            raise RuntimeError(f"LLM communication error: {e}")

        answer = data.get("response", "")
        logger.info(f"Ollama responded successfully ({len(answer)} chars)")
        return Generation(
            text=answer,
            model_id=self.model_id,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            temperature=temperature,
        )


class OpenRouterProvider(LLMProvider):
    name = "openrouter"

    @property
    def supports_temperature(self) -> Optional[bool]:
        params = _openrouter_supported_params().get(self.model)
        return None if params is None else "temperature" in params

    def generate(self, prompt: str, temperature: Optional[float] = None) -> Generation:
        temperature = self.resolve_temperature(temperature)
        if not OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to backend/.env or the environment, "
                "or pick a local Ollama model."
            )

        url = f"{OPENROUTER_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": PROJECT_URL,
            "X-Title": "notes-rag-assistant",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        # Reasoning models (GPT-5.x, Claude 4.5+) don't accept temperature; OpenRouter would drop it silently,
        # so we don't send it and report temperature=None instead of claiming a value that wasn't applied.
        if temperature is not None:
            payload["temperature"] = temperature
        reasoning = self.reasoning_off_payload()
        if reasoning:
            payload["reasoning"] = reasoning
        logger.info(f"Sending request to OpenRouter ({self.model}), timeout={OPENROUTER_TIMEOUT}s")

        try:
            data = self._post(url, payload, headers)
        except RuntimeError as e:
            # Some models make reasoning mandatory and reject turning it off (rejected requests are not
            # billed): retry once with the lowest effort they accept. The applied setting is reported.
            if not reasoning or "400" not in str(e):
                raise
            fallback = {"effort": "minimal"} if self.model.startswith("openai/") else {"effort": "low"}
            logger.warning(f"{self.model} rejected reasoning {reasoning}: {e}; retrying with {fallback}")
            payload["reasoning"] = fallback
            data = self._post(url, payload, headers)

        text = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
        logger.info(f"OpenRouter responded successfully ({len(text)} chars)")
        return Generation(
            text=text,
            model_id=self.model_id,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            temperature=temperature,
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            reasoning=",".join(f"{k}={v}" for k, v in payload.get("reasoning", {}).items()) or None,
        )

    def reasoning_off_payload(self) -> Optional[Dict]:
        """Disables hidden reasoning for models that support it (REASONING=off, the default):
        grounded answering doesn't need it and it would dominate cost and latency.
        OpenAI-style models take effort "none"; Anthropic and others take enabled=false."""
        if REASONING != "off":
            return None
        params = _openrouter_supported_params().get(self.model) or []
        if "reasoning" not in params:
            return None
        return {"effort": "none"} if self.model.startswith("openai/") else {"enabled": False}

    def _post(self, url: str, payload: Dict, headers: Dict) -> Dict:
        try:
            with httpx.Client(timeout=OPENROUTER_TIMEOUT) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.ReadTimeout:
            raise RuntimeError(f"OpenRouter ({self.model}) took longer than {OPENROUTER_TIMEOUT}s to respond.")
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            # Never log request headers — they contain the API key.
            logger.error(f"OpenRouter HTTP error {status} for {self.model}: {e.response.text[:300]}")
            hints = {
                401: "invalid OPENROUTER_API_KEY",
                402: "out of OpenRouter credits",
                404: f"model '{self.model}' not found on OpenRouter",
                429: "rate limited — retry later",
            }
            try:
                detail = e.response.json()["error"]["message"]
            except Exception:
                detail = ""
            raise RuntimeError(f"OpenRouter error {status}: {hints.get(status, detail or 'request failed')}")
        except httpx.HTTPError as e:
            raise RuntimeError(f"Cannot reach OpenRouter at {OPENROUTER_BASE_URL}: {e}")

        if "error" in data:
            raise RuntimeError(f"OpenRouter error: {data['error'].get('message', data['error'])}")
        return data


@lru_cache(maxsize=1)
def _openrouter_supported_params() -> Dict[str, List[str]]:
    """supported_parameters per OpenRouter model (public endpoint, fetched once per process)."""
    try:
        data = httpx.get(f"{OPENROUTER_BASE_URL}/models", timeout=15).json()["data"]
    except Exception as e:
        logger.warning(f"Could not fetch OpenRouter model capabilities; temperature support unknown: {e}")
        return {}
    return {m["id"]: m.get("supported_parameters") or [] for m in data}


_PROVIDERS = {cls.name: cls for cls in (OllamaProvider, OpenRouterProvider)}


def _allowed_ids() -> List[str]:
    return [f"ollama:{OLLAMA_MODEL}"] + [f"openrouter:{name}" for name in OPENROUTER_MODELS]


def list_models() -> List[Dict]:
    """Selectable models; hosted ones are marked unavailable when no API key is configured."""
    models = []
    for model_id in _allowed_ids():
        provider = get_provider(model_id)
        models.append({
            "id": model_id,
            "provider": provider.name,
            "model": provider.model,
            "local": provider.name == "ollama",
            "available": provider.name == "ollama" or bool(OPENROUTER_API_KEY),
            "supports_temperature": provider.supports_temperature,
        })
    return models


def get_provider(model_id: Optional[str] = None) -> LLMProvider:
    """Resolves an allowlisted model id to a provider instance (default: DEFAULT_MODEL_ID)."""
    model_id = model_id or DEFAULT_MODEL_ID
    allowed = set(_allowed_ids())
    if model_id not in allowed:
        raise ValueError(f"Model '{model_id}' is not in the allowlist. Allowed: {sorted(allowed)}")
    provider_name, model = model_id.split(":", 1)
    return _PROVIDERS[provider_name](model)
