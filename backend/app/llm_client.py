import httpx
import logging
from .config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

def generate_from_ollama(prompt: str) -> str:
    """
    Sends a prompt to the local Ollama API.
    Handles connection errors and missing models gracefully.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "")
            
    except httpx.ConnectError:
        logger.error(f"Failed to connect to Ollama at {OLLAMA_BASE_URL}")
        raise RuntimeError("Ollama is not running or unreachable.")
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama API returned an error: {e.response.text}")
        if e.response.status_code == 404:
            raise RuntimeError(f"Ollama model '{OLLAMA_MODEL}' not found. Please pull it first.")
        raise RuntimeError(f"Ollama API error: {e.response.status_code}")
    except Exception as e:
        logger.error(f"Unexpected error communicating with Ollama: {e}")
        raise RuntimeError(f"LLM communication error: {e}")
