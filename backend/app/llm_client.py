import httpx
import logging
from .config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

# Local models on CPU can be slow with long prompts
OLLAMA_TIMEOUT = 180.0

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
    
    logger.info(f"Sending request to Ollama ({OLLAMA_MODEL}) at {url}, timeout={OLLAMA_TIMEOUT}s")
    
    try:
        with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            answer = data.get("response", "")
            logger.info(f"Ollama responded successfully ({len(answer)} chars)")
            return answer
            
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
                f"Ollama model '{OLLAMA_MODEL}' not found. "
                f"Run 'ollama pull {OLLAMA_MODEL}' first."
            )
        raise RuntimeError(f"Ollama API error: {e.response.status_code}")
    except Exception as e:
        logger.error(f"Unexpected Ollama error ({type(e).__name__}): {e}")
        raise RuntimeError(f"LLM communication error: {e}")
