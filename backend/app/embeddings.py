import logging
from typing import List
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Global model instance
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            _model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            logger.error(f"Failed to load embedding model {EMBEDDING_MODEL}: {e}")
            raise RuntimeError(f"Embedding model load failure: {e}")
    return _model

def embed_text(text: str) -> List[float]:
    """Generates an embedding for a single text string."""
    model = get_model()
    embedding = model.encode(text)
    return embedding.tolist()

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of text strings."""
    if not texts:
        return []
    model = get_model()
    embeddings = model.encode(texts)
    return embeddings.tolist()
