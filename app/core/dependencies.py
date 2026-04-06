"""
FastAPI dependency injectors.
"""

from functools import lru_cache

from app.core.config import settings


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the embedding model once and reuse across requests (lazy import)."""
    from sentence_transformers import SentenceTransformer  # deferred — torch is slow to import
    return SentenceTransformer(settings.embedding_model)
