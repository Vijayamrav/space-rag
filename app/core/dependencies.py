"""
FastAPI dependency injectors.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse across requests."""
    return SentenceTransformer(settings.embedding_model)
