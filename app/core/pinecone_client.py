"""
Singleton Pinecone client and index accessor.
Import `get_index` wherever you need to talk to Pinecone.
"""

import logging

from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings

logger = logging.getLogger(__name__)

_pc: Pinecone | None = None


def get_client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
    return _pc


def get_index():
    pc = get_client()
    _ensure_index(pc)
    return pc.Index(settings.pinecone_index_name)


def _ensure_index(pc: Pinecone) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        logger.info(
            "Creating Pinecone index '%s' (dim=%d)",
            settings.pinecone_index_name, settings.embedding_dim,
        )
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings.pinecone_environment),
        )
    else:
        logger.debug("Pinecone index '%s' already exists", settings.pinecone_index_name)
