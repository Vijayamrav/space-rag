"""
Retrieval Service
-----------------
Hybrid search: dense ANN + sparse BM25 via Pinecone native hybrid query.
alpha=1.0 -> pure dense, alpha=0.0 -> pure sparse, 0.75 -> dense-leaning hybrid.
"""

import logging
from typing import TypedDict

from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.dependencies import get_embedding_model
from app.core.pinecone_client import get_index

logger = logging.getLogger(__name__)

# Singleton BM25 encoder for query-time sparse encoding
_bm25_encoder: BM25Encoder | None = None


def _get_bm25_encoder() -> BM25Encoder:
    global _bm25_encoder
    if _bm25_encoder is None:
        # Use default pre-fitted encoder for query encoding
        _bm25_encoder = BM25Encoder().default()
    return _bm25_encoder


class RetrievedChunk(TypedDict):
    chunk_id:     str
    arxiv_id:     str
    title:        str
    authors:      list[str]
    published:    str
    text:         str
    chunk_index:  int
    total_chunks: int
    score:        float


def _hybrid_search(
    query: str,
    model: SentenceTransformer,
    top_k: int,
    alpha: float = 0.75,
) -> list[RetrievedChunk]:
    """
    Query Pinecone with both dense and sparse vectors.
    alpha controls the blend: 1.0 = pure dense, 0.0 = pure sparse.
    """
    index        = get_index()
    dense_vector = model.encode(query, convert_to_numpy=True).tolist()
    sparse_vector = _get_bm25_encoder().encode_queries(query)

    # Scale vectors by alpha for weighted hybrid
    scaled_dense  = [v * alpha for v in dense_vector]
    scaled_sparse = {
        "indices": sparse_vector["indices"],
        "values":  [v * (1 - alpha) for v in sparse_vector["values"]],
    }

    response = index.query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True,
    )

    results = []
    for match in response["matches"]:
        meta = match["metadata"]
        results.append(RetrievedChunk(
            chunk_id     = match["id"],
            arxiv_id     = meta.get("arxiv_id", ""),
            title        = meta.get("title", ""),
            authors      = meta.get("authors", []),
            published    = meta.get("published", ""),
            text         = meta.get("text", ""),
            chunk_index  = int(meta.get("chunk_index", 0)),
            total_chunks = int(meta.get("total_chunks", 0)),
            score        = float(match["score"]),
        ))
    return results


def retrieve(
    query: str,
    top_k: int | None = None,
    alpha: float = 0.75,
) -> list[RetrievedChunk]:
    """
    Hybrid retrieval using Pinecone native sparse-dense search.

    Args:
        query:  natural language question
        top_k:  number of chunks to return
        alpha:  blend weight — 0.75 means 75% dense, 25% sparse (BM25)
    """
    top_k = top_k or settings.top_k
    model = get_embedding_model()

    logger.info("Hybrid search | query='%s' top_k=%d alpha=%.2f", query, top_k, alpha)
    results = _hybrid_search(query, model, top_k=top_k, alpha=alpha)

    if not results:
        logger.warning("No results from Pinecone for query: '%s'", query)

    return results
