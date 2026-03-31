"""
Retrieval Service
-----------------
Hybrid search: dense ANN (Pinecone) + BM25 re-ranking.
"""

import logging
from typing import TypedDict

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.dependencies import get_embedding_model
from app.core.pinecone_client import get_index

logger = logging.getLogger(__name__)


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


def _dense_search(query: str, model: SentenceTransformer, top_k: int) -> list[RetrievedChunk]:
    index        = get_index()
    query_vector = model.encode(query, convert_to_numpy=True).tolist()
    response     = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

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


def _bm25_rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    tokenized_corpus = [c["text"].lower().split() for c in candidates]
    bm25             = BM25Okapi(tokenized_corpus)
    scores           = bm25.get_scores(query.lower().split())

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [{**chunk, "score": float(score)} for score, chunk in ranked[:top_k]]


def retrieve(query: str, top_k: int | None = None, use_rerank: bool = True) -> list[RetrievedChunk]:
    top_k   = top_k or settings.top_k
    fetch_k = top_k * 3 if use_rerank else top_k
    model   = get_embedding_model()

    logger.info("Dense search | query='%s' fetch_k=%d", query, fetch_k)
    candidates = _dense_search(query, model, top_k=fetch_k)

    if not candidates:
        logger.warning("No results from Pinecone for query: '%s'", query)
        return []

    if use_rerank and len(candidates) > 1:
        logger.info("BM25 re-ranking %d candidates -> top %d", len(candidates), top_k)
        return _bm25_rerank(query, candidates, top_k=top_k)

    return candidates[:top_k]
