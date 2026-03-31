"""
Module 4: Retriever
-------------------
Given a user query, finds the most relevant chunks from Pinecone
using dense vector search (and optional BM25 sparse re-ranking).

Flow:
    query string
        --> embed query with same model used at index time
            --> Pinecone ANN search (top_k)
                --> optional BM25 re-rank
                    --> list[RetrievedChunk]
                        --> passed to generator (Module 5)

Why hybrid retrieval:
    Dense search catches semantic similarity; BM25 catches exact keyword
    matches (e.g. specific paper IDs, author names, technical terms).
    Combining both improves precision for scientific queries.
"""

import logging
from typing import TypedDict

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from app.config import settings

logger = logging.getLogger(__name__)


# ── Data contract between retriever and generator ────────────────────────────

class RetrievedChunk(TypedDict):
    chunk_id:    str
    arxiv_id:    str
    title:       str
    authors:     list[str]
    published:   str
    text:        str
    chunk_index: int
    total_chunks: int
    score:       float   # cosine similarity (dense) or BM25 score (re-ranked)


# ── Core retrieval ────────────────────────────────────────────────────────────

def _dense_search(
    query: str,
    model: SentenceTransformer,
    index,
    top_k: int,
) -> list[RetrievedChunk]:
    """Embed query and run ANN search against Pinecone."""
    query_vector: list[float] = model.encode(query, convert_to_numpy=True).tolist()

    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    results: list[RetrievedChunk] = []
    for match in response["matches"]:
        meta = match["metadata"]
        results.append(
            RetrievedChunk(
                chunk_id     = match["id"],
                arxiv_id     = meta.get("arxiv_id", ""),
                title        = meta.get("title", ""),
                authors      = meta.get("authors", []),
                published    = meta.get("published", ""),
                text         = meta.get("text", ""),
                chunk_index  = int(meta.get("chunk_index", 0)),
                total_chunks = int(meta.get("total_chunks", 0)),
                score        = float(match["score"]),
            )
        )
    return results


def _bm25_rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Re-rank dense results with BM25 over the candidate texts.
    Returns top_k results sorted by BM25 score descending.
    """
    tokenized_corpus = [c["text"].lower().split() for c in candidates]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    reranked: list[RetrievedChunk] = []
    for score, chunk in ranked[:top_k]:
        reranked.append({**chunk, "score": float(score)})
    return reranked


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int | None = None,
    use_rerank: bool = True,
    embedding_model: str | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve the most relevant chunks for a query.

    Args:
        query:           natural language question or search string
        top_k:           number of results to return (defaults to config)
        use_rerank:      whether to apply BM25 re-ranking on dense results
        embedding_model: override config default model name

    Returns:
        List of RetrievedChunk dicts sorted by relevance score
    """
    top_k      = top_k or settings.top_k
    model_name = embedding_model or settings.embedding_model

    # fetch more candidates for re-ranking, then trim to top_k
    fetch_k = top_k * 3 if use_rerank else top_k

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    pc    = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    logger.info("Dense search | query='%s' fetch_k=%d", query, fetch_k)
    candidates = _dense_search(query, model, index, top_k=fetch_k)

    if not candidates:
        logger.warning("No results returned from Pinecone for query: '%s'", query)
        return []

    if use_rerank and len(candidates) > 1:
        logger.info("BM25 re-ranking %d candidates -> top %d", len(candidates), top_k)
        results = _bm25_rerank(query, candidates, top_k=top_k)
    else:
        results = candidates[:top_k]

    logger.info("Retrieved %d chunks for query='%s'", len(results), query)
    return results


# ── Debug entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    query   = "How does JWST detect exoplanet atmospheres?"
    results = retrieve(query, top_k=3)

    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}\n")
    for i, r in enumerate(results):
        print(f"[{i+1}] {r['title']} ({r['arxiv_id']})")
        print(f"     score={r['score']:.4f} | chunk {r['chunk_index']}/{r['total_chunks']}")
        print(f"     {r['text'][:200]}\n")
