"""
Module 3: Embedder
------------------
Converts ChunkRecords into dense vector embeddings and upserts them
into a Pinecone index for later retrieval.

Flow:
    list[ChunkRecord]
        --> sentence-transformers encode (batched)
            --> build Pinecone vectors with metadata
                --> upsert to Pinecone index
                    --> list[EmbeddingRecord]  (passed to retriever, Module 4)

Why sentence-transformers + Pinecone:
    all-MiniLM-L6-v2 is fast, lightweight (384-dim), and strong on
    semantic similarity tasks — ideal for RAG retrieval over scientific text.
    Pinecone handles ANN search at scale without managing an index server.
"""

import logging
from typing import TypedDict

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from app.chunker import ChunkRecord
from app.config import settings

logger = logging.getLogger(__name__)


# ── Data contract between embedder and retriever ─────────────────────────────

class EmbeddingRecord(TypedDict):
    chunk_id:    str
    arxiv_id:    str
    title:       str
    authors:     list[str]
    published:   str
    text:        str
    chunk_index: int
    total_chunks: int
    embedding:   list[float]   # 384-dim vector


# ── Pinecone helpers ──────────────────────────────────────────────────────────

def _get_or_create_index(pc: Pinecone, index_name: str, dim: int) -> None:
    """Create the Pinecone index if it doesn't already exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        logger.info("Creating Pinecone index '%s' (dim=%d)", index_name, dim)
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings.pinecone_environment),
        )
    else:
        logger.debug("Pinecone index '%s' already exists", index_name)


def _build_vectors(chunks: list[ChunkRecord], embeddings: list[list[float]]) -> list[dict]:
    """Pair each chunk with its embedding and metadata for Pinecone upsert."""
    vectors = []
    for chunk, vector in zip(chunks, embeddings):
        vectors.append({
            "id": chunk["chunk_id"],
            "values": vector,
            "metadata": {
                "arxiv_id":    chunk["arxiv_id"],
                "title":       chunk["title"],
                "authors":     chunk["authors"],
                "published":   chunk["published"],
                "text":        chunk["text"],          # stored for retrieval context
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
            },
        })
    return vectors


# ── Public API ────────────────────────────────────────────────────────────────

def embed_and_store(
    chunks: list[ChunkRecord],
    batch_size: int = 64,
    embedding_model: str | None = None,
) -> list[EmbeddingRecord]:
    """
    Embed chunks and upsert them into Pinecone.

    Args:
        chunks:          output from chunker.chunk_papers()
        batch_size:      how many chunks to encode + upsert at once
        embedding_model: override config default model name

    Returns:
        List of EmbeddingRecord dicts (chunks + their embedding vectors)
    """
    model_name = embedding_model or settings.embedding_model

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    pc    = Pinecone(api_key=settings.pinecone_api_key)
    _get_or_create_index(pc, settings.pinecone_index_name, settings.embedding_dim)
    index = pc.Index(settings.pinecone_index_name)

    all_records: list[EmbeddingRecord] = []

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [c["text"] for c in batch]

        logger.info(
            "Encoding batch %d-%d / %d",
            batch_start, batch_start + len(batch) - 1, len(chunks),
        )
        embeddings: list[list[float]] = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        vectors = _build_vectors(batch, embeddings)
        index.upsert(vectors=vectors)
        logger.debug("Upserted %d vectors to Pinecone", len(vectors))

        for chunk, vector in zip(batch, embeddings):
            all_records.append(
                EmbeddingRecord(
                    chunk_id     = chunk["chunk_id"],
                    arxiv_id     = chunk["arxiv_id"],
                    title        = chunk["title"],
                    authors      = chunk["authors"],
                    published    = chunk["published"],
                    text         = chunk["text"],
                    chunk_index  = chunk["chunk_index"],
                    total_chunks = chunk["total_chunks"],
                    embedding    = vector,
                )
            )

    logger.info(
        "Embedding complete | %d records upserted to index '%s'",
        len(all_records), settings.pinecone_index_name,
    )
    return all_records


# ── Debug entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from app.ingestion import fetch_papers
    from app.chunker import chunk_papers

    papers  = fetch_papers("james webb telescope exoplanet", max_results=1)
    chunks  = chunk_papers(papers)
    records = embed_and_store(chunks)

    print(f"\nTotal embedded: {len(records)}")
    print(f"Sample chunk_id : {records[0]['chunk_id']}")
    print(f"Embedding dim   : {len(records[0]['embedding'])}")
    print(f"Text preview    : {records[0]['text'][:200]}")
