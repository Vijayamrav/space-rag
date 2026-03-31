"""
Ingestion Service
-----------------
Orchestrates the full ingest pipeline:
    multi-source fetch + rank --> PDF download --> text extraction --> chunking --> embedding --> Pinecone upsert
"""

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict

import fitz
import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.pinecone_client import get_index
from app.services.sources.aggregator import fetch_and_rank
from app.services.sources.base import PaperRecord

logger = logging.getLogger(__name__)


class ChunkRecord(TypedDict):
    chunk_id:         str
    arxiv_id:         str
    title:            str
    authors:          list[str]
    published:        str
    text:             str
    chunk_index:      int
    total_chunks:     int
    source:           str
    citation_count:   int
    concept_tags:     list[str]


# ── Chunking ──────────────────────────────────────────────────────────────────

def _download_pdf(pdf_url: str, retries: int = 3) -> bytes:
    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                r = client.get(pdf_url)
                r.raise_for_status()
                return r.content
        except Exception as exc:
            logger.warning("PDF download attempt %d failed: %s", attempt, exc)
            if attempt < retries:
                time.sleep(2.0)
    raise RuntimeError(f"Failed to download PDF after {retries} attempts: {pdf_url}")


def _extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    text = "\n".join(page.get_text("text") for page in doc).strip()
    doc.close()
    return text


def _split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


def _process_paper(paper: PaperRecord) -> list[ChunkRecord]:
    """Download, extract and chunk a single paper. Runs in a thread."""
    pdf_bytes  = _download_pdf(paper["pdf_url"])
    text       = _extract_text(pdf_bytes)
    if not text:
        logger.warning("Empty text | arxiv_id=%s — skipping", paper["arxiv_id"])
        return []
    raw_chunks = _split_text(text)
    total      = len(raw_chunks)
    logger.info("Produced %d chunks | arxiv_id=%s", total, paper["arxiv_id"])
    return [
        ChunkRecord(
            chunk_id       = f"{paper['arxiv_id']}::chunk::{idx}",
            arxiv_id       = paper["arxiv_id"],
            title          = paper["title"],
            authors        = paper["authors"],
            published      = paper["published"],
            text           = chunk_text,
            chunk_index    = idx,
            total_chunks   = total,
            source         = paper.get("source", "arxiv"),
            citation_count = paper.get("citation_count", 0),
            concept_tags   = paper.get("concept_tags", []),
        )
        for idx, chunk_text in enumerate(raw_chunks)
    ]


def chunk_papers(papers: list[PaperRecord], max_workers: int = 8) -> list[ChunkRecord]:
    all_chunks: list[ChunkRecord] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_paper, paper): paper["arxiv_id"] for paper in papers}
        for future in as_completed(futures):
            arxiv_id = futures[future]
            try:
                all_chunks.extend(future.result())
            except Exception as exc:
                logger.error("Failed to chunk %s: %s", arxiv_id, exc)

    if not all_chunks:
        raise RuntimeError("Chunker produced zero chunks")
    return all_chunks


# ── Embedding + Upsert ────────────────────────────────────────────────────────

def embed_and_store(chunks: list[ChunkRecord], batch_size: int = 64) -> int:
    """Embed chunks and upsert to Pinecone. Returns number of vectors stored."""
    model = SentenceTransformer(settings.embedding_model)
    index = get_index()
    total = 0

    for start in range(0, len(chunks), batch_size):
        batch      = chunks[start : start + batch_size]
        texts      = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

        vectors = [
            {
                "id":     c["chunk_id"],
                "values": emb,
                "metadata": {
                    "arxiv_id":    c["arxiv_id"],
                    "title":       c["title"],
                    "authors":     c["authors"],
                    "published":   c["published"],
                    "text":        c["text"],
                    "chunk_index": c["chunk_index"],
                    "total_chunks": c["total_chunks"],
                    "source":          c.get("source", "arxiv"),
                    "citation_count":  c.get("citation_count", 0),
                    "concept_tags":    c.get("concept_tags", []),
                },
            }
            for c, emb in zip(batch, embeddings)
        ]
        index.upsert(vectors=vectors)
        total += len(vectors)
        logger.debug("Upserted %d vectors", len(vectors))

    logger.info("Embedding complete | %d vectors stored", total)
    return total


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_ingest(query: str, max_results: int) -> dict:
    papers  = fetch_and_rank(query, max_results=max_results)
    if not papers:
        raise RuntimeError("All sources returned zero papers for query: '{query}'")
    chunks  = chunk_papers(papers)
    stored  = embed_and_store(chunks)
    return {"papers_fetched": len(papers), "chunks_created": len(chunks), "vectors_stored": stored}
