"""
Module 2: Chunker
-----------------
Downloads PDFs from arXiv and splits them into overlapping text chunks.

Flow:
    list[PaperRecord]
        --> download PDF (in-memory, no disk I/O)
            --> extract text via PyMuPDF
                --> split into overlapping chunks
                    --> list[ChunkRecord]
                        --> passed to embedder (Module 3)

Why chunking matters for RAG:
    LLMs have context limits. We split papers into small overlapping
    windows so the retriever can find the exact passage relevant to
    a query, rather than stuffing an entire 20-page paper into the prompt.
"""

import io
import logging
import time
from typing import TypedDict

import fitz          # PyMuPDF
import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.ingestion import PaperRecord

logger = logging.getLogger(__name__)


# ── Data contract between chunker and embedder ───────────────────────────────

class ChunkRecord(TypedDict):
    chunk_id:  str        # "{arxiv_id}::chunk::{index}"
    arxiv_id:  str
    title:     str
    authors:   list[str]
    published: str
    text:      str        # the actual chunk text fed to the embedder
    chunk_index: int
    total_chunks: int     # useful for debugging chunk distribution


# ── PDF download ─────────────────────────────────────────────────────────────

def _download_pdf(pdf_url: str, retries: int = 3) -> bytes:
    """Download PDF bytes in-memory. No temp files."""
    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(pdf_url)
                response.raise_for_status()
                logger.debug("Downloaded PDF | url=%s | size=%d bytes", pdf_url, len(response.content))
                return response.content
        except Exception as exc:
            logger.warning("PDF download attempt %d failed: %s", attempt, exc)
            if attempt < retries:
                time.sleep(2.0)

    raise RuntimeError(f"Failed to download PDF after {retries} attempts: {pdf_url}")


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_text(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using PyMuPDF."""
    doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    full_text = "\n".join(pages).strip()
    logger.debug("Extracted %d characters from PDF", len(full_text))
    return full_text


# ── Text splitter ─────────────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text using LangChain's RecursiveCharacterTextSplitter.

    Tries to split on paragraphs → sentences → words → characters in order,
    so chunks break at natural language boundaries rather than mid-word.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # priority order
        length_function=len,
    )
    return splitter.split_text(text)


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_papers(
    papers: list[PaperRecord],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[ChunkRecord]:
    """
    Download and chunk a list of papers.

    Args:
        papers:        output from ingestion.fetch_papers()
        chunk_size:    override config default (characters per chunk)
        chunk_overlap: override config default (overlap between chunks)

    Returns:
        Flat list of ChunkRecord dicts across all papers
    """
    chunk_size    = chunk_size    or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    all_chunks: list[ChunkRecord] = []

    for paper in papers:
        logger.info("Chunking paper | arxiv_id=%s | title='%s'", paper["arxiv_id"], paper["title"][:60])

        try:
            pdf_bytes = _download_pdf(paper["pdf_url"])
            text      = _extract_text(pdf_bytes)

            if not text:
                logger.warning("Empty text extracted | arxiv_id=%s — skipping", paper["arxiv_id"])
                continue

            raw_chunks = _split_text(text, chunk_size, chunk_overlap)
            total      = len(raw_chunks)

            for idx, chunk_text in enumerate(raw_chunks):
                all_chunks.append(
                    ChunkRecord(
                        chunk_id     = f"{paper['arxiv_id']}::chunk::{idx}",
                        arxiv_id     = paper["arxiv_id"],
                        title        = paper["title"],
                        authors      = paper["authors"],
                        published    = paper["published"],
                        text         = chunk_text,
                        chunk_index  = idx,
                        total_chunks = total,
                    )
                )

            logger.info(
                "Produced %d chunks | arxiv_id=%s | avg_chars=%d",
                total,
                paper["arxiv_id"],
                sum(len(c["text"]) for c in all_chunks[-total:]) // max(total, 1),
            )

        except Exception as exc:
            # Log and continue — one bad PDF shouldn't kill the whole pipeline
            logger.error("Failed to chunk paper %s: %s", paper["arxiv_id"], exc)
            continue

    if not all_chunks:
        raise RuntimeError("Chunker produced zero chunks — check PDF downloads and text extraction logs")

    logger.info("Total chunks produced: %d across %d papers", len(all_chunks), len(papers))
    return all_chunks


# ── Debug entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")

    from app.ingestion import fetch_papers

    papers = fetch_papers("james webb telescope exoplanet", max_results=1)
    chunks = chunk_papers(papers)

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"\n--- Chunk 0 preview ---")
    print(chunks[0]["text"][:300])
    print(f"\n--- Chunk 1 preview ---")
    print(chunks[1]["text"][:300] if len(chunks) > 1 else "only one chunk")
