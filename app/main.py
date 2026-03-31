"""
Module 6: API Layer
-------------------
FastAPI application exposing the RAG pipeline over HTTP.

Endpoints:
    POST /ingest   — fetch papers, chunk, embed, store in Pinecone
    POST /query    — retrieve + generate answer for a user question
    GET  /health   — liveness check

Flow:
    /ingest:  query string --> ingestion --> chunker --> embedder --> Pinecone
    /query:   question     --> retriever --> generator --> answer + sources
"""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.ingestion import fetch_papers
from app.chunker import chunk_papers
from app.embedder import embed_and_store
from app.retriever import retrieve
from app.generator import generate_answer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Space RAG API",
    description="Retrieval-Augmented Generation over arXiv space research papers.",
    version="1.0.0",
)


# ── Request / Response schemas ────────────────────────────────────────────────

class IngestRequest(BaseModel):
    query:       str = Field(..., description="arXiv search query", example="james webb telescope exoplanet")
    max_results: int = Field(10, ge=1, le=100, description="Number of papers to ingest")

class IngestResponse(BaseModel):
    papers_fetched: int
    chunks_created: int
    vectors_stored: int

class QueryRequest(BaseModel):
    question:   str  = Field(..., description="Natural language question", example="How does JWST detect exoplanet atmospheres?")
    top_k:      int  = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    use_rerank: bool = Field(True, description="Apply BM25 re-ranking on dense results")

class SourceMeta(BaseModel):
    chunk_id:  str
    arxiv_id:  str
    title:     str
    authors:   list[str]
    published: str
    score:     float

class QueryResponse(BaseModel):
    answer:  str
    sources: list[SourceMeta]
    model:   str
    query:   str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """
    Fetch arXiv papers matching the query, chunk them, embed and store in Pinecone.
    """
    try:
        logger.info("Ingest request | query='%s' max_results=%d", req.query, req.max_results)

        papers  = fetch_papers(req.query, max_results=req.max_results)
        chunks  = chunk_papers(papers)
        records = embed_and_store(chunks)

        return IngestResponse(
            papers_fetched = len(papers),
            chunks_created = len(chunks),
            vectors_stored = len(records),
        )

    except Exception as exc:
        logger.error("Ingest failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Retrieve relevant chunks and generate a grounded answer.
    """
    try:
        logger.info("Query request | question='%s'", req.question[:80])

        chunks = retrieve(req.question, top_k=req.top_k, use_rerank=req.use_rerank)
        result = generate_answer(req.question, chunks)

        return QueryResponse(
            answer  = result["answer"],
            sources = [SourceMeta(**s) for s in result["sources"]],
            model   = result["model"],
            query   = result["query"],
        )

    except Exception as exc:
        logger.error("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
