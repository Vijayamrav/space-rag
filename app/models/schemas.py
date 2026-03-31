"""
Pydantic schemas shared across routers and services.
"""

from pydantic import BaseModel, Field


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    query:       str = Field(..., description="arXiv search query", example="james webb telescope exoplanet")
    max_results: int = Field(10, ge=1, le=100, description="Number of papers to ingest")


class IngestResponse(BaseModel):
    papers_fetched: int
    chunks_created: int
    vectors_stored: int
    message: str = "Ingestion started in background. Check server logs for progress."


# ── Query ─────────────────────────────────────────────────────────────────────

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
