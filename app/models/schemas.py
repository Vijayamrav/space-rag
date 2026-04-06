"""
Pydantic schemas shared across routers and services.
"""

from pydantic import BaseModel, Field


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    query:       str = Field(..., description="arXiv search query", example="james webb telescope exoplanet")
    max_results: int = Field(10, ge=1, le=100, description="Number of papers to ingest")


class BulkIngestRequest(BaseModel):
    queries:     list[str] = Field(..., description="List of search queries to ingest in parallel")
    max_results: int        = Field(10, ge=1, le=100, description="Number of papers per query")


# arXiv astrophysics categories
ARXIV_CATEGORIES = {
    "astro-ph.EP": "Exoplanets",
    "astro-ph.HE": "High Energy (Black Holes, Neutron Stars)",
    "astro-ph.GA": "Galaxies & Quasars",
    "astro-ph.CO": "Cosmology & Dark Matter",
    "astro-ph.SR": "Stellar & Solar",
    "astro-ph.IM": "Instrumentation & Methods",
    "gr-qc":       "General Relativity & Quantum Cosmology",
    "hep-ph":      "High Energy Physics",
}

class CategoryIngestRequest(BaseModel):
    categories:  list[str] = Field(
        default=list(ARXIV_CATEGORIES.keys()),
        description="arXiv category identifiers e.g. ['astro-ph.EP', 'astro-ph.HE']",
        example=["astro-ph.EP", "astro-ph.HE"],
    )
    max_results: int = Field(50, ge=1, le=100, description="Papers per category")


class IngestResponse(BaseModel):
    papers_fetched: int
    chunks_created: int
    vectors_stored: int
    message: str = "Ingestion started in background. Check server logs for progress."


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str   = Field(..., description="Natural language question", example="How does JWST detect exoplanet atmospheres?")
    top_k:      int   = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    alpha:      float = Field(0.75, ge=0.0, le=1.0, description="Hybrid blend: 1.0=pure dense, 0.0=pure sparse BM25")


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
