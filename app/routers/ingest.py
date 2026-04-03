from fastapi import APIRouter, BackgroundTasks

from app.models.schemas import IngestRequest, BulkIngestRequest, CategoryIngestRequest, IngestResponse, ARXIV_CATEGORIES
from app.services.ingestion_service import run_ingest

router = APIRouter(prefix="/ingest", tags=["Ingest"])


@router.post("", response_model=IngestResponse)
def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest a single query in the background."""
    background_tasks.add_task(run_ingest, req.query, req.max_results)
    return IngestResponse(
        papers_fetched = -1,
        chunks_created = -1,
        vectors_stored = -1,
    )


@router.post("/bulk", response_model=IngestResponse)
def bulk_ingest(req: BulkIngestRequest, background_tasks: BackgroundTasks):
    """Ingest multiple queries at once, each as a separate background task."""
    for query in req.queries:
        background_tasks.add_task(run_ingest, query, req.max_results)
    return IngestResponse(
        papers_fetched = -1,
        chunks_created = -1,
        vectors_stored = -1,
        message        = f"Bulk ingestion started for {len(req.queries)} queries. Check server logs for progress.",
    )


@router.post("/categories", response_model=IngestResponse)
def ingest_by_categories(req: CategoryIngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest papers from arXiv categories directly.
    Each category maps to a cat: query e.g. cat:astro-ph.EP.
    Available categories:
        astro-ph.EP  — Exoplanets
        astro-ph.HE  — High Energy (Black Holes, Neutron Stars)
        astro-ph.GA  — Galaxies & Quasars
        astro-ph.CO  — Cosmology & Dark Matter
        astro-ph.SR  — Stellar & Solar
        astro-ph.IM  — Instrumentation & Methods
        gr-qc        — General Relativity & Quantum Cosmology
        hep-ph       — High Energy Physics
    """
    for cat in req.categories:
        background_tasks.add_task(run_ingest, f"cat:{cat}", req.max_results)
    return IngestResponse(
        papers_fetched = -1,
        chunks_created = -1,
        vectors_stored = -1,
        message        = f"Category ingestion started for {len(req.categories)} categories "
                         f"× {req.max_results} papers each. Check server logs.",
    )


@router.get("/categories", tags=["Ingest"])
def list_categories():
    """List all supported arXiv categories for ingestion."""
    return {"categories": [{"id": k, "name": v} for k, v in ARXIV_CATEGORIES.items()]}
