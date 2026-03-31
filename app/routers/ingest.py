from fastapi import APIRouter, BackgroundTasks

from app.models.schemas import IngestRequest, BulkIngestRequest, IngestResponse
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
