from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.models.schemas import IngestRequest, IngestResponse
from app.services.ingestion_service import run_ingest

router = APIRouter(prefix="/ingest", tags=["Ingest"])


@router.post("", response_model=IngestResponse)
def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """
    Kicks off ingestion in the background and returns immediately.
    Check server logs to track progress.
    """
    try:
        # validate the request first with a quick dry-run check
        background_tasks.add_task(run_ingest, req.query, req.max_results)
        return IngestResponse(
            papers_fetched = -1,   # -1 = in progress
            chunks_created = -1,
            vectors_stored = -1,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
