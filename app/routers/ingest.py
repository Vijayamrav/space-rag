from fastapi import APIRouter, HTTPException

from app.models.schemas import IngestRequest, IngestResponse
from app.services.ingestion_service import run_ingest

router = APIRouter(prefix="/ingest", tags=["Ingest"])


@router.post("", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """Fetch arXiv papers, chunk, embed, and store in Pinecone."""
    try:
        result = run_ingest(req.query, req.max_results)
        return IngestResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
