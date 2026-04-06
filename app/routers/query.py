from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse, SourceMeta
from app.services.retrieval_service import retrieve
from app.services.generation_service import generate_answer

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
def query(req: QueryRequest):
    """Retrieve relevant chunks and generate a grounded answer."""
    try:
        chunks = retrieve(req.question, top_k=req.top_k, alpha=req.alpha)
        result = generate_answer(req.question, chunks)
        return QueryResponse(
            answer  = result["answer"],
            sources = [SourceMeta(**s) for s in result["sources"]],
            model   = result["model"],
            query   = result["query"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
