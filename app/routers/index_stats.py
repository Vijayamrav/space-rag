from fastapi import APIRouter
from app.core.pinecone_client import get_index

router = APIRouter(prefix="/index", tags=["Index"])


@router.get("/stats")
def index_stats():
    """Total vector count and dimension of the Pinecone index."""
    index = get_index()
    stats = index.describe_index_stats()
    return {
        "total_vectors": stats["total_vector_count"],
        "dimension":     stats["dimension"],
    }


@router.get("/papers")
def list_papers():
    """
    List all unique papers ingested into Pinecone.
    Returns deduplicated list of papers with arxiv_id, title, source, citation_count.
    """
    index = get_index()

    # Pinecone doesn't support full scan — query with a zero vector to sample
    # Use a large top_k to get as many as possible (max 10000)
    stats     = index.describe_index_stats()
    total     = stats["total_vector_count"]
    dim       = stats["dimension"]
    fetch_k   = min(total, 10000)

    if fetch_k == 0:
        return {"total_papers": 0, "papers": []}

    response = index.query(
        vector    = [0.0] * dim,
        top_k     = fetch_k,
        include_metadata = True,
    )

    # Deduplicate by arxiv_id
    seen: dict[str, dict] = {}
    for match in response["matches"]:
        meta     = match["metadata"]
        arxiv_id = meta.get("arxiv_id", "unknown")
        if arxiv_id not in seen:
            seen[arxiv_id] = {
                "arxiv_id":       arxiv_id,
                "title":          meta.get("title", ""),
                "published":      meta.get("published", ""),
                "source":         meta.get("source", "arxiv"),
                "citation_count": meta.get("citation_count", 0),
                "concept_tags":   meta.get("concept_tags", []),
                "chunk_count":    1,
            }
        else:
            seen[arxiv_id]["chunk_count"] += 1

    papers = sorted(seen.values(), key=lambda x: x["citation_count"], reverse=True)

    return {
        "total_vectors": total,
        "total_papers":  len(papers),
        "papers":        papers,
    }
