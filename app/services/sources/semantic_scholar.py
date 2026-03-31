"""
Semantic Scholar source — uses the public S2 API (no key required for basic use).
Enriches results with citation counts and influential citation counts.
"""

import logging
import time

import httpx

from app.services.sources.base import PaperRecord

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS   = "paperId,externalIds,title,authors,abstract,year,citationCount,influentialCitationCount,fieldsOfStudy,openAccessPdf,publicationDate"


def fetch(query: str, max_results: int = 20, retries: int = 3) -> list[PaperRecord]:
    params = {
        "query":  query,
        "limit":  min(max_results, 100),
        "fields": _FIELDS,
    }

    for attempt in range(1, retries + 1):
        try:
            logger.info("Semantic Scholar fetch | query='%s' attempt=%d", query, attempt)
            with httpx.Client(timeout=20.0) as client:
                r = client.get(f"{_BASE_URL}/paper/search", params=params)
                r.raise_for_status()
                data = r.json().get("data", [])

            papers = []
            for item in data:
                pdf_url  = _get_pdf_url(item)
                arxiv_id = _get_arxiv_id(item)

                if not pdf_url:
                    continue  # skip papers we can't download

                papers.append(PaperRecord(
                    arxiv_id          = arxiv_id or item.get("paperId", ""),
                    title             = (item.get("title") or "").strip(),
                    authors           = [a["name"] for a in item.get("authors", [])],
                    abstract          = (item.get("abstract") or "").strip(),
                    pdf_url           = pdf_url,
                    published         = _parse_date(item),
                    source            = "semantic_scholar",
                    citation_count    = item.get("citationCount") or 0,
                    influential_count = item.get("influentialCitationCount") or 0,
                    concept_tags      = item.get("fieldsOfStudy") or [],
                ))

            logger.info("Semantic Scholar returned %d usable papers", len(papers))
            return papers

        except Exception as exc:
            logger.warning("Semantic Scholar attempt %d failed: %s", attempt, exc)
            if attempt < retries:
                time.sleep(5.0)  # longer backoff for rate limits

    logger.error("Semantic Scholar fetch failed after %d attempts", retries)
    return []


def _get_pdf_url(item: dict) -> str | None:
    # prefer open access PDF
    oa = item.get("openAccessPdf")
    if oa and oa.get("url"):
        return oa["url"]
    # fall back to arXiv PDF if we have the ID
    arxiv_id = _get_arxiv_id(item)
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}"
    return None


def _get_arxiv_id(item: dict) -> str | None:
    return (item.get("externalIds") or {}).get("ArXiv")


def _parse_date(item: dict) -> str:
    if item.get("publicationDate"):
        return item["publicationDate"]
    year = item.get("year")
    return f"{year}-01-01" if year else "1970-01-01"
