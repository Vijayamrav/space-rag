"""
OpenAlex source — uses the public OpenAlex API (no key required).
Adds citation counts and concept tags for ranking.
"""

import logging
import time

import httpx

from app.services.sources.base import PaperRecord

logger = logging.getLogger(__name__)

_BASE_URL  = "https://api.openalex.org/works"
_MAILTO    = "space-rag@example.com"   # polite pool — speeds up responses


def fetch(query: str, max_results: int = 20, retries: int = 3) -> list[PaperRecord]:
    params = {
        "search":    query,
        "per-page":  min(max_results, 100),
        "filter":    "is_oa:true",          # open access only so we can get PDFs
        "select":    "id,doi,title,authorships,abstract_inverted_index,publication_date,cited_by_count,concepts,open_access,primary_location",
        "mailto":    _MAILTO,
    }

    for attempt in range(1, retries + 1):
        try:
            logger.info("OpenAlex fetch | query='%s' attempt=%d", query, attempt)
            with httpx.Client(timeout=20.0) as client:
                r = client.get(_BASE_URL, params=params)
                r.raise_for_status()
                results = r.json().get("results", [])

            papers = []
            for item in results:
                pdf_url  = _get_pdf_url(item)
                arxiv_id = _get_arxiv_id(item)

                if not pdf_url:
                    continue

                papers.append(PaperRecord(
                    arxiv_id          = arxiv_id or item.get("id", "").split("/")[-1],
                    title             = (item.get("title") or "").strip(),
                    authors           = _get_authors(item),
                    abstract          = _decode_abstract(item.get("abstract_inverted_index")),
                    pdf_url           = pdf_url,
                    published         = item.get("publication_date") or "1970-01-01",
                    source            = "openalex",
                    citation_count    = item.get("cited_by_count") or 0,
                    influential_count = 0,   # not available in OpenAlex
                    concept_tags      = [c["display_name"] for c in (item.get("concepts") or [])[:5]],
                ))

            logger.info("OpenAlex returned %d usable papers", len(papers))
            return papers

        except Exception as exc:
            logger.warning("OpenAlex attempt %d failed: %s", attempt, exc)
            if attempt < retries:
                time.sleep(2.0)

    logger.error("OpenAlex fetch failed after %d attempts", retries)
    return []


def _get_pdf_url(item: dict) -> str | None:
    oa = item.get("open_access") or {}
    if oa.get("oa_url"):
        return oa["oa_url"]
    loc = item.get("primary_location") or {}
    return loc.get("pdf_url")


def _get_arxiv_id(item: dict) -> str | None:
    doi = item.get("doi") or ""
    # OpenAlex sometimes stores arXiv DOIs like 10.48550/arxiv.2301.12345
    if "arxiv" in doi.lower():
        return doi.split("arxiv.")[-1]
    loc = item.get("primary_location") or {}
    landing = (loc.get("landing_page_url") or "")
    if "arxiv.org" in landing:
        return landing.rstrip("/").split("/")[-1]
    return None


def _get_authors(item: dict) -> list[str]:
    return [
        (a.get("author") or {}).get("display_name", "")
        for a in (item.get("authorships") or [])
    ]


def _decode_abstract(inverted_index: dict | None) -> str:
    """OpenAlex stores abstracts as inverted index {word: [positions]}. Reconstruct."""
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions))
