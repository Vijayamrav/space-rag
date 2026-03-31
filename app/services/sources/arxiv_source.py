"""
arXiv source — wraps the arxiv SDK.
"""

import logging
import time

import arxiv

from app.services.sources.base import PaperRecord

logger = logging.getLogger(__name__)


def fetch(query: str, max_results: int = 20, retries: int = 3) -> list[PaperRecord]:
    client = arxiv.Client(page_size=min(max_results, 100), delay_seconds=1.0, num_retries=retries)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    for attempt in range(1, retries + 1):
        try:
            logger.info("arXiv fetch | query='%s' attempt=%d", query, attempt)
            papers = []
            for r in client.results(search):
                arxiv_id = r.entry_id.split("/")[-1]
                papers.append(PaperRecord(
                    arxiv_id          = arxiv_id,
                    title             = r.title.strip(),
                    authors           = [a.name for a in r.authors],
                    abstract          = r.summary.strip(),
                    pdf_url           = r.pdf_url,
                    published         = r.published.isoformat(),
                    source            = "arxiv",
                    citation_count    = 0,
                    influential_count = 0,
                    concept_tags      = list(r.categories) if hasattr(r, "categories") else [],
                ))
            logger.info("arXiv returned %d papers", len(papers))
            return papers
        except Exception as exc:
            logger.warning("arXiv attempt %d failed: %s", attempt, exc)
            if attempt < retries:
                time.sleep(2.0)

    logger.error("arXiv fetch failed after %d attempts", retries)
    return []
