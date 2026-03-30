"""
Module 1: Ingestion
-------------------
Fetches space research papers from the arXiv API.

Flow:
    query string
        --> arXiv API
            --> list[PaperRecord]
                --> passed to chunker (Module 2)

Each PaperRecord contains metadata + pdf_url so the chunker
can download and parse the full text.
"""

import logging
import time
from typing import TypedDict

import arxiv

logger = logging.getLogger(__name__)


# ── Data contract between ingestion and chunker ──────────────────────────────

class PaperRecord(TypedDict):
    arxiv_id:  str
    title:     str
    authors:   list[str]
    abstract:  str
    pdf_url:   str
    published: str   # ISO-8601 string, safe for JSON serialisation


# ── Core fetcher ─────────────────────────────────────────────────────────────

def fetch_papers(
    query: str,
    max_results: int = 20,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> list[PaperRecord]:
    """
    Query arXiv and return structured paper records.

    Args:
        query:        arXiv search string  e.g. "black holes gravitational waves"
        max_results:  number of papers to retrieve (max 100 per arXiv policy)
        retries:      how many times to retry on network failure
        retry_delay:  seconds to wait between retries

    Returns:
        List of PaperRecord dicts

    Raises:
        RuntimeError: if all retry attempts fail
    """
    client = arxiv.Client(
        page_size=min(max_results, 100),
        delay_seconds=1.0,   # be polite to arXiv
        num_retries=retries,
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: list[PaperRecord] = []

    for attempt in range(1, retries + 1):
        try:
            logger.info("arXiv fetch attempt %d | query='%s' max=%d", attempt, query, max_results)
            for result in client.results(search):
                papers.append(
                    PaperRecord(
                        arxiv_id  = result.entry_id.split("/")[-1],
                        title     = result.title.strip(),
                        authors   = [a.name for a in result.authors],
                        abstract  = result.summary.strip(),
                        pdf_url   = result.pdf_url,
                        published = result.published.isoformat(),
                    )
                )
            logger.info("Fetched %d papers", len(papers))
            return papers

        except Exception as exc:
            logger.warning("Attempt %d failed: %s", attempt, exc)
            if attempt < retries:
                time.sleep(retry_delay)

    raise RuntimeError(f"arXiv fetch failed after {retries} attempts for query: '{query}'")


# ── Debug entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    papers = fetch_papers("james webb telescope exoplanet atmosphere", max_results=3)
    for p in papers:
        print(f"\n[{p['arxiv_id']}] {p['title']}")
        print(f"  Authors   : {', '.join(p['authors'][:3])}")
        print(f"  Published : {p['published']}")
        print(f"  PDF       : {p['pdf_url']}")
