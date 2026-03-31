"""
Aggregator — fetches from all sources in parallel, deduplicates, and ranks.

Ranking signal (higher = better):
    score = 0.5 * norm_citations
          + 0.3 * norm_influential
          + 0.2 * recency_score

Papers with the same arXiv ID from multiple sources are merged,
taking the highest citation count and combining concept tags.
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from app.services.sources import arxiv_source, semantic_scholar, openalex
from app.services.sources.base import PaperRecord

logger = logging.getLogger(__name__)

_SOURCES = {
    "arxiv":            arxiv_source.fetch,
    "semantic_scholar": semantic_scholar.fetch,
    "openalex":         openalex.fetch,
}


def fetch_and_rank(
    query: str,
    max_results: int = 20,
    top_n: int | None = None,
) -> list[PaperRecord]:
    """
    Fetch from all sources in parallel, deduplicate by arxiv_id,
    rank by citation + recency signal, return top_n papers.

    Args:
        query:       search query
        max_results: how many to request from each source
        top_n:       final number to return after ranking (defaults to max_results)

    Returns:
        Ranked, deduplicated list of PaperRecord
    """
    top_n = top_n or max_results
    raw: list[PaperRecord] = []

    # fetch all sources concurrently
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(fn, query, max_results): name
            for name, fn in _SOURCES.items()
        }
        for future in as_completed(futures):
            source_name = futures[future]
            try:
                results = future.result()
                logger.info("Source '%s' returned %d papers", source_name, len(results))
                raw.extend(results)
            except Exception as exc:
                logger.error("Source '%s' raised: %s", source_name, exc)

    deduped = _deduplicate(raw)
    logger.info("After dedup: %d unique papers from %d raw", len(deduped), len(raw))

    ranked = _rank(deduped)
    return ranked[:top_n]


# ── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate(papers: list[PaperRecord]) -> list[PaperRecord]:
    """
    Merge papers with the same arxiv_id.
    Keeps highest citation count and union of concept tags.
    Prefers arXiv source metadata when available.
    """
    seen: dict[str, PaperRecord] = {}

    for paper in papers:
        key = paper["arxiv_id"].lower().strip()
        if not key:
            continue

        if key not in seen:
            seen[key] = paper
        else:
            existing = seen[key]
            # merge: take max citations, union tags, prefer arxiv source
            seen[key] = PaperRecord(
                arxiv_id          = existing["arxiv_id"],
                title             = existing["title"] or paper["title"],
                authors           = existing["authors"] or paper["authors"],
                abstract          = existing["abstract"] or paper["abstract"],
                pdf_url           = existing["pdf_url"] or paper["pdf_url"],
                published         = existing["published"] or paper["published"],
                source            = "arxiv" if "arxiv" in (existing["source"], paper["source"]) else existing["source"],
                citation_count    = max(existing["citation_count"], paper["citation_count"]),
                influential_count = max(existing["influential_count"], paper["influential_count"]),
                concept_tags      = list(set(existing["concept_tags"] + paper["concept_tags"])),
            )

    return list(seen.values())


# ── Ranking ───────────────────────────────────────────────────────────────────

def _rank(papers: list[PaperRecord]) -> list[PaperRecord]:
    if not papers:
        return []

    max_cit  = max(p["citation_count"]    for p in papers) or 1
    max_inf  = max(p["influential_count"] for p in papers) or 1
    now_year = datetime.now().year

    def score(p: PaperRecord) -> float:
        norm_cit = math.log1p(p["citation_count"])    / math.log1p(max_cit)
        norm_inf = math.log1p(p["influential_count"]) / math.log1p(max_inf)
        try:
            pub_year = datetime.fromisoformat(p["published"]).year
        except Exception:
            pub_year = now_year - 5
        recency  = max(0.0, 1.0 - (now_year - pub_year) / 10.0)
        return 0.5 * norm_cit + 0.3 * norm_inf + 0.2 * recency

    return sorted(papers, key=score, reverse=True)
