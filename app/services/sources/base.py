"""
Shared PaperRecord type used across all sources.
Citation metadata added for ranking.
"""

from typing import TypedDict


class PaperRecord(TypedDict):
    arxiv_id:          str
    title:             str
    authors:           list[str]
    abstract:          str
    pdf_url:           str
    published:         str          # ISO-8601
    source:            str          # "arxiv" | "semantic_scholar" | "openalex"
    citation_count:    int          # 0 if unavailable
    influential_count: int          # Semantic Scholar influential citations (0 if N/A)
    concept_tags:      list[str]    # OpenAlex concepts / S2 fields of study
