"""
Generation Service
------------------
Builds a grounded prompt from retrieved chunks and calls the LLM via OpenRouter.
"""

import logging
from typing import TypedDict

from openai import OpenAI

from app.core.config import settings
from app.services.retrieval_service import RetrievedChunk

logger = logging.getLogger(__name__)


class GeneratorResponse(TypedDict):
    answer:  str
    sources: list[dict]
    model:   str
    query:   str


_SYSTEM_PROMPT = """\
You are a scientific research assistant specialising in space science and astrophysics.
Answer the user's question using ONLY the context passages provided below.
If the context does not contain enough information to answer, say so clearly.
Cite the paper title and arXiv ID when referencing specific findings.
Be concise, accurate, and avoid speculation beyond what the context supports.\
"""


def _build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    blocks = [
        f"[{i}] Title: {c['title']}\n"
        f"    arXiv: {c['arxiv_id']} | Published: {c['published']}\n"
        f"    Authors: {', '.join(c['authors'][:3])}\n"
        f"    Passage:\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    ]
    return f"Context:\n{'---'.join(blocks)}\n\nQuestion: {query}"


def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> GeneratorResponse:
    model = model or settings.model_name

    if not chunks:
        return GeneratorResponse(
            answer  = "I could not find relevant passages in the knowledge base to answer your question.",
            sources = [],
            model   = model,
            query   = query,
        )

    client = OpenAI(api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url)

    logger.info("LLM call | model=%s | chunks=%d | query='%s'", model, len(chunks), query[:80])

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(query, chunks)},
        ],
    )

    answer = response.choices[0].message.content.strip()
    logger.info("LLM response | usage=%s", response.usage)

    return GeneratorResponse(
        answer  = answer,
        sources = [
            {
                "chunk_id":  c["chunk_id"],
                "arxiv_id":  c["arxiv_id"],
                "title":     c["title"],
                "authors":   c["authors"],
                "published": c["published"],
                "score":     c["score"],
            }
            for c in chunks
        ],
        model = model,
        query = query,
    )
