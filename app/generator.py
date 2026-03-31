"""
Module 5: Generator
-------------------
Takes retrieved chunks and a user query, builds a prompt, and calls
the LLM via OpenRouter to produce a grounded answer.

Flow:
    query + list[RetrievedChunk]
        --> build system + user prompt with context
            --> OpenRouter chat completion (OpenAI-compatible SDK)
                --> GeneratorResponse
                    --> returned to API layer (Module 6)

Why OpenRouter:
    Single endpoint to swap models without changing code. The model is
    configured in settings (default: stepfun-ai/step-3-5-flash).
"""

import logging
from typing import TypedDict

from openai import OpenAI

from app.config import settings
from app.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


# ── Data contract between generator and API layer ────────────────────────────

class GeneratorResponse(TypedDict):
    answer:  str
    sources: list[dict]   # [{arxiv_id, title, authors, published, chunk_id}]
    model:   str
    query:   str


# ── Prompt builder ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a scientific research assistant specialising in space science and astrophysics.
Answer the user's question using ONLY the context passages provided below.
If the context does not contain enough information to answer, say so clearly.
Cite the paper title and arXiv ID when referencing specific findings.
Be concise, accurate, and avoid speculation beyond what the context supports.\
"""


def _build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[{i}] Title: {chunk['title']}\n"
            f"    arXiv: {chunk['arxiv_id']} | Published: {chunk['published']}\n"
            f"    Authors: {', '.join(chunk['authors'][:3])}\n"
            f"    Passage:\n{chunk['text']}\n"
        )

    context = "\n---\n".join(context_blocks)
    return f"Context:\n{context}\n\nQuestion: {query}"


# ── Public API ────────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> GeneratorResponse:
    """
    Generate a grounded answer from retrieved chunks.

    Args:
        query:       the user's original question
        chunks:      output from retriever.retrieve()
        model:       override config default model
        temperature: lower = more factual (default 0.2)
        max_tokens:  max response length

    Returns:
        GeneratorResponse with answer text and source metadata
    """
    model = model or settings.model_name

    if not chunks:
        return GeneratorResponse(
            answer  = "I could not find any relevant passages in the knowledge base to answer your question.",
            sources = [],
            model   = model,
            query   = query,
        )

    client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )

    user_prompt = _build_user_prompt(query, chunks)

    logger.info("Calling LLM | model=%s | chunks=%d | query='%s'", model, len(chunks), query[:80])

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content.strip()
    logger.info("LLM response received | tokens_used=%s", response.usage)

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
    ]

    return GeneratorResponse(
        answer  = answer,
        sources = sources,
        model   = model,
        query   = query,
    )


# ── Debug entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from app.retriever import retrieve

    query   = "How does JWST detect exoplanet atmospheres?"
    chunks  = retrieve(query, top_k=3)
    result  = generate_answer(query, chunks)

    print(f"\nQuery : {result['query']}")
    print(f"Model : {result['model']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources ({len(result['sources'])}):")
    for s in result["sources"]:
        print(f"  [{s['arxiv_id']}] {s['title']} (score={s['score']:.4f})")
