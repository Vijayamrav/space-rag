from fastapi import FastAPI

from app.utils.logging import setup_logging
from app.utils.errors import generic_exception_handler
from app.routers import ingest, query

setup_logging()

app = FastAPI(
    title="Space RAG API",
    description="Retrieval-Augmented Generation over arXiv space research papers.",
    version="2.0.0",
)

app.add_exception_handler(Exception, generic_exception_handler)

app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
