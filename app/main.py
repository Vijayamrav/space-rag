from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.utils.logging import setup_logging
from app.utils.errors import generic_exception_handler
from app.routers import ingest, query, index_stats

setup_logging()

app = FastAPI(
    title="AstroRAG API",
    description="Retrieval-Augmented Generation over arXiv space research papers.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(Exception, generic_exception_handler)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(index_stats.router)


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
