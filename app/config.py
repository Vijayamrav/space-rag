"""
config.py — Central settings loaded from environment variables.
All modules import from here. Never hardcode secrets anywhere else.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenRouter
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "stepfun-ai/step-3-5-flash"

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "space-rag"
    pinecone_environment: str = "us-east-1"

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
