from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "codebase"
    qdrant_vector_size: int = 768

    # Redis
    redis_url: str = "redis://localhost:6379"
    cache_exact_ttl_seconds: int = 3600
    cache_semantic_ttl_seconds: int = 1800
    cache_semantic_threshold: float = 0.92

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-coder:6.7b"
    ollama_timeout_seconds: int = 120

    # Embedding
    embed_model_name: str = "nomic-ai/nomic-embed-code"
    embed_batch_size: int = 32
    embed_device: str = "cpu"

    # Reranking
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_top_k: int = 20
    rerank_top_n: int = 5

    # Chunking
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 64
    context_token_budget: int = 3000

    # Ingestion
    sqlite_db_path: str = "data/tracking.db"
    repos_base_dir: str = "data/repos"


@lru_cache
def get_settings() -> Settings:
    return Settings()
