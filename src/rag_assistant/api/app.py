from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_assistant.api.middleware import RequestIDMiddleware
from rag_assistant.api.routers import admin, query
from rag_assistant.cache.redis_cache import TwoLevelCache
from rag_assistant.config import get_settings
from rag_assistant.embedding.embedder import CodeEmbedder
from rag_assistant.generation.llm import OllamaClient
from rag_assistant.ingestion.tracker import SQLiteTracker
from rag_assistant.jobs.job_store import RedisJobStore
from rag_assistant.retrieval.reranker import CrossEncoderReranker
from rag_assistant.retrieval.vector_store import QdrantStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start and stop all long-lived resources with the application."""
    settings = get_settings()
    logger.info("Starting up RAG assistant...")

    # Qdrant
    vector_store = QdrantStore.from_settings(settings)
    await vector_store.initialize()
    app.state.vector_store = vector_store

    # Embedding model (lazy — loads on first request)
    app.state.embedder = CodeEmbedder.from_settings(settings)

    # Reranker (lazy)
    app.state.reranker = CrossEncoderReranker(settings.reranker_model_name)

    # Redis cache
    cache = TwoLevelCache.from_settings(settings)
    await cache.initialize()
    app.state.cache = cache

    # Job store
    app.state.job_store = RedisJobStore.from_settings(settings)

    # SQLite tracker
    app.state.tracker = SQLiteTracker(settings.sqlite_db_path)

    # Ollama client
    app.state.ollama_client = OllamaClient.from_settings(settings)

    logger.info("All services initialised. Ready to serve requests.")
    yield

    # Shutdown
    logger.info("Shutting down...")
    await vector_store.close()
    await cache.close()
    app.state.tracker.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Local RAG Codebase Assistant",
        description="Index a GitHub repo and ask natural language questions about its code.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestIDMiddleware)
    app.include_router(query.router)
    app.include_router(admin.router)

    @app.get("/health", tags=["health"])
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
