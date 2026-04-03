from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from rag_assistant.cache.redis_cache import TwoLevelCache
from rag_assistant.config import Settings, get_settings
from rag_assistant.embedding.embedder import CodeEmbedder
from rag_assistant.generation.citation_parser import CitationParser
from rag_assistant.generation.llm import OllamaClient
from rag_assistant.generation.prompt import PromptBuilder
from rag_assistant.ingestion.pipeline import IngestionPipeline
from rag_assistant.ingestion.tracker import SQLiteTracker
from rag_assistant.jobs.job_store import RedisJobStore
from rag_assistant.retrieval.reranker import CrossEncoderReranker
from rag_assistant.retrieval.retriever import Retriever
from rag_assistant.retrieval.vector_store import QdrantStore

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def get_settings_dep() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


# ---------------------------------------------------------------------------
# Singletons stored on app.state (set during lifespan)
# ---------------------------------------------------------------------------

def get_vector_store(settings: SettingsDep) -> QdrantStore:
    from fastapi import Request
    # Resolved via app.state in lifespan; re-exported here for type safety
    raise NotImplementedError("Use app.state.vector_store")


def get_embedder(settings: SettingsDep) -> CodeEmbedder:
    raise NotImplementedError("Use app.state.embedder")


# ---------------------------------------------------------------------------
# Per-request dependencies (constructed from app.state singletons)
# ---------------------------------------------------------------------------

from fastapi import Request  # noqa: E402


def vector_store(request: Request) -> QdrantStore:
    return request.app.state.vector_store


def embedder(request: Request) -> CodeEmbedder:
    return request.app.state.embedder


def reranker(request: Request) -> CrossEncoderReranker:
    return request.app.state.reranker


def cache(request: Request) -> TwoLevelCache:
    return request.app.state.cache


def job_store(request: Request) -> RedisJobStore:
    return request.app.state.job_store


def tracker(request: Request) -> SQLiteTracker:
    return request.app.state.tracker


def retriever(request: Request) -> Retriever:
    settings = get_settings()
    return Retriever(
        embedder=request.app.state.embedder,
        vector_store=request.app.state.vector_store,
        reranker=request.app.state.reranker,
        settings=settings,
    )


def pipeline(request: Request) -> IngestionPipeline:
    settings = get_settings()
    return IngestionPipeline(
        settings=settings,
        vector_store=request.app.state.vector_store,
        embedder=request.app.state.embedder,
        tracker=request.app.state.tracker,
    )


def prompt_builder() -> PromptBuilder:
    return PromptBuilder()


def citation_parser() -> CitationParser:
    return CitationParser()


def ollama_client(request: Request) -> OllamaClient:
    return request.app.state.ollama_client


# Annotated shorthands for router use
VectorStoreDep = Annotated[QdrantStore, Depends(vector_store)]
EmbedderDep = Annotated[CodeEmbedder, Depends(embedder)]
RerankerDep = Annotated[CrossEncoderReranker, Depends(reranker)]
CacheDep = Annotated[TwoLevelCache, Depends(cache)]
JobStoreDep = Annotated[RedisJobStore, Depends(job_store)]
TrackerDep = Annotated[SQLiteTracker, Depends(tracker)]
RetrieverDep = Annotated[Retriever, Depends(retriever)]
PipelineDep = Annotated[IngestionPipeline, Depends(pipeline)]
PromptBuilderDep = Annotated[PromptBuilder, Depends(prompt_builder)]
CitationParserDep = Annotated[CitationParser, Depends(citation_parser)]
OllamaClientDep = Annotated[OllamaClient, Depends(ollama_client)]
