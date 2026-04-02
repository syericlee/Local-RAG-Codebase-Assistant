from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from rag_assistant.config import Settings
from rag_assistant.models.chunk import ChunkType, CodeChunk
from rag_assistant.models.search import RerankedResult, SearchResult
from rag_assistant.retrieval.reranker import CrossEncoderReranker
from rag_assistant.retrieval.retriever import Retriever
from rag_assistant.retrieval.vector_store import QdrantStore

REPO_URL = "https://github.com/foo/bar"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(symbol_name: str = "foo", start_line: int = 1) -> CodeChunk:
    return CodeChunk(
        id=CodeChunk.make_id(REPO_URL, "src/main.py", start_line),
        content=f"def {symbol_name}(): pass",
        file_path="src/main.py",
        start_line=start_line,
        end_line=start_line,
        language="python",
        chunk_type=ChunkType.FUNCTION,
        symbol_name=symbol_name,
        repo_url=REPO_URL,
        file_hash="abc123",
    )


def make_search_result(symbol_name: str = "foo", score: float = 0.9) -> SearchResult:
    return SearchResult(chunk=make_chunk(symbol_name), vector_score=score)


# ---------------------------------------------------------------------------
# QdrantStore
# ---------------------------------------------------------------------------

class TestQdrantStore:
    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        client = AsyncMock()
        client.collection_exists.return_value = False
        return client

    @pytest.fixture
    def store(self, mock_client: AsyncMock, settings: Settings) -> QdrantStore:
        return QdrantStore(mock_client, settings.qdrant_collection_name, settings.qdrant_vector_size)

    @pytest.mark.asyncio
    async def test_initialize_creates_collection_when_missing(
        self, store: QdrantStore, mock_client: AsyncMock
    ) -> None:
        await store.initialize()
        mock_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_skips_creation_when_exists(
        self, store: QdrantStore, mock_client: AsyncMock
    ) -> None:
        mock_client.collection_exists.return_value = True
        await store.initialize()
        mock_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_upsert_chunks_calls_client(
        self, store: QdrantStore, mock_client: AsyncMock
    ) -> None:
        chunks = [make_chunk("foo", 1), make_chunk("bar", 10)]
        embeddings = np.random.rand(2, 768).astype(np.float32)
        await store.upsert_chunks(chunks, embeddings)
        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args.kwargs
        assert len(call_kwargs["points"]) == 2

    @pytest.mark.asyncio
    async def test_upsert_point_id_is_int(
        self, store: QdrantStore, mock_client: AsyncMock
    ) -> None:
        chunks = [make_chunk("foo", 1)]
        embeddings = np.random.rand(1, 768).astype(np.float32)
        await store.upsert_chunks(chunks, embeddings)
        points = mock_client.upsert.call_args.kwargs["points"]
        assert isinstance(points[0].id, int)

    @pytest.mark.asyncio
    async def test_search_returns_search_results(
        self, store: QdrantStore, mock_client: AsyncMock
    ) -> None:
        chunk = make_chunk("foo")
        scored_point = MagicMock()
        scored_point.payload = chunk.model_dump()
        scored_point.score = 0.95
        mock_client.search.return_value = [scored_point]

        query_vector = np.random.rand(768).astype(np.float32)
        results = await store.search(query_vector, top_k=5)

        assert len(results) == 1
        assert results[0].vector_score == 0.95
        assert results[0].chunk.symbol_name == "foo"

    @pytest.mark.asyncio
    async def test_search_with_repo_filter_passes_filter(
        self, store: QdrantStore, mock_client: AsyncMock
    ) -> None:
        mock_client.search.return_value = []
        query_vector = np.random.rand(768).astype(np.float32)
        await store.search(query_vector, top_k=5, repo_url=REPO_URL)
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["query_filter"] is not None

    @pytest.mark.asyncio
    async def test_search_without_repo_filter_passes_none(
        self, store: QdrantStore, mock_client: AsyncMock
    ) -> None:
        mock_client.search.return_value = []
        query_vector = np.random.rand(768).astype(np.float32)
        await store.search(query_vector, top_k=5)
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["query_filter"] is None

    @pytest.mark.asyncio
    async def test_delete_points_converts_hex_to_int(
        self, store: QdrantStore, mock_client: AsyncMock
    ) -> None:
        chunk_id = make_chunk().id  # 16-char hex
        await store.delete_points([chunk_id])
        mock_client.delete.assert_called_once()


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------

class TestCrossEncoderReranker:
    @pytest.fixture
    def mock_cross_encoder(self) -> MagicMock:
        model = MagicMock()
        # Return descending scores so we can predict sort order
        model.predict.side_effect = lambda pairs: np.array(
            [10.0 - i for i in range(len(pairs))]
        )
        return model

    @pytest.fixture
    def reranker(self, mock_cross_encoder: MagicMock) -> CrossEncoderReranker:
        r = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        r._model = mock_cross_encoder
        return r

    def test_returns_top_n(self, reranker: CrossEncoderReranker) -> None:
        results = [make_search_result(f"fn{i}", 0.9 - i * 0.1) for i in range(5)]
        reranked = reranker.rerank("query", results, top_n=3)
        assert len(reranked) == 3

    def test_sorted_by_rerank_score_descending(
        self, reranker: CrossEncoderReranker
    ) -> None:
        results = [make_search_result(f"fn{i}") for i in range(4)]
        reranked = reranker.rerank("query", results, top_n=4)
        scores = [r.rerank_score for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input_returns_empty(self, reranker: CrossEncoderReranker) -> None:
        assert reranker.rerank("query", [], top_n=5) == []

    def test_preserves_vector_score(self, reranker: CrossEncoderReranker) -> None:
        result = make_search_result("foo", score=0.77)
        reranked = reranker.rerank("query", [result], top_n=1)
        assert reranked[0].vector_score == 0.77

    def test_model_loaded_lazily(self) -> None:
        r = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert r._model is None

    def test_pairs_passed_to_model(
        self, reranker: CrossEncoderReranker, mock_cross_encoder: MagicMock
    ) -> None:
        results = [make_search_result("foo")]
        reranker.rerank("my query", results, top_n=1)
        pairs = mock_cross_encoder.predict.call_args[0][0]
        assert pairs[0][0] == "my query"
        assert "foo" in pairs[0][1]


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class TestRetriever:
    @pytest.fixture
    def mock_embedder(self) -> MagicMock:
        embedder = MagicMock()
        embedder.embed_query.return_value = np.random.rand(768).astype(np.float32)
        return embedder

    @pytest.fixture
    def mock_vector_store(self) -> AsyncMock:
        store = AsyncMock()
        store.search.return_value = [make_search_result(f"fn{i}") for i in range(5)]
        return store

    @pytest.fixture
    def mock_reranker(self) -> MagicMock:
        reranker = MagicMock()
        reranker.rerank.return_value = [
            RerankedResult(
                chunk=make_chunk(f"fn{i}", i + 1),
                vector_score=0.9,
                rerank_score=float(5 - i),
            )
            for i in range(3)
        ]
        return reranker

    @pytest.fixture
    def retriever(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: AsyncMock,
        mock_reranker: MagicMock,
        settings: Settings,
    ) -> Retriever:
        return Retriever(mock_embedder, mock_vector_store, mock_reranker, settings)

    @pytest.mark.asyncio
    async def test_returns_retrieval_response(self, retriever: Retriever) -> None:
        response = await retriever.retrieve("how does auth work?")
        assert len(response.results) > 0
        assert response.total_tokens_in_context > 0

    @pytest.mark.asyncio
    async def test_embed_query_called(
        self, retriever: Retriever, mock_embedder: MagicMock
    ) -> None:
        await retriever.retrieve("test query")
        mock_embedder.embed_query.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_vector_search_called_with_top_k(
        self, retriever: Retriever, mock_vector_store: AsyncMock, settings: Settings
    ) -> None:
        await retriever.retrieve("test query")
        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["top_k"] == settings.retrieval_top_k

    @pytest.mark.asyncio
    async def test_repo_url_forwarded_to_search(
        self, retriever: Retriever, mock_vector_store: AsyncMock
    ) -> None:
        await retriever.retrieve("query", repo_url=REPO_URL)
        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["repo_url"] == REPO_URL

    @pytest.mark.asyncio
    async def test_reranker_called_with_query(
        self, retriever: Retriever, mock_reranker: MagicMock
    ) -> None:
        await retriever.retrieve("my question")
        assert mock_reranker.rerank.call_args[0][0] == "my question"

    @pytest.mark.asyncio
    async def test_context_budget_respected(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: AsyncMock,
        mock_reranker: MagicMock,
        settings: Settings,
    ) -> None:
        # Give each chunk content of ~1000 tokens; budget is 3000 → max 3 chunks
        long_content = " ".join(["token"] * 1000)
        mock_reranker.rerank.return_value = [
            RerankedResult(
                chunk=CodeChunk(
                    id=CodeChunk.make_id(REPO_URL, "f.py", i),
                    content=long_content,
                    file_path="f.py",
                    start_line=i,
                    end_line=i + 10,
                    language="python",
                    chunk_type=ChunkType.FUNCTION,
                    repo_url=REPO_URL,
                    file_hash="abc",
                ),
                vector_score=0.9,
                rerank_score=float(5 - i),
            )
            for i in range(5)
        ]
        retriever = Retriever(mock_embedder, mock_vector_store, mock_reranker, settings)
        response = await retriever.retrieve("query")
        assert response.total_tokens_in_context <= settings.context_token_budget

    @pytest.mark.asyncio
    async def test_timing_fields_are_non_negative(self, retriever: Retriever) -> None:
        response = await retriever.retrieve("query")
        assert response.query_embedding_ms >= 0
        assert response.vector_search_ms >= 0
        assert response.rerank_ms >= 0
