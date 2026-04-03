from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import fakeredis.aioredis

from rag_assistant.cache.redis_cache import TwoLevelCache, _query_key
from rag_assistant.models.api import Citation, QueryResponse

QUERY = "How does login work?"
EMBEDDING = np.zeros(768, dtype=np.float32)


def _make_response(answer: str = "It uses bcrypt.") -> QueryResponse:
    return QueryResponse(
        answer=answer,
        citations=[Citation(file_path="src/auth.py", start_line=1, end_line=10)],
        cache_hit="miss",
    )


@pytest.fixture
async def cache(settings) -> TwoLevelCache:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=False)
    c = TwoLevelCache(redis, settings)
    # Skip FT.CREATE — fakeredis doesn't support RediSearch commands
    yield c
    await redis.aclose()


class TestExactCache:
    @pytest.mark.asyncio
    async def test_miss_when_empty(self, cache: TwoLevelCache) -> None:
        result, hit = await cache.get(QUERY, EMBEDDING)
        assert result is None
        assert hit == "miss"

    @pytest.mark.asyncio
    async def test_exact_hit_after_set(self, cache: TwoLevelCache) -> None:
        response = _make_response()
        # Bypass semantic store (fakeredis has no FT.SEARCH)
        await cache._set_exact(QUERY, response.model_dump_json())

        raw = await cache._redis.get(_query_key(QUERY))
        assert raw is not None
        cached = QueryResponse.model_validate_json(raw)
        assert cached.answer == response.answer

    @pytest.mark.asyncio
    async def test_exact_hit_returns_correct_response(self, cache: TwoLevelCache) -> None:
        response = _make_response("Exact answer.")
        await cache._set_exact(QUERY, response.model_dump_json())

        result = await cache._get_exact(QUERY)
        assert result is not None
        assert result.answer == "Exact answer."

    @pytest.mark.asyncio
    async def test_different_query_is_miss(self, cache: TwoLevelCache) -> None:
        response = _make_response()
        await cache._set_exact(QUERY, response.model_dump_json())

        result = await cache._get_exact("Something completely different?")
        assert result is None

    @pytest.mark.asyncio
    async def test_query_normalized_before_hashing(self, cache: TwoLevelCache) -> None:
        response = _make_response()
        await cache._set_exact(QUERY, response.model_dump_json())

        # Same query with different casing / whitespace
        result = await cache._get_exact("  HOW DOES LOGIN WORK?  ")
        assert result is not None

    @pytest.mark.asyncio
    async def test_set_stores_in_exact_layer(self, cache: TwoLevelCache) -> None:
        response = _make_response()
        with patch.object(cache, "_set_semantic", new=AsyncMock()):
            await cache.set(QUERY, EMBEDDING, response)

        result = await cache._get_exact(QUERY)
        assert result is not None
        assert result.answer == response.answer


class TestSemanticCache:
    @pytest.mark.asyncio
    async def test_semantic_miss_below_threshold(self, cache: TwoLevelCache) -> None:
        """If Redis returns a low similarity score, treat as miss."""
        low_score = b"0.5"  # distance 0.5 → similarity 0.5, below threshold 0.92

        mock_result = [1, b"cache:sem:abc", [b"response", b'{"answer":"x","citations":[],"cache_hit":"miss"}', b"score", low_score]]
        with patch.object(cache._redis, "execute_command", new=AsyncMock(return_value=mock_result)):
            result = await cache._get_semantic(EMBEDDING)
        assert result is None

    @pytest.mark.asyncio
    async def test_semantic_hit_above_threshold(self, cache: TwoLevelCache) -> None:
        """If Redis returns a high similarity score, return the cached response."""
        response = _make_response("Semantic answer.")
        high_score = b"0.05"  # distance 0.05 → similarity 0.95, above threshold 0.92

        mock_result = [
            1,
            b"cache:sem:abc",
            [b"response", response.model_dump_json().encode(), b"score", high_score],
        ]
        with patch.object(cache._redis, "execute_command", new=AsyncMock(return_value=mock_result)):
            result = await cache._get_semantic(EMBEDDING)

        assert result is not None
        assert result.answer == "Semantic answer."

    @pytest.mark.asyncio
    async def test_semantic_no_results(self, cache: TwoLevelCache) -> None:
        mock_result = [0]
        with patch.object(cache._redis, "execute_command", new=AsyncMock(return_value=mock_result)):
            result = await cache._get_semantic(EMBEDDING)
        assert result is None

    @pytest.mark.asyncio
    async def test_semantic_exception_returns_none(self, cache: TwoLevelCache) -> None:
        with patch.object(cache._redis, "execute_command", new=AsyncMock(side_effect=Exception("no index"))):
            result = await cache._get_semantic(EMBEDDING)
        assert result is None


class TestGetOrdering:
    @pytest.mark.asyncio
    async def test_exact_checked_before_semantic(self, cache: TwoLevelCache) -> None:
        response = _make_response()
        await cache._set_exact(QUERY, response.model_dump_json())

        sem_mock = AsyncMock(return_value=None)
        with patch.object(cache, "_get_semantic", new=sem_mock):
            result, hit = await cache.get(QUERY, EMBEDDING)

        assert hit == "exact"
        sem_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_semantic_checked_when_exact_misses(self, cache: TwoLevelCache) -> None:
        response = _make_response()
        sem_mock = AsyncMock(return_value=response)
        with patch.object(cache, "_get_semantic", new=sem_mock):
            result, hit = await cache.get("unknown query", EMBEDDING)

        assert hit == "semantic"
        assert result is not None
