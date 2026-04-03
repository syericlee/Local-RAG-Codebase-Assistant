from __future__ import annotations

import hashlib
import json
import struct
import uuid
from typing import Optional

import numpy as np
from redis.asyncio import Redis

from rag_assistant.config import Settings
from rag_assistant.models.api import QueryResponse

# Redis key prefixes
_EXACT_PREFIX = "cache:exact:"
_SEM_PREFIX = "cache:sem:"
_SEM_INDEX = "cache_semantic_idx"


def _query_key(query: str) -> str:
    """Stable key for exact-match cache: sha256 of the lowercased, stripped query."""
    normalized = query.strip().lower()
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return f"{_EXACT_PREFIX}{digest}"


def _pack_embedding(embedding: np.ndarray) -> bytes:
    """Serialize a float32 numpy vector to raw bytes for Redis."""
    return struct.pack(f"{len(embedding)}f", *embedding.tolist())


class TwoLevelCache:
    """Two-level query cache backed by Redis Stack.

    Level 1 — Exact match:
        Key:   cache:exact:{sha256(normalized_query)}
        Value: JSON-serialized QueryResponse
        TTL:   settings.cache_exact_ttl_seconds (default 3600 s)

    Level 2 — Semantic match (RediSearch HNSW):
        Each entry is a Redis Hash at cache:sem:{uuid} with fields:
          embedding  – raw float32 bytes (768 dims)
          response   – JSON-serialized QueryResponse
          query      – original query text (for debugging)
        TTL: settings.cache_semantic_ttl_seconds (default 1800 s)

    Requires Redis Stack (redis/redis-stack image) for the FT.* commands.
    """

    def __init__(self, redis: Redis, settings: Settings) -> None:
        self._redis = redis
        self._exact_ttl = settings.cache_exact_ttl_seconds
        self._sem_ttl = settings.cache_semantic_ttl_seconds
        self._threshold = settings.cache_semantic_threshold
        self._vector_size = settings.qdrant_vector_size

    @classmethod
    def from_settings(cls, settings: Settings) -> TwoLevelCache:
        redis = Redis.from_url(settings.redis_url, decode_responses=False)
        return cls(redis, settings)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the RediSearch HNSW index if it does not already exist."""
        try:
            await self._redis.execute_command("FT.INFO", _SEM_INDEX)
        except Exception:
            await self._redis.execute_command(
                "FT.CREATE", _SEM_INDEX,
                "ON", "HASH",
                "PREFIX", "1", _SEM_PREFIX,
                "SCHEMA",
                "embedding", "VECTOR", "HNSW", "6",
                "TYPE", "FLOAT32",
                "DIM", str(self._vector_size),
                "DISTANCE_METRIC", "COSINE",
            )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    async def get(
        self, query: str, query_embedding: np.ndarray
    ) -> tuple[Optional[QueryResponse], str]:
        """Look up a query in the cache.

        Returns:
            (response, cache_hit) where cache_hit is "exact", "semantic", or "miss".
        """
        # Level 1: exact match
        exact_result = await self._get_exact(query)
        if exact_result is not None:
            return exact_result, "exact"

        # Level 2: semantic match
        sem_result = await self._get_semantic(query_embedding)
        if sem_result is not None:
            return sem_result, "semantic"

        return None, "miss"

    async def _get_exact(self, query: str) -> Optional[QueryResponse]:
        key = _query_key(query)
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return QueryResponse.model_validate_json(raw)

    async def _get_semantic(self, embedding: np.ndarray) -> Optional[QueryResponse]:
        embedding_bytes = _pack_embedding(embedding)

        try:
            results = await self._redis.execute_command(
                "FT.SEARCH", _SEM_INDEX,
                f"*=>[KNN 1 @embedding $vec AS score]",
                "PARAMS", "2", "vec", embedding_bytes,
                "SORTBY", "score",
                "RETURN", "2", "response", "score",
                "DIALECT", "2",
            )
        except Exception:
            return None

        # results format: [total_count, key, [field, value, ...], ...]
        if not results or results[0] == 0:
            return None

        fields = results[2]
        field_dict = dict(zip(fields[::2], fields[1::2]))

        score_raw = field_dict.get(b"score")
        if score_raw is None:
            return None

        # Redis returns cosine distance (0 = identical, 1 = opposite).
        # Convert to similarity: similarity = 1 - distance
        similarity = 1.0 - float(score_raw)
        if similarity < self._threshold:
            return None

        response_raw = field_dict.get(b"response")
        if response_raw is None:
            return None

        return QueryResponse.model_validate_json(response_raw)

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    async def set(
        self,
        query: str,
        query_embedding: np.ndarray,
        response: QueryResponse,
    ) -> None:
        """Store a response in both cache levels."""
        response_json = response.model_dump_json()
        await self._set_exact(query, response_json)
        await self._set_semantic(query, query_embedding, response_json)

    async def _set_exact(self, query: str, response_json: str) -> None:
        key = _query_key(query)
        await self._redis.set(key, response_json.encode(), ex=self._exact_ttl)

    async def _set_semantic(
        self, query: str, embedding: np.ndarray, response_json: str
    ) -> None:
        key = f"{_SEM_PREFIX}{uuid.uuid4()}"
        await self._redis.hset(key, mapping={
            "embedding": _pack_embedding(embedding),
            "response": response_json.encode(),
            "query": query.encode(),
        })
        await self._redis.expire(key, self._sem_ttl)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self._redis.aclose()
