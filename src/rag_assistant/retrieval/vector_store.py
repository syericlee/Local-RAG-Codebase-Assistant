from __future__ import annotations

from typing import Optional

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from rag_assistant.config import Settings
from rag_assistant.models.chunk import CodeChunk
from rag_assistant.models.search import SearchResult


def _chunk_to_point(chunk: CodeChunk, embedding: np.ndarray) -> PointStruct:
    """Convert a CodeChunk to a Qdrant PointStruct.

    Qdrant point IDs must be unsigned ints or UUIDs. We convert our
    16-char hex chunk ID to an integer deterministically.
    """
    return PointStruct(
        id=int(chunk.id, 16),
        vector=embedding.tolist(),
        payload=chunk.model_dump(),
    )


class QdrantStore:
    """Manages the Qdrant collection: creation, upsert, search, and deletion."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        vector_size: int,
    ) -> None:
        self._client = client
        self._collection = collection_name
        self._vector_size = vector_size

    @classmethod
    def from_settings(cls, settings: Settings) -> QdrantStore:
        client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        return cls(client, settings.qdrant_collection_name, settings.qdrant_vector_size)

    async def initialize(self) -> None:
        """Create the collection if it does not already exist."""
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )

    async def upsert_chunks(
        self, chunks: list[CodeChunk], embeddings: np.ndarray
    ) -> None:
        """Upsert chunks with their embeddings. Idempotent — same chunk ID overwrites."""
        points = [
            _chunk_to_point(chunk, embeddings[i])
            for i, chunk in enumerate(chunks)
        ]
        await self._client.upsert(collection_name=self._collection, points=points)

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        repo_url: Optional[str] = None,
    ) -> list[SearchResult]:
        """Vector search, optionally filtered to a single repo."""
        query_filter: Optional[Filter] = None
        if repo_url is not None:
            query_filter = Filter(
                must=[FieldCondition(key="repo_url", match=MatchValue(value=repo_url))]
            )

        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            SearchResult(chunk=CodeChunk(**hit.payload), vector_score=hit.score)
            for hit in hits
        ]

    async def delete_points(self, point_ids: list[str]) -> None:
        """Delete Qdrant points by their hex chunk IDs."""
        int_ids = [int(pid, 16) for pid in point_ids]
        await self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=int_ids),
        )

    async def get_collection_info(self) -> dict:
        info = await self._client.get_collection(self._collection)
        return info.model_dump()

    async def close(self) -> None:
        await self._client.close()
