from __future__ import annotations

from typing import Optional

from sentence_transformers import CrossEncoder

from rag_assistant.models.search import RerankedResult, SearchResult


class CrossEncoderReranker:
    """Reranks vector search candidates using a cross-encoder model.

    The cross-encoder scores each (query, document) pair jointly, which is
    more accurate than vector similarity alone but too slow to run over the
    full index — so it only runs over the top-K vector search candidates.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Optional[CrossEncoder] = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int,
    ) -> list[RerankedResult]:
        """Score all candidates, return the top_n sorted by rerank score descending."""
        if not results:
            return []

        model = self._get_model()
        pairs = [(query, r.chunk.content) for r in results]
        scores: list[float] = model.predict(pairs).tolist()

        reranked = [
            RerankedResult(
                chunk=r.chunk,
                vector_score=r.vector_score,
                rerank_score=score,
            )
            for r, score in zip(results, scores)
        ]
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)
        return reranked[:top_n]
