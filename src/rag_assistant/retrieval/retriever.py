from __future__ import annotations

import time

import tiktoken

from rag_assistant.config import Settings
from rag_assistant.embedding.embedder import CodeEmbedder
from rag_assistant.models.search import RerankedResult, RetrievalResponse

from .reranker import CrossEncoderReranker
from .vector_store import QdrantStore

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text, disallowed_special=()))


class Retriever:
    """Full retrieval pipeline: embed → vector search → rerank → context assembly."""

    def __init__(
        self,
        embedder: CodeEmbedder,
        vector_store: QdrantStore,
        reranker: CrossEncoderReranker,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self._top_k = settings.retrieval_top_k
        self._rerank_top_n = settings.rerank_top_n
        self._token_budget = settings.context_token_budget

    async def retrieve(
        self,
        query: str,
        repo_url: str | None = None,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        effective_top_n = top_k or self._rerank_top_n

        # 1. Embed the query
        t0 = time.perf_counter()
        query_vector = self._embedder.embed_query(query)
        embed_ms = (time.perf_counter() - t0) * 1000

        # 2. Vector search — cast a wide net
        t1 = time.perf_counter()
        candidates = await self._vector_store.search(
            query_vector, top_k=self._top_k, repo_url=repo_url
        )
        search_ms = (time.perf_counter() - t1) * 1000

        # 3. Cross-encoder rerank — precision filter
        t2 = time.perf_counter()
        reranked = self._reranker.rerank(query, candidates, top_n=effective_top_n)
        rerank_ms = (time.perf_counter() - t2) * 1000

        # 4. Greedy context assembly within token budget
        selected, total_tokens = self._assemble_context(reranked)

        return RetrievalResponse(
            results=selected,
            total_tokens_in_context=total_tokens,
            query_embedding_ms=round(embed_ms, 2),
            vector_search_ms=round(search_ms, 2),
            rerank_ms=round(rerank_ms, 2),
        )

    def _assemble_context(
        self, results: list[RerankedResult]
    ) -> tuple[list[RerankedResult], int]:
        """Add chunks in score order until the token budget is exhausted."""
        selected: list[RerankedResult] = []
        total_tokens = 0
        for result in results:
            tokens = _count_tokens(result.chunk.content)
            if total_tokens + tokens > self._token_budget:
                break
            selected.append(result)
            total_tokens += tokens
        return selected, total_tokens
