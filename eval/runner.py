from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from eval.metrics import (
    EvalReport,
    ItemResult,
    correctness,
    faithfulness,
    mrr,
    recall_at_k,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalItem:
    id: str
    question: str
    relevant_chunk_ids: list[str]
    ground_truth_answer: str
    repo_url: Optional[str] = None


def load_dataset(path: Path) -> list[EvalItem]:
    """Load evaluation items from a JSONL file."""
    items = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(EvalItem(
                id=data["id"],
                question=data["question"],
                relevant_chunk_ids=data["relevant_chunk_ids"],
                ground_truth_answer=data["ground_truth_answer"],
                repo_url=data.get("repo_url"),
            ))
    return items


class EvalRunner:
    """Runs evaluation items concurrently and aggregates metrics.

    Args:
        retriever: Retriever instance for vector search + reranking.
        embedder: CodeEmbedder for query embedding.
        ollama_client: OllamaClient for correctness scoring.
        concurrency: Max simultaneous items (default 4).
        skip_generation_metrics: If True, skip faithfulness and correctness
            (useful when Ollama is not available).
    """

    def __init__(
        self,
        retriever,
        embedder,
        ollama_client,
        concurrency: int = 4,
        skip_generation_metrics: bool = False,
    ) -> None:
        self._retriever = retriever
        self._embedder = embedder
        self._ollama = ollama_client
        self._semaphore = asyncio.Semaphore(concurrency)
        self._skip_gen = skip_generation_metrics

    async def run(self, items: list[EvalItem]) -> EvalReport:
        """Evaluate all items and return an EvalReport."""
        tasks = [self._evaluate_item(item) for item in items]
        results = await asyncio.gather(*tasks)
        return EvalReport.from_items(list(results))

    async def _evaluate_item(self, item: EvalItem) -> ItemResult:
        async with self._semaphore:
            try:
                return await self._score_item(item)
            except Exception as exc:
                logger.warning("Error evaluating item %s: %s", item.id, exc)
                return ItemResult(
                    id=item.id,
                    question=item.question,
                    error=str(exc),
                )

    async def _score_item(self, item: EvalItem) -> ItemResult:
        # Retrieve top-10 so we can compute Recall@1, @5, @10
        retrieval = await self._retriever.retrieve(
            item.question,
            repo_url=item.repo_url,
            top_k=10,
        )

        retrieved_ids = [r.chunk.id for r in retrieval.results]

        result = ItemResult(
            id=item.id,
            question=item.question,
            recall_at_1=recall_at_k(retrieved_ids, item.relevant_chunk_ids, k=1),
            recall_at_5=recall_at_k(retrieved_ids, item.relevant_chunk_ids, k=5),
            recall_at_10=recall_at_k(retrieved_ids, item.relevant_chunk_ids, k=10),
            mrr=mrr(retrieved_ids, item.relevant_chunk_ids),
        )

        if not self._skip_gen:
            context = "\n\n".join(r.chunk.content for r in retrieval.results)
            answer_for_eval = retrieval.results[0].chunk.content if retrieval.results else ""

            result.faithfulness = faithfulness(answer_for_eval, context)
            result.correctness = await correctness(
                question=item.question,
                answer=answer_for_eval,
                ground_truth=item.ground_truth_answer,
                ollama_client=self._ollama,
            )

        return result
