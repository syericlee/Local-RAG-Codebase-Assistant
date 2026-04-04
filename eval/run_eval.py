#!/usr/bin/env python3
"""CLI entry point for the evaluation pipeline.

Usage:
    python eval/run_eval.py --dataset eval/dataset.jsonl
    python eval/run_eval.py --dataset eval/dataset.jsonl --skip-generation
    python eval/run_eval.py --dataset eval/dataset.jsonl --output eval/report.json
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main(args: argparse.Namespace) -> None:
    # Imports here so startup is fast for --help
    from eval.runner import EvalRunner, load_dataset
    from rag_assistant.config import get_settings
    from rag_assistant.embedding.embedder import CodeEmbedder
    from rag_assistant.generation.llm import OllamaClient
    from rag_assistant.retrieval.reranker import CrossEncoderReranker
    from rag_assistant.retrieval.retriever import Retriever
    from rag_assistant.retrieval.vector_store import QdrantStore

    settings = get_settings()

    logger.info("Loading dataset from %s", args.dataset)
    items = load_dataset(Path(args.dataset))
    logger.info("Loaded %d evaluation items", len(items))

    logger.info("Initialising services...")
    embedder = CodeEmbedder.from_settings(settings)
    vector_store = QdrantStore.from_settings(settings)
    reranker = CrossEncoderReranker(settings.reranker_model_name)
    retriever = Retriever(embedder, vector_store, reranker, settings)
    ollama = OllamaClient.from_settings(settings)

    runner = EvalRunner(
        retriever=retriever,
        embedder=embedder,
        ollama_client=ollama,
        concurrency=args.concurrency,
        skip_generation_metrics=args.skip_generation,
    )

    logger.info("Running evaluation (concurrency=%d)...", args.concurrency)
    report = await runner.run(items)

    # Print summary
    print("\n=== Evaluation Report ===")
    print(f"Total items:      {report.total_items}")
    print(f"Failed items:     {report.failed_items}")
    print(f"Recall@1:         {report.mean_recall_at_1:.3f}")
    print(f"Recall@5:         {report.mean_recall_at_5:.3f}")
    print(f"Recall@10:        {report.mean_recall_at_10:.3f}")
    print(f"MRR:              {report.mean_mrr:.3f}")
    if not args.skip_generation:
        print(f"Faithfulness:     {report.mean_faithfulness:.3f}")
        print(f"Correctness:      {report.mean_correctness:.3f}")

    # Optionally write full report to JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(dataclasses.asdict(report), f, indent=2)
        logger.info("Full report written to %s", args.output)

    await vector_store.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG evaluation pipeline")
    parser.add_argument(
        "--dataset",
        default="eval/dataset.jsonl",
        help="Path to JSONL evaluation dataset (default: eval/dataset.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write full JSON report",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent evaluation items (default: 4)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip faithfulness and correctness metrics (no Ollama needed)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
