"""Unit tests for evaluation metrics (no models, no network)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from eval.metrics import (
    EvalReport,
    ItemResult,
    correctness,
    faithfulness,
    mrr,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_all_relevant_found(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=5) == 1.0

    def test_none_relevant_found(self) -> None:
        assert recall_at_k(["x", "y"], ["a", "b"], k=5) == 0.0

    def test_partial_recall(self) -> None:
        assert recall_at_k(["a", "x", "y"], ["a", "b"], k=5) == 0.5

    def test_k_limits_window(self) -> None:
        # relevant chunk is at position 3, outside k=2
        assert recall_at_k(["x", "y", "a"], ["a"], k=2) == 0.0

    def test_k_includes_relevant(self) -> None:
        assert recall_at_k(["x", "y", "a"], ["a"], k=3) == 1.0

    def test_empty_relevant_returns_zero(self) -> None:
        assert recall_at_k(["a", "b"], [], k=5) == 0.0

    def test_empty_retrieved_returns_zero(self) -> None:
        assert recall_at_k([], ["a"], k=5) == 0.0


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------

class TestMRR:
    def test_first_result_relevant(self) -> None:
        assert mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_second_result_relevant(self) -> None:
        assert mrr(["x", "a", "c"], ["a"]) == pytest.approx(0.5)

    def test_fifth_result_relevant(self) -> None:
        assert mrr(["x", "y", "z", "w", "a"], ["a"]) == pytest.approx(0.2)

    def test_not_found_returns_zero(self) -> None:
        assert mrr(["x", "y"], ["a"]) == 0.0

    def test_empty_retrieved_returns_zero(self) -> None:
        assert mrr([], ["a"]) == 0.0

    def test_empty_relevant_returns_zero(self) -> None:
        assert mrr(["a", "b"], []) == 0.0

    def test_first_of_multiple_relevant(self) -> None:
        # Both b and c are relevant; b is at rank 2
        assert mrr(["x", "b", "c"], ["b", "c"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Faithfulness (mocked NLI model)
# ---------------------------------------------------------------------------

class TestFaithfulness:
    def test_all_sentences_entailed(self) -> None:
        # entailment label index = 1; argmax([0.05, 0.90, 0.05]) = 1
        scores = np.array([[0.05, 0.90, 0.05], [0.05, 0.90, 0.05]])
        with patch("eval.metrics._get_nli_model") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = scores
            mock_get.return_value = mock_model
            score = faithfulness("Sentence one. Sentence two.", "some context")
        assert score == 1.0

    def test_no_sentences_entailed(self) -> None:
        # contradiction label index = 0; argmax([0.90, 0.05, 0.05]) = 0
        scores = np.array([[0.90, 0.05, 0.05]])
        with patch("eval.metrics._get_nli_model") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = scores
            mock_get.return_value = mock_model
            score = faithfulness("One sentence.", "context")
        assert score == 0.0

    def test_partial_entailment(self) -> None:
        # First entailed, second not
        scores = np.array([[0.05, 0.90, 0.05], [0.90, 0.05, 0.05]])
        with patch("eval.metrics._get_nli_model") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = scores
            mock_get.return_value = mock_model
            score = faithfulness("First sentence. Second sentence.", "context")
        assert score == pytest.approx(0.5)

    def test_empty_answer_returns_one(self) -> None:
        score = faithfulness("", "some context")
        assert score == 1.0


# ---------------------------------------------------------------------------
# Correctness (mocked Ollama)
# ---------------------------------------------------------------------------

class TestCorrectness:
    @pytest.mark.asyncio
    async def test_parses_score_from_response(self) -> None:
        mock_ollama = MagicMock()
        mock_ollama.generate = AsyncMock(return_value="0.8")
        score = await correctness("q", "answer", "ground truth", mock_ollama)
        assert score == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_score_clamped_to_one(self) -> None:
        mock_ollama = MagicMock()
        mock_ollama.generate = AsyncMock(return_value="1.5")
        score = await correctness("q", "answer", "ground truth", mock_ollama)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_score_clamped_to_zero(self) -> None:
        mock_ollama = MagicMock()
        mock_ollama.generate = AsyncMock(return_value="-0.3")
        score = await correctness("q", "answer", "ground truth", mock_ollama)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_unparseable_returns_zero(self) -> None:
        mock_ollama = MagicMock()
        mock_ollama.generate = AsyncMock(return_value="I cannot score this.")
        score = await correctness("q", "answer", "ground truth", mock_ollama)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_exception_returns_zero(self) -> None:
        mock_ollama = MagicMock()
        mock_ollama.generate = AsyncMock(side_effect=Exception("connection error"))
        score = await correctness("q", "answer", "ground truth", mock_ollama)
        assert score == 0.0


# ---------------------------------------------------------------------------
# EvalReport aggregation
# ---------------------------------------------------------------------------

class TestEvalReport:
    def test_means_computed_correctly(self) -> None:
        items = [
            ItemResult(id="1", question="q1", recall_at_1=1.0, recall_at_5=1.0,
                       recall_at_10=1.0, mrr=1.0, faithfulness=0.8, correctness=0.9),
            ItemResult(id="2", question="q2", recall_at_1=0.0, recall_at_5=0.5,
                       recall_at_10=1.0, mrr=0.5, faithfulness=0.6, correctness=0.7),
        ]
        report = EvalReport.from_items(items)
        assert report.total_items == 2
        assert report.failed_items == 0
        assert report.mean_recall_at_1 == pytest.approx(0.5)
        assert report.mean_recall_at_5 == pytest.approx(0.75)
        assert report.mean_mrr == pytest.approx(0.75)
        assert report.mean_faithfulness == pytest.approx(0.7)
        assert report.mean_correctness == pytest.approx(0.8)

    def test_failed_items_excluded_from_means(self) -> None:
        items = [
            ItemResult(id="1", question="q1", recall_at_1=1.0, mrr=1.0),
            ItemResult(id="2", question="q2", error="timeout"),
        ]
        report = EvalReport.from_items(items)
        assert report.total_items == 2
        assert report.failed_items == 1
        assert report.mean_recall_at_1 == 1.0

    def test_all_failed_returns_zero_means(self) -> None:
        items = [ItemResult(id="1", question="q", error="error")]
        report = EvalReport.from_items(items)
        assert report.mean_recall_at_1 == 0.0
        assert report.mean_mrr == 0.0

    def test_empty_items(self) -> None:
        report = EvalReport.from_items([])
        assert report.total_items == 0
        assert report.mean_mrr == 0.0
