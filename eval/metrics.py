from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Retrieval metrics (pure functions, no models needed)
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of relevant chunks found in the top-k retrieved results.

    Returns 0.0 if relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & set(relevant_ids)) / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean Reciprocal Rank for a single query.

    Returns 1/rank of the first relevant chunk, or 0 if none found.
    """
    relevant_set = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Faithfulness (NLI model)
# ---------------------------------------------------------------------------

_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
_nli_model = None  # lazy singleton


def _get_nli_model():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder(_NLI_MODEL_NAME)
    return _nli_model


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!', '?' boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def faithfulness(answer: str, context: str) -> float:
    """Fraction of answer sentences entailed by the retrieved context.

    Uses an NLI model: a sentence is 'entailed' if the model's entailment
    score (label index 1) is the highest of the three classes
    (contradiction=0, entailment=1, neutral=2).

    Returns 1.0 if the answer has no sentences.
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return 1.0

    model = _get_nli_model()
    pairs = [(context, sentence) for sentence in sentences]

    # scores shape: (N, 3) — [contradiction, entailment, neutral]
    scores = model.predict(pairs, apply_softmax=True)
    entailed = sum(1 for s in scores if int(np.argmax(s)) == 1)
    return entailed / len(sentences)


# ---------------------------------------------------------------------------
# Correctness (LLM-as-judge)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an impartial judge evaluating the quality of an AI-generated answer.

Question: {question}

Ground truth answer: {ground_truth}

Generated answer: {answer}

Rate how well the generated answer matches the ground truth on a scale from 0.0 to 1.0:
- 1.0: completely correct and complete
- 0.5: partially correct, missing important details
- 0.0: incorrect or completely off-topic

Respond with ONLY a single decimal number between 0.0 and 1.0. No explanation.\
"""


async def correctness(
    question: str,
    answer: str,
    ground_truth: str,
    ollama_client,
) -> float:
    """LLM-as-judge correctness score in [0, 1].

    Sends a scoring prompt to Ollama and parses the numeric response.
    Returns 0.0 if the response cannot be parsed.
    """
    prompt = _JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        raw = await ollama_client.generate(messages)
        match = re.search(r"(?<![-\d])([01](?:\.\d+)?)(?!\d)", raw.strip())
        if match:
            return min(1.0, max(0.0, float(match.group(1))))
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Per-item result
# ---------------------------------------------------------------------------

@dataclass
class ItemResult:
    id: str
    question: str
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    faithfulness: float = 0.0
    correctness: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    """Aggregate metrics across all evaluated items."""
    total_items: int = 0
    failed_items: int = 0

    mean_recall_at_1: float = 0.0
    mean_recall_at_5: float = 0.0
    mean_recall_at_10: float = 0.0
    mean_mrr: float = 0.0
    mean_faithfulness: float = 0.0
    mean_correctness: float = 0.0

    item_results: list[ItemResult] = field(default_factory=list)

    @classmethod
    def from_items(cls, items: list[ItemResult]) -> EvalReport:
        successful = [it for it in items if it.error is None]
        n = len(successful)

        def mean(values: list[float]) -> float:
            return sum(values) / n if n else 0.0

        return cls(
            total_items=len(items),
            failed_items=len(items) - n,
            mean_recall_at_1=mean([it.recall_at_1 for it in successful]),
            mean_recall_at_5=mean([it.recall_at_5 for it in successful]),
            mean_recall_at_10=mean([it.recall_at_10 for it in successful]),
            mean_mrr=mean([it.mrr for it in successful]),
            mean_faithfulness=mean([it.faithfulness for it in successful]),
            mean_correctness=mean([it.correctness for it in successful]),
            item_results=items,
        )
