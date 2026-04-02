from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_assistant.config import Settings


class CodeEmbedder:
    """Wraps nomic-ai/nomic-embed-code for retrieval-optimized code embeddings.

    nomic-embed-code requires instruction prefixes to produce retrieval-quality
    embeddings. Queries must use QUERY_PREFIX; indexed documents must use
    DOCUMENT_PREFIX. Skipping the prefixes noticeably degrades retrieval quality.
    """

    QUERY_PREFIX = "search_query: "
    DOCUMENT_PREFIX = "search_document: "

    def __init__(self, model_name: str, device: str, batch_size: int) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
                trust_remote_code=True,
            )
        return self._model

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (768,)."""
        return self.embed_texts([f"{self.QUERY_PREFIX}{query}"])[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a list of code documents. Returns shape (N, 768)."""
        prefixed = [f"{self.DOCUMENT_PREFIX}{t}" for t in texts]
        return self.embed_texts(prefixed)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts as-is (no prefix injection). Returns shape (N, 768) float32."""
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.array(embeddings, dtype=np.float32)

    @classmethod
    def from_settings(cls, settings: Settings) -> CodeEmbedder:
        return cls(
            model_name=settings.embed_model_name,
            device=settings.embed_device,
            batch_size=settings.embed_batch_size,
        )
