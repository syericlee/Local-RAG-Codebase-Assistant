from unittest.mock import MagicMock

import numpy as np
import pytest

from rag_assistant.config import Settings
from rag_assistant.embedding.embedder import CodeEmbedder


@pytest.fixture
def mock_model() -> MagicMock:
    """Mock SentenceTransformer that returns random 768-dim float32 embeddings."""
    model = MagicMock()
    model.encode.side_effect = lambda texts, **kwargs: np.random.rand(
        len(texts), 768
    ).astype(np.float32)
    return model


@pytest.fixture
def embedder(mock_model: MagicMock) -> CodeEmbedder:
    e = CodeEmbedder(model_name="nomic-ai/nomic-embed-code", device="cpu", batch_size=32)
    e._model = mock_model
    return e


class TestCodeEmbedder:
    def test_model_loaded_lazily(self) -> None:
        e = CodeEmbedder(model_name="nomic-ai/nomic-embed-code", device="cpu", batch_size=32)
        assert e._model is None

    def test_embed_query_shape(self, embedder: CodeEmbedder) -> None:
        result = embedder.embed_query("how does auth work?")
        assert result.shape == (768,)

    def test_embed_query_injects_prefix(self, embedder: CodeEmbedder, mock_model: MagicMock) -> None:
        embedder.embed_query("how does auth work?")
        sent = mock_model.encode.call_args[0][0]
        assert sent == ["search_query: how does auth work?"]

    def test_embed_documents_shape(self, embedder: CodeEmbedder) -> None:
        texts = ["def foo(): pass", "class Bar: pass", "x = 1"]
        result = embedder.embed_documents(texts)
        assert result.shape == (3, 768)

    def test_embed_documents_injects_prefix(self, embedder: CodeEmbedder, mock_model: MagicMock) -> None:
        embedder.embed_documents(["def foo(): pass"])
        sent = mock_model.encode.call_args[0][0]
        assert sent == ["search_document: def foo(): pass"]

    def test_embed_documents_prefixes_all(self, embedder: CodeEmbedder, mock_model: MagicMock) -> None:
        texts = ["chunk a", "chunk b", "chunk c"]
        embedder.embed_documents(texts)
        sent = mock_model.encode.call_args[0][0]
        assert all(s.startswith("search_document: ") for s in sent)
        assert len(sent) == 3

    def test_embed_texts_no_prefix(self, embedder: CodeEmbedder, mock_model: MagicMock) -> None:
        embedder.embed_texts(["raw text"])
        sent = mock_model.encode.call_args[0][0]
        assert sent == ["raw text"]

    def test_output_dtype_is_float32(self, embedder: CodeEmbedder) -> None:
        result = embedder.embed_query("test")
        assert result.dtype == np.float32

    def test_embed_documents_float32(self, embedder: CodeEmbedder) -> None:
        result = embedder.embed_documents(["def foo(): pass"])
        assert result.dtype == np.float32

    def test_embed_large_batch(self, embedder: CodeEmbedder) -> None:
        texts = [f"def func_{i}(): pass" for i in range(100)]
        result = embedder.embed_documents(texts)
        assert result.shape == (100, 768)

    def test_from_settings(self, settings: Settings) -> None:
        e = CodeEmbedder.from_settings(settings)
        assert e._model_name == settings.embed_model_name
        assert e._device == settings.embed_device
        assert e._batch_size == settings.embed_batch_size

    def test_model_reused_across_calls(self, embedder: CodeEmbedder, mock_model: MagicMock) -> None:
        embedder.embed_query("first")
        embedder.embed_query("second")
        assert embedder._model is mock_model  # same instance, not re-initialized

    def test_query_and_document_prefixes_differ(self, embedder: CodeEmbedder, mock_model: MagicMock) -> None:
        embedder.embed_query("a question")
        query_sent = mock_model.encode.call_args[0][0][0]

        embedder.embed_documents(["some code"])
        doc_sent = mock_model.encode.call_args[0][0][0]

        assert query_sent.startswith("search_query: ")
        assert doc_sent.startswith("search_document: ")
        assert query_sent != doc_sent
