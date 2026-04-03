from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_assistant.generation.llm import OllamaClient

MESSAGES = [
    {"role": "system", "content": "You are a code assistant."},
    {"role": "user", "content": "Question: How does login work?"},
]


def _make_client() -> OllamaClient:
    return OllamaClient(
        base_url="http://localhost:11434",
        model="deepseek-coder:6.7b",
        timeout=120,
    )


def _make_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    response = MagicMock()
    response.message = msg
    return response


def _make_chunk(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    chunk = MagicMock()
    chunk.message = msg
    return chunk


class TestOllamaClientGenerate:
    @pytest.mark.asyncio
    async def test_returns_answer_string(self) -> None:
        client = _make_client()
        mock_response = _make_response("Authentication uses bcrypt.")

        with patch.object(client._client, "chat", new=AsyncMock(return_value=mock_response)):
            result = await client.generate(MESSAGES)

        assert result == "Authentication uses bcrypt."

    @pytest.mark.asyncio
    async def test_passes_messages_to_ollama(self) -> None:
        client = _make_client()
        mock_chat = AsyncMock(return_value=_make_response("answer"))

        with patch.object(client._client, "chat", new=mock_chat):
            await client.generate(MESSAGES)

        mock_chat.assert_awaited_once()
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["messages"] == MESSAGES
        assert call_kwargs["stream"] is False

    @pytest.mark.asyncio
    async def test_uses_configured_model(self) -> None:
        client = _make_client()
        mock_chat = AsyncMock(return_value=_make_response("answer"))

        with patch.object(client._client, "chat", new=mock_chat):
            await client.generate(MESSAGES)

        assert mock_chat.call_args.kwargs["model"] == "deepseek-coder:6.7b"

    @pytest.mark.asyncio
    async def test_empty_response_returned_as_is(self) -> None:
        client = _make_client()
        with patch.object(client._client, "chat", new=AsyncMock(return_value=_make_response(""))):
            result = await client.generate(MESSAGES)
        assert result == ""


class TestOllamaClientStream:
    async def _make_async_iter(self, items):
        for item in items:
            yield item

    @pytest.mark.asyncio
    async def test_yields_tokens(self) -> None:
        client = _make_client()
        chunks = [_make_chunk("Auth"), _make_chunk(" uses"), _make_chunk(" bcrypt.")]

        with patch.object(
            client._client,
            "chat",
            new=AsyncMock(return_value=self._make_async_iter(chunks)),
        ):
            tokens = [t async for t in client.stream_generate(MESSAGES)]

        assert tokens == ["Auth", " uses", " bcrypt."]

    @pytest.mark.asyncio
    async def test_empty_tokens_skipped(self) -> None:
        client = _make_client()
        chunks = [_make_chunk("Hello"), _make_chunk(""), _make_chunk(" world")]

        with patch.object(
            client._client,
            "chat",
            new=AsyncMock(return_value=self._make_async_iter(chunks)),
        ):
            tokens = [t async for t in client.stream_generate(MESSAGES)]

        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_passes_stream_true(self) -> None:
        client = _make_client()
        mock_chat = AsyncMock(return_value=self._make_async_iter([]))

        with patch.object(client._client, "chat", new=mock_chat):
            async for _ in client.stream_generate(MESSAGES):
                pass

        assert mock_chat.call_args.kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_from_settings(self, settings) -> None:
        client = OllamaClient.from_settings(settings)
        assert client._model == settings.ollama_model
