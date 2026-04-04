from __future__ import annotations

from collections.abc import AsyncIterator

from ollama import AsyncClient

from rag_assistant.config import Settings


class OllamaClient:
    """Async wrapper around the Ollama chat API.

    Supports both full-response and token-streaming modes.
    """

    def __init__(self, base_url: str, model: str, timeout: int) -> None:
        self._model = model
        self._client = AsyncClient(host=base_url, timeout=timeout)

    @classmethod
    def from_settings(cls, settings: Settings) -> OllamaClient:
        return cls(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout=settings.ollama_timeout_seconds,
        )

    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Send messages to Ollama and return the full answer as a string.

        Args:
            messages: List of {role, content} dicts from PromptBuilder.

        Returns:
            The assistant's reply as a plain string.
        """
        response = await self._client.chat(
            model=self._model,
            messages=messages,
            stream=False,
        )
        return response.message.content

    async def stream_generate(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Stream the answer token by token.

        Yields each text fragment as Ollama produces it. The caller can
        forward these directly to the client as SSE events.

        Args:
            messages: List of {role, content} dicts from PromptBuilder.

        Yields:
            Successive text fragments from the model.
        """
        async for chunk in await self._client.chat(
            model=self._model,
            messages=messages,
            stream=True,
        ):
            token = chunk.message.content
            if token:
                yield token
