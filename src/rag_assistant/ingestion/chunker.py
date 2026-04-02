from __future__ import annotations

import os

from rag_assistant.config import Settings
from rag_assistant.models.chunk import CodeChunk

from .ast_chunker import EXTENSION_TO_LANGUAGE, ASTChunker
from .sliding_chunker import SlidingWindowChunker


class CodeChunker:
    """Dispatches to ASTChunker for supported languages, SlidingWindowChunker otherwise."""

    def __init__(self, settings: Settings) -> None:
        self._ast = ASTChunker(
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        self._sliding = SlidingWindowChunker(
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )

    def chunk_file(self, source: str, file_path: str, repo_url: str) -> list[CodeChunk]:
        ext = os.path.splitext(file_path)[1].lower()
        language = EXTENSION_TO_LANGUAGE.get(ext, "unknown")

        if language != "unknown":
            return self._ast.chunk(source, file_path, language, repo_url)

        return self._sliding.chunk(source, file_path, language, repo_url)
