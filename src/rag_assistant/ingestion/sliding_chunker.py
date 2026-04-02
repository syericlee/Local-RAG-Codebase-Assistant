from __future__ import annotations

import hashlib

import tiktoken

from rag_assistant.models.chunk import ChunkType, CodeChunk

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text, disallowed_special=()))


class SlidingWindowChunker:
    """Splits source code into overlapping line-based chunks.

    Used as a fallback for file types not supported by the AST chunker.
    Chunks always contain at least one line, even if that line alone exceeds
    max_tokens (oversized lines are never split mid-line).
    """

    def __init__(self, max_tokens: int, overlap_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(
        self, source: str, file_path: str, language: str, repo_url: str
    ) -> list[CodeChunk]:
        lines = source.splitlines()
        if not lines:
            return []

        file_hash = hashlib.sha256(source.encode()).hexdigest()
        chunks: list[CodeChunk] = []
        start_idx = 0  # 0-based index into lines

        while start_idx < len(lines):
            # Accumulate lines until we hit the token budget.
            # Always include at least one line even if it exceeds max_tokens.
            token_count = 0
            end_idx = start_idx

            while end_idx < len(lines):
                line_tokens = _count_tokens(lines[end_idx])
                if token_count + line_tokens > self.max_tokens and end_idx > start_idx:
                    break
                token_count += line_tokens
                end_idx += 1

            # lines[start_idx:end_idx] — end_idx is exclusive.
            content = "\n".join(lines[start_idx:end_idx])
            chunks.append(
                CodeChunk(
                    id=CodeChunk.make_id(repo_url, file_path, start_idx + 1),
                    content=content,
                    file_path=file_path,
                    start_line=start_idx + 1,   # convert to 1-based
                    end_line=end_idx,            # end_idx is exclusive 0-based = inclusive 1-based
                    language=language,
                    chunk_type=ChunkType.BLOCK,
                    symbol_name=None,
                    repo_url=repo_url,
                    file_hash=file_hash,
                )
            )

            if end_idx >= len(lines):
                break

            # Find the next start by walking back overlap_tokens from end_idx.
            new_start = end_idx
            if self.overlap_tokens > 0:
                accumulated = 0
                while new_start > start_idx + 1:
                    new_start -= 1
                    accumulated += _count_tokens(lines[new_start])
                    if accumulated >= self.overlap_tokens:
                        break

            # Guard against no progress (e.g. overlap >= chunk size).
            if new_start <= start_idx:
                new_start = start_idx + 1

            start_idx = new_start

        return chunks
