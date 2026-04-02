from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Optional

import tiktoken
from tree_sitter import Language, Node, Parser

from rag_assistant.models.chunk import ChunkType, CodeChunk

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# File extension → language name
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
}

# AST node types to extract as chunks, keyed by language
CHUNK_NODE_TYPES: dict[str, dict[str, ChunkType]] = {
    "python": {
        "function_definition": ChunkType.FUNCTION,
        "class_definition": ChunkType.CLASS,
        "decorated_definition": ChunkType.FUNCTION,
    },
    "javascript": {
        "function_declaration": ChunkType.FUNCTION,
        "method_definition": ChunkType.METHOD,
        "class_declaration": ChunkType.CLASS,
    },
    "typescript": {
        "function_declaration": ChunkType.FUNCTION,
        "method_definition": ChunkType.METHOD,
        "class_declaration": ChunkType.CLASS,
    },
    "go": {
        "function_declaration": ChunkType.FUNCTION,
        "method_declaration": ChunkType.METHOD,
    },
    "rust": {
        "function_item": ChunkType.FUNCTION,
        "impl_item": ChunkType.CLASS,
    },
}


@lru_cache(maxsize=8)
def _get_ts_language(language: str) -> Language:
    """Load and cache a tree-sitter Language. Imports are deferred so unused
    language packages don't slow down module import."""
    if language == "python":
        import tree_sitter_python as m
        return Language(m.language())
    if language == "javascript":
        import tree_sitter_javascript as m
        return Language(m.language())
    if language == "typescript":
        import tree_sitter_typescript as m
        return Language(m.language_typescript())
    if language == "go":
        import tree_sitter_go as m
        return Language(m.language())
    if language == "rust":
        import tree_sitter_rust as m
        return Language(m.language())
    raise ValueError(f"Unsupported language for AST chunking: {language!r}")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text, disallowed_special=()))


def _get_symbol_name(node: Node, source_bytes: bytes) -> Optional[str]:
    """Return the symbol name for a chunking node, or None if not found."""
    # decorated_definition wraps a function/class — delegate to the inner node
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return _get_symbol_name(child, source_bytes)
        return None

    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
    return None


class ASTChunker:
    """Extracts functions, classes, and methods as chunks using tree-sitter.

    Walk strategy (top-down):
    - When a chunking node fits within max_tokens: emit it, stop recursing.
    - When a chunking node is too large: recurse into its children.
    - If no children produce chunks (e.g. a large flat function): fall back to
      SlidingWindowChunker on that node's content, correcting line numbers.
    """

    def __init__(self, max_tokens: int, overlap_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(
        self, source: str, file_path: str, language: str, repo_url: str
    ) -> list[CodeChunk]:
        ts_lang = _get_ts_language(language)
        parser = Parser(ts_lang)
        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)
        file_hash = hashlib.sha256(source_bytes).hexdigest()

        return self._walk(
            tree.root_node, source_bytes, file_path, language, repo_url, file_hash
        )

    def _walk(
        self,
        node: Node,
        source_bytes: bytes,
        file_path: str,
        language: str,
        repo_url: str,
        file_hash: str,
    ) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        chunk_type = CHUNK_NODE_TYPES.get(language, {}).get(node.type)

        if chunk_type is not None:
            content = source_bytes[node.start_byte:node.end_byte].decode("utf-8")

            if _count_tokens(content) <= self.max_tokens:
                # Node fits — emit it and stop here.
                start_line = node.start_point[0] + 1  # tree-sitter uses 0-based rows
                end_line = node.end_point[0] + 1
                chunks.append(
                    CodeChunk(
                        id=CodeChunk.make_id(repo_url, file_path, start_line),
                        content=content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        language=language,
                        chunk_type=chunk_type,
                        symbol_name=_get_symbol_name(node, source_bytes),
                        repo_url=repo_url,
                        file_hash=file_hash,
                    )
                )
                return chunks

            # Node is too large — try children first.
            child_chunks: list[CodeChunk] = []
            for child in node.children:
                child_chunks.extend(
                    self._walk(child, source_bytes, file_path, language, repo_url, file_hash)
                )
            if child_chunks:
                return child_chunks

            # No children produced chunks (e.g. a large flat function).
            return self._sub_chunk(node, source_bytes, file_path, language, repo_url, file_hash)

        # Not a chunking node — recurse into children.
        for child in node.children:
            chunks.extend(
                self._walk(child, source_bytes, file_path, language, repo_url, file_hash)
            )
        return chunks

    def _sub_chunk(
        self,
        node: Node,
        source_bytes: bytes,
        file_path: str,
        language: str,
        repo_url: str,
        file_hash: str,
    ) -> list[CodeChunk]:
        """Sliding-window fallback for an oversized node with no sub-chunks."""
        from .sliding_chunker import SlidingWindowChunker

        content = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
        line_offset = node.start_point[0]  # 0-based line in the file

        raw = SlidingWindowChunker(self.max_tokens, self.overlap_tokens).chunk(
            content, file_path, language, repo_url
        )

        corrected: list[CodeChunk] = []
        for c in raw:
            abs_start = c.start_line + line_offset
            abs_end = c.end_line + line_offset
            corrected.append(
                CodeChunk(
                    id=CodeChunk.make_id(repo_url, file_path, abs_start),
                    content=c.content,
                    file_path=file_path,
                    start_line=abs_start,
                    end_line=abs_end,
                    language=language,
                    chunk_type=ChunkType.BLOCK,
                    symbol_name=None,
                    repo_url=repo_url,
                    file_hash=file_hash,
                )
            )
        return corrected
