from __future__ import annotations

import hashlib
from enum import Enum

from pydantic import BaseModel


class ChunkType(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    BLOCK = "block"
    IMPORT_BLOCK = "import_block"


class CodeChunk(BaseModel):
    id: str
    content: str
    file_path: str      # relative path within repo
    start_line: int
    end_line: int
    language: str       # "python", "javascript", "typescript", "go", "rust", "unknown"
    chunk_type: ChunkType
    symbol_name: str | None = None   # function/class name if AST-extracted
    repo_url: str
    file_hash: str      # sha256 of the source file at index time

    @staticmethod
    def make_id(repo_url: str, file_path: str, start_line: int) -> str:
        """Deterministic chunk ID: sha256(repo_url:file_path:start_line)[:16]."""
        key = f"{repo_url}:{file_path}:{start_line}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
