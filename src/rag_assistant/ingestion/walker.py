from __future__ import annotations

import hashlib
from pathlib import Path

# Directories that are never useful to index
SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", "target", "vendor",
    ".mypy_cache", ".ruff_cache", ".pytest_cache",
})

# File extensions to index (AST-supported + common code/config types)
CODE_EXTENSIONS: frozenset[str] = frozenset({
    # AST-aware (Python, JS, TS, Go, Rust)
    ".py", ".js", ".mjs", ".jsx", ".ts", ".tsx", ".go", ".rs",
    # Other languages (sliding-window)
    ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php",
    # Config / data
    ".sh", ".bash", ".zsh",
    ".yaml", ".yml", ".toml", ".json",
    ".sql",
})


class FileWalker:
    """Walks a repo directory and returns indexable files with their hashes."""

    def walk(self, repo_dir: Path) -> dict[str, tuple[str, float]]:
        """Return {relative_path: (sha256_hash, mtime)} for all indexable files.

        Skips hidden directories, dependency directories, and non-code extensions.
        """
        result: dict[str, tuple[str, float]] = {}

        for path in sorted(repo_dir.rglob("*")):
            if not path.is_file():
                continue
            if self._in_skipped_dir(path, repo_dir):
                continue
            if path.suffix.lower() not in CODE_EXTENSIONS:
                continue

            rel = str(path.relative_to(repo_dir))
            file_hash = self._hash_file(path)
            mtime = path.stat().st_mtime
            result[rel] = (file_hash, mtime)

        return result

    @staticmethod
    def _in_skipped_dir(path: Path, base: Path) -> bool:
        """Return True if any component of the path (relative to base) is a skip dir."""
        try:
            rel = path.relative_to(base)
        except ValueError:
            return False
        return any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1])

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()
