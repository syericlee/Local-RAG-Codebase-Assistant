from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class FileRecord:
    repo_url: str
    file_path: str
    file_hash: str
    mtime: float
    chunk_ids: list[str]
    indexed_at: datetime


@dataclass
class RepoDiff:
    """Result of comparing current repo state against the tracker database."""
    new_paths: list[str] = field(default_factory=list)       # never indexed
    modified_paths: list[str] = field(default_factory=list)  # content changed
    deleted_paths: list[str] = field(default_factory=list)   # removed from disk


_SCHEMA = """
CREATE TABLE IF NOT EXISTS indexed_files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_url    TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    file_hash   TEXT NOT NULL,
    mtime       REAL NOT NULL,
    chunk_ids   TEXT NOT NULL,
    indexed_at  TEXT NOT NULL,
    UNIQUE(repo_url, file_path)
);
CREATE INDEX IF NOT EXISTS idx_indexed_files_repo ON indexed_files(repo_url);
"""


class SQLiteTracker:
    """Tracks which files have been indexed and what chunks they produced.

    Uses SQLite so state survives process restarts. Pass db_path=':memory:'
    for tests — each instance gets a fresh in-memory database.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_file_record(self, repo_url: str, file_path: str) -> FileRecord | None:
        row = self._conn.execute(
            "SELECT * FROM indexed_files WHERE repo_url = ? AND file_path = ?",
            (repo_url, file_path),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_all_file_paths(self, repo_url: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT file_path FROM indexed_files WHERE repo_url = ?",
            (repo_url,),
        ).fetchall()
        return {row["file_path"] for row in rows}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_file_record(
        self,
        repo_url: str,
        file_path: str,
        file_hash: str,
        mtime: float,
        chunk_ids: list[str],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO indexed_files
                    (repo_url, file_path, file_hash, mtime, chunk_ids, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_url, file_path) DO UPDATE SET
                    file_hash  = excluded.file_hash,
                    mtime      = excluded.mtime,
                    chunk_ids  = excluded.chunk_ids,
                    indexed_at = excluded.indexed_at
                """,
                (repo_url, file_path, file_hash, mtime, json.dumps(chunk_ids), now),
            )

    def delete_file_record(self, repo_url: str, file_path: str) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM indexed_files WHERE repo_url = ? AND file_path = ?",
                (repo_url, file_path),
            )

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def diff(
        self,
        repo_url: str,
        current_files: dict[str, tuple[str, float]],
    ) -> RepoDiff:
        """Compare current repo state against the database.

        Args:
            repo_url: The repo being indexed.
            current_files: Mapping of file_path → (file_hash, mtime) for every
                file currently on disk.

        Returns:
            RepoDiff with new, modified, and deleted file paths.
        """
        known_paths = self.get_all_file_paths(repo_url)
        current_paths = set(current_files)

        result = RepoDiff()

        # Files on disk but not in DB → new
        result.new_paths = sorted(current_paths - known_paths)

        # Files in DB but not on disk → deleted
        result.deleted_paths = sorted(known_paths - current_paths)

        # Files in both — check if content changed
        for path in sorted(current_paths & known_paths):
            current_hash, _ = current_files[path]
            record = self.get_file_record(repo_url, path)
            if record is not None and record.file_hash != current_hash:
                result.modified_paths.append(path)

        return result

    def get_chunk_ids_for_paths(
        self, repo_url: str, file_paths: list[str]
    ) -> list[str]:
        """Return all chunk IDs stored for the given file paths."""
        chunk_ids: list[str] = []
        for path in file_paths:
            record = self.get_file_record(repo_url, path)
            if record:
                chunk_ids.extend(record.chunk_ids)
        return chunk_ids

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> FileRecord:
        return FileRecord(
            repo_url=row["repo_url"],
            file_path=row["file_path"],
            file_hash=row["file_hash"],
            mtime=row["mtime"],
            chunk_ids=json.loads(row["chunk_ids"]),
            indexed_at=datetime.fromisoformat(row["indexed_at"]),
        )
