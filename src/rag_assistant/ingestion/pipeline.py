from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rag_assistant.config import Settings
from rag_assistant.embedding.embedder import CodeEmbedder
from rag_assistant.retrieval.vector_store import QdrantStore

from .chunker import CodeChunker
from .cloner import RepoCloner
from .tracker import SQLiteTracker
from .walker import FileWalker

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Summary of a completed ingestion run."""
    repo_url: str
    files_new: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    files_unchanged: int = 0
    chunks_upserted: int = 0
    chunks_deleted: int = 0

    @property
    def files_indexed(self) -> int:
        return self.files_new + self.files_modified


@dataclass
class PipelineProgress:
    """Emitted periodically during ingestion for job-status updates."""
    files_processed: int = 0
    files_total: int = 0
    chunks_upserted: int = 0


# Callback type: receives a PipelineProgress snapshot
ProgressCallback = Callable[[PipelineProgress], None]


class IngestionPipeline:
    """Orchestrates the full ingestion flow for a single repository.

    Stages:
      1. Clone or pull the repo via gitpython.
      2. Walk the repo tree to get {file_path: (hash, mtime)}.
      3. Diff against SQLite to find new / modified / deleted files.
      4. Delete Qdrant points for modified and deleted files.
      5. Chunk + embed new and modified files.
      6. Upsert embeddings to Qdrant.
      7. Update the SQLite tracker.
    """

    # How often (in files) to fire the progress callback
    PROGRESS_EVERY = 10

    def __init__(
        self,
        settings: Settings,
        vector_store: QdrantStore,
        embedder: CodeEmbedder | None = None,
        tracker: SQLiteTracker | None = None,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._embedder = embedder or CodeEmbedder.from_settings(settings)
        self._tracker = tracker or SQLiteTracker(settings.sqlite_db_path)
        self._cloner = RepoCloner(settings.repos_base_dir)
        self._walker = FileWalker()
        self._chunker = CodeChunker(settings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        repo_url: str,
        branch: str = "main",
        force_reindex: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        """Run the full ingestion pipeline. Returns a PipelineResult summary.

        Args:
            repo_url: GitHub / any git URL to clone or pull.
            branch: Branch to check out on first clone.
            force_reindex: If True, treat every file as new (ignores SQLite state).
            progress_callback: Optional callable invoked every PROGRESS_EVERY files.
        """
        result = PipelineResult(repo_url=repo_url)

        # Stage 1: clone / pull
        logger.info("Cloning or pulling %s", repo_url)
        repo_path = self._cloner.clone_or_pull(repo_url, branch=branch)

        # Stage 2: walk
        logger.info("Walking repo tree at %s", repo_path)
        current_files = self._walker.walk(repo_path)
        logger.info("Found %d indexable files", len(current_files))

        # Stage 3: diff
        if force_reindex:
            from .tracker import RepoDiff
            diff = RepoDiff(
                new_paths=sorted(current_files.keys()),
                modified_paths=[],
                deleted_paths=sorted(self._tracker.get_all_file_paths(repo_url)),
            )
        else:
            diff = self._tracker.diff(repo_url, current_files)

        result.files_new = len(diff.new_paths)
        result.files_modified = len(diff.modified_paths)
        result.files_deleted = len(diff.deleted_paths)
        result.files_unchanged = (
            len(current_files) - result.files_new - result.files_modified
        )

        logger.info(
            "Diff: %d new, %d modified, %d deleted, %d unchanged",
            result.files_new,
            result.files_modified,
            result.files_deleted,
            result.files_unchanged,
        )

        # Stage 4: delete Qdrant points for modified + deleted files
        stale_paths = diff.modified_paths + diff.deleted_paths
        if stale_paths:
            old_chunk_ids = self._tracker.get_chunk_ids_for_paths(repo_url, stale_paths)
            if old_chunk_ids:
                logger.info("Deleting %d stale Qdrant points", len(old_chunk_ids))
                await self._vector_store.delete_points(old_chunk_ids)
                result.chunks_deleted = len(old_chunk_ids)

            for path in stale_paths:
                self._tracker.delete_file_record(repo_url, path)

        # Stage 5–7: chunk → embed → upsert for new + modified files
        to_index = diff.new_paths + diff.modified_paths
        if to_index:
            await self._index_files(
                repo_url=repo_url,
                repo_path=repo_path,
                file_paths=to_index,
                current_files=current_files,
                result=result,
                progress_callback=progress_callback,
            )

        logger.info(
            "Ingestion complete: %d chunks upserted, %d chunks deleted",
            result.chunks_upserted,
            result.chunks_deleted,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _index_files(
        self,
        repo_url: str,
        repo_path: Path,
        file_paths: list[str],
        current_files: dict[str, tuple[str, float]],
        result: PipelineResult,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """Chunk, embed, upsert, and track each file in file_paths."""
        progress = PipelineProgress(files_total=len(file_paths))

        for rel_path in file_paths:
            abs_path = repo_path / rel_path
            try:
                source = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Skipping unreadable file %s: %s", rel_path, exc)
                continue

            chunks = self._chunker.chunk_file(source, rel_path, repo_url)
            if not chunks:
                # Empty file or no extractable chunks — still record it
                file_hash, mtime = current_files[rel_path]
                self._tracker.upsert_file_record(repo_url, rel_path, file_hash, mtime, [])
                progress.files_processed += 1
                self._maybe_fire_progress(progress, progress_callback)
                continue

            # Embed all chunks for this file in one batch
            texts = [c.content for c in chunks]
            embeddings = self._embedder.embed_documents(texts)

            await self._vector_store.upsert_chunks(chunks, embeddings)

            chunk_ids = [c.id for c in chunks]
            file_hash, mtime = current_files[rel_path]
            self._tracker.upsert_file_record(repo_url, rel_path, file_hash, mtime, chunk_ids)

            result.chunks_upserted += len(chunks)
            progress.chunks_upserted += len(chunks)
            progress.files_processed += 1

            logger.debug("Indexed %s: %d chunks", rel_path, len(chunks))
            self._maybe_fire_progress(progress, progress_callback)

    @staticmethod
    def _maybe_fire_progress(
        progress: PipelineProgress,
        callback: ProgressCallback | None,
    ) -> None:
        if callback is None:
            return
        if progress.files_processed % IngestionPipeline.PROGRESS_EVERY == 0:
            callback(progress)
