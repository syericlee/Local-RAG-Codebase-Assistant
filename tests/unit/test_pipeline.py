"""Unit tests for IngestionPipeline.

All external I/O (git, Qdrant, sentence-transformers) is mocked so these
tests run without any live services or model downloads.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from rag_assistant.ingestion.pipeline import IngestionPipeline, PipelineProgress
from rag_assistant.ingestion.tracker import SQLiteTracker
from rag_assistant.models.chunk import ChunkType, CodeChunk

REPO_URL = "https://github.com/foo/bar"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(file_path: str = "main.py", start: int = 1) -> CodeChunk:
    return CodeChunk(
        id="abcdef1234567890",
        content="def foo(): pass",
        file_path=file_path,
        start_line=start,
        end_line=start + 2,
        language="python",
        chunk_type=ChunkType.FUNCTION,
        symbol_name="foo",
        repo_url=REPO_URL,
        file_hash="deadbeef" * 8,
    )


def _make_pipeline(settings, tracker: SQLiteTracker) -> tuple[IngestionPipeline, AsyncMock]:
    """Return (pipeline, mock_vector_store)."""
    mock_vs = AsyncMock()
    mock_vs.upsert_chunks = AsyncMock()
    mock_vs.delete_points = AsyncMock()

    mock_embedder = MagicMock()
    mock_embedder.embed_documents.return_value = np.zeros((1, 768), dtype=np.float32)

    pipeline = IngestionPipeline(
        settings=settings,
        vector_store=mock_vs,
        embedder=mock_embedder,
        tracker=tracker,
    )
    return pipeline, mock_vs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineNewFile:
    @pytest.fixture
    def tracker(self) -> SQLiteTracker:
        t = SQLiteTracker(":memory:")
        yield t
        t.close()

    @pytest.mark.asyncio
    async def test_new_file_is_indexed(self, settings, tracker: SQLiteTracker, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("def foo(): pass\n")
        pipeline, mock_vs = _make_pipeline(settings, tracker)

        chunk = _make_chunk()
        pipeline._chunker = MagicMock()
        pipeline._chunker.chunk_file.return_value = [chunk]
        pipeline._cloner = MagicMock()
        pipeline._cloner.clone_or_pull.return_value = tmp_path

        result = await pipeline.run(REPO_URL)

        assert result.files_new == 1
        assert result.files_modified == 0
        assert result.files_deleted == 0
        assert result.chunks_upserted == 1
        mock_vs.upsert_chunks.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tracker_updated_after_index(self, settings, tracker: SQLiteTracker, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("def foo(): pass\n")
        pipeline, _ = _make_pipeline(settings, tracker)

        chunk = _make_chunk()
        pipeline._chunker = MagicMock()
        pipeline._chunker.chunk_file.return_value = [chunk]
        pipeline._cloner = MagicMock()
        pipeline._cloner.clone_or_pull.return_value = tmp_path

        await pipeline.run(REPO_URL)

        record = tracker.get_file_record(REPO_URL, "main.py")
        assert record is not None
        assert record.chunk_ids == [chunk.id]


class TestPipelineUnchangedFile:
    @pytest.fixture
    def tracker(self) -> SQLiteTracker:
        t = SQLiteTracker(":memory:")
        yield t
        t.close()

    @pytest.mark.asyncio
    async def test_unchanged_file_not_reindexed(self, settings, tracker: SQLiteTracker, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("def foo(): pass\n")
        pipeline, mock_vs = _make_pipeline(settings, tracker)
        pipeline._cloner = MagicMock()
        pipeline._cloner.clone_or_pull.return_value = tmp_path

        # Index once
        chunk = _make_chunk()
        pipeline._chunker = MagicMock()
        pipeline._chunker.chunk_file.return_value = [chunk]
        await pipeline.run(REPO_URL)

        # Run again — same file, same hash
        mock_vs.upsert_chunks.reset_mock()
        result = await pipeline.run(REPO_URL)

        assert result.files_unchanged == 1
        assert result.files_new == 0
        mock_vs.upsert_chunks.assert_not_awaited()


class TestPipelineModifiedFile:
    @pytest.fixture
    def tracker(self) -> SQLiteTracker:
        t = SQLiteTracker(":memory:")
        yield t
        t.close()

    @pytest.mark.asyncio
    async def test_modified_file_deletes_old_points_and_reindexes(
        self, settings, tracker: SQLiteTracker, tmp_path: Path
    ) -> None:
        py_file = tmp_path / "main.py"
        py_file.write_text("x = 1\n")

        pipeline, mock_vs = _make_pipeline(settings, tracker)
        pipeline._cloner = MagicMock()
        pipeline._cloner.clone_or_pull.return_value = tmp_path

        chunk_v1 = _make_chunk()
        pipeline._chunker = MagicMock()
        pipeline._chunker.chunk_file.return_value = [chunk_v1]
        await pipeline.run(REPO_URL)

        # Modify the file (different content → different hash)
        py_file.write_text("x = 2\n")
        chunk_v2 = CodeChunk(
            id="1111111111111111",
            content="x = 2",
            file_path="main.py",
            start_line=1,
            end_line=1,
            language="python",
            chunk_type=ChunkType.BLOCK,
            symbol_name=None,
            repo_url=REPO_URL,
            file_hash="newhashnewhashnewhashnewhashnewhashnewhashnewhashnewhashnewh1234",
        )
        pipeline._chunker.chunk_file.return_value = [chunk_v2]

        mock_vs.delete_points.reset_mock()
        mock_vs.upsert_chunks.reset_mock()
        result = await pipeline.run(REPO_URL)

        assert result.files_modified == 1
        mock_vs.delete_points.assert_awaited_once()
        mock_vs.upsert_chunks.assert_awaited_once()


class TestPipelineDeletedFile:
    @pytest.fixture
    def tracker(self) -> SQLiteTracker:
        t = SQLiteTracker(":memory:")
        yield t
        t.close()

    @pytest.mark.asyncio
    async def test_deleted_file_removes_points_and_record(
        self, settings, tracker: SQLiteTracker, tmp_path: Path
    ) -> None:
        py_file = tmp_path / "main.py"
        py_file.write_text("x = 1\n")

        pipeline, mock_vs = _make_pipeline(settings, tracker)
        pipeline._cloner = MagicMock()
        pipeline._cloner.clone_or_pull.return_value = tmp_path

        chunk = _make_chunk()
        pipeline._chunker = MagicMock()
        pipeline._chunker.chunk_file.return_value = [chunk]
        await pipeline.run(REPO_URL)

        # Delete the file from disk
        py_file.unlink()
        mock_vs.delete_points.reset_mock()
        result = await pipeline.run(REPO_URL)

        assert result.files_deleted == 1
        mock_vs.delete_points.assert_awaited_once_with([chunk.id])
        assert tracker.get_file_record(REPO_URL, "main.py") is None


class TestPipelineForceReindex:
    @pytest.fixture
    def tracker(self) -> SQLiteTracker:
        t = SQLiteTracker(":memory:")
        yield t
        t.close()

    @pytest.mark.asyncio
    async def test_force_reindex_treats_all_as_new(
        self, settings, tracker: SQLiteTracker, tmp_path: Path
    ) -> None:
        (tmp_path / "main.py").write_text("x = 1\n")
        pipeline, mock_vs = _make_pipeline(settings, tracker)
        pipeline._cloner = MagicMock()
        pipeline._cloner.clone_or_pull.return_value = tmp_path

        chunk = _make_chunk()
        pipeline._chunker = MagicMock()
        pipeline._chunker.chunk_file.return_value = [chunk]

        # First run to populate tracker
        await pipeline.run(REPO_URL)
        mock_vs.upsert_chunks.reset_mock()

        # Force reindex — should re-embed even though file unchanged
        result = await pipeline.run(REPO_URL, force_reindex=True)

        assert result.files_new == 1
        mock_vs.upsert_chunks.assert_awaited_once()


class TestPipelineProgress:
    @pytest.fixture
    def tracker(self) -> SQLiteTracker:
        t = SQLiteTracker(":memory:")
        yield t
        t.close()

    @pytest.mark.asyncio
    async def test_progress_callback_fires(
        self, settings, tracker: SQLiteTracker, tmp_path: Path
    ) -> None:
        # Create PROGRESS_EVERY files so callback fires exactly once
        n = IngestionPipeline.PROGRESS_EVERY
        for i in range(n):
            (tmp_path / f"f{i}.py").write_text(f"x = {i}\n")

        pipeline, _ = _make_pipeline(settings, tracker)
        pipeline._cloner = MagicMock()
        pipeline._cloner.clone_or_pull.return_value = tmp_path
        pipeline._chunker = MagicMock()
        pipeline._chunker.chunk_file.return_value = [_make_chunk()]

        received: list[PipelineProgress] = []
        await pipeline.run(REPO_URL, progress_callback=received.append)

        assert len(received) >= 1
        assert received[-1].files_processed == n


class TestPipelineEmptyFile:
    @pytest.fixture
    def tracker(self) -> SQLiteTracker:
        t = SQLiteTracker(":memory:")
        yield t
        t.close()

    @pytest.mark.asyncio
    async def test_empty_chunk_list_still_records_file(
        self, settings, tracker: SQLiteTracker, tmp_path: Path
    ) -> None:
        (tmp_path / "empty.py").write_text("")
        pipeline, mock_vs = _make_pipeline(settings, tracker)
        pipeline._cloner = MagicMock()
        pipeline._cloner.clone_or_pull.return_value = tmp_path
        pipeline._chunker = MagicMock()
        pipeline._chunker.chunk_file.return_value = []  # no chunks

        await pipeline.run(REPO_URL)

        mock_vs.upsert_chunks.assert_not_awaited()
        record = tracker.get_file_record(REPO_URL, "empty.py")
        assert record is not None
        assert record.chunk_ids == []
