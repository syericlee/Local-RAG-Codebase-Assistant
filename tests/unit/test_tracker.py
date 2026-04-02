import pytest

from rag_assistant.ingestion.tracker import RepoDiff, SQLiteTracker

REPO_URL = "https://github.com/foo/bar"


@pytest.fixture
def tracker() -> SQLiteTracker:
    """Fresh in-memory tracker for each test."""
    t = SQLiteTracker(":memory:")
    yield t
    t.close()


class TestSQLiteTrackerRead:
    def test_get_file_record_returns_none_when_missing(self, tracker: SQLiteTracker) -> None:
        assert tracker.get_file_record(REPO_URL, "src/main.py") is None

    def test_get_all_file_paths_empty_when_no_records(self, tracker: SQLiteTracker) -> None:
        assert tracker.get_all_file_paths(REPO_URL) == set()

    def test_get_file_record_after_upsert(self, tracker: SQLiteTracker) -> None:
        tracker.upsert_file_record(REPO_URL, "src/main.py", "abc123", 1000.0, ["id1", "id2"])
        record = tracker.get_file_record(REPO_URL, "src/main.py")
        assert record is not None
        assert record.file_hash == "abc123"
        assert record.mtime == 1000.0
        assert record.chunk_ids == ["id1", "id2"]

    def test_get_all_file_paths_returns_all(self, tracker: SQLiteTracker) -> None:
        tracker.upsert_file_record(REPO_URL, "a.py", "h1", 1.0, [])
        tracker.upsert_file_record(REPO_URL, "b.py", "h2", 2.0, [])
        assert tracker.get_all_file_paths(REPO_URL) == {"a.py", "b.py"}

    def test_get_all_file_paths_isolated_by_repo(self, tracker: SQLiteTracker) -> None:
        tracker.upsert_file_record(REPO_URL, "a.py", "h1", 1.0, [])
        tracker.upsert_file_record("https://github.com/other/repo", "b.py", "h2", 2.0, [])
        assert tracker.get_all_file_paths(REPO_URL) == {"a.py"}


class TestSQLiteTrackerWrite:
    def test_upsert_stores_record(self, tracker: SQLiteTracker) -> None:
        tracker.upsert_file_record(REPO_URL, "src/main.py", "abc", 1.0, ["c1"])
        record = tracker.get_file_record(REPO_URL, "src/main.py")
        assert record is not None

    def test_upsert_overwrites_existing_record(self, tracker: SQLiteTracker) -> None:
        tracker.upsert_file_record(REPO_URL, "src/main.py", "old_hash", 1.0, ["c1"])
        tracker.upsert_file_record(REPO_URL, "src/main.py", "new_hash", 2.0, ["c2", "c3"])
        record = tracker.get_file_record(REPO_URL, "src/main.py")
        assert record is not None
        assert record.file_hash == "new_hash"
        assert record.chunk_ids == ["c2", "c3"]

    def test_delete_removes_record(self, tracker: SQLiteTracker) -> None:
        tracker.upsert_file_record(REPO_URL, "src/main.py", "abc", 1.0, ["c1"])
        tracker.delete_file_record(REPO_URL, "src/main.py")
        assert tracker.get_file_record(REPO_URL, "src/main.py") is None

    def test_delete_nonexistent_is_safe(self, tracker: SQLiteTracker) -> None:
        tracker.delete_file_record(REPO_URL, "nonexistent.py")  # should not raise

    def test_chunk_ids_preserved_as_list(self, tracker: SQLiteTracker) -> None:
        ids = ["a1b2", "c3d4", "e5f6"]
        tracker.upsert_file_record(REPO_URL, "f.py", "h", 1.0, ids)
        record = tracker.get_file_record(REPO_URL, "f.py")
        assert record is not None
        assert record.chunk_ids == ids


class TestSQLiteTrackerDiff:
    def test_all_new_when_db_empty(self, tracker: SQLiteTracker) -> None:
        current = {"a.py": ("h1", 1.0), "b.py": ("h2", 2.0)}
        diff = tracker.diff(REPO_URL, current)
        assert set(diff.new_paths) == {"a.py", "b.py"}
        assert diff.modified_paths == []
        assert diff.deleted_paths == []

    def test_all_deleted_when_current_empty(self, tracker: SQLiteTracker) -> None:
        tracker.upsert_file_record(REPO_URL, "a.py", "h1", 1.0, ["c1"])
        tracker.upsert_file_record(REPO_URL, "b.py", "h2", 2.0, ["c2"])
        diff = tracker.diff(REPO_URL, {})
        assert set(diff.deleted_paths) == {"a.py", "b.py"}
        assert diff.new_paths == []
        assert diff.modified_paths == []

    def test_unchanged_file_not_in_diff(self, tracker: SQLiteTracker) -> None:
        tracker.upsert_file_record(REPO_URL, "a.py", "h1", 1.0, ["c1"])
        diff = tracker.diff(REPO_URL, {"a.py": ("h1", 2.0)})  # mtime changed, hash same
        assert diff.new_paths == []
        assert diff.modified_paths == []
        assert diff.deleted_paths == []

    def test_modified_file_detected_by_hash(self, tracker: SQLiteTracker) -> None:
        tracker.upsert_file_record(REPO_URL, "a.py", "old_hash", 1.0, ["c1"])
        diff = tracker.diff(REPO_URL, {"a.py": ("new_hash", 2.0)})
        assert "a.py" in diff.modified_paths

    def test_mixed_diff(self, tracker: SQLiteTracker) -> None:
        tracker.upsert_file_record(REPO_URL, "unchanged.py", "h1", 1.0, ["c1"])
        tracker.upsert_file_record(REPO_URL, "modified.py", "old", 1.0, ["c2"])
        tracker.upsert_file_record(REPO_URL, "deleted.py", "h3", 1.0, ["c3"])

        current = {
            "unchanged.py": ("h1", 1.0),
            "modified.py": ("new", 2.0),
            "new.py": ("h4", 3.0),
        }
        diff = tracker.diff(REPO_URL, current)

        assert diff.new_paths == ["new.py"]
        assert diff.modified_paths == ["modified.py"]
        assert diff.deleted_paths == ["deleted.py"]


class TestSQLiteTrackerChunkIds:
    def test_get_chunk_ids_for_paths(self, tracker: SQLiteTracker) -> None:
        tracker.upsert_file_record(REPO_URL, "a.py", "h1", 1.0, ["c1", "c2"])
        tracker.upsert_file_record(REPO_URL, "b.py", "h2", 2.0, ["c3"])
        ids = tracker.get_chunk_ids_for_paths(REPO_URL, ["a.py", "b.py"])
        assert set(ids) == {"c1", "c2", "c3"}

    def test_get_chunk_ids_skips_missing_paths(self, tracker: SQLiteTracker) -> None:
        tracker.upsert_file_record(REPO_URL, "a.py", "h1", 1.0, ["c1"])
        ids = tracker.get_chunk_ids_for_paths(REPO_URL, ["a.py", "nonexistent.py"])
        assert ids == ["c1"]

    def test_get_chunk_ids_empty_when_no_paths(self, tracker: SQLiteTracker) -> None:
        assert tracker.get_chunk_ids_for_paths(REPO_URL, []) == []
