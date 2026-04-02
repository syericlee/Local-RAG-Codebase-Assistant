from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_assistant.ingestion.cloner import RepoCloner
from rag_assistant.ingestion.walker import FileWalker

REPO_URL = "https://github.com/foo/bar"


# ---------------------------------------------------------------------------
# RepoCloner
# ---------------------------------------------------------------------------

class TestRepoCloner:
    @pytest.fixture
    def cloner(self, tmp_path: Path) -> RepoCloner:
        return RepoCloner(str(tmp_path / "repos"))

    def test_repo_dir_strips_https_prefix(self, cloner: RepoCloner) -> None:
        d = cloner._repo_dir("https://github.com/foo/bar")
        assert d.name == "github.com_foo_bar"

    def test_repo_dir_strips_git_suffix(self, cloner: RepoCloner) -> None:
        d = cloner._repo_dir("https://github.com/foo/bar.git")
        assert d.name == "github.com_foo_bar"

    def test_repo_dir_strips_http_prefix(self, cloner: RepoCloner) -> None:
        d = cloner._repo_dir("http://github.com/foo/bar")
        assert d.name == "github.com_foo_bar"

    def test_clone_called_when_dir_missing(self, cloner: RepoCloner, tmp_path: Path) -> None:
        with patch("git.Repo.clone_from") as mock_clone:
            mock_clone.return_value = MagicMock()
            cloner.clone_or_pull(REPO_URL, branch="main")
            mock_clone.assert_called_once_with(
                REPO_URL,
                cloner._repo_dir(REPO_URL),
                branch="main",
            )

    def test_pull_called_when_dir_exists(self, cloner: RepoCloner) -> None:
        repo_dir = cloner._repo_dir(REPO_URL)
        repo_dir.mkdir(parents=True)

        mock_remote = MagicMock()
        mock_repo = MagicMock()
        mock_repo.remotes.origin = mock_remote

        with patch("git.Repo", return_value=mock_repo):
            cloner.clone_or_pull(REPO_URL)
            mock_remote.pull.assert_called_once()

    def test_clone_creates_base_dir(self, cloner: RepoCloner, tmp_path: Path) -> None:
        base = tmp_path / "repos"
        assert not base.exists()
        with patch("git.Repo.clone_from"):
            cloner.clone_or_pull(REPO_URL)
        assert base.exists()

    def test_returns_repo_path(self, cloner: RepoCloner) -> None:
        with patch("git.Repo.clone_from"):
            result = cloner.clone_or_pull(REPO_URL)
        assert result == cloner._repo_dir(REPO_URL)


# ---------------------------------------------------------------------------
# FileWalker
# ---------------------------------------------------------------------------

class TestFileWalker:
    @pytest.fixture
    def walker(self) -> FileWalker:
        return FileWalker()

    def test_finds_python_files(self, walker: FileWalker, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("x = 1")
        result = walker.walk(tmp_path)
        assert "main.py" in result

    def test_finds_go_files(self, walker: FileWalker, tmp_path: Path) -> None:
        (tmp_path / "main.go").write_text("package main")
        result = walker.walk(tmp_path)
        assert "main.go" in result

    def test_skips_unsupported_extension(self, walker: FileWalker, tmp_path: Path) -> None:
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        result = walker.walk(tmp_path)
        assert "image.png" not in result

    def test_skips_git_directory(self, walker: FileWalker, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("[core]")
        result = walker.walk(tmp_path)
        assert not any(".git" in p for p in result)

    def test_skips_node_modules(self, walker: FileWalker, tmp_path: Path) -> None:
        nm = tmp_path / "node_modules" / "lodash"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("module.exports = {}")
        result = walker.walk(tmp_path)
        assert not any("node_modules" in p for p in result)

    def test_skips_pycache(self, walker: FileWalker, tmp_path: Path) -> None:
        cache = tmp_path / "src" / "__pycache__"
        cache.mkdir(parents=True)
        (cache / "main.cpython-310.pyc").write_bytes(b"bytecode")
        result = walker.walk(tmp_path)
        assert not any("__pycache__" in p for p in result)

    def test_skips_hidden_directories(self, walker: FileWalker, tmp_path: Path) -> None:
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "config.py").write_text("x = 1")
        result = walker.walk(tmp_path)
        assert not any(".hidden" in p for p in result)

    def test_returns_relative_paths(self, walker: FileWalker, tmp_path: Path) -> None:
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("x = 1")
        result = walker.walk(tmp_path)
        assert "src/main.py" in result

    def test_hash_is_sha256(self, walker: FileWalker, tmp_path: Path) -> None:
        (tmp_path / "f.py").write_text("hello")
        result = walker.walk(tmp_path)
        file_hash, _ = result["f.py"]
        assert len(file_hash) == 64  # sha256 hex digest

    def test_same_content_same_hash(self, walker: FileWalker, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("x = 1")
        result = walker.walk(tmp_path)
        assert result["a.py"][0] == result["b.py"][0]

    def test_different_content_different_hash(self, walker: FileWalker, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("x = 2")
        result = walker.walk(tmp_path)
        assert result["a.py"][0] != result["b.py"][0]

    def test_empty_directory_returns_empty(self, walker: FileWalker, tmp_path: Path) -> None:
        assert walker.walk(tmp_path) == {}

    def test_mtime_is_float(self, walker: FileWalker, tmp_path: Path) -> None:
        (tmp_path / "f.py").write_text("x = 1")
        result = walker.walk(tmp_path)
        _, mtime = result["f.py"]
        assert isinstance(mtime, float)
