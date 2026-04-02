from __future__ import annotations

from pathlib import Path

import git


class RepoCloner:
    """Clones a git repo on first call, pulls on subsequent calls."""

    def __init__(self, repos_base_dir: str) -> None:
        self._base_dir = Path(repos_base_dir)

    def clone_or_pull(self, repo_url: str, branch: str = "main") -> Path:
        """Return the local path to the repo, cloning or pulling as needed."""
        repo_dir = self._repo_dir(repo_url)
        if repo_dir.exists():
            repo = git.Repo(repo_dir)
            repo.remotes.origin.pull()
        else:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            git.Repo.clone_from(repo_url, repo_dir, branch=branch)
        return repo_dir

    def _repo_dir(self, repo_url: str) -> Path:
        """Derive a safe directory name from a repo URL.

        e.g. 'https://github.com/foo/bar.git' → '<base>/github.com_foo_bar'
        """
        slug = repo_url
        for prefix in ("https://", "http://"):
            slug = slug.removeprefix(prefix)
        slug = slug.removesuffix(".git")
        slug = slug.replace("/", "_")
        return self._base_dir / slug
