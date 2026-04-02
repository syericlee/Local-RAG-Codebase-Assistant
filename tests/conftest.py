import pytest

from rag_assistant.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Settings instance with test-safe defaults (no real services required)."""
    return Settings(
        qdrant_host="localhost",
        qdrant_port=6333,
        redis_url="redis://localhost:6379",
        ollama_base_url="http://localhost:11434",
        sqlite_db_path=":memory:",
        repos_base_dir="/tmp/test_repos",
    )
