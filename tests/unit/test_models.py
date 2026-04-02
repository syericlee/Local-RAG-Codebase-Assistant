from datetime import datetime

import pytest

from rag_assistant.models.api import Citation, IndexRequest, JobStatus, QueryRequest, QueryResponse
from rag_assistant.models.chunk import ChunkType, CodeChunk
from rag_assistant.models.search import RerankedResult, RetrievalResponse, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(**overrides) -> CodeChunk:
    defaults = dict(
        id=CodeChunk.make_id("https://github.com/foo/bar", "src/main.py", 1),
        content="def hello(): pass",
        file_path="src/main.py",
        start_line=1,
        end_line=1,
        language="python",
        chunk_type=ChunkType.FUNCTION,
        symbol_name="hello",
        repo_url="https://github.com/foo/bar",
        file_hash="abc123",
    )
    defaults.update(overrides)
    return CodeChunk(**defaults)


# ---------------------------------------------------------------------------
# CodeChunk
# ---------------------------------------------------------------------------

class TestCodeChunk:
    def test_make_id_is_deterministic(self):
        id1 = CodeChunk.make_id("https://github.com/foo/bar", "src/main.py", 10)
        id2 = CodeChunk.make_id("https://github.com/foo/bar", "src/main.py", 10)
        assert id1 == id2

    def test_make_id_length(self):
        chunk_id = CodeChunk.make_id("https://github.com/foo/bar", "src/main.py", 1)
        assert len(chunk_id) == 16

    def test_make_id_differs_on_different_start_line(self):
        id1 = CodeChunk.make_id("https://github.com/foo/bar", "src/main.py", 10)
        id2 = CodeChunk.make_id("https://github.com/foo/bar", "src/main.py", 20)
        assert id1 != id2

    def test_make_id_differs_on_different_file(self):
        id1 = CodeChunk.make_id("https://github.com/foo/bar", "src/a.py", 1)
        id2 = CodeChunk.make_id("https://github.com/foo/bar", "src/b.py", 1)
        assert id1 != id2

    def test_make_id_differs_on_different_repo(self):
        id1 = CodeChunk.make_id("https://github.com/foo/bar", "src/main.py", 1)
        id2 = CodeChunk.make_id("https://github.com/foo/baz", "src/main.py", 1)
        assert id1 != id2

    def test_instantiation(self):
        chunk = make_chunk()
        assert chunk.chunk_type == ChunkType.FUNCTION
        assert chunk.symbol_name == "hello"
        assert chunk.language == "python"

    def test_optional_symbol_name_defaults_none(self):
        chunk = make_chunk(symbol_name=None, chunk_type=ChunkType.BLOCK)
        assert chunk.symbol_name is None

    def test_chunk_type_enum_values(self):
        assert ChunkType.FUNCTION == "function"
        assert ChunkType.CLASS == "class"
        assert ChunkType.METHOD == "method"
        assert ChunkType.MODULE == "module"
        assert ChunkType.BLOCK == "block"
        assert ChunkType.IMPORT_BLOCK == "import_block"


# ---------------------------------------------------------------------------
# SearchResult / RerankedResult / RetrievalResponse
# ---------------------------------------------------------------------------

class TestSearchModels:
    def test_search_result(self):
        chunk = make_chunk()
        result = SearchResult(chunk=chunk, vector_score=0.95)
        assert result.vector_score == 0.95
        assert result.chunk.id == chunk.id

    def test_reranked_result(self):
        chunk = make_chunk()
        result = RerankedResult(chunk=chunk, vector_score=0.85, rerank_score=8.3)
        assert result.rerank_score == 8.3
        assert result.vector_score == 0.85

    def test_retrieval_response_empty(self):
        resp = RetrievalResponse(
            results=[],
            total_tokens_in_context=0,
            query_embedding_ms=5.2,
            vector_search_ms=10.1,
            rerank_ms=50.3,
        )
        assert resp.results == []
        assert resp.total_tokens_in_context == 0

    def test_retrieval_response_with_results(self):
        chunk = make_chunk()
        result = RerankedResult(chunk=chunk, vector_score=0.9, rerank_score=7.5)
        resp = RetrievalResponse(
            results=[result],
            total_tokens_in_context=250,
            query_embedding_ms=3.0,
            vector_search_ms=8.0,
            rerank_ms=45.0,
        )
        assert len(resp.results) == 1
        assert resp.total_tokens_in_context == 250


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class TestCitation:
    def test_instantiation(self):
        c = Citation(file_path="src/auth.py", start_line=10, end_line=25)
        assert c.file_path == "src/auth.py"
        assert c.start_line == 10
        assert c.end_line == 25


class TestQueryRequest:
    def test_defaults(self):
        req = QueryRequest(query="how does auth work?")
        assert req.repo_url is None
        assert req.top_k is None

    def test_with_overrides(self):
        req = QueryRequest(
            query="explain caching",
            repo_url="https://github.com/foo/bar",
            top_k=3,
        )
        assert req.top_k == 3
        assert req.repo_url == "https://github.com/foo/bar"


class TestQueryResponse:
    def test_cache_hit_miss(self):
        resp = QueryResponse(answer="The answer", citations=[], cache_hit="miss")
        assert resp.cache_hit == "miss"
        assert resp.retrieval is None

    def test_cache_hit_exact(self):
        resp = QueryResponse(answer="Cached", citations=[], cache_hit="exact")
        assert resp.cache_hit == "exact"

    def test_cache_hit_semantic(self):
        resp = QueryResponse(answer="Semantic", citations=[], cache_hit="semantic")
        assert resp.cache_hit == "semantic"

    def test_with_citations(self):
        citations = [Citation(file_path="main.py", start_line=1, end_line=5)]
        resp = QueryResponse(answer="See main.py", citations=citations, cache_hit="miss")
        assert len(resp.citations) == 1


class TestIndexRequest:
    def test_defaults(self):
        req = IndexRequest(repo_url="https://github.com/foo/bar")
        assert req.branch == "main"
        assert req.force_reindex is False

    def test_force_reindex(self):
        req = IndexRequest(repo_url="https://github.com/foo/bar", force_reindex=True)
        assert req.force_reindex is True


class TestJobStatus:
    def test_instantiation(self):
        now = datetime.utcnow()
        job = JobStatus(
            job_id="abc-123",
            status="pending",
            repo_url="https://github.com/foo/bar",
            created_at=now,
            updated_at=now,
        )
        assert job.status == "pending"
        assert job.files_indexed == 0
        assert job.chunks_upserted == 0
        assert job.error is None

    def test_completed_status(self):
        now = datetime.utcnow()
        job = JobStatus(
            job_id="abc-123",
            status="completed",
            repo_url="https://github.com/foo/bar",
            created_at=now,
            updated_at=now,
            files_indexed=42,
            chunks_upserted=380,
        )
        assert job.files_indexed == 42
        assert job.chunks_upserted == 380

    def test_failed_status_with_error(self):
        now = datetime.utcnow()
        job = JobStatus(
            job_id="abc-123",
            status="failed",
            repo_url="https://github.com/foo/bar",
            created_at=now,
            updated_at=now,
            error="Connection refused",
        )
        assert job.status == "failed"
        assert job.error == "Connection refused"
