import pytest

from rag_assistant.ingestion.sliding_chunker import SlidingWindowChunker
from rag_assistant.models.chunk import ChunkType

REPO_URL = "https://github.com/foo/bar"
FILE_PATH = "src/main.py"


def make_source(n_lines: int, tokens_per_line: int = 5) -> str:
    """Generate source with predictable lines.
    Each line is 'a a a ...' repeated tokens_per_line times.
    Single-letter tokens separated by spaces are reliably 1 token each in cl100k_base.
    """
    return "\n".join(" ".join(["a"] * tokens_per_line) for _ in range(n_lines))


class TestSlidingWindowChunker:
    def test_empty_source_returns_no_chunks(self) -> None:
        chunker = SlidingWindowChunker(max_tokens=512, overlap_tokens=64)
        assert chunker.chunk("", FILE_PATH, "python", REPO_URL) == []

    def test_single_line_produces_one_chunk(self) -> None:
        chunker = SlidingWindowChunker(max_tokens=512, overlap_tokens=64)
        chunks = chunker.chunk("def foo(): pass", FILE_PATH, "python", REPO_URL)
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 1

    def test_chunk_type_is_block(self) -> None:
        chunker = SlidingWindowChunker(max_tokens=512, overlap_tokens=64)
        chunks = chunker.chunk("x = 1\ny = 2", FILE_PATH, "python", REPO_URL)
        assert all(c.chunk_type == ChunkType.BLOCK for c in chunks)

    def test_symbol_name_is_none(self) -> None:
        chunker = SlidingWindowChunker(max_tokens=512, overlap_tokens=64)
        chunks = chunker.chunk("x = 1\ny = 2", FILE_PATH, "python", REPO_URL)
        assert all(c.symbol_name is None for c in chunks)

    def test_small_file_fits_in_one_chunk(self) -> None:
        chunker = SlidingWindowChunker(max_tokens=512, overlap_tokens=64)
        source = make_source(n_lines=10, tokens_per_line=5)  # ~50 tokens total
        chunks = chunker.chunk(source, FILE_PATH, "python", REPO_URL)
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 10

    def test_large_file_produces_multiple_chunks(self) -> None:
        # 10 tokens per line, max 50 tokens → ~5 lines per chunk
        chunker = SlidingWindowChunker(max_tokens=50, overlap_tokens=10)
        source = make_source(n_lines=30, tokens_per_line=10)
        chunks = chunker.chunk(source, FILE_PATH, "python", REPO_URL)
        assert len(chunks) > 1

    def test_chunks_cover_entire_file(self) -> None:
        chunker = SlidingWindowChunker(max_tokens=50, overlap_tokens=10)
        source = make_source(n_lines=30, tokens_per_line=10)
        chunks = chunker.chunk(source, FILE_PATH, "python", REPO_URL)
        assert chunks[0].start_line == 1
        assert chunks[-1].end_line == 30

    def test_overlap_makes_consecutive_chunks_share_lines(self) -> None:
        # With overlap, chunk N+1 should start at or before where chunk N ended.
        # (start_line == end_line is valid when a single line fills the overlap budget.)
        chunker = SlidingWindowChunker(max_tokens=50, overlap_tokens=15)
        source = make_source(n_lines=30, tokens_per_line=10)
        chunks = chunker.chunk(source, FILE_PATH, "python", REPO_URL)
        for i in range(len(chunks) - 1):
            assert chunks[i + 1].start_line <= chunks[i].end_line

    def test_no_overlap_chunks_are_contiguous(self) -> None:
        # With zero overlap, chunk N+1 starts where chunk N ended + 1
        chunker = SlidingWindowChunker(max_tokens=50, overlap_tokens=0)
        source = make_source(n_lines=20, tokens_per_line=10)
        chunks = chunker.chunk(source, FILE_PATH, "python", REPO_URL)
        for i in range(len(chunks) - 1):
            assert chunks[i + 1].start_line == chunks[i].end_line + 1

    def test_oversized_single_line_produces_one_chunk(self) -> None:
        # A line with more tokens than max_tokens must still be chunked (not dropped)
        chunker = SlidingWindowChunker(max_tokens=5, overlap_tokens=1)
        long_line = " ".join([f"word{i}" for i in range(50)])  # ~50 tokens
        chunks = chunker.chunk(long_line, FILE_PATH, "python", REPO_URL)
        assert len(chunks) >= 1
        assert chunks[0].content == long_line

    def test_line_numbers_are_one_based(self) -> None:
        chunker = SlidingWindowChunker(max_tokens=512, overlap_tokens=0)
        source = "line one\nline two\nline three"
        chunks = chunker.chunk(source, FILE_PATH, "python", REPO_URL)
        assert chunks[0].start_line == 1

    def test_chunk_content_matches_source_lines(self) -> None:
        chunker = SlidingWindowChunker(max_tokens=512, overlap_tokens=0)
        source = "line one\nline two\nline three"
        chunks = chunker.chunk(source, FILE_PATH, "python", REPO_URL)
        assert chunks[0].content == source

    def test_metadata_fields(self) -> None:
        chunker = SlidingWindowChunker(max_tokens=512, overlap_tokens=0)
        chunks = chunker.chunk("x = 1", FILE_PATH, "python", REPO_URL)
        c = chunks[0]
        assert c.file_path == FILE_PATH
        assert c.repo_url == REPO_URL
        assert c.language == "python"
        assert len(c.file_hash) == 64  # sha256 hex digest

    def test_chunk_ids_are_deterministic(self) -> None:
        chunker = SlidingWindowChunker(max_tokens=512, overlap_tokens=0)
        source = "x = 1\ny = 2"
        chunks1 = chunker.chunk(source, FILE_PATH, "python", REPO_URL)
        chunks2 = chunker.chunk(source, FILE_PATH, "python", REPO_URL)
        assert [c.id for c in chunks1] == [c.id for c in chunks2]

    def test_chunk_ids_differ_across_files(self) -> None:
        chunker = SlidingWindowChunker(max_tokens=512, overlap_tokens=0)
        source = "x = 1"
        chunks_a = chunker.chunk(source, "a.py", "python", REPO_URL)
        chunks_b = chunker.chunk(source, "b.py", "python", REPO_URL)
        assert chunks_a[0].id != chunks_b[0].id
