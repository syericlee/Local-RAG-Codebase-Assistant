import pytest

from rag_assistant.config import Settings
from rag_assistant.ingestion.chunker import CodeChunker
from rag_assistant.models.chunk import ChunkType

REPO_URL = "https://github.com/foo/bar"


@pytest.fixture
def chunker(settings: Settings) -> CodeChunker:
    return CodeChunker(settings)


class TestCodeChunker:
    def test_python_file_uses_ast_chunker(self, chunker: CodeChunker) -> None:
        source = "def hello():\n    pass\n"
        chunks = chunker.chunk_file(source, "src/main.py", REPO_URL)
        assert len(chunks) >= 1
        assert chunks[0].chunk_type == ChunkType.FUNCTION
        assert chunks[0].symbol_name == "hello"

    def test_go_file_uses_ast_chunker(self, chunker: CodeChunker) -> None:
        source = "package main\n\nfunc Hello() {}\n"
        chunks = chunker.chunk_file(source, "main.go", REPO_URL)
        assert any(c.symbol_name == "Hello" for c in chunks)

    def test_unknown_extension_uses_sliding_window(self, chunker: CodeChunker) -> None:
        source = "some plain text\nthat is not code\n"
        chunks = chunker.chunk_file(source, "notes.txt", REPO_URL)
        assert len(chunks) >= 1
        assert all(c.chunk_type == ChunkType.BLOCK for c in chunks)

    def test_unknown_extension_language_is_unknown(self, chunker: CodeChunker) -> None:
        chunks = chunker.chunk_file("hello\nworld\n", "README.md", REPO_URL)
        assert all(c.language == "unknown" for c in chunks)

    def test_typescript_file_dispatched_to_ast(self, chunker: CodeChunker) -> None:
        source = "function greet(name: string): string {\n    return name;\n}\n"
        chunks = chunker.chunk_file(source, "greet.ts", REPO_URL)
        assert any(c.symbol_name == "greet" for c in chunks)

    def test_javascript_file_dispatched_to_ast(self, chunker: CodeChunker) -> None:
        source = "function add(a, b) {\n    return a + b;\n}\n"
        chunks = chunker.chunk_file(source, "math.js", REPO_URL)
        assert any(c.symbol_name == "add" for c in chunks)

    def test_rust_file_dispatched_to_ast(self, chunker: CodeChunker) -> None:
        source = "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n"
        chunks = chunker.chunk_file(source, "lib.rs", REPO_URL)
        assert any(c.symbol_name == "add" for c in chunks)

    def test_file_path_stored_on_chunk(self, chunker: CodeChunker) -> None:
        source = "def foo():\n    pass\n"
        chunks = chunker.chunk_file(source, "src/foo.py", REPO_URL)
        assert all(c.file_path == "src/foo.py" for c in chunks)
