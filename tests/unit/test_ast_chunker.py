import pytest

from rag_assistant.ingestion.ast_chunker import ASTChunker
from rag_assistant.models.chunk import ChunkType

REPO_URL = "https://github.com/foo/bar"
FILE_PATH = "src/main"


# ---------------------------------------------------------------------------
# Fixture sources
# ---------------------------------------------------------------------------

PYTHON_SOURCE = """\
def hello(name: str) -> str:
    return f"Hello, {name}"


class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

    def subtract(self, a: int, b: int) -> int:
        return a - b
"""

GO_SOURCE = """\
package main

func Add(a, b int) int {
\treturn a + b
}

func Subtract(a, b int) int {
\treturn a - b
}
"""

RUST_SOURCE = """\
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn subtract(a: i32, b: i32) -> i32 {
    a - b
}
"""

TS_SOURCE = """\
function greet(name: string): string {
    return `Hello, ${name}`;
}

class Greeter {
    greet(name: string): string {
        return `Hi, ${name}`;
    }
}
"""

JS_SOURCE = """\
function add(a, b) {
    return a + b;
}

class Calculator {
    subtract(a, b) {
        return a - b;
    }
}
"""


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

class TestPythonChunker:
    def setup_method(self) -> None:
        self.chunker = ASTChunker(max_tokens=512, overlap_tokens=64)

    def test_extracts_top_level_function(self) -> None:
        chunks = self.chunker.chunk(PYTHON_SOURCE, FILE_PATH + ".py", "python", REPO_URL)
        names = [c.symbol_name for c in chunks]
        assert "hello" in names

    def test_top_level_function_chunk_type(self) -> None:
        chunks = self.chunker.chunk(PYTHON_SOURCE, FILE_PATH + ".py", "python", REPO_URL)
        hello = next(c for c in chunks if c.symbol_name == "hello")
        assert hello.chunk_type == ChunkType.FUNCTION

    def test_extracts_class(self) -> None:
        chunks = self.chunker.chunk(PYTHON_SOURCE, FILE_PATH + ".py", "python", REPO_URL)
        names = [c.symbol_name for c in chunks]
        assert "Calculator" in names

    def test_class_chunk_type(self) -> None:
        chunks = self.chunker.chunk(PYTHON_SOURCE, FILE_PATH + ".py", "python", REPO_URL)
        calc = next(c for c in chunks if c.symbol_name == "Calculator")
        assert calc.chunk_type == ChunkType.CLASS

    def test_function_line_numbers(self) -> None:
        chunks = self.chunker.chunk(PYTHON_SOURCE, FILE_PATH + ".py", "python", REPO_URL)
        hello = next(c for c in chunks if c.symbol_name == "hello")
        assert hello.start_line == 1
        assert hello.end_line == 2

    def test_language_field(self) -> None:
        chunks = self.chunker.chunk(PYTHON_SOURCE, FILE_PATH + ".py", "python", REPO_URL)
        assert all(c.language == "python" for c in chunks)

    def test_oversized_class_splits_into_methods(self) -> None:
        # Set max_tokens very low so the whole Calculator class doesn't fit,
        # forcing the chunker to recurse into individual methods.
        chunker = ASTChunker(max_tokens=20, overlap_tokens=0)
        chunks = chunker.chunk(PYTHON_SOURCE, FILE_PATH + ".py", "python", REPO_URL)
        names = [c.symbol_name for c in chunks]
        assert "add" in names or "subtract" in names
        # The Calculator class as a whole should not appear
        assert "Calculator" not in names

    def test_decorated_function_name_extracted(self) -> None:
        # A top-level decorated function — name should come from the inner def, not the decorator
        source = """\
@some_decorator
def bar():
    pass
"""
        chunks = self.chunker.chunk(source, FILE_PATH + ".py", "python", REPO_URL)
        names = [c.symbol_name for c in chunks]
        assert "bar" in names

    def test_empty_source_returns_no_chunks(self) -> None:
        chunks = self.chunker.chunk("", FILE_PATH + ".py", "python", REPO_URL)
        assert chunks == []

    def test_chunk_ids_are_deterministic(self) -> None:
        chunks1 = self.chunker.chunk(PYTHON_SOURCE, FILE_PATH + ".py", "python", REPO_URL)
        chunks2 = self.chunker.chunk(PYTHON_SOURCE, FILE_PATH + ".py", "python", REPO_URL)
        assert [c.id for c in chunks1] == [c.id for c in chunks2]


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------

class TestGoChunker:
    def setup_method(self) -> None:
        self.chunker = ASTChunker(max_tokens=512, overlap_tokens=64)

    def test_extracts_functions(self) -> None:
        chunks = self.chunker.chunk(GO_SOURCE, FILE_PATH + ".go", "go", REPO_URL)
        names = [c.symbol_name for c in chunks]
        assert "Add" in names
        assert "Subtract" in names

    def test_function_chunk_type(self) -> None:
        chunks = self.chunker.chunk(GO_SOURCE, FILE_PATH + ".go", "go", REPO_URL)
        assert all(c.chunk_type == ChunkType.FUNCTION for c in chunks)

    def test_language_field(self) -> None:
        chunks = self.chunker.chunk(GO_SOURCE, FILE_PATH + ".go", "go", REPO_URL)
        assert all(c.language == "go" for c in chunks)


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------

class TestRustChunker:
    def setup_method(self) -> None:
        self.chunker = ASTChunker(max_tokens=512, overlap_tokens=64)

    def test_extracts_functions(self) -> None:
        chunks = self.chunker.chunk(RUST_SOURCE, FILE_PATH + ".rs", "rust", REPO_URL)
        names = [c.symbol_name for c in chunks]
        assert "add" in names
        assert "subtract" in names

    def test_function_chunk_type(self) -> None:
        chunks = self.chunker.chunk(RUST_SOURCE, FILE_PATH + ".rs", "rust", REPO_URL)
        assert all(c.chunk_type == ChunkType.FUNCTION for c in chunks)


# ---------------------------------------------------------------------------
# TypeScript
# ---------------------------------------------------------------------------

class TestTypeScriptChunker:
    def setup_method(self) -> None:
        self.chunker = ASTChunker(max_tokens=512, overlap_tokens=64)

    def test_extracts_function(self) -> None:
        chunks = self.chunker.chunk(TS_SOURCE, FILE_PATH + ".ts", "typescript", REPO_URL)
        names = [c.symbol_name for c in chunks]
        assert "greet" in names

    def test_extracts_class(self) -> None:
        chunks = self.chunker.chunk(TS_SOURCE, FILE_PATH + ".ts", "typescript", REPO_URL)
        names = [c.symbol_name for c in chunks]
        assert "Greeter" in names


# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------

class TestJavaScriptChunker:
    def setup_method(self) -> None:
        self.chunker = ASTChunker(max_tokens=512, overlap_tokens=64)

    def test_extracts_function(self) -> None:
        chunks = self.chunker.chunk(JS_SOURCE, FILE_PATH + ".js", "javascript", REPO_URL)
        names = [c.symbol_name for c in chunks]
        assert "add" in names

    def test_extracts_class(self) -> None:
        chunks = self.chunker.chunk(JS_SOURCE, FILE_PATH + ".js", "javascript", REPO_URL)
        names = [c.symbol_name for c in chunks]
        assert "Calculator" in names


# ---------------------------------------------------------------------------
# Oversized node fallback
# ---------------------------------------------------------------------------

class TestOversizedFallback:
    def test_large_function_falls_back_to_sliding_window(self) -> None:
        # A function large enough to exceed max_tokens with no child functions
        lines = ["def big_function():"] + [f"    x_{i} = {i}" for i in range(100)]
        source = "\n".join(lines)
        chunker = ASTChunker(max_tokens=30, overlap_tokens=0)
        chunks = chunker.chunk(source, FILE_PATH + ".py", "python", REPO_URL)
        # Should produce multiple BLOCK chunks from the sliding window fallback
        assert len(chunks) > 1
        assert all(c.chunk_type == ChunkType.BLOCK for c in chunks)

    def test_fallback_line_numbers_are_file_absolute(self) -> None:
        # Function starts at line 5 — fallback chunks must have line numbers >= 5
        preamble = "# comment\n" * 4  # 4 lines before the function
        lines = ["def big_function():"] + [f"    x_{i} = {i}" for i in range(100)]
        source = preamble + "\n".join(lines)
        chunker = ASTChunker(max_tokens=30, overlap_tokens=0)
        chunks = chunker.chunk(source, FILE_PATH + ".py", "python", REPO_URL)
        assert all(c.start_line >= 5 for c in chunks)
