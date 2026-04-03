from rag_assistant.generation.prompt import PromptBuilder
from rag_assistant.models.chunk import ChunkType, CodeChunk
from rag_assistant.models.search import RerankedResult, SearchResult

REPO_URL = "https://github.com/foo/bar"


def _make_result(file_path: str, start: int, end: int, content: str) -> RerankedResult:
    chunk = CodeChunk(
        id="abcdef1234567890",
        content=content,
        file_path=file_path,
        start_line=start,
        end_line=end,
        language="python",
        chunk_type=ChunkType.FUNCTION,
        symbol_name="foo",
        repo_url=REPO_URL,
        file_hash="a" * 64,
    )
    return RerankedResult(chunk=chunk, vector_score=0.9, rerank_score=0.8)


class TestPromptBuilder:
    def setup_method(self) -> None:
        self.builder = PromptBuilder()

    def test_returns_two_messages(self) -> None:
        messages = self.builder.build("How does login work?", [])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_message_mentions_citation_format(self) -> None:
        messages = self.builder.build("q", [])
        system = messages[0]["content"]
        assert "[file_path:start_line-end_line]" in system

    def test_query_appears_in_user_message(self) -> None:
        messages = self.builder.build("How does login work?", [])
        assert "How does login work?" in messages[1]["content"]

    def test_chunk_header_in_user_message(self) -> None:
        result = _make_result("src/auth.py", 10, 20, "def login(): pass")
        messages = self.builder.build("q", [result])
        user = messages[1]["content"]
        assert "[src/auth.py:10-20]" in user

    def test_chunk_content_in_user_message(self) -> None:
        result = _make_result("src/auth.py", 10, 20, "def login(): pass")
        messages = self.builder.build("q", [result])
        assert "def login(): pass" in messages[1]["content"]

    def test_multiple_chunks_all_present(self) -> None:
        r1 = _make_result("a.py", 1, 5, "x = 1")
        r2 = _make_result("b.py", 10, 15, "y = 2")
        messages = self.builder.build("q", [r1, r2])
        user = messages[1]["content"]
        assert "[a.py:1-5]" in user
        assert "[b.py:10-15]" in user
        assert "x = 1" in user
        assert "y = 2" in user

    def test_context_label_present_when_results_given(self) -> None:
        result = _make_result("a.py", 1, 5, "x = 1")
        messages = self.builder.build("q", [result])
        assert "Context:" in messages[1]["content"]

    def test_no_context_label_when_no_results(self) -> None:
        messages = self.builder.build("q", [])
        assert "Context:" not in messages[1]["content"]

    def test_question_label_present(self) -> None:
        messages = self.builder.build("What is foo?", [])
        assert "Question: What is foo?" in messages[1]["content"]

    def test_chunks_ordered_as_given(self) -> None:
        r1 = _make_result("first.py", 1, 2, "first")
        r2 = _make_result("second.py", 1, 2, "second")
        messages = self.builder.build("q", [r1, r2])
        user = messages[1]["content"]
        assert user.index("first.py") < user.index("second.py")
