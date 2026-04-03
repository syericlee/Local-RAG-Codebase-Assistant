import pytest

from rag_assistant.generation.citation_parser import CitationParser


@pytest.fixture
def parser() -> CitationParser:
    return CitationParser()


class TestCitationParser:
    def test_single_citation(self, parser: CitationParser) -> None:
        answer = "See [src/auth.py:10-25] for details."
        citations = parser.parse(answer)
        assert len(citations) == 1
        assert citations[0].file_path == "src/auth.py"
        assert citations[0].start_line == 10
        assert citations[0].end_line == 25

    def test_multiple_citations(self, parser: CitationParser) -> None:
        answer = "Login [src/auth.py:10-25] uses hashing [src/utils.py:5-12]."
        citations = parser.parse(answer)
        assert len(citations) == 2
        assert citations[0].file_path == "src/auth.py"
        assert citations[1].file_path == "src/utils.py"

    def test_no_citations(self, parser: CitationParser) -> None:
        answer = "There is no relevant code in the context."
        assert parser.parse(answer) == []

    def test_preserves_order(self, parser: CitationParser) -> None:
        answer = "[b.py:1-2] then [a.py:3-4]"
        citations = parser.parse(answer)
        assert citations[0].file_path == "b.py"
        assert citations[1].file_path == "a.py"

    def test_deduplicates_identical_citations(self, parser: CitationParser) -> None:
        answer = "[src/auth.py:10-25] and again [src/auth.py:10-25]"
        citations = parser.parse(answer)
        assert len(citations) == 1

    def test_same_file_different_lines_not_deduplicated(self, parser: CitationParser) -> None:
        answer = "[src/auth.py:10-25] and [src/auth.py:30-40]"
        citations = parser.parse(answer)
        assert len(citations) == 2

    def test_nested_path_with_slashes(self, parser: CitationParser) -> None:
        answer = "See [src/api/routers/query.py:5-10]."
        citations = parser.parse(answer)
        assert citations[0].file_path == "src/api/routers/query.py"

    def test_single_line_citation(self, parser: CitationParser) -> None:
        answer = "Defined at [main.py:42-42]."
        citations = parser.parse(answer)
        assert citations[0].start_line == 42
        assert citations[0].end_line == 42

    def test_citation_mid_sentence(self, parser: CitationParser) -> None:
        answer = "The function [foo.py:1-5] does X and [bar.py:10-20] does Y."
        citations = parser.parse(answer)
        assert len(citations) == 2

    def test_empty_string(self, parser: CitationParser) -> None:
        assert parser.parse("") == []

    def test_malformed_no_lines_ignored(self, parser: CitationParser) -> None:
        # Missing line range — should not match
        answer = "See [src/auth.py] for details."
        assert parser.parse(answer) == []
