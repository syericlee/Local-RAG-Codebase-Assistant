from __future__ import annotations

import re

from rag_assistant.models.api import Citation

# Matches [some/file/path.py:10-25]
_CITATION_RE = re.compile(r"\[([^\]]+?):(\d+)-(\d+)\]")


class CitationParser:
    """Extracts structured Citation objects from an LLM answer string."""

    def parse(self, answer: str) -> list[Citation]:
        """Return all citations found in answer, in order of appearance.

        Duplicate citations (same file + lines) are deduplicated while
        preserving first-occurrence order.
        """
        seen: set[tuple[str, int, int]] = set()
        citations: list[Citation] = []

        for match in _CITATION_RE.finditer(answer):
            file_path = match.group(1)
            start_line = int(match.group(2))
            end_line = int(match.group(3))

            key = (file_path, start_line, end_line)
            if key not in seen:
                seen.add(key)
                citations.append(
                    Citation(
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                    )
                )

        return citations
