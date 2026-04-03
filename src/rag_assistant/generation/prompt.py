from __future__ import annotations

from rag_assistant.models.search import RerankedResult

_SYSTEM_PROMPT = """\
You are an expert code assistant. Answer the user's question using only the \
provided code context below. Do not use outside knowledge.

When you reference a specific piece of code, cite its source inline using \
exactly this format: [file_path:start_line-end_line]
For example: [src/auth/login.py:10-25]

If the context does not contain enough information to answer, say so clearly.\
"""


def _format_chunk(result: RerankedResult) -> str:
    chunk = result.chunk
    header = f"[{chunk.file_path}:{chunk.start_line}-{chunk.end_line}]"
    return f"{header}\n{chunk.content}"


class PromptBuilder:
    """Builds the message list sent to Ollama for a RAG query."""

    def build(
        self,
        query: str,
        results: list[RerankedResult],
    ) -> list[dict[str, str]]:
        """Return a list of {role, content} messages.

        Args:
            query: The user's natural language question.
            results: Reranked retrieval results, ordered best-first.

        Returns:
            [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        """
        context_blocks = "\n\n".join(_format_chunk(r) for r in results)

        if context_blocks:
            user_content = f"Context:\n\n{context_blocks}\n\nQuestion: {query}"
        else:
            user_content = f"Question: {query}"

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
