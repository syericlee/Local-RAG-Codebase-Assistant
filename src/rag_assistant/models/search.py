from pydantic import BaseModel

from .chunk import CodeChunk


class SearchResult(BaseModel):
    chunk: CodeChunk
    vector_score: float     # cosine similarity from Qdrant (0–1)


class RerankedResult(BaseModel):
    chunk: CodeChunk
    vector_score: float
    rerank_score: float     # cross-encoder logit (higher = more relevant)


class RetrievalResponse(BaseModel):
    results: list[RerankedResult]
    total_tokens_in_context: int
    query_embedding_ms: float
    vector_search_ms: float
    rerank_ms: float
