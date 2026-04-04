from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from .search import RetrievalResponse


class Citation(BaseModel):
    file_path: str
    start_line: int
    end_line: int


class QueryRequest(BaseModel):
    query: str
    repo_url: str | None = None      # filter search to a specific indexed repo
    top_k: int | None = None         # override default rerank_top_n


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    cache_hit: Literal["exact", "semantic", "miss"]
    retrieval: RetrievalResponse | None = None   # None when served from cache


class IndexRequest(BaseModel):
    repo_url: str
    branch: str = "main"
    force_reindex: bool = False     # ignore SQLite tracker, re-chunk everything


class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    repo_url: str
    created_at: datetime
    updated_at: datetime
    files_indexed: int = 0
    chunks_upserted: int = 0
    error: str | None = None
