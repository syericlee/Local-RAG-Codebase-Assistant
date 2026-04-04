# Local RAG Codebase Assistant

A fully local RAG system: GitHub repo URL → code embeddings → natural language Q&A with file/line citations.
No external APIs. Everything runs on the local machine.

## Stack

| Component | Technology |
|-----------|-----------|
| Vector store | Qdrant |
| Embeddings | nomic-ai/nomic-embed-code (sentence-transformers) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | deepseek-coder:6.7b via Ollama |
| Cache | Redis Stack (exact match + semantic HNSW) |
| Chunking | tree-sitter (AST-aware) + sliding-window fallback |
| API | FastAPI |

## Prerequisites

- Python 3.10+ (developed on 3.10.7). Do not raise the minimum to 3.11+ without
  testing — the `requires-python` field in `pyproject.toml` reflects this.
- Docker + Docker Compose (for Qdrant and Redis Stack)
- Ollama running locally with `deepseek-coder:6.7b` pulled

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Copy and edit config
cp .env.example .env

# 3. Start infrastructure
docker compose up -d

# 4. Run tests
make test

# 5. Index a repo
make index URL=https://github.com/owner/repo

# 6. Query
make query Q="How does authentication work?"
```

## Common Commands

```bash
make dev          # Start app with hot-reload (uvicorn)
make test         # Run full test suite with coverage
make test-unit    # Unit tests only (no Docker needed)
make lint         # ruff + mypy
make index URL=<repo_url>    # Index a GitHub repo
make eval         # Run evaluation pipeline
```

## Project Structure

```
src/rag_assistant/
├── config.py           # All settings via Pydantic BaseSettings (.env)
├── models/             # Pydantic data models (CodeChunk, SearchResult, etc.)
├── ingestion/          # clone → walk → chunk → embed → upsert
├── embedding/          # nomic-embed-code wrapper
├── retrieval/          # vector search → cross-encoder rerank → context assembly
├── generation/         # Ollama client, prompt builder, citation parser
├── cache/              # Two-level Redis cache (exact + semantic)
├── jobs/               # Background job state (Redis hashes)
└── api/                # FastAPI app, routers, middleware
```

## Key Design Decisions

- **Chunk IDs are deterministic**: `sha256(repo_url:file_path:start_line)[:16]` — re-indexing is idempotent.
- **nomic-embed-code requires instruction prefixes**: queries get `"search_query: "`, documents get `"search_document: "`.
- **Semantic cache uses Redis Stack** (RediSearch module) for HNSW vector similarity search with built-in TTL.
- **AST chunking uses tree-sitter 0.23.x** with separate per-language pip packages.
- **Ollama runs on the host** (not containerized); connect via `OLLAMA_BASE_URL=http://localhost:11434`.
- **Background indexing** uses FastAPI `BackgroundTasks` — no Celery needed. Job state is stored in Redis hashes.

## Build Phases

| Phase | Status | What |
|-------|--------|------|
| 1 | ✅ | Foundation: config, models, scaffolding |
| 2 | ✅ | Embedding |
| 3 | ✅ | Chunking (AST + sliding window) |
| 4 | ✅ | Retrieval (Qdrant + reranker) |
| 5 | ✅ | Ingestion pipeline |
| 6 | ✅ | Generation (Ollama + citations) |
| 7 | ✅ | Caching (Redis two-level) |
| 8 | ✅ | API layer |
| 9 | ✅ | Evaluation pipeline |
| 10 | ✅ | Polish (README, Makefile, lint) |

## Dependency Notes

- `torch` is pinned to 2.7.1 to match the system `torchvision` installation.
  Changing it will cause a `torchvision::nms` runtime error.
- `ollama` is pinned to 0.6.1 — older versions conflict with `httpx>=0.28`.
- `qdrant-client` is used without the `[fastembed]` extra.

## Qdrant Point IDs

Qdrant requires point IDs to be unsigned integers or UUIDs. Our chunk IDs are
16-char hex strings, so we convert them with `int(chunk_id, 16)` on every
upsert, search, and delete call. The original string ID is preserved in the
point payload so `CodeChunk` can be reconstructed from search results.

## Chunking Behavior

- AST chunker emits a whole class as one chunk if it fits within `max_tokens`.
  It only recurses into individual methods when the class is too large.
- `SlidingWindowChunker` requires `overlap_tokens > 0` to produce overlapping
  chunks — zero overlap is intentionally non-overlapping and is only used in tests.
- tiktoken tokenizes compound identifiers (e.g. `word0_0`) as multiple tokens.
  Test fixtures that need predictable token counts should use single-character
  repeated tokens (`"a a a a a"`) rather than multi-part words.

## Environment Variables

See `.env.example` for all configuration options. Key settings:

- `EMBED_DEVICE`: set to `cuda` if you have a GPU
- `OLLAMA_MODEL`: change if using a different model
- `CACHE_SEMANTIC_THRESHOLD`: cosine similarity threshold for semantic cache hit (default 0.92)
- `RETRIEVAL_TOP_K` / `RERANK_TOP_N`: vector search candidates / reranked results returned
