# Local RAG Codebase Assistant

A fully local RAG (Retrieval-Augmented Generation) system that indexes a GitHub repository and answers natural language questions about its code with file and line citations. No external APIs — everything runs on your machine.

## How It Works

1. **Index** — clone a repo, chunk the code with AST-aware parsing, embed chunks with `nomic-embed-code`, store vectors in Qdrant
2. **Query** — embed the question, retrieve the top relevant chunks, rerank with a cross-encoder, generate an answer with Ollama, return citations
3. **Cache** — repeated or semantically similar queries are served from a two-level Redis cache without hitting Ollama or Qdrant again

## Stack

| Component | Technology |
|-----------|-----------|
| Vector store | Qdrant |
| Embeddings | nomic-ai/nomic-embed-code |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | deepseek-coder:6.7b via Ollama |
| Cache | Redis Stack (exact + semantic HNSW) |
| Chunking | tree-sitter (AST-aware) + sliding-window fallback |
| API | FastAPI |
| Tracking | SQLite (incremental re-indexing) |

## Prerequisites

- Python 3.10+
- Docker + Docker Compose
- [Ollama](https://ollama.com) running locally with `deepseek-coder:6.7b` pulled

```bash
ollama pull deepseek-coder:6.7b
```

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Copy and configure environment
cp .env.example .env

# 3. Start Qdrant and Redis Stack
docker compose up -d

# 4. Start the API server
make dev

# 5. Index a repository
make index URL=https://github.com/owner/repo

# 6. Ask a question
make query Q="How does authentication work?"
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Answer a question, return full response |
| `POST` | `/query/stream` | Stream answer tokens as SSE events |
| `POST` | `/admin/index` | Start background indexing of a repo |
| `GET` | `/admin/status/{job_id}` | Check indexing job progress |
| `GET` | `/health` | Health check |

### Example: Index a repo

```bash
curl -X POST http://localhost:8000/admin/index \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/owner/repo", "branch": "main"}'
```

Response:
```json
{"job_id": "abc-123", "status": "pending", "repo_url": "..."}
```

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does authentication work?"}'
```

Response:
```json
{
  "answer": "Authentication works by hashing the password [src/auth/login.py:10-25]...",
  "citations": [
    {"file_path": "src/auth/login.py", "start_line": 10, "end_line": 25}
  ],
  "cache_hit": "miss"
}
```

## Common Commands

```bash
make dev          # Start API server with hot-reload
make test         # Run full test suite with coverage
make test-unit    # Unit tests only (no Docker needed)
make lint         # ruff + mypy
make index URL=<repo_url>   # Index a GitHub repo
make query Q="<question>"   # Ask a question
make eval         # Run evaluation pipeline
```

## Evaluation

Measure retrieval and generation quality against a labelled dataset:

```bash
python eval/run_eval.py --dataset eval/dataset.jsonl
```

Metrics reported: Recall@1/5/10, MRR, Faithfulness, Correctness.

To skip generation metrics (no Ollama needed):

```bash
python eval/run_eval.py --dataset eval/dataset.jsonl --skip-generation
```

## Configuration

All settings are controlled via environment variables or a `.env` file. See `.env.example` for the full list. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_DEVICE` | `cpu` | Set to `cuda` for GPU acceleration |
| `OLLAMA_MODEL` | `deepseek-coder:6.7b` | Ollama model to use |
| `RETRIEVAL_TOP_K` | `20` | Vector search candidates |
| `RERANK_TOP_N` | `5` | Results returned after reranking |
| `CACHE_SEMANTIC_THRESHOLD` | `0.92` | Cosine similarity for semantic cache hit |

## Project Structure

```
src/rag_assistant/
├── config.py           # All settings via Pydantic BaseSettings
├── models/             # Pydantic data models
├── ingestion/          # clone → walk → chunk → embed → upsert
├── embedding/          # nomic-embed-code wrapper
├── retrieval/          # vector search → rerank → context assembly
├── generation/         # Ollama client, prompt builder, citation parser
├── cache/              # Two-level Redis cache
├── jobs/               # Background job state
└── api/                # FastAPI app and routers

eval/
├── metrics.py          # Recall@K, MRR, faithfulness, correctness
├── runner.py           # Async evaluation runner
├── run_eval.py         # CLI entry point
└── dataset.jsonl       # Sample evaluation questions
```
