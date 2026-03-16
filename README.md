# DocMind — Intelligent Document Q&A Agent

> Upload PDFs and DOCX files, ask questions in plain English, get answers with source citations — powered by Claude, LangGraph, and a production-grade retrieval pipeline.

---

## What it does

1. **Upload** a PDF or DOCX → document is chunked, embedded, and stored in a vector database
2. **Ask a question** → a multi-step AI agent retrieves relevant passages, reasons over them, and returns an answer with exact source citations
3. **Every decision is traced** in LangSmith — latency, token counts, and tool calls visible per query

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FastAPI (port 8000)                 │
│   POST /documents/upload    POST /query/                 │
└────────────┬───────────────────────┬────────────────────┘
             │                       │
             ▼                       ▼
   ┌──────────────────┐    ┌──────────────────────────────┐
   │  Celery Worker   │    │       LangGraph Agent        │
   │                  │    │                              │
   │ 1. Extract text  │    │  retrieve ──► reason         │
   │    (PyMuPDF /    │    │               │              │
   │    python-docx)  │    │          tool_executor       │
   │ 2. Chunk text    │    │               │              │
   │ 3. Embed chunks  │    │          reason (loop)       │
   │ 4. Store vectors │    │               │              │
   └──────┬───────────┘    │           answer             │
          │                └──────────────┬───────────────┘
          │                               │
          ▼                               ▼
   ┌─────────────┐  ┌──────────┐  ┌─────────────┐
   │   Qdrant    │  │PostgreSQL│  │    Redis     │
   │ (vectors +  │  │(doc meta │  │ (embedding   │
   │  metadata)  │  │+ chunks) │  │  cache +     │
   │  port 6333  │  │ port 5432│  │  task queue) │
   └─────────────┘  └──────────┘  │  port 6379  │
                                   └─────────────┘
                                          │
                                   ┌──────▼──────┐
                                   │  LangSmith  │
                                   │  (traces)   │
                                   └─────────────┘
```

### Agent Flow

```
User Query
    │
    ▼
[retrieve]  ── embed query → search Qdrant top-5 chunks automatically
    │
    ▼
[reason]    ── Claude reads context + bound tools, decides next action
    │
    ├── tool call? ──► [tool_executor] ──► [reason]  (loop until done)
    │
    └── final answer? ──► [answer] ── extract citations ──► Response
```

### Tools available to Claude

| Tool | When Claude uses it |
|---|---|
| `vector_search` | Primary — semantic search across all chunks |
| `get_chunk_by_id` | Fetching a specific passage by ID |
| `summarize_doc` | User asks for a summary of a document |
| `compare_docs` | User asks to compare two or more documents |

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| **LLM** | Claude (`claude-sonnet-4-6`) via Anthropic API | — |
| **Agent framework** | LangGraph | ≥ 1.0.10 |
| **LLM client** | LangChain Anthropic | ≥ 1.3.4 |
| **Embeddings** | OpenAI `text-embedding-3-small` | openai ≥ 2.24.0 |
| **PDF parsing** | PyMuPDF | ≥ 1.27.1 |
| **DOCX parsing** | python-docx | ≥ 1.2.0 |
| **Chunking** | LangChain RecursiveCharacterTextSplitter | langchain ≥ 1.2.10 |
| **Vector DB** | Qdrant (Docker) | latest |
| **Metadata DB** | PostgreSQL 16 (Docker) | SQLAlchemy ≥ 2.0.48 |
| **Cache** | Redis 7 (Docker) | redis ≥ 7.2.1 |
| **Task queue** | Celery | ≥ 5.6.2 |
| **API** | FastAPI + Uvicorn | ≥ 0.135.1 / ≥ 0.41.0 |
| **Observability** | LangSmith | ≥ 0.7.11 |
| **Package manager** | uv | — |
| **Runtime** | Python | ≥ 3.14 |

---

## Project Structure

```
docmind/
├── main.py                      # FastAPI app entrypoint
├── app/
│   ├── api/routes/
│   │   ├── documents.py         # Upload, list, delete, status endpoints
│   │   └── query.py             # Q&A endpoint
│   ├── agent/
│   │   ├── graph.py             # LangGraph agent (4 nodes + edge logic)
│   │   ├── tools.py             # 4 LangChain tools Claude can call
│   │   └── prompts.py           # System prompt + citation format
│   ├── ingestion/
│   │   ├── extractor.py         # PDF + DOCX text extraction
│   │   ├── chunker.py           # RecursiveCharacterTextSplitter wrapper
│   │   └── embedder.py          # OpenAI embeddings + Redis cache
│   ├── storage/
│   │   ├── vector_store.py      # Qdrant client wrapper
│   │   ├── db.py                # SQLAlchemy models + session
│   │   └── cache.py             # Redis embedding cache
│   ├── observability/
│   │   └── tracing.py           # Timing context managers
│   └── tasks.py                 # Celery task: async ingestion
├── tests/
│   ├── test_ingestion.py        # Unit tests (no services needed)
│   ├── test_agent.py            # Unit tests (no services needed)
│   └── test_api.py              # Integration tests (requires docker-compose)
├── Dockerfile                   # FastAPI app image
├── Dockerfile.worker            # Celery worker image
├── docker-compose.yml           # Full stack: Qdrant + PostgreSQL + Redis + API + Worker
└── pyproject.toml
```

---

## Getting Started

### Prerequisites

- [Docker + Docker Compose](https://docs.docker.com/get-docker/)
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Anthropic API key
- OpenAI API key

### 1. Clone and configure

```bash
git clone https://github.com/your-username/docmind.git
cd docmind

cp .env.example .env
# Fill in your keys:
#   ANTHROPIC_API_KEY=...
#   OPENAI_API_KEY=...
#   LANGSMITH_API_KEY=...   (optional, for tracing)
```

### 2. Option A — Run everything in Docker

```bash
docker-compose up --build
```

API available at `http://localhost:8000`. Swagger UI at `http://localhost:8000/docs`.

### 2. Option B — Local development

```bash
# Start infrastructure
docker-compose up -d qdrant postgres redis

# Install dependencies
uv sync

# Start API
uv run uvicorn main:app --reload

# Start Celery worker (separate terminal)
uv run celery -A app.tasks worker --loglevel=info
```

---

## API Reference

### Documents

#### `POST /documents/upload`
Upload a PDF or DOCX for async ingestion. Returns immediately — poll status to know when ready.

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@contract.pdf"
```

**Response:**
```json
{
  "doc_id": "3f2a8b1c-...",
  "filename": "contract.pdf",
  "status": "processing"
}
```

---

#### `GET /documents/{doc_id}/status`
Poll ingestion status. Values: `pending → processing → ready | failed`

```bash
curl http://localhost:8000/documents/3f2a8b1c-.../status
```

**Response:**
```json
{
  "doc_id": "3f2a8b1c-...",
  "filename": "contract.pdf",
  "status": "ready",
  "chunk_count": 42
}
```

---

#### `GET /documents/`
List all uploaded documents.

```bash
curl http://localhost:8000/documents/
```

**Response:**
```json
[
  {
    "doc_id": "3f2a8b1c-...",
    "filename": "contract.pdf",
    "file_type": "pdf",
    "status": "ready",
    "chunk_count": 42,
    "created_at": "2026-03-15T10:00:00"
  }
]
```

---

#### `GET /documents/{doc_id}`
Get details for a single document.

```bash
curl http://localhost:8000/documents/3f2a8b1c-...
```

---

#### `DELETE /documents/{doc_id}`
Remove a document from PostgreSQL and Qdrant.

```bash
curl -X DELETE http://localhost:8000/documents/3f2a8b1c-...
```

**Response:**
```json
{ "deleted": "3f2a8b1c-..." }
```

---

### Query

#### `POST /query/`
Ask a question against uploaded documents. The agent retrieves context, reasons over it, and returns an answer with citations.

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the termination clauses?"}'
```

**Scope to specific documents (optional):**
```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Compare the liability sections",
    "doc_ids": ["3f2a8b1c-...", "7a9d3e2f-..."]
  }'
```

**Response:**
```json
{
  "answer": "The contract can be terminated with 30 days written notice by either party... [Source: doc_id=3f2a8b1c, page=4, chunk=12]",
  "citations": [
    {
      "chunk_id": "f1a2b3c4-...",
      "doc_id": "3f2a8b1c-...",
      "chunk_index": 12,
      "page_num": 4
    }
  ]
}
```

---

#### `GET /health`
Health check.

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Running Tests

```bash
# Unit tests only (no services needed)
uv run pytest tests/test_ingestion.py tests/test_agent.py -v

# Integration tests (requires docker-compose up -d)
uv run pytest tests/test_api.py -v

# All tests
uv run pytest tests/ -v
```

---

## Observability

When `LANGSMITH_API_KEY` is set, every query produces a full trace in [LangSmith](https://smith.langchain.com):

```
query_documents
├── retrieve        (embed + Qdrant search, ~100-200ms)
├── ChatAnthropic   (LLM call 1, tokens + latency)
│   └── vector_search (tool call — input query, output chunks)
├── ChatAnthropic   (LLM call 2, tokens + latency)
└── answer          (citation extraction)
```

The terminal also logs structured spans on every upload and query:

```
[cache] MISS key=3f2a8b1c...
[span:ingest] doc_id=abc chunks=42 embed_latency=2.1s total=4.3s

[cache] HIT  key=3f2a8b1c...
[span:query] query='What are the risks?' top_k=5 tool_calls=2 total=3.8s
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `OPENAI_API_KEY` | Yes | For `text-embedding-3-small` |
| `LANGSMITH_API_KEY` | No | Enables LangSmith tracing |
| `LANGSMITH_PROJECT` | No | LangSmith project name (default: `docmind`) |
| `QDRANT_URL` | No | Default: `http://localhost:6333` |
| `POSTGRES_DSN` | No | Default: `postgresql://docmind:docmind@localhost:5432/docmind` |
| `REDIS_URL` | No | Default: `redis://localhost:6379` |

---

## Key Design Decisions

**Why LangGraph instead of a simple chain?**
LangGraph gives us an explicit, inspectable loop — Claude can call tools multiple times before answering. A chain would force a fixed number of retrieval steps.

**Why Celery for ingestion?**
Embedding a large document takes 5-30s. Blocking an HTTP thread is bad UX and limits concurrency. Celery decouples upload (instant) from processing (async).

**Why Redis for embedding cache?**
The same text chunk queried repeatedly (e.g. from different questions) would be re-embedded unnecessarily. A SHA-256 keyed Redis cache cuts OpenAI costs and speeds up repeated queries significantly.

**Chunk size: 500 tokens, 50 overlap**
Small enough for precise retrieval, large enough to preserve sentence context. The 50-token overlap prevents answers being cut at chunk boundaries.

---

## License

MIT