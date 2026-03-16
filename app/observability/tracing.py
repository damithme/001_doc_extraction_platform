"""
Observability helpers for DocMind.

LangSmith auto-traces all LangChain/LangGraph LLM calls when
LANGCHAIN_TRACING_V2=true is set. This module adds timing context managers
for the ingestion pipeline and query path so we can log custom metrics.

Usage:
    with ingestion_span(doc_id) as span:
        # ... do ingestion ...
        span["chunk_count"] = len(chunks)
        span["embedding_latency_s"] = elapsed
"""

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def ingestion_span(doc_id: str) -> Generator[dict, None, None]:
    """
    Time the full ingestion pipeline for a document.
    Yields a dict you fill in with metrics; prints a structured log on exit.

    Metrics to set:
        chunk_count (int)
        embedding_latency_s (float)
    """
    t0 = time.perf_counter()
    data: dict = {}
    yield data
    elapsed = round(time.perf_counter() - t0, 3)
    print(
        f"[span:ingest] doc_id={doc_id} "
        f"chunks={data.get('chunk_count', '?')} "
        f"embed_latency={data.get('embedding_latency_s', '?')}s "
        f"total={elapsed}s"
    )


@contextmanager
def query_span(query: str) -> Generator[dict, None, None]:
    """
    Time the full agent query path.
    Yields a dict you fill in with metrics; prints a structured log on exit.

    Metrics to set:
        top_k_chunks (int)    — number of chunks returned by initial retrieve
        tool_calls_made (int) — number of tool calls Claude made
    """
    t0 = time.perf_counter()
    data: dict = {}
    yield data
    elapsed = round(time.perf_counter() - t0, 3)
    print(
        f"[span:query] query={query!r} "
        f"top_k={data.get('top_k_chunks', '?')} "
        f"tool_calls={data.get('tool_calls_made', '?')} "
        f"total={elapsed}s"
    )