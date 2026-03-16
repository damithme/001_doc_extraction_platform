"""
Celery tasks for async document ingestion.

Why Celery instead of running ingestion inline?
- Large PDFs can take 10-30s to embed. Blocking the API thread is bad UX.
- Celery decouples upload (fast) from processing (slow).
- Workers can be scaled independently from the API.

Flow:
  POST /documents/upload
    → saves file bytes in Redis (1hr TTL)
    → creates Document(status="pending") in PostgreSQL
    → dispatches ingest_document.delay(doc_id, filename)
    → returns {doc_id, status: "processing"} immediately

  Celery worker picks up the task:
    → reads file bytes from Redis
    → runs extract → chunk → embed → store
    → updates Document(status="ready") in PostgreSQL
    → deletes file bytes from Redis
"""

import os
import time
import uuid

import redis as _redis
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

celery_app = Celery(
    "docmind",
    broker=REDIS_URL,
    backend=REDIS_URL,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_expires=3600,
)


def _raw_redis() -> _redis.Redis:
    """Binary Redis client (no decode_responses) for storing file bytes."""
    return _redis.Redis.from_url(REDIS_URL)


@celery_app.task(name="docmind.ingest_document", bind=True, max_retries=2)
def ingest_document(self, doc_id: str, filename: str) -> dict:
    """
    Full ingestion pipeline for a single document.

    Args:
        doc_id:   UUID of the Document row already created in PostgreSQL
        filename: Original filename (used to dispatch extractor by extension)
    """
    # Import inside task to avoid circular imports at module load time
    from app.ingestion.chunker import chunk_pages
    from app.ingestion.embedder import embed_batch
    from app.ingestion.extractor import extract
    from app.observability.tracing import ingestion_span
    from app.storage.db import Chunk, Document, SessionLocal
    from app.storage.vector_store import ensure_collection, upsert_chunks

    db = SessionLocal()
    try:
        # ── 1. Fetch file bytes from Redis ────────────────────────────────────
        r = _raw_redis()
        file_bytes: bytes | None = r.get(f"upload:{doc_id}")
        if file_bytes is None:
            raise ValueError(
                f"File bytes for doc_id={doc_id} not found in Redis "
                "(may have expired — TTL is 1 hour)."
            )

        # ── 2. Mark as processing ─────────────────────────────────────────────
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise ValueError(f"Document {doc_id} not found in DB.")
        doc.status = "processing"
        db.commit()

        # ── 3. Extract → chunk → embed → store ───────────────────────────────
        with ingestion_span(doc_id) as span:
            pages = extract(filename, file_bytes)
            chunks = chunk_pages(pages)
            span["chunk_count"] = len(chunks)

            t0 = time.perf_counter()
            texts = [c["text"] for c in chunks]
            vectors = embed_batch(texts)
            span["embedding_latency_s"] = round(time.perf_counter() - t0, 3)

            ensure_collection()
            point_ids = upsert_chunks(doc_id, chunks, vectors)

            for chunk, point_id in zip(chunks, point_ids):
                db.add(Chunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    chunk_index=chunk["chunk_index"],
                    page_num=chunk.get("page_num"),
                    text=chunk["text"],
                    embedding_id=point_id,
                ))

            doc.status = "ready"
            doc.chunk_count = len(chunks)
            db.commit()

        # ── 4. Clean up temp bytes from Redis ─────────────────────────────────
        r.delete(f"upload:{doc_id}")

        return {"doc_id": doc_id, "chunk_count": len(chunks), "status": "ready"}

    except Exception as exc:
        # Mark as failed so the status endpoint reflects it
        try:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.status = "failed"
                db.commit()
        except Exception:
            pass

        # Retry up to max_retries times before giving up
        raise self.retry(exc=exc, countdown=5)

    finally:
        db.close()