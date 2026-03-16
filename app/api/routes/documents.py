import os
import uuid

import redis as _redis
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.storage.db import Document, get_db
from app.storage.vector_store import delete_by_doc_id

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_TYPES = {"pdf", "docx"}
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


def _raw_redis() -> _redis.Redis:
    return _redis.Redis.from_url(REDIS_URL)


@router.post("/upload")
async def upload_document(file: UploadFile, db: Session = Depends(get_db)):
    """
    Upload a document for async ingestion.

    Returns immediately with doc_id and status="processing".
    Poll GET /documents/{doc_id}/status to know when processing is done.

    Why async?
    Embedding a large document can take 10-30s. Blocking the HTTP thread
    is bad UX. Celery picks up the job in the background so the API stays fast.
    """
    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename else ""
    if ext not in ALLOWED_TYPES:
        raise HTTPException(status_code=422, detail=f"Unsupported file type: .{ext}")

    file_bytes = await file.read()
    doc_id = str(uuid.uuid4())

    # Store raw bytes in Redis for the Celery worker to pick up (1-hour TTL)
    _raw_redis().setex(f"upload:{doc_id}", 3600, file_bytes)

    # Create document record immediately so callers can poll the status
    doc = Document(id=doc_id, filename=file.filename, file_type=ext, status="pending")
    db.add(doc)
    db.commit()

    # Dispatch the ingestion task — non-blocking
    from app.tasks import ingest_document
    ingest_document.delay(doc_id, file.filename)

    return {"doc_id": doc_id, "filename": file.filename, "status": "processing"}


@router.get("/{doc_id}/status")
def get_document_status(doc_id: str, db: Session = Depends(get_db)):
    """Poll ingestion status. Status values: pending → processing → ready | failed."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "doc_id": doc.id,
        "filename": doc.filename,
        "status": doc.status,
        "chunk_count": doc.chunk_count,
    }


@router.get("/")
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return [
        {
            "doc_id": d.id,
            "filename": d.filename,
            "file_type": d.file_type,
            "status": d.status,
            "chunk_count": d.chunk_count,
            "created_at": d.created_at,
        }
        for d in docs
    ]


@router.get("/{doc_id}")
def get_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "doc_id": doc.id,
        "filename": doc.filename,
        "file_type": doc.file_type,
        "status": doc.status,
        "chunk_count": doc.chunk_count,
        "created_at": doc.created_at,
    }


@router.delete("/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    delete_by_doc_id(doc_id)  # remove from Qdrant
    db.delete(doc)            # cascades to chunks
    db.commit()

    return {"deleted": doc_id}