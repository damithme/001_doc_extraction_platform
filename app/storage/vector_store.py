import os
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "docmind_chunks"
VECTOR_SIZE = 1536  # text-embedding-3-small


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection() -> None:
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection: {COLLECTION_NAME}")


def upsert_chunks(doc_id: str, chunks: list[dict], vectors: list[list[float]]) -> list[str]:
    """Store chunks with their embeddings. Returns list of point IDs."""
    client = get_client()
    points = []
    point_ids = []

    for chunk, vector in zip(chunks, vectors):
        point_id = str(uuid.uuid4())
        point_ids.append(point_id)
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "doc_id": doc_id,
                    "chunk_index": chunk["chunk_index"],
                    "page_num": chunk.get("page_num"),
                    "text": chunk["text"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return point_ids


def search(query_vector: list[float], top_k: int = 5, doc_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Semantic search. Optionally filter to specific doc_ids."""
    client = get_client()

    query_filter = None
    if doc_ids:
        from qdrant_client.models import FieldCondition, Filter, MatchAny
        query_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))]
        )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    return [
        {
            "id": str(r.id),
            "score": r.score,
            "text": r.payload["text"],
            "doc_id": r.payload["doc_id"],
            "chunk_index": r.payload["chunk_index"],
            "page_num": r.payload.get("page_num"),
        }
        for r in response.points
    ]


def get_chunk_by_id(point_id: str) -> dict[str, Any] | None:
    client = get_client()
    results = client.retrieve(collection_name=COLLECTION_NAME, ids=[point_id], with_payload=True)
    if not results:
        return None
    r = results[0]
    return {
        "id": str(r.id),
        "text": r.payload["text"],
        "doc_id": r.payload["doc_id"],
        "chunk_index": r.payload["chunk_index"],
        "page_num": r.payload.get("page_num"),
    }


def get_chunks_for_doc(doc_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Retrieve up to `limit` chunks for a document, ordered by chunk_index."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    client = get_client()
    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
        limit=limit,
        with_payload=True,
    )
    chunks = [
        {
            "id": str(r.id),
            "text": r.payload["text"],
            "doc_id": r.payload["doc_id"],
            "chunk_index": r.payload["chunk_index"],
            "page_num": r.payload.get("page_num"),
        }
        for r in results
    ]
    return sorted(chunks, key=lambda c: c["chunk_index"])


def delete_by_doc_id(doc_id: str) -> None:
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    client = get_client()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
    )