import os
import time

from openai import OpenAI

from app.storage.cache import cache_embedding, get_cached_embedding

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def embed_text(text: str) -> list[float]:
    """Embed a single string. Checks Redis cache first. Retries up to 2 times on failure."""
    cached = get_cached_embedding(text)
    if cached is not None:
        return cached

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            response = _get_client().embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            vector = response.data[0].embedding
            cache_embedding(text, vector)
            return vector
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(1)

    raise RuntimeError(f"Embedding failed after 3 attempts: {last_err}")


def embed_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed a list of texts in batches. Cache-aware: only calls OpenAI for uncached texts."""
    all_vectors: list[list[float] | None] = [None] * len(texts)

    # Separate cached from uncached
    uncached_indices: list[int] = []
    for i, text in enumerate(texts):
        cached = get_cached_embedding(text)
        if cached is not None:
            all_vectors[i] = cached
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        return all_vectors  # type: ignore[return-value]

    # Batch-embed only the uncached ones
    uncached_texts = [texts[i] for i in uncached_indices]
    client = _get_client()

    new_vectors: list[list[float]] = []
    last_err: Exception | None = None

    for start in range(0, len(uncached_texts), batch_size):
        batch = uncached_texts[start : start + batch_size]
        for attempt in range(3):
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                )
                new_vectors.extend([r.embedding for r in response.data])
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(1)
        else:
            raise RuntimeError(f"Batch embedding failed after 3 attempts: {last_err}")

    # Write new vectors back to cache and result list
    for idx, vector in zip(uncached_indices, new_vectors):
        cache_embedding(texts[idx], vector)
        all_vectors[idx] = vector

    return all_vectors  # type: ignore[return-value]