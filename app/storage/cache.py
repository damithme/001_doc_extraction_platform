"""
Redis-backed embedding cache.

Caches embedding vectors by SHA-256 hash of the input text with a 24-hour TTL.
This avoids paying for duplicate OpenAI embedding calls on repeated text.

Usage:
    vector = get_cached_embedding(text)
    if vector is None:
        vector = call_openai(text)
        cache_embedding(text, vector)
"""

import hashlib
import json
import os

import redis

_client: redis.Redis | None = None
_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_TTL_SECONDS = 86_400  # 24 hours


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(_REDIS_URL, decode_responses=True)
    return _client


def _key(text: str) -> str:
    return "emb:" + hashlib.sha256(text.encode()).hexdigest()


def get_cached_embedding(text: str) -> list[float] | None:
    """Return cached embedding vector or None on miss."""
    try:
        raw = _get_client().get(_key(text))
        if raw:
            print(f"[cache] HIT  key={_key(text)[:20]}...")
            return json.loads(raw)
        print(f"[cache] MISS key={_key(text)[:20]}...")
    except Exception as e:
        print(f"[cache] Redis error (get): {e}")
    return None


def cache_embedding(text: str, vector: list[float]) -> None:
    """Store embedding vector in Redis with a 24-hour TTL."""
    try:
        _get_client().setex(_key(text), _TTL_SECONDS, json.dumps(vector))
    except Exception as e:
        print(f"[cache] Redis error (set): {e}")