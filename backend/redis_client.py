# ============================================================
# redis_client.py — Async Redis Singleton for Image Storage
# ============================================================
# Implements the "Lean State" pattern: large image blobs go to
# Redis, and only lightweight UUID pointers enter LangGraph state.
# ============================================================

import uuid
import redis.asyncio as aioredis
from config import REDIS_URL

# ── Singleton Connection Pool ─────────────────────────────────
_redis_pool: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """Get or create the async Redis connection. Falls back to fakeredis if no server is running."""
    global _redis_pool
    if _redis_pool is None:
        try:
            # Try connecting to real Redis
            _redis_pool = aioredis.from_url(
                REDIS_URL,
                decode_responses=False,
                max_connections=20,
            )
            await _redis_pool.ping()
            print("✅ Connected to Redis server")
        except Exception:
            # Fallback to in-memory fakeredis (no Docker needed for dev)
            import fakeredis.aioredis
            _redis_pool = fakeredis.aioredis.FakeRedis(decode_responses=False)
            print("⚠️  No Redis server found — using in-memory fakeredis (dev mode)")
    return _redis_pool


async def store_image(session_id: str, suffix: str, image_bytes: bytes, ttl: int = 600) -> str:
    """
    Store an image blob in Redis and return the reference key.
    
    Args:
        session_id: Unique session identifier
        suffix: Key suffix (e.g., 'full' or 'snippet')
        image_bytes: Raw image bytes to store
        ttl: Time-to-live in seconds (default 10 minutes)
    
    Returns:
        The Redis key string (e.g., "abc123_full")
    """
    r = await get_redis()
    ref_key = f"{session_id}_{suffix}"
    await r.set(ref_key, image_bytes, ex=ttl)
    return ref_key


async def get_image(ref_key: str) -> bytes | None:
    """Retrieve an image blob from Redis by its reference key."""
    r = await get_redis()
    return await r.get(ref_key)


async def cleanup_session(session_id: str):
    """Remove all image blobs for a session."""
    r = await get_redis()
    for suffix in ["full", "snippet"]:
        await r.delete(f"{session_id}_{suffix}")


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return uuid.uuid4().hex[:12]
