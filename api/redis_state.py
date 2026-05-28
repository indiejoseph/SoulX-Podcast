"""
Redis client helpers for API coordination state.

Redis is optional for local development. When REDIS_URL is configured, task
metadata and prompt-cache metadata use Redis while CUDA tensors remain local to
the API process.
"""
import logging
from typing import Any

from api.config import config

logger = logging.getLogger(__name__)

_sync_client: Any = None
_async_client: Any = None


def redis_enabled() -> bool:
    """Return whether Redis-backed state is configured."""
    return bool(config.redis_url)


def redis_key(*parts: str) -> str:
    """Build a namespaced Redis key."""
    cleaned = [str(part).strip(":") for part in parts if str(part)]
    return ":".join([config.redis_key_prefix, *cleaned])


def get_redis_client():
    """Return a shared synchronous Redis client, or None when disabled."""
    global _sync_client
    if not redis_enabled():
        return None
    if _sync_client is None:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("REDIS_URL is set but the redis package is not installed") from exc
        _sync_client = redis.Redis.from_url(config.redis_url, decode_responses=True)
    return _sync_client


async def get_async_redis_client():
    """Return a shared asyncio Redis client, or None when disabled."""
    global _async_client
    if not redis_enabled():
        return None
    if _async_client is None:
        try:
            import redis.asyncio as redis
        except ImportError as exc:
            raise RuntimeError("REDIS_URL is set but the redis package is not installed") from exc
        _async_client = redis.Redis.from_url(config.redis_url, decode_responses=True)
    return _async_client


async def close_async_redis_client() -> None:
    """Close the shared asyncio Redis connection on shutdown."""
    global _async_client
    if _async_client is None:
        return
    await _async_client.aclose()
    _async_client = None


def ping_redis() -> bool | None:
    """Return Redis availability, or None when Redis is not configured."""
    client = get_redis_client()
    if client is None:
        return None
    try:
        return bool(client.ping())
    except Exception:
        logger.exception("Redis health check failed")
        return False


async def async_ping_redis() -> bool | None:
    """Return Redis availability without blocking the event loop."""
    client = await get_async_redis_client()
    if client is None:
        return None
    try:
        return bool(await client.ping())
    except Exception:
        logger.exception("Redis health check failed")
        return False
