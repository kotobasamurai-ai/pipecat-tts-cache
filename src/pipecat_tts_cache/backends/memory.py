#
# Copyright (c) 2026, Om Chauhan
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""In-memory LRU cache backend for TTS caching."""

import asyncio
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from pipecat_tts_cache.backends.base import CacheBackend
from pipecat_tts_cache.models import CachedTTSResponse


class MemoryCacheBackend(CacheBackend):
    """In-memory LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000, refresh_ttl_on_hit: bool = True):
        """Initialize in-memory cache backend.

        Args:
            max_size: Maximum number of cache entries to store.
            refresh_ttl_on_hit: When True, each cache hit resets the entry's TTL
                so frequently accessed phrases stay cached indefinitely.
        """
        self._cache: OrderedDict[str, Tuple[CachedTTSResponse, float, int]] = OrderedDict()
        self._max_size = max_size
        self._refresh_ttl_on_hit = refresh_ttl_on_hit
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        logger.debug(f"Initialized MemoryCacheBackend: max_size={max_size}")

    async def get(self, key: str) -> Optional[CachedTTSResponse]:
        """Retrieve cached response, or None if not found/expired.

        When refresh_ttl_on_hit is enabled, a successful get resets the
        entry's expiry so frequently used phrases stay cached.
        """
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            response, expiry, original_ttl = self._cache[key]

            if expiry > 0 and time.time() > expiry:
                del self._cache[key]
                self._misses += 1
                return None

            # Refresh TTL on hit
            if self._refresh_ttl_on_hit and original_ttl > 0:
                new_expiry = time.time() + original_ttl
                self._cache[key] = (response, new_expiry, original_ttl)

            self._cache.move_to_end(key)
            self._hits += 1
            return response

    async def set(self, key: str, response: CachedTTSResponse, ttl: Optional[int] = None) -> bool:
        """Store a TTS response in cache. Returns True on success."""
        async with self._lock:
            try:
                expiry = time.time() + ttl if ttl else 0.0

                if len(self._cache) >= self._max_size and key not in self._cache:
                    self._cache.popitem(last=False)
                    self._evictions += 1

                self._cache[key] = (response, expiry, ttl or 0)
                self._cache.move_to_end(key)
                return True

            except Exception as e:
                logger.error(f"Memory cache set error: {e}")
                return False

    async def delete(self, key: str) -> bool:
        """Delete a cache entry. Returns True if deleted."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self, namespace: Optional[str] = None) -> int:
        """Clear cache entries. Returns number of entries deleted."""
        async with self._lock:
            if namespace is None:
                count = len(self._cache)
                self._cache.clear()
                return count

            to_delete = [k for k in self._cache.keys() if k.startswith(namespace)]
            for key in to_delete:
                del self._cache[key]
            return len(to_delete)

    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        async with self._lock:
            if key not in self._cache:
                return False

            _, expiry, _ = self._cache[key]
            if expiry > 0 and time.time() > expiry:
                del self._cache[key]
                return False
            return True

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        async with self._lock:
            total = self._hits + self._misses
            return {
                "type": "memory",
                "size": len(self._cache),
                "max_size": self._max_size,
                "backend_hits": self._hits,
                "backend_misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }

    async def close(self) -> None:
        """Close backend connections and cleanup resources."""
        async with self._lock:
            self._cache.clear()
