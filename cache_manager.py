"""
Redis Cache Manager for NeuroVest API

Provides caching layer for prediction data to reduce database load
and improve API response times.

Features:
- 5-minute TTL for prediction data
- Automatic cache invalidation
- Fallback to database if Redis unavailable
- JSON serialization for complex objects

Usage:
    from cache_manager import cache

    # Get cached prediction
    prediction = cache.get("prediction:SPY")

    # Set prediction with 5-min TTL
    cache.set("prediction:SPY", data, ttl=300)
"""

import redis
import json
import os
import logging
from typing import Any, Optional
from datetime import timedelta

logger = logging.getLogger(__name__)

class CacheManager:
    """Redis cache manager with fallback"""

    def __init__(self):
        self.enabled = False
        self.client = None

        # Try to connect to Redis
        redis_url = os.getenv('REDIS_URL')
        if redis_url:
            try:
                self.client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2
                )
                # Test connection
                self.client.ping()
                self.enabled = True
                logger.info("✅ Redis cache connected")
            except Exception as e:
                logger.warning(f"⚠️  Redis unavailable: {e}")
                logger.info("   API will run without cache (slower responses)")
                self.enabled = False
        else:
            logger.info("ℹ️  REDIS_URL not set - caching disabled")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/cache disabled
        """
        if not self.enabled:
            return None

        try:
            value = self.client.get(key)
            if value:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(value)
            logger.debug(f"Cache MISS: {key}")
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 300):
        """
        Set value in cache with TTL

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time to live in seconds (default 300 = 5 minutes)
        """
        if not self.enabled:
            return

        try:
            serialized = json.dumps(value)
            self.client.setex(key, ttl, serialized)
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    def delete(self, key: str):
        """Delete key from cache"""
        if not self.enabled:
            return

        try:
            self.client.delete(key)
            logger.debug(f"Cache DELETE: {key}")
        except Exception as e:
            logger.error(f"Cache delete error: {e}")

    def clear_pattern(self, pattern: str):
        """
        Clear all keys matching pattern.

        Uses SCAN instead of KEYS to avoid blocking Redis on large key sets.

        Args:
            pattern: Redis key pattern (e.g., "prediction:*")
        """
        if not self.enabled:
            return

        try:
            deleted = 0
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=100)
                if keys:
                    self.client.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            if deleted:
                logger.info(f"Cache CLEAR: {deleted} keys matching {pattern}")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    def get_stats(self) -> dict:
        """Get cache statistics"""
        if not self.enabled:
            return {
                "enabled": False,
                "message": "Redis not connected"
            }

        try:
            info = self.client.info('stats')
            return {
                "enabled": True,
                "total_commands": info.get('total_commands_processed', 0),
                "keyspace_hits": info.get('keyspace_hits', 0),
                "keyspace_misses": info.get('keyspace_misses', 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get('keyspace_hits', 0),
                    info.get('keyspace_misses', 0)
                )
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"enabled": True, "error": str(e)}

    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage"""
        total = hits + misses
        if total == 0:
            return 0.0
        return round((hits / total) * 100, 2)

    def invalidate_predictions(self):
        """Invalidate all prediction caches (call after new predictions generated)"""
        self.clear_pattern("prediction:*")
        logger.info("♻️  Prediction cache invalidated")


# Global cache instance
cache = CacheManager()
