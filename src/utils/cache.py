"""Cache management with LRU eviction policy.

This module provides cache size management and LRU (Least Recently Used)
eviction for the photo generation pipeline's prompt caching system.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Metadata for a cached entry.
    
    Attributes:
        key: SHA256 cache key
        size_bytes: Size of cached content in bytes
        created_at: ISO8601 timestamp when entry was created
        accessed_at: ISO8601 timestamp of last access
        hit_count: Number of times this entry was accessed
    """
    key: str
    size_bytes: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    accessed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hit_count: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "key": self.key,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "hit_count": self.hit_count,
        }
    
    @staticmethod
    def from_dict(data: dict[str, Any]) -> CacheEntry:
        """Create from dictionary."""
        return CacheEntry(
            key=str(data.get("key", "")),
            size_bytes=int(data.get("size_bytes", 0)),
            created_at=str(data.get("created_at", "")),
            accessed_at=str(data.get("accessed_at", "")),
            hit_count=int(data.get("hit_count", 0)),
        )


class CacheManager:
    """LRU cache manager for prompt caching.
    
    Features:
    - Tracks cache entry metadata (size, timestamps, hit count)
    - Automatic LRU eviction when max size exceeded
    - Per-user cache isolation
    - Cache statistics (total size, hit rate, entry count)
    """
    
    METADATA_FILENAME = ".cache_metadata.json"
    
    def __init__(
        self,
        cache_dir: Path,
        max_size_mb: int = 500,
        eviction_policy: str = "lru",
    ):
        """Initialize cache manager.
        
        Parameters:
            cache_dir: Root directory for cache storage
            max_size_mb: Maximum cache size in megabytes (per user)
            eviction_policy: Eviction strategy ("lru" is currently supported)
            
        Raises:
            ValueError: If parameters are invalid
        """
        if max_size_mb <= 0:
            raise ValueError(f"max_size_mb must be positive, got {max_size_mb}")
        if eviction_policy != "lru":
            raise ValueError(f"Unsupported eviction policy: {eviction_policy}")
        
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.entries: dict[Path, CacheEntry] = {}
        
        self._load_metadata()
    
    def get(self, cache_path: Path) -> CacheEntry | None:
        """Retrieve and update cache entry.
        
        Parameters:
            cache_path: Path to cached file
            
        Returns:
            CacheEntry if found and valid, None otherwise
        """
        if cache_path not in self.entries:
            return None
        
        entry = self.entries[cache_path]
        now = datetime.now(timezone.utc).isoformat()
        entry.accessed_at = now
        entry.hit_count += 1
        
        logger.debug(
            "Cache hit: %s (hit_count=%d, size=%d bytes)",
            cache_path.name,
            entry.hit_count,
            entry.size_bytes,
        )
        
        return entry
    
    def put(self, cache_path: Path, content: str, cache_key: str) -> None:
        """Record cache entry and evict if necessary.
        
        Parameters:
            cache_path: Path to cached file
            content: Content being cached
            cache_key: SHA256 cache key
            
        Side effects:
            - Updates metadata
            - May evict old entries if size limit exceeded
        """
        size_bytes = len(content.encode("utf-8"))
        now = datetime.now(timezone.utc).isoformat()
        
        entry = CacheEntry(
            key=cache_key,
            size_bytes=size_bytes,
            created_at=now,
            accessed_at=now,
            hit_count=0,
        )
        
        self.entries[cache_path] = entry
        
        logger.debug(
            "Cache put: %s (size=%d bytes, key=%s...)",
            cache_path.name,
            size_bytes,
            cache_key[:8],
        )
        
        # Check if eviction is needed
        total_size = self._total_size()
        if total_size > self.max_size_bytes:
            self._evict_lru(total_size)
    
    def _total_size(self) -> int:
        """Calculate total cache size in bytes."""
        return sum(e.size_bytes for e in self.entries.values())
    
    def _evict_lru(self, current_size: int) -> None:
        """Evict least recently used entries until under limit.
        
        Parameters:
            current_size: Current total cache size in bytes
        """
        target_size = int(self.max_size_bytes * 0.8)  # Target 80% of max
        to_evict_bytes = current_size - target_size
        
        logger.info(
            "Cache eviction triggered: %s→%s MB (target: %s MB)",
            current_size / (1024 * 1024),
            target_size / (1024 * 1024),
            self.max_size_bytes / (1024 * 1024),
        )
        
        # Sort by accessed_at (least recently used first)
        sorted_entries = sorted(
            self.entries.items(),
            key=lambda x: x[1].accessed_at,
        )
        
        evicted = 0
        evicted_bytes = 0
        
        for cache_path, entry in sorted_entries:
            if evicted_bytes >= to_evict_bytes:
                break
            
            # Delete cache files
            try:
                if cache_path.exists():
                    cache_path.unlink()
                    logger.debug("Evicted cache file: %s", cache_path.name)
                
                # Delete hash file
                hash_path = cache_path.with_suffix(cache_path.suffix + ".sha256")
                if hash_path.exists():
                    hash_path.unlink()
                
                # Delete layout file if present
                layout_path = cache_path.with_suffix(cache_path.suffix + ".layout.json")
                if layout_path.exists():
                    layout_path.unlink()
                
                del self.entries[cache_path]
                evicted += 1
                evicted_bytes += entry.size_bytes
            except Exception as e:
                logger.warning("Error evicting cache file %s: %s", cache_path, e)
        
        logger.info(
            "Cache eviction complete: removed %d entries (%s MB)",
            evicted,
            evicted_bytes / (1024 * 1024),
        )
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with keys:
            - total_size_mb: Total cache size in MB
            - entry_count: Number of cached entries
            - max_size_mb: Maximum allowed size in MB
            - usage_percent: Percentage of max size used
            - avg_hit_count: Average hit count per entry
        """
        total_size = self._total_size()
        entry_count = len(self.entries)
        avg_hits = (
            sum(e.hit_count for e in self.entries.values()) / entry_count
            if entry_count > 0
            else 0
        )
        
        return {
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "entry_count": entry_count,
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "usage_percent": round(100 * total_size / self.max_size_bytes, 1),
            "avg_hit_count": round(avg_hits, 1),
        }
    
    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        metadata_path = self.cache_dir / self.METADATA_FILENAME
        if not metadata_path.exists():
            logger.debug("No cache metadata found at %s", metadata_path)
            return
        
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            for entry_path, entry_data in data.items():
                cache_path = Path(entry_path)
                
                # Validate file still exists
                if not cache_path.exists():
                    logger.debug("Cache file no longer exists, skipping: %s", cache_path)
                    continue
                
                try:
                    self.entries[cache_path] = CacheEntry.from_dict(entry_data)
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning("Invalid cache metadata for %s: %s", cache_path, e)
            
            logger.info("Loaded %d cache entries from metadata", len(self.entries))
        except Exception as e:
            logger.error("Error loading cache metadata: %s", e)
    
    def save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = self.cache_dir / self.METADATA_FILENAME
            
            data = {str(path): entry.to_dict() for path, entry in self.entries.items()}
            metadata_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            logger.debug("Saved cache metadata (%d entries)", len(self.entries))
        except Exception as e:
            logger.error("Error saving cache metadata: %s", e)
    
    def clear(self) -> None:
        """Clear all cache entries and metadata."""
        logger.info("Clearing all cache entries (%d entries)", len(self.entries))
        
        for cache_path in list(self.entries.keys()):
            try:
                if cache_path.exists():
                    cache_path.unlink()
                
                hash_path = cache_path.with_suffix(cache_path.suffix + ".sha256")
                if hash_path.exists():
                    hash_path.unlink()
            except Exception as e:
                logger.warning("Error deleting cache file %s: %s", cache_path, e)
        
        self.entries.clear()
        
        # Remove metadata file
        metadata_path = self.cache_dir / self.METADATA_FILENAME
        if metadata_path.exists():
            try:
                metadata_path.unlink()
            except Exception as e:
                logger.warning("Error deleting metadata: %s", e)
