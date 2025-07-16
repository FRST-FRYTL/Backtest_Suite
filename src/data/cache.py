"""Data caching utilities for efficient data management."""

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass

import pandas as pd
from diskcache import Cache


class CacheError(Exception):
    """Exception raised for cache-related errors."""
    pass


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    data: Any
    timestamp: datetime
    size_bytes: int
    hit_count: int = 0
    
    @property
    def age(self) -> timedelta:
        """Get age of cache entry."""
        return datetime.now() - self.timestamp
    
    def is_expired(self, max_age: timedelta) -> bool:
        """Check if entry is expired."""
        return self.age > max_age


class DataCache:
    """Disk-based cache for stock data."""
    
    def __init__(self, cache_dir: str = "data/cache", size_limit: int = 1e9):
        """
        Initialize the data cache.
        
        Args:
            cache_dir: Directory for cache storage
            size_limit: Maximum cache size in bytes (default: 1GB)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache = Cache(
            str(self.cache_dir),
            size_limit=size_limit,
            eviction_policy='least-recently-used'
        )
        
    def get(self, key: str, default: Any = None) -> Optional[pd.DataFrame]:
        """
        Get data from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached DataFrame or default value
        """
        try:
            data = self.cache.get(key, default)
            if data is not None and not isinstance(data, pd.DataFrame):
                # Handle legacy pickle format
                if isinstance(data, bytes):
                    data = pickle.loads(data)
            return data
        except Exception as e:
            print(f"Cache get error: {e}")
            return default
            
    def set(self, key: str, data: pd.DataFrame, expire: Optional[int] = None) -> bool:
        """
        Store data in cache.
        
        Args:
            key: Cache key
            data: DataFrame to cache
            expire: Expiration time in seconds (default: 24 hours)
            
        Returns:
            True if successful
        """
        if expire is None:
            expire = 24 * 60 * 60  # 24 hours default
            
        try:
            return self.cache.set(key, data, expire=expire)
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
            
    def delete(self, key: str) -> bool:
        """
        Delete data from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        try:
            return self.cache.delete(key)
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False
            
    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        
    def close(self) -> None:
        """Close the cache."""
        self.cache.close()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def get_size(self) -> int:
        """
        Get current cache size in bytes.
        
        Returns:
            Cache size in bytes
        """
        return self.cache.volume()
        
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            'size': self.get_size(),
            'size_mb': self.get_size() / (1024 * 1024),
            'hits': self.cache.stats(enable=True)[0],
            'misses': self.cache.stats(enable=True)[1],
            'keys': len(list(self.cache.iterkeys()))
        }