"""Comprehensive tests for fetcher and cache modules to improve coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import pickle
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock, mock_open
import aiohttp
import yfinance as yf

from src.data.fetcher import StockDataFetcher, DataSource, DataError
from src.data.cache import DataCache, CacheEntry, CacheError


class TestStockDataFetcherAdditional:
    """Additional comprehensive tests for StockDataFetcher to improve coverage."""
    
    @pytest.fixture
    def fetcher(self, tmp_path):
        """Create StockDataFetcher instance."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(exist_ok=True)
        return StockDataFetcher(cache_dir=str(cache_dir))
    
    @pytest.mark.asyncio
    async def test_fetch_with_all_parameters(self, fetcher):
        """Test fetch with all optional parameters."""
        # Mock yfinance download
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000000, 1100000, 1200000],
            'Dividends': [0, 0.5, 0],
            'Stock Splits': [0, 0, 0]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        with patch('yfinance.download', return_value=mock_data):
            async with fetcher:
                result = await fetcher.fetch(
                    symbol='AAPL',
                    start=datetime(2023, 1, 1),
                    end=datetime(2023, 1, 3),
                    interval='1d',
                    prepost=True,
                    auto_adjust=False,
                    actions=True,
                    repair=True
                )
                
                assert result is not None
                assert len(result) == 3
                assert 'Dividends' in result.columns
                assert 'Stock Splits' in result.columns
    
    @pytest.mark.asyncio
    async def test_fetch_with_invalid_interval(self, fetcher):
        """Test fetch with invalid interval."""
        with patch('yfinance.download', side_effect=Exception("Invalid interval")):
            async with fetcher:
                with pytest.raises(DataError, match="Failed to fetch data"):
                    await fetcher.fetch(
                        symbol='AAPL',
                        start=datetime(2023, 1, 1),
                        end=datetime(2023, 1, 3),
                        interval='invalid'
                    )
    
    @pytest.mark.asyncio
    async def test_context_manager_reuse(self, fetcher):
        """Test context manager can be used multiple times."""
        mock_data = pd.DataFrame({'Close': [100]})
        
        with patch('yfinance.download', return_value=mock_data):
            # First use
            async with fetcher:
                result1 = await fetcher.fetch('AAPL', datetime(2023, 1, 1), datetime(2023, 1, 2))
                assert result1 is not None
            
            # Second use
            async with fetcher:
                result2 = await fetcher.fetch('GOOGL', datetime(2023, 1, 1), datetime(2023, 1, 2))
                assert result2 is not None
    
    @pytest.mark.asyncio
    async def test_fetch_multiple(self, fetcher):
        """Test fetching multiple symbols."""
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        # Mock different data for each symbol
        def mock_download(tickers, **kwargs):
            if 'AAPL' in tickers:
                return pd.DataFrame({'Close': [150]})
            elif 'GOOGL' in tickers:
                return pd.DataFrame({'Close': [2800]})
            elif 'MSFT' in tickers:
                return pd.DataFrame({'Close': [350]})
            return pd.DataFrame()
        
        with patch('yfinance.download', side_effect=mock_download):
            async with fetcher:
                results = await fetcher.fetch_multiple(
                    symbols,
                    datetime(2023, 1, 1),
                    datetime(2023, 1, 2)
                )
                
                assert isinstance(results, dict)
                assert len(results) > 0
    
    def test_fetch_sync(self, fetcher):
        """Test synchronous fetch method."""
        mock_data = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        with patch('yfinance.download', return_value=mock_data):
            result = fetcher.fetch_sync(
                'AAPL',
                datetime(2023, 1, 1),
                datetime(2023, 1, 3)
            )
            
            assert result is not None
            pd.testing.assert_frame_equal(result, mock_data)
    
    def test_get_info(self, fetcher):
        """Test getting symbol info."""
        mock_info = {
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'marketCap': 3000000000000
        }
        
        with patch('yfinance.Ticker') as mock_ticker_class:
            mock_ticker = Mock()
            mock_ticker.info = mock_info
            mock_ticker_class.return_value = mock_ticker
            
            info = fetcher.get_info('AAPL')
            
            assert info == mock_info
            assert info['longName'] == 'Apple Inc.'
    
    def test_get_dividends(self, fetcher):
        """Test getting dividends."""
        mock_dividends = pd.Series([0.22, 0.23, 0.24], 
                                  index=pd.date_range('2023-01-01', periods=3, freq='Q'))
        
        with patch('yfinance.Ticker') as mock_ticker_class:
            mock_ticker = Mock()
            mock_ticker.dividends = mock_dividends
            mock_ticker_class.return_value = mock_ticker
            
            dividends = fetcher.get_dividends('AAPL')
            
            assert isinstance(dividends, pd.Series)
            assert len(dividends) == 3
    
    def test_get_splits(self, fetcher):
        """Test getting stock splits."""
        mock_splits = pd.Series([2.0, 4.0], 
                               index=pd.date_range('2020-01-01', periods=2, freq='Y'))
        
        with patch('yfinance.Ticker') as mock_ticker_class:
            mock_ticker = Mock()
            mock_ticker.splits = mock_splits
            mock_ticker_class.return_value = mock_ticker
            
            splits = fetcher.get_splits('AAPL')
            
            assert isinstance(splits, pd.Series)
            assert len(splits) == 2
    
    def test_get_options_chain(self, fetcher):
        """Test getting options chain."""
        mock_calls = pd.DataFrame({
            'strike': [150, 155, 160],
            'lastPrice': [10, 8, 6]
        })
        mock_puts = pd.DataFrame({
            'strike': [140, 145, 150],
            'lastPrice': [5, 7, 9]
        })
        
        with patch('yfinance.Ticker') as mock_ticker_class:
            mock_ticker = Mock()
            mock_ticker.option_chain.return_value = (mock_calls, mock_puts)
            mock_ticker_class.return_value = mock_ticker
            
            calls, puts = fetcher.get_options_chain('AAPL')
            
            assert isinstance(calls, pd.DataFrame)
            assert isinstance(puts, pd.DataFrame)
            assert len(calls) == 3
            assert len(puts) == 3


class TestCacheEntry:
    """Tests for CacheEntry class."""
    
    def test_cache_entry_creation(self):
        """Test CacheEntry creation."""
        data = pd.DataFrame({'value': [1, 2, 3]})
        timestamp = datetime.now()
        
        entry = CacheEntry(
            key='test_key',
            data=data,
            timestamp=timestamp,
            size_bytes=1024,
            hit_count=0
        )
        
        assert entry.key == 'test_key'
        assert entry.timestamp == timestamp
        assert entry.size_bytes == 1024
        assert entry.hit_count == 0
    
    def test_cache_entry_age(self):
        """Test CacheEntry age calculation."""
        # Create entry with past timestamp
        past_time = datetime.now() - timedelta(hours=2)
        entry = CacheEntry(
            key='old_entry',
            data=pd.DataFrame(),
            timestamp=past_time,
            size_bytes=100
        )
        
        age = entry.age
        assert age >= timedelta(hours=2)
        assert age < timedelta(hours=3)
    
    def test_cache_entry_is_expired(self):
        """Test CacheEntry expiration check."""
        # Create entry
        timestamp = datetime.now() - timedelta(hours=1)
        entry = CacheEntry(
            key='test',
            data=pd.DataFrame(),
            timestamp=timestamp,
            size_bytes=100
        )
        
        # Should not be expired for 2 hour limit
        assert not entry.is_expired(timedelta(hours=2))
        
        # Should be expired for 30 minute limit
        assert entry.is_expired(timedelta(minutes=30))


class TestDataCacheAdditional:
    """Additional comprehensive tests for DataCache to improve coverage."""
    
    @pytest.fixture
    def cache(self, tmp_path):
        """Create DataCache instance."""
        return DataCache(cache_dir=str(tmp_path))
    
    def test_cache_get_set(self, cache):
        """Test basic cache get/set operations."""
        key = "test_key"
        data = pd.DataFrame({'value': [1, 2, 3]})
        
        # Store data
        cache.set(key, data)
        
        # Retrieve data
        retrieved = cache.get(key)
        
        assert retrieved is not None
        pd.testing.assert_frame_equal(retrieved, data)
    
    def test_cache_delete(self, cache):
        """Test cache delete operation."""
        # Add data
        cache.set('key1', pd.DataFrame({'a': [1, 2]}))
        cache.set('key2', pd.DataFrame({'b': [3, 4]}))
        
        # Verify data exists
        assert cache.get('key1') is not None
        assert cache.get('key2') is not None
        
        # Delete one key
        cache.delete('key1')
        
        # Verify deletion
        assert cache.get('key1') is None
        assert cache.get('key2') is not None
    
    def test_cache_clear(self, cache):
        """Test cache clear operation."""
        # Add multiple items
        for i in range(5):
            cache.set(f'key{i}', pd.DataFrame({'value': [i]}))
        
        # Clear cache
        cache.clear()
        
        # Verify all items removed
        for i in range(5):
            assert cache.get(f'key{i}') is None
    
    def test_cache_size(self, cache):
        """Test cache size calculation."""
        # Add data of known size
        data = pd.DataFrame({
            'col1': np.arange(100),
            'col2': np.arange(100) * 2
        })
        
        cache.set('test_key', data)
        
        size = cache.size()
        assert size > 0  # Should have some size
        
    def test_cache_keys(self, cache):
        """Test listing cache keys."""
        # Add some data
        keys = ['key1', 'key2', 'key3']
        for key in keys:
            cache.set(key, pd.DataFrame({'data': [1]}))
        
        cached_keys = cache.keys()
        assert len(cached_keys) == 3
        for key in keys:
            assert key in cached_keys
    
    def test_cache_ttl(self, cache):
        """Test cache TTL functionality."""
        import time
        
        # Set data with short TTL
        cache.set('expiring_key', pd.DataFrame({'a': [1]}), ttl=0.1)
        
        # Should exist immediately
        assert cache.get('expiring_key') is not None
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        assert cache.get('expiring_key') is None




# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])