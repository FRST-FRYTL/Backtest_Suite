"""Comprehensive tests for data fetching and caching modules."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import json
import pickle
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import aiohttp
import yfinance as yf

from src.data import DataFetcher, CacheManager
from src.data.cache import CacheEntry, CacheError
from src.data.fetcher import DataSource, DataError


class TestCacheManager:
    """Test cache management functionality."""
    
    def test_cache_initialization(self, temp_cache_dir):
        """Test cache manager initialization."""
        cache = CacheManager(cache_dir=temp_cache_dir)
        
        assert cache.cache_dir == temp_cache_dir
        assert os.path.exists(temp_cache_dir)
        assert cache.max_cache_size == 1024 * 1024 * 1024  # 1GB default
        assert cache.ttl == 86400  # 24 hours default
    
    def test_cache_custom_initialization(self, temp_cache_dir):
        """Test cache with custom parameters."""
        cache = CacheManager(
            cache_dir=temp_cache_dir,
            max_cache_size=500 * 1024 * 1024,  # 500MB
            ttl=3600  # 1 hour
        )
        
        assert cache.max_cache_size == 500 * 1024 * 1024
        assert cache.ttl == 3600
    
    def test_cache_key_generation(self, cache_manager):
        """Test cache key generation."""
        # Test with different parameters
        key1 = cache_manager._generate_key('AAPL', '2023-01-01', '2023-12-31', 'stock')
        key2 = cache_manager._generate_key('AAPL', '2023-01-01', '2023-12-31', 'stock')
        key3 = cache_manager._generate_key('GOOGL', '2023-01-01', '2023-12-31', 'stock')
        
        # Same parameters should generate same key
        assert key1 == key2
        
        # Different parameters should generate different keys
        assert key1 != key3
    
    def test_cache_store_and_retrieve(self, cache_manager, sample_ohlcv_data):
        """Test storing and retrieving data from cache."""
        key = 'test_data'
        
        # Store data
        cache_manager.store(key, sample_ohlcv_data)
        
        # Retrieve data
        retrieved = cache_manager.retrieve(key)
        
        assert retrieved is not None
        assert isinstance(retrieved, pd.DataFrame)
        pd.testing.assert_frame_equal(retrieved, sample_ohlcv_data)
    
    def test_cache_expiration(self, cache_manager, sample_ohlcv_data):
        """Test cache expiration."""
        key = 'expiring_data'
        
        # Store with short TTL
        cache_manager.store(key, sample_ohlcv_data, ttl=1)
        
        # Should retrieve immediately
        assert cache_manager.retrieve(key) is not None
        
        # Wait for expiration
        import time
        time.sleep(2)
        
        # Should return None after expiration
        assert cache_manager.retrieve(key) is None
    
    def test_cache_invalidation(self, cache_manager, sample_ohlcv_data):
        """Test cache invalidation."""
        key = 'invalid_data'
        
        # Store data
        cache_manager.store(key, sample_ohlcv_data)
        assert cache_manager.retrieve(key) is not None
        
        # Invalidate
        cache_manager.invalidate(key)
        assert cache_manager.retrieve(key) is None
    
    def test_cache_clear(self, cache_manager, sample_ohlcv_data):
        """Test clearing entire cache."""
        # Store multiple items
        for i in range(5):
            cache_manager.store(f'data_{i}', sample_ohlcv_data)
        
        # Verify all stored
        for i in range(5):
            assert cache_manager.retrieve(f'data_{i}') is not None
        
        # Clear cache
        cache_manager.clear()
        
        # Verify all cleared
        for i in range(5):
            assert cache_manager.retrieve(f'data_{i}') is None
    
    def test_cache_size_limit(self, temp_cache_dir, sample_ohlcv_data):
        """Test cache size limiting."""
        # Create cache with small size limit
        cache = CacheManager(
            cache_dir=temp_cache_dir,
            max_cache_size=1024  # 1KB - very small
        )
        
        # Try to store large data
        large_data = pd.concat([sample_ohlcv_data] * 100)
        
        # Should enforce size limit
        with pytest.raises(CacheError):
            cache.store('large_data', large_data)
    
    def test_cache_lru_eviction(self, temp_cache_dir):
        """Test LRU eviction policy."""
        # Small cache that can hold ~3 items
        cache = CacheManager(
            cache_dir=temp_cache_dir,
            max_cache_size=10 * 1024,  # 10KB
            eviction_policy='lru'
        )
        
        # Create small test data
        data = pd.DataFrame({'value': range(100)})
        
        # Store items
        for i in range(5):
            cache.store(f'data_{i}', data)
        
        # Access first items to make them recently used
        cache.retrieve('data_0')
        cache.retrieve('data_1')
        
        # Older items should be evicted
        assert cache.retrieve('data_0') is not None
        assert cache.retrieve('data_1') is not None
        # data_2 or data_3 might be evicted
    
    def test_cache_statistics(self, cache_manager, sample_ohlcv_data):
        """Test cache statistics tracking."""
        # Store and retrieve data
        cache_manager.store('data1', sample_ohlcv_data)
        cache_manager.retrieve('data1')  # Hit
        cache_manager.retrieve('data2')  # Miss
        
        stats = cache_manager.get_statistics()
        
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
        assert stats['total_size'] > 0
        assert stats['item_count'] == 1
    
    def test_cache_compression(self, temp_cache_dir, sample_ohlcv_data):
        """Test cache compression."""
        # Cache with compression
        cache_compressed = CacheManager(
            cache_dir=temp_cache_dir,
            compression='gzip'
        )
        
        # Cache without compression
        cache_uncompressed = CacheManager(
            cache_dir=temp_cache_dir + '_uncompressed',
            compression=None
        )
        
        # Store same data
        cache_compressed.store('data', sample_ohlcv_data)
        cache_uncompressed.store('data', sample_ohlcv_data)
        
        # Compressed should use less space
        compressed_size = cache_compressed.get_statistics()['total_size']
        uncompressed_size = cache_uncompressed.get_statistics()['total_size']
        
        assert compressed_size < uncompressed_size
    
    def test_cache_concurrent_access(self, cache_manager, sample_ohlcv_data):
        """Test concurrent cache access."""
        import threading
        
        results = []
        errors = []
        
        def access_cache(thread_id):
            try:
                # Each thread stores and retrieves
                key = f'thread_{thread_id}'
                cache_manager.store(key, sample_ohlcv_data)
                retrieved = cache_manager.retrieve(key)
                results.append(retrieved is not None)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=access_cache, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # All should succeed
        assert len(errors) == 0
        assert all(results)


class TestDataFetcher:
    """Test data fetching functionality."""
    
    def test_fetcher_initialization(self):
        """Test data fetcher initialization."""
        fetcher = DataFetcher()
        
        assert fetcher.default_source == DataSource.YAHOO
        assert isinstance(fetcher.session, aiohttp.ClientSession) or fetcher.session is None
    
    def test_fetcher_with_cache(self, cache_manager):
        """Test fetcher with cache integration."""
        fetcher = DataFetcher(cache_manager=cache_manager)
        
        assert fetcher.cache_manager == cache_manager
    
    @pytest.mark.asyncio
    async def test_fetch_stock_data_yahoo(self, mock_yfinance_data):
        """Test fetching stock data from Yahoo Finance."""
        fetcher = DataFetcher()
        
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = generate_stock_data()
            
            data = await fetcher.fetch_stock_data(
                'AAPL',
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            
            assert isinstance(data, pd.DataFrame)
            assert not data.empty
            assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    @pytest.mark.asyncio
    async def test_fetch_multiple_symbols(self):
        """Test fetching data for multiple symbols."""
        fetcher = DataFetcher()
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        with patch('yfinance.download') as mock_download:
            # Return different data for each symbol
            def side_effect(symbol, *args, **kwargs):
                return generate_stock_data(initial_price=100 + symbols.index(symbol) * 50)
            
            mock_download.side_effect = side_effect
            
            results = await fetcher.fetch_multiple_stocks(
                symbols,
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            
            assert len(results) == len(symbols)
            assert all(symbol in results for symbol in symbols)
            assert all(isinstance(data, pd.DataFrame) for data in results.values())
    
    @pytest.mark.asyncio
    async def test_fetch_with_cache_hit(self, cache_manager, sample_ohlcv_data):
        """Test fetching with cache hit."""
        fetcher = DataFetcher(cache_manager=cache_manager)
        
        # Pre-populate cache
        cache_key = cache_manager._generate_key('AAPL', '2023-01-01', '2023-12-31', 'stock')
        cache_manager.store(cache_key, sample_ohlcv_data)
        
        # Fetch should return cached data
        with patch('yfinance.download') as mock_download:
            data = await fetcher.fetch_stock_data(
                'AAPL',
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            
            # Should not call yfinance
            mock_download.assert_not_called()
            
            # Should return cached data
            pd.testing.assert_frame_equal(data, sample_ohlcv_data)
    
    @pytest.mark.asyncio
    async def test_fetch_with_cache_miss(self, cache_manager):
        """Test fetching with cache miss."""
        fetcher = DataFetcher(cache_manager=cache_manager)
        
        with patch('yfinance.download') as mock_download:
            mock_data = generate_stock_data()
            mock_download.return_value = mock_data
            
            data = await fetcher.fetch_stock_data(
                'AAPL',
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            
            # Should call yfinance
            mock_download.assert_called_once()
            
            # Should cache the result
            cache_key = cache_manager._generate_key('AAPL', '2023-01-01', '2023-12-31', 'stock')
            cached = cache_manager.retrieve(cache_key)
            assert cached is not None
            pd.testing.assert_frame_equal(cached, mock_data)
    
    @pytest.mark.asyncio
    async def test_fetch_options_data(self, sample_options_data):
        """Test fetching options data."""
        fetcher = DataFetcher()
        calls, puts = sample_options_data
        
        with patch.object(fetcher, '_fetch_options_yahoo') as mock_fetch:
            mock_fetch.return_value = (calls, puts)
            
            result_calls, result_puts = await fetcher.fetch_options_data(
                'AAPL',
                expiry_date='2023-12-15'
            )
            
            assert isinstance(result_calls, pd.DataFrame)
            assert isinstance(result_puts, pd.DataFrame)
            assert 'strike' in result_calls.columns
            assert 'openInterest' in result_calls.columns
    
    @pytest.mark.asyncio
    async def test_fetch_intraday_data(self):
        """Test fetching intraday data."""
        fetcher = DataFetcher()
        
        # Generate 1-minute data
        intraday_data = generate_stock_data(
            start_date='2023-01-01 09:30:00',
            end_date='2023-01-01 16:00:00'
        )
        intraday_data.index = pd.date_range(
            '2023-01-01 09:30:00',
            '2023-01-01 16:00:00',
            freq='1min'
        )
        
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = intraday_data
            
            data = await fetcher.fetch_intraday_data(
                'AAPL',
                date='2023-01-01',
                interval='1m'
            )
            
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 300  # Should have many minute bars
            
            # Verify interval
            time_diff = data.index[1] - data.index[0]
            assert time_diff == timedelta(minutes=1)
    
    @pytest.mark.asyncio
    async def test_fetch_fundamental_data(self):
        """Test fetching fundamental data."""
        fetcher = DataFetcher()
        
        mock_info = {
            'symbol': 'AAPL',
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'marketCap': 3000000000000,
            'trailingPE': 30.5,
            'forwardPE': 28.2,
            'dividendYield': 0.005,
            'profitMargins': 0.25
        }
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.info = mock_info
            
            fundamentals = await fetcher.fetch_fundamental_data('AAPL')
            
            assert isinstance(fundamentals, dict)
            assert fundamentals['symbol'] == 'AAPL'
            assert fundamentals['marketCap'] > 0
            assert 'trailingPE' in fundamentals
    
    @pytest.mark.asyncio
    async def test_fetch_earnings_data(self):
        """Test fetching earnings data."""
        fetcher = DataFetcher()
        
        mock_earnings = pd.DataFrame({
            'Revenue': [100e9, 110e9, 120e9, 130e9],
            'Earnings': [20e9, 22e9, 25e9, 28e9]
        }, index=pd.date_range('2023-01-01', periods=4, freq='Q'))
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.quarterly_earnings = mock_earnings
            
            earnings = await fetcher.fetch_earnings_data('AAPL')
            
            assert isinstance(earnings, pd.DataFrame)
            assert 'Revenue' in earnings.columns
            assert 'Earnings' in earnings.columns
    
    @pytest.mark.asyncio
    async def test_fetch_news_sentiment(self):
        """Test fetching news and sentiment data."""
        fetcher = DataFetcher()
        
        mock_news = [
            {
                'title': 'Apple Reports Record Earnings',
                'publisher': 'Reuters',
                'link': 'https://example.com/1',
                'publishedDate': '2023-01-01',
                'sentiment': 'positive'
            },
            {
                'title': 'Apple Faces Supply Chain Issues',
                'publisher': 'Bloomberg',
                'link': 'https://example.com/2',
                'publishedDate': '2023-01-02',
                'sentiment': 'negative'
            }
        ]
        
        with patch.object(fetcher, '_fetch_news_data') as mock_fetch:
            mock_fetch.return_value = mock_news
            
            news = await fetcher.fetch_news_sentiment('AAPL', days=7)
            
            assert isinstance(news, list)
            assert len(news) == 2
            assert all('sentiment' in item for item in news)
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting functionality."""
        fetcher = DataFetcher(rate_limit=2)  # 2 requests per second
        
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = generate_stock_data()
            
            # Make rapid requests
            start_time = datetime.now()
            tasks = []
            
            for i in range(5):
                task = fetcher.fetch_stock_data(
                    f'SYMBOL{i}',
                    start_date='2023-01-01',
                    end_date='2023-12-31'
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            # Should take at least 2 seconds for 5 requests at 2/sec
            elapsed = (datetime.now() - start_time).total_seconds()
            assert elapsed >= 2.0
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry logic on failures."""
        fetcher = DataFetcher(max_retries=3)
        
        call_count = 0
        
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Network error")
            return generate_stock_data()
        
        with patch('yfinance.download') as mock_download:
            mock_download.side_effect = side_effect
            
            data = await fetcher.fetch_stock_data(
                'AAPL',
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            
            # Should retry and eventually succeed
            assert call_count == 3
            assert isinstance(data, pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_data_validation(self):
        """Test data validation after fetching."""
        fetcher = DataFetcher()
        
        # Create data with issues
        bad_data = generate_stock_data()
        bad_data.iloc[10, bad_data.columns.get_loc('high')] = 50  # High < Low
        bad_data.iloc[20, bad_data.columns.get_loc('volume')] = -100  # Negative volume
        
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = bad_data
            
            data = await fetcher.fetch_stock_data(
                'AAPL',
                start_date='2023-01-01',
                end_date='2023-12-31',
                validate=True
            )
            
            # Should fix data issues
            assert (data['high'] >= data['low']).all()
            assert (data['volume'] >= 0).all()
    
    @pytest.mark.asyncio
    async def test_fetch_crypto_data(self):
        """Test fetching cryptocurrency data."""
        fetcher = DataFetcher()
        
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = generate_stock_data(volatility=0.05)
            
            data = await fetcher.fetch_crypto_data(
                'BTC-USD',
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            
            assert isinstance(data, pd.DataFrame)
            # Crypto trades 24/7, so might have weekend data
            assert len(data) >= 250


class TestDataIntegration:
    """Test integration between fetcher and cache."""
    
    @pytest.mark.asyncio
    async def test_parallel_fetching_with_cache(self, cache_manager):
        """Test parallel fetching with shared cache."""
        fetcher = DataFetcher(cache_manager=cache_manager)
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        with patch('yfinance.download') as mock_download:
            def side_effect(symbol, *args, **kwargs):
                return generate_stock_data(seed=hash(symbol))
            
            mock_download.side_effect = side_effect
            
            # Fetch all symbols in parallel
            tasks = [
                fetcher.fetch_stock_data(symbol, '2023-01-01', '2023-12-31')
                for symbol in symbols
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            assert len(results) == len(symbols)
            assert all(isinstance(r, pd.DataFrame) for r in results)
            
            # All should be cached
            stats = cache_manager.get_statistics()
            assert stats['item_count'] == len(symbols)
    
    @pytest.mark.asyncio
    async def test_update_cached_data(self, cache_manager):
        """Test updating cached data with new information."""
        fetcher = DataFetcher(cache_manager=cache_manager)
        
        # Initial fetch
        with patch('yfinance.download') as mock_download:
            initial_data = generate_stock_data(
                start_date='2023-01-01',
                end_date='2023-06-30'
            )
            mock_download.return_value = initial_data
            
            data1 = await fetcher.fetch_stock_data(
                'AAPL',
                start_date='2023-01-01',
                end_date='2023-06-30'
            )
        
        # Fetch extended period
        with patch('yfinance.download') as mock_download:
            extended_data = generate_stock_data(
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            mock_download.return_value = extended_data
            
            data2 = await fetcher.fetch_stock_data(
                'AAPL',
                start_date='2023-01-01',
                end_date='2023-12-31',
                update_cache=True
            )
        
        # Should have more data
        assert len(data2) > len(data1)
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing of data requests."""
        fetcher = DataFetcher()
        
        # Create batch request
        requests = [
            {'symbol': 'AAPL', 'start': '2023-01-01', 'end': '2023-03-31'},
            {'symbol': 'GOOGL', 'start': '2023-01-01', 'end': '2023-03-31'},
            {'symbol': 'MSFT', 'start': '2023-04-01', 'end': '2023-06-30'},
            {'symbol': 'AMZN', 'start': '2023-04-01', 'end': '2023-06-30'},
        ]
        
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = generate_stock_data()
            
            results = await fetcher.batch_fetch(requests)
            
            assert len(results) == len(requests)
            
            # Check each result
            for i, result in enumerate(results):
                assert result['symbol'] == requests[i]['symbol']
                assert isinstance(result['data'], pd.DataFrame)
                assert result['status'] == 'success'


class TestDataQuality:
    """Test data quality and cleaning."""
    
    def test_detect_missing_data(self):
        """Test detection of missing data."""
        data = generate_stock_data()
        
        # Create gaps
        data.iloc[10:15, :] = np.nan
        data.iloc[50, data.columns.get_loc('volume')] = np.nan
        
        fetcher = DataFetcher()
        issues = fetcher._detect_data_issues(data)
        
        assert 'missing_rows' in issues
        assert len(issues['missing_rows']) == 5
        assert 'missing_values' in issues
        assert 'volume' in issues['missing_values']
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        data = generate_stock_data()
        
        # Add various issues
        data.iloc[10, data.columns.get_loc('high')] = 50  # High < Low
        data.iloc[20, data.columns.get_loc('volume')] = -100  # Negative volume
        data.iloc[30, data.columns.get_loc('close')] = 0  # Zero price
        data.iloc[40:45, :] = np.nan  # Missing rows
        
        fetcher = DataFetcher()
        cleaned = fetcher._clean_data(data)
        
        # Should fix all issues
        assert (cleaned['high'] >= cleaned['low']).all()
        assert (cleaned['volume'] >= 0).all()
        assert (cleaned['close'] > 0).all()
        assert cleaned.isna().sum().sum() == 0  # No NaN values
    
    def test_validate_data_integrity(self):
        """Test data integrity validation."""
        fetcher = DataFetcher()
        
        # Good data
        good_data = generate_stock_data()
        assert fetcher._validate_data_integrity(good_data) is True
        
        # Bad data - prices out of order
        bad_data = good_data.copy()
        bad_data = bad_data.iloc[::-1]  # Reverse order
        assert fetcher._validate_data_integrity(bad_data) is False
        
        # Bad data - duplicate timestamps
        bad_data2 = good_data.copy()
        bad_data2 = pd.concat([bad_data2, bad_data2.iloc[0:1]])
        assert fetcher._validate_data_integrity(bad_data2) is False
    
    def test_handle_splits_dividends(self):
        """Test handling of stock splits and dividends."""
        data = generate_stock_data()
        
        # Simulate a 2:1 split
        split_date = data.index[100]
        data.loc[split_date:, ['open', 'high', 'low', 'close']] *= 0.5
        data.loc[split_date:, 'volume'] *= 2
        
        fetcher = DataFetcher()
        adjusted = fetcher._adjust_for_splits(data, {'2023-04-10': 2.0})
        
        # Prices before split should be halved
        assert adjusted.loc[:split_date, 'close'].iloc[-1] == pytest.approx(
            data.loc[:split_date, 'close'].iloc[-1] / 2
        )


class TestErrorHandling:
    """Test error handling in data module."""
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self):
        """Test handling of network errors."""
        fetcher = DataFetcher()
        
        with patch('yfinance.download') as mock_download:
            mock_download.side_effect = aiohttp.ClientError("Network error")
            
            with pytest.raises(DataError) as exc_info:
                await fetcher.fetch_stock_data('AAPL', '2023-01-01', '2023-12-31')
            
            assert "Network error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_invalid_symbol_handling(self):
        """Test handling of invalid symbols."""
        fetcher = DataFetcher()
        
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = pd.DataFrame()  # Empty dataframe
            
            with pytest.raises(DataError) as exc_info:
                await fetcher.fetch_stock_data('INVALID_SYMBOL', '2023-01-01', '2023-12-31')
            
            assert "No data" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_invalid_date_range(self):
        """Test handling of invalid date ranges."""
        fetcher = DataFetcher()
        
        # End date before start date
        with pytest.raises(ValueError):
            await fetcher.fetch_stock_data('AAPL', '2023-12-31', '2023-01-01')
        
        # Future dates
        future_date = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
        with pytest.raises(ValueError):
            await fetcher.fetch_stock_data('AAPL', future_date, future_date)
    
    def test_corrupted_cache_handling(self, temp_cache_dir):
        """Test handling of corrupted cache files."""
        cache = CacheManager(cache_dir=temp_cache_dir)
        
        # Write corrupted data
        cache_file = os.path.join(temp_cache_dir, 'corrupted.pkl')
        with open(cache_file, 'wb') as f:
            f.write(b'corrupted data')
        
        # Should handle gracefully
        result = cache.retrieve('corrupted')
        assert result is None
        
        # Should log error but not crash
        stats = cache.get_statistics()
        assert stats['errors'] > 0


class TestPerformanceOptimization:
    """Test performance optimizations."""
    
    @pytest.mark.asyncio
    async def test_bulk_fetch_performance(self, performance_monitor):
        """Test performance of bulk data fetching."""
        fetcher = DataFetcher()
        symbols = [f'SYMBOL{i}' for i in range(20)]
        
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = generate_stock_data()
            
            performance_monitor.start('bulk_fetch')
            
            results = await fetcher.fetch_multiple_stocks(
                symbols,
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            
            performance_monitor.stop('bulk_fetch')
            
            # Should complete reasonably fast
            assert performance_monitor.get_duration('bulk_fetch') < 5.0
            assert len(results) == len(symbols)
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, cache_manager, performance_monitor):
        """Test cache performance improvement."""
        fetcher = DataFetcher(cache_manager=cache_manager)
        
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = generate_stock_data()
            
            # First fetch (cache miss)
            performance_monitor.start('first_fetch')
            await fetcher.fetch_stock_data('AAPL', '2023-01-01', '2023-12-31')
            performance_monitor.stop('first_fetch')
            
            # Second fetch (cache hit)
            performance_monitor.start('cached_fetch')
            await fetcher.fetch_stock_data('AAPL', '2023-01-01', '2023-12-31')
            performance_monitor.stop('cached_fetch')
            
            # Cached fetch should be much faster
            first_duration = performance_monitor.get_duration('first_fetch')
            cached_duration = performance_monitor.get_duration('cached_fetch')
            
            assert cached_duration < first_duration * 0.1  # At least 10x faster
    
    def test_data_compression_performance(self, sample_ohlcv_data, performance_monitor):
        """Test data compression performance."""
        cache = CacheManager()
        
        # Large dataset
        large_data = pd.concat([sample_ohlcv_data] * 100)
        
        # Time uncompressed
        performance_monitor.start('uncompressed')
        cache.compression = None
        cache.store('uncompressed', large_data)
        uncompressed = cache.retrieve('uncompressed')
        performance_monitor.stop('uncompressed')
        
        # Time compressed
        performance_monitor.start('compressed')
        cache.compression = 'gzip'
        cache.store('compressed', large_data)
        compressed = cache.retrieve('compressed')
        performance_monitor.stop('compressed')
        
        # Both should return same data
        pd.testing.assert_frame_equal(uncompressed, compressed)
        
        # Compression adds overhead but saves space
        # Performance depends on data size and system


if __name__ == "__main__":
    pytest.main([__file__, "-v"])