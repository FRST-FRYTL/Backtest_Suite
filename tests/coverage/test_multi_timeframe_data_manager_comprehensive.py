"""Comprehensive tests for multi_timeframe_data_manager module to achieve 100% coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import yfinance as yf

from src.data.multi_timeframe_data_manager import (
    MultiTimeframeDataManager, Timeframe, TimeframeConfig
)


class TestTimeframeEnum:
    """Test Timeframe enum."""
    
    def test_timeframe_values(self):
        """Test Timeframe enum values."""
        assert Timeframe.HOUR_1.value == "1H"
        assert Timeframe.HOUR_4.value == "4H"
        assert Timeframe.DAY_1.value == "1D"
        assert Timeframe.WEEK_1.value == "1W"
        assert Timeframe.MONTH_1.value == "1M"


class TestTimeframeConfig:
    """Test TimeframeConfig dataclass."""
    
    def test_timeframe_config_creation(self):
        """Test TimeframeConfig creation."""
        config = TimeframeConfig(
            timeframe=Timeframe.DAY_1,
            weight=0.25,
            description="Daily timeframe",
            yfinance_interval="1d"
        )
        
        assert config.timeframe == Timeframe.DAY_1
        assert config.weight == 0.25
        assert config.description == "Daily timeframe"
        assert config.yfinance_interval == "1d"


class TestMultiTimeframeDataManager:
    """Comprehensive tests for MultiTimeframeDataManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create MultiTimeframeDataManager instance."""
        return MultiTimeframeDataManager()
    
    def test_initialization_default(self, manager):
        """Test default initialization."""
        assert manager.cache_dir == Path("data/cache")
        assert manager.data_store == {}
        assert manager.aligned_data == {}
        assert len(manager.TIMEFRAME_CONFIGS) == 5
        
        # Check timeframe configurations
        assert Timeframe.MONTH_1 in manager.TIMEFRAME_CONFIGS
        assert manager.TIMEFRAME_CONFIGS[Timeframe.MONTH_1].weight == 0.35
        assert manager.TIMEFRAME_CONFIGS[Timeframe.WEEK_1].weight == 0.30
        assert manager.TIMEFRAME_CONFIGS[Timeframe.DAY_1].weight == 0.25
        assert manager.TIMEFRAME_CONFIGS[Timeframe.HOUR_4].weight == 0.07
        assert manager.TIMEFRAME_CONFIGS[Timeframe.HOUR_1].weight == 0.03
    
    def test_initialization_custom_cache(self):
        """Test initialization with custom cache directory."""
        custom_cache = Path("/tmp/custom_cache")
        manager = MultiTimeframeDataManager(cache_dir=custom_cache)
        assert manager.cache_dir == custom_cache
    
    @pytest.mark.asyncio
    async def test_load_data_success(self, manager):
        """Test successful data loading."""
        # Create sample data
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.rand(len(dates)) * 100,
            'High': np.random.rand(len(dates)) * 100 + 10,
            'Low': np.random.rand(len(dates)) * 100 - 10,
            'Close': np.random.rand(len(dates)) * 100,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Mock the individual data loading method
        with patch.object(manager, '_load_timeframe_data', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = sample_data
            
            await manager.load_data("SPY", datetime(2023, 1, 1), datetime(2023, 1, 10))
            
            # Check that data was loaded for all timeframes
            assert "SPY" in manager.data_store
            assert len(manager.data_store["SPY"]) == len(manager.TIMEFRAME_CONFIGS)
            
            # Verify each timeframe has data
            for timeframe in manager.TIMEFRAME_CONFIGS:
                assert timeframe in manager.data_store["SPY"]
                pd.testing.assert_frame_equal(manager.data_store["SPY"][timeframe], sample_data)
    
    @pytest.mark.asyncio
    async def test_load_data_with_progress_callback(self, manager):
        """Test data loading with progress callback."""
        progress_updates = []
        
        def progress_callback(symbol, timeframe, status):
            progress_updates.append((symbol, timeframe, status))
        
        # Create sample data
        sample_data = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        with patch.object(manager, '_load_timeframe_data', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = sample_data
            
            await manager.load_data(
                "AAPL",
                datetime(2023, 1, 1),
                datetime(2023, 1, 3),
                progress_callback=progress_callback
            )
            
            # Should have progress updates for each timeframe
            assert len(progress_updates) == len(manager.TIMEFRAME_CONFIGS)
            assert all(update[0] == "AAPL" for update in progress_updates)
            assert all(update[2] == "completed" for update in progress_updates)
    
    @pytest.mark.asyncio
    async def test_load_data_with_some_failures(self, manager):
        """Test data loading with some timeframes failing."""
        # Mock different responses for different timeframes
        async def mock_load(symbol, timeframe, start, end):
            if timeframe in [Timeframe.HOUR_1, Timeframe.HOUR_4]:
                raise Exception("API limit reached")
            return pd.DataFrame({'Close': [100]}, index=[datetime(2023, 1, 1)])
        
        with patch.object(manager, '_load_timeframe_data', side_effect=mock_load):
            await manager.load_data("SPY", datetime(2023, 1, 1), datetime(2023, 1, 2))
            
            # Should still have data for successful timeframes
            assert "SPY" in manager.data_store
            assert Timeframe.DAY_1 in manager.data_store["SPY"]
            assert Timeframe.WEEK_1 in manager.data_store["SPY"]
            assert Timeframe.MONTH_1 in manager.data_store["SPY"]
            
            # Failed timeframes should not be present
            assert Timeframe.HOUR_1 not in manager.data_store["SPY"]
            assert Timeframe.HOUR_4 not in manager.data_store["SPY"]
    
    @pytest.mark.asyncio
    async def test_load_timeframe_data_from_cache(self, manager, tmp_path):
        """Test loading timeframe data from cache."""
        manager.cache_dir = tmp_path / "cache"
        manager.cache_dir.mkdir(exist_ok=True)
        
        # Create cached data
        cached_data = pd.DataFrame({
            'Open': [100, 101],
            'Close': [100.5, 101.5]
        }, index=pd.date_range('2023-01-01', periods=2))
        
        cache_file = manager.cache_dir / "SPY_1D_20230101_20230102.parquet"
        cached_data.to_parquet(cache_file)
        
        result = await manager._load_timeframe_data(
            "SPY",
            Timeframe.DAY_1,
            datetime(2023, 1, 1),
            datetime(2023, 1, 2)
        )
        
        pd.testing.assert_frame_equal(result, cached_data)
    
    @pytest.mark.asyncio
    async def test_load_timeframe_data_from_yfinance(self, manager, tmp_path):
        """Test loading timeframe data from yfinance."""
        manager.cache_dir = tmp_path / "cache"
        manager.cache_dir.mkdir(exist_ok=True)
        
        # Mock yfinance download
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        with patch('yfinance.download', return_value=mock_data):
            result = await manager._load_timeframe_data(
                "SPY",
                Timeframe.DAY_1,
                datetime(2023, 1, 1),
                datetime(2023, 1, 3)
            )
            
            pd.testing.assert_frame_equal(result, mock_data)
            
            # Check that data was cached
            cache_file = manager.cache_dir / "SPY_1D_20230101_20230103.parquet"
            assert cache_file.exists()
    
    @pytest.mark.asyncio
    async def test_load_timeframe_data_empty_response(self, manager, tmp_path):
        """Test handling empty yfinance response."""
        manager.cache_dir = tmp_path / "cache"
        manager.cache_dir.mkdir(exist_ok=True)
        
        with patch('yfinance.download', return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="No data available"):
                await manager._load_timeframe_data(
                    "INVALID",
                    Timeframe.DAY_1,
                    datetime(2023, 1, 1),
                    datetime(2023, 1, 3)
                )
    
    def test_align_timeframes(self, manager):
        """Test timeframe alignment."""
        # Create data with different frequencies
        base_index = pd.date_range('2023-01-01', '2023-01-31', freq='D')
        
        manager.data_store = {
            "SPY": {
                Timeframe.DAY_1: pd.DataFrame({
                    'Close': np.random.rand(len(base_index)) * 100
                }, index=base_index),
                
                Timeframe.WEEK_1: pd.DataFrame({
                    'Close': np.random.rand(5) * 100
                }, index=pd.date_range('2023-01-01', periods=5, freq='W')),
                
                Timeframe.MONTH_1: pd.DataFrame({
                    'Close': [100]
                }, index=[datetime(2023, 1, 1)])
            }
        }
        
        manager.align_timeframes("SPY")
        
        assert "SPY" in manager.aligned_data
        aligned = manager.aligned_data["SPY"]
        
        # All timeframes should have same index (daily)
        assert len(aligned[Timeframe.DAY_1]) == len(base_index)
        assert len(aligned[Timeframe.WEEK_1]) == len(base_index)
        assert len(aligned[Timeframe.MONTH_1]) == len(base_index)
        
        # Check that forward fill was applied
        week_data = aligned[Timeframe.WEEK_1]
        assert not week_data['Close'].isna().any()
    
    def test_align_timeframes_missing_symbol(self, manager):
        """Test alignment with missing symbol."""
        with pytest.raises(ValueError, match="No data loaded for symbol"):
            manager.align_timeframes("MISSING")
    
    def test_align_timeframes_partial_data(self, manager):
        """Test alignment with some timeframes missing."""
        manager.data_store = {
            "SPY": {
                Timeframe.DAY_1: pd.DataFrame({
                    'Close': [100, 101, 102]
                }, index=pd.date_range('2023-01-01', periods=3)),
                # Only daily data available
            }
        }
        
        # Should still work with partial data
        manager.align_timeframes("SPY")
        
        assert "SPY" in manager.aligned_data
        assert Timeframe.DAY_1 in manager.aligned_data["SPY"]
        assert len(manager.aligned_data["SPY"]) == 1
    
    def test_get_aligned_data(self, manager):
        """Test getting aligned data."""
        # Setup aligned data
        aligned_data = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        manager.aligned_data = {
            "SPY": {
                Timeframe.DAY_1: aligned_data
            }
        }
        
        result = manager.get_aligned_data("SPY", Timeframe.DAY_1)
        pd.testing.assert_frame_equal(result, aligned_data)
    
    def test_get_aligned_data_not_aligned(self, manager):
        """Test getting data when not aligned."""
        with pytest.raises(ValueError, match="Data not aligned"):
            manager.get_aligned_data("SPY", Timeframe.DAY_1)
    
    def test_get_aligned_data_missing_timeframe(self, manager):
        """Test getting data for missing timeframe."""
        manager.aligned_data = {"SPY": {}}
        
        with pytest.raises(ValueError, match="Timeframe.*not available"):
            manager.get_aligned_data("SPY", Timeframe.DAY_1)
    
    def test_get_all_aligned_data(self, manager):
        """Test getting all aligned data."""
        # Setup aligned data
        manager.aligned_data = {
            "SPY": {
                Timeframe.DAY_1: pd.DataFrame({'Close': [100]}),
                Timeframe.WEEK_1: pd.DataFrame({'Close': [100]})
            }
        }
        
        result = manager.get_all_aligned_data("SPY")
        
        assert isinstance(result, dict)
        assert len(result) == 2
        assert Timeframe.DAY_1 in result
        assert Timeframe.WEEK_1 in result
    
    def test_get_all_aligned_data_not_aligned(self, manager):
        """Test getting all data when not aligned."""
        with pytest.raises(ValueError, match="Data not aligned"):
            manager.get_all_aligned_data("SPY")
    
    def test_calculate_confluence_score(self, manager):
        """Test confluence score calculation."""
        # Setup signals for different timeframes
        signals = {
            Timeframe.MONTH_1: pd.Series([1, 1, -1], index=pd.date_range('2023-01-01', periods=3)),
            Timeframe.WEEK_1: pd.Series([1, -1, -1], index=pd.date_range('2023-01-01', periods=3)),
            Timeframe.DAY_1: pd.Series([1, 1, 1], index=pd.date_range('2023-01-01', periods=3)),
            Timeframe.HOUR_4: pd.Series([-1, 1, -1], index=pd.date_range('2023-01-01', periods=3)),
            Timeframe.HOUR_1: pd.Series([1, -1, 1], index=pd.date_range('2023-01-01', periods=3))
        }
        
        scores = manager.calculate_confluence_score(signals)
        
        assert isinstance(scores, pd.Series)
        assert len(scores) == 3
        
        # Check first timestamp calculation
        # Month: 1 * 0.35 = 0.35
        # Week: 1 * 0.30 = 0.30
        # Day: 1 * 0.25 = 0.25
        # 4H: -1 * 0.07 = -0.07
        # 1H: 1 * 0.03 = 0.03
        # Total: 0.35 + 0.30 + 0.25 - 0.07 + 0.03 = 0.86
        assert abs(scores.iloc[0] - 0.86) < 0.001
    
    def test_calculate_confluence_score_missing_timeframes(self, manager):
        """Test confluence score with missing timeframes."""
        # Only provide some timeframes
        signals = {
            Timeframe.DAY_1: pd.Series([1, 1, 1]),
            Timeframe.WEEK_1: pd.Series([1, -1, 1])
        }
        
        scores = manager.calculate_confluence_score(signals)
        
        assert isinstance(scores, pd.Series)
        assert len(scores) == 3
        # Should still calculate with available timeframes
    
    def test_get_timeframe_hierarchy(self, manager):
        """Test getting timeframe hierarchy."""
        hierarchy = manager.get_timeframe_hierarchy()
        
        assert isinstance(hierarchy, list)
        assert len(hierarchy) == 5
        
        # Check order (highest weight first)
        assert hierarchy[0] == Timeframe.MONTH_1
        assert hierarchy[1] == Timeframe.WEEK_1
        assert hierarchy[2] == Timeframe.DAY_1
        assert hierarchy[3] == Timeframe.HOUR_4
        assert hierarchy[4] == Timeframe.HOUR_1
    
    def test_get_summary_statistics(self, manager):
        """Test summary statistics generation."""
        # Setup data
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        manager.data_store = {
            "SPY": {
                Timeframe.DAY_1: pd.DataFrame({
                    'Open': np.random.rand(len(dates)) * 100,
                    'High': np.random.rand(len(dates)) * 100 + 10,
                    'Low': np.random.rand(len(dates)) * 100 - 10,
                    'Close': np.random.rand(len(dates)) * 100,
                    'Volume': np.random.randint(1000000, 10000000, len(dates))
                }, index=dates),
                Timeframe.WEEK_1: pd.DataFrame({
                    'Close': [100, 105]
                }, index=pd.date_range('2023-01-01', periods=2, freq='W'))
            }
        }
        
        stats = manager.get_summary_statistics("SPY")
        
        assert isinstance(stats, dict)
        assert Timeframe.DAY_1 in stats
        assert Timeframe.WEEK_1 in stats
        
        # Check daily stats
        daily_stats = stats[Timeframe.DAY_1]
        assert 'date_range' in daily_stats
        assert 'num_bars' in daily_stats
        assert 'columns' in daily_stats
        assert 'missing_data_pct' in daily_stats
        
        assert daily_stats['num_bars'] == len(dates)
        assert set(daily_stats['columns']) == {'Open', 'High', 'Low', 'Close', 'Volume'}
    
    def test_get_summary_statistics_no_data(self, manager):
        """Test summary statistics with no data."""
        stats = manager.get_summary_statistics("MISSING")
        assert stats == {}
    
    def test_clear_cache(self, manager, tmp_path):
        """Test cache clearing."""
        manager.cache_dir = tmp_path / "cache"
        manager.cache_dir.mkdir(exist_ok=True)
        
        # Create some cache files
        cache_files = [
            manager.cache_dir / "SPY_1D_20230101_20230110.parquet",
            manager.cache_dir / "AAPL_1H_20230101_20230102.parquet"
        ]
        
        for file in cache_files:
            file.touch()
        
        # Clear cache for SPY
        manager.clear_cache("SPY")
        
        # SPY cache should be deleted
        assert not cache_files[0].exists()
        # AAPL cache should remain
        assert cache_files[1].exists()
    
    def test_clear_cache_all_symbols(self, manager, tmp_path):
        """Test clearing all cache."""
        manager.cache_dir = tmp_path / "cache"
        manager.cache_dir.mkdir(exist_ok=True)
        
        # Create cache files
        cache_files = [
            manager.cache_dir / "SPY_1D_20230101_20230110.parquet",
            manager.cache_dir / "AAPL_1H_20230101_20230102.parquet"
        ]
        
        for file in cache_files:
            file.touch()
        
        # Clear all cache
        manager.clear_cache()
        
        # All cache files should be deleted
        assert not any(file.exists() for file in cache_files)
    
    def test_clear_cache_no_cache_dir(self, manager):
        """Test clearing cache when directory doesn't exist."""
        manager.cache_dir = Path("/nonexistent/cache")
        
        # Should not raise error
        manager.clear_cache()
    
    @pytest.mark.asyncio
    async def test_load_multiple_symbols(self, manager):
        """Test loading data for multiple symbols."""
        symbols = ["SPY", "AAPL", "GOOGL"]
        
        # Mock data loading
        sample_data = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        with patch.object(manager, '_load_timeframe_data', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = sample_data
            
            await manager.load_multiple_symbols(
                symbols,
                datetime(2023, 1, 1),
                datetime(2023, 1, 3)
            )
            
            # Check all symbols loaded
            for symbol in symbols:
                assert symbol in manager.data_store
                assert len(manager.data_store[symbol]) == len(manager.TIMEFRAME_CONFIGS)
    
    @pytest.mark.asyncio
    async def test_load_multiple_symbols_with_progress(self, manager):
        """Test loading multiple symbols with progress callback."""
        symbols = ["SPY", "AAPL"]
        progress_updates = []
        
        def progress_callback(symbol, timeframe, status):
            progress_updates.append((symbol, timeframe, status))
        
        sample_data = pd.DataFrame({'Close': [100]})
        
        with patch.object(manager, '_load_timeframe_data', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = sample_data
            
            await manager.load_multiple_symbols(
                symbols,
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                progress_callback=progress_callback
            )
            
            # Should have updates for each symbol and timeframe
            total_expected = len(symbols) * len(manager.TIMEFRAME_CONFIGS)
            assert len(progress_updates) == total_expected


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])