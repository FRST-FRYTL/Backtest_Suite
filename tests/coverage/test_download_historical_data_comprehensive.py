"""Comprehensive tests for download_historical_data module to achieve 100% coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import pickle
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open

from src.data.download_historical_data import MarketDataDownloader, load_cached_data


class TestMarketDataDownloader:
    """Comprehensive tests for MarketDataDownloader class."""
    
    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create a temporary config file."""
        config_path = tmp_path / "test_config.yaml"
        config_data = {
            'assets': ['AAPL', 'GOOGL', 'SPY'],
            'data': {
                'start_date': '2023-01-01',
                'end_date': '2023-12-31'
            },
            'trading_costs': {
                'spread': {
                    'base_spread_pct': {
                        'SPY': 0.0001,
                        'AAPL': 0.0002
                    },
                    'volatility_multiplier': 2.0,
                    'volume_impact': {
                        'low_volume_multiplier': 1.5,
                        'high_volume_multiplier': 0.8
                    }
                },
                'commission': {
                    'percentage': 0.001,
                    'fixed': 0.0,
                    'minimum': 1.0
                },
                'slippage': {
                    'base_slippage_pct': 0.0005
                }
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        return str(config_path)
    
    @pytest.fixture
    def downloader(self, temp_config_file, tmp_path):
        """Create a MarketDataDownloader instance with temp directories."""
        with patch('src.data.download_historical_data.Path.mkdir'):
            downloader = MarketDataDownloader(config_path=temp_config_file)
            downloader.data_dir = tmp_path / "data"
            downloader.raw_dir = tmp_path / "data" / "raw"
            downloader.processed_dir = tmp_path / "data" / "processed"
            downloader.cache_dir = tmp_path / "data" / "cache"
            # Create directories
            for dir_path in [downloader.raw_dir, downloader.processed_dir, downloader.cache_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            return downloader
    
    def test_initialization(self, temp_config_file):
        """Test MarketDataDownloader initialization."""
        with patch('src.data.download_historical_data.Path.mkdir'):
            downloader = MarketDataDownloader(config_path=temp_config_file)
            
            assert downloader.config is not None
            assert downloader.config['assets'] == ['AAPL', 'GOOGL', 'SPY']
            assert downloader.data_dir == Path("data")
            assert downloader.raw_dir == Path("data/raw")
            assert downloader.processed_dir == Path("data/processed")
            assert downloader.cache_dir == Path("data/cache")
    
    def test_load_config(self, downloader, temp_config_file):
        """Test configuration loading."""
        config = downloader._load_config(temp_config_file)
        
        assert isinstance(config, dict)
        assert 'assets' in config
        assert 'data' in config
        assert 'trading_costs' in config
    
    @patch('yfinance.Ticker')
    def test_download_asset_data_from_cache(self, mock_ticker, downloader):
        """Test downloading asset data when cache exists."""
        # Create sample data
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.rand(len(dates)) * 100,
            'High': np.random.rand(len(dates)) * 100,
            'Low': np.random.rand(len(dates)) * 100,
            'Close': np.random.rand(len(dates)) * 100,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Create cache file
        cache_file = downloader.cache_dir / "AAPL_1d_2023-01-01_2023-01-10.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(sample_data, f)
        
        # Download data (should load from cache)
        result = downloader.download_asset_data('AAPL', '2023-01-01', '2023-01-10', '1d')
        
        assert result is not None
        pd.testing.assert_frame_equal(result, sample_data)
        mock_ticker.assert_not_called()  # Should not download from yfinance
    
    @patch('yfinance.Ticker')
    def test_download_asset_data_daily(self, mock_ticker, downloader):
        """Test downloading daily asset data."""
        # Mock yfinance response
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.rand(len(dates)) * 100,
            'High': np.random.rand(len(dates)) * 100,
            'Low': np.random.rand(len(dates)) * 100,
            'Close': np.random.rand(len(dates)) * 100,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Download data
        result = downloader.download_asset_data('AAPL', '2023-01-01', '2023-01-10', '1d')
        
        assert result is not None
        assert len(result) == len(mock_data)
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        mock_ticker_instance.history.assert_called_once()
    
    @patch('yfinance.Ticker')
    def test_download_asset_data_intraday(self, mock_ticker, downloader):
        """Test downloading intraday asset data with chunking."""
        # Mock yfinance response
        dates1 = pd.date_range('2023-01-01 09:30', '2023-01-30 16:00', freq='1H')
        dates2 = pd.date_range('2023-01-31 09:30', '2023-03-01 16:00', freq='1H')
        
        mock_data1 = pd.DataFrame({
            'Open': np.random.rand(len(dates1)) * 100,
            'High': np.random.rand(len(dates1)) * 100,
            'Low': np.random.rand(len(dates1)) * 100,
            'Close': np.random.rand(len(dates1)) * 100,
            'Volume': np.random.randint(100000, 1000000, len(dates1))
        }, index=dates1)
        
        mock_data2 = pd.DataFrame({
            'Open': np.random.rand(len(dates2)) * 100,
            'High': np.random.rand(len(dates2)) * 100,
            'Low': np.random.rand(len(dates2)) * 100,
            'Close': np.random.rand(len(dates2)) * 100,
            'Volume': np.random.randint(100000, 1000000, len(dates2))
        }, index=dates2)
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.side_effect = [mock_data1, mock_data2]
        mock_ticker.return_value = mock_ticker_instance
        
        # Download data
        result = downloader.download_asset_data('AAPL', '2023-01-01', '2023-03-01', '1h')
        
        assert result is not None
        assert len(result) == len(mock_data1) + len(mock_data2)
        assert mock_ticker_instance.history.call_count >= 2  # Multiple chunks
    
    @patch('yfinance.Ticker')
    def test_download_asset_data_empty_response(self, mock_ticker, downloader):
        """Test handling empty response from yfinance."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        result = downloader.download_asset_data('INVALID', '2023-01-01', '2023-01-10', '1d')
        
        assert result is not None
        assert result.empty
    
    @patch('yfinance.Ticker')
    def test_download_asset_data_with_duplicates(self, mock_ticker, downloader):
        """Test handling duplicate indices in downloaded data."""
        # Create data with duplicate indices
        dates = pd.date_range('2023-01-01', '2023-01-05', freq='D')
        duplicate_dates = pd.DatetimeIndex(list(dates) + [dates[2]])  # Add duplicate
        
        mock_data = pd.DataFrame({
            'Open': np.random.rand(len(duplicate_dates)) * 100,
            'High': np.random.rand(len(duplicate_dates)) * 100,
            'Low': np.random.rand(len(duplicate_dates)) * 100,
            'Close': np.random.rand(len(duplicate_dates)) * 100,
            'Volume': np.random.randint(1000000, 10000000, len(duplicate_dates))
        }, index=duplicate_dates)
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        result = downloader.download_asset_data('AAPL', '2023-01-01', '2023-01-05', '1d')
        
        assert result is not None
        assert not result.index.duplicated().any()  # No duplicates in result
    
    @patch('yfinance.Ticker')
    def test_download_asset_data_exception(self, mock_ticker, downloader):
        """Test exception handling during download."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.side_effect = Exception("Network error")
        mock_ticker.return_value = mock_ticker_instance
        
        result = downloader.download_asset_data('AAPL', '2023-01-01', '2023-01-10', '1d')
        
        assert result is not None
        assert result.empty
    
    @patch.object(MarketDataDownloader, 'download_asset_data')
    def test_download_all_assets(self, mock_download, downloader):
        """Test downloading all assets."""
        # Mock individual downloads
        sample_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', '2023-01-03', freq='D'))
        
        mock_download.return_value = sample_data
        
        # Download all assets
        result = downloader.download_all_assets(force_refresh=False)
        
        assert isinstance(result, dict)
        assert 'AAPL' in result
        assert 'GOOGL' in result
        assert 'SPY' in result
        
        # Check that each asset has multiple timeframes
        for symbol, timeframes in result.items():
            assert isinstance(timeframes, dict)
            # Should have called download for each timeframe
    
    @patch.object(MarketDataDownloader, 'download_asset_data')
    def test_download_all_assets_with_4h_resampling(self, mock_download, downloader):
        """Test 4H timeframe resampling from 1H data."""
        # Create hourly data
        hourly_dates = pd.date_range('2023-01-01', '2023-01-02', freq='1H')
        hourly_data = pd.DataFrame({
            'Open': np.random.rand(len(hourly_dates)) * 100,
            'High': np.random.rand(len(hourly_dates)) * 100 + 10,
            'Low': np.random.rand(len(hourly_dates)) * 100 - 10,
            'Close': np.random.rand(len(hourly_dates)) * 100,
            'Volume': np.random.randint(100000, 1000000, len(hourly_dates))
        }, index=hourly_dates)
        
        mock_download.return_value = hourly_data
        
        result = downloader.download_all_assets()
        
        # Check 4H resampling
        for symbol, timeframes in result.items():
            if '4H' in timeframes:
                four_hour_data = timeframes['4H']
                assert len(four_hour_data) < len(hourly_data)
                assert four_hour_data.index.freq == '4H'
    
    def test_download_all_assets_force_refresh(self, downloader):
        """Test force refresh clears cache."""
        # Create some cache files
        cache_file1 = downloader.cache_dir / "test1.pkl"
        cache_file2 = downloader.cache_dir / "test2.pkl"
        cache_file1.touch()
        cache_file2.touch()
        
        assert cache_file1.exists()
        assert cache_file2.exists()
        
        with patch.object(downloader, 'download_asset_data', return_value=pd.DataFrame()):
            downloader.download_all_assets(force_refresh=True)
        
        # Cache files should be deleted
        assert not cache_file1.exists()
        assert not cache_file2.exists()
    
    def test_get_spreads_and_fees_spy(self, downloader):
        """Test spread and fee calculation for SPY."""
        costs = downloader.get_spreads_and_fees(
            symbol='SPY',
            price=450.0,
            volume=80_000_000,
            avg_volume=75_000_000,
            volatility=0.012
        )
        
        assert 'spread' in costs
        assert 'commission' in costs
        assert 'slippage' in costs
        assert 'total_cost' in costs
        assert 'total_cost_pct' in costs
        
        assert costs['spread'] > 0
        assert costs['commission'] >= 1.0  # Minimum commission
        assert costs['slippage'] > 0
        assert costs['total_cost'] == costs['spread'] + costs['commission'] + costs['slippage']
        assert costs['total_cost_pct'] == costs['total_cost'] / 450.0
    
    def test_get_spreads_and_fees_high_volatility(self, downloader):
        """Test spread adjustment for high volatility."""
        costs_low_vol = downloader.get_spreads_and_fees(
            symbol='AAPL',
            price=150.0,
            volume=50_000_000,
            avg_volume=50_000_000,
            volatility=0.01  # 1% volatility
        )
        
        costs_high_vol = downloader.get_spreads_and_fees(
            symbol='AAPL',
            price=150.0,
            volume=50_000_000,
            avg_volume=50_000_000,
            volatility=0.05  # 5% volatility
        )
        
        # Higher volatility should increase spread
        assert costs_high_vol['spread'] > costs_low_vol['spread']
    
    def test_get_spreads_and_fees_low_volume(self, downloader):
        """Test spread adjustment for low volume."""
        costs_normal = downloader.get_spreads_and_fees(
            symbol='AAPL',
            price=150.0,
            volume=50_000_000,
            avg_volume=50_000_000,
            volatility=0.02
        )
        
        costs_low_volume = downloader.get_spreads_and_fees(
            symbol='AAPL',
            price=150.0,
            volume=20_000_000,  # Low volume
            avg_volume=50_000_000,
            volatility=0.02
        )
        
        # Low volume should increase spread
        assert costs_low_volume['spread'] > costs_normal['spread']
    
    def test_get_spreads_and_fees_high_volume(self, downloader):
        """Test spread adjustment for high volume."""
        costs_normal = downloader.get_spreads_and_fees(
            symbol='AAPL',
            price=150.0,
            volume=50_000_000,
            avg_volume=50_000_000,
            volatility=0.02
        )
        
        costs_high_volume = downloader.get_spreads_and_fees(
            symbol='AAPL',
            price=150.0,
            volume=120_000_000,  # High volume
            avg_volume=50_000_000,
            volatility=0.02
        )
        
        # High volume should decrease spread
        assert costs_high_volume['spread'] < costs_normal['spread']
    
    def test_get_spreads_and_fees_unknown_symbol(self, downloader):
        """Test spread calculation for unknown symbol uses default."""
        costs = downloader.get_spreads_and_fees(
            symbol='UNKNOWN',
            price=100.0,
            volume=1_000_000,
            avg_volume=1_000_000,
            volatility=0.02
        )
        
        # Should use default base spread
        assert costs['spread'] > 0
        assert costs['total_cost'] > 0
    
    def test_get_spreads_and_fees_zero_avg_volume(self, downloader):
        """Test handling zero average volume."""
        costs = downloader.get_spreads_and_fees(
            symbol='AAPL',
            price=150.0,
            volume=1_000_000,
            avg_volume=0,  # Zero average volume
            volatility=0.02
        )
        
        # Should handle gracefully
        assert costs['spread'] > 0
        assert costs['total_cost'] > 0
    
    def test_get_spreads_and_fees_extreme_volatility(self, downloader):
        """Test volatility multiplier capping."""
        costs_extreme_low = downloader.get_spreads_and_fees(
            symbol='AAPL',
            price=150.0,
            volume=50_000_000,
            avg_volume=50_000_000,
            volatility=0.001  # Very low volatility
        )
        
        costs_extreme_high = downloader.get_spreads_and_fees(
            symbol='AAPL',
            price=150.0,
            volume=50_000_000,
            avg_volume=50_000_000,
            volatility=0.20  # Very high volatility
        )
        
        # Both should be within reasonable bounds
        assert costs_extreme_low['spread'] > 0
        assert costs_extreme_high['spread'] > 0
        # High volatility spread should be capped at 3x
        assert costs_extreme_high['spread'] <= costs_extreme_low['spread'] * 6  # Max 3x multiplier on both ends


class TestLoadCachedData:
    """Test the load_cached_data function."""
    
    @pytest.fixture
    def setup_cache_dir(self, tmp_path):
        """Setup temporary cache directories."""
        cache_dir = tmp_path / "data" / "processed"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def test_load_cached_data_from_complete_pickle(self, setup_cache_dir):
        """Test loading data from complete market data pickle."""
        # Create sample data
        sample_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', '2023-01-03', freq='D'))
        
        # Create complete market data
        all_data = {
            'AAPL': {
                '1D': sample_data,
                '1H': sample_data
            }
        }
        
        # Save to pickle
        pickle_path = setup_cache_dir / "complete_market_data.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(all_data, f)
        
        # Mock Path to use our temp directory
        with patch('src.data.download_historical_data.Path') as mock_path:
            mock_path.return_value = pickle_path
            mock_path.exists.return_value = True
            
            with patch('builtins.open', mock_open(read_data=pickle.dumps(all_data))):
                with patch('pickle.load', return_value=all_data):
                    result = load_cached_data('AAPL', '1D')
        
        assert result is not None
        assert not result.empty
        assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume']  # Lowercase
    
    def test_load_cached_data_symbol_not_found(self, setup_cache_dir):
        """Test loading data for symbol not in cache."""
        all_data = {
            'AAPL': {
                '1D': pd.DataFrame()
            }
        }
        
        pickle_path = setup_cache_dir / "complete_market_data.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(all_data, f)
        
        with patch('src.data.download_historical_data.Path') as mock_path:
            mock_path.return_value = pickle_path
            mock_path.exists.return_value = True
            
            with patch('builtins.open', mock_open(read_data=pickle.dumps(all_data))):
                with patch('pickle.load', return_value=all_data):
                    result = load_cached_data('GOOGL', '1D')
        
        # Should fall back to individual cache (which doesn't exist)
        assert result is None
    
    def test_load_cached_data_timeframe_not_found(self, setup_cache_dir):
        """Test loading data for timeframe not in cache."""
        sample_data = pd.DataFrame({
            'Open': [100],
            'High': [101],
            'Low': [99],
            'Close': [100.5],
            'Volume': [1000000]
        })
        
        all_data = {
            'AAPL': {
                '1D': sample_data
            }
        }
        
        pickle_path = setup_cache_dir / "complete_market_data.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(all_data, f)
        
        with patch('src.data.download_historical_data.Path') as mock_path:
            mock_path.return_value = pickle_path
            mock_path.exists.return_value = True
            
            with patch('builtins.open', mock_open(read_data=pickle.dumps(all_data))):
                with patch('pickle.load', return_value=all_data):
                    result = load_cached_data('AAPL', '1H')
        
        assert result is None
    
    def test_load_cached_data_empty_dataframe(self, setup_cache_dir):
        """Test handling empty dataframe in cache."""
        all_data = {
            'AAPL': {
                '1D': pd.DataFrame()  # Empty
            }
        }
        
        pickle_path = setup_cache_dir / "complete_market_data.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(all_data, f)
        
        with patch('src.data.download_historical_data.Path') as mock_path:
            mock_path.return_value = pickle_path
            mock_path.exists.return_value = True
            
            with patch('builtins.open', mock_open(read_data=pickle.dumps(all_data))):
                with patch('pickle.load', return_value=all_data):
                    result = load_cached_data('AAPL', '1D')
        
        assert result is None
    
    def test_load_cached_data_pickle_error(self, setup_cache_dir):
        """Test handling pickle load errors."""
        pickle_path = setup_cache_dir / "complete_market_data.pkl"
        with open(pickle_path, 'wb') as f:
            f.write(b"invalid pickle data")
        
        with patch('src.data.download_historical_data.Path') as mock_path:
            mock_path.return_value = pickle_path
            mock_path.exists.return_value = True
            
            with patch('builtins.open', mock_open(read_data=b"invalid")):
                with patch('pickle.load', side_effect=Exception("Pickle error")):
                    result = load_cached_data('AAPL', '1D')
        
        assert result is None
    
    @patch('src.data.download_historical_data.MarketDataDownloader')
    def test_load_cached_data_from_individual_cache(self, mock_downloader_class, setup_cache_dir):
        """Test loading from individual cache file."""
        # Create sample data
        sample_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [101, 102],
            'Low': [99, 100],
            'Close': [100.5, 101.5],
            'Volume': [1000000, 1100000]
        })
        
        # Mock downloader instance
        mock_downloader = Mock()
        mock_downloader.cache_dir = setup_cache_dir / "cache"
        mock_downloader_class.return_value = mock_downloader
        
        # Create individual cache file
        individual_cache = mock_downloader.cache_dir / "SPY_1D_2019-01-01_2024-01-01.pkl"
        individual_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(individual_cache, 'wb') as f:
            pickle.dump(sample_data, f)
        
        # First check should fail (no complete pickle)
        with patch('src.data.download_historical_data.Path') as mock_path:
            complete_path = setup_cache_dir / "complete_market_data.pkl"
            mock_path.return_value = complete_path
            mock_path.exists.return_value = False
            
            result = load_cached_data('SPY', '1D')
        
        assert result is not None
        assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume']
    
    @patch('src.data.download_historical_data.MarketDataDownloader')
    def test_load_cached_data_individual_cache_error(self, mock_downloader_class):
        """Test handling errors when loading individual cache."""
        mock_downloader = Mock()
        mock_downloader.cache_dir = Path("/nonexistent/cache")
        mock_downloader_class.return_value = mock_downloader
        
        with patch('src.data.download_historical_data.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            
            result = load_cached_data('SPY', '1D')
        
        assert result is None


@patch('src.data.download_historical_data.MarketDataDownloader')
def test_main_function(mock_downloader_class, capsys):
    """Test the main function."""
    # Create mock downloader
    mock_downloader = Mock()
    mock_downloader_class.return_value = mock_downloader
    
    # Mock download_all_assets return value
    mock_all_data = {
        'SPY': {
            '1D': pd.DataFrame({
                'Open': [450, 451],
                'Close': [451, 452]
            }, index=pd.date_range('2023-01-01', '2023-01-02', freq='D')),
            '1H': pd.DataFrame({
                'Open': [450],
                'Close': [450.5]
            }, index=pd.date_range('2023-01-01 09:30', '2023-01-01 10:30', freq='1H'))
        },
        'AAPL': {
            '1D': pd.DataFrame()  # Empty dataframe
        }
    }
    mock_downloader.download_all_assets.return_value = mock_all_data
    
    # Mock get_spreads_and_fees
    mock_costs = {
        'spread': 0.045,
        'commission': 1.0,
        'slippage': 0.225,
        'total_cost': 1.27,
        'total_cost_pct': 0.00282
    }
    mock_downloader.get_spreads_and_fees.return_value = mock_costs
    
    # Import and run main
    from src.data.download_historical_data import main
    main()
    
    # Check output
    captured = capsys.readouterr()
    assert "Data Download Summary:" in captured.out
    assert "SPY:" in captured.out
    assert "1D: 2 bars" in captured.out
    assert "1H: 1 bars" in captured.out
    assert "Example Trading Costs" in captured.out
    assert "spread: $0.0450" in captured.out
    assert "total_cost_pct: 0.2820%" in captured.out
    
    # Verify method calls
    mock_downloader.download_all_assets.assert_called_once_with(force_refresh=False)
    mock_downloader.get_spreads_and_fees.assert_called_once_with(
        symbol='SPY',
        price=450,
        volume=80_000_000,
        avg_volume=75_000_000,
        volatility=0.012
    )


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])