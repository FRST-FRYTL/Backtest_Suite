"""Comprehensive tests for spx_multi_timeframe_fetcher module to achieve 100% coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock, mock_open
import pytz

from src.data.spx_multi_timeframe_fetcher import SPXMultiTimeframeFetcher


class TestSPXMultiTimeframeFetcher:
    """Comprehensive tests for SPXMultiTimeframeFetcher class."""
    
    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary data directory."""
        data_dir = tmp_path / "data" / "SPX"
        data_dir.mkdir(parents=True, exist_ok=True)
        return str(data_dir)
    
    @pytest.fixture
    def fetcher(self, temp_data_dir):
        """Create SPXMultiTimeframeFetcher instance."""
        with patch('src.data.spx_multi_timeframe_fetcher.StockDataFetcher'):
            fetcher = SPXMultiTimeframeFetcher(data_dir=temp_data_dir)
            return fetcher
    
    def test_initialization(self, temp_data_dir):
        """Test SPXMultiTimeframeFetcher initialization."""
        with patch('src.data.spx_multi_timeframe_fetcher.StockDataFetcher') as mock_fetcher:
            fetcher = SPXMultiTimeframeFetcher(data_dir=temp_data_dir)
            
            assert fetcher.data_dir == Path(temp_data_dir)
            assert fetcher.symbol == "SPY"
            assert fetcher.summary_data['symbol'] == "SPY"
            assert 'fetch_date' in fetcher.summary_data
            assert fetcher.summary_data['timeframes'] == {}
            
            # Check that StockDataFetcher was initialized with correct cache dir
            mock_fetcher.assert_called_once_with(cache_dir=str(Path(temp_data_dir) / "cache"))
    
    @pytest.mark.asyncio
    async def test_fetch_all_timeframes_success(self, fetcher):
        """Test successful fetching of all timeframes."""
        # Create sample data for each timeframe
        sample_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', '2023-01-03', freq='D'))
        
        # Mock the individual timeframe fetch
        with patch.object(fetcher, '_fetch_single_timeframe', new_callable=AsyncMock) as mock_fetch:
            with patch.object(fetcher, '_run_post_edit_hook', new_callable=AsyncMock):
                mock_fetch.return_value = sample_data
                
                result = await fetcher.fetch_all_timeframes()
                
                assert isinstance(result, dict)
                assert len(result) == len(fetcher.TIMEFRAMES)
                for timeframe in fetcher.TIMEFRAMES:
                    assert timeframe in result
                    pd.testing.assert_frame_equal(result[timeframe], sample_data)
    
    @pytest.mark.asyncio
    async def test_fetch_all_timeframes_with_failures(self, fetcher):
        """Test fetching with some timeframes failing."""
        sample_data = pd.DataFrame({
            'Open': [100],
            'Close': [100.5],
            'Volume': [1000000]
        }, index=pd.date_range('2023-01-01', periods=1, freq='D'))
        
        # Mock some timeframes to fail
        async def mock_fetch(timeframe, config, start, end):
            if timeframe in ['1min', '5min']:
                raise Exception("Network error")
            return sample_data
        
        with patch.object(fetcher, '_fetch_single_timeframe', side_effect=mock_fetch):
            with patch.object(fetcher, '_run_post_edit_hook', new_callable=AsyncMock):
                result = await fetcher.fetch_all_timeframes()
                
                # Should still return successful timeframes
                assert len(result) < len(fetcher.TIMEFRAMES)
                assert '1min' not in result
                assert '5min' not in result
                assert '1D' in result
    
    @pytest.mark.asyncio
    async def test_fetch_all_timeframes_empty_data(self, fetcher):
        """Test handling of empty dataframes."""
        with patch.object(fetcher, '_fetch_single_timeframe', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = pd.DataFrame()  # Empty dataframe
            
            result = await fetcher.fetch_all_timeframes()
            
            assert isinstance(result, dict)
            assert len(result) == 0  # No timeframes added for empty data
    
    @pytest.mark.asyncio
    async def test_fetch_single_timeframe_success(self, fetcher):
        """Test successful single timeframe fetch."""
        # Create sample data
        dates = pd.date_range('2023-01-01 09:30', '2023-01-01 16:00', freq='1H')
        sample_data = pd.DataFrame({
            'Open': np.random.rand(len(dates)) * 100,
            'High': np.random.rand(len(dates)) * 100 + 10,
            'Low': np.random.rand(len(dates)) * 100 - 10,
            'Close': np.random.rand(len(dates)) * 100,
            'Volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)
        
        # Mock the fetcher
        mock_async_fetcher = AsyncMock()
        mock_async_fetcher.fetch.return_value = sample_data
        
        async def mock_context_manager():
            return mock_async_fetcher
        
        fetcher.fetcher.__aenter__ = mock_context_manager
        fetcher.fetcher.__aexit__ = AsyncMock()
        
        with patch.object(fetcher, '_save_timeframe_data') as mock_save:
            mock_save.return_value = Path("test_output.csv")
            
            config = fetcher.TIMEFRAMES['1H']
            start = datetime.now() - timedelta(days=180)
            end = datetime.now()
            
            result = await fetcher._fetch_single_timeframe('1H', config, start, end)
            
            assert result is not None
            assert len(result) == len(sample_data)
            mock_async_fetcher.fetch.assert_called_once()
            mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fetch_single_timeframe_4h_resampling(self, fetcher):
        """Test 4H timeframe resampling."""
        # Create hourly data
        dates = pd.date_range('2023-01-01', '2023-01-02', freq='1H')
        hourly_data = pd.DataFrame({
            'Open': [100 + i for i in range(len(dates))],
            'High': [102 + i for i in range(len(dates))],
            'Low': [98 + i for i in range(len(dates))],
            'Close': [101 + i for i in range(len(dates))],
            'Volume': [1000000 + i*10000 for i in range(len(dates))]
        }, index=dates)
        
        # Mock the fetcher
        mock_async_fetcher = AsyncMock()
        mock_async_fetcher.fetch.return_value = hourly_data
        
        async def mock_context_manager():
            return mock_async_fetcher
        
        fetcher.fetcher.__aenter__ = mock_context_manager
        fetcher.fetcher.__aexit__ = AsyncMock()
        
        with patch.object(fetcher, '_save_timeframe_data') as mock_save:
            mock_save.return_value = Path("test_4h.csv")
            
            config = fetcher.TIMEFRAMES['4H']
            start = datetime.now() - timedelta(days=365)
            end = datetime.now()
            
            result = await fetcher._fetch_single_timeframe('4H', config, start, end)
            
            assert result is not None
            assert len(result) < len(hourly_data)  # Should be resampled
            assert result.index.freq == '4H'
            
            # Check resampling aggregation
            assert result['High'].iloc[0] == hourly_data['High'][:4].max()
            assert result['Low'].iloc[0] == hourly_data['Low'][:4].min()
            assert result['Volume'].iloc[0] == hourly_data['Volume'][:4].sum()
    
    @pytest.mark.asyncio
    async def test_fetch_single_timeframe_timezone_handling(self, fetcher):
        """Test timezone handling for naive timestamps."""
        # Create data with naive timestamps
        dates = pd.date_range('2023-01-01', '2023-01-02', freq='D')
        sample_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [101, 102],
            'Low': [99, 100],
            'Close': [100.5, 101.5],
            'Volume': [1000000, 1100000]
        }, index=dates)  # Naive timestamps
        
        # Mock the fetcher
        mock_async_fetcher = AsyncMock()
        mock_async_fetcher.fetch.return_value = sample_data
        
        async def mock_context_manager():
            return mock_async_fetcher
        
        fetcher.fetcher.__aenter__ = mock_context_manager
        fetcher.fetcher.__aexit__ = AsyncMock()
        
        with patch.object(fetcher, '_save_timeframe_data') as mock_save:
            mock_save.return_value = Path("test_tz.csv")
            
            config = fetcher.TIMEFRAMES['1D']
            start = datetime.now() - timedelta(days=730)
            end = datetime.now()
            
            result = await fetcher._fetch_single_timeframe('1D', config, start, end)
            
            assert result is not None
            assert result.index.tz is not None  # Should be timezone-aware
            assert result.index.tz.zone == 'America/New_York'
    
    @pytest.mark.asyncio
    async def test_fetch_single_timeframe_exception(self, fetcher):
        """Test exception handling in single timeframe fetch."""
        # Mock the fetcher to raise exception
        mock_async_fetcher = AsyncMock()
        mock_async_fetcher.fetch.side_effect = Exception("API error")
        
        async def mock_context_manager():
            return mock_async_fetcher
        
        fetcher.fetcher.__aenter__ = mock_context_manager
        fetcher.fetcher.__aexit__ = AsyncMock()
        
        config = fetcher.TIMEFRAMES['1D']
        start = datetime.now() - timedelta(days=730)
        end = datetime.now()
        
        with pytest.raises(Exception, match="API error"):
            await fetcher._fetch_single_timeframe('1D', config, start, end)
    
    def test_resample_to_4h(self, fetcher):
        """Test 4H resampling function."""
        # Create hourly data
        dates = pd.date_range('2023-01-01', '2023-01-02', freq='1H')
        hourly_data = pd.DataFrame({
            'Open': np.random.rand(len(dates)) * 100,
            'High': np.random.rand(len(dates)) * 100 + 20,
            'Low': np.random.rand(len(dates)) * 100 - 20,
            'Close': np.random.rand(len(dates)) * 100,
            'Volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)
        
        result = fetcher._resample_to_4h(hourly_data)
        
        assert len(result) == len(dates) // 4
        assert all(col in result.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Verify aggregation functions
        first_4h = hourly_data.iloc[:4]
        assert result['Open'].iloc[0] == first_4h['Open'].iloc[0]
        assert result['High'].iloc[0] == first_4h['High'].max()
        assert result['Low'].iloc[0] == first_4h['Low'].min()
        assert result['Close'].iloc[0] == first_4h['Close'].iloc[-1]
        assert result['Volume'].iloc[0] == first_4h['Volume'].sum()
    
    def test_resample_to_4h_with_nan(self, fetcher):
        """Test 4H resampling with NaN values."""
        dates = pd.date_range('2023-01-01', '2023-01-01 12:00', freq='1H')
        hourly_data = pd.DataFrame({
            'Open': [100, np.nan, 102, 103, 104, np.nan, 106, 107, 108, 109, 110, 111, 112],
            'High': [101, np.nan, 103, 104, 105, np.nan, 107, 108, 109, 110, 111, 112, 113],
            'Low': [99, np.nan, 101, 102, 103, np.nan, 105, 106, 107, 108, 109, 110, 111],
            'Close': [100, np.nan, 102, 103, 104, np.nan, 106, 107, 108, 109, 110, 111, 112],
            'Volume': [1000000, np.nan, 1200000, 1300000, 1400000, np.nan, 1600000, 1700000, 1800000, 1900000, 2000000, 2100000, 2200000]
        }, index=dates)
        
        result = fetcher._resample_to_4h(hourly_data)
        
        # Should drop rows with all NaN
        assert len(result) > 0
        assert not result.isna().all(axis=1).any()
    
    def test_save_timeframe_data(self, fetcher):
        """Test saving timeframe data."""
        # Create sample data
        dates = pd.date_range('2023-01-01', '2023-01-05', freq='D')
        sample_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=dates)
        
        output_path = fetcher._save_timeframe_data('1D', sample_data)
        
        # Check that files were created
        assert output_path.exists()
        assert output_path.name == "SPY_1D_20230101_20230105.csv"
        
        # Check latest file
        latest_path = output_path.parent / "SPY_1D_latest.csv"
        assert latest_path.exists()
        
        # Verify content
        saved_data = pd.read_csv(output_path, index_col=0, parse_dates=True)
        assert len(saved_data) == len(sample_data)
    
    @pytest.mark.asyncio
    async def test_run_post_edit_hook_success(self, fetcher):
        """Test successful post-edit hook execution."""
        sample_data = pd.DataFrame({'Close': [100]}, index=[datetime.now()])
        
        # Mock subprocess
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"Success", b"")
        
        with patch('asyncio.create_subprocess_shell', return_value=mock_proc):
            await fetcher._run_post_edit_hook('1D', sample_data)
            
            # Should complete without error
            mock_proc.communicate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_post_edit_hook_failure(self, fetcher):
        """Test post-edit hook failure handling."""
        sample_data = pd.DataFrame({'Close': [100]}, index=[datetime.now()])
        
        # Mock subprocess with failure
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"Error message")
        
        with patch('asyncio.create_subprocess_shell', return_value=mock_proc):
            await fetcher._run_post_edit_hook('1D', sample_data)
            
            # Should handle error gracefully
            mock_proc.communicate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_post_edit_hook_exception(self, fetcher):
        """Test post-edit hook exception handling."""
        sample_data = pd.DataFrame({'Close': [100]}, index=[datetime.now()])
        
        with patch('asyncio.create_subprocess_shell', side_effect=Exception("Process error")):
            await fetcher._run_post_edit_hook('1D', sample_data)
            
            # Should handle exception gracefully
    
    @pytest.mark.asyncio
    async def test_store_notification_success(self, fetcher):
        """Test successful notification storage."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"Success", b"")
        
        with patch('asyncio.create_subprocess_shell', return_value=mock_proc):
            await fetcher._store_notification("Test message")
            
            mock_proc.communicate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_notification_exception(self, fetcher):
        """Test notification storage exception handling."""
        with patch('asyncio.create_subprocess_shell', side_effect=Exception("Process error")):
            await fetcher._store_notification("Test message")
            
            # Should handle exception gracefully
    
    def test_validate_data_quality(self, fetcher):
        """Test data quality validation."""
        # Create sample data with some issues
        dates = pd.date_range('2023-01-01', '2023-01-05', freq='D')
        data_by_timeframe = {
            '1D': pd.DataFrame({
                'Open': [100, 101, np.nan, 103, 104],
                'High': [101, 102, 103, 104, 105],
                'Low': [99, 100, 101, 102, 103],
                'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
                'Volume': [1000000, 1100000, 0, 1300000, 1400000]
            }, index=dates),
            '1H': pd.DataFrame({
                'Open': [100, 101],
                'High': [99, 102],  # Invalid: High < Open
                'Low': [99, 100],
                'Close': [100.5, 101.5],
                'Volume': [-1000, 1100000]  # Invalid: negative volume
            }, index=pd.date_range('2023-01-01', '2023-01-01 01:00', freq='1H'))
        }
        
        results = fetcher.validate_data_quality(data_by_timeframe)
        
        assert '1D' in results
        assert '1H' in results
        
        # Check 1D validation
        assert results['1D']['rows'] == 5
        assert results['1D']['missing_values']['Open'] == 1
        assert results['1D']['price_consistency']['high_low_valid'] == True
        assert results['1D']['price_consistency']['positive_volume'] == True
        
        # Check 1H validation
        assert results['1H']['rows'] == 2
        assert results['1H']['price_consistency']['high_low_valid'] == False  # High < Low issue
        assert results['1H']['price_consistency']['positive_volume'] == False  # Negative volume
    
    def test_calculate_data_freshness_timezone_aware(self, fetcher):
        """Test data freshness calculation with timezone-aware timestamps."""
        # Create data with timezone-aware timestamps
        tz = pytz.timezone('America/New_York')
        dates = pd.date_range('2023-01-01', periods=5, freq='D', tz=tz)
        sample_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104]
        }, index=dates)
        
        # Mock current time
        with patch('src.data.spx_multi_timeframe_fetcher.pd.Timestamp.now') as mock_now:
            mock_now.return_value = pd.Timestamp('2023-01-10', tz='UTC')
            
            freshness = fetcher._calculate_data_freshness(sample_data)
            
            assert freshness == 5  # 5 days old
    
    def test_calculate_data_freshness_naive(self, fetcher):
        """Test data freshness calculation with naive timestamps."""
        # Create data with naive timestamps
        dates = pd.date_range('2023-01-01', '2023-01-05', freq='D')
        sample_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104]
        }, index=dates)
        
        with patch('src.data.spx_multi_timeframe_fetcher.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 10)
            
            freshness = fetcher._calculate_data_freshness(sample_data)
            
            assert freshness == 5  # 5 days old
    
    def test_check_date_gaps_daily(self, fetcher):
        """Test date gap checking for daily data."""
        # Create data with a gap (missing weekend is OK for business days)
        dates = pd.DatetimeIndex([
            '2023-01-02',  # Monday
            '2023-01-03',  # Tuesday
            '2023-01-04',  # Wednesday
            # Missing Thursday
            '2023-01-06',  # Friday
        ])
        sample_data = pd.DataFrame({'Close': [100, 101, 102, 104]}, index=dates)
        
        result = fetcher._check_date_gaps(sample_data, '1D')
        
        assert result['checked'] == True
        assert result['missing_count'] > 0  # Should detect missing Thursday
        assert 'largest_gap' in result
    
    def test_check_date_gaps_intraday(self, fetcher):
        """Test date gap checking for intraday data."""
        # Create intraday data with gaps
        dates = pd.DatetimeIndex([
            '2023-01-02 09:30:00',
            '2023-01-02 09:35:00',
            # Missing 09:40
            '2023-01-02 09:45:00',
            '2023-01-02 09:50:00',
        ])
        sample_data = pd.DataFrame({'Close': [100, 101, 102, 103]}, index=dates)
        
        result = fetcher._check_date_gaps(sample_data, '5min')
        
        assert result['checked'] == True
        assert result['missing_count'] > 0
        assert result['missing_percentage'] > 0
    
    def test_check_date_gaps_unknown_timeframe(self, fetcher):
        """Test date gap checking for unknown timeframe."""
        sample_data = pd.DataFrame({'Close': [100]}, index=[datetime.now()])
        
        result = fetcher._check_date_gaps(sample_data, 'UNKNOWN')
        
        assert result['checked'] == False
        assert result['reason'] == 'Unknown timeframe'
    
    def test_check_date_gaps_exception(self, fetcher):
        """Test date gap checking exception handling."""
        sample_data = pd.DataFrame({'Close': [100]}, index=[datetime.now()])
        
        with patch('pandas.date_range', side_effect=Exception("Date range error")):
            result = fetcher._check_date_gaps(sample_data, '1D')
            
            assert result['checked'] == False
            assert 'Date range error' in result['reason']
    
    def test_find_largest_gap(self, fetcher):
        """Test finding largest gap in datetime index."""
        # Create data with varying gaps
        dates = pd.DatetimeIndex([
            '2023-01-01',
            '2023-01-02',
            # 3-day gap
            '2023-01-05',
            '2023-01-06',
            # 2-day gap
            '2023-01-08',
        ])
        
        result = fetcher._find_largest_gap(dates)
        
        assert result is not None
        assert '3 days' in result['duration']
        assert '2023-01-02' in result['start']
        assert '2023-01-05' in result['end']
    
    def test_find_largest_gap_single_element(self, fetcher):
        """Test finding largest gap with single element."""
        dates = pd.DatetimeIndex(['2023-01-01'])
        
        result = fetcher._find_largest_gap(dates)
        
        assert result is None
    
    def test_generate_summary_report(self, fetcher):
        """Test summary report generation."""
        # Create sample data
        dates_daily = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        dates_hourly = pd.date_range('2023-01-01', '2023-01-01 12:00', freq='1H')
        
        data_by_timeframe = {
            '1D': pd.DataFrame({
                'Open': np.random.rand(len(dates_daily)) * 100,
                'High': np.random.rand(len(dates_daily)) * 100 + 10,
                'Low': np.random.rand(len(dates_daily)) * 100 - 10,
                'Close': np.random.rand(len(dates_daily)) * 100,
                'Volume': np.random.randint(1000000, 10000000, len(dates_daily))
            }, index=dates_daily),
            '1H': pd.DataFrame({
                'Open': np.random.rand(len(dates_hourly)) * 100,
                'High': np.random.rand(len(dates_hourly)) * 100 + 10,
                'Low': np.random.rand(len(dates_hourly)) * 100 - 10,
                'Close': np.random.rand(len(dates_hourly)) * 100,
                'Volume': np.random.randint(100000, 1000000, len(dates_hourly))
            }, index=dates_hourly)
        }
        
        validation_results = {
            '1D': {
                'rows': len(dates_daily),
                'missing_percentage': {'Open': 0, 'Close': 0, 'Volume': 0},
                'price_consistency': {'high_low_valid': True, 'ohlc_valid': True, 'positive_volume': True}
            },
            '1H': {
                'rows': len(dates_hourly),
                'missing_percentage': {'Open': 2.0, 'Close': 2.0, 'Volume': 0},
                'price_consistency': {'high_low_valid': True, 'ohlc_valid': True, 'positive_volume': True}
            }
        }
        
        with patch.object(fetcher, '_generate_markdown_report'):
            report_path = fetcher.generate_summary_report(data_by_timeframe, validation_results)
            
            assert report_path == str(fetcher.data_dir / "summary_report.json")
            
            # Check JSON report was created
            json_path = Path(report_path)
            assert json_path.exists()
            
            with open(json_path, 'r') as f:
                report_data = json.load(f)
                assert 'summary' in report_data
                assert 'validation' in report_data
                assert 'fetch_timestamp' in report_data
    
    def test_generate_markdown_report(self, fetcher):
        """Test markdown report generation."""
        # Create sample data
        dates = pd.date_range('2023-01-01', '2023-01-05', freq='D')
        data_by_timeframe = {
            '1D': pd.DataFrame({
                'Close': [100, 101, 102, 103, 104]
            }, index=dates)
        }
        
        validation_results = {
            '1D': {
                'price_consistency': {
                    'high_low_valid': True,
                    'ohlc_valid': True,
                    'positive_volume': True
                },
                'date_gaps': {
                    'checked': True,
                    'missing_count': 0,
                    'missing_percentage': 0.0
                }
            }
        }
        
        output_path = fetcher.data_dir / "test_report.md"
        fetcher._generate_markdown_report(data_by_timeframe, validation_results, output_path)
        
        assert output_path.exists()
        
        content = output_path.read_text()
        assert "# SPX Multi-Timeframe Data Summary Report" in content
        assert "SPY (SPY ETF as S&P 500 proxy)" in content
        assert "1D - Position trading and trend following" in content
        assert "Data Quality:" in content
        assert "Usage Instructions" in content
        assert "pd.read_csv" in content


@pytest.mark.asyncio
async def test_main_function(tmp_path, capsys):
    """Test the main function."""
    # Create mock fetcher
    mock_fetcher = Mock(spec=SPXMultiTimeframeFetcher)
    mock_fetcher.data_dir = tmp_path / "data" / "SPX"
    mock_fetcher.fetch_all_timeframes = AsyncMock()
    mock_fetcher.validate_data_quality = Mock()
    mock_fetcher.generate_summary_report = Mock()
    mock_fetcher._store_notification = AsyncMock()
    
    # Mock data
    sample_data = {
        '1D': pd.DataFrame({'Close': [100, 101]}, index=pd.date_range('2023-01-01', periods=2))
    }
    mock_fetcher.fetch_all_timeframes.return_value = sample_data
    mock_fetcher.validate_data_quality.return_value = {'1D': {'rows': 2}}
    mock_fetcher.generate_summary_report.return_value = str(tmp_path / "report.json")
    
    # Mock subprocess for post-task hook
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"", b"")
    
    with patch('src.data.spx_multi_timeframe_fetcher.SPXMultiTimeframeFetcher', return_value=mock_fetcher):
        with patch('asyncio.create_subprocess_shell', return_value=mock_proc):
            from src.data.spx_multi_timeframe_fetcher import main
            await main()
    
    # Verify calls
    mock_fetcher.fetch_all_timeframes.assert_called_once()
    mock_fetcher.validate_data_quality.assert_called_once()
    mock_fetcher.generate_summary_report.assert_called_once()
    mock_fetcher._store_notification.assert_called_once()
    
    # Check output
    captured = capsys.readouterr()
    assert "Data fetch complete!" in captured.out


@pytest.mark.asyncio
async def test_main_function_error_handling(tmp_path, capsys):
    """Test main function error handling."""
    mock_fetcher = Mock(spec=SPXMultiTimeframeFetcher)
    mock_fetcher.fetch_all_timeframes = AsyncMock(side_effect=Exception("Test error"))
    
    with patch('src.data.spx_multi_timeframe_fetcher.SPXMultiTimeframeFetcher', return_value=mock_fetcher):
        from src.data.spx_multi_timeframe_fetcher import main
        
        with pytest.raises(Exception, match="Test error"):
            await main()
    
    captured = capsys.readouterr()
    assert "Error during data fetch: Test error" in captured.out


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])