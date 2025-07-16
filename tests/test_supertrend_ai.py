"""Tests for SuperTrend AI indicator."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.indicators.supertrend_ai import SuperTrendAI


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic price data with trend
    trend = np.linspace(100, 150, len(dates))
    noise = np.random.randn(len(dates)) * 2
    close_prices = trend + noise
    
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(len(dates)) * 0.5,
        'high': close_prices + np.abs(np.random.randn(len(dates))) * 2,
        'low': close_prices - np.abs(np.random.randn(len(dates))) * 2,
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Ensure high/low are correct
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


@pytest.fixture
def volatile_data():
    """Create volatile market data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
    np.random.seed(123)
    
    # Generate volatile price data
    close_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 5)
    
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(len(dates)) * 2,
        'high': close_prices + np.abs(np.random.randn(len(dates))) * 5,
        'low': close_prices - np.abs(np.random.randn(len(dates))) * 5,
        'close': close_prices,
        'volume': np.random.randint(500000, 10000000, len(dates))
    }, index=dates)
    
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


@pytest.fixture
def sideways_data():
    """Create sideways market data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
    np.random.seed(456)
    
    # Generate sideways price data
    close_prices = 100 + np.sin(np.linspace(0, 10 * np.pi, len(dates))) * 5 + np.random.randn(len(dates)) * 1
    
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(len(dates)) * 0.3,
        'high': close_prices + np.abs(np.random.randn(len(dates))) * 1,
        'low': close_prices - np.abs(np.random.randn(len(dates))) * 1,
        'close': close_prices,
        'volume': np.random.randint(800000, 2000000, len(dates))
    }, index=dates)
    
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


class TestSuperTrendAI:
    """Test SuperTrend AI indicator."""
    
    def test_initialization(self):
        """Test indicator initialization with various parameters."""
        # Default initialization
        st = SuperTrendAI()
        assert st.atr_length == 10
        assert st.factor_min == 1.0
        assert st.factor_max == 5.0
        assert st.factor_step == 0.5
        assert st.perf_alpha == 10.0
        assert st.cluster_from == 'best'
        assert st.max_iter == 1000
        assert st.max_data == 10000
        
        # Custom initialization
        st_custom = SuperTrendAI(
            atr_length=20,
            factor_min=0.5,
            factor_max=3.0,
            factor_step=0.25,
            perf_alpha=5.0,
            cluster_from='average'
        )
        assert st_custom.atr_length == 20
        assert st_custom.factor_min == 0.5
        assert st_custom.factor_max == 3.0
        assert st_custom.factor_step == 0.25
        assert st_custom.perf_alpha == 5.0
        assert st_custom.cluster_from == 'average'
    
    def test_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        # Min factor > Max factor
        with pytest.raises(ValueError, match="Minimum factor.*greater than maximum factor"):
            SuperTrendAI(factor_min=5.0, factor_max=1.0)
        
        # Invalid cluster_from
        with pytest.raises(ValueError, match="cluster_from must be one of"):
            SuperTrendAI(cluster_from='invalid')
        
        # Invalid step
        with pytest.raises(ValueError, match="factor_step must be positive"):
            SuperTrendAI(factor_step=0)
        
        # Invalid ATR length
        with pytest.raises(ValueError, match="atr_length must be positive"):
            SuperTrendAI(atr_length=0)
    
    def test_calculate_basic(self, sample_data):
        """Test basic calculation returns expected structure."""
        st = SuperTrendAI()
        result = st.calculate(sample_data)
        
        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        
        # Check required columns
        required_columns = [
            'trend', 'upper_band', 'lower_band', 
            'signal_strength', 'selected_factor', 'cluster_performance'
        ]
        for col in required_columns:
            assert col in result.columns
        
        # Check data types
        assert result['trend'].dtype == np.int64
        assert result['signal_strength'].dtype == np.int64
        assert result['upper_band'].dtype == np.float64
        assert result['lower_band'].dtype == np.float64
    
    def test_trend_values(self, sample_data):
        """Test trend values are binary (0 or 1)."""
        st = SuperTrendAI()
        result = st.calculate(sample_data)
        
        unique_trends = result['trend'].unique()
        assert all(t in [0, 1] for t in unique_trends)
    
    def test_signal_strength_range(self, sample_data):
        """Test signal strength is in valid range (0-10)."""
        st = SuperTrendAI()
        result = st.calculate(sample_data)
        
        assert result['signal_strength'].min() >= 0
        assert result['signal_strength'].max() <= 10
    
    def test_bands_logic(self, sample_data):
        """Test upper and lower bands logic."""
        st = SuperTrendAI()
        result = st.calculate(sample_data)
        
        # Upper band should be above HL2
        hl2 = (sample_data['high'] + sample_data['low']) / 2
        assert (result['upper_band'] >= hl2).all()
        
        # Lower band should be below HL2
        assert (result['lower_band'] <= hl2).all()
        
        # Bands should not cross
        assert (result['upper_band'] > result['lower_band']).all()
    
    def test_trend_changes(self, sample_data):
        """Test trend change signals."""
        st = SuperTrendAI()
        result = st.calculate(sample_data)
        
        # Find trend changes
        trend_diff = result['trend'].diff()
        bullish_signals = trend_diff == 1
        bearish_signals = trend_diff == -1
        
        # Should have at least some signals in a year of data
        assert bullish_signals.sum() > 0
        assert bearish_signals.sum() > 0
        
        # Signals should alternate (no consecutive same signals)
        signal_indices = result.index[bullish_signals | bearish_signals]
        if len(signal_indices) > 1:
            for i in range(1, len(signal_indices)):
                prev_trend = result.loc[signal_indices[i-1], 'trend']
                curr_trend = result.loc[signal_indices[i], 'trend']
                assert prev_trend != curr_trend
    
    def test_factor_selection(self, sample_data):
        """Test adaptive factor selection."""
        st = SuperTrendAI()
        result = st.calculate(sample_data)
        
        # Selected factors should be within range
        factors = result['selected_factor'].dropna()
        assert factors.min() >= st.factor_min
        assert factors.max() <= st.factor_max
        
        # Factors should be multiples of step
        expected_factors = np.arange(st.factor_min, st.factor_max + st.factor_step, st.factor_step)
        for factor in factors.unique():
            assert any(np.isclose(factor, ef) for ef in expected_factors)
    
    def test_cluster_selection(self, sample_data):
        """Test different cluster selections."""
        clusters = ['best', 'average', 'worst']
        results = {}
        
        for cluster in clusters:
            st = SuperTrendAI(cluster_from=cluster)
            result = st.calculate(sample_data)
            results[cluster] = result
        
        # Different clusters should produce different results
        assert not results['best']['selected_factor'].equals(results['worst']['selected_factor'])
        
        # Best cluster should generally have higher performance
        best_perf = results['best']['cluster_performance'].mean()
        worst_perf = results['worst']['cluster_performance'].mean()
        assert best_perf >= worst_perf
    
    def test_volatile_market_behavior(self, volatile_data):
        """Test indicator behavior in volatile markets."""
        st = SuperTrendAI()
        result = st.calculate(volatile_data)
        
        # In volatile markets, expect more frequent trend changes
        trend_changes = result['trend'].diff().abs().sum()
        data_days = len(volatile_data)
        change_frequency = trend_changes / data_days
        
        # Should have reasonable number of changes (not too many, not too few)
        assert 0.02 < change_frequency < 0.5  # 2% to 50% of days have trend changes
        
        # Signal strength should vary more in volatile markets
        signal_std = result['signal_strength'].std()
        assert signal_std > 1  # Some variation in signal strength
    
    def test_sideways_market_behavior(self, sideways_data):
        """Test indicator behavior in sideways markets."""
        st = SuperTrendAI()
        result = st.calculate(sideways_data)
        
        # In sideways markets, expect frequent but weak signals
        avg_signal_strength = result['signal_strength'].mean()
        assert avg_signal_strength < 7  # Weaker signals on average
        
        # More whipsaws expected
        trend_changes = result['trend'].diff().abs().sum()
        assert trend_changes > 10  # Multiple trend changes
    
    def test_performance_memory(self, sample_data):
        """Test performance memory calculation."""
        # Test with different alpha values
        alphas = [5.0, 10.0, 20.0]
        results = {}
        
        for alpha in alphas:
            st = SuperTrendAI(perf_alpha=alpha)
            result = st.calculate(sample_data)
            results[alpha] = result
        
        # Different alphas should produce different results
        assert not results[5.0]['cluster_performance'].equals(results[20.0]['cluster_performance'])
        
        # Lower alpha (faster adaptation) should have more variable performance
        perf_std_5 = results[5.0]['cluster_performance'].std()
        perf_std_20 = results[20.0]['cluster_performance'].std()
        assert perf_std_5 > perf_std_20
    
    def test_kmeans_clustering(self, sample_data):
        """Test K-means clustering functionality."""
        st = SuperTrendAI()
        result = st.calculate(sample_data)
        
        # Get internal state to verify clustering
        if hasattr(st, '_cluster_info'):
            cluster_info = st._cluster_info
            
            # Should have 3 clusters
            assert len(cluster_info['centroids']) == 3
            
            # Centroids should be ordered (worst < average < best)
            centroids = cluster_info['centroids']
            assert centroids[0] < centroids[1] < centroids[2]
            
            # Each cluster should have some factors
            for cluster_factors in cluster_info['factor_clusters']:
                assert len(cluster_factors) > 0
    
    def test_data_windowing(self, sample_data):
        """Test max_data parameter for computational efficiency."""
        # Test with limited data window
        st = SuperTrendAI(max_data=50)
        result = st.calculate(sample_data)
        
        # Should still produce valid results
        assert len(result) == len(sample_data)
        assert result['trend'].notna().all()
    
    def test_empty_data(self):
        """Test handling of empty data."""
        st = SuperTrendAI()
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Input data is empty"):
            st.calculate(empty_data)
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        st = SuperTrendAI(atr_length=14)
        
        # Create data with less bars than ATR length
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        small_data = pd.DataFrame({
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000000] * 10
        }, index=dates)
        
        with pytest.raises(ValueError, match="Insufficient data"):
            st.calculate(small_data)
    
    def test_missing_columns(self, sample_data):
        """Test handling of missing required columns."""
        st = SuperTrendAI()
        
        # Remove required column
        incomplete_data = sample_data.drop(columns=['high'])
        
        with pytest.raises(ValueError, match="Missing required columns"):
            st.calculate(incomplete_data)
    
    def test_signal_generation(self, sample_data):
        """Test signal generation for entry/exit."""
        st = SuperTrendAI()
        result = st.calculate(sample_data)
        
        # Add signal columns
        signals = st.generate_signals(result)
        
        assert 'long_entry' in signals.columns
        assert 'short_entry' in signals.columns
        assert 'exit_long' in signals.columns
        assert 'exit_short' in signals.columns
        
        # Signals should be boolean
        assert signals['long_entry'].dtype == bool
        assert signals['short_entry'].dtype == bool
        
        # Should not have simultaneous long and short entries
        simultaneous = signals['long_entry'] & signals['short_entry']
        assert simultaneous.sum() == 0
    
    def test_reproducibility(self, sample_data):
        """Test that calculations are reproducible."""
        st1 = SuperTrendAI(random_seed=42)
        st2 = SuperTrendAI(random_seed=42)
        
        result1 = st1.calculate(sample_data)
        result2 = st2.calculate(sample_data)
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_performance_metrics(self, sample_data):
        """Test performance metric calculations."""
        st = SuperTrendAI()
        result = st.calculate(sample_data)
        
        # Calculate performance metrics
        metrics = st.calculate_performance_metrics(sample_data, result)
        
        expected_metrics = [
            'total_signals', 'win_rate', 'avg_gain', 'avg_loss',
            'profit_factor', 'max_drawdown', 'sharpe_ratio'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_real_time_updates(self, sample_data):
        """Test real-time bar updates."""
        st = SuperTrendAI()
        
        # Calculate on all but last bar
        historical_data = sample_data[:-1]
        result = st.calculate(historical_data)
        
        # Update with new bar
        new_bar = sample_data.iloc[-1:]
        updated_result = st.update(new_bar, result)
        
        # Should have one more row
        assert len(updated_result) == len(result) + 1
        
        # Last row should have valid data
        last_row = updated_result.iloc[-1]
        assert last_row['trend'] in [0, 1]
        assert 0 <= last_row['signal_strength'] <= 10