"""Optimized tests for VWMA indicator to achieve 100% coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch

from src.indicators.vwma import VWMABands
from src.indicators.base import IndicatorError


class TestVWMAComplete:
    """Complete test suite for 100% VWMA coverage."""
    
    def test_initialization_coverage(self):
        """Test lines 19-36: __init__ method."""
        # Default initialization
        vwma1 = VWMABands()
        assert vwma1.name == "VWMA_Bands"
        assert vwma1.period == 20
        assert vwma1.band_multiplier == 2.0
        assert vwma1.price_column == "close"
        
        # Custom initialization  
        vwma2 = VWMABands(period=50, band_multiplier=3.0, price_column="high")
        assert vwma2.period == 50
        assert vwma2.band_multiplier == 3.0
        assert vwma2.price_column == "high"
        
    def test_calculate_complete(self):
        """Test lines 38-81: calculate method with all branches."""
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        data = pd.DataFrame({
            'close': 100 + np.random.randn(30) * 2,
            'volume': np.random.randint(900000, 1100000, 30)
        }, index=dates)
        
        vwma = VWMABands(period=10)
        result = vwma.calculate(data)
        
        # Verify all outputs (lines 69-81)
        assert 'vwma' in result.columns
        assert 'vwma_upper' in result.columns
        assert 'vwma_lower' in result.columns
        assert 'vwma_width' in result.columns
        assert 'vwma_signal' in result.columns
        
        # Test signal generation (lines 77-79)
        # Create data to trigger all signal conditions
        test_data = pd.DataFrame({
            'close': [95, 100, 105, 110, 90],  # Below, neutral, neutral, above, below
            'volume': [1000000] * 5
        })
        
        vwma2 = VWMABands(period=2)
        result2 = vwma2.calculate(test_data)
        
        # Verify signals
        assert result2['vwma_signal'].iloc[-1] == 1  # Buy signal when price < lower band
        
    def test_calculate_vwma_method(self):
        """Test lines 83-111: _calculate_vwma method."""
        vwma = VWMABands(period=5)
        
        # Test normal case
        price = pd.Series([100, 101, 102, 103, 104])
        volume = pd.Series([1000, 1100, 1200, 1300, 1400])
        
        result = vwma._calculate_vwma(price, volume)
        
        # Manually verify calculation
        pv = price * volume
        pv_sum = pv.rolling(window=5, min_periods=1).sum()
        vol_sum = volume.rolling(window=5, min_periods=1).sum()
        expected = pv_sum / vol_sum
        
        pd.testing.assert_series_equal(result, expected)
        
        # Test zero volume case (line 109)
        zero_volume = pd.Series([0, 0, 0, 0, 0])
        result_zero = vwma._calculate_vwma(price, zero_volume)
        
        # Should fall back to SMA
        expected_sma = price.rolling(window=5, min_periods=1).mean()
        pd.testing.assert_series_equal(result_zero, expected_sma)
        
    def test_calculate_rolling_std_method(self):
        """Test lines 113-149: _calculate_rolling_std method."""
        vwma = VWMABands(period=5)
        
        # Test normal case
        price = pd.Series([100, 101, 99, 102, 98])
        volume = pd.Series([1000, 1100, 1200, 1300, 1400])
        vwma_series = pd.Series([100, 100.5, 100, 100.5, 100])
        
        result = vwma._calculate_rolling_std(price, volume, vwma_series)
        
        # Verify it returns a series
        assert isinstance(result, pd.Series)
        assert len(result) == len(price)
        
        # Test with zero volume (line 147)
        zero_volume = pd.Series([0, 0, 0, 0, 0])
        result_zero = vwma._calculate_rolling_std(price, zero_volume, vwma_series)
        
        # Should fall back to regular std
        expected_std = price.rolling(window=5, min_periods=1).std()
        pd.testing.assert_series_equal(result_zero, expected_std)
        
    def test_get_signals_method(self):
        """Test lines 151-205: get_signals method."""
        # Create test data with specific patterns
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        
        # Create price that crosses bands
        price = pd.Series(100.0, index=dates)
        price.iloc[10] = 105  # Will cross above upper
        price.iloc[20] = 95   # Will cross below lower
        price.iloc[30] = 101  # Will cross above vwma
        price.iloc[40] = 99   # Will cross below vwma
        
        data = pd.DataFrame({
            'close': price,
            'volume': np.ones(50) * 1000000
        })
        
        vwma = VWMABands(period=5, band_multiplier=1.0)  # Tight bands
        vwma_data = vwma.calculate(data)
        
        # Add some variation to band width for squeeze/expansion
        vwma_data['vwma_width'].iloc[25:30] = vwma_data['vwma_width'].iloc[25:30] * 0.5  # Squeeze
        vwma_data['vwma_width'].iloc[35:40] = vwma_data['vwma_width'].iloc[35:40] * 2.0  # Expansion
        
        signals = vwma.get_signals(data, vwma_data)
        
        # Verify all signal columns exist (lines 166-204)
        assert 'touch_upper' in signals.columns
        assert 'touch_lower' in signals.columns
        assert 'cross_above_upper' in signals.columns
        assert 'cross_below_lower' in signals.columns
        assert 'cross_above_vwma' in signals.columns
        assert 'cross_below_vwma' in signals.columns
        assert 'band_squeeze' in signals.columns
        assert 'band_expansion' in signals.columns
        
        # Verify some signals were generated
        assert signals['cross_above_upper'].any()
        assert signals['cross_below_lower'].any()
        assert signals['band_squeeze'].any()
        assert signals['band_expansion'].any()
        
    def test_calculate_percent_b(self):
        """Test lines 207-224: calculate_percent_b method."""
        vwma = VWMABands()
        
        # Test data
        price = pd.Series([95, 100, 105])
        vwma_data = pd.DataFrame({
            'vwma_lower': [90, 95, 100],
            'vwma_upper': [110, 105, 110]
        })
        
        # This executes line 222
        percent_b = vwma.calculate_percent_b(price, vwma_data)
        
        # Manual calculation
        expected = (price - vwma_data['vwma_lower']) / (vwma_data['vwma_upper'] - vwma_data['vwma_lower'])
        pd.testing.assert_series_equal(percent_b, expected)
        
    def test_volume_confirmation_complete(self):
        """Test lines 226-269: volume_confirmation method."""
        # Create comprehensive test data
        dates = pd.date_range(start='2023-01-01', periods=40, freq='D')
        
        # Base values
        price = pd.Series(100.0, index=dates)
        volume = pd.Series(1000000, index=dates)
        
        # Create different scenarios
        # Scenario 1: Price above VWMA with high volume (bullish)
        price.iloc[10:15] = 105
        volume.iloc[10:15] = 1600000
        
        # Scenario 2: Price below VWMA with high volume (bearish)
        price.iloc[20:25] = 95
        volume.iloc[20:25] = 1500000
        
        # Scenario 3: Large price move with low volume
        price.iloc[30] = 103  # 3% move
        volume.iloc[30] = 600000
        
        data = pd.DataFrame({'close': price, 'volume': volume})
        
        vwma = VWMABands(period=10)
        vwma_data = vwma.calculate(data)
        
        # Force VWMA to be 100 for clear above/below testing
        vwma_data['vwma'] = 100.0
        
        # Execute volume_confirmation (lines 243-269)
        vol_signals = vwma.volume_confirmation(data, vwma_data)
        
        # Verify DataFrame creation (line 243)
        assert isinstance(vol_signals, pd.DataFrame)
        assert vol_signals.index.equals(data.index)
        
        # Verify all columns exist
        assert 'bullish_volume' in vol_signals.columns
        assert 'bearish_volume' in vol_signals.columns  
        assert 'low_volume_move' in vol_signals.columns
        
        # Verify calculations
        avg_volume = volume.rolling(window=10).mean()
        
        # Bullish volume (lines 252-255)
        bullish_expected = (price > 100.0) & (volume > avg_volume * 1.5)
        pd.testing.assert_series_equal(vol_signals['bullish_volume'], bullish_expected)
        
        # Bearish volume (lines 258-261)
        bearish_expected = (price < 100.0) & (volume > avg_volume * 1.5)
        pd.testing.assert_series_equal(vol_signals['bearish_volume'], bearish_expected)
        
        # Low volume move (lines 264-267)
        low_vol_expected = (abs(price.pct_change()) > 0.02) & (volume < avg_volume * 0.7)
        pd.testing.assert_series_equal(vol_signals['low_volume_move'], low_vol_expected)
        
        # Verify we have all types of signals
        assert vol_signals['bullish_volume'].any()
        assert vol_signals['bearish_volume'].any()
        assert vol_signals['low_volume_move'].any()
        
    def test_edge_cases_for_coverage(self):
        """Test edge cases to ensure complete coverage."""
        vwma = VWMABands(period=5)
        
        # Test with custom volume column
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'vol': [1000, 1100, 1200, 1300, 1400]
        })
        
        result = vwma.calculate(data, volume_column='vol')
        assert not result.empty
        
        # Test different price columns
        data['high'] = data['close'] + 1
        vwma_high = VWMABands(price_column='high')
        result_high = vwma_high.calculate(data, volume_column='vol')
        assert not result_high.empty
        
    def test_validation_errors(self):
        """Test error handling with invalid data."""
        vwma = VWMABands()
        
        # Test empty DataFrame
        with pytest.raises(IndicatorError, match="Data is empty or None"):
            vwma.calculate(pd.DataFrame())
            
        # Test missing columns
        data = pd.DataFrame({'close': [100, 101, 102]})
        with pytest.raises(IndicatorError, match="Missing required columns"):
            vwma.calculate(data)
            
        # Test None data
        with pytest.raises(IndicatorError, match="Data is empty or None"):
            vwma.calculate(None)
            
    def test_full_integration(self):
        """Integration test covering all methods together."""
        # Create realistic market data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Trending market with volatility
        trend = np.linspace(100, 110, 100)
        noise = np.random.randn(100) * 2
        price = trend + noise
        
        # Volume correlated with price changes
        price_changes = np.abs(np.diff(price, prepend=price[0]))
        volume = 1000000 + price_changes * 100000 + np.random.randint(-50000, 50000, 100)
        
        data = pd.DataFrame({
            'close': price,
            'volume': volume
        }, index=dates)
        
        # Test full workflow
        vwma = VWMABands(period=20, band_multiplier=2.0)
        vwma_result = vwma.calculate(data)
        signals = vwma.get_signals(data, vwma_result)
        percent_b = vwma.calculate_percent_b(data['close'], vwma_result)
        vol_conf = vwma.volume_confirmation(data, vwma_result)
        
        # Verify all components work together
        assert not vwma_result.empty
        assert not signals.empty
        assert not percent_b.empty
        assert not vol_conf.empty
        
        # Verify data integrity
        assert len(vwma_result) == len(data)
        assert len(signals) == len(data)
        assert len(percent_b) == len(data)
        assert len(vol_conf) == len(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/indicators/vwma", "--cov-report=term-missing"])