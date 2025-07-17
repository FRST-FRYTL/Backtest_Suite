"""Comprehensive tests for Bollinger Bands indicator to achieve 100% coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.indicators.bollinger import BollingerBands
from src.indicators.base import IndicatorError


class TestBollingerBandsComplete:
    """Complete test suite for 100% Bollinger Bands coverage."""
    
    def test_initialization(self):
        """Test lines 19-39: __init__ method with all branches."""
        # Default initialization
        bb1 = BollingerBands()
        assert bb1.name == "Bollinger_Bands"
        assert bb1.period == 20
        assert bb1.std_dev == 2.0
        assert bb1.ma_type == "sma"
        
        # Custom initialization with SMA
        bb2 = BollingerBands(period=30, std_dev=1.5, ma_type="SMA")
        assert bb2.period == 30
        assert bb2.std_dev == 1.5
        assert bb2.ma_type == "sma"  # Should be lowercase
        
        # Custom initialization with EMA
        bb3 = BollingerBands(period=10, std_dev=3.0, ma_type="EMA")
        assert bb3.ma_type == "ema"
        
        # Test invalid ma_type (line 39)
        with pytest.raises(ValueError, match="Invalid ma_type: wma"):
            BollingerBands(ma_type="wma")
            
    def test_calculate_with_sma(self):
        """Test lines 41-77: calculate method with SMA."""
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        prices = 100 + np.cumsum(np.random.randn(30) * 0.5)
        data = pd.DataFrame({'close': prices}, index=dates)
        
        bb = BollingerBands(period=10, std_dev=2.0, ma_type="sma")
        result = bb.calculate(data)
        
        # Verify all columns exist
        expected_cols = ['bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_percent']
        for col in expected_cols:
            assert col in result.columns
            
        # Verify relationships
        assert (result['bb_upper'] > result['bb_middle']).all()
        assert (result['bb_lower'] < result['bb_middle']).all()
        assert (result['bb_width'] > 0).all()
        
        # Test bb_percent calculation (line 75)
        expected_percent = (data['close'] - result['bb_lower']) / result['bb_width']
        pd.testing.assert_series_equal(result['bb_percent'], expected_percent)
        
    def test_calculate_with_ema(self):
        """Test calculate method with EMA (lines 63-64)."""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        prices = 100 + np.cumsum(np.random.randn(30) * 0.5)
        data = pd.DataFrame({'close': prices}, index=dates)
        
        bb = BollingerBands(period=10, std_dev=2.0, ma_type="ema")
        result = bb.calculate(data)
        
        # Should have all columns
        assert 'bb_middle' in result.columns
        assert 'bb_upper' in result.columns
        
        # EMA should be different from SMA
        bb_sma = BollingerBands(period=10, std_dev=2.0, ma_type="sma")
        result_sma = bb_sma.calculate(data)
        
        # After warm-up period, should be different
        assert not result['bb_middle'].iloc[15:].equals(result_sma['bb_middle'].iloc[15:])
        
    def test_calculate_with_custom_column(self):
        """Test calculate with different price column."""
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        data = pd.DataFrame({
            'close': 100 + np.random.randn(20),
            'high': 101 + np.random.randn(20),
            'low': 99 + np.random.randn(20),
            'open': 100 + np.random.randn(20)
        }, index=dates)
        
        bb = BollingerBands()
        
        # Calculate with high prices
        result_high = bb.calculate(data, price_column='high')
        assert not result_high.empty
        
        # Calculate with low prices
        result_low = bb.calculate(data, price_column='low')
        assert not result_low.empty
        
        # Should be different
        assert result_high['bb_middle'].mean() > result_low['bb_middle'].mean()
        
    def test_get_signals(self):
        """Test lines 79-146: get_signals method with all signal types."""
        # Create data that will trigger various signals
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        
        # Create price pattern that touches bands
        base_price = 100
        prices = []
        for i in range(50):
            if i < 10:
                prices.append(base_price)
            elif i < 20:
                prices.append(base_price + i - 10)  # Rising to upper band
            elif i < 30:
                prices.append(base_price + 10 - (i - 20))  # Falling to lower band
            elif i < 40:
                prices.append(base_price - 10 + (i - 30))  # Rising back
            else:
                prices.append(base_price)
                
        data = pd.DataFrame({
            'close': prices,
            'high': [p + 0.5 for p in prices],
            'low': [p - 0.5 for p in prices],
            'open': [p + np.random.randn() * 0.1 for p in prices]
        }, index=dates)
        
        bb = BollingerBands(period=10, std_dev=2.0)
        bb_data = bb.calculate(data)
        signals = bb.get_signals(data, bb_data)
        
        # Verify all signal columns exist
        expected_signals = [
            'touch_upper', 'touch_lower', 'break_above_upper', 'break_below_lower',
            'reenter_from_above', 'reenter_from_below', 'cross_above_middle',
            'cross_below_middle', 'squeeze', 'squeeze_release'
        ]
        for signal in expected_signals:
            assert signal in signals.columns
            
        # Should have some signals
        assert signals['touch_upper'].any()
        assert signals['touch_lower'].any()
        assert signals['cross_above_middle'].any()
        assert signals['cross_below_middle'].any()
        
    def test_get_signals_without_close_column(self):
        """Test get_signals when 'close' column doesn't exist (line 97)."""
        # Create data without 'close' column
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        data = pd.DataFrame({
            'open': 100 + np.random.randn(20),
            'high': 101 + np.random.randn(20),
            'low': 99 + np.random.randn(20),
            'price': 100 + np.random.randn(20)  # 4th column will be used
        }, index=dates)
        
        bb = BollingerBands()
        bb_data = bb.calculate(data, price_column='price')
        
        # Should use iloc[:, 3] when 'close' not found
        signals = bb.get_signals(data, bb_data)
        assert not signals.empty
        
    def test_calculate_bandwidth(self):
        """Test lines 148-158: calculate_bandwidth method."""
        bb = BollingerBands()
        
        # Create sample BB data
        bb_data = pd.DataFrame({
            'bb_upper': [102, 103, 104, 105],
            'bb_middle': [100, 100, 100, 100],
            'bb_lower': [98, 97, 96, 95]
        })
        
        bandwidth = bb.calculate_bandwidth(bb_data)
        
        # Verify calculation: (upper - lower) / middle
        expected = (bb_data['bb_upper'] - bb_data['bb_lower']) / bb_data['bb_middle']
        pd.testing.assert_series_equal(bandwidth, expected)
        
    def test_detect_patterns(self):
        """Test lines 160-204: detect_patterns method."""
        # Create data with patterns
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        
        # Create W-bottom and M-top patterns
        prices = []
        for i in range(50):
            if i < 10:
                prices.append(100)
            elif i < 15:
                prices.append(95 - (i - 10))  # First low
            elif i < 20:
                prices.append(90 + (i - 15))  # Rise
            elif i < 25:
                prices.append(95 - (i - 20) * 0.5)  # Second low (higher)
            elif i < 30:
                prices.append(92.5 + (i - 25) * 2)  # Strong rise (W complete)
            elif i < 35:
                prices.append(102.5 + (i - 30))  # First high
            elif i < 40:
                prices.append(107.5 - (i - 35))  # Fall
            elif i < 45:
                prices.append(102.5 + (i - 40) * 0.5)  # Second high (lower)
            else:
                prices.append(105 - (i - 45))  # Fall (M complete)
                
        data = pd.DataFrame({
            'close': prices,
            'high': [p + 0.5 for p in prices],
            'low': [p - 0.5 for p in prices],
            'open': prices
        }, index=dates)
        
        bb = BollingerBands(period=10)
        bb_data = bb.calculate(data)
        patterns = bb.detect_patterns(data, bb_data, lookback=20)
        
        # Check all pattern columns exist
        assert 'w_bottom' in patterns.columns
        assert 'm_top' in patterns.columns
        assert 'walking_upper' in patterns.columns
        assert 'walking_lower' in patterns.columns
        
        # Should detect some patterns
        assert patterns['walking_upper'].any() or patterns['walking_lower'].any()
        
    def test_detect_patterns_without_ohlc_columns(self):
        """Test detect_patterns when OHLC columns don't exist (lines 179-181)."""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        
        # Create data with non-standard column names
        data = pd.DataFrame({
            'price': 100 + np.random.randn(30),
            'max': 101 + np.random.randn(30),
            'min': 99 + np.random.randn(30),
            'first': 100 + np.random.randn(30)
        }, index=dates)
        
        bb = BollingerBands()
        bb_data = bb.calculate(data, price_column='price')
        
        # Should use iloc to get columns
        patterns = bb.detect_patterns(data, bb_data, lookback=10)
        assert not patterns.empty
        
    def test_detect_w_bottom(self):
        """Test lines 206-234: _detect_w_bottom method."""
        bb = BollingerBands()
        
        # Create data with W-bottom pattern
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        
        # Pattern: decline -> first low -> rise -> second low (higher) -> rise
        close = pd.Series([100, 98, 96, 94, 92, 90,  # Decline
                          91, 92, 93, 94,  # First rise
                          93, 92, 91,  # Second decline (higher low)
                          92, 93, 94, 95, 96, 97, 98,  # Strong rise
                          99, 100, 101, 102, 103, 104, 105, 106, 107, 108], index=dates)
        
        low = close - 0.5
        lower_band = pd.Series([95] * 30, index=dates)  # Fixed for simplicity
        lower_band.iloc[5:7] = 90  # At first low
        lower_band.iloc[11:13] = 91  # At second low
        
        result = bb._detect_w_bottom(close, low, lower_band, lookback=15)
        
        # Should detect W-bottom after second low rises
        assert result.any()
        
    def test_detect_m_top(self):
        """Test lines 236-264: _detect_m_top method."""
        bb = BollingerBands()
        
        # Create data with M-top pattern
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        
        # Pattern: rise -> first high -> fall -> second high (lower) -> fall
        close = pd.Series([100, 102, 104, 106, 108, 110,  # Rise
                          109, 108, 107, 106,  # First fall
                          107, 108, 109,  # Second rise (lower high)
                          108, 107, 106, 105, 104, 103, 102,  # Strong fall
                          101, 100, 99, 98, 97, 96, 95, 94, 93, 92], index=dates)
        
        high = close + 0.5
        upper_band = pd.Series([105] * 30, index=dates)  # Fixed for simplicity
        upper_band.iloc[5:7] = 110  # At first high
        upper_band.iloc[11:13] = 109  # At second high
        
        result = bb._detect_m_top(close, high, upper_band, lookback=15)
        
        # Should detect M-top after second high falls
        assert result.any()
        
    def test_validation_errors(self):
        """Test error handling."""
        bb = BollingerBands()
        
        # Empty DataFrame
        with pytest.raises(IndicatorError, match="Data is empty or None"):
            bb.calculate(pd.DataFrame())
            
        # Missing column
        data = pd.DataFrame({'open': [100, 101, 102]})
        with pytest.raises(IndicatorError, match="Missing required columns"):
            bb.calculate(data)
            
        # None data
        with pytest.raises(IndicatorError, match="Data is empty or None"):
            bb.calculate(None)
            
    def test_edge_cases(self):
        """Test edge cases for complete coverage."""
        bb = BollingerBands(period=3, std_dev=1.0)
        
        # Very small dataset
        data = pd.DataFrame({'close': [100, 101, 102, 103]})
        result = bb.calculate(data)
        assert len(result) == len(data)
        
        # Constant prices (zero std dev)
        const_data = pd.DataFrame({'close': [100] * 10})
        result_const = bb.calculate(const_data)
        
        # Bands should collapse to middle when no volatility
        assert (result_const['bb_width'].iloc[3:] == 0).all()
        
        # Extreme values
        extreme_data = pd.DataFrame({'close': [1e-10, 1e10, 1e-10, 1e10, 1e-10]})
        result_extreme = bb.calculate(extreme_data)
        assert not np.isinf(result_extreme).any().any()
        
    def test_squeeze_signals(self):
        """Test squeeze detection in get_signals (lines 137-144)."""
        # Create data with volatility contraction and expansion
        dates = pd.date_range(start='2023-01-01', periods=40, freq='D')
        
        # High volatility -> low volatility -> high volatility
        prices = []
        for i in range(40):
            if i < 10:
                # High volatility
                prices.append(100 + np.random.randn() * 3)
            elif i < 20:
                # Low volatility (squeeze)
                prices.append(100 + np.random.randn() * 0.5)
            elif i < 30:
                # Transitioning back to high volatility
                prices.append(100 + np.random.randn() * 1.5)
            else:
                # High volatility again
                prices.append(100 + np.random.randn() * 3)
                
        data = pd.DataFrame({'close': prices}, index=dates)
        
        bb = BollingerBands(period=10)
        bb_data = bb.calculate(data)
        signals = bb.get_signals(data, bb_data)
        
        # Should detect squeeze during low volatility period
        assert signals['squeeze'].iloc[15:20].any()
        
        # Should detect squeeze release
        assert signals['squeeze_release'].any()
        
    def test_full_workflow(self):
        """Test complete Bollinger Bands workflow."""
        # Create realistic market data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Trending market with varying volatility
        trend = np.linspace(100, 110, 100)
        volatility = np.sin(np.linspace(0, 4*np.pi, 100)) * 2 + 3
        prices = trend + np.random.randn(100) * volatility
        
        data = pd.DataFrame({
            'open': prices + np.random.randn(100) * 0.1,
            'high': prices + np.abs(np.random.randn(100)) * 0.5,
            'low': prices - np.abs(np.random.randn(100)) * 0.5,
            'close': prices
        }, index=dates)
        
        # Test full workflow
        bb = BollingerBands(period=20, std_dev=2.0, ma_type='sma')
        bb_result = bb.calculate(data)
        signals = bb.get_signals(data, bb_result)
        bandwidth = bb.calculate_bandwidth(bb_result)
        patterns = bb.detect_patterns(data, bb_result, lookback=20)
        
        # Verify all components work
        assert not bb_result.empty
        assert not signals.empty
        assert not bandwidth.empty
        assert not patterns.empty
        
        # Verify data integrity
        assert len(bb_result) == len(data)
        assert len(signals) == len(data)
        assert len(bandwidth) == len(data)
        assert len(patterns) == len(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/indicators/bollinger", "--cov-report=term-missing"])