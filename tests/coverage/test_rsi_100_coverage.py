"""Comprehensive tests for RSI indicator to achieve 100% coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.indicators.rsi import RSI
from src.indicators.base import IndicatorError


class TestRSIComplete:
    """Complete test suite for 100% RSI coverage."""
    
    def test_initialization(self):
        """Test lines 19-36: __init__ method."""
        # Default initialization
        rsi1 = RSI()
        assert rsi1.name == "RSI"
        assert rsi1.period == 14
        assert rsi1.overbought == 70.0
        assert rsi1.oversold == 30.0
        
        # Custom initialization
        rsi2 = RSI(period=21, overbought=80.0, oversold=20.0)
        assert rsi2.period == 21
        assert rsi2.overbought == 80.0
        assert rsi2.oversold == 20.0
        
    def test_calculate_basic(self):
        """Test lines 38-80: calculate method basic functionality."""
        # Create test data with known price movements
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                  111, 110, 112, 114, 113, 115, 117, 116, 118, 120,
                  119, 121, 123, 122, 124, 126, 125, 127, 129, 128]
        
        data = pd.DataFrame({'close': prices}, index=dates)
        
        rsi = RSI(period=14)
        result = rsi.calculate(data)
        
        # Verify result is a Series
        assert isinstance(result, pd.Series)
        assert result.name == 'rsi'
        
        # RSI should be between 0 and 100
        assert (result.dropna() >= 0).all()
        assert (result.dropna() <= 100).all()
        
        # First 14 values should be NaN
        assert result.iloc[:14].isna().all()
        assert not result.iloc[14:].isna().any()
        
    def test_calculate_with_losses_only(self):
        """Test RSI calculation when prices only go down."""
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        prices = list(range(100, 80, -1))  # Decreasing prices
        
        data = pd.DataFrame({'close': prices}, index=dates)
        
        rsi = RSI(period=14)
        result = rsi.calculate(data)
        
        # RSI should be very low (near 0) when only losses
        assert (result.dropna() < 30).all()
        
    def test_calculate_with_gains_only(self):
        """Test RSI calculation when prices only go up."""
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        prices = list(range(100, 120))  # Increasing prices
        
        data = pd.DataFrame({'close': prices}, index=dates)
        
        rsi = RSI(period=14)
        result = rsi.calculate(data)
        
        # RSI should be very high (near 100) when only gains
        assert (result.dropna() > 70).all()
        
    def test_calculate_with_no_change(self):
        """Test RSI when prices don't change (line 75 - division by zero handling)."""
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        prices = [100] * 20  # No price change
        
        data = pd.DataFrame({'close': prices}, index=dates)
        
        rsi = RSI(period=14)
        result = rsi.calculate(data)
        
        # When no losses (avg_losses = 0), RSI should be 50
        # First 14 values are NaN, rest should be 50
        assert result.iloc[14:].eq(50).all()
        
    def test_calculate_with_custom_column(self):
        """Test calculation with different price column."""
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        data = pd.DataFrame({
            'close': range(100, 120),
            'high': range(101, 121),
            'low': range(99, 119)
        }, index=dates)
        
        rsi = RSI()
        
        # Calculate with high prices
        result_high = rsi.calculate(data, price_column='high')
        assert not result_high.empty
        
        # Calculate with low prices
        result_low = rsi.calculate(data, price_column='low')
        assert not result_low.empty
        
        # Results should be different
        assert not result_high.equals(result_low)
        
    def test_calculate_average_method(self):
        """Test lines 82-100: _calculate_average method."""
        rsi = RSI(period=14)
        
        # Create test series
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                           11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        
        result = rsi._calculate_average(series, 14)
        
        # First 14 values should be NaN (line 98)
        assert result.iloc[:14].isna().all()
        
        # Rest should have values
        assert not result.iloc[14:].isna().any()
        
        # Test with different period
        result2 = rsi._calculate_average(series, 5)
        assert result2.iloc[:5].isna().all()
        assert not result2.iloc[5:].isna().any()
        
    def test_get_signals(self):
        """Test lines 102-142: get_signals method."""
        # Create RSI values that trigger all signal types
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        rsi_values = [25, 28, 32, 35, 40, 45, 48, 52, 55, 60,
                      65, 72, 75, 73, 68, 55, 45, 35, 25, 20]
        
        rsi_series = pd.Series(rsi_values, index=dates, name='rsi')
        
        rsi = RSI(period=14, overbought=70, oversold=30)
        signals = rsi.get_signals(rsi_series)
        
        # Check all signal columns exist
        expected_cols = ['oversold', 'overbought', 'cross_above_oversold',
                        'cross_below_overbought', 'cross_above_50', 'cross_below_50']
        for col in expected_cols:
            assert col in signals.columns
            
        # Test oversold signals (line 115)
        assert signals['oversold'].iloc[0] == True  # 25 < 30
        assert signals['oversold'].iloc[9] == False  # 60 > 30
        
        # Test overbought signals (line 118)
        assert signals['overbought'].iloc[11] == True  # 72 > 70
        assert signals['overbought'].iloc[0] == False  # 25 < 70
        
        # Test cross above oversold (lines 121-124)
        # RSI crosses from 28 to 32 (crosses above 30)
        assert signals['cross_above_oversold'].iloc[2] == True
        
        # Test cross below overbought (lines 126-129)
        # RSI crosses from 73 to 68 (crosses below 70)
        assert signals['cross_below_overbought'].iloc[14] == True
        
        # Test cross above 50 (lines 132-135)
        # RSI crosses from 48 to 52
        assert signals['cross_above_50'].iloc[7] == True
        
        # Test cross below 50 (lines 137-140)
        # RSI crosses from 55 to 45
        assert signals['cross_below_50'].iloc[16] == True
        
    def test_divergence_detection(self):
        """Test lines 144-185: divergence method."""
        # Create price and RSI data with divergences
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        
        # Price making higher highs
        price_pattern = [100, 102, 104, 106, 108, 110, 108, 106, 104, 102,
                        100, 102, 104, 106, 108, 112, 110, 108, 106, 104,
                        102, 104, 106, 108, 110, 114, 112, 110, 108, 106,
                        104, 102, 100, 98, 96, 94, 92, 90, 88, 86,
                        84, 82, 80, 78, 76, 74, 72, 70, 68, 66]
        
        # RSI making lower highs (bearish divergence)
        rsi_pattern = [50, 55, 60, 65, 70, 75, 70, 65, 60, 55,
                      50, 55, 60, 65, 70, 73, 68, 63, 58, 53,
                      48, 53, 58, 63, 68, 71, 66, 61, 56, 51,
                      46, 41, 36, 31, 26, 21, 26, 31, 36, 41,
                      46, 51, 56, 61, 66, 71, 76, 81, 86, 91]
        
        price = pd.Series(price_pattern, index=dates)
        rsi_series = pd.Series(rsi_pattern, index=dates)
        
        rsi = RSI()
        divergences = rsi.divergence(price, rsi_series, window=5)
        
        # Check columns exist
        assert 'bearish' in divergences.columns
        assert 'bullish' in divergences.columns
        
        # Should detect at least one divergence
        assert divergences['bearish'].any() or divergences['bullish'].any()
        
    def test_find_peaks_and_troughs(self):
        """Test lines 187-199: _find_peaks and _find_troughs methods."""
        rsi = RSI()
        
        # Create series with clear peaks and troughs
        dates = pd.date_range(start='2023-01-01', periods=21, freq='D')
        values = [50, 55, 60, 65, 70, 65, 60, 55, 50, 45, 40, 45, 50, 55, 60, 55, 50, 45, 40, 35, 30]
        series = pd.Series(values, index=dates)
        
        # Find peaks (line 189-192)
        peaks = rsi._find_peaks(series, window=3)
        assert len(peaks) > 0
        assert 70 in peaks.values  # Should find the peak at 70
        
        # Find troughs (line 196-199)
        troughs = rsi._find_troughs(series, window=3)
        assert len(troughs) > 0
        assert 40 in troughs.values  # Should find the trough at 40
        
    def test_validation_errors(self):
        """Test error handling."""
        rsi = RSI()
        
        # Empty DataFrame
        with pytest.raises(IndicatorError, match="Data is empty or None"):
            rsi.calculate(pd.DataFrame())
            
        # Missing column
        data = pd.DataFrame({'open': [100, 101, 102]})
        with pytest.raises(IndicatorError, match="Missing required columns"):
            rsi.calculate(data)
            
        # None data
        with pytest.raises(IndicatorError, match="Data is empty or None"):
            rsi.calculate(None)
            
    def test_edge_cases(self):
        """Test edge cases for complete coverage."""
        rsi = RSI(period=5)
        
        # Very small dataset
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105]
        })
        
        result = rsi.calculate(data)
        assert len(result) == len(data)
        assert result.iloc[:5].isna().all()  # First 5 should be NaN
        assert not pd.isna(result.iloc[5])  # 6th should have value
        
        # Extreme values
        extreme_data = pd.DataFrame({
            'close': [1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10]
        })
        
        result_extreme = rsi.calculate(extreme_data)
        assert not np.isinf(result_extreme.dropna()).any()
        
    def test_full_workflow(self):
        """Test complete RSI workflow."""
        # Create realistic market data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Simulate price with trend and volatility
        trend = np.linspace(100, 110, 100)
        noise = np.random.randn(100) * 2
        price = trend + noise
        
        data = pd.DataFrame({'close': price}, index=dates)
        
        # Full workflow
        rsi = RSI(period=14, overbought=70, oversold=30)
        rsi_values = rsi.calculate(data)
        signals = rsi.get_signals(rsi_values)
        divergences = rsi.divergence(data['close'], rsi_values, window=7)
        
        # Verify all components work
        assert not rsi_values.empty
        assert not signals.empty
        assert not divergences.empty
        
        # Verify data integrity
        assert len(rsi_values) == len(data)
        assert len(signals) == len(data)
        assert len(divergences) == len(data)
        
        # RSI values should be reasonable
        assert rsi_values.dropna().min() >= 0
        assert rsi_values.dropna().max() <= 100
        
    def test_different_periods(self):
        """Test RSI with different periods."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(50))
        }, index=dates)
        
        # Test different periods
        for period in [5, 9, 14, 21, 30]:
            rsi = RSI(period=period)
            result = rsi.calculate(data)
            
            # First 'period' values should be NaN
            assert result.iloc[:period].isna().all()
            assert not result.iloc[period:].isna().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/indicators/rsi", "--cov-report=term-missing"])