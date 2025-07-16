"""Comprehensive tests for technical indicators to achieve >95% coverage."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.indicators import (
    RSI, BollingerBands, VWMABands, TSV, VWAP, AnchoredVWAP
)
from src.indicators.base import Indicator, IndicatorError


class TestRSIComprehensive:
    """Comprehensive RSI tests for maximum coverage."""
    
    def test_rsi_initialization(self):
        """Test RSI initialization with various parameters."""
        # Default parameters
        rsi = RSI()
        assert rsi.period == 14
        assert rsi.overbought == 70.0
        assert rsi.oversold == 30.0
        
        # Custom parameters
        rsi_custom = RSI(period=21, overbought=80, oversold=20)
        assert rsi_custom.period == 21
        assert rsi_custom.overbought == 80
        assert rsi_custom.oversold == 20
    
    def test_rsi_calculation_full_coverage(self, sample_ohlcv_data):
        """Test RSI calculation with full method coverage."""
        rsi = RSI(period=14)
        
        # Test with default close column
        result = rsi.calculate(sample_ohlcv_data)
        assert isinstance(result, pd.Series)
        assert result.name == 'rsi'
        assert len(result) == len(sample_ohlcv_data)
        
        # Test with custom price column
        result_open = rsi.calculate(sample_ohlcv_data, price_column='open')
        assert isinstance(result_open, pd.Series)
        assert result_open.name == 'rsi'
        
        # Test values are within bounds
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()
    
    def test_rsi_wilder_smoothing(self, sample_ohlcv_data):
        """Test Wilder's smoothing calculation."""
        rsi = RSI(period=14)
        prices = sample_ohlcv_data['close']
        
        # Test the internal _calculate_average method
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gains = rsi._calculate_average(gains, 14)
        avg_losses = rsi._calculate_average(losses, 14)
        
        assert isinstance(avg_gains, pd.Series)
        assert isinstance(avg_losses, pd.Series)
        assert len(avg_gains) == len(prices)
        assert len(avg_losses) == len(prices)
        assert (avg_gains >= 0).all()
        assert (avg_losses >= 0).all()
    
    def test_rsi_signals_comprehensive(self, sample_ohlcv_data):
        """Test all RSI signal generation methods."""
        rsi = RSI(period=14, overbought=70, oversold=30)
        rsi_values = rsi.calculate(sample_ohlcv_data)
        signals = rsi.get_signals(rsi_values)
        
        # Check all signal columns exist
        expected_columns = [
            'oversold', 'overbought', 'cross_above_oversold',
            'cross_below_overbought', 'cross_above_50', 'cross_below_50'
        ]
        for col in expected_columns:
            assert col in signals.columns
        
        # Check signal types
        for col in expected_columns:
            assert signals[col].dtype == bool
        
        # Test signal logic
        assert (signals['oversold'] == (rsi_values < 30)).all()
        assert (signals['overbought'] == (rsi_values > 70)).all()
    
    def test_rsi_divergence_detection(self, sample_ohlcv_data):
        """Test RSI divergence detection."""
        rsi = RSI(period=14)
        rsi_values = rsi.calculate(sample_ohlcv_data)
        prices = sample_ohlcv_data['close']
        
        # Test divergence detection with different windows
        for window in [10, 14, 21]:
            divergences = rsi.divergence(prices, rsi_values, window)
            
            assert isinstance(divergences, pd.DataFrame)
            assert 'bearish' in divergences.columns
            assert 'bullish' in divergences.columns
            assert divergences['bearish'].dtype == bool
            assert divergences['bullish'].dtype == bool
    
    def test_rsi_peak_trough_detection(self, sample_ohlcv_data):
        """Test peak and trough detection methods."""
        rsi = RSI(period=14)
        prices = sample_ohlcv_data['close']
        
        # Test peak detection
        peaks = rsi._find_peaks(prices, window=5)
        assert isinstance(peaks, pd.Series)
        
        # Test trough detection
        troughs = rsi._find_troughs(prices, window=5)
        assert isinstance(troughs, pd.Series)
    
    def test_rsi_edge_cases(self):
        """Test RSI edge cases and error conditions."""
        rsi = RSI(period=14)
        
        # Test with minimal data
        minimal_data = pd.DataFrame({
            'close': [100, 101, 102],
            'open': [100, 101, 102],
            'high': [100, 101, 102],
            'low': [100, 101, 102],
            'volume': [1000, 1000, 1000]
        })
        
        result = rsi.calculate(minimal_data)
        assert isinstance(result, pd.Series)
        assert result.name == 'rsi'
        
        # Test with constant prices (no price changes)
        constant_data = pd.DataFrame({
            'close': [100] * 20,
            'open': [100] * 20,
            'high': [100] * 20,
            'low': [100] * 20,
            'volume': [1000] * 20
        })
        
        result_constant = rsi.calculate(constant_data)
        # Should fill with 50 (neutral) when no price changes
        assert (result_constant.dropna() == 50).all()
        
        # Test with only gains (no losses)
        gains_only = pd.DataFrame({
            'close': list(range(100, 120)),
            'open': list(range(100, 120)),
            'high': list(range(100, 120)),
            'low': list(range(100, 120)),
            'volume': [1000] * 20
        })
        
        result_gains = rsi.calculate(gains_only)
        assert isinstance(result_gains, pd.Series)
        # RSI should be near 100 with only gains
        assert result_gains.iloc[-1] > 90
    
    def test_rsi_data_validation(self):
        """Test RSI data validation."""
        rsi = RSI(period=14)
        
        # Test with missing required column
        invalid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [100, 101, 102],
            'low': [100, 101, 102],
            'volume': [1000, 1000, 1000]
        })
        
        with pytest.raises(IndicatorError):
            rsi.calculate(invalid_data)
        
        # Test with empty DataFrame
        with pytest.raises(IndicatorError):
            rsi.calculate(pd.DataFrame())
    
    def test_rsi_various_periods(self, sample_ohlcv_data):
        """Test RSI with various period settings."""
        periods = [5, 9, 14, 21, 30, 50]
        
        for period in periods:
            rsi = RSI(period=period)
            result = rsi.calculate(sample_ohlcv_data)
            
            assert isinstance(result, pd.Series)
            assert result.name == 'rsi'
            
            # Check that early values are NaN based on period
            if period <= len(sample_ohlcv_data):
                assert result.iloc[0].isna()  # First value should be NaN
                if period < len(sample_ohlcv_data):
                    assert result.iloc[period:].notna().any()  # Later values should exist


class TestBollingerBandsComprehensive:
    """Comprehensive Bollinger Bands tests for maximum coverage."""
    
    def test_bollinger_initialization(self):
        """Test Bollinger Bands initialization."""
        bb = BollingerBands()
        assert bb.period == 20
        assert bb.std_dev == 2.0
        
        bb_custom = BollingerBands(period=10, std_dev=2.5)
        assert bb_custom.period == 10
        assert bb_custom.std_dev == 2.5
    
    def test_bollinger_calculation_comprehensive(self, sample_ohlcv_data):
        """Test comprehensive Bollinger Bands calculation."""
        bb = BollingerBands(period=20, std_dev=2.0)
        result = bb.calculate(sample_ohlcv_data)
        
        # Check result structure
        assert isinstance(result, pd.DataFrame)
        expected_columns = ['middle', 'upper', 'lower', 'bandwidth', 'percent_b']
        for col in expected_columns:
            assert col in result.columns
        
        # Check relationships
        assert (result['upper'] >= result['middle']).all()
        assert (result['lower'] <= result['middle']).all()
        assert (result['bandwidth'] >= 0).all()
    
    def test_bollinger_signals(self, sample_ohlcv_data):
        """Test Bollinger Bands signal generation."""
        bb = BollingerBands(period=20)
        bands = bb.calculate(sample_ohlcv_data)
        signals = bb.get_signals(sample_ohlcv_data, bands)
        
        # Check signal columns
        expected_signals = [
            'squeeze', 'expansion', 'upper_breakout', 'lower_breakout',
            'mean_reversion_upper', 'mean_reversion_lower'
        ]
        for col in expected_signals:
            assert col in signals.columns
            assert signals[col].dtype == bool
    
    def test_bollinger_squeeze_detection(self, sample_ohlcv_data):
        """Test Bollinger Bands squeeze detection."""
        bb = BollingerBands(period=20)
        
        # Test squeeze detection
        is_squeeze = bb.detect_squeeze(sample_ohlcv_data)
        assert isinstance(is_squeeze, pd.Series)
        assert is_squeeze.dtype == bool
        
        # Test with custom parameters
        is_squeeze_custom = bb.detect_squeeze(sample_ohlcv_data, squeeze_threshold=0.1)
        assert isinstance(is_squeeze_custom, pd.Series)


class TestVWMABandsComprehensive:
    """Comprehensive VWMA Bands tests for maximum coverage."""
    
    def test_vwma_initialization(self):
        """Test VWMA Bands initialization."""
        vwma = VWMABands()
        assert vwma.period == 20
        assert vwma.std_dev == 2.0
        
        vwma_custom = VWMABands(period=14, std_dev=1.5)
        assert vwma_custom.period == 14
        assert vwma_custom.std_dev == 1.5
    
    def test_vwma_calculation_comprehensive(self, sample_ohlcv_data):
        """Test comprehensive VWMA calculation."""
        vwma = VWMABands(period=20)
        result = vwma.calculate(sample_ohlcv_data)
        
        # Check result structure
        assert isinstance(result, pd.DataFrame)
        expected_columns = ['vwma', 'upper', 'lower', 'bandwidth']
        for col in expected_columns:
            assert col in result.columns
        
        # Check volume-weighted nature
        assert (result['upper'] >= result['vwma']).all()
        assert (result['lower'] <= result['vwma']).all()
    
    def test_vwma_volume_impact(self, sample_ohlcv_data):
        """Test that volume impacts VWMA calculation."""
        vwma = VWMABands(period=20)
        
        # Test with normal volume
        result_normal = vwma.calculate(sample_ohlcv_data)
        
        # Test with high volume on specific days
        data_high_volume = sample_ohlcv_data.copy()
        data_high_volume.loc[data_high_volume.index[10:15], 'volume'] *= 10
        
        result_high_volume = vwma.calculate(data_high_volume)
        
        # VWMA should be different with different volume patterns
        assert not result_normal['vwma'].equals(result_high_volume['vwma'])


class TestTSVComprehensive:
    """Comprehensive Time Segmented Volume tests for maximum coverage."""
    
    def test_tsv_initialization(self):
        """Test TSV initialization."""
        tsv = TSV()
        assert tsv.period == 20
        
        tsv_custom = TSV(period=14)
        assert tsv_custom.period == 14
    
    def test_tsv_calculation_comprehensive(self, sample_ohlcv_data):
        """Test comprehensive TSV calculation."""
        tsv = TSV(period=20)
        result = tsv.calculate(sample_ohlcv_data)
        
        assert isinstance(result, pd.Series)
        assert result.name == 'tsv'
        assert len(result) == len(sample_ohlcv_data)
    
    def test_tsv_accumulation_distribution(self, sample_ohlcv_data):
        """Test TSV accumulation/distribution logic."""
        tsv = TSV(period=20)
        result = tsv.calculate(sample_ohlcv_data)
        
        # TSV should reflect accumulation/distribution
        assert isinstance(result, pd.Series)
        
        # Test with zero volume data
        zero_volume_data = sample_ohlcv_data.copy()
        zero_volume_data['volume'] = 0
        
        result_zero = tsv.calculate(zero_volume_data)
        assert isinstance(result_zero, pd.Series)
    
    def test_tsv_signals(self, sample_ohlcv_data):
        """Test TSV signal generation."""
        tsv = TSV(period=20)
        tsv_values = tsv.calculate(sample_ohlcv_data)
        signals = tsv.get_signals(sample_ohlcv_data, tsv_values)
        
        # Check signal structure
        assert isinstance(signals, pd.DataFrame)
        expected_columns = ['accumulation', 'distribution', 'divergence']
        for col in expected_columns:
            assert col in signals.columns


class TestVWAPComprehensive:
    """Comprehensive VWAP tests for maximum coverage."""
    
    def test_vwap_initialization(self):
        """Test VWAP initialization."""
        vwap = VWAP()
        assert vwap.period == 20
        
        vwap_custom = VWAP(period=50)
        assert vwap_custom.period == 50
    
    def test_vwap_calculation_comprehensive(self, sample_ohlcv_data):
        """Test comprehensive VWAP calculation."""
        vwap = VWAP(period=20)
        result = vwap.calculate(sample_ohlcv_data)
        
        assert isinstance(result, pd.DataFrame)
        expected_columns = ['vwap', 'upper_band', 'lower_band']
        for col in expected_columns:
            assert col in result.columns
        
        # Check VWAP properties
        assert (result['upper_band'] >= result['vwap']).all()
        assert (result['lower_band'] <= result['vwap']).all()
    
    def test_anchored_vwap_comprehensive(self, sample_ohlcv_data):
        """Test comprehensive Anchored VWAP calculation."""
        anchor_date = sample_ohlcv_data.index[10]
        anchored_vwap = AnchoredVWAP(anchor_date=anchor_date)
        
        result = anchored_vwap.calculate(sample_ohlcv_data)
        
        assert isinstance(result, pd.Series)
        assert result.name == 'anchored_vwap'
        
        # Test multiple anchors
        anchor_dates = [sample_ohlcv_data.index[5], sample_ohlcv_data.index[15]]
        multi_vwap = AnchoredVWAP(anchor_dates=anchor_dates)
        
        result_multi = multi_vwap.calculate(sample_ohlcv_data)
        assert isinstance(result_multi, pd.DataFrame)
    
    def test_vwap_signals(self, sample_ohlcv_data):
        """Test VWAP signal generation."""
        vwap = VWAP(period=20)
        vwap_data = vwap.calculate(sample_ohlcv_data)
        signals = vwap.get_signals(sample_ohlcv_data, vwap_data)
        
        # Check signal structure
        assert isinstance(signals, pd.DataFrame)
        expected_columns = ['above_vwap', 'below_vwap', 'cross_above', 'cross_below']
        for col in expected_columns:
            assert col in signals.columns
            assert signals[col].dtype == bool


class TestIndicatorIntegration:
    """Integration tests for multiple indicators."""
    
    def test_multiple_indicators_workflow(self, sample_ohlcv_data):
        """Test workflow with multiple indicators."""
        # Initialize indicators
        rsi = RSI(period=14)
        bb = BollingerBands(period=20)
        vwap = VWAP(period=20)
        
        # Calculate all indicators
        rsi_values = rsi.calculate(sample_ohlcv_data)
        bb_values = bb.calculate(sample_ohlcv_data)
        vwap_values = vwap.calculate(sample_ohlcv_data)
        
        # Create combined signals
        combined_signals = pd.DataFrame(index=sample_ohlcv_data.index)
        
        # RSI signals
        rsi_signals = rsi.get_signals(rsi_values)
        combined_signals['rsi_oversold'] = rsi_signals['oversold']
        combined_signals['rsi_overbought'] = rsi_signals['overbought']
        
        # Bollinger signals
        bb_signals = bb.get_signals(sample_ohlcv_data, bb_values)
        combined_signals['bb_squeeze'] = bb_signals['squeeze']
        
        # VWAP signals
        vwap_signals = vwap.get_signals(sample_ohlcv_data, vwap_values)
        combined_signals['above_vwap'] = vwap_signals['above_vwap']
        
        # Test combined analysis
        assert isinstance(combined_signals, pd.DataFrame)
        assert len(combined_signals) == len(sample_ohlcv_data)
        
        # Test confluence signals
        confluence_buy = (
            combined_signals['rsi_oversold'] & 
            combined_signals['above_vwap'] & 
            ~combined_signals['bb_squeeze']
        )
        
        confluence_sell = (
            combined_signals['rsi_overbought'] & 
            ~combined_signals['above_vwap']
        )
        
        assert isinstance(confluence_buy, pd.Series)
        assert isinstance(confluence_sell, pd.Series)
        assert confluence_buy.dtype == bool
        assert confluence_sell.dtype == bool
    
    def test_indicator_performance_comparison(self, sample_ohlcv_data):
        """Test performance comparison between indicators."""
        indicators = {
            'rsi': RSI(period=14),
            'bb': BollingerBands(period=20),
            'vwap': VWAP(period=20)
        }
        
        performance_results = {}
        
        for name, indicator in indicators.items():
            import time
            start_time = time.time()
            
            result = indicator.calculate(sample_ohlcv_data)
            
            end_time = time.time()
            performance_results[name] = {
                'calculation_time': end_time - start_time,
                'result_type': type(result).__name__,
                'result_shape': result.shape if hasattr(result, 'shape') else len(result)
            }
        
        # Verify all indicators completed successfully
        assert len(performance_results) == len(indicators)
        for name, perf in performance_results.items():
            assert perf['calculation_time'] > 0
            assert perf['result_type'] in ['Series', 'DataFrame']