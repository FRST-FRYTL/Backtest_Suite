"""Tests for technical indicators."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.indicators import (
    RSI, BollingerBands, VWMABands, TSV, VWAP, AnchoredVWAP,
    FearGreedIndex, InsiderTrading, MaxPain
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    close_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
    
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


class TestRSI:
    """Test RSI indicator."""
    
    def test_rsi_calculation(self, sample_data):
        """Test basic RSI calculation."""
        rsi = RSI(period=14)
        result = rsi.calculate(sample_data)
        
        # Check output
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.name == 'rsi'
        
        # RSI should be between 0 and 100
        assert result.min() >= 0
        assert result.max() <= 100
        
        # Check for NaN handling
        assert result.iloc[14:].notna().all()
        
    def test_rsi_signals(self, sample_data):
        """Test RSI signal generation."""
        rsi = RSI(period=14, overbought=70, oversold=30)
        rsi_values = rsi.calculate(sample_data)
        signals = rsi.get_signals(rsi_values)
        
        # Check signal columns
        expected_columns = [
            'oversold', 'overbought', 'cross_above_oversold',
            'cross_below_overbought', 'cross_above_50', 'cross_below_50'
        ]
        assert all(col in signals.columns for col in expected_columns)
        
        # Signals should be boolean
        for col in signals.columns:
            assert signals[col].dtype == bool


class TestBollingerBands:
    """Test Bollinger Bands indicator."""
    
    def test_bollinger_calculation(self, sample_data):
        """Test Bollinger Bands calculation."""
        bb = BollingerBands(period=20, std_dev=2.0)
        result = bb.calculate(sample_data)
        
        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert all(col in result.columns for col in 
                  ['bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_percent'])
        
        # Bands relationship
        assert (result['bb_upper'] > result['bb_middle']).all()
        assert (result['bb_middle'] > result['bb_lower']).all()
        
    def test_bollinger_patterns(self, sample_data):
        """Test pattern detection."""
        bb = BollingerBands()
        bb_data = bb.calculate(sample_data)
        patterns = bb.detect_patterns(sample_data, bb_data)
        
        # Check pattern columns
        assert 'w_bottom' in patterns.columns
        assert 'm_top' in patterns.columns
        assert 'walking_upper' in patterns.columns
        assert 'walking_lower' in patterns.columns


class TestVWMABands:
    """Test VWMA Bands indicator."""
    
    def test_vwma_calculation(self, sample_data):
        """Test VWMA calculation."""
        vwma = VWMABands(period=20)
        result = vwma.calculate(sample_data)
        
        # Check output
        assert 'vwma' in result.columns
        assert 'vwma_upper' in result.columns
        assert 'vwma_lower' in result.columns
        
        # VWMA should be different from SMA
        sma = sample_data['close'].rolling(20).mean()
        assert not result['vwma'].equals(sma)


class TestTSV:
    """Test Time Segmented Volume indicator."""
    
    def test_tsv_calculation(self, sample_data):
        """Test TSV calculation."""
        tsv = TSV(period=13, signal_period=9)
        result = tsv.calculate(sample_data)
        
        # Check output
        assert 'tsv' in result.columns
        assert 'tsv_signal' in result.columns
        assert 'tsv_histogram' in result.columns
        
        # TSV should respond to price changes
        assert result['tsv'].std() > 0


class TestVWAP:
    """Test VWAP indicators."""
    
    def test_rolling_vwap(self, sample_data):
        """Test rolling VWAP calculation."""
        vwap = VWAP(window=20)
        result = vwap.calculate(sample_data)
        
        # Check output
        assert 'vwap' in result.columns
        assert 'vwap_std' in result.columns
        assert 'vwap_upper_1.0' in result.columns
        assert 'vwap_lower_1.0' in result.columns
        
    def test_anchored_vwap(self, sample_data):
        """Test anchored VWAP calculation."""
        anchor_date = sample_data.index[50]
        avwap = AnchoredVWAP(anchor_date)
        result = avwap.calculate(sample_data)
        
        # Check output
        assert 'avwap' in result.columns
        
        # Should be NaN before anchor date
        assert result.loc[:anchor_date].iloc[:-1]['avwap'].isna().all()
        
        # Should have values after anchor date
        assert result.loc[anchor_date:]['avwap'].notna().all()


class TestMetaIndicators:
    """Test meta indicators."""
    
    @pytest.mark.asyncio
    async def test_fear_greed_index(self):
        """Test Fear and Greed Index fetching."""
        fg = FearGreedIndex()
        
        # Test current value fetch
        current = await fg.fetch_current()
        assert isinstance(current, dict)
        assert 'value' in current
        assert 0 <= current['value'] <= 100
        
    def test_max_pain_calculation(self):
        """Test max pain calculation with mock data."""
        # Create mock options data
        strikes = [90, 95, 100, 105, 110]
        
        calls = pd.DataFrame({
            'strike': strikes,
            'openInterest': [100, 200, 500, 300, 150],
            'lastPrice': [12, 8, 5, 2, 0.5]
        })
        
        puts = pd.DataFrame({
            'strike': strikes,
            'openInterest': [150, 300, 500, 200, 100],
            'lastPrice': [0.5, 2, 5, 8, 12]
        })
        
        max_pain = MaxPain()
        result = max_pain._calculate_max_pain(calls, puts)
        
        # Check result structure
        assert 'max_pain_price' in result
        assert 'pain_distribution' in result
        assert 'resistance_levels' in result
        assert 'support_levels' in result
        
        # Max pain should be one of the strikes
        assert result['max_pain_price'] in strikes


class TestIndicatorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_data(self):
        """Test indicators with empty data."""
        empty_df = pd.DataFrame()
        
        rsi = RSI()
        with pytest.raises(ValueError):
            rsi.calculate(empty_df)
            
    def test_insufficient_data(self):
        """Test with insufficient data points."""
        small_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        
        # RSI needs more than 3 points for period=14
        rsi = RSI(period=14)
        result = rsi.calculate(small_data)
        assert result.isna().sum() == len(small_data)
        
    def test_missing_columns(self, sample_data):
        """Test with missing required columns."""
        incomplete_data = sample_data[['close']]
        
        bb = BollingerBands()
        # Should work with just close prices
        result = bb.calculate(incomplete_data)
        assert not result.empty
        
        # VWMA should fail without volume
        vwma = VWMABands()
        with pytest.raises(ValueError):
            vwma.calculate(incomplete_data)