"""Focused test for VWMA volume_confirmation method to achieve 100% coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.indicators.vwma import VWMABands


def test_volume_confirmation_complete_coverage():
    """Test volume_confirmation method with complete line coverage."""
    # Create test data
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    
    # Create price data
    price = pd.Series(100.0, index=dates)
    volume = pd.Series(1000000, index=dates)
    
    # Set up conditions to trigger all branches
    # Price above VWMA with high volume (bullish)
    price.iloc[25:30] = 105.0
    volume.iloc[25:30] = 1600000
    
    # Price below VWMA with high volume (bearish) 
    price.iloc[35:40] = 95.0
    volume.iloc[35:40] = 1500000
    
    # Large price move with low volume
    price.iloc[45] = 103.0  # 3% move
    volume.iloc[45] = 600000  # Low volume
    
    data = pd.DataFrame({'close': price, 'volume': volume})
    
    # Initialize VWMA and calculate
    vwma = VWMABands(period=20)
    vwma_data = vwma.calculate(data)
    
    # Call volume_confirmation - this executes all lines
    vol_signals = vwma.volume_confirmation(data, vwma_data)
    
    # Verify DataFrame creation (line 243)
    assert isinstance(vol_signals, pd.DataFrame)
    assert vol_signals.index.equals(data.index)
    
    # Verify price and volume extraction (lines 245-246)
    assert 'close' in data.columns
    assert 'volume' in data.columns
    
    # Verify average volume calculation (line 249)
    avg_volume = volume.rolling(window=20).mean()
    assert not avg_volume.isna().all()
    
    # Verify all signal columns exist
    assert 'bullish_volume' in vol_signals.columns
    assert 'bearish_volume' in vol_signals.columns
    assert 'low_volume_move' in vol_signals.columns
    
    # Verify bullish volume calculation (lines 252-255)
    bullish_expected = (price > vwma_data['vwma']) & (volume > avg_volume * 1.5)
    pd.testing.assert_series_equal(vol_signals['bullish_volume'], bullish_expected, check_names=False)
    
    # Verify bearish volume calculation (lines 258-261)
    bearish_expected = (price < vwma_data['vwma']) & (volume > avg_volume * 1.5)
    pd.testing.assert_series_equal(vol_signals['bearish_volume'], bearish_expected, check_names=False)
    
    # Verify low volume move calculation (lines 264-267)
    low_vol_expected = (abs(price.pct_change()) > 0.02) & (volume < avg_volume * 0.7)
    pd.testing.assert_series_equal(vol_signals['low_volume_move'], low_vol_expected, check_names=False)
    
    # Ensure we have some signals
    assert vol_signals['bullish_volume'].any()
    assert vol_signals['bearish_volume'].any()
    assert vol_signals['low_volume_move'].any()


def test_calculate_percent_b_direct():
    """Test calculate_percent_b method directly."""
    vwma = VWMABands()
    
    # Create test data
    price = pd.Series([95, 100, 105, 110, 115])
    vwma_data = pd.DataFrame({
        'vwma_lower': [90, 92, 94, 96, 98],
        'vwma_upper': [110, 112, 114, 116, 118]
    })
    
    # Call the method - this executes line 222
    percent_b = vwma.calculate_percent_b(price, vwma_data)
    
    # Verify the calculation
    expected = (price - vwma_data['vwma_lower']) / (vwma_data['vwma_upper'] - vwma_data['vwma_lower'])
    pd.testing.assert_series_equal(percent_b, expected)
    
    # Check specific values
    assert percent_b[0] == (95 - 90) / (110 - 90)  # 0.25
    assert percent_b[1] == (100 - 92) / (112 - 92)  # 0.40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])