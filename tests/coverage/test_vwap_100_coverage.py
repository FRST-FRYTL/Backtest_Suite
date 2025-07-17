"""Comprehensive tests for VWAP indicators to achieve 100% coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

from src.indicators.vwap import VWAP, AnchoredVWAP
from src.indicators.base import IndicatorError


class TestVWAPComplete:
    """Complete test suite for 100% VWAP coverage."""
    
    def test_initialization(self):
        """Test lines 19-36: VWAP __init__ method."""
        # Default initialization
        vwap1 = VWAP()
        assert vwap1.name == "VWAP"
        assert vwap1.window is None
        assert vwap1.price_type == "typical"
        assert vwap1.std_dev_bands == [1.0, 2.0, 3.0]
        
        # Custom initialization
        vwap2 = VWAP(window=20, price_type="CLOSE", std_dev_bands=[0.5, 1.0, 1.5])
        assert vwap2.window == 20
        assert vwap2.price_type == "close"  # Should be lowercase
        assert vwap2.std_dev_bands == [0.5, 1.0, 1.5]
        
    def test_calculate_rolling_vwap(self):
        """Test calculate method with rolling window."""
        # Create test data
        dates = pd.date_range(start='2023-01-01 09:30:00', periods=30, freq='5min')
        data = pd.DataFrame({
            'open': 100 + np.random.randn(30) * 0.5,
            'high': 101 + np.random.randn(30) * 0.5,
            'low': 99 + np.random.randn(30) * 0.5,
            'close': 100 + np.cumsum(np.random.randn(30) * 0.2),
            'volume': np.random.randint(1000, 5000, 30)
        }, index=dates)
        
        vwap = VWAP(window=10)
        result = vwap.calculate(data)
        
        # Check columns exist
        assert 'vwap' in result.columns
        assert 'vwap_volume' in result.columns
        assert 'vwap_std' in result.columns
        assert 'vwap_upper_1.0' in result.columns
        assert 'vwap_lower_1.0' in result.columns
        
        # VWAP should be between high and low
        assert (result['vwap'] <= data['high'] * 1.1).all()
        assert (result['vwap'] >= data['low'] * 0.9).all()
        
    def test_calculate_session_vwap(self):
        """Test calculate method with session-based VWAP (lines 64-66)."""
        # Create intraday data spanning two days
        start = datetime(2023, 1, 1, 9, 30)
        dates = []
        for day in range(2):
            for minute in range(390):  # 6.5 hours of trading
                dates.append(start + timedelta(days=day, minutes=minute))
                
        data = pd.DataFrame({
            'high': 101 + np.random.randn(len(dates)) * 0.5,
            'low': 99 + np.random.randn(len(dates)) * 0.5,
            'close': 100 + np.random.randn(len(dates)) * 0.5,
            'volume': np.random.randint(1000, 5000, len(dates))
        }, index=pd.DatetimeIndex(dates))
        
        vwap = VWAP(window=None)  # Session-based
        result = vwap.calculate(data, reset_time=time(9, 30))
        
        # Check that VWAP resets at session start
        # First data point of each day should have VWAP equal to price
        assert np.isclose(result['vwap'].iloc[0], 
                         (data['high'].iloc[0] + data['low'].iloc[0] + data['close'].iloc[0]) / 3)
        assert np.isclose(result['vwap'].iloc[390], 
                         (data['high'].iloc[390] + data['low'].iloc[390] + data['close'].iloc[390]) / 3)
        
    def test_calculate_price_types(self):
        """Test lines 76-87: _calculate_price method with all price types."""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })
        
        # Test typical/hlc3
        vwap_typical = VWAP(price_type="typical")
        result = vwap_typical.calculate(data)
        assert not result.empty
        
        # Test close
        vwap_close = VWAP(price_type="close")
        result = vwap_close.calculate(data)
        assert not result.empty
        
        # Test hl2
        vwap_hl2 = VWAP(price_type="hl2")
        result = vwap_hl2.calculate(data)
        assert not result.empty
        
        # Test ohlc4
        vwap_ohlc4 = VWAP(price_type="ohlc4")
        result = vwap_ohlc4.calculate(data)
        assert not result.empty
        
        # Test invalid price type
        vwap_invalid = VWAP(price_type="invalid")
        with pytest.raises(ValueError, match="Invalid price_type"):
            vwap_invalid.calculate(data)
            
    def test_get_required_columns(self):
        """Test lines 89-100: _get_required_columns method."""
        # Test all price types
        vwap = VWAP(price_type="typical")
        assert vwap._get_required_columns() == ['high', 'low', 'close']
        
        vwap = VWAP(price_type="close")
        assert vwap._get_required_columns() == ['close']
        
        vwap = VWAP(price_type="hl2")
        assert vwap._get_required_columns() == ['high', 'low']
        
        vwap = VWAP(price_type="ohlc4")
        assert vwap._get_required_columns() == ['open', 'high', 'low', 'close']
        
        # Test invalid type defaults to close
        vwap = VWAP(price_type="invalid")
        vwap.price_type = "invalid"  # Force invalid
        assert vwap._get_required_columns() == ['close']
        
    def test_calculate_rolling_vwap_method(self):
        """Test lines 102-130: _calculate_rolling_vwap method."""
        vwap = VWAP(window=5)
        
        price = pd.Series([100, 101, 102, 103, 104, 105, 106])
        volume = pd.Series([1000, 1100, 1200, 1300, 1400, 1500, 1600])
        
        result = vwap._calculate_rolling_vwap(price, volume)
        
        assert 'vwap' in result.columns
        assert 'vwap_volume' in result.columns
        assert len(result) == len(price)
        
        # Test with window larger than data (lines 117-120)
        vwap_large = VWAP(window=100)
        result_large = vwap_large._calculate_rolling_vwap(price, volume)
        assert not result_large['vwap'].isna().all()
        
        # Test with None window (line 108)
        vwap_none = VWAP(window=None)
        result_none = vwap_none._calculate_rolling_vwap(price, volume)
        assert not result_none['vwap'].isna().all()
        
    def test_calculate_session_vwap_method(self):
        """Test lines 132-165: _calculate_session_vwap method."""
        vwap = VWAP()
        
        # Create multi-day intraday data
        dates = []
        base_date = datetime(2023, 1, 1)
        for day in range(3):
            for hour in range(7):
                for minute in range(0, 60, 5):
                    dt = base_date + timedelta(days=day, hours=9+hour, minutes=minute)
                    dates.append(dt)
                    
        index = pd.DatetimeIndex(dates)
        price = pd.Series(100 + np.random.randn(len(dates)) * 0.5, index=index)
        volume = pd.Series(np.random.randint(1000, 2000, len(dates)), index=index)
        
        result = vwap._calculate_session_vwap(price, volume, index, time(9, 30))
        
        assert 'vwap' in result.columns
        assert 'vwap_volume' in result.columns
        assert len(result) == len(price)
        
        # Check session resets
        # Volume should reset at start of each day
        vol_at_930 = result[result.index.time == time(9, 30)]['vwap_volume']
        assert (vol_at_930 == volume[volume.index.time == time(9, 30)]).all()
        
    def test_add_std_bands(self):
        """Test lines 167-209: _add_std_bands method."""
        vwap = VWAP(window=10, std_dev_bands=[1.0, 2.0, 3.0])
        
        # Create sample data
        price = pd.Series(100 + np.random.randn(30) * 2)
        volume = pd.Series(np.random.randint(1000, 2000, 30))
        vwap_data = pd.DataFrame({
            'vwap': price.rolling(10).mean(),
            'vwap_volume': volume.rolling(10).sum()
        })
        
        result = vwap._add_std_bands(vwap_data, price, volume)
        
        # Check all bands exist
        assert 'vwap_upper_1.0' in result.columns
        assert 'vwap_lower_1.0' in result.columns
        assert 'vwap_upper_2.0' in result.columns
        assert 'vwap_lower_2.0' in result.columns
        assert 'vwap_upper_3.0' in result.columns
        assert 'vwap_lower_3.0' in result.columns
        assert 'vwap_std' in result.columns
        
        # Test cumulative calculation (lines 192-200)
        vwap_cum = VWAP(window=None, std_dev_bands=[1.5])
        result_cum = vwap_cum._add_std_bands(vwap_data, price, volume)
        assert 'vwap_upper_1.5' in result_cum.columns
        
    def test_validation_errors(self):
        """Test error handling."""
        vwap = VWAP()
        
        # Empty DataFrame
        with pytest.raises(IndicatorError, match="Data is empty or None"):
            vwap.calculate(pd.DataFrame())
            
        # Missing required columns
        data = pd.DataFrame({'close': [100, 101, 102]})
        with pytest.raises(IndicatorError, match="Missing required columns"):
            vwap.calculate(data)
            
    def test_edge_cases(self):
        """Test edge cases."""
        # Zero volume
        data = pd.DataFrame({
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
            'volume': [0, 0, 0]
        })
        
        vwap = VWAP(window=2)
        result = vwap.calculate(data)
        # Should handle division by zero gracefully (fillna with price)
        assert not result['vwap'].isna().all()
        
        # Single data point
        single_data = pd.DataFrame({
            'high': [101],
            'low': [99],
            'close': [100],
            'volume': [1000]
        }, index=pd.DatetimeIndex(['2023-01-01']))
        
        result_single = vwap.calculate(single_data)
        assert len(result_single) == 1
        

class TestAnchoredVWAPComplete:
    """Complete test suite for AnchoredVWAP coverage."""
    
    def test_anchored_initialization(self):
        """Test lines 217-239: AnchoredVWAP __init__ method."""
        # String anchor date
        avwap1 = AnchoredVWAP("2023-01-01")
        assert avwap1.name == "Anchored_VWAP"
        assert avwap1.anchor_date == pd.to_datetime("2023-01-01")
        assert avwap1.price_type == "typical"
        assert avwap1.std_dev_bands == [1.0, 2.0]
        
        # Datetime anchor date
        anchor_dt = datetime(2023, 1, 15)
        avwap2 = AnchoredVWAP(anchor_dt, price_type="CLOSE", std_dev_bands=[1.5])
        assert avwap2.anchor_date == anchor_dt
        assert avwap2.price_type == "close"
        assert avwap2.std_dev_bands == [1.5]
        
    def test_anchored_calculate(self):
        """Test lines 241-280: calculate method."""
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'open': 100 + np.random.randn(50) * 0.5,
            'high': 101 + np.random.randn(50) * 0.5,
            'low': 99 + np.random.randn(50) * 0.5,
            'close': 100 + np.cumsum(np.random.randn(50) * 0.2),
            'volume': np.random.randint(100000, 200000, 50)
        }, index=dates)
        
        # Test with exact anchor date in data
        avwap = AnchoredVWAP("2023-01-10")
        result = avwap.calculate(data)
        
        # Check renamed columns
        assert 'avwap' in result.columns
        assert 'avwap_std' in result.columns
        
        # Values before anchor should be NaN
        assert result.loc[:'2023-01-09', 'avwap'].isna().all()
        
        # Values from anchor onward should not be NaN
        assert not result.loc['2023-01-10':, 'avwap'].isna().all()
        
    def test_anchored_calculate_missing_date(self):
        """Test calculate when anchor date not in index (lines 260-267)."""
        # Create data without the specific anchor date
        dates = pd.date_range(start='2023-01-01', periods=30, freq='2D')  # Every 2 days
        data = pd.DataFrame({
            'high': 101 + np.random.randn(30),
            'low': 99 + np.random.randn(30),
            'close': 100 + np.random.randn(30),
            'volume': np.random.randint(100000, 200000, 30)
        }, index=dates)
        
        # Anchor on a missing date
        avwap = AnchoredVWAP("2023-01-02")  # This date is not in index
        result = avwap.calculate(data)
        
        # Should start from next available date
        assert result.loc['2023-01-01', 'avwap'].isna()
        assert not result.loc['2023-01-03', 'avwap'].isna()
        
    def test_anchored_calculate_future_date(self):
        """Test calculate with future anchor date."""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        data = pd.DataFrame({
            'high': 101 + np.random.randn(30),
            'low': 99 + np.random.randn(30),
            'close': 100 + np.random.randn(30),
            'volume': np.random.randint(100000, 200000, 30)
        }, index=dates)
        
        # Anchor date after all data
        avwap = AnchoredVWAP("2024-01-01")
        with pytest.raises(ValueError, match="No data available after anchor date"):
            avwap.calculate(data)
            
    def test_create_multiple_anchors(self):
        """Test lines 283-317: create_multiple_anchors static method."""
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        data = pd.DataFrame({
            'high': 101 + np.random.randn(60),
            'low': 99 + np.random.randn(60),
            'close': 100 + np.random.randn(60),
            'volume': np.random.randint(100000, 200000, 60)
        }, index=dates)
        
        # Test with multiple anchor dates
        anchor_dates = ["2023-01-10", datetime(2023, 1, 20), "2023-01-30"]
        result = AnchoredVWAP.create_multiple_anchors(
            data, anchor_dates, price_type="close"
        )
        
        # Check columns exist for each anchor
        assert 'avwap_2023-01-10' in result.columns
        assert 'avwap_20230120' in result.columns
        assert 'avwap_2023-01-30' in result.columns
        
        # Each should have appropriate NaN pattern
        assert result.loc[:'2023-01-09', 'avwap_2023-01-10'].isna().all()
        assert not result.loc['2023-01-10':, 'avwap_2023-01-10'].isna().all()
        
    def test_anchor_from_events(self):
        """Test lines 320-365: anchor_from_events static method."""
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'high': 101 + np.random.randn(100),
            'low': 99 + np.random.randn(100),
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'volume': np.random.randint(100000, 200000, 100)
        }, index=dates)
        
        # Create events (e.g., new highs)
        events = pd.Series(False, index=dates)
        events.iloc[20] = True  # Event on day 20
        events.iloc[50] = True  # Event on day 50
        events.iloc[80] = True  # Event on day 80
        
        result = AnchoredVWAP.anchor_from_events(
            data, events, price_type="close", lookback_days=30
        )
        
        # Should have columns for each event
        assert any('avwap_event_' in col for col in result.columns)
        
        # Test with no events
        no_events = pd.Series(False, index=dates)
        result_no_events = AnchoredVWAP.anchor_from_events(
            data, no_events
        )
        assert result_no_events.empty
        
    def test_full_workflow(self):
        """Test complete VWAP workflow with both classes."""
        # Create realistic market data
        dates = pd.date_range(start='2023-01-01 09:30:00', periods=1000, freq='5min')
        np.random.seed(42)
        
        # Simulate price with trend and intraday patterns
        trend = np.linspace(100, 105, 1000)
        intraday = np.sin(np.linspace(0, 10*np.pi, 1000)) * 0.5
        noise = np.random.randn(1000) * 0.1
        close = trend + intraday + noise
        
        data = pd.DataFrame({
            'open': close + np.random.randn(1000) * 0.05,
            'high': close + np.abs(np.random.randn(1000)) * 0.1,
            'low': close - np.abs(np.random.randn(1000)) * 0.1,
            'close': close,
            'volume': np.random.randint(50000, 150000, 1000)
        }, index=dates)
        
        # Test rolling VWAP
        vwap_rolling = VWAP(window=50, price_type="typical")
        result_rolling = vwap_rolling.calculate(data)
        
        # Test session VWAP
        vwap_session = VWAP(window=None, price_type="typical")
        result_session = vwap_session.calculate(data, reset_time=time(9, 30))
        
        # Test anchored VWAP
        avwap = AnchoredVWAP(dates[100], price_type="typical")
        result_anchored = avwap.calculate(data)
        
        # Test multiple anchors
        anchors = [dates[50], dates[200], dates[500]]
        result_multi = AnchoredVWAP.create_multiple_anchors(data, anchors)
        
        # Verify all results
        assert not result_rolling.empty
        assert not result_session.empty
        assert not result_anchored.empty
        assert not result_multi.empty
        
        # Verify data integrity
        assert len(result_rolling) == len(data)
        assert len(result_session) == len(data)
        assert len(result_anchored) == len(data)
        assert len(result_multi) == len(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/indicators/vwap", "--cov-report=term-missing"])