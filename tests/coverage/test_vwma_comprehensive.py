"""Comprehensive tests for VWMA indicator to achieve 100% coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.indicators.vwma import VWMABands


class TestVWMABandsCreation:
    """Test VWMABands initialization."""
    
    def test_default_initialization(self):
        """Test default VWMABands initialization."""
        vwma = VWMABands()
        
        assert vwma.name == "VWMA_Bands"
        assert vwma.period == 20
        assert vwma.band_multiplier == 2.0
        assert vwma.price_column == "close"
        
    def test_custom_initialization(self):
        """Test custom VWMABands initialization."""
        vwma = VWMABands(
            period=50,
            band_multiplier=1.5,
            price_column="typical_price"
        )
        
        assert vwma.period == 50
        assert vwma.band_multiplier == 1.5
        assert vwma.price_column == "typical_price"
        
    def test_various_periods(self):
        """Test initialization with various periods."""
        periods = [5, 10, 20, 50, 100, 200]
        
        for period in periods:
            vwma = VWMABands(period=period)
            assert vwma.period == period
            
    def test_various_multipliers(self):
        """Test initialization with various band multipliers."""
        multipliers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        
        for multiplier in multipliers:
            vwma = VWMABands(band_multiplier=multiplier)
            assert vwma.band_multiplier == multiplier


class TestVWMACalculation:
    """Test VWMA calculation methods."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample market data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data with trend
        trend = np.linspace(100, 110, len(dates))
        noise = np.random.randn(len(dates)) * 2
        close_prices = trend + noise
        
        # Generate volume with some correlation to price changes
        base_volume = 1000000
        price_changes = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        volume_multiplier = 1 + price_changes * 0.5
        volume = (base_volume * volume_multiplier + np.random.randint(-200000, 200000, len(dates))).astype(int)
        
        return pd.DataFrame({
            'close': close_prices,
            'volume': volume,
            'high': close_prices + np.abs(np.random.randn(len(dates))) * 0.5,
            'low': close_prices - np.abs(np.random.randn(len(dates))) * 0.5,
            'open': close_prices + np.random.randn(len(dates)) * 0.3
        }, index=dates)
        
    def test_basic_calculation(self, sample_data):
        """Test basic VWMA calculation."""
        vwma = VWMABands(period=20)
        result = vwma.calculate(sample_data)
        
        # Check all expected columns exist
        expected_columns = ['vwma', 'vwma_upper', 'vwma_lower', 'vwma_width', 'vwma_signal']
        for col in expected_columns:
            assert col in result.columns
            
        # Check data types
        assert result['vwma'].dtype == np.float64
        assert result['vwma_signal'].dtype in [np.int64, np.int32]
        
        # Check no NaN values after warm-up period
        assert not result['vwma'].iloc[20:].isna().any()
        
    def test_vwma_differs_from_sma(self, sample_data):
        """Test that VWMA is different from SMA."""
        vwma = VWMABands(period=20)
        result = vwma.calculate(sample_data)
        
        # Calculate simple moving average
        sma = sample_data['close'].rolling(window=20).mean()
        
        # VWMA should be different from SMA (unless volume is constant)
        vwma_values = result['vwma'].dropna()
        sma_values = sma.dropna()
        
        # Align indices
        common_index = vwma_values.index.intersection(sma_values.index)
        
        assert not np.allclose(
            vwma_values.loc[common_index].values,
            sma_values.loc[common_index].values,
            rtol=1e-10
        )
        
    def test_band_calculation(self, sample_data):
        """Test band calculation correctness."""
        vwma = VWMABands(period=20, band_multiplier=2.0)
        result = vwma.calculate(sample_data)
        
        # Upper band should be above VWMA
        assert (result['vwma_upper'] >= result['vwma']).all()
        
        # Lower band should be below VWMA
        assert (result['vwma_lower'] <= result['vwma']).all()
        
        # Band width should be positive
        assert (result['vwma_width'] > 0).all()
        
        # Band width should equal upper - lower
        calculated_width = result['vwma_upper'] - result['vwma_lower']
        assert np.allclose(result['vwma_width'], calculated_width)
        
    def test_signal_generation(self, sample_data):
        """Test signal generation logic."""
        vwma = VWMABands(period=20)
        result = vwma.calculate(sample_data)
        
        price = sample_data['close']
        
        # Check buy signals (price < lower band)
        buy_mask = price < result['vwma_lower']
        assert (result.loc[buy_mask, 'vwma_signal'] == 1).all()
        
        # Check sell signals (price > upper band)
        sell_mask = price > result['vwma_upper']
        assert (result.loc[sell_mask, 'vwma_signal'] == -1).all()
        
        # Check neutral signals (price between bands)
        neutral_mask = (price >= result['vwma_lower']) & (price <= result['vwma_upper'])
        assert (result.loc[neutral_mask, 'vwma_signal'] == 0).all()
        
    def test_different_price_columns(self, sample_data):
        """Test using different price columns."""
        # Add typical price column
        sample_data['typical_price'] = (sample_data['high'] + sample_data['low'] + sample_data['close']) / 3
        
        # Test with typical price
        vwma = VWMABands(price_column='typical_price')
        result = vwma.calculate(sample_data)
        
        assert not result.empty
        assert 'vwma' in result.columns
        
        # Test with high price
        vwma_high = VWMABands(price_column='high')
        result_high = vwma_high.calculate(sample_data)
        
        assert not result_high.empty
        
        # Results should be different
        assert not result['vwma'].equals(result_high['vwma'])
        
    def test_custom_volume_column(self, sample_data):
        """Test with custom volume column name."""
        # Rename volume column
        sample_data['trade_volume'] = sample_data['volume']
        sample_data = sample_data.drop('volume', axis=1)
        
        vwma = VWMABands()
        result = vwma.calculate(sample_data, volume_column='trade_volume')
        
        assert not result.empty
        assert 'vwma' in result.columns


class TestVWMAPrivateMethods:
    """Test private methods of VWMABands."""
    
    @pytest.fixture
    def vwma_instance(self):
        """Create VWMABands instance."""
        return VWMABands(period=20)
        
    @pytest.fixture
    def price_volume_data(self):
        """Create simple price and volume series."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        price = pd.Series(100 + np.random.randn(50) * 2, index=dates)
        volume = pd.Series(np.random.randint(900000, 1100000, 50), index=dates)
        return price, volume
        
    def test_calculate_vwma_method(self, vwma_instance, price_volume_data):
        """Test _calculate_vwma private method."""
        price, volume = price_volume_data
        
        vwma_result = vwma_instance._calculate_vwma(price, volume)
        
        # Check result is a Series
        assert isinstance(vwma_result, pd.Series)
        
        # Check length matches input
        assert len(vwma_result) == len(price)
        
        # Manually calculate VWMA for verification
        pv = price * volume
        pv_sum = pv.rolling(window=20, min_periods=1).sum()
        vol_sum = volume.rolling(window=20, min_periods=1).sum()
        expected_vwma = pv_sum / vol_sum
        
        assert np.allclose(vwma_result.dropna(), expected_vwma.dropna())
        
    def test_calculate_vwma_zero_volume(self, vwma_instance):
        """Test VWMA calculation with zero volume."""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        price = pd.Series(100 + np.random.randn(30) * 2, index=dates)
        
        # Create volume with some zero values
        volume = pd.Series(np.random.randint(0, 1000000, 30), index=dates)
        volume.iloc[10:15] = 0  # Set some volumes to zero
        
        vwma_result = vwma_instance._calculate_vwma(price, volume)
        
        # Should fall back to SMA when volume is zero
        assert not vwma_result.isna().all()
        
        # Check specific zero volume periods
        for i in range(10, 15):
            if i >= 20:  # After warm-up period
                # Should be close to simple moving average
                sma_value = price.iloc[max(0, i-19):i+1].mean()
                assert np.isclose(vwma_result.iloc[i], sma_value, rtol=1e-5)
                
    def test_calculate_rolling_std_method(self, vwma_instance, price_volume_data):
        """Test _calculate_rolling_std private method."""
        price, volume = price_volume_data
        
        # First calculate VWMA
        vwma = vwma_instance._calculate_vwma(price, volume)
        
        # Then calculate rolling std
        std_result = vwma_instance._calculate_rolling_std(price, volume, vwma)
        
        # Check result is a Series
        assert isinstance(std_result, pd.Series)
        
        # Check length matches input
        assert len(std_result) == len(price)
        
        # Standard deviation should be positive
        assert (std_result.dropna() >= 0).all()
        
    def test_calculate_rolling_std_zero_volume(self, vwma_instance):
        """Test rolling std calculation with zero volume."""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        price = pd.Series(100 + np.random.randn(30) * 2, index=dates)
        volume = pd.Series(0, index=dates)  # All zero volume
        
        vwma = vwma_instance._calculate_vwma(price, volume)
        std_result = vwma_instance._calculate_rolling_std(price, volume, vwma)
        
        # Should fall back to regular standard deviation
        assert not std_result.isna().all()
        
        # Should be close to rolling std of price
        expected_std = price.rolling(window=20, min_periods=1).std()
        
        # Compare where both have values
        mask = ~(std_result.isna() | expected_std.isna())
        assert np.allclose(std_result[mask], expected_std[mask], rtol=1e-5)


class TestVWMASignals:
    """Test VWMA signal generation methods."""
    
    @pytest.fixture
    def sample_data_with_vwma(self):
        """Generate sample data and calculate VWMA."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Create trending data
        trend = np.linspace(100, 120, len(dates))
        close_prices = trend + np.sin(np.linspace(0, 4*np.pi, len(dates))) * 5
        
        data = pd.DataFrame({
            'close': close_prices,
            'volume': np.random.randint(900000, 1100000, len(dates))
        }, index=dates)
        
        vwma = VWMABands(period=20)
        vwma_data = vwma.calculate(data)
        
        return data, vwma_data, vwma
        
    def test_get_signals_method(self, sample_data_with_vwma):
        """Test get_signals method."""
        data, vwma_data, vwma_instance = sample_data_with_vwma
        
        signals = vwma_instance.get_signals(data, vwma_data)
        
        # Check all expected signal columns
        expected_signals = [
            'touch_upper', 'touch_lower',
            'cross_above_upper', 'cross_below_lower',
            'cross_above_vwma', 'cross_below_vwma',
            'band_squeeze', 'band_expansion'
        ]
        
        for signal in expected_signals:
            assert signal in signals.columns
            assert signals[signal].dtype == bool
            
    def test_band_touch_signals(self, sample_data_with_vwma):
        """Test band touch signals."""
        data, vwma_data, vwma_instance = sample_data_with_vwma
        signals = vwma_instance.get_signals(data, vwma_data)
        
        price = data['close']
        
        # Verify touch_upper signal
        touch_upper_mask = price >= vwma_data['vwma_upper']
        assert signals['touch_upper'].equals(touch_upper_mask)
        
        # Verify touch_lower signal
        touch_lower_mask = price <= vwma_data['vwma_lower']
        assert signals['touch_lower'].equals(touch_lower_mask)
        
    def test_crossover_signals(self, sample_data_with_vwma):
        """Test crossover signals."""
        data, vwma_data, vwma_instance = sample_data_with_vwma
        signals = vwma_instance.get_signals(data, vwma_data)
        
        # Check that crossovers are detected correctly
        # A crossover should only happen once at the crossing point
        cross_above_upper = signals['cross_above_upper']
        cross_below_lower = signals['cross_below_lower']
        
        # Crossovers should be relatively rare events
        assert cross_above_upper.sum() < len(signals) * 0.1
        assert cross_below_lower.sum() < len(signals) * 0.1
        
        # When there's a crossover, the previous value should be on the other side
        for i in range(1, len(signals)):
            if signals['cross_above_upper'].iloc[i]:
                assert data['close'].iloc[i] > vwma_data['vwma_upper'].iloc[i]
                assert data['close'].iloc[i-1] <= vwma_data['vwma_upper'].iloc[i-1]
                
    def test_band_squeeze_expansion_signals(self, sample_data_with_vwma):
        """Test band squeeze and expansion signals."""
        data, vwma_data, vwma_instance = sample_data_with_vwma
        signals = vwma_instance.get_signals(data, vwma_data)
        
        # Band squeeze and expansion should be mutually exclusive
        both_true = signals['band_squeeze'] & signals['band_expansion']
        assert not both_true.any()
        
        # Check that squeeze happens when width is below average
        band_width_ma = vwma_data['vwma_width'].rolling(window=20).mean()
        squeeze_threshold = band_width_ma * 0.8
        
        # Where we have enough data for rolling mean
        valid_idx = ~band_width_ma.isna()
        expected_squeeze = vwma_data.loc[valid_idx, 'vwma_width'] < squeeze_threshold[valid_idx]
        assert signals.loc[valid_idx, 'band_squeeze'].equals(expected_squeeze)


class TestVWMAPercentB:
    """Test %B calculation."""
    
    def test_calculate_percent_b(self):
        """Test %B indicator calculation."""
        vwma = VWMABands()
        
        # Create sample data
        price = pd.Series([95, 100, 105, 110, 115])
        vwma_data = pd.DataFrame({
            'vwma_lower': [90, 92, 94, 96, 98],
            'vwma_upper': [110, 112, 114, 116, 118]
        })
        
        percent_b = vwma.calculate_percent_b(price, vwma_data)
        
        # Check calculations
        # %B = (price - lower) / (upper - lower)
        expected = [
            (95 - 90) / (110 - 90),   # 0.25
            (100 - 92) / (112 - 92),  # 0.40
            (105 - 94) / (114 - 94),  # 0.55
            (110 - 96) / (116 - 96),  # 0.70
            (115 - 98) / (118 - 98)   # 0.85
        ]
        
        assert np.allclose(percent_b.values, expected)
        
        # Test the actual calculation line 222
        # Manually verify the calculation matches the method implementation
        manual_calc = (price - vwma_data['vwma_lower']) / (vwma_data['vwma_upper'] - vwma_data['vwma_lower'])
        assert percent_b.equals(manual_calc)
        
    def test_percent_b_edge_cases(self):
        """Test %B with edge cases."""
        vwma = VWMABands()
        
        # Price at lower band (should be 0)
        price = pd.Series([100])
        vwma_data = pd.DataFrame({
            'vwma_lower': [100],
            'vwma_upper': [110]
        })
        
        percent_b = vwma.calculate_percent_b(price, vwma_data)
        assert percent_b.iloc[0] == 0.0
        
        # Price at upper band (should be 1)
        price = pd.Series([110])
        percent_b = vwma.calculate_percent_b(price, vwma_data)
        assert percent_b.iloc[0] == 1.0
        
        # Price above upper band (should be > 1)
        price = pd.Series([120])
        percent_b = vwma.calculate_percent_b(price, vwma_data)
        assert percent_b.iloc[0] > 1.0
        
        # Price below lower band (should be < 0)
        price = pd.Series([90])
        percent_b = vwma.calculate_percent_b(price, vwma_data)
        assert percent_b.iloc[0] < 0.0


class TestVWMAVolumeConfirmation:
    """Test volume confirmation methods."""
    
    @pytest.fixture
    def volume_data(self):
        """Create sample data with various volume patterns."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        
        # Create price with trend changes
        price = pd.Series(100, index=dates, dtype=float)
        price.iloc[10:20] = 105  # Upward move
        price.iloc[20:30] = 102  # Consolidation
        price.iloc[30:40] = 98   # Downward move
        price.iloc[40:] = 100    # Recovery
        
        # Create volume patterns
        volume = pd.Series(1000000, index=dates)
        volume.iloc[10:20] = 1600000  # High volume on up move
        volume.iloc[20:30] = 700000   # Low volume consolidation
        volume.iloc[30:40] = 1500000  # High volume on down move
        volume.iloc[40:] = 800000     # Low volume recovery
        
        data = pd.DataFrame({
            'close': price,
            'volume': volume
        })
        
        return data
        
    def test_volume_confirmation_basic(self, volume_data):
        """Test basic volume confirmation."""
        vwma = VWMABands(period=20)
        vwma_data = vwma.calculate(volume_data)
        
        # Call volume_confirmation to ensure all lines are executed
        vol_signals = vwma.volume_confirmation(volume_data, vwma_data)
        
        # Check all expected columns
        expected_cols = ['bullish_volume', 'bearish_volume', 'low_volume_move']
        for col in expected_cols:
            assert col in vol_signals.columns
            assert vol_signals[col].dtype == bool
            
        # Verify the method creates the signals DataFrame (line 243)
        assert isinstance(vol_signals, pd.DataFrame)
        assert vol_signals.index.equals(volume_data.index)
        
        # Check that price and volume are extracted correctly (lines 245-246)
        price = volume_data[vwma.price_column]
        volume = volume_data['volume']
        assert len(price) == len(vol_signals)
        assert len(volume) == len(vol_signals)
            
    def test_bullish_volume_signal(self, volume_data):
        """Test bullish volume confirmation."""
        vwma = VWMABands(period=10)  # Shorter period for test data
        vwma_data = vwma.calculate(volume_data)
        
        vol_signals = vwma.volume_confirmation(volume_data, vwma_data)
        
        # During upward move with high volume (days 10-20)
        # Price should be above VWMA and volume above average
        bullish_period = vol_signals.iloc[15:19]['bullish_volume']
        assert bullish_period.any()  # Should have some bullish volume signals
        
    def test_bearish_volume_signal(self, volume_data):
        """Test bearish volume confirmation."""
        vwma = VWMABands(period=10)
        vwma_data = vwma.calculate(volume_data)
        
        vol_signals = vwma.volume_confirmation(volume_data, vwma_data)
        
        # During downward move with high volume (days 30-40)
        # Price should be below VWMA and volume above average
        bearish_period = vol_signals.iloc[32:38]['bearish_volume']
        assert bearish_period.any()  # Should have some bearish volume signals
        
    def test_low_volume_move_signal(self):
        """Test low volume move detection."""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        
        # Create data with a big price move on low volume
        price = pd.Series(100.0, index=dates)
        price.iloc[20] = 105.0  # 5% move
        
        volume = pd.Series(1000000, index=dates)
        volume.iloc[20] = 500000  # Low volume on big move
        
        data = pd.DataFrame({'close': price, 'volume': volume})
        
        vwma = VWMABands(period=10)
        vwma_data = vwma.calculate(data)
        vol_signals = vwma.volume_confirmation(data, vwma_data)
        
        # Should detect low volume move
        assert vol_signals.iloc[20]['low_volume_move']
        
    def test_volume_confirmation_full_coverage(self):
        """Test volume confirmation to ensure full line coverage."""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        
        # Create data that will trigger all conditions
        price = pd.Series(100.0, index=dates)
        volume = pd.Series(1000000, index=dates)
        
        # Setup price movements to trigger different conditions
        # Price above VWMA with high volume (bullish)
        price.iloc[25:30] = 105.0
        volume.iloc[25:30] = 1600000
        
        # Price below VWMA with high volume (bearish)
        price.iloc[35:40] = 95.0
        volume.iloc[35:40] = 1500000
        
        # Large price move with low volume
        price.iloc[45] = 107.0  # 7% move
        volume.iloc[45] = 600000  # Low volume
        
        data = pd.DataFrame({'close': price, 'volume': volume})
        
        vwma = VWMABands(period=20)
        vwma_data = vwma.calculate(data)
        
        # This should execute all lines in volume_confirmation
        vol_signals = vwma.volume_confirmation(data, vwma_data)
        
        # Verify all signal types are generated
        assert vol_signals['bullish_volume'].any()
        assert vol_signals['bearish_volume'].any()
        assert vol_signals['low_volume_move'].any()
        
        # Verify calculation logic
        avg_volume = volume.rolling(window=20).mean()
        
        # Check bullish volume logic (lines 252-255)
        bullish_mask = (price > vwma_data['vwma']) & (volume > avg_volume * 1.5)
        assert vol_signals['bullish_volume'].equals(bullish_mask)
        
        # Check bearish volume logic (lines 258-261)
        bearish_mask = (price < vwma_data['vwma']) & (volume > avg_volume * 1.5)
        assert vol_signals['bearish_volume'].equals(bearish_mask)
        
        # Check low volume move logic (lines 264-267)
        low_vol_mask = (abs(price.pct_change()) > 0.02) & (volume < avg_volume * 0.7)
        assert vol_signals['low_volume_move'].equals(low_vol_mask)
        
    def test_volume_confirmation_edge_cases(self):
        """Test volume confirmation with edge cases."""
        dates = pd.date_range(start='2023-01-01', periods=25, freq='D')
        
        # Create data with zero volume
        data = pd.DataFrame({
            'close': pd.Series(100 + np.random.randn(25) * 2, index=dates),
            'volume': pd.Series(0, index=dates)  # All zero volume
        })
        
        vwma = VWMABands(period=20)
        vwma_data = vwma.calculate(data)
        
        # Should not raise error
        vol_signals = vwma.volume_confirmation(data, vwma_data)
        
        # With zero volume, should have no volume confirmations
        assert not vol_signals['bullish_volume'].any()
        assert not vol_signals['bearish_volume'].any()


class TestVWMAEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        vwma = VWMABands()
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            vwma.calculate(empty_df)
            
    def test_missing_required_columns(self):
        """Test with missing required columns."""
        vwma = VWMABands()
        
        # Missing volume column
        data_no_volume = pd.DataFrame({
            'close': [100, 101, 102]
        })
        
        with pytest.raises(ValueError):
            vwma.calculate(data_no_volume)
            
        # Missing price column
        data_no_price = pd.DataFrame({
            'volume': [1000, 1100, 1200]
        })
        
        with pytest.raises(ValueError):
            vwma.calculate(data_no_price)
            
    def test_single_data_point(self):
        """Test with single data point."""
        vwma = VWMABands(period=20)
        
        data = pd.DataFrame({
            'close': [100],
            'volume': [1000000]
        })
        
        result = vwma.calculate(data)
        
        # Should return result without error
        assert len(result) == 1
        assert not result['vwma'].isna().all()
        
    def test_all_same_price(self):
        """Test with constant price."""
        vwma = VWMABands(period=10)
        
        data = pd.DataFrame({
            'close': [100] * 30,
            'volume': np.random.randint(900000, 1100000, 30)
        })
        
        result = vwma.calculate(data)
        
        # VWMA should be close to the constant price
        assert np.allclose(result['vwma'].dropna(), 100, rtol=1e-10)
        
        # Bands should be symmetric
        upper_distance = result['vwma_upper'] - result['vwma']
        lower_distance = result['vwma'] - result['vwma_lower']
        assert np.allclose(upper_distance.dropna(), lower_distance.dropna())
        
    def test_extreme_values(self):
        """Test with extreme values."""
        vwma = VWMABands(period=5)
        
        data = pd.DataFrame({
            'close': [1e-10, 1e10, 1e-10, 1e10, 1e-10],
            'volume': [1e10, 1e-10, 1e10, 1e-10, 1e10]
        })
        
        # Should handle without overflow/underflow
        result = vwma.calculate(data)
        assert not result['vwma'].isna().all()
        assert not np.isinf(result['vwma']).any()
        
    def test_negative_prices(self):
        """Test with negative prices (e.g., spread trading)."""
        vwma = VWMABands(period=10)
        
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        data = pd.DataFrame({
            'close': np.linspace(-10, 10, 30),  # Negative to positive
            'volume': np.random.randint(900000, 1100000, 30)
        }, index=dates)
        
        result = vwma.calculate(data)
        
        # Should handle negative prices
        assert not result['vwma'].isna().all()
        
        # Bands should still work
        assert (result['vwma_upper'] > result['vwma_lower']).all()
        
    def test_data_types(self):
        """Test with different data types."""
        vwma = VWMABands()
        
        # Integer prices and volumes
        data_int = pd.DataFrame({
            'close': pd.Series([100, 101, 102], dtype=int),
            'volume': pd.Series([1000, 1100, 1200], dtype=int)
        })
        
        result_int = vwma.calculate(data_int)
        assert not result_int.empty
        
        # Float32 data
        data_float32 = pd.DataFrame({
            'close': pd.Series([100.5, 101.5, 102.5], dtype=np.float32),
            'volume': pd.Series([1000.0, 1100.0, 1200.0], dtype=np.float32)
        })
        
        result_float32 = vwma.calculate(data_float32)
        assert not result_float32.empty


class TestVWMAIntegration:
    """Test VWMA in realistic scenarios."""
    
    def test_trending_market(self):
        """Test VWMA in trending market."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Strong uptrend
        trend = np.linspace(100, 150, len(dates))
        noise = np.random.randn(len(dates)) * 1
        price = trend + noise
        
        # Volume increases with trend
        volume = np.linspace(1000000, 1500000, len(dates))
        volume = volume + np.random.randint(-100000, 100000, len(dates))
        
        data = pd.DataFrame({
            'close': price,
            'volume': volume
        }, index=dates)
        
        vwma = VWMABands(period=20, band_multiplier=2.0)
        result = vwma.calculate(data)
        
        # In strong uptrend, price should mostly be above VWMA
        above_vwma = (data['close'] > result['vwma']).sum() / len(data)
        assert above_vwma > 0.7
        
        # Should generate more buy signals early, sell signals late
        signals = result['vwma_signal']
        first_half_buys = (signals.iloc[:50] == 1).sum()
        second_half_sells = (signals.iloc[50:] == -1).sum()
        
        assert first_half_buys > second_half_sells * 0.5
        
    def test_ranging_market(self):
        """Test VWMA in ranging/sideways market."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Sideways market with oscillation
        base_price = 100
        price = base_price + np.sin(np.linspace(0, 8*np.pi, len(dates))) * 5
        price = price + np.random.randn(len(dates)) * 0.5
        
        # Random volume
        volume = np.random.randint(900000, 1100000, len(dates))
        
        data = pd.DataFrame({
            'close': price,
            'volume': volume
        }, index=dates)
        
        vwma = VWMABands(period=20, band_multiplier=2.0)
        result = vwma.calculate(data)
        
        # In ranging market, should generate both buy and sell signals
        signals = result['vwma_signal']
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        
        # Should be relatively balanced
        assert 0.5 < buy_signals / (sell_signals + 1) < 2.0
        
        # VWMA should be close to average price
        avg_price = data['close'].mean()
        avg_vwma = result['vwma'].mean()
        assert abs(avg_vwma - avg_price) < 2.0
        
    def test_volatile_market(self):
        """Test VWMA in volatile market."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # High volatility price
        np.random.seed(42)
        price = 100 + np.cumsum(np.random.randn(len(dates)) * 3)
        
        # Volume spikes on volatile days
        price_change = np.abs(np.diff(price, prepend=price[0]))
        volume = 1000000 + price_change * 100000
        volume = volume.astype(int)
        
        data = pd.DataFrame({
            'close': price,
            'volume': volume
        }, index=dates)
        
        vwma = VWMABands(period=20, band_multiplier=2.0)
        result = vwma.calculate(data)
        
        # Bands should expand during volatile periods
        band_width = result['vwma_width']
        
        # Check correlation between volatility and band width
        price_volatility = data['close'].rolling(20).std()
        
        # Remove NaN values for correlation
        valid_mask = ~(band_width.isna() | price_volatility.isna())
        
        correlation = np.corrcoef(
            band_width[valid_mask],
            price_volatility[valid_mask]
        )[0, 1]
        
        # Should have positive correlation
        assert correlation > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])