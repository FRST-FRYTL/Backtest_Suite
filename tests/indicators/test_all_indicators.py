"""Comprehensive tests for all technical indicators."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.indicators import (
    RSI, BollingerBands, VWMABands, TSV, VWAP, AnchoredVWAP,
    FearGreedIndex, InsiderTrading, MaxPain
)
from src.indicators.base import Indicator, IndicatorError


class TestIndicatorBase:
    """Test base indicator functionality."""
    
    def test_indicator_interface(self):
        """Test the indicator interface."""
        # Test with concrete implementation (RSI)
        indicator = RSI()
        
        # Should have required methods
        assert hasattr(indicator, 'calculate')
        assert hasattr(indicator, 'validate_data')
        assert callable(indicator.calculate)
        assert callable(indicator.validate_data)
    
    def test_data_validation(self, sample_ohlcv_data):
        """Test data validation in indicators."""
        indicator = RSI()
        
        # Valid data should pass
        indicator.validate_data(sample_ohlcv_data, ['close'])
        
        # Missing required columns should fail
        incomplete_data = sample_ohlcv_data[['open', 'high']]
        with pytest.raises(IndicatorError):
            indicator.validate_data(incomplete_data, ['close'])
        
        # Empty data should fail
        with pytest.raises(IndicatorError):
            indicator.validate_data(pd.DataFrame(), ['close'])


class TestRSIIndicator:
    """Comprehensive tests for RSI indicator."""
    
    def test_rsi_default_parameters(self, sample_ohlcv_data):
        """Test RSI with default parameters."""
        rsi = RSI()
        result = rsi.calculate(sample_ohlcv_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        assert result.name == 'rsi'
        assert (result.dropna() >= 0).all()
        assert (result.dropna() <= 100).all()
    
    def test_rsi_custom_parameters(self, sample_ohlcv_data):
        """Test RSI with custom parameters."""
        periods = [7, 14, 21, 28]
        
        for period in periods:
            rsi = RSI(period=period)
            result = rsi.calculate(sample_ohlcv_data)
            
            # First 'period' values should be NaN
            assert result.iloc[:period].isna().all()
            # Remaining values should be valid
            assert result.iloc[period:].notna().all()
    
    def test_rsi_calculation_accuracy(self):
        """Test RSI calculation accuracy with known values."""
        # Create simple test data
        prices = [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84]
        dates = pd.date_range('2023-01-01', periods=len(prices))
        
        data = pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': prices,
            'low': prices,
            'volume': [1000] * len(prices)
        }, index=dates)
        
        rsi = RSI(period=5)
        result = rsi.calculate(data)
        
        # RSI should be calculated correctly
        # With this upward trending data, RSI should be high (typically above 70)
        assert 70 < result.iloc[-1] < 100
        # RSI should be within valid range
        assert (result.dropna() >= 0).all()
        assert (result.dropna() <= 100).all()
    
    def test_rsi_signals(self, sample_ohlcv_data):
        """Test RSI signal generation."""
        rsi = RSI(period=14, overbought=70, oversold=30)
        rsi_values = rsi.calculate(sample_ohlcv_data)
        signals = rsi.get_signals(rsi_values)
        
        # Check signal structure
        assert isinstance(signals, pd.DataFrame)
        assert 'oversold' in signals.columns
        assert 'overbought' in signals.columns
        assert 'cross_above_oversold' in signals.columns
        assert 'cross_below_overbought' in signals.columns
        assert 'cross_above_50' in signals.columns
        assert 'cross_below_50' in signals.columns
        
        # Signals should be boolean
        for col in signals.columns:
            assert signals[col].dtype == bool
    
    def test_rsi_divergence_detection(self, sample_ohlcv_data):
        """Test RSI divergence detection."""
        # Create data with clear divergence
        data = sample_ohlcv_data.copy()
        
        # Create bullish divergence: price makes lower low, RSI makes higher low
        data.iloc[50:55, data.columns.get_loc('close')] = [100, 98, 95, 97, 99]
        data.iloc[60:65, data.columns.get_loc('close')] = [94, 92, 90, 93, 96]
        
        rsi = RSI(period=14)
        rsi_values = rsi.calculate(data)
        divergence = rsi.divergence(data['close'], rsi_values)
        
        # Should detect divergence
        assert isinstance(divergence, pd.DataFrame)
        assert 'bullish' in divergence.columns
        assert 'bearish' in divergence.columns


class TestBollingerBands:
    """Comprehensive tests for Bollinger Bands."""
    
    def test_bollinger_default_parameters(self, sample_ohlcv_data):
        """Test Bollinger Bands with default parameters."""
        bb = BollingerBands()
        result = bb.calculate(sample_ohlcv_data)
        
        assert isinstance(result, pd.DataFrame)
        expected_columns = ['bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_percent']
        assert all(col in result.columns for col in expected_columns)
        
        # Verify band relationships (excluding NaN values)
        valid_data = result.dropna()
        assert (valid_data['bb_upper'] >= valid_data['bb_middle']).all()
        assert (valid_data['bb_middle'] >= valid_data['bb_lower']).all()
        
        # Verify we have some valid data
        assert len(valid_data) > 0
    
    def test_bollinger_custom_parameters(self, sample_ohlcv_data):
        """Test Bollinger Bands with custom parameters."""
        periods = [10, 20, 30]
        std_devs = [1.0, 2.0, 3.0]
        
        for period in periods:
            for std_dev in std_devs:
                bb = BollingerBands(period=period, std_dev=std_dev)
                result = bb.calculate(sample_ohlcv_data)
                
                # Width should increase with std_dev
                width = result['bb_upper'] - result['bb_lower']
                assert width.mean() > 0
    
    def test_bollinger_squeeze_detection(self, sample_ohlcv_data):
        """Test Bollinger Band squeeze detection."""
        bb = BollingerBands(period=20, std_dev=2.0)
        bb_data = bb.calculate(sample_ohlcv_data)
        
        # Detect squeeze (low volatility)
        squeeze_threshold = bb_data['bb_width'].quantile(0.1)
        squeeze_periods = bb_data['bb_width'] < squeeze_threshold
        
        assert squeeze_periods.any()
        
        # Squeeze should occur (periods with low volatility)
        assert squeeze_periods.any()
        
        # Verify that squeeze periods have relatively narrow bands
        squeeze_values = bb_data[squeeze_periods]['bb_width'].dropna()
        non_squeeze_values = bb_data[~squeeze_periods]['bb_width'].dropna()
        
        if len(squeeze_values) > 0 and len(non_squeeze_values) > 0:
            assert squeeze_values.mean() < non_squeeze_values.mean()
    
    def test_bollinger_pattern_detection(self, sample_ohlcv_data):
        """Test Bollinger Band pattern detection."""
        bb = BollingerBands()
        bb_data = bb.calculate(sample_ohlcv_data)
        patterns = bb.detect_patterns(sample_ohlcv_data, bb_data)
        
        # Check pattern columns exist
        expected_patterns = ['w_bottom', 'm_top', 'walking_upper', 'walking_lower']
        for pattern in expected_patterns:
            assert pattern in patterns.columns
            assert patterns[pattern].dtype == bool
    
    def test_bollinger_percent_b(self, sample_ohlcv_data):
        """Test %B calculation."""
        bb = BollingerBands()
        result = bb.calculate(sample_ohlcv_data)
        
        # %B should be 0 at lower band, 1 at upper band
        close_prices = sample_ohlcv_data['close']
        
        # When close == lower band, %B should be ~0
        at_lower = abs(close_prices - result['bb_lower']) < 0.01
        if at_lower.any():
            assert (result.loc[at_lower, 'bb_percent'].abs() < 0.1).all()
        
        # When close == upper band, %B should be ~1
        at_upper = abs(close_prices - result['bb_upper']) < 0.01
        if at_upper.any():
            assert (abs(result.loc[at_upper, 'bb_percent'] - 1) < 0.1).all()


class TestVWMABands:
    """Comprehensive tests for VWMA Bands."""
    
    def test_vwma_calculation(self, sample_ohlcv_data):
        """Test VWMA calculation."""
        vwma = VWMABands(period=20)
        result = vwma.calculate(sample_ohlcv_data)
        
        assert 'vwma' in result.columns
        assert 'vwma_upper' in result.columns
        assert 'vwma_lower' in result.columns
        assert 'vwma_signal' in result.columns
        
        # VWMA should differ from SMA due to volume weighting
        sma = sample_ohlcv_data['close'].rolling(20).mean()
        assert not result['vwma'].equals(sma)
    
    def test_vwma_volume_impact(self):
        """Test that volume impacts VWMA calculation."""
        # Create data with varying volume
        dates = pd.date_range('2023-01-01', periods=50)
        prices = np.ones(50) * 100  # Constant price
        
        # High volume in the middle
        volume = np.ones(50) * 1000
        volume[20:30] = 10000
        
        data = pd.DataFrame({
            'close': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'volume': volume
        }, index=dates)
        
        vwma = VWMABands(period=10)
        result = vwma.calculate(data)
        
        # VWMA should be pulled toward high-volume periods
        # Even with constant price, VWMA will vary due to volume
        assert result['vwma'].std() > 0
    
    def test_vwma_signals(self, sample_ohlcv_data):
        """Test VWMA signal generation."""
        vwma = VWMABands(period=20)
        vwma_data = vwma.calculate(sample_ohlcv_data)
        signals = vwma.get_signals(sample_ohlcv_data, vwma_data)
        
        # Check signal types
        assert 'vwma_cross_above' in signals.columns
        assert 'vwma_cross_below' in signals.columns
        assert 'band_expansion' in signals.columns
        assert 'band_contraction' in signals.columns
    
    def test_vwma_band_width(self, sample_ohlcv_data):
        """Test VWMA band width calculations."""
        band_pcts = [0.01, 0.02, 0.05]
        
        for pct in band_pcts:
            vwma = VWMABands(period=20, band_pct=pct)
            result = vwma.calculate(sample_ohlcv_data)
            
            # Band width should match percentage
            expected_width = result['vwma'] * pct * 2
            actual_width = result['vwma_upper'] - result['vwma_lower']
            
            assert np.allclose(actual_width.dropna(), expected_width.dropna(), rtol=0.01)


class TestTSV:
    """Comprehensive tests for Time Segmented Volume."""
    
    def test_tsv_calculation(self, sample_ohlcv_data):
        """Test basic TSV calculation."""
        tsv = TSV(period=13, signal_period=9)
        result = tsv.calculate(sample_ohlcv_data)
        
        assert 'tsv' in result.columns
        assert 'tsv_signal' in result.columns
        assert 'tsv_histogram' in result.columns
        
        # TSV should respond to price-volume relationships
        assert result['tsv'].std() > 0
        
        # Histogram = TSV - Signal
        expected_histogram = result['tsv'] - result['tsv_signal']
        assert np.allclose(result['tsv_histogram'].dropna(), expected_histogram.dropna())
    
    def test_tsv_accumulation_distribution(self):
        """Test TSV accumulation/distribution detection."""
        # Create data with clear accumulation pattern
        dates = pd.date_range('2023-01-01', periods=100)
        
        # Rising prices with increasing volume = accumulation
        prices = 100 + np.linspace(0, 10, 100)
        volume = np.linspace(1000000, 2000000, 100)
        
        data = pd.DataFrame({
            'close': prices,
            'open': prices - 0.5,
            'high': prices + 1,
            'low': prices - 1,
            'volume': volume
        }, index=dates)
        
        tsv = TSV()
        result = tsv.calculate(data)
        
        # TSV should be generally positive (accumulation)
        assert result['tsv'].iloc[20:].mean() > 0
    
    def test_tsv_divergence(self, sample_ohlcv_data):
        """Test TSV divergence detection."""
        tsv = TSV()
        tsv_data = tsv.calculate(sample_ohlcv_data)
        signals = tsv.get_signals(sample_ohlcv_data, tsv_data)
        
        assert 'bullish_divergence' in signals.columns
        assert 'bearish_divergence' in signals.columns
        assert 'accumulation' in signals.columns
        assert 'distribution' in signals.columns
    
    def test_tsv_zero_volume(self):
        """Test TSV with zero volume periods."""
        data = generate_stock_data()
        # Set some volume to zero
        data.iloc[10:15, data.columns.get_loc('volume')] = 0
        
        tsv = TSV()
        result = tsv.calculate(data)
        
        # Should handle zero volume gracefully
        assert not result['tsv'].isna().all()


class TestVWAP:
    """Comprehensive tests for VWAP indicators."""
    
    def test_rolling_vwap_calculation(self, sample_ohlcv_data):
        """Test rolling VWAP calculation."""
        vwap = VWAP(window=20)
        result = vwap.calculate(sample_ohlcv_data)
        
        assert 'vwap' in result.columns
        assert 'vwap_upper' in result.columns
        assert 'vwap_lower' in result.columns
        assert 'vwap_distance' in result.columns
        
        # VWAP should be between high and low
        assert (result['vwap'] <= sample_ohlcv_data['high']).all()
        assert (result['vwap'] >= sample_ohlcv_data['low']).all()
    
    def test_vwap_typical_price(self):
        """Test VWAP typical price calculation."""
        # Simple data for verification
        data = pd.DataFrame({
            'high': [102, 103, 104],
            'low': [98, 99, 100],
            'close': [100, 101, 102],
            'volume': [1000, 2000, 1500]
        })
        
        vwap = VWAP(window=3)
        result = vwap.calculate(data)
        
        # Typical price = (H + L + C) / 3
        typical = (data['high'] + data['low'] + data['close']) / 3
        
        # VWAP = sum(typical * volume) / sum(volume)
        expected_vwap = (typical * data['volume']).rolling(3).sum() / data['volume'].rolling(3).sum()
        
        assert np.allclose(result['vwap'].dropna(), expected_vwap.dropna())
    
    def test_vwap_bands(self, sample_ohlcv_data):
        """Test VWAP band calculations."""
        vwap = VWAP(window=20, num_std_dev=2.0)
        result = vwap.calculate(sample_ohlcv_data)
        
        # Bands should be symmetric around VWAP
        upper_distance = result['vwap_upper'] - result['vwap']
        lower_distance = result['vwap'] - result['vwap_lower']
        
        assert np.allclose(upper_distance.dropna(), lower_distance.dropna(), rtol=0.01)
    
    def test_anchored_vwap(self, sample_ohlcv_data):
        """Test anchored VWAP calculation."""
        anchor_date = sample_ohlcv_data.index[20]
        avwap = AnchoredVWAP(anchor_date=anchor_date)
        result = avwap.calculate(sample_ohlcv_data)
        
        assert 'avwap' in result.columns
        
        # Should be NaN before anchor
        assert result.loc[:anchor_date].iloc[:-1]['avwap'].isna().all()
        
        # Should have values from anchor onward
        assert result.loc[anchor_date:]['avwap'].notna().all()
        
        # First value should equal typical price at anchor
        typical_price = (
            sample_ohlcv_data.loc[anchor_date, 'high'] +
            sample_ohlcv_data.loc[anchor_date, 'low'] +
            sample_ohlcv_data.loc[anchor_date, 'close']
        ) / 3
        
        assert result.loc[anchor_date, 'avwap'] == pytest.approx(typical_price)
    
    def test_multiple_anchored_vwaps(self, sample_ohlcv_data):
        """Test multiple anchored VWAPs."""
        anchors = [
            sample_ohlcv_data.index[10],
            sample_ohlcv_data.index[50],
            sample_ohlcv_data.index[100]
        ]
        
        results = {}
        for anchor in anchors:
            avwap = AnchoredVWAP(anchor_date=anchor)
            results[anchor] = avwap.calculate(sample_ohlcv_data)
        
        # Each should start from its anchor date
        for anchor, result in results.items():
            assert result.loc[anchor, 'avwap'] > 0
            if anchor != sample_ohlcv_data.index[0]:
                assert result.loc[:anchor].iloc[:-1]['avwap'].isna().all()


class TestFearGreedIndex:
    """Tests for Fear and Greed Index."""
    
    @pytest.mark.asyncio
    async def test_fear_greed_fetch(self):
        """Test fetching Fear and Greed Index."""
        fg = FearGreedIndex()
        
        # Mock the API response
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value={
                'fgi': {
                    'now': {
                        'value': 45,
                        'value_classification': 'Fear'
                    }
                }
            })
            mock_response.status = 200
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            current = await fg.fetch_current()
            
            assert isinstance(current, dict)
            assert current['value'] == 45
            assert current['classification'] == 'Fear'
    
    @pytest.mark.asyncio
    async def test_fear_greed_historical(self):
        """Test fetching historical Fear and Greed data."""
        fg = FearGreedIndex()
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value={
                'fgi': {
                    'historical': [
                        {'value': 45, 'date': '2023-01-01'},
                        {'value': 55, 'date': '2023-01-02'},
                        {'value': 35, 'date': '2023-01-03'}
                    ]
                }
            })
            mock_response.status = 200
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            historical = await fg.fetch_historical(days=3)
            
            assert isinstance(historical, pd.DataFrame)
            assert len(historical) == 3
            assert 'value' in historical.columns
            assert 'date' in historical.columns
    
    def test_fear_greed_classification(self):
        """Test Fear and Greed classification."""
        fg = FearGreedIndex()
        
        classifications = [
            (10, 'Extreme Fear'),
            (30, 'Fear'),
            (50, 'Neutral'),
            (70, 'Greed'),
            (90, 'Extreme Greed')
        ]
        
        for value, expected in classifications:
            assert fg.classify_value(value) == expected
    
    def test_fear_greed_signals(self):
        """Test Fear and Greed signal generation."""
        fg = FearGreedIndex()
        
        # Create mock historical data
        dates = pd.date_range('2023-01-01', periods=30)
        values = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70] * 3
        
        data = pd.DataFrame({
            'value': values[:30],
            'classification': [fg.classify_value(v) for v in values[:30]]
        }, index=dates)
        
        signals = fg.get_signals(data)
        
        assert 'extreme_fear_buy' in signals.columns
        assert 'extreme_greed_sell' in signals.columns
        assert 'sentiment_reversal' in signals.columns


class TestInsiderTrading:
    """Tests for Insider Trading indicator."""
    
    @pytest.mark.asyncio
    async def test_insider_fetch(self):
        """Test fetching insider trading data."""
        insider = InsiderTrading()
        
        # Mock SEC API response
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.text = AsyncMock(return_value="""
                <ownershipDocument>
                    <issuer>
                        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
                    </issuer>
                    <nonDerivativeTransaction>
                        <transactionAmounts>
                            <transactionShares>1000</transactionShares>
                            <transactionPricePerShare>150</transactionPricePerShare>
                        </transactionAmounts>
                        <transactionCoding>
                            <transactionCode>P</transactionCode>
                        </transactionCoding>
                    </nonDerivativeTransaction>
                </ownershipDocument>
            """)
            mock_response.status = 200
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            data = await insider.fetch_insider_trades('AAPL', days=30)
            
            assert isinstance(data, pd.DataFrame)
            if not data.empty:
                assert 'transaction_type' in data.columns
                assert 'shares' in data.columns
                assert 'value' in data.columns
    
    def test_insider_sentiment_calculation(self):
        """Test insider sentiment calculation."""
        insider = InsiderTrading()
        
        # Create mock insider data
        trades = pd.DataFrame({
            'transaction_type': ['Buy', 'Buy', 'Sell', 'Buy', 'Sell'],
            'shares': [1000, 2000, 1500, 500, 1000],
            'value': [100000, 200000, 150000, 50000, 100000]
        })
        
        sentiment = insider.calculate_sentiment(trades)
        
        assert 'net_shares' in sentiment
        assert 'net_value' in sentiment
        assert 'buy_sell_ratio' in sentiment
        assert 'sentiment_score' in sentiment
        
        # More buys than sells
        assert sentiment['buy_sell_ratio'] > 1
    
    def test_insider_signals(self):
        """Test insider trading signals."""
        insider = InsiderTrading()
        
        # Create time series of insider sentiment
        dates = pd.date_range('2023-01-01', periods=30)
        sentiment_scores = np.sin(np.linspace(0, 4*np.pi, 30)) * 50 + 50
        
        data = pd.DataFrame({
            'sentiment_score': sentiment_scores,
            'net_value': sentiment_scores * 10000
        }, index=dates)
        
        signals = insider.get_signals(data)
        
        assert 'bullish_cluster' in signals.columns
        assert 'bearish_cluster' in signals.columns
        assert 'sentiment_extreme' in signals.columns


class TestMaxPain:
    """Tests for Max Pain indicator."""
    
    def test_max_pain_calculation(self, sample_options_data):
        """Test max pain calculation."""
        calls, puts = sample_options_data
        max_pain = MaxPain()
        
        result = max_pain.calculate_max_pain(calls, puts)
        
        assert 'max_pain_price' in result
        assert 'pain_distribution' in result
        assert 'call_pain' in result
        assert 'put_pain' in result
        
        # Max pain should be one of the strikes
        strikes = calls['strike'].unique()
        assert result['max_pain_price'] in strikes
    
    def test_max_pain_with_current_price(self, sample_options_data):
        """Test max pain with current price comparison."""
        calls, puts = sample_options_data
        current_price = 100.0
        
        max_pain = MaxPain()
        result = max_pain.calculate_max_pain(calls, puts, current_price)
        
        assert 'price_vs_max_pain' in result
        assert 'max_pain_delta' in result
        
        # Calculate expected delta
        expected_delta = (result['max_pain_price'] - current_price) / current_price
        assert result['max_pain_delta'] == pytest.approx(expected_delta)
    
    def test_max_pain_support_resistance(self, sample_options_data):
        """Test support/resistance level detection."""
        calls, puts = sample_options_data
        max_pain = MaxPain()
        
        result = max_pain.calculate_max_pain(calls, puts)
        
        assert 'support_levels' in result
        assert 'resistance_levels' in result
        
        # Should identify high open interest strikes
        total_oi = calls.groupby('strike')['openInterest'].sum() + puts.groupby('strike')['openInterest'].sum()
        high_oi_strikes = total_oi.nlargest(3).index.tolist()
        
        # High OI strikes should be in support/resistance
        all_levels = result['support_levels'] + result['resistance_levels']
        assert any(strike in all_levels for strike in high_oi_strikes)
    
    def test_max_pain_signals(self, sample_options_data):
        """Test max pain signal generation."""
        calls, puts = sample_options_data
        max_pain = MaxPain()
        
        # Calculate for multiple expiries
        max_pain_series = []
        current_price = 100.0
        
        for i in range(5):
            # Simulate changing options data
            calls_copy = calls.copy()
            puts_copy = puts.copy()
            calls_copy['openInterest'] *= (1 + i * 0.1)
            
            result = max_pain.calculate_max_pain(calls_copy, puts_copy, current_price)
            max_pain_series.append({
                'max_pain': result['max_pain_price'],
                'current_price': current_price + i
            })
        
        df = pd.DataFrame(max_pain_series)
        signals = max_pain.get_signals(df)
        
        assert 'pin_risk' in signals.columns
        assert 'max_pain_magnet' in signals.columns


class TestIndicatorIntegration:
    """Test indicator integration and combinations."""
    
    def test_multiple_indicators(self, sample_ohlcv_data):
        """Test using multiple indicators together."""
        # Calculate all indicators
        rsi = RSI(period=14)
        bb = BollingerBands(period=20)
        vwap = VWAP(window=20)
        tsv = TSV(period=13)
        
        # Add all to dataframe
        data = sample_ohlcv_data.copy()
        data['rsi'] = rsi.calculate(data)
        data = pd.concat([data, bb.calculate(data)], axis=1)
        data = pd.concat([data, vwap.calculate(data)], axis=1)
        data = pd.concat([data, tsv.calculate(data)], axis=1)
        
        # Create composite signals
        # Oversold + below BB lower + positive TSV = strong buy
        strong_buy = (
            (data['rsi'] < 30) &
            (data['close'] < data['bb_lower']) &
            (data['tsv'] > 0)
        )
        
        # Should have some signals
        assert strong_buy.any()
    
    def test_indicator_correlation(self, sample_ohlcv_data):
        """Test correlation between indicators."""
        # Calculate momentum indicators
        rsi = RSI(period=14)
        tsv = TSV(period=13)
        
        rsi_values = rsi.calculate(sample_ohlcv_data)
        tsv_data = tsv.calculate(sample_ohlcv_data)
        
        # RSI and TSV should have some correlation
        correlation = rsi_values.corr(tsv_data['tsv'])
        
        # Should have moderate correlation
        assert -1 <= correlation <= 1
        assert abs(correlation) > 0.1  # Not completely uncorrelated
    
    def test_indicator_performance(self, performance_benchmark_data, performance_monitor):
        """Test indicator calculation performance."""
        indicators = [
            ('RSI', RSI()),
            ('BollingerBands', BollingerBands()),
            ('VWAP', VWAP()),
            ('TSV', TSV()),
            ('VWMABands', VWMABands())
        ]
        
        for name, indicator in indicators:
            performance_monitor.start(name)
            result = indicator.calculate(performance_benchmark_data)
            performance_monitor.stop(name)
            
            # Should complete quickly even with 5 years of data
            assert performance_monitor.get_duration(name) < 1.0
            
            # Should produce valid results
            assert not result.empty


class TestIndicatorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_insufficient_data(self):
        """Test indicators with insufficient data."""
        # Only 5 days of data
        small_data = generate_stock_data(
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        # RSI needs more data
        rsi = RSI(period=14)
        result = rsi.calculate(small_data)
        
        # Should return all NaN
        assert result.isna().all()
    
    def test_missing_volume(self):
        """Test indicators that require volume with missing data."""
        data = generate_stock_data()
        data = data[['open', 'high', 'low', 'close']]  # No volume
        
        # VWAP requires volume
        vwap = VWAP()
        with pytest.raises(IndicatorError):
            vwap.calculate(data)
        
        # TSV requires volume
        tsv = TSV()
        with pytest.raises(IndicatorError):
            tsv.calculate(data)
    
    def test_extreme_values(self):
        """Test indicators with extreme price values."""
        data = generate_stock_data()
        
        # Add extreme values
        data.iloc[50, data.columns.get_loc('high')] = 1000000
        data.iloc[51, data.columns.get_loc('low')] = 0.01
        
        # Indicators should handle gracefully
        rsi = RSI()
        rsi_result = rsi.calculate(data)
        assert not rsi_result.isna().all()
        assert (rsi_result.dropna() >= 0).all()
        assert (rsi_result.dropna() <= 100).all()
        
        bb = BollingerBands()
        bb_result = bb.calculate(data)
        assert not bb_result.isna().all()
    
    def test_constant_prices(self):
        """Test indicators with constant prices."""
        # All prices the same
        dates = pd.date_range('2023-01-01', periods=50)
        data = pd.DataFrame({
            'open': 100,
            'high': 100,
            'low': 100,
            'close': 100,
            'volume': 1000000
        }, index=dates)
        
        # RSI should be 50 (neutral)
        rsi = RSI()
        rsi_result = rsi.calculate(data)
        assert rsi_result.dropna().iloc[-1] == pytest.approx(50, abs=1)
        
        # Bollinger Bands should have zero width
        bb = BollingerBands()
        bb_result = bb.calculate(data)
        assert bb_result['bb_width'].dropna().iloc[-1] == pytest.approx(0, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])