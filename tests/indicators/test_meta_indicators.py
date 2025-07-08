"""
Tests for meta indicators: Fear & Greed, Insider Trading, and Max Pain.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from src.indicators.fear_greed import FearGreedIndex
from src.indicators.insider import InsiderTrading
from src.indicators.max_pain import MaxPain


class TestFearGreedIndex:
    """Test Fear and Greed Index indicator."""
    
    @pytest.fixture
    def fg_index(self):
        """Create FearGreedIndex instance."""
        return FearGreedIndex(source="alternative")
    
    @pytest.fixture
    def mock_api_response(self):
        """Mock API response data."""
        return {
            "data": [
                {
                    "value": "45",
                    "value_classification": "Fear",
                    "timestamp": "1234567890",
                    "time_until_update": "3600"
                },
                {
                    "value": "65",
                    "value_classification": "Greed",
                    "timestamp": "1234481490",
                },
                {
                    "value": "25",
                    "value_classification": "Extreme Fear",
                    "timestamp": "1234395090",
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_fetch_current(self, fg_index, mock_api_response):
        """Test fetching current Fear & Greed value."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value=mock_api_response)
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            result = await fg_index.fetch_current()
            
            assert result['value'] == 45
            assert result['classification'] == 'Fear'
            assert isinstance(result['timestamp'], datetime)
    
    @pytest.mark.asyncio
    async def test_fetch_historical(self, fg_index, mock_api_response):
        """Test fetching historical Fear & Greed data."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value=mock_api_response)
            mock_get.return_value.__aenter__.return_value = mock_resp
            
            result = await fg_index.fetch_historical(limit=30)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert 'value' in result.columns
            assert 'classification' in result.columns
            assert result['value'].mean() == 45  # (45 + 65 + 25) / 3
    
    def test_classify_value(self, fg_index):
        """Test value classification."""
        assert fg_index._classify_value(10) == "Extreme Fear"
        assert fg_index._classify_value(30) == "Fear"
        assert fg_index._classify_value(50) == "Neutral"
        assert fg_index._classify_value(70) == "Greed"
        assert fg_index._classify_value(85) == "Extreme Greed"
    
    def test_get_signals(self, fg_index):
        """Test signal generation."""
        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'value': [20, 25, 30, 45, 50, 60, 75, 80, 75, 70],
            'classification': ['Extreme Fear'] * 3 + ['Fear', 'Neutral', 'Greed'] * 2 + ['Greed']
        }, index=dates)
        
        signals = fg_index.get_signals(data)
        
        assert 'extreme_fear' in signals.columns
        assert 'extreme_greed' in signals.columns
        assert 'entering_fear' in signals.columns
        assert 'entering_greed' in signals.columns
        assert signals['extreme_fear'].iloc[0] == True  # value=20
        assert signals['extreme_greed'].iloc[7] == True  # value=80


class TestInsiderTrading:
    """Test Insider Trading indicator."""
    
    @pytest.fixture
    def mock_html_response(self):
        """Mock HTML response from OpenInsider."""
        return """
        <html>
        <body>
            <table class="tinytable">
                <thead>
                    <tr>
                        <th>Filing Date</th>
                        <th>Company</th>
                        <th>Insider</th>
                        <th>Title</th>
                        <th>Type</th>
                        <th>Shares</th>
                        <th>Price</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>2024-01-15</td>
                        <td>AAPL Apple Inc</td>
                        <td>John Doe</td>
                        <td>CEO</td>
                        <td>P</td>
                        <td>10,000</td>
                        <td>$150.00</td>
                        <td>$1.5M</td>
                    </tr>
                    <tr>
                        <td>2024-01-14</td>
                        <td>MSFT Microsoft Corp</td>
                        <td>Jane Smith</td>
                        <td>CFO</td>
                        <td>S</td>
                        <td>5,000</td>
                        <td>$300.00</td>
                        <td>$1.5M</td>
                    </tr>
                </tbody>
            </table>
        </body>
        </html>
        """
    
    @pytest.mark.asyncio
    async def test_fetch_latest_trades(self, mock_html_response):
        """Test fetching latest insider trades."""
        async with InsiderTrading() as insider:
            with patch('aiohttp.ClientSession.get') as mock_get:
                mock_resp = AsyncMock()
                mock_resp.text = AsyncMock(return_value=mock_html_response)
                mock_get.return_value.__aenter__.return_value = mock_resp
                
                result = await insider.fetch_latest_trades(ticker="AAPL", limit=10)
                
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 2
                assert 'ticker' in result.columns
                assert result.iloc[0]['ticker'] == 'AAPL'
    
    def test_parse_numeric(self):
        """Test numeric parsing."""
        insider = InsiderTrading()
        
        assert insider._parse_numeric('1,000') == 1000.0
        assert insider._parse_numeric('$1.5M') == 1500000.0
        assert insider._parse_numeric('2.5K') == 2500.0
        assert insider._parse_numeric('1B') == 1000000000.0
        assert insider._parse_numeric('') == 0.0
    
    def test_extract_ticker(self):
        """Test ticker extraction."""
        insider = InsiderTrading()
        
        assert insider._extract_ticker('AAPL Apple Inc') == 'AAPL'
        assert insider._extract_ticker('MSFT Microsoft Corp') == 'MSFT'
        assert insider._extract_ticker({'text': 'TSLA Tesla Inc'}) == 'TSLA'
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis."""
        insider = InsiderTrading()
        
        # Create test data
        data = pd.DataFrame({
            'transaction_type': ['Purchase', 'Purchase', 'Sale', 'Purchase'],
            'value': [1000000, 500000, 2000000, 300000],
            'ticker': ['AAPL', 'AAPL', 'MSFT', 'TSLA'],
            'insider': ['CEO', 'CFO', 'Director', 'VP'],
            'title': ['CEO', 'CFO', 'Director', 'VP']
        })
        
        sentiment = insider.analyze_sentiment(data)
        
        assert sentiment['total_trades'] == 4
        assert sentiment['buy_count'] == 3
        assert sentiment['sell_count'] == 1
        assert sentiment['buy_sell_ratio'] == 3.0
        assert sentiment['buy_value'] == 1800000
        assert sentiment['sell_value'] == 2000000


class TestMaxPain:
    """Test Max Pain calculator."""
    
    @pytest.fixture
    def max_pain_calc(self):
        """Create MaxPain instance."""
        return MaxPain()
    
    @pytest.fixture
    def mock_options_data(self):
        """Mock options chain data."""
        strikes = [145, 150, 155, 160, 165]
        
        calls = pd.DataFrame({
            'strike': strikes,
            'openInterest': [1000, 2000, 3000, 2000, 1000],
            'lastPrice': [7.5, 5.0, 2.5, 1.0, 0.5],
            'gamma': [0.05, 0.08, 0.10, 0.07, 0.03]
        })
        
        puts = pd.DataFrame({
            'strike': strikes,
            'openInterest': [1000, 2000, 3000, 2000, 1000],
            'lastPrice': [0.5, 1.0, 2.5, 5.0, 7.5],
            'gamma': [0.03, 0.07, 0.10, 0.08, 0.05]
        })
        
        return calls, puts
    
    def test_calculate_max_pain(self, max_pain_calc, mock_options_data):
        """Test max pain calculation."""
        calls, puts = mock_options_data
        
        result = max_pain_calc._calculate_max_pain(calls, puts)
        
        assert 'max_pain_price' in result
        assert 'pain_distribution' in result
        assert 'resistance_levels' in result
        assert 'support_levels' in result
        assert 'put_call_ratio' in result
        assert result['put_call_ratio'] == 1.0  # Equal OI
    
    def test_calculate_call_pain(self, max_pain_calc, mock_options_data):
        """Test call pain calculation."""
        calls, _ = mock_options_data
        
        # At strike 155, calls at 145 and 150 are ITM
        pain = max_pain_calc._calculate_call_pain(calls, 155)
        
        # Pain = (155-145)*1000*100 + (155-150)*2000*100
        expected = 10*1000*100 + 5*2000*100
        assert pain == expected
    
    def test_calculate_put_pain(self, max_pain_calc, mock_options_data):
        """Test put pain calculation."""
        _, puts = mock_options_data
        
        # At strike 155, puts at 160 and 165 are ITM
        pain = max_pain_calc._calculate_put_pain(puts, 155)
        
        # Pain = (160-155)*2000*100 + (165-155)*1000*100
        expected = 5*2000*100 + 10*1000*100
        assert pain == expected
    
    def test_calculate_gamma_levels(self, max_pain_calc, mock_options_data):
        """Test gamma level calculation."""
        calls, puts = mock_options_data
        
        gamma_levels = max_pain_calc._calculate_gamma_levels(calls, puts)
        
        assert isinstance(gamma_levels, list)
        assert len(gamma_levels) <= 5
        if gamma_levels:
            assert 'strike' in gamma_levels[0]
            assert 'gamma_exposure' in gamma_levels[0]
            assert 'significance' in gamma_levels[0]
    
    def test_get_signals(self, max_pain_calc):
        """Test signal generation."""
        max_pain_data = {
            'max_pain_price': 150,
            'resistance_levels': [155, 160, 165],
            'support_levels': [145, 140, 135],
            'gamma_levels': [
                {'strike': 150, 'significance': 'high'},
                {'strike': 155, 'significance': 'medium'}
            ]
        }
        
        # Test with price near max pain
        signals = max_pain_calc.get_signals('AAPL', 151, max_pain_data)
        assert signals['max_pain_magnet'] == True
        assert signals['above_max_pain'] == True
        
        # Test with price far from max pain
        signals = max_pain_calc.get_signals('AAPL', 165, max_pain_data)
        assert signals['extreme_deviation'] == True
        assert signals['bullish_deviation'] == True