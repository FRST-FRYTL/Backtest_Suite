"""
Integration tests for meta indicators with real data sources.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.indicators.fear_greed import FearGreedIndex
from src.indicators.insider import InsiderTrading
from src.indicators.max_pain import MaxPain
from src.backtesting.strategy import Strategy
from src.backtesting.engine import BacktestEngine
import pandas as pd


class MetaIndicatorStrategy(Strategy):
    """Example strategy using meta indicators."""
    
    def __init__(self, fear_greed_threshold=30, max_pain_deviation=5):
        super().__init__("MetaIndicatorStrategy")
        self.fear_greed_threshold = fear_greed_threshold
        self.max_pain_deviation = max_pain_deviation
        self.fg_index = FearGreedIndex()
        self.max_pain_calc = MaxPain()
        
    async def setup(self):
        """Setup indicators."""
        # Fetch latest Fear & Greed
        self.current_fg = await self.fg_index.fetch_current()
        
    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze data with meta indicators."""
        signals = pd.DataFrame(index=data.index)
        
        # Simple strategy: Buy when fear is extreme
        if self.current_fg['value'] < self.fear_greed_threshold:
            signals['buy'] = 1
        else:
            signals['buy'] = 0
            
        signals['sell'] = 0  # Hold positions
        
        return signals


@pytest.mark.integration
class TestMetaIndicatorsIntegration:
    """Integration tests for meta indicators."""
    
    @pytest.mark.asyncio
    async def test_fear_greed_with_strategy(self):
        """Test Fear & Greed integration with backtesting."""
        # Mock the API response to avoid external dependencies
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = MagicMock()
            mock_resp.json = asyncio.coroutine(lambda: {
                "data": [{
                    "value": "25",
                    "value_classification": "Extreme Fear",
                    "timestamp": str(int(datetime.now().timestamp()))
                }]
            })
            mock_resp.__aenter__ = asyncio.coroutine(lambda self: mock_resp)
            mock_resp.__aexit__ = asyncio.coroutine(lambda self, *args: None)
            mock_get.return_value = mock_resp
            
            # Create strategy
            strategy = MetaIndicatorStrategy(fear_greed_threshold=30)
            await strategy.setup()
            
            # Verify Fear & Greed was fetched
            assert strategy.current_fg['value'] == 25
            assert strategy.current_fg['classification'] == "Extreme Fear"
    
    @pytest.mark.asyncio
    async def test_insider_trading_filtering(self):
        """Test insider trading data filtering and analysis."""
        # Mock HTML response
        mock_html = """
        <html><body><table class="tinytable">
        <thead><tr>
            <th>Filing Date</th><th>Company</th><th>Insider</th>
            <th>Title</th><th>Type</th><th>Value</th>
        </tr></thead>
        <tbody>
            <tr>
                <td>2024-01-15</td><td>AAPL Apple Inc</td><td>Tim Cook</td>
                <td>CEO</td><td>P</td><td>$5M</td>
            </tr>
            <tr>
                <td>2024-01-14</td><td>AAPL Apple Inc</td><td>John Doe</td>
                <td>CFO</td><td>P</td><td>$2M</td>
            </tr>
            <tr>
                <td>2024-01-13</td><td>MSFT Microsoft</td><td>Jane Smith</td>
                <td>CEO</td><td>S</td><td>$3M</td>
            </tr>
        </tbody>
        </table></body></html>
        """
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_resp = MagicMock()
            mock_resp.text = asyncio.coroutine(lambda: mock_html)
            mock_resp.__aenter__ = asyncio.coroutine(lambda self: mock_resp)
            mock_resp.__aexit__ = asyncio.coroutine(lambda self, *args: None)
            mock_get.return_value = mock_resp
            
            async with InsiderTrading() as insider:
                # Test ticker filtering
                aapl_trades = await insider.fetch_latest_trades(ticker="AAPL")
                
                # Should only have AAPL trades
                assert len(aapl_trades) == 2
                assert all(aapl_trades['ticker'] == 'AAPL')
                
                # Test sentiment analysis
                sentiment = insider.analyze_sentiment(aapl_trades)
                assert sentiment['buy_count'] == 2
                assert sentiment['sell_count'] == 0
                assert sentiment['bullish_score'] > 50
    
    def test_max_pain_with_mock_options(self):
        """Test max pain calculation with mock options data."""
        with patch.object(MaxPain, 'fetcher') as mock_fetcher:
            # Mock options chain
            strikes = [95, 100, 105, 110]
            calls = pd.DataFrame({
                'strike': strikes,
                'openInterest': [100, 500, 800, 200],
                'lastPrice': [6.0, 3.0, 1.0, 0.5]
            })
            puts = pd.DataFrame({
                'strike': strikes,
                'openInterest': [200, 800, 500, 100],
                'lastPrice': [0.5, 1.0, 3.0, 6.0]
            })
            
            mock_fetcher.get_options_chain.return_value = (calls, puts)
            mock_fetcher.get_info.return_value = {'expirationDates': ['2024-01-19']}
            
            max_pain_calc = MaxPain()
            result = max_pain_calc.calculate("TEST")
            
            assert result['max_pain_price'] is not None
            assert 95 <= result['max_pain_price'] <= 110
            assert 'resistance_levels' in result
            assert 'support_levels' in result
    
    @pytest.mark.asyncio
    async def test_combined_indicators_signal_generation(self):
        """Test combining multiple meta indicators for signals."""
        # Mock all external calls
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Mock Fear & Greed response
            fg_resp = MagicMock()
            fg_resp.json = asyncio.coroutine(lambda: {
                "data": [{
                    "value": "20",  # Extreme fear
                    "value_classification": "Extreme Fear",
                    "timestamp": str(int(datetime.now().timestamp()))
                }]
            })
            
            # Mock insider trading response
            insider_resp = MagicMock()
            insider_resp.text = asyncio.coroutine(lambda: """
                <html><body><table class="tinytable">
                <thead><tr><th>Filing Date</th><th>Company</th><th>Type</th><th>Value</th></tr></thead>
                <tbody>
                    <tr><td>2024-01-15</td><td>SPY SPDR</td><td>P</td><td>$10M</td></tr>
                </tbody>
                </table></body></html>
            """)
            
            # Setup response routing
            async def get_mock(*args, **kwargs):
                url = args[0] if args else kwargs.get('url', '')
                if 'alternative.me' in url:
                    return fg_resp
                else:
                    return insider_resp
            
            mock_get.side_effect = lambda *args, **kwargs: MagicMock(
                __aenter__=asyncio.coroutine(lambda self: get_mock(*args, **kwargs)),
                __aexit__=asyncio.coroutine(lambda self, *args: None)
            )
            
            # Initialize indicators
            fg_index = FearGreedIndex()
            
            # Get current fear & greed
            fg_data = await fg_index.fetch_current()
            
            # Generate combined signal
            bullish_signals = []
            
            # Fear & Greed signal
            if fg_data['value'] < 25:
                bullish_signals.append("extreme_fear")
            
            # Verify we have bullish signals
            assert len(bullish_signals) > 0
            assert "extreme_fear" in bullish_signals