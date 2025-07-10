"""Pytest configuration and shared fixtures for test suite."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import asyncio

from src.backtesting import BacktestEngine, Portfolio
from src.strategies import StrategyBuilder
from src.data import StockDataFetcher
from src.indicators import RSI, BollingerBands, VWAP


# Test data generators
def generate_stock_data(
    symbol: str = "TEST",
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    initial_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    seed: int = 42
) -> pd.DataFrame:
    """Generate realistic stock price data for testing."""
    np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Generate returns with trend and volatility
    returns = np.random.normal(trend, volatility, n_days)
    
    # Calculate prices
    price_series = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame(index=dates)
    data['close'] = price_series
    
    # Generate open prices (close of previous day with small gap)
    data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, 0.001, n_days))
    data['open'].iloc[0] = initial_price
    
    # Generate high/low with realistic intraday movements
    daily_range = np.abs(np.random.normal(0.01, 0.005, n_days))
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + daily_range)
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - daily_range)
    
    # Generate volume (higher volume on larger price movements)
    base_volume = 1000000
    price_change = np.abs(data['close'].pct_change()).fillna(0)
    data['volume'] = base_volume * (1 + price_change * 50) * np.random.uniform(0.8, 1.2, n_days)
    data['volume'] = data['volume'].astype(int)
    
    return data


def generate_options_chain(
    spot_price: float = 100.0,
    strikes: list = None,
    expiry_days: int = 30,
    seed: int = 42
) -> tuple:
    """Generate mock options chain data for testing."""
    np.random.seed(seed)
    
    if strikes is None:
        strikes = list(range(int(spot_price * 0.8), int(spot_price * 1.2), 5))
    
    # Generate call options
    calls = []
    for strike in strikes:
        moneyness = spot_price / strike
        # Higher open interest near the money
        oi_factor = np.exp(-abs(moneyness - 1) * 10)
        open_interest = int(10000 * oi_factor * np.random.uniform(0.5, 1.5))
        
        # Simple option pricing
        intrinsic = max(0, spot_price - strike)
        time_value = (expiry_days / 365) * 5 * np.exp(-abs(moneyness - 1) * 2)
        price = intrinsic + time_value + np.random.uniform(-0.1, 0.1)
        
        calls.append({
            'strike': strike,
            'openInterest': open_interest,
            'lastPrice': max(0.01, price),
            'volume': int(open_interest * 0.1 * np.random.uniform(0.5, 1.5)),
            'impliedVolatility': 0.2 + np.random.uniform(-0.05, 0.05)
        })
    
    # Generate put options
    puts = []
    for strike in strikes:
        moneyness = strike / spot_price
        # Higher open interest near the money
        oi_factor = np.exp(-abs(moneyness - 1) * 10)
        open_interest = int(10000 * oi_factor * np.random.uniform(0.5, 1.5))
        
        # Simple option pricing
        intrinsic = max(0, strike - spot_price)
        time_value = (expiry_days / 365) * 5 * np.exp(-abs(moneyness - 1) * 2)
        price = intrinsic + time_value + np.random.uniform(-0.1, 0.1)
        
        puts.append({
            'strike': strike,
            'openInterest': open_interest,
            'lastPrice': max(0.01, price),
            'volume': int(open_interest * 0.1 * np.random.uniform(0.5, 1.5)),
            'impliedVolatility': 0.2 + np.random.uniform(-0.05, 0.05)
        })
    
    return pd.DataFrame(calls), pd.DataFrame(puts)


# Fixtures
@pytest.fixture
def sample_ohlcv_data():
    """Provide sample OHLCV data for testing."""
    return generate_stock_data()


@pytest.fixture
def multi_symbol_data():
    """Provide data for multiple symbols."""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    data = {}
    for i, symbol in enumerate(symbols):
        data[symbol] = generate_stock_data(
            symbol=symbol,
            initial_price=100 + i * 50,
            volatility=0.02 + i * 0.005,
            seed=42 + i
        )
    return data


@pytest.fixture
def sample_options_data():
    """Provide sample options chain data."""
    return generate_options_chain()


@pytest.fixture
def temp_cache_dir():
    """Provide a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_data_fetcher():
    """Provide a mock data fetcher."""
    fetcher = Mock(spec=StockDataFetcher)
    fetcher.fetch_stock_data = AsyncMock(return_value=generate_stock_data())
    fetcher.fetch_options_chain = AsyncMock(return_value=generate_options_chain())
    return fetcher


@pytest.fixture
def sample_portfolio():
    """Provide a sample portfolio for testing."""
    return Portfolio(initial_capital=100000, commission_rate=0.001)


@pytest.fixture
def sample_strategy():
    """Provide a sample trading strategy."""
    builder = StrategyBuilder("Test Strategy")
    
    # Add some basic rules
    builder.add_entry_rule("rsi < 30")
    builder.add_entry_rule("close > vwap")
    builder.add_exit_rule("rsi > 70")
    builder.add_exit_rule("close < vwap * 0.98")
    
    # Risk management
    builder.set_risk_management(
        stop_loss=0.05,
        take_profit=0.10,
        max_positions=3,
        position_size=0.1
    )
    
    return builder.build()


@pytest.fixture
def backtest_engine():
    """Provide a configured backtest engine."""
    return BacktestEngine(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005
    )


@pytest.fixture
def cache_manager(temp_cache_dir):
    """Provide a cache manager with temporary directory."""
    # CacheManager removed - using temp directory directly
    return temp_cache_dir


@pytest.fixture
async def async_event_loop():
    """Provide an async event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_yfinance_data():
    """Mock yfinance responses for testing."""
    mock_ticker = Mock()
    mock_ticker.history.return_value = generate_stock_data()
    mock_ticker.info = {
        'symbol': 'TEST',
        'longName': 'Test Company',
        'sector': 'Technology',
        'marketCap': 1000000000,
        'regularMarketPrice': 100.0
    }
    return mock_ticker


@pytest.fixture
def performance_benchmark_data():
    """Large dataset for performance testing."""
    # 5 years of data
    return generate_stock_data(
        start_date="2019-01-01",
        end_date="2023-12-31",
        volatility=0.025
    )


# Test helpers
class TestHelpers:
    """Helper functions for tests."""
    
    @staticmethod
    def assert_dataframe_structure(df: pd.DataFrame, required_columns: list):
        """Assert that a DataFrame has required columns."""
        assert isinstance(df, pd.DataFrame)
        assert all(col in df.columns for col in required_columns)
    
    @staticmethod
    def assert_series_bounded(series: pd.Series, lower: float, upper: float):
        """Assert that a Series values are within bounds."""
        assert series.min() >= lower
        assert series.max() <= upper
    
    @staticmethod
    def assert_monotonic_increasing(series: pd.Series):
        """Assert that a Series is monotonically increasing."""
        assert series.is_monotonic_increasing
    
    @staticmethod
    def create_test_signals(data: pd.DataFrame, entry_pct: float = 0.1) -> pd.DataFrame:
        """Create random test signals for backtesting."""
        n_signals = int(len(data) * entry_pct)
        signal_indices = np.random.choice(data.index, n_signals, replace=False)
        
        signals = pd.DataFrame(index=data.index)
        signals['entry'] = False
        signals.loc[signal_indices, 'entry'] = True
        
        # Add exit signals after entries
        signals['exit'] = False
        for idx in signal_indices:
            exit_idx = data.index.get_loc(idx) + np.random.randint(5, 20)
            if exit_idx < len(data):
                signals.iloc[exit_idx, signals.columns.get_loc('exit')] = True
        
        return signals


@pytest.fixture
def test_helpers():
    """Provide test helper functions."""
    return TestHelpers()


# Async helpers
@pytest.fixture
def async_mock_factory():
    """Factory for creating async mocks."""
    def create_async_mock(return_value=None, side_effect=None):
        mock = AsyncMock()
        if return_value is not None:
            mock.return_value = return_value
        if side_effect is not None:
            mock.side_effect = side_effect
        return mock
    return create_async_mock


# Performance monitoring
@pytest.fixture
def performance_monitor():
    """Monitor test performance."""
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.timings = {}
        
        def start(self, name: str):
            self.timings[name] = {'start': time.time()}
        
        def stop(self, name: str):
            if name in self.timings:
                self.timings[name]['end'] = time.time()
                self.timings[name]['duration'] = (
                    self.timings[name]['end'] - self.timings[name]['start']
                )
        
        def get_duration(self, name: str) -> float:
            return self.timings.get(name, {}).get('duration', 0)
        
        def report(self):
            for name, timing in self.timings.items():
                if 'duration' in timing:
                    print(f"{name}: {timing['duration']:.3f}s")
    
    return PerformanceMonitor()


# Database fixtures
@pytest.fixture
def test_database_url():
    """Provide a test database URL."""
    return "sqlite:///:memory:"


@pytest.fixture
def mock_redis_client():
    """Provide a mock Redis client."""
    client = Mock()
    client.get = Mock(return_value=None)
    client.set = Mock(return_value=True)
    client.delete = Mock(return_value=True)
    client.exists = Mock(return_value=False)
    return client