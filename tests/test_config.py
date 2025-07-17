"""
Test configuration and utilities for the comprehensive test suite.

This module provides common configuration, fixtures, and utilities
used across all test modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
import warnings

# Suppress specific warnings during testing
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="plotly")


class TestConfig:
    """Global test configuration."""
    
    # Data generation seeds for reproducible tests
    RANDOM_SEED = 42
    
    # Test data parameters
    DEFAULT_START_DATE = '2023-01-01'
    DEFAULT_END_DATE = '2023-12-31'
    DEFAULT_FREQUENCY = 'D'
    DEFAULT_PERIODS = 252
    
    # Portfolio parameters
    DEFAULT_INITIAL_CAPITAL = 100000
    DEFAULT_COMMISSION = 0.001
    DEFAULT_SLIPPAGE = 0.0005
    
    # Risk parameters
    DEFAULT_MAX_POSITION_SIZE = 0.05
    DEFAULT_MAX_RISK_PER_TRADE = 0.02
    DEFAULT_VAR_CONFIDENCE = 0.95
    
    # ML parameters
    DEFAULT_TRAIN_TEST_SPLIT = 0.8
    DEFAULT_VALIDATION_SPLIT = 0.2
    DEFAULT_CV_FOLDS = 5
    
    # Performance benchmarks
    MIN_SHARPE_RATIO = 0.5
    MAX_DRAWDOWN_THRESHOLD = 0.3
    MIN_WIN_RATE = 0.4
    
    # Test timeouts (seconds)
    FAST_TEST_TIMEOUT = 10
    MEDIUM_TEST_TIMEOUT = 60
    SLOW_TEST_TIMEOUT = 300


class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_price_data(
        start_date: str = TestConfig.DEFAULT_START_DATE,
        end_date: str = TestConfig.DEFAULT_END_DATE,
        frequency: str = TestConfig.DEFAULT_FREQUENCY,
        initial_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.001,
        seed: int = TestConfig.RANDOM_SEED
    ) -> pd.DataFrame:
        """Generate realistic OHLCV price data."""
        np.random.seed(seed)
        
        dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
        n_periods = len(dates)
        
        # Generate returns with trend and volatility
        returns = np.random.normal(trend, volatility, n_periods)
        
        # Calculate prices
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = pd.DataFrame(index=dates)
        
        # Close prices
        data['close'] = prices
        
        # Open prices (close of previous day + small gap)
        data['open'] = data['close'].shift(1) + np.random.normal(0, 0.001, n_periods)
        data['open'].iloc[0] = initial_price
        
        # High and low prices
        intraday_range = np.abs(np.random.normal(0, volatility * 0.5, n_periods))
        data['high'] = np.maximum(data['open'], data['close']) + intraday_range
        data['low'] = np.minimum(data['open'], data['close']) - intraday_range
        
        # Volume (realistic pattern)
        base_volume = 1000000
        volume_multiplier = 1 + np.random.normal(0, 0.5, n_periods)
        # Higher volume on high volatility days
        volatility_factor = np.abs(returns) / volatility
        data['volume'] = (base_volume * volume_multiplier * (1 + volatility_factor)).astype(int)
        
        return data
    
    @staticmethod
    def generate_multi_asset_data(
        assets: List[str] = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
        start_date: str = TestConfig.DEFAULT_START_DATE,
        end_date: str = TestConfig.DEFAULT_END_DATE,
        correlations: Optional[np.ndarray] = None,
        seed: int = TestConfig.RANDOM_SEED
    ) -> Dict[str, pd.DataFrame]:
        """Generate multi-asset price data with realistic correlations."""
        np.random.seed(seed)
        
        n_assets = len(assets)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_periods = len(dates)
        
        # Create correlation matrix if not provided
        if correlations is None:
            correlations = np.random.uniform(0.3, 0.7, (n_assets, n_assets))
            correlations = (correlations + correlations.T) / 2
            np.fill_diagonal(correlations, 1.0)
        
        # Generate correlated returns
        mean_returns = np.random.uniform(0.0005, 0.0015, n_assets)
        volatilities = np.random.uniform(0.015, 0.035, n_assets)
        
        # Create covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlations
        
        # Generate correlated returns
        returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_periods)
        
        # Create price data for each asset
        asset_data = {}
        for i, asset in enumerate(assets):
            initial_price = np.random.uniform(50, 200)
            prices = initial_price * np.exp(np.cumsum(returns[:, i]))
            
            data = pd.DataFrame(index=dates)
            data['close'] = prices
            data['open'] = data['close'].shift(1)
            data['open'].iloc[0] = initial_price
            
            intraday_range = np.abs(np.random.normal(0, volatilities[i] * 0.5, n_periods))
            data['high'] = np.maximum(data['open'], data['close']) + intraday_range
            data['low'] = np.minimum(data['open'], data['close']) - intraday_range
            
            data['volume'] = np.random.randint(1000000, 10000000, n_periods)
            
            asset_data[asset] = data
        
        return asset_data
    
    @staticmethod
    def generate_trade_data(
        n_trades: int = 100,
        start_date: str = TestConfig.DEFAULT_START_DATE,
        end_date: str = TestConfig.DEFAULT_END_DATE,
        win_rate: float = 0.55,
        avg_win: float = 0.05,
        avg_loss: float = -0.03,
        seed: int = TestConfig.RANDOM_SEED
    ) -> pd.DataFrame:
        """Generate realistic trade data."""
        np.random.seed(seed)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate trade entry/exit times
        entry_times = np.random.choice(dates[:-5], n_trades, replace=True)
        exit_times = []
        
        for entry_time in entry_times:
            # Random holding period (1-5 days)
            holding_period = np.random.randint(1, 6)
            exit_time = entry_time + timedelta(days=holding_period)
            if exit_time > dates[-1]:
                exit_time = dates[-1]
            exit_times.append(exit_time)
        
        # Generate P&L based on win rate
        is_winner = np.random.random(n_trades) < win_rate
        pnl = np.where(
            is_winner,
            np.random.normal(avg_win, avg_win * 0.5, n_trades),
            np.random.normal(avg_loss, abs(avg_loss) * 0.5, n_trades)
        )
        
        # Generate other trade attributes
        sides = np.random.choice(['long', 'short'], n_trades)
        sizes = np.random.uniform(100, 1000, n_trades)
        entry_prices = np.random.uniform(50, 200, n_trades)
        exit_prices = entry_prices * (1 + pnl)
        
        trades = pd.DataFrame({
            'trade_id': range(1, n_trades + 1),
            'entry_time': entry_times,
            'exit_time': exit_times,
            'side': sides,
            'size': sizes,
            'entry_price': entry_prices,
            'exit_price': exit_prices,
            'pnl': pnl * sizes * entry_prices,
            'pnl_pct': pnl,
            'duration': [(exit - entry).total_seconds() / 3600 for entry, exit in zip(entry_times, exit_times)]
        })
        
        return trades
    
    @staticmethod
    def generate_portfolio_data(
        assets: List[str] = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
        weights: Optional[List[float]] = None,
        start_date: str = TestConfig.DEFAULT_START_DATE,
        end_date: str = TestConfig.DEFAULT_END_DATE,
        seed: int = TestConfig.RANDOM_SEED
    ) -> Dict[str, Any]:
        """Generate portfolio data including returns, weights, and metrics."""
        np.random.seed(seed)
        
        if weights is None:
            weights = np.random.dirichlet(np.ones(len(assets)))
        
        # Generate asset data
        asset_data = TestDataGenerator.generate_multi_asset_data(
            assets, start_date, end_date, seed=seed
        )
        
        # Calculate portfolio returns
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        portfolio_returns = pd.Series(0.0, index=dates)
        
        for i, asset in enumerate(assets):
            asset_returns = asset_data[asset]['close'].pct_change().fillna(0)
            portfolio_returns += weights[i] * asset_returns
        
        # Calculate portfolio value
        portfolio_value = TestConfig.DEFAULT_INITIAL_CAPITAL * (1 + portfolio_returns).cumprod()
        
        # Calculate basic metrics
        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (portfolio_returns.mean() * 252) / volatility
        max_drawdown = ((portfolio_value / portfolio_value.cummax()) - 1).min()
        
        return {
            'asset_data': asset_data,
            'weights': dict(zip(assets, weights)),
            'portfolio_returns': portfolio_returns,
            'portfolio_value': portfolio_value,
            'metrics': {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
        }


class TestFixtures:
    """Common test fixtures and utilities."""
    
    @staticmethod
    def create_temp_directory():
        """Create a temporary directory for test files."""
        return tempfile.mkdtemp()
    
    @staticmethod
    def create_mock_strategy():
        """Create a mock trading strategy for testing."""
        strategy = Mock()
        strategy.generate_signals.return_value = pd.DataFrame({
            'long_signal': [True, False, True, False, True],
            'short_signal': [False, True, False, True, False],
            'signal_strength': [0.8, 0.6, 0.9, 0.7, 0.75]
        })
        return strategy
    
    @staticmethod
    def create_mock_backtest_engine():
        """Create a mock backtesting engine for testing."""
        engine = Mock()
        engine.initial_capital = TestConfig.DEFAULT_INITIAL_CAPITAL
        engine.commission = TestConfig.DEFAULT_COMMISSION
        engine.slippage = TestConfig.DEFAULT_SLIPPAGE
        
        # Mock run method
        engine.run.return_value = {
            'equity_curve': pd.Series([100000, 101000, 102000, 101500, 103000]),
            'trades': TestDataGenerator.generate_trade_data(10),
            'metrics': {
                'total_return': 0.03,
                'sharpe_ratio': 0.8,
                'max_drawdown': -0.05,
                'win_rate': 0.6
            }
        }
        
        return engine
    
    @staticmethod
    def create_mock_ml_model():
        """Create a mock ML model for testing."""
        model = Mock()
        model.is_trained = True
        model.predict.return_value = pd.DataFrame({
            'prediction': [1, -1, 1, -1, 1],
            'probability': [0.8, 0.7, 0.9, 0.6, 0.85],
            'confidence': [0.6, 0.5, 0.8, 0.4, 0.7]
        })
        model.train.return_value = {'accuracy': 0.75, 'loss': 0.3}
        return model


class TestValidators:
    """Validation utilities for test assertions."""
    
    @staticmethod
    def validate_ohlcv_data(data: pd.DataFrame) -> bool:
        """Validate OHLCV data structure and relationships."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check required columns
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check OHLC relationships
        if not (data['high'] >= data[['open', 'close']].max(axis=1)).all():
            return False
        
        if not (data['low'] <= data[['open', 'close']].min(axis=1)).all():
            return False
        
        # Check for positive volume
        if not (data['volume'] > 0).all():
            return False
        
        return True
    
    @staticmethod
    def validate_portfolio_weights(weights: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Validate portfolio weights sum to 1 and are non-negative."""
        return (
            abs(weights.sum() - 1.0) < tolerance and
            (weights >= 0).all()
        )
    
    @staticmethod
    def validate_trade_data(trades: pd.DataFrame) -> bool:
        """Validate trade data structure and consistency."""
        required_columns = ['entry_time', 'exit_time', 'side', 'pnl']
        
        # Check required columns
        if not all(col in trades.columns for col in required_columns):
            return False
        
        # Check time relationships
        if not (trades['exit_time'] >= trades['entry_time']).all():
            return False
        
        # Check side values
        if not trades['side'].isin(['long', 'short']).all():
            return False
        
        return True
    
    @staticmethod
    def validate_performance_metrics(metrics: Dict[str, float]) -> bool:
        """Validate performance metrics are within reasonable ranges."""
        checks = [
            # Sharpe ratio should be reasonable
            -5 <= metrics.get('sharpe_ratio', 0) <= 10,
            # Max drawdown should be negative or zero
            metrics.get('max_drawdown', 0) <= 0,
            # Win rate should be between 0 and 1
            0 <= metrics.get('win_rate', 0.5) <= 1,
            # Volatility should be positive
            metrics.get('volatility', 0.1) > 0
        ]
        
        return all(checks)


class TestDecorators:
    """Test decorators for common functionality."""
    
    @staticmethod
    def timeout(seconds: int):
        """Decorator to add timeout to test functions."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Test timed out after {seconds} seconds")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
                
                try:
                    result = func(*args, **kwargs)
                    signal.alarm(0)
                    return result
                except TimeoutError:
                    signal.alarm(0)
                    raise
            
            return wrapper
        return decorator
    
    @staticmethod
    def skip_if_slow(reason: str = "Slow test"):
        """Decorator to skip slow tests unless explicitly enabled."""
        def decorator(func):
            return pytest.mark.skipif(
                not os.getenv('RUN_SLOW_TESTS', False),
                reason=reason
            )(func)
        return decorator
    
    @staticmethod
    def requires_data(data_type: str):
        """Decorator to skip tests if required data is not available."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Check if required data exists
                # This is a placeholder - implement actual data checks
                return func(*args, **kwargs)
            return wrapper
        return decorator


# Global test configuration instance
test_config = TestConfig()
test_data_generator = TestDataGenerator()
test_fixtures = TestFixtures()
test_validators = TestValidators()
test_decorators = TestDecorators()

# Common pytest fixtures
@pytest.fixture
def sample_ohlcv_data():
    """Fixture providing sample OHLCV data."""
    return test_data_generator.generate_price_data()

@pytest.fixture
def sample_multi_asset_data():
    """Fixture providing multi-asset data."""
    return test_data_generator.generate_multi_asset_data()

@pytest.fixture
def sample_trade_data():
    """Fixture providing sample trade data."""
    return test_data_generator.generate_trade_data()

@pytest.fixture
def sample_portfolio_data():
    """Fixture providing sample portfolio data."""
    return test_data_generator.generate_portfolio_data()

@pytest.fixture
def temp_directory():
    """Fixture providing temporary directory."""
    temp_dir = test_fixtures.create_temp_directory()
    yield temp_dir
    # Cleanup is handled by tempfile

@pytest.fixture
def mock_strategy():
    """Fixture providing mock trading strategy."""
    return test_fixtures.create_mock_strategy()

@pytest.fixture
def mock_backtest_engine():
    """Fixture providing mock backtesting engine."""
    return test_fixtures.create_mock_backtest_engine()

@pytest.fixture
def mock_ml_model():
    """Fixture providing mock ML model."""
    return test_fixtures.create_mock_ml_model()