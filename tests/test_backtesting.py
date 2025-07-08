"""Tests for backtesting engine and components."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtesting import (
    BacktestEngine, Portfolio, Position, Order, OrderType, OrderStatus,
    MarketEvent, SignalEvent, OrderEvent, FillEvent
)
from src.strategies import StrategyBuilder, Rule
from src.utils import PerformanceMetrics


@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
    np.random.seed(42)
    
    # Generate trending data
    trend = np.linspace(100, 120, len(dates))
    noise = np.random.randn(len(dates)) * 2
    close_prices = trend + noise
    
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(len(dates)) * 0.5,
        'high': close_prices + np.abs(np.random.randn(len(dates))) * 1.5,
        'low': close_prices - np.abs(np.random.randn(len(dates))) * 1.5,
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Fix high/low
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    # Add indicators for strategy testing
    data['rsi'] = 50 + np.random.randn(len(dates)) * 20  # Simplified RSI
    data['rsi'] = data['rsi'].clip(0, 100)
    
    return data


@pytest.fixture
def simple_strategy():
    """Create a simple test strategy."""
    builder = StrategyBuilder("Test Strategy")
    
    # Simple RSI strategy
    builder.add_entry_rule("rsi < 30")
    builder.add_exit_rule("rsi > 70")
    
    # Risk management
    builder.set_risk_management(
        stop_loss=0.05,  # 5% stop loss
        take_profit=0.10,  # 10% take profit
        max_positions=3
    )
    
    return builder.build()


class TestPosition:
    """Test Position class."""
    
    def test_position_creation(self):
        """Test creating a position."""
        pos = Position(symbol="AAPL")
        
        assert pos.symbol == "AAPL"
        assert pos.quantity == 0
        assert pos.is_open() is False
        
    def test_position_trades(self):
        """Test adding trades to position."""
        pos = Position(symbol="AAPL")
        
        # Open position
        pos.add_trade(datetime.now(), 100, 150.0, 1.0)
        
        assert pos.quantity == 100
        assert pos.avg_price == 150.0
        assert pos.is_open() is True
        assert pos.is_long() is True
        
        # Add to position
        pos.add_trade(datetime.now(), 50, 155.0, 0.5)
        
        assert pos.quantity == 150
        assert pos.avg_price == pytest.approx(151.67, 0.01)
        
        # Reduce position
        pos.current_price = 160.0
        pos.add_trade(datetime.now(), -50, 160.0, 0.5)
        
        assert pos.quantity == 100
        assert pos.realized_pnl > 0  # Should have profit
        
    def test_position_pnl(self):
        """Test P&L calculations."""
        pos = Position(symbol="AAPL")
        pos.add_trade(datetime.now(), 100, 100.0, 1.0)
        
        # Test unrealized P&L
        pos.current_price = 110.0
        assert pos.unrealized_pnl() == pytest.approx(1000.0)
        assert pos.return_pct() == pytest.approx(10.0)
        
        # Test realized P&L
        pos.add_trade(datetime.now(), -100, 110.0, 1.0)
        assert pos.realized_pnl == pytest.approx(998.0)  # 1000 - 2 commission
        
    def test_position_stops(self):
        """Test stop loss and take profit."""
        pos = Position(symbol="AAPL")
        pos.add_trade(datetime.now(), 100, 100.0)
        pos.stop_loss = 95.0
        pos.take_profit = 110.0
        
        # Test stop loss hit
        pos.update_price(94.0)
        assert pos.check_stops() == 'stop_loss'
        
        # Test take profit hit
        pos.update_price(111.0)
        assert pos.check_stops() == 'take_profit'
        
        # Test no stop hit
        pos.update_price(105.0)
        assert pos.check_stops() is None


class TestPortfolio:
    """Test Portfolio class."""
    
    def test_portfolio_creation(self):
        """Test creating a portfolio."""
        portfolio = Portfolio(initial_capital=100000)
        
        assert portfolio.cash == 100000
        assert portfolio.current_value() == 100000
        assert len(portfolio.positions) == 0
        
    def test_order_placement(self):
        """Test placing orders."""
        portfolio = Portfolio()
        
        order = portfolio.place_order(
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET
        )
        
        assert order.symbol == "AAPL"
        assert order.quantity == 100
        assert order.direction == "BUY"
        assert order.status == OrderStatus.SUBMITTED
        
    def test_order_execution(self):
        """Test executing orders."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Place buy order
        order = portfolio.place_order("AAPL", 100, "BUY")
        
        # Execute order
        success = portfolio.execute_order(order, 150.0, datetime.now())
        
        assert success is True
        assert portfolio.cash < 100000  # Cash reduced
        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].quantity == 100
        
    def test_position_sizing(self):
        """Test position sizing calculations."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Fixed sizing
        size = portfolio.calculate_position_size("AAPL", 100.0, position_pct=0.1)
        assert size == 100  # 10% of 100k / $100 = 100 shares
        
        # Risk-based sizing
        size = portfolio.calculate_position_size("AAPL", 100.0, risk_amount=1000)
        assert size == 10  # $1000 / $100 = 10 shares
        
    def test_performance_tracking(self):
        """Test performance summary."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Make some trades
        order1 = portfolio.place_order("AAPL", 100, "BUY")
        portfolio.execute_order(order1, 100.0, datetime.now())
        
        # Update price
        portfolio.update_prices({"AAPL": 110.0})
        
        # Take snapshot
        snapshot = portfolio.take_snapshot(datetime.now())
        
        assert snapshot.total_value > 100000  # Should have profit
        assert snapshot.unrealized_pnl > 0
        
        # Get performance summary
        summary = portfolio.get_performance_summary()
        
        assert 'total_return' in summary
        assert 'total_trades' in summary
        assert summary['total_trades'] == 1


class TestBacktestEngine:
    """Test backtesting engine."""
    
    def test_engine_initialization(self):
        """Test engine creation."""
        engine = BacktestEngine(initial_capital=100000)
        
        assert engine.initial_capital == 100000
        assert engine.portfolio is None  # Not initialized until run
        
    def test_simple_backtest(self, sample_market_data, simple_strategy):
        """Test running a simple backtest."""
        engine = BacktestEngine(initial_capital=100000)
        
        results = engine.run(
            sample_market_data,
            simple_strategy,
            progress_bar=False
        )
        
        # Check results structure
        assert 'equity_curve' in results
        assert 'trades' in results
        assert 'performance' in results
        assert 'final_portfolio' in results
        
        # Check equity curve
        assert not results['equity_curve'].empty
        assert 'total_value' in results['equity_curve'].columns
        
    def test_event_processing(self, sample_market_data):
        """Test event processing."""
        engine = BacktestEngine()
        
        # Create market event
        event = MarketEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            open=100, high=101, low=99, close=100.5,
            volume=1000000
        )
        
        # Process event (need to initialize first)
        engine._initialize_backtest(
            sample_market_data,
            StrategyBuilder("Test").build()
        )
        
        engine._process_event(event)
        assert engine.processed_events == 1
        
    def test_performance_metrics(self, sample_market_data, simple_strategy):
        """Test performance metrics calculation."""
        engine = BacktestEngine()
        results = engine.run(sample_market_data, simple_strategy, progress_bar=False)
        
        # Calculate metrics
        metrics = PerformanceMetrics.calculate(
            results['equity_curve'],
            results['trades']
        )
        
        # Check metrics
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'win_rate')


class TestEvents:
    """Test event classes."""
    
    def test_market_event(self):
        """Test MarketEvent."""
        event = MarketEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            open=100, high=101, low=99, close=100.5,
            volume=1000000
        )
        
        assert event.symbol == "AAPL"
        assert event.close == 100.5
        assert event.get_type().value == "MARKET"
        
    def test_signal_event(self):
        """Test SignalEvent."""
        event = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type="LONG",
            strength=0.8,
            stop_loss=95.0,
            take_profit=110.0
        )
        
        assert event.signal_type == "LONG"
        assert event.strength == 0.8
        assert event.get_type().value == "SIGNAL"
        
    def test_order_event(self):
        """Test OrderEvent."""
        event = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type="MARKET",
            quantity=100,
            direction="BUY"
        )
        
        assert event.quantity == 100
        assert event.direction == "BUY"
        assert event.get_type().value == "ORDER"


class TestIntegration:
    """Integration tests."""
    
    def test_full_backtest_flow(self, sample_market_data):
        """Test complete backtest flow."""
        # Create strategy
        builder = StrategyBuilder("Integration Test")
        builder.add_entry_rule("close > open")  # Buy on green candles
        builder.add_exit_rule("close < open")   # Sell on red candles
        builder.set_risk_management(stop_loss=0.02, max_positions=1)
        
        strategy = builder.build()
        
        # Run backtest
        engine = BacktestEngine(
            initial_capital=100000,
            commission_rate=0.001
        )
        
        results = engine.run(
            sample_market_data,
            strategy,
            progress_bar=False
        )
        
        # Verify results
        assert results['statistics']['total_events'] > 0
        assert len(results['equity_curve']) == len(sample_market_data)
        
        # Should have made some trades
        if not results['trades'].empty:
            assert results['statistics']['orders_executed'] > 0
            
        # Calculate and verify metrics
        metrics = PerformanceMetrics.calculate(
            results['equity_curve'],
            results['trades']
        )
        
        assert metrics.total_return != 0  # Should have some return
        assert metrics.volatility > 0  # Should have volatility