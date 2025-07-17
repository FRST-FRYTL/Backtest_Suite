"""Optimized comprehensive tests for BacktestEngine - matches actual implementation."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import queue

from src.backtesting.engine import BacktestEngine, BacktestResults
from src.backtesting.events import (
    Event, MarketEvent, SignalEvent, OrderEvent, FillEvent, EventType
)
from src.backtesting.portfolio import Portfolio
from src.backtesting.order import Order, OrderType, OrderSide
from src.strategies.base import BaseStrategy
from src.strategies.builder import StrategyBuilder


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    close = 100
    data = []
    
    for date in dates:
        open_price = close * (1 + np.random.uniform(-0.02, 0.02))
        high = max(open_price, close) * (1 + np.random.uniform(0, 0.02))
        low = min(open_price, close) * (1 - np.random.uniform(0, 0.02))
        close = open_price * (1 + np.random.uniform(-0.03, 0.03))
        volume = np.random.randint(1000000, 5000000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    df.attrs['symbol'] = 'TEST'
    return df


@pytest.fixture
def simple_strategy():
    """Create a simple test strategy."""
    builder = StrategyBuilder("Simple Test Strategy")
    builder.add_entry_rule("close > open")
    builder.add_exit_rule("close < open")
    return builder.build()


class TestBacktestEngine:
    """Comprehensive tests for BacktestEngine covering all methods."""
    
    def test_initialization(self):
        """Test engine initialization with various parameters."""
        # Default initialization
        engine = BacktestEngine()
        assert engine.initial_capital == 100000.0
        assert engine.commission_rate == 0.001
        assert engine.slippage_rate == 0.0005
        assert engine.max_positions == 10
        assert engine.generate_report == True
        assert engine.portfolio is None
        assert engine.strategy is None
        assert engine.data is None
        assert isinstance(engine.events_queue, queue.Queue)
        
        # Custom initialization
        engine2 = BacktestEngine(
            initial_capital=50000,
            commission_rate=0.002,
            slippage_rate=0.001,
            max_positions=5,
            generate_report=False,
            report_dir="custom_reports"
        )
        assert engine2.initial_capital == 50000
        assert engine2.commission_rate == 0.002
        assert engine2.slippage_rate == 0.001
        assert engine2.max_positions == 5
        assert engine2.generate_report == False
        assert str(engine2.report_dir).endswith("custom_reports")
    
    def test_backtest_results_class(self):
        """Test BacktestResults wrapper class."""
        results_dict = {
            'returns': 0.15,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.10
        }
        
        results = BacktestResults(results_dict)
        
        # Test dictionary-like access
        assert results['returns'] == 0.15
        assert results['sharpe_ratio'] == 1.5
        
        # Test attribute-like access
        assert results.returns == 0.15
        assert results.sharpe_ratio == 1.5
        
        # Test get method
        assert results.get('max_drawdown') == -0.10
        assert results.get('missing_key', 'default') == 'default'
        
        # Test keys
        assert set(results.keys()) == {'returns', 'sharpe_ratio', 'max_drawdown'}
    
    def test_run_method_basic(self, sample_data, simple_strategy):
        """Test basic run method functionality."""
        engine = BacktestEngine(generate_report=False)
        
        results = engine.run(
            data=sample_data,
            strategy=simple_strategy,
            progress_bar=False
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'metrics' in results
        assert 'portfolio_value' in results
        assert 'positions' in results
        assert 'trades' in results
        
        # Verify engine state after run
        assert engine.portfolio is not None
        assert engine.strategy is not None
        assert engine.data is not None
        assert engine.processed_events > 0
    
    def test_run_with_date_range(self, sample_data, simple_strategy):
        """Test run method with date range filtering."""
        engine = BacktestEngine(generate_report=False)
        
        start_date = sample_data.index[50]
        end_date = sample_data.index[100]
        
        results = engine.run(
            data=sample_data,
            strategy=simple_strategy,
            start_date=start_date,
            end_date=end_date,
            progress_bar=False
        )
        
        # Verify data was filtered
        assert len(engine.data) == 51  # 50 to 100 inclusive
        assert engine.data.index[0] >= start_date
        assert engine.data.index[-1] <= end_date
    
    def test_initialize_backtest(self, sample_data, simple_strategy):
        """Test _initialize_backtest method."""
        engine = BacktestEngine()
        
        engine._initialize_backtest(
            data=sample_data,
            strategy=simple_strategy
        )
        
        # Verify initialization
        assert engine.portfolio is not None
        assert isinstance(engine.portfolio, Portfolio)
        assert engine.portfolio.initial_capital == engine.initial_capital
        assert engine.strategy == simple_strategy
        assert engine.data.equals(sample_data)
        assert engine.market_data_index == 0
        assert engine.processed_events == 0
        assert engine.generated_signals == 0
        assert engine.executed_orders == 0
    
    def test_generate_market_event(self, sample_data, simple_strategy):
        """Test _generate_market_event method."""
        engine = BacktestEngine()
        engine._initialize_backtest(sample_data, simple_strategy)
        
        # Generate first market event
        engine._generate_market_event()
        
        # Check event was created
        assert not engine.events_queue.empty()
        event = engine.events_queue.get()
        
        assert isinstance(event, MarketEvent)
        assert event.symbol == 'TEST'
        assert event.timestamp == sample_data.index[0]
        assert event.open == sample_data.iloc[0]['open']
        assert event.close == sample_data.iloc[0]['close']
        assert engine.market_data_index == 1
        
        # Test at end of data
        engine.market_data_index = len(sample_data)
        engine._generate_market_event()
        assert engine.events_queue.empty()
    
    def test_process_event_dispatcher(self, sample_data, simple_strategy):
        """Test _process_event method dispatches correctly."""
        engine = BacktestEngine()
        engine._initialize_backtest(sample_data, simple_strategy)
        
        # Test market event
        market_event = MarketEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            open=100, high=101, low=99, close=100.5, volume=1000
        )
        engine._process_event(market_event)
        assert engine.processed_events == 1
        
        # Test signal event
        signal_event = SignalEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            signal_type='BUY',
            strength=0.8,
            price=100.0
        )
        engine._process_event(signal_event)
        assert engine.processed_events == 2
        
        # Test order event
        order_event = OrderEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            order_type='MARKET',
            quantity=100,
            side='BUY'
        )
        engine._process_event(order_event)
        assert engine.processed_events == 3
        
        # Test fill event
        fill_event = FillEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            quantity=100,
            price=100.0,
            commission=1.0,
            slippage=0.1,
            side='BUY'
        )
        engine._process_event(fill_event)
        assert engine.processed_events == 4
    
    def test_handle_market_event(self, sample_data, simple_strategy):
        """Test _handle_market_event method."""
        engine = BacktestEngine()
        engine._initialize_backtest(sample_data, simple_strategy)
        
        # Create market event
        event = MarketEvent(
            timestamp=sample_data.index[0],
            symbol='TEST',
            open=100, high=101, low=99, close=100.5, volume=1000
        )
        
        # Mock strategy to capture calls
        with patch.object(engine.strategy, 'generate_signals') as mock_generate:
            mock_generate.return_value = {'signal': 1, 'confidence': 0.8}
            
            engine._handle_market_event(event)
            
            # Verify portfolio was updated
            assert 'TEST' in engine.portfolio.current_prices
            assert engine.portfolio.current_prices['TEST'] == 100.5
            
            # Verify strategy was called
            mock_generate.assert_called_once()
            
            # Verify signal was generated
            assert engine.generated_signals == 1
    
    def test_handle_signal_event(self, sample_data, simple_strategy):
        """Test _handle_signal_event method."""
        engine = BacktestEngine()
        engine._initialize_backtest(sample_data, simple_strategy)
        
        # Create signal event
        signal = SignalEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            signal_type='BUY',
            strength=0.8,
            price=100.0,
            size=100
        )
        
        # Handle signal
        engine._handle_signal_event(signal)
        
        # Check order was created
        assert not engine.events_queue.empty()
        order_event = engine.events_queue.get()
        assert isinstance(order_event, OrderEvent)
        assert order_event.symbol == 'TEST'
        assert order_event.quantity == 100
    
    def test_handle_order_event(self, sample_data, simple_strategy):
        """Test _handle_order_event method."""
        engine = BacktestEngine()
        engine._initialize_backtest(sample_data, simple_strategy)
        
        # Set current price
        engine.portfolio.current_prices['TEST'] = 100.0
        
        # Create order event
        order = OrderEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            order_type='MARKET',
            quantity=100,
            side='BUY'
        )
        
        # Handle order
        engine._handle_order_event(order)
        
        # Check fill was created
        assert not engine.events_queue.empty()
        fill_event = engine.events_queue.get()
        assert isinstance(fill_event, FillEvent)
        assert fill_event.symbol == 'TEST'
        assert fill_event.quantity == 100
        assert engine.executed_orders == 1
    
    def test_handle_fill_event(self, sample_data, simple_strategy):
        """Test _handle_fill_event method."""
        engine = BacktestEngine()
        engine._initialize_backtest(sample_data, simple_strategy)
        
        # Create fill event
        fill = FillEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            quantity=100,
            price=100.0,
            commission=1.0,
            slippage=0.1,
            side='BUY'
        )
        
        # Handle fill
        engine._handle_fill_event(fill)
        
        # Check position was created
        assert 'TEST' in engine.portfolio.positions
        position = engine.portfolio.positions['TEST']
        assert position.quantity == 100
        assert position.avg_entry_price == 100.0
    
    def test_calculate_slippage(self, sample_data, simple_strategy):
        """Test _calculate_slippage method."""
        engine = BacktestEngine(slippage_rate=0.001)
        engine._initialize_backtest(sample_data, simple_strategy)
        
        # Test buy side slippage
        slippage = engine._calculate_slippage(100.0, 'BUY')
        assert slippage == 0.1  # 100 * 0.001
        
        # Test sell side slippage
        slippage = engine._calculate_slippage(100.0, 'SELL')
        assert slippage == -0.1  # -100 * 0.001
    
    def test_calculate_commission(self, sample_data, simple_strategy):
        """Test _calculate_commission method."""
        engine = BacktestEngine(commission_rate=0.001)
        engine._initialize_backtest(sample_data, simple_strategy)
        
        commission = engine._calculate_commission(100.0, 100)
        assert commission == 10.0  # 100 * 100 * 0.001
    
    def test_generate_results(self, sample_data, simple_strategy):
        """Test _generate_results method."""
        engine = BacktestEngine(generate_report=False)
        
        # Run a simple backtest
        results = engine.run(
            data=sample_data,
            strategy=simple_strategy,
            progress_bar=False
        )
        
        # Verify results structure
        assert 'metrics' in results
        assert 'portfolio_value' in results
        assert 'trades' in results
        assert 'positions' in results
        
        # Verify metrics
        metrics = results['metrics']
        assert 'returns' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'total_trades' in metrics
        
        # Verify timing info
        assert engine.start_time is not None
        assert engine.end_time is not None
        assert engine.end_time > engine.start_time
    
    def test_event_queue_processing(self, sample_data, simple_strategy):
        """Test complete event queue processing."""
        engine = BacktestEngine(generate_report=False)
        
        # Create a strategy that generates signals
        class ActiveStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("Active Strategy")
                self.trade_count = 0
            
            def generate_signals(self, data):
                if len(data) == 50 and self.trade_count == 0:
                    self.trade_count += 1
                    return {'signal': 1, 'confidence': 0.9, 'size': 100}
                elif len(data) == 100 and self.trade_count == 1:
                    self.trade_count += 1
                    return {'signal': -1, 'confidence': 0.8, 'size': 100}
                return {'signal': 0}
        
        results = engine.run(
            data=sample_data,
            strategy=ActiveStrategy(),
            progress_bar=False
        )
        
        # Verify events were processed
        assert engine.processed_events > len(sample_data)
        assert engine.generated_signals >= 2
        assert engine.executed_orders >= 2
        assert len(results['trades']) >= 1
    
    def test_empty_data_handling(self, simple_strategy):
        """Test handling of empty data."""
        engine = BacktestEngine(generate_report=False)
        
        empty_data = pd.DataFrame({
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        })
        
        results = engine.run(
            data=empty_data,
            strategy=simple_strategy,
            progress_bar=False
        )
        
        assert results is not None
        assert results['metrics']['total_trades'] == 0
    
    def test_report_generation(self, sample_data, simple_strategy, tmp_path):
        """Test report generation functionality."""
        # Create engine with report generation enabled
        engine = BacktestEngine(
            generate_report=True,
            report_dir=str(tmp_path)
        )
        
        with patch('src.backtesting.engine.StandardReportGenerator') as mock_generator:
            mock_instance = Mock()
            mock_generator.return_value = mock_instance
            
            # Run backtest
            results = engine.run(
                data=sample_data,
                strategy=simple_strategy,
                progress_bar=False
            )
            
            # Verify report generator was called
            mock_generator.assert_called_once()
            mock_instance.generate_report.assert_called_once()


class TestBacktestEngineIntegration:
    """Integration tests for BacktestEngine."""
    
    def test_complete_trading_cycle(self, sample_data):
        """Test complete buy-hold-sell cycle."""
        engine = BacktestEngine(
            initial_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005,
            generate_report=False
        )
        
        # Create strategy with specific buy/sell points
        class CycleStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("Cycle Strategy")
                self.bought = False
                self.sold = False
            
            def generate_signals(self, data):
                if len(data) == 50 and not self.bought:
                    self.bought = True
                    return {'signal': 1, 'confidence': 1.0, 'size': 1000}
                elif len(data) == 150 and self.bought and not self.sold:
                    self.sold = True
                    return {'signal': -1, 'confidence': 1.0, 'size': 1000}
                return {'signal': 0}
        
        results = engine.run(
            data=sample_data,
            strategy=CycleStrategy(),
            progress_bar=False
        )
        
        # Verify trade cycle
        trades = results['trades']
        assert len(trades) >= 1
        
        if trades:
            trade = trades[0]
            assert trade['quantity'] == 1000
            assert trade['entry_date'] < trade['exit_date']
            assert 'pnl' in trade
            assert 'return' in trade
    
    def test_multiple_positions(self, sample_data):
        """Test handling multiple concurrent positions."""
        engine = BacktestEngine(
            initial_capital=100000,
            max_positions=3,
            generate_report=False
        )
        
        # Strategy that opens multiple positions
        class MultiPositionStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("Multi Position")
                self.signals_sent = 0
            
            def generate_signals(self, data):
                # Open 3 positions at different times
                if len(data) in [20, 40, 60] and self.signals_sent < 3:
                    self.signals_sent += 1
                    return {'signal': 1, 'confidence': 0.8, 'size': 100}
                # Close all at once
                elif len(data) == 100:
                    return {'signal': -1, 'confidence': 1.0, 'size': 300}
                return {'signal': 0}
        
        results = engine.run(
            data=sample_data,
            strategy=MultiPositionStrategy(),
            progress_bar=False
        )
        
        # Verify multiple positions were handled
        assert engine.executed_orders >= 4  # 3 buys + 1 sell
        assert len(results['trades']) >= 1
    
    def test_performance_under_stress(self, sample_data):
        """Test engine performance with high-frequency signals."""
        engine = BacktestEngine(generate_report=False)
        
        # High-frequency strategy
        class HighFreqStrategy(BaseStrategy):
            def generate_signals(self, data):
                # Generate signal every other bar
                if len(data) % 2 == 0:
                    signal = 1 if (len(data) // 2) % 2 == 0 else -1
                    return {'signal': signal, 'confidence': 0.7, 'size': 50}
                return {'signal': 0}
        
        import time
        start_time = time.time()
        
        results = engine.run(
            data=sample_data,
            strategy=HighFreqStrategy(),
            progress_bar=False
        )
        
        execution_time = time.time() - start_time
        
        # Verify performance
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert engine.processed_events > 0
        assert engine.generated_signals > 100  # Many signals generated