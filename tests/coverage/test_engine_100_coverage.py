"""Comprehensive tests for backtesting engine to achieve 100% coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import queue

from src.backtesting.engine import BacktestEngine, BacktestResults
from src.backtesting.portfolio import Portfolio
from src.backtesting.position import Position
from src.backtesting.order import Order, OrderType, OrderSide, OrderStatus
from src.backtesting.events import (
    MarketEvent, OrderEvent, FillEvent, SignalEvent, EventType
)
from src.strategies.base import BaseStrategy
from src.strategies.builder import StrategyBuilder
from src.reporting import StandardReportGenerator, ReportConfig


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-02-01', freq='D')
    data = pd.DataFrame({
        'open': np.random.uniform(90, 110, len(dates)),
        'high': np.random.uniform(95, 115, len(dates)),
        'low': np.random.uniform(85, 105, len(dates)),
        'close': np.random.uniform(90, 110, len(dates)),
        'volume': np.random.uniform(1000000, 2000000, len(dates))
    }, index=dates)
    data.attrs['symbol'] = 'TEST'
    return data


@pytest.fixture
def simple_strategy():
    """Create a simple test strategy."""
    strategy = StrategyBuilder("Test Strategy")
    strategy.add_entry_rule("close > open")
    strategy.add_exit_rule("close < open")
    return strategy.build()


class TestBacktestResults:
    """Test BacktestResults class."""
    
    def test_backtest_results_initialization(self):
        """Test BacktestResults initialization and access methods."""
        results_dict = {
            'total_return': 0.15,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.10,
            'win_rate': 0.60
        }
        
        results = BacktestResults(results_dict)
        
        # Test dictionary-like access
        assert results['total_return'] == 0.15
        assert results['sharpe_ratio'] == 1.5
        
        # Test attribute-like access
        assert results.total_return == 0.15
        assert results.sharpe_ratio == 1.5
        
        # Test keys method
        assert set(results.keys()) == set(results_dict.keys())
        
        # Test get method with default
        assert results.get('total_return') == 0.15
        assert results.get('nonexistent', 'default') == 'default'
        
        # Test attribute access for non-existent key
        assert results.nonexistent is None


class TestBacktestEngine:
    """Test BacktestEngine class."""
    
    def test_engine_initialization_all_parameters(self):
        """Test engine initialization with all parameters."""
        report_config = ReportConfig()
        engine = BacktestEngine(
            initial_capital=50000,
            commission_rate=0.002,
            slippage_rate=0.001,
            max_positions=5,
            generate_report=False,
            report_config=report_config,
            report_dir="test_reports"
        )
        
        assert engine.initial_capital == 50000
        assert engine.commission_rate == 0.002
        assert engine.slippage_rate == 0.001
        assert engine.max_positions == 5
        assert engine.generate_report == False
        assert engine.report_config == report_config
        assert str(engine.report_dir) == "test_reports"
        assert engine.events_queue.empty()
        assert engine.portfolio is None
        assert engine.strategy is None
        assert engine.data is None
        assert engine.market_data_index == 0
        assert engine.current_time is None
        assert engine.is_running == False
        assert engine.processed_events == 0
        assert engine.generated_signals == 0
        assert engine.executed_orders == 0
    
    def test_initialize_backtest_with_date_filtering(self, sample_data, simple_strategy):
        """Test _initialize_backtest with date filtering."""
        engine = BacktestEngine()
        
        start_date = datetime(2023, 1, 10)
        end_date = datetime(2023, 1, 20)
        
        engine._initialize_backtest(
            data=sample_data,
            strategy=simple_strategy,
            start_date=start_date,
            end_date=end_date
        )
        
        # Check data filtering
        assert engine.data.index[0] >= start_date
        assert engine.data.index[-1] <= end_date
        assert len(engine.data) < len(sample_data)
        
        # Check initialization
        assert engine.market_data_index == 0
        assert isinstance(engine.portfolio, Portfolio)
        assert engine.strategy == simple_strategy
        assert engine.events_queue.empty()
        assert engine.processed_events == 0
        assert engine.generated_signals == 0
        assert engine.executed_orders == 0
    
    def test_generate_market_event_all_column_formats(self):
        """Test _generate_market_event with different column formats."""
        engine = BacktestEngine()
        
        # Test with lowercase columns
        data_lower = pd.DataFrame({
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [102],
            'volume': [1000000]
        }, index=[datetime(2023, 1, 1)])
        data_lower.attrs['symbol'] = 'TEST'
        
        engine.data = data_lower
        engine.market_data_index = 0
        engine.events_queue = queue.Queue()
        
        engine._generate_market_event()
        
        event = engine.events_queue.get()
        assert isinstance(event, MarketEvent)
        assert event.symbol == 'TEST'
        assert event.open == 100
        assert event.close == 102
        assert engine.market_data_index == 1
        
        # Test with uppercase columns
        data_upper = pd.DataFrame({
            'Open': [100],
            'High': [105],
            'Low': [95],
            'Close': [102],
            'Volume': [1000000]
        }, index=[datetime(2023, 1, 2)])
        
        engine.data = data_upper
        engine.market_data_index = 0
        
        engine._generate_market_event()
        
        event = engine.events_queue.get()
        assert event.symbol == 'UNKNOWN'  # No symbol attribute
        assert event.open == 100
        assert event.close == 102
    
    def test_generate_market_event_boundary_conditions(self):
        """Test _generate_market_event at data boundaries."""
        engine = BacktestEngine()
        data = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [95, 96],
            'close': [102, 103],
            'volume': [1000000, 1100000]
        }, index=pd.date_range('2023-01-01', periods=2))
        
        engine.data = data
        engine.events_queue = queue.Queue()
        
        # Test at end of data
        engine.market_data_index = 2
        engine._generate_market_event()
        assert engine.events_queue.empty()
        
        # Test normal generation
        engine.market_data_index = 0
        engine._generate_market_event()
        assert not engine.events_queue.empty()
    
    def test_process_event_all_event_types(self):
        """Test _process_event with all event types."""
        engine = BacktestEngine()
        engine.processed_events = 0
        
        # Mock handler methods
        engine._handle_market_event = Mock()
        engine._handle_signal_event = Mock()
        engine._handle_order_event = Mock()
        engine._handle_fill_event = Mock()
        
        # Test market event
        market_event = MarketEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            open=100, high=105, low=95, close=102, volume=1000000
        )
        engine._process_event(market_event)
        assert engine.processed_events == 1
        engine._handle_market_event.assert_called_once_with(market_event)
        
        # Test signal event
        signal_event = SignalEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            signal_type='LONG',
            strength=0.8
        )
        engine._process_event(signal_event)
        assert engine.processed_events == 2
        engine._handle_signal_event.assert_called_once_with(signal_event)
        
        # Test order event
        order_event = OrderEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            order_type='MARKET',
            quantity=100,
            direction='BUY'
        )
        engine._process_event(order_event)
        assert engine.processed_events == 3
        engine._handle_order_event.assert_called_once_with(order_event)
        
        # Test fill event
        fill_event = FillEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            exchange='BACKTEST',
            quantity=100,
            direction='BUY',
            fill_price=102.5,
            commission=0.1
        )
        engine._process_event(fill_event)
        assert engine.processed_events == 4
        engine._handle_fill_event.assert_called_once_with(fill_event)
    
    def test_should_generate_signal(self):
        """Test _should_generate_signal method."""
        engine = BacktestEngine()
        
        # Test with insufficient data
        small_data = pd.DataFrame({'close': [100, 101, 102]})
        assert not engine._should_generate_signal(small_data)
        
        # Test with sufficient data
        large_data = pd.DataFrame({'close': np.random.uniform(90, 110, 60)})
        assert engine._should_generate_signal(large_data)
    
    def test_handle_market_event_with_stops(self, sample_data):
        """Test _handle_market_event with stop orders."""
        engine = BacktestEngine()
        engine._initialize_backtest(sample_data, Mock())
        engine.events_queue = queue.Queue()
        
        # Create a position with stop loss
        position = Position(symbol='TEST')
        position.quantity = 100
        position.entry_price = 100
        position.stop_loss = 95
        position.current_price = 100
        engine.portfolio.positions['TEST'] = position
        
        # Create stop order
        stop_order = Order(
            symbol='TEST',
            quantity=100,
            order_type=OrderType.STOP,
            side=OrderSide.SELL
        )
        stop_order.stop_price = 95
        
        # Mock check_stops to return stop order
        engine.portfolio.check_stops = Mock(return_value=[stop_order])
        
        # Create market event that triggers stop
        market_event = MarketEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            open=96, high=97, low=94, close=94, volume=1000000
        )
        
        engine._handle_market_event(market_event)
        
        # Check stop order was generated
        assert not engine.events_queue.empty()
        order_event = engine.events_queue.get()
        assert isinstance(order_event, OrderEvent)
        assert order_event.symbol == 'TEST'
        assert order_event.quantity == 100
    
    def test_handle_market_event_signal_generation(self, sample_data, simple_strategy):
        """Test _handle_market_event with signal generation."""
        engine = BacktestEngine()
        engine._initialize_backtest(sample_data, simple_strategy)
        engine.market_data_index = 60  # Enough data for signals
        engine.generated_signals = 0
        
        # Mock signal generator to return entry signal
        signals_df = pd.DataFrame({
            'entry': [True],
            'exit': [False],
            'signal_strength': [0.8],
            'stop_loss': [95],
            'take_profit': [105]
        })
        engine.signal_generator.generate = Mock(return_value=signals_df)
        engine.portfolio.can_open_position = Mock(return_value=True)
        
        market_event = MarketEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            open=100, high=105, low=95, close=102, volume=1000000
        )
        
        engine._handle_market_event(market_event)
        
        # Check signal was generated
        assert engine.generated_signals == 1
        assert not engine.events_queue.empty()
        signal_event = engine.events_queue.get()
        assert isinstance(signal_event, SignalEvent)
        assert signal_event.signal_type == 'LONG'
        assert signal_event.strength == 0.8
        assert signal_event.stop_loss == 95
        assert signal_event.take_profit == 105
    
    def test_handle_market_event_exit_signal(self, sample_data, simple_strategy):
        """Test _handle_market_event with exit signal generation."""
        engine = BacktestEngine()
        engine._initialize_backtest(sample_data, simple_strategy)
        engine.market_data_index = 60
        engine.generated_signals = 0
        
        # Add existing position
        engine.portfolio.positions['TEST'] = Position(symbol='TEST')
        
        # Mock signal generator to return exit signal
        signals_df = pd.DataFrame({
            'entry': [False],
            'exit': [True]
        })
        engine.signal_generator.generate = Mock(return_value=signals_df)
        
        market_event = MarketEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            open=100, high=105, low=95, close=102, volume=1000000
        )
        
        engine._handle_market_event(market_event)
        
        # Check exit signal was generated
        assert engine.generated_signals == 1
        signal_event = engine.events_queue.get()
        assert signal_event.signal_type == 'EXIT'
    
    def test_handle_signal_event_long_entry(self):
        """Test _handle_signal_event for long entry."""
        engine = BacktestEngine()
        engine.portfolio = Mock()
        engine.portfolio.calculate_position_size = Mock(return_value=100)
        engine.portfolio.place_order = Mock()
        engine.portfolio.positions = {}
        engine.events_queue = queue.Queue()
        
        # Create mock data
        engine.data = pd.DataFrame({'close': [100]}, index=[datetime.now()])
        engine.market_data_index = 1
        
        signal_event = SignalEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            signal_type='LONG',
            strength=0.8,
            stop_loss=95,
            take_profit=105
        )
        
        engine._handle_signal_event(signal_event)
        
        # Check order was placed
        engine.portfolio.place_order.assert_called_once()
        
        # Check position was created with stops
        assert 'TEST' in engine.portfolio.positions
        position = engine.portfolio.positions['TEST']
        assert position.stop_loss == 95
        assert position.take_profit == 105
        
        # Check order event was generated
        assert not engine.events_queue.empty()
        order_event = engine.events_queue.get()
        assert order_event.direction == 'BUY'
        assert order_event.quantity == 100
    
    def test_handle_signal_event_exit(self):
        """Test _handle_signal_event for exit signal."""
        engine = BacktestEngine()
        engine.portfolio = Mock()
        engine.events_queue = queue.Queue()
        
        # Create existing long position
        position = Mock()
        position.is_open = Mock(return_value=True)
        position.is_long = Mock(return_value=True)
        position.quantity = 100
        engine.portfolio.positions = {'TEST': position}
        
        signal_event = SignalEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            signal_type='EXIT',
            strength=1.0
        )
        
        engine._handle_signal_event(signal_event)
        
        # Check order was placed
        engine.portfolio.place_order.assert_called_once_with(
            symbol='TEST',
            quantity=100,
            direction='SELL',
            order_type=OrderType.MARKET
        )
        
        # Check order event was generated
        assert not engine.events_queue.empty()
        order_event = engine.events_queue.get()
        assert order_event.direction == 'SELL'
    
    def test_handle_order_event(self):
        """Test _handle_order_event."""
        engine = BacktestEngine()
        engine.portfolio = Mock()
        engine.portfolio.execute_order = Mock(return_value=100)  # Filled quantity
        engine.executed_orders = 0
        engine.events_queue = queue.Queue()
        
        order_event = OrderEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            order_type='MARKET',
            quantity=100,
            direction='BUY'
        )
        
        engine._handle_order_event(order_event)
        
        # Check order was executed
        engine.portfolio.execute_order.assert_called_once()
        assert engine.executed_orders == 1
        
        # Check fill event was generated
        assert not engine.events_queue.empty()
        fill_event = engine.events_queue.get()
        assert isinstance(fill_event, FillEvent)
        assert fill_event.quantity == 100
    
    def test_handle_fill_event(self):
        """Test _handle_fill_event."""
        engine = BacktestEngine()
        engine.portfolio = Mock()
        
        fill_event = FillEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            exchange='BACKTEST',
            quantity=100,
            direction='BUY',
            fill_price=102.5,
            commission=0.1
        )
        
        engine._handle_fill_event(fill_event)
        
        # Check portfolio was updated
        engine.portfolio.update_from_fill.assert_called_once_with(fill_event)
    
    def test_generate_results(self):
        """Test _generate_results method."""
        engine = BacktestEngine()
        
        # Mock portfolio with metrics
        engine.portfolio = Mock()
        engine.portfolio.get_metrics = Mock(return_value={
            'total_return': 0.15,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.10
        })
        engine.portfolio.total_value = 115000
        engine.portfolio.closed_positions = [Mock(), Mock()]
        engine.portfolio.get_snapshots = Mock(return_value=pd.DataFrame({
            'total_value': [100000, 105000, 110000, 115000]
        }))
        
        # Set timing
        engine.start_time = datetime.now()
        engine.end_time = engine.start_time + timedelta(seconds=30)
        engine.processed_events = 1000
        engine.generated_signals = 50
        engine.executed_orders = 25
        
        results = engine._generate_results()
        
        assert isinstance(results, dict)
        assert results['metrics']['total_return'] == 0.15
        assert results['portfolio_value'] == 115000
        assert results['trades'] == 2
        assert results['processing_time'] == 30.0
        assert results['events_processed'] == 1000
        assert results['signals_generated'] == 50
        assert results['orders_executed'] == 25
        assert 'portfolio_history' in results
    
    @patch('src.backtesting.engine.StandardReportGenerator')
    def test_generate_standard_report(self, mock_generator_class):
        """Test _generate_standard_report method."""
        engine = BacktestEngine(
            report_config=ReportConfig(save_html=True),
            report_dir="test_reports"
        )
        
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        results = {
            'metrics': {'total_return': 0.15},
            'portfolio_value': 115000
        }
        
        engine._generate_standard_report(results)
        
        # Check report generator was created and called
        mock_generator_class.assert_called_once_with(
            results=results,
            config=engine.report_config,
            output_dir=engine.report_dir
        )
        mock_generator.generate.assert_called_once()
    
    def test_run_full_backtest_with_progress(self, sample_data, simple_strategy):
        """Test full backtest run with progress bar."""
        engine = BacktestEngine(generate_report=False)
        
        # Mock signal generator to avoid issues
        engine.signal_generator.generate = Mock(return_value=pd.DataFrame())
        
        results = engine.run(
            data=sample_data,
            strategy=simple_strategy,
            progress_bar=True
        )
        
        assert isinstance(results, dict)
        assert 'metrics' in results
        assert 'portfolio_value' in results
        assert engine.start_time is not None
        assert engine.end_time is not None
        assert engine.is_running == False
    
    def test_run_with_empty_queue_exception(self, sample_data, simple_strategy):
        """Test run method handles empty queue exception."""
        engine = BacktestEngine(generate_report=False)
        
        # Make data very small to test the loop
        small_data = sample_data.iloc[:2]
        
        results = engine.run(
            data=small_data,
            strategy=simple_strategy,
            progress_bar=False
        )
        
        assert isinstance(results, dict)
        assert engine.is_running == False
    
    def test_edge_cases_and_error_handling(self):
        """Test various edge cases."""
        engine = BacktestEngine()
        
        # Test with no positions for calculate_position_size
        engine.portfolio = Mock()
        engine.portfolio.positions = {}
        engine.portfolio.calculate_position_size = Mock(return_value=0)
        engine.events_queue = queue.Queue()
        
        signal_event = SignalEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            signal_type='LONG',
            strength=0.8
        )
        
        # Should not generate order event if quantity is 0
        engine._handle_signal_event(signal_event)
        assert engine.events_queue.empty()
        
        # Test exit signal with no position
        exit_event = SignalEvent(
            timestamp=datetime.now(),
            symbol='NOPOS',
            signal_type='EXIT',
            strength=1.0
        )
        
        engine._handle_signal_event(exit_event)
        # Should not crash or generate events
        assert engine.events_queue.empty()