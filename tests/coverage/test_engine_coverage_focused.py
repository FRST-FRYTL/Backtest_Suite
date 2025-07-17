"""Focused tests for maximum BacktestEngine coverage with minimal complexity."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import queue

from src.backtesting.engine import BacktestEngine, BacktestResults
from src.backtesting.events import (
    MarketEvent, SignalEvent, OrderEvent, FillEvent, EventType
)
from src.strategies.builder import StrategyBuilder


@pytest.fixture
def minimal_data():
    """Minimal OHLCV data for testing."""
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    data = pd.DataFrame({
        'Open': [100] * 10,
        'High': [101] * 10,
        'Low': [99] * 10,
        'Close': [100.5] * 10,
        'Volume': [1000] * 10
    }, index=dates)
    data.attrs['symbol'] = 'TEST'
    return data


class TestBacktestEngineCoverage:
    """Tests focused on achieving maximum coverage."""
    
    def test_backtest_results_wrapper(self):
        """Test BacktestResults class methods."""
        results = BacktestResults({'key1': 'value1', 'key2': 2})
        
        # Test all access methods
        assert results['key1'] == 'value1'
        assert results.key2 == 2
        assert results.get('key1') == 'value1'
        assert results.get('missing', 'default') == 'default'
        assert list(results.keys()) == ['key1', 'key2']
    
    def test_engine_init_and_properties(self):
        """Test engine initialization and properties."""
        # Test with all parameters
        engine = BacktestEngine(
            initial_capital=50000,
            commission_rate=0.002,
            slippage_rate=0.001,
            max_positions=5,
            generate_report=False,
            report_config=Mock(),
            report_dir="test_reports"
        )
        
        assert engine.initial_capital == 50000
        assert engine.commission_rate == 0.002
        assert engine.slippage_rate == 0.001
        assert engine.max_positions == 5
        assert not engine.generate_report
        assert engine.report_config is not None
    
    def test_run_method_coverage(self, minimal_data):
        """Test run method with minimal viable execution."""
        engine = BacktestEngine(generate_report=False)
        
        # Create simple strategy
        strategy = StrategyBuilder("Test")
        strategy.add_entry_rule("close > 0")  # Always true
        built_strategy = strategy.build()
        
        # Mock signal generator to avoid complex logic
        with patch.object(engine.signal_generator, 'generate') as mock_gen:
            mock_gen.return_value = {}  # No signals
            
            results = engine.run(
                data=minimal_data,
                strategy=built_strategy,
                start_date=minimal_data.index[2],
                end_date=minimal_data.index[7],
                progress_bar=True  # Test progress bar branch
            )
            
            assert results is not None
            assert engine.start_time is not None
            assert engine.end_time is not None
    
    def test_initialize_backtest_branches(self, minimal_data):
        """Test _initialize_backtest with date filtering."""
        engine = BacktestEngine()
        strategy = Mock()
        
        # Test with date range
        engine._initialize_backtest(
            data=minimal_data,
            strategy=strategy,
            start_date=minimal_data.index[2],
            end_date=minimal_data.index[7]
        )
        
        assert len(engine.data) == 6  # Filtered data
        assert engine.portfolio is not None
        assert engine.strategy == strategy
        assert engine.market_data_index == 0
    
    def test_generate_market_event_branches(self, minimal_data):
        """Test _generate_market_event edge cases."""
        engine = BacktestEngine()
        engine._initialize_backtest(minimal_data, Mock())
        
        # Test normal generation
        engine._generate_market_event()
        assert not engine.events_queue.empty()
        event = engine.events_queue.get()
        assert isinstance(event, MarketEvent)
        
        # Test at end of data
        engine.market_data_index = len(engine.data)
        engine._generate_market_event()
        assert engine.events_queue.empty()
    
    def test_process_event_all_types(self, minimal_data):
        """Test _process_event with all event types."""
        engine = BacktestEngine()
        engine._initialize_backtest(minimal_data, Mock())
        
        # Test each event type
        market_event = MarketEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            open=100, high=101, low=99, close=100.5, volume=1000
        )
        engine._process_event(market_event)
        assert engine.processed_events == 1
        
        signal_event = SignalEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            signal_type='LONG',
            strength=0.8
        )
        engine._process_event(signal_event)
        assert engine.processed_events == 2
        
        order_event = OrderEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            order_type='MARKET',
            quantity=100,
            direction='BUY'
        )
        engine._process_event(order_event)
        assert engine.processed_events == 3
        
        fill_event = FillEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            quantity=100,
            direction='BUY',
            fill_price=100.0,
            commission=1.0,
            slippage=0.1
        )
        engine._process_event(fill_event)
        assert engine.processed_events == 4
    
    def test_handle_market_event_with_signals(self, minimal_data):
        """Test _handle_market_event generating signals."""
        engine = BacktestEngine()
        strategy = Mock()
        engine._initialize_backtest(minimal_data, strategy)
        
        # Mock signal generator to return a signal
        with patch.object(engine.signal_generator, 'generate') as mock_gen:
            mock_gen.return_value = {
                'signal': 1,
                'confidence': 0.8,
                'size': 100
            }
            
            event = MarketEvent(
                timestamp=minimal_data.index[0],
                symbol='TEST',
                open=100, high=101, low=99, close=100.5, volume=1000
            )
            
            engine._handle_market_event(event)
            
            # Check signal was generated
            assert engine.generated_signals == 1
            assert not engine.events_queue.empty()
    
    def test_handle_market_event_check_stops(self, minimal_data):
        """Test _handle_market_event with stop orders."""
        engine = BacktestEngine()
        engine._initialize_backtest(minimal_data, Mock())
        
        # Mock portfolio to return stop orders
        mock_stop_order = Mock()
        mock_stop_order.symbol = 'TEST'
        mock_stop_order.order_type.value = 'STOP'
        mock_stop_order.quantity = 100
        mock_stop_order.direction.value = 'SELL'
        
        with patch.object(engine.portfolio, 'check_stops') as mock_check:
            mock_check.return_value = [mock_stop_order]
            
            event = MarketEvent(
                timestamp=minimal_data.index[0],
                symbol='TEST',
                open=100, high=101, low=99, close=100.5, volume=1000
            )
            
            engine._handle_market_event(event)
            
            # Check stop order was processed
            assert not engine.events_queue.empty()
    
    def test_handle_signal_event_all_branches(self, minimal_data):
        """Test _handle_signal_event with different signal types."""
        engine = BacktestEngine()
        engine._initialize_backtest(minimal_data, Mock())
        
        # Test LONG signal
        signal = SignalEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            signal_type='LONG',
            strength=0.8,
            quantity=100
        )
        engine._handle_signal_event(signal)
        assert not engine.events_queue.empty()
        
        # Test SHORT signal
        signal.signal_type = 'SHORT'
        engine._handle_signal_event(signal)
        
        # Test EXIT signal
        signal.signal_type = 'EXIT'
        engine._handle_signal_event(signal)
        
        # Test with stop_loss and take_profit
        signal.signal_type = 'LONG'
        signal.stop_loss = 95.0
        signal.take_profit = 105.0
        engine._handle_signal_event(signal)
    
    def test_handle_order_event_branches(self, minimal_data):
        """Test _handle_order_event with different order types."""
        engine = BacktestEngine()
        engine._initialize_backtest(minimal_data, Mock())
        
        # Set up portfolio prices
        engine.portfolio.update_prices({'TEST': 100.0})
        
        # Test MARKET order
        order = OrderEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            order_type='MARKET',
            quantity=100,
            direction='BUY'
        )
        engine._handle_order_event(order)
        assert engine.executed_orders == 1
        
        # Test LIMIT order
        order.order_type = 'LIMIT'
        order.price = 99.0
        engine._handle_order_event(order)
        
        # Test STOP order
        order.order_type = 'STOP'
        order.stop_price = 101.0
        engine._handle_order_event(order)
    
    def test_handle_fill_event(self, minimal_data):
        """Test _handle_fill_event."""
        engine = BacktestEngine()
        engine._initialize_backtest(minimal_data, Mock())
        
        fill = FillEvent(
            timestamp=datetime.now(),
            symbol='TEST',
            quantity=100,
            direction='BUY',
            fill_price=100.0,
            commission=1.0,
            slippage=0.1
        )
        
        engine._handle_fill_event(fill)
        
        # Check position was updated
        assert 'TEST' in engine.portfolio.positions
    
    def test_calculate_slippage(self, minimal_data):
        """Test _calculate_slippage method."""
        engine = BacktestEngine(slippage_rate=0.001)
        engine._initialize_backtest(minimal_data, Mock())
        
        # Test BUY side
        slippage = engine._calculate_slippage(100.0, 'BUY')
        assert slippage == 0.1
        
        # Test SELL side
        slippage = engine._calculate_slippage(100.0, 'SELL')
        assert slippage == -0.1
    
    def test_calculate_commission(self, minimal_data):
        """Test _calculate_commission method."""
        engine = BacktestEngine(commission_rate=0.001)
        engine._initialize_backtest(minimal_data, Mock())
        
        commission = engine._calculate_commission(100.0, 100)
        assert commission == 10.0
    
    def test_generate_results(self, minimal_data):
        """Test _generate_results method."""
        engine = BacktestEngine()
        engine._initialize_backtest(minimal_data, Mock())
        
        # Set up some data
        engine.start_time = datetime.now()
        engine.end_time = datetime.now()
        
        # Mock portfolio methods
        with patch.object(engine.portfolio, 'get_equity_curve') as mock_equity:
            mock_equity.return_value = pd.Series([100000, 101000, 102000])
            
            with patch.object(engine.portfolio, 'get_all_trades') as mock_trades:
                mock_trades.return_value = []
                
                with patch.object(engine.portfolio, 'get_positions') as mock_pos:
                    mock_pos.return_value = {}
                    
                    with patch.object(engine.portfolio, 'calculate_metrics') as mock_metrics:
                        mock_metrics.return_value = {
                            'returns': 0.02,
                            'sharpe_ratio': 1.5,
                            'max_drawdown': -0.05,
                            'total_trades': 10
                        }
                        
                        results = engine._generate_results()
                        
                        assert 'metrics' in results
                        assert 'portfolio_value' in results
                        assert 'trades' in results
                        assert 'positions' in results
                        assert 'execution_time' in results
                        assert 'events_processed' in results
    
    def test_generate_standard_report(self, minimal_data, tmp_path):
        """Test _generate_standard_report method."""
        engine = BacktestEngine(
            generate_report=True,
            report_dir=str(tmp_path)
        )
        
        results = {
            'metrics': {'returns': 0.1},
            'portfolio_value': pd.Series([100000, 110000]),
            'trades': [],
            'data': minimal_data
        }
        
        with patch('src.backtesting.engine.StandardReportGenerator') as mock_gen:
            mock_instance = Mock()
            mock_gen.return_value = mock_instance
            
            engine._generate_standard_report(results)
            
            mock_gen.assert_called_once()
            mock_instance.generate_report.assert_called_once()
    
    def test_event_loop_empty_queue(self, minimal_data):
        """Test event loop with queue.Empty exception."""
        engine = BacktestEngine(generate_report=False)
        strategy = StrategyBuilder("Test").build()
        
        # Mock to raise queue.Empty
        with patch.object(engine.events_queue, 'get') as mock_get:
            mock_get.side_effect = queue.Empty
            
            # This should handle the empty queue gracefully
            results = engine.run(
                data=minimal_data[:2],  # Very short data
                strategy=strategy,
                progress_bar=False
            )
            
            assert results is not None