"""Comprehensive tests for backtesting engine to achieve >95% coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.backtesting.engine import BacktestEngine
from src.backtesting.portfolio import Portfolio
from src.backtesting.position import Position
from src.backtesting.order import Order, OrderType, OrderSide
from src.backtesting.events import MarketEvent, OrderEvent, FillEvent
from src.strategies.base import BaseStrategy
from src.strategies.builder import StrategyBuilder


class TestBacktestEngineComprehensive:
    """Comprehensive backtesting engine tests for maximum coverage."""
    
    def test_engine_initialization(self):
        """Test backtesting engine initialization."""
        # Default initialization
        engine = BacktestEngine()
        assert engine.initial_capital == 100000
        assert engine.commission_rate == 0.001
        assert engine.slippage_rate == 0.0005
        assert engine.portfolio is None  # Portfolio is created during run
        assert engine.strategy is None
        assert engine.data is None
        assert engine.events_queue is not None
        assert engine.max_positions == 10
        assert engine.generate_report == True
        
        # Custom initialization
        engine_custom = BacktestEngine(
            initial_capital=50000,
            commission_rate=0.002,
            slippage_rate=0.001,
            max_positions=5,
            generate_report=False
        )
        assert engine_custom.initial_capital == 50000
        assert engine_custom.commission_rate == 0.002
        assert engine_custom.slippage_rate == 0.001
        assert engine_custom.max_positions == 5
        assert engine_custom.generate_report == False
    
    def test_engine_run_method(self, sample_ohlcv_data):
        """Test engine run method."""
        engine = BacktestEngine(generate_report=False)
        
        # Create a simple strategy
        strategy = StrategyBuilder("Test Strategy")
        strategy.add_entry_rule("close > open")
        strategy.add_exit_rule("close < open")
        built_strategy = strategy.build()
        
        # Run backtest
        results = engine.run(
            data=sample_ohlcv_data,
            strategy=built_strategy,
            progress_bar=False
        )
        
        # Verify results
        assert isinstance(results, dict)
        assert 'metrics' in results
        assert 'portfolio_value' in results
        assert 'positions' in results
        assert 'trades' in results
    
    def test_engine_initialization_methods(self, sample_ohlcv_data):
        """Test internal initialization methods."""
        engine = BacktestEngine()
        
        # Create simple strategy
        strategy = StrategyBuilder("Test Strategy")
        strategy.add_entry_rule("rsi < 30")
        strategy.add_exit_rule("rsi > 70")
        built_strategy = strategy.build()
        
        # Test _initialize_backtest
        engine._initialize_backtest(
            data=sample_ohlcv_data,
            strategy=built_strategy
        )
        
        assert engine.portfolio is not None
        assert engine.strategy is not None
        assert engine.data is not None
        assert engine.market_data_index == 0
        assert engine.portfolio.initial_capital == engine.initial_capital
    
    def test_engine_full_backtest_workflow(self, sample_ohlcv_data):
        """Test complete backtest workflow."""
        engine = BacktestEngine(initial_capital=100000, generate_report=False)
        
        # Create strategy
        strategy = StrategyBuilder("Test Strategy")
        strategy.add_entry_rule("close > open")  # Simple rule
        strategy.add_exit_rule("close < open")
        strategy.set_risk_management(position_size=0.1)
        built_strategy = strategy.build()
        
        # Run backtest with date range
        start_date = sample_ohlcv_data.index[10]
        end_date = sample_ohlcv_data.index[-10]
        
        results = engine.run(
            data=sample_ohlcv_data,
            strategy=built_strategy,
            start_date=start_date,
            end_date=end_date,
            progress_bar=False
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'metrics' in results
        assert 'portfolio_value' in results
        assert 'trades' in results
        assert 'positions' in results
        
        # Check portfolio value
        pv = results['portfolio_value']
        assert isinstance(pv, pd.Series)
        assert len(pv) > 0
        
        # Check metrics
        metrics = results['metrics']
        assert 'returns' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
    
    def test_engine_event_processing(self, sample_ohlcv_data):
        """Test event processing system."""
        engine = BacktestEngine()
        engine.setup_data_handler(sample_ohlcv_data)
        
        # Create mock strategy that generates signals
        class MockStrategy(BaseStrategy):
            def __init__(self):
                super().__init__("Mock Strategy")
                self.signal_count = 0
            
            def generate_signals(self, data: pd.DataFrame) -> Dict:
                # Generate a buy signal every 10 bars
                if len(data) % 10 == 0 and self.signal_count < 5:
                    self.signal_count += 1
                    return {
                        'signal': 1,  # Buy signal
                        'confidence': 0.8,
                        'size': 100
                    }
                elif len(data) % 15 == 0:  # Sell signal
                    return {
                        'signal': -1,  # Sell signal
                        'confidence': 0.7,
                        'size': 100
                    }
                return {'signal': 0}  # No signal
        
        engine.setup_strategy(MockStrategy())
        
        # Process events manually
        engine._initialize_backtest()
        
        # Test market event processing
        market_event = MarketEvent(sample_ohlcv_data.index[0])
        engine._process_market_event(market_event)
        
        # Test order event processing
        order_event = OrderEvent(
            symbol='TEST',
            order_type=OrderType.MARKET,
            quantity=100,
            side=OrderSide.BUY,
            timestamp=sample_ohlcv_data.index[0]
        )
        engine._process_order_event(order_event)
        
        # Test fill event processing
        fill_event = FillEvent(
            symbol='TEST',
            quantity=100,
            side=OrderSide.BUY,
            fill_price=100.0,
            commission=0.1,
            timestamp=sample_ohlcv_data.index[0]
        )
        engine._process_fill_event(fill_event)
    
    def test_engine_position_management(self, sample_ohlcv_data):
        """Test position management functionality."""
        engine = BacktestEngine()
        engine.setup_data_handler(sample_ohlcv_data)
        
        # Test opening position
        fill_event = FillEvent(
            symbol='TEST',
            quantity=100,
            side=OrderSide.BUY,
            fill_price=100.0,
            commission=0.1,
            timestamp=sample_ohlcv_data.index[0]
        )
        
        engine._process_fill_event(fill_event)
        
        # Check position was created
        assert 'TEST' in engine.portfolio.positions
        position = engine.portfolio.positions['TEST']
        assert position.quantity == 100
        assert position.avg_price == 100.0
        
        # Test closing position
        close_fill = FillEvent(
            symbol='TEST',
            quantity=100,
            side=OrderSide.SELL,
            fill_price=105.0,
            commission=0.1,
            timestamp=sample_ohlcv_data.index[1]
        )
        
        engine._process_fill_event(close_fill)
        
        # Check position was closed
        position = engine.portfolio.positions['TEST']
        assert position.quantity == 0
    
    def test_engine_risk_management(self, sample_ohlcv_data):
        """Test risk management features."""
        engine = BacktestEngine()
        engine.setup_data_handler(sample_ohlcv_data)
        
        # Test position sizing
        max_position_size = 0.1  # 10% of capital
        order_size = engine._calculate_position_size(
            price=100.0,
            max_position_pct=max_position_size
        )
        
        expected_size = int((engine.initial_capital * max_position_size) / 100.0)
        assert order_size == expected_size
        
        # Test risk limits
        engine.max_portfolio_risk = 0.02  # 2% max risk
        
        # Test stop loss calculation
        stop_price = engine._calculate_stop_loss(
            entry_price=100.0,
            side=OrderSide.BUY,
            stop_pct=0.05
        )
        
        assert stop_price == 95.0  # 5% below entry for long position
        
        # Test take profit calculation
        take_profit = engine._calculate_take_profit(
            entry_price=100.0,
            side=OrderSide.BUY,
            profit_pct=0.10
        )
        
        assert take_profit == 110.0  # 10% above entry for long position
    
    def test_engine_performance_calculation(self, sample_ohlcv_data):
        """Test performance metrics calculation."""
        engine = BacktestEngine()
        engine.setup_data_handler(sample_ohlcv_data)
        
        # Simulate some trades
        trades = [
            {
                'symbol': 'TEST',
                'entry_date': sample_ohlcv_data.index[0],
                'exit_date': sample_ohlcv_data.index[10],
                'entry_price': 100.0,
                'exit_price': 105.0,
                'quantity': 100,
                'side': 'BUY',
                'pnl': 500.0,
                'commission': 0.2,
                'duration': 10
            },
            {
                'symbol': 'TEST',
                'entry_date': sample_ohlcv_data.index[20],
                'exit_date': sample_ohlcv_data.index[30],
                'entry_price': 110.0,
                'exit_price': 108.0,
                'quantity': 100,
                'side': 'BUY',
                'pnl': -200.0,
                'commission': 0.2,
                'duration': 10
            }
        ]
        
        engine.portfolio.trades = trades
        
        # Calculate performance metrics
        metrics = engine._calculate_performance_metrics()
        
        # Check required metrics
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'avg_win' in metrics
        assert 'avg_loss' in metrics
        assert 'profit_factor' in metrics
        assert 'total_trades' in metrics
        
        # Check metric values
        assert metrics['total_trades'] == 2
        assert metrics['win_rate'] == 0.5  # 1 win out of 2 trades
        assert metrics['avg_win'] == 500.0
        assert metrics['avg_loss'] == -200.0
    
    def test_engine_slippage_and_commission(self, sample_ohlcv_data):
        """Test slippage and commission calculations."""
        engine = BacktestEngine(
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        engine.setup_data_handler(sample_ohlcv_data)
        
        # Test commission calculation
        commission = engine._calculate_commission(
            quantity=100,
            price=100.0
        )
        
        assert commission == 10.0  # 100 * 100 * 0.001
        
        # Test slippage calculation
        slippage = engine._calculate_slippage(
            price=100.0,
            side=OrderSide.BUY
        )
        
        assert slippage == 0.05  # 100 * 0.0005
        
        # Test adjusted fill price
        adjusted_price = engine._get_adjusted_fill_price(
            price=100.0,
            side=OrderSide.BUY
        )
        
        expected_price = 100.0 + 0.05  # price + slippage
        assert adjusted_price == expected_price
    
    def test_engine_market_data_handling(self, sample_ohlcv_data):
        """Test market data handling."""
        engine = BacktestEngine()
        engine.setup_data_handler(sample_ohlcv_data)
        
        # Test getting latest bar
        latest_bar = engine.data_handler.get_latest_bar('TEST')
        assert latest_bar is not None
        assert isinstance(latest_bar, pd.Series)
        
        # Test getting multiple bars
        latest_bars = engine.data_handler.get_latest_bars('TEST', N=5)
        assert len(latest_bars) == 5
        assert isinstance(latest_bars, pd.DataFrame)
        
        # Test data iteration
        for i, bar in enumerate(engine.data_handler):
            assert isinstance(bar, pd.Series)
            if i > 5:  # Test first few bars
                break
    
    def test_engine_edge_cases(self, sample_ohlcv_data):
        """Test edge cases and error conditions."""
        engine = BacktestEngine()
        
        # Test running backtest without data
        with pytest.raises(ValueError):
            engine.run_backtest()
        
        # Test running backtest without strategy
        engine.setup_data_handler(sample_ohlcv_data)
        with pytest.raises(ValueError):
            engine.run_backtest()
        
        # Test with empty data
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError):
            engine.setup_data_handler(empty_data)
        
        # Test with insufficient capital
        engine_low_capital = BacktestEngine(initial_capital=100)
        engine_low_capital.setup_data_handler(sample_ohlcv_data)
        
        strategy = StrategyBuilder("Test Strategy")
        strategy.add_entry_rule("close > open")
        strategy.set_risk_management(position_size=1.0)  # 100% of capital
        
        engine_low_capital.setup_strategy(strategy.build())
        
        # Should handle insufficient capital gracefully
        results = engine_low_capital.run_backtest()
        assert isinstance(results, dict)
    
    def test_engine_multi_symbol_support(self, multi_symbol_data):
        """Test multi-symbol backtesting."""
        engine = BacktestEngine()
        engine.setup_data_handler(multi_symbol_data)
        
        # Create multi-symbol strategy
        strategy = StrategyBuilder("Multi Symbol Strategy")
        strategy.add_entry_rule("close > open")
        strategy.add_exit_rule("close < open")
        strategy.set_risk_management(position_size=0.05)  # 5% per symbol
        
        engine.setup_strategy(strategy.build())
        
        # Run backtest
        results = engine.run_backtest()
        
        # Check results include all symbols
        assert isinstance(results, dict)
        assert 'portfolio_value' in results
        assert 'trades' in results
        
        # Check that trades might include different symbols
        trades = results['trades']
        if len(trades) > 0:
            symbols = set(trade.get('symbol', 'UNKNOWN') for trade in trades)
            assert len(symbols) >= 1  # At least one symbol traded
    
    def test_engine_benchmark_comparison(self, sample_ohlcv_data):
        """Test benchmark comparison functionality."""
        engine = BacktestEngine()
        engine.setup_data_handler(sample_ohlcv_data)
        
        # Use the same data as benchmark
        benchmark_data = sample_ohlcv_data.copy()
        engine.setup_benchmark(benchmark_data)
        
        strategy = StrategyBuilder("Benchmark Test Strategy")
        strategy.add_entry_rule("close > open")
        strategy.set_risk_management(position_size=0.1)
        
        engine.setup_strategy(strategy.build())
        
        # Run backtest
        results = engine.run_backtest()
        
        # Check benchmark metrics
        metrics = results['performance_metrics']
        assert 'benchmark_return' in metrics
        assert 'alpha' in metrics
        assert 'beta' in metrics
        assert 'information_ratio' in metrics
    
    def test_engine_transaction_costs(self, sample_ohlcv_data):
        """Test transaction cost impact."""
        # Test with no transaction costs
        engine_no_costs = BacktestEngine(
            commission_rate=0.0,
            slippage_rate=0.0
        )
        engine_no_costs.setup_data_handler(sample_ohlcv_data)
        
        strategy = StrategyBuilder("No Cost Strategy")
        strategy.add_entry_rule("close > open")
        strategy.set_risk_management(position_size=0.1)
        
        engine_no_costs.setup_strategy(strategy.build())
        results_no_costs = engine_no_costs.run_backtest()
        
        # Test with high transaction costs
        engine_high_costs = BacktestEngine(
            commission_rate=0.01,  # 1% commission
            slippage_rate=0.005   # 0.5% slippage
        )
        engine_high_costs.setup_data_handler(sample_ohlcv_data)
        engine_high_costs.setup_strategy(strategy.build())
        results_high_costs = engine_high_costs.run_backtest()
        
        # High cost results should be worse
        no_cost_return = results_no_costs['performance_metrics']['total_return']
        high_cost_return = results_high_costs['performance_metrics']['total_return']
        
        assert high_cost_return <= no_cost_return
    
    def test_engine_walk_forward_analysis(self, sample_ohlcv_data):
        """Test walk-forward analysis capability."""
        engine = BacktestEngine()
        engine.setup_data_handler(sample_ohlcv_data)
        
        # Split data into train/test periods
        split_date = sample_ohlcv_data.index[len(sample_ohlcv_data) // 2]
        
        train_data = sample_ohlcv_data[:split_date]
        test_data = sample_ohlcv_data[split_date:]
        
        # Test in-sample period
        engine.setup_data_handler(train_data)
        
        strategy = StrategyBuilder("Walk Forward Strategy")
        strategy.add_entry_rule("close > open")
        strategy.set_risk_management(position_size=0.1)
        
        engine.setup_strategy(strategy.build())
        
        in_sample_results = engine.run_backtest()
        
        # Test out-of-sample period
        engine.setup_data_handler(test_data)
        out_sample_results = engine.run_backtest()
        
        # Both should produce valid results
        assert isinstance(in_sample_results, dict)
        assert isinstance(out_sample_results, dict)
        
        # Check performance degradation (common in walk-forward)
        in_sample_return = in_sample_results['performance_metrics']['total_return']
        out_sample_return = out_sample_results['performance_metrics']['total_return']
        
        # Both should be numbers (could be positive or negative)
        assert isinstance(in_sample_return, (int, float))
        assert isinstance(out_sample_return, (int, float))


class TestBacktestEngineIntegration:
    """Integration tests for backtesting engine."""
    
    def test_engine_with_real_strategy(self, sample_ohlcv_data):
        """Test engine with realistic trading strategy."""
        engine = BacktestEngine(initial_capital=100000)
        engine.setup_data_handler(sample_ohlcv_data)
        
        # Create RSI mean reversion strategy
        strategy = StrategyBuilder("RSI Mean Reversion")
        strategy.add_entry_rule("rsi < 30")  # Oversold entry
        strategy.add_entry_rule("close > vwap")  # Above VWAP
        strategy.add_exit_rule("rsi > 70")  # Overbought exit
        strategy.add_exit_rule("close < vwap * 0.98")  # Stop loss
        
        strategy.set_risk_management(
            stop_loss=0.05,
            take_profit=0.10,
            position_size=0.1,
            max_positions=3
        )
        
        engine.setup_strategy(strategy.build())
        
        # Run comprehensive backtest
        results = engine.run_backtest()
        
        # Verify comprehensive results
        assert isinstance(results, dict)
        
        # Check all required components
        required_keys = [
            'portfolio_value', 'trades', 'positions',
            'performance_metrics', 'daily_returns'
        ]
        
        for key in required_keys:
            assert key in results
        
        # Check performance metrics completeness
        metrics = results['performance_metrics']
        required_metrics = [
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'max_drawdown', 'win_rate',
            'avg_win', 'avg_loss', 'profit_factor'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_engine_stress_testing(self, sample_ohlcv_data):
        """Test engine under stress conditions."""
        engine = BacktestEngine(initial_capital=10000)  # Low capital
        engine.setup_data_handler(sample_ohlcv_data)
        
        # Aggressive strategy with high turnover
        strategy = StrategyBuilder("Aggressive Strategy")
        strategy.add_entry_rule("close > open")  # Simple entry
        strategy.add_exit_rule("close < open")  # Simple exit
        strategy.set_risk_management(
            position_size=0.5,  # 50% of capital per trade
            max_positions=5     # Allow many positions
        )
        
        engine.setup_strategy(strategy.build())
        
        # Should handle stress gracefully
        results = engine.run_backtest()
        
        assert isinstance(results, dict)
        assert 'portfolio_value' in results
        assert 'performance_metrics' in results
        
        # Check that engine handled capital constraints
        final_value = results['portfolio_value'].iloc[-1]
        assert final_value >= 0  # Should not go negative
    
    def test_engine_performance_benchmarking(self, performance_benchmark_data):
        """Test engine performance with large dataset."""
        engine = BacktestEngine()
        engine.setup_data_handler(performance_benchmark_data)
        
        # Simple but realistic strategy
        strategy = StrategyBuilder("Performance Test Strategy")
        strategy.add_entry_rule("close > open")
        strategy.set_risk_management(position_size=0.1)
        
        engine.setup_strategy(strategy.build())
        
        # Measure execution time
        import time
        start_time = time.time()
        
        results = engine.run_backtest()
        
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert execution_time < 30  # 30 seconds max
        
        # Should produce valid results
        assert isinstance(results, dict)
        assert 'performance_metrics' in results
        
        # Check data volume handling
        assert len(results['portfolio_value']) == len(performance_benchmark_data)
        
        # Performance should be reasonable
        metrics = results['performance_metrics']
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert isinstance(metrics['total_return'], (int, float))
        assert isinstance(metrics['sharpe_ratio'], (int, float))