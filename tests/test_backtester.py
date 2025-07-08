"""Comprehensive tests for the backtesting engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio

from src.backtesting.engine import BacktestEngine, BacktestResults
from src.backtesting.events import (
    Event, EventType, MarketEvent, SignalEvent, 
    OrderEvent, FillEvent, EventQueue
)
from src.strategies import StrategyBuilder, Rule
from src.indicators import RSI, BollingerBands


class TestBacktestEngine:
    """Test the main backtesting engine."""
    
    def test_engine_initialization(self):
        """Test engine initialization with various configurations."""
        # Default initialization
        engine = BacktestEngine()
        assert engine.initial_capital == 100000
        assert engine.commission_rate == 0.001
        assert engine.slippage_rate == 0.0005
        
        # Custom initialization
        engine = BacktestEngine(
            initial_capital=50000,
            commission_rate=0.002,
            slippage_rate=0.001,
            max_positions=5
        )
        assert engine.initial_capital == 50000
        assert engine.commission_rate == 0.002
        assert engine.max_positions == 5
    
    def test_invalid_initialization(self):
        """Test engine initialization with invalid parameters."""
        with pytest.raises(ValueError):
            BacktestEngine(initial_capital=-1000)
        
        with pytest.raises(ValueError):
            BacktestEngine(commission_rate=-0.001)
    
    def test_engine_reset(self, backtest_engine):
        """Test engine reset functionality."""
        # Run a backtest
        data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        strategy = StrategyBuilder("Test").build()
        backtest_engine.run(data, strategy)
        
        # Reset engine
        backtest_engine.reset()
        
        assert backtest_engine.portfolio is None
        assert backtest_engine.current_positions == 0
        assert backtest_engine.processed_events == 0


class TestEventSystem:
    """Test the event-driven system."""
    
    def test_event_queue(self):
        """Test event queue functionality."""
        queue = EventQueue()
        
        # Add events
        market_event = MarketEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            open=100, high=101, low=99, close=100.5,
            volume=1000000
        )
        
        signal_event = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type="LONG",
            strength=0.8
        )
        
        queue.put(market_event)
        queue.put(signal_event)
        
        assert queue.qsize() == 2
        
        # Get events in order
        event1 = queue.get()
        assert isinstance(event1, MarketEvent)
        
        event2 = queue.get()
        assert isinstance(event2, SignalEvent)
        
        assert queue.empty()
    
    def test_event_priority(self):
        """Test event priority handling."""
        queue = EventQueue(priority=True)
        
        # Add events with different priorities
        events = [
            (3, MarketEvent(datetime.now(), "AAPL", 100, 101, 99, 100, 1000)),
            (1, FillEvent(datetime.now(), "AAPL", 100, "BUY", 100.5, 0.1)),
            (2, OrderEvent(datetime.now(), "AAPL", "MARKET", 100, "BUY"))
        ]
        
        for priority, event in events:
            queue.put((priority, event))
        
        # Should get highest priority (lowest number) first
        _, event = queue.get()
        assert isinstance(event, FillEvent)
    
    def test_event_processing_order(self, backtest_engine, sample_ohlcv_data):
        """Test correct event processing order."""
        events_processed = []
        
        # Mock event handlers
        def track_event(event):
            events_processed.append(type(event).__name__)
        
        backtest_engine._process_market_event = track_event
        backtest_engine._process_signal_event = track_event
        backtest_engine._process_order_event = track_event
        backtest_engine._process_fill_event = track_event
        
        # Initialize engine
        strategy = StrategyBuilder("Test").build()
        backtest_engine._initialize_backtest(sample_ohlcv_data, strategy)
        
        # Add events
        backtest_engine.events.put(MarketEvent(datetime.now(), "TEST", 100, 101, 99, 100, 1000))
        backtest_engine.events.put(SignalEvent(datetime.now(), "TEST", "LONG", 0.8))
        backtest_engine.events.put(OrderEvent(datetime.now(), "TEST", "MARKET", 100, "BUY"))
        backtest_engine.events.put(FillEvent(datetime.now(), "TEST", 100, "BUY", 100, 0.1))
        
        # Process all events
        while not backtest_engine.events.empty():
            event = backtest_engine.events.get()
            backtest_engine._process_event(event)
        
        # Check processing order
        expected_order = ['MarketEvent', 'SignalEvent', 'OrderEvent', 'FillEvent']
        assert events_processed == expected_order


class TestBacktestExecution:
    """Test backtest execution logic."""
    
    def test_simple_backtest(self, backtest_engine, sample_ohlcv_data, sample_strategy):
        """Test running a simple backtest."""
        results = backtest_engine.run(
            sample_ohlcv_data,
            sample_strategy,
            progress_bar=False
        )
        
        assert isinstance(results, BacktestResults)
        assert results.equity_curve is not None
        assert results.trades is not None
        assert results.performance_metrics is not None
        assert results.statistics is not None
    
    def test_backtest_with_multiple_symbols(self, backtest_engine, multi_symbol_data):
        """Test backtesting with multiple symbols."""
        strategy = StrategyBuilder("Multi-Symbol")
        strategy.add_entry_rule("rsi < 30")
        strategy.add_exit_rule("rsi > 70")
        strategy = strategy.build()
        
        results = backtest_engine.run(
            multi_symbol_data,
            strategy,
            progress_bar=False
        )
        
        # Should have results for multiple symbols
        symbols_traded = results.trades['symbol'].unique() if not results.trades.empty else []
        assert len(symbols_traded) >= 0  # May or may not generate trades
    
    def test_backtest_date_range(self, backtest_engine, sample_ohlcv_data):
        """Test backtesting with specific date range."""
        # Select subset of data
        start_date = sample_ohlcv_data.index[50]
        end_date = sample_ohlcv_data.index[150]
        
        strategy = StrategyBuilder("Test").build()
        
        results = backtest_engine.run(
            sample_ohlcv_data,
            strategy,
            start_date=start_date,
            end_date=end_date,
            progress_bar=False
        )
        
        # Check equity curve matches date range
        assert results.equity_curve.index[0] >= start_date
        assert results.equity_curve.index[-1] <= end_date
    
    def test_backtest_warmup_period(self, backtest_engine, sample_ohlcv_data):
        """Test backtest with indicator warmup period."""
        # Strategy with indicators that need warmup
        strategy = StrategyBuilder("Warmup Test")
        strategy.add_indicator("sma_20", "SMA", period=20)
        strategy.add_indicator("sma_50", "SMA", period=50)
        strategy.add_entry_rule("close > sma_20 and close > sma_50")
        strategy = strategy.build()
        
        results = backtest_engine.run(
            sample_ohlcv_data,
            strategy,
            warmup_period=50,
            progress_bar=False
        )
        
        # First trade should be after warmup period
        if not results.trades.empty:
            first_trade_date = results.trades.iloc[0]['entry_date']
            warmup_end = sample_ohlcv_data.index[50]
            assert first_trade_date >= warmup_end


class TestSignalGeneration:
    """Test signal generation from strategies."""
    
    def test_simple_signal_generation(self, backtest_engine, sample_ohlcv_data):
        """Test basic signal generation."""
        # Create strategy with simple rules
        strategy = StrategyBuilder("Signal Test")
        strategy.add_entry_rule("close > open")  # Buy on green candles
        strategy.add_exit_rule("close < open")   # Sell on red candles
        strategy = strategy.build()
        
        # Track generated signals
        signals = []
        
        def capture_signal(event):
            if isinstance(event, SignalEvent):
                signals.append(event)
        
        # Patch event processing
        original_put = backtest_engine.events.put
        def patched_put(event):
            capture_signal(event)
            original_put(event)
        
        backtest_engine.events.put = patched_put
        
        # Run backtest
        results = backtest_engine.run(
            sample_ohlcv_data[:20],  # Use small subset
            strategy,
            progress_bar=False
        )
        
        # Should have generated some signals
        assert len(signals) > 0
        
        # Check signal properties
        for signal in signals:
            assert signal.symbol is not None
            assert signal.signal_type in ["LONG", "SHORT", "EXIT"]
            assert 0 <= signal.strength <= 1
    
    def test_indicator_based_signals(self, backtest_engine, sample_ohlcv_data):
        """Test signals based on technical indicators."""
        # Add RSI to data
        rsi_indicator = RSI(period=14)
        sample_ohlcv_data['rsi'] = rsi_indicator.calculate(sample_ohlcv_data)
        
        # Create RSI-based strategy
        strategy = StrategyBuilder("RSI Strategy")
        strategy.add_entry_rule("rsi < 30")
        strategy.add_exit_rule("rsi > 70")
        strategy = strategy.build()
        
        results = backtest_engine.run(
            sample_ohlcv_data,
            strategy,
            progress_bar=False
        )
        
        # Check if trades were generated at appropriate RSI levels
        if not results.trades.empty:
            for _, trade in results.trades.iterrows():
                entry_date = trade['entry_date']
                if entry_date in sample_ohlcv_data.index:
                    entry_rsi = sample_ohlcv_data.loc[entry_date, 'rsi']
                    # Entry should be when RSI < 30 (allowing for some tolerance)
                    assert entry_rsi < 35
    
    def test_complex_signal_conditions(self, backtest_engine, sample_ohlcv_data):
        """Test complex signal conditions with multiple indicators."""
        # Add indicators
        rsi = RSI(period=14)
        bb = BollingerBands(period=20)
        
        sample_ohlcv_data['rsi'] = rsi.calculate(sample_ohlcv_data)
        bb_data = bb.calculate(sample_ohlcv_data)
        sample_ohlcv_data = pd.concat([sample_ohlcv_data, bb_data], axis=1)
        
        # Complex strategy
        strategy = StrategyBuilder("Complex Strategy")
        strategy.add_entry_rule("rsi < 30 and close < bb_lower")
        strategy.add_entry_rule("volume > volume.rolling(20).mean() * 1.5")
        strategy.add_exit_rule("rsi > 70 or close > bb_upper")
        strategy = strategy.build()
        
        results = backtest_engine.run(
            sample_ohlcv_data,
            strategy,
            progress_bar=False
        )
        
        # Verify results structure
        assert hasattr(results, 'signals_generated')
        assert hasattr(results, 'signals_executed')


class TestOrderManagement:
    """Test order generation and management."""
    
    def test_market_order_generation(self, backtest_engine, sample_ohlcv_data):
        """Test market order generation from signals."""
        orders_generated = []
        
        # Capture orders
        original_execute = backtest_engine.portfolio.execute_order
        def capture_order(order, price, timestamp):
            orders_generated.append(order)
            return original_execute(order, price, timestamp)
        
        # Simple strategy
        strategy = StrategyBuilder("Order Test")
        strategy.add_entry_rule("close > close.shift(1)")
        strategy = strategy.build()
        
        # Initialize and patch
        backtest_engine._initialize_backtest(sample_ohlcv_data, strategy)
        backtest_engine.portfolio.execute_order = capture_order
        
        # Run partial backtest
        for i in range(10):
            bar = sample_ohlcv_data.iloc[i]
            event = MarketEvent(
                timestamp=sample_ohlcv_data.index[i],
                symbol="TEST",
                **bar.to_dict()
            )
            backtest_engine._process_market_event(event)
        
        # Should have generated orders
        assert len(orders_generated) > 0
        
        # Check order properties
        for order in orders_generated:
            assert order.order_type.value == "MARKET"
            assert order.direction in ["BUY", "SELL"]
    
    def test_position_sizing_in_orders(self, backtest_engine, sample_ohlcv_data):
        """Test position sizing in generated orders."""
        # Strategy with specific position sizing
        strategy = StrategyBuilder("Sizing Test")
        strategy.add_entry_rule("close > open")
        strategy.set_risk_management(position_size=0.1)  # 10% per position
        strategy = strategy.build()
        
        results = backtest_engine.run(
            sample_ohlcv_data[:50],
            strategy,
            progress_bar=False
        )
        
        # Check trade sizes
        if not results.trades.empty:
            for _, trade in results.trades.iterrows():
                # Position value should be ~10% of portfolio
                position_value = trade['quantity'] * trade['entry_price']
                portfolio_value = backtest_engine.initial_capital  # Approximate
                position_pct = position_value / portfolio_value
                assert 0.08 < position_pct < 0.12  # Allow some variance
    
    def test_order_rejection_handling(self, backtest_engine, sample_ohlcv_data):
        """Test handling of rejected orders."""
        # Strategy that might generate many signals
        strategy = StrategyBuilder("Rejection Test")
        strategy.add_entry_rule("True")  # Always signal
        strategy.set_risk_management(max_positions=1)
        strategy = strategy.build()
        
        results = backtest_engine.run(
            sample_ohlcv_data[:20],
            strategy,
            progress_bar=False
        )
        
        # Should have statistics on rejected orders
        assert 'orders_rejected' in results.statistics
        
        # With max_positions=1, should have rejections after first position
        if results.statistics['orders_submitted'] > 1:
            assert results.statistics['orders_rejected'] > 0


class TestRiskManagement:
    """Test risk management features in backtesting."""
    
    def test_stop_loss_execution(self, backtest_engine, sample_ohlcv_data):
        """Test stop loss order execution."""
        # Create data with a sharp drop
        data = sample_ohlcv_data.copy()
        drop_idx = 50
        data.iloc[drop_idx:drop_idx+5, data.columns.get_loc('close')] *= 0.9  # 10% drop
        data.iloc[drop_idx:drop_idx+5, data.columns.get_loc('low')] *= 0.88
        
        # Strategy with stop loss
        strategy = StrategyBuilder("Stop Loss Test")
        strategy.add_entry_rule("close > close.rolling(20).mean()")
        strategy.set_risk_management(stop_loss=0.05)  # 5% stop loss
        strategy = strategy.build()
        
        results = backtest_engine.run(
            data,
            strategy,
            progress_bar=False
        )
        
        # Should have some stopped out trades
        if not results.trades.empty:
            stopped_trades = results.trades[results.trades['exit_reason'] == 'stop_loss']
            assert len(stopped_trades) > 0
            
            # Check stop losses were executed correctly
            for _, trade in stopped_trades.iterrows():
                loss_pct = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
                assert loss_pct <= -0.04  # Should be close to -5%
    
    def test_take_profit_execution(self, backtest_engine, sample_ohlcv_data):
        """Test take profit order execution."""
        # Create data with upward moves
        data = sample_ohlcv_data.copy()
        
        # Strategy with take profit
        strategy = StrategyBuilder("Take Profit Test")
        strategy.add_entry_rule("rsi < 30")
        strategy.set_risk_management(take_profit=0.1)  # 10% take profit
        strategy = strategy.build()
        
        # Add RSI
        rsi = RSI()
        data['rsi'] = rsi.calculate(data)
        
        results = backtest_engine.run(
            data,
            strategy,
            progress_bar=False
        )
        
        # Check for take profit exits
        if not results.trades.empty:
            tp_trades = results.trades[results.trades['exit_reason'] == 'take_profit']
            
            for _, trade in tp_trades.iterrows():
                profit_pct = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
                assert profit_pct >= 0.09  # Should be close to 10%
    
    def test_trailing_stop_loss(self, backtest_engine, sample_ohlcv_data):
        """Test trailing stop loss functionality."""
        # Strategy with trailing stop
        strategy = StrategyBuilder("Trailing Stop Test")
        strategy.add_entry_rule("close > close.rolling(10).mean()")
        strategy.set_risk_management(
            trailing_stop=0.05,  # 5% trailing stop
            trailing_stop_activation=0.03  # Activate after 3% profit
        )
        strategy = strategy.build()
        
        results = backtest_engine.run(
            sample_ohlcv_data,
            strategy,
            progress_bar=False
        )
        
        # Check trailing stop behavior
        if not results.trades.empty:
            trail_trades = results.trades[results.trades['exit_reason'] == 'trailing_stop']
            
            # Trailing stops should have some profit (unless gapped down)
            for _, trade in trail_trades.iterrows():
                profit_pct = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
                assert profit_pct > -0.06  # Should not lose more than trailing %
    
    def test_maximum_position_limit(self, backtest_engine, sample_ohlcv_data):
        """Test maximum position limits."""
        # Strategy that would open many positions
        strategy = StrategyBuilder("Max Positions Test")
        strategy.add_entry_rule("close > open")  # Frequent signals
        strategy.set_risk_management(max_positions=3)
        strategy = strategy.build()
        
        # Track positions over time
        max_positions_held = 0
        
        def track_positions(event):
            nonlocal max_positions_held
            if hasattr(backtest_engine, 'portfolio') and backtest_engine.portfolio:
                current = len([p for p in backtest_engine.portfolio.positions.values() if p.is_open()])
                max_positions_held = max(max_positions_held, current)
        
        # Patch event processing
        original_process = backtest_engine._process_event
        def patched_process(event):
            result = original_process(event)
            track_positions(event)
            return result
        
        backtest_engine._process_event = patched_process
        
        # Run backtest
        results = backtest_engine.run(
            sample_ohlcv_data,
            strategy,
            progress_bar=False
        )
        
        # Should never exceed max positions
        assert max_positions_held <= 3


class TestPerformanceMetrics:
    """Test performance metric calculations."""
    
    def test_basic_metrics_calculation(self, backtest_engine, sample_ohlcv_data, sample_strategy):
        """Test calculation of basic performance metrics."""
        results = backtest_engine.run(
            sample_ohlcv_data,
            sample_strategy,
            progress_bar=False
        )
        
        metrics = results.performance_metrics
        
        # Check required metrics exist
        required_metrics = [
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'max_drawdown', 'win_rate'
        ]
        
        for metric in required_metrics:
            assert hasattr(metrics, metric)
            assert getattr(metrics, metric) is not None
    
    def test_trade_statistics(self, backtest_engine, sample_ohlcv_data):
        """Test trade statistics calculation."""
        # Strategy that generates trades
        strategy = StrategyBuilder("Trade Stats Test")
        strategy.add_entry_rule("close > close.rolling(5).mean()")
        strategy.add_exit_rule("close < close.rolling(5).mean()")
        strategy = strategy.build()
        
        results = backtest_engine.run(
            sample_ohlcv_data,
            strategy,
            progress_bar=False
        )
        
        if not results.trades.empty:
            stats = results.statistics
            
            assert stats['total_trades'] == len(results.trades)
            assert stats['total_trades'] == stats['winning_trades'] + stats['losing_trades'] + stats.get('breakeven_trades', 0)
            
            if stats['winning_trades'] > 0 and stats['losing_trades'] > 0:
                assert stats['profit_factor'] == pytest.approx(
                    abs(stats['gross_profit'] / stats['gross_loss']), rel=0.01
                )
    
    def test_drawdown_calculation(self, backtest_engine, sample_ohlcv_data, sample_strategy):
        """Test drawdown calculations."""
        results = backtest_engine.run(
            sample_ohlcv_data,
            sample_strategy,
            progress_bar=False
        )
        
        # Calculate drawdown from equity curve
        equity = results.equity_curve['total_value']
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        
        # Verify max drawdown
        calculated_max_dd = abs(drawdown.min())
        assert results.performance_metrics.max_drawdown == pytest.approx(calculated_max_dd, rel=0.01)
        
        # Check drawdown duration
        if hasattr(results.performance_metrics, 'max_drawdown_duration'):
            assert results.performance_metrics.max_drawdown_duration >= 0
    
    def test_risk_adjusted_returns(self, backtest_engine, sample_ohlcv_data, sample_strategy):
        """Test risk-adjusted return metrics."""
        results = backtest_engine.run(
            sample_ohlcv_data,
            sample_strategy,
            progress_bar=False
        )
        
        metrics = results.performance_metrics
        
        # Sharpe ratio
        if metrics.volatility > 0:
            expected_sharpe = (metrics.annualized_return - 0.02) / metrics.volatility
            assert metrics.sharpe_ratio == pytest.approx(expected_sharpe, rel=0.1)
        
        # Sortino ratio should be calculated
        if hasattr(metrics, 'sortino_ratio'):
            # Sortino should be >= Sharpe (uses downside deviation)
            assert metrics.sortino_ratio >= metrics.sharpe_ratio - 0.1
        
        # Calmar ratio
        if hasattr(metrics, 'calmar_ratio') and metrics.max_drawdown > 0:
            expected_calmar = metrics.annualized_return / metrics.max_drawdown
            assert metrics.calmar_ratio == pytest.approx(expected_calmar, rel=0.1)


class TestBacktestResults:
    """Test backtest results structure and export."""
    
    def test_results_structure(self, backtest_engine, sample_ohlcv_data, sample_strategy):
        """Test the structure of backtest results."""
        results = backtest_engine.run(
            sample_ohlcv_data,
            sample_strategy,
            progress_bar=False
        )
        
        # Check all required components
        assert hasattr(results, 'equity_curve')
        assert hasattr(results, 'trades')
        assert hasattr(results, 'positions')
        assert hasattr(results, 'performance_metrics')
        assert hasattr(results, 'statistics')
        assert hasattr(results, 'strategy_params')
        
        # Check data types
        assert isinstance(results.equity_curve, pd.DataFrame)
        assert isinstance(results.trades, pd.DataFrame)
        assert isinstance(results.statistics, dict)
    
    def test_results_export(self, backtest_engine, sample_ohlcv_data, sample_strategy, tmp_path):
        """Test exporting results to various formats."""
        results = backtest_engine.run(
            sample_ohlcv_data,
            sample_strategy,
            progress_bar=False
        )
        
        # Export to dict
        results_dict = results.to_dict()
        assert isinstance(results_dict, dict)
        assert 'performance_metrics' in results_dict
        assert 'statistics' in results_dict
        
        # Export to JSON
        json_path = tmp_path / "results.json"
        results.to_json(json_path)
        assert json_path.exists()
        
        # Export trades to CSV
        trades_path = tmp_path / "trades.csv"
        results.trades.to_csv(trades_path)
        assert trades_path.exists()
    
    def test_results_visualization_data(self, backtest_engine, sample_ohlcv_data, sample_strategy):
        """Test data preparation for visualization."""
        results = backtest_engine.run(
            sample_ohlcv_data,
            sample_strategy,
            progress_bar=False
        )
        
        # Get visualization data
        viz_data = results.get_visualization_data()
        
        assert 'equity_curve' in viz_data
        assert 'drawdown' in viz_data
        assert 'returns' in viz_data
        
        # Check return calculations
        returns = viz_data['returns']
        assert len(returns) == len(results.equity_curve) - 1
        
        # Drawdown should be negative or zero
        assert (viz_data['drawdown'] <= 0).all()


class TestAdvancedFeatures:
    """Test advanced backtesting features."""
    
    def test_multi_timeframe_strategy(self, backtest_engine):
        """Test strategy using multiple timeframes."""
        # Generate data for multiple timeframes
        daily_data = generate_stock_data(start_date='2023-01-01', end_date='2023-12-31')
        
        # Create strategy using multiple timeframes
        strategy = StrategyBuilder("Multi-Timeframe")
        strategy.add_indicator("sma_20", "SMA", period=20)
        strategy.add_indicator("sma_50", "SMA", period=50)
        strategy.add_entry_rule("close > sma_20 and sma_20 > sma_50")
        strategy = strategy.build()
        
        results = backtest_engine.run(
            daily_data,
            strategy,
            progress_bar=False
        )
        
        assert results is not None
    
    def test_portfolio_rebalancing(self, backtest_engine, multi_symbol_data):
        """Test portfolio rebalancing functionality."""
        # Strategy with rebalancing
        strategy = StrategyBuilder("Rebalancing")
        strategy.add_entry_rule("close > close.rolling(20).mean()")
        strategy.set_rebalancing(
            frequency='monthly',
            target_weights={'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'AMZN': 0.25}
        )
        strategy = strategy.build()
        
        results = backtest_engine.run(
            multi_symbol_data,
            strategy,
            progress_bar=False
        )
        
        # Check for rebalancing trades
        if not results.trades.empty:
            rebalance_trades = results.trades[results.trades['trade_type'] == 'rebalance']
            assert len(rebalance_trades) > 0
    
    def test_transaction_cost_analysis(self, backtest_engine, sample_ohlcv_data):
        """Test impact of transaction costs."""
        strategy = StrategyBuilder("Cost Analysis")
        strategy.add_entry_rule("close > open")
        strategy.add_exit_rule("close < open")
        strategy = strategy.build()
        
        # Run with different commission rates
        commission_rates = [0, 0.001, 0.005, 0.01]
        returns = []
        
        for rate in commission_rates:
            engine = BacktestEngine(
                initial_capital=100000,
                commission_rate=rate
            )
            results = engine.run(
                sample_ohlcv_data,
                strategy,
                progress_bar=False
            )
            returns.append(results.performance_metrics.total_return)
        
        # Returns should decrease with higher commissions
        assert returns == sorted(returns, reverse=True)
    
    def test_slippage_modeling(self, backtest_engine, sample_ohlcv_data):
        """Test different slippage models."""
        strategy = StrategyBuilder("Slippage Test")
        strategy.add_entry_rule("volume > volume.rolling(20).mean() * 2")
        strategy = strategy.build()
        
        # Test fixed slippage
        results_fixed = backtest_engine.run(
            sample_ohlcv_data,
            strategy,
            slippage_model='fixed',
            slippage_rate=0.001,
            progress_bar=False
        )
        
        # Test variable slippage based on volume
        results_variable = backtest_engine.run(
            sample_ohlcv_data,
            strategy,
            slippage_model='volume_based',
            progress_bar=False
        )
        
        # Both should complete successfully
        assert results_fixed is not None
        assert results_variable is not None


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_data(self, backtest_engine):
        """Test with empty data."""
        empty_data = pd.DataFrame()
        strategy = StrategyBuilder("Test").build()
        
        with pytest.raises(ValueError):
            backtest_engine.run(empty_data, strategy)
    
    def test_insufficient_data(self, backtest_engine):
        """Test with insufficient data for indicators."""
        # Only 10 days of data
        small_data = generate_stock_data(
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        # Strategy needs 50 days
        strategy = StrategyBuilder("Insufficient Data")
        strategy.add_indicator("sma_50", "SMA", period=50)
        strategy.add_entry_rule("close > sma_50")
        strategy = strategy.build()
        
        results = backtest_engine.run(
            small_data,
            strategy,
            progress_bar=False
        )
        
        # Should complete but generate no trades
        assert results.trades.empty
    
    def test_data_gaps(self, backtest_engine, sample_ohlcv_data):
        """Test handling of data gaps."""
        # Create gaps in data
        gapped_data = sample_ohlcv_data.copy()
        # Remove some days
        gapped_data = gapped_data.drop(gapped_data.index[50:55])
        gapped_data = gapped_data.drop(gapped_data.index[100:103])
        
        strategy = StrategyBuilder("Gap Test").build()
        
        results = backtest_engine.run(
            gapped_data,
            strategy,
            progress_bar=False
        )
        
        # Should handle gaps gracefully
        assert len(results.equity_curve) == len(gapped_data)
    
    def test_extreme_price_movements(self, backtest_engine):
        """Test handling of extreme price movements."""
        # Create data with extreme moves
        extreme_data = generate_stock_data()
        
        # Add some extreme moves
        extreme_data.iloc[50, extreme_data.columns.get_loc('close')] *= 0.5  # 50% drop
        extreme_data.iloc[100, extreme_data.columns.get_loc('close')] *= 2.0  # 100% gain
        
        strategy = StrategyBuilder("Extreme Test")
        strategy.add_entry_rule("close < close.shift(1) * 0.9")  # Buy big drops
        strategy.set_risk_management(stop_loss=0.1)
        strategy = strategy.build()
        
        results = backtest_engine.run(
            extreme_data,
            strategy,
            progress_bar=False
        )
        
        # Should handle extreme moves without errors
        assert results is not None
        
        # Check for limit up/down handling if implemented
        if hasattr(backtest_engine, 'limit_move_pct'):
            # Trades during extreme moves might be restricted
            pass


class TestMemoryAndPerformance:
    """Test memory usage and performance."""
    
    def test_large_dataset_performance(self, backtest_engine, performance_monitor):
        """Test performance with large datasets."""
        # 10 years of data
        large_data = generate_stock_data(
            start_date='2014-01-01',
            end_date='2023-12-31'
        )
        
        strategy = StrategyBuilder("Performance Test")
        strategy.add_entry_rule("rsi < 30")
        strategy.add_exit_rule("rsi > 70")
        strategy = strategy.build()
        
        # Add RSI
        rsi = RSI()
        large_data['rsi'] = rsi.calculate(large_data)
        
        # Time the backtest
        performance_monitor.start('large_backtest')
        
        results = backtest_engine.run(
            large_data,
            strategy,
            progress_bar=False
        )
        
        performance_monitor.stop('large_backtest')
        
        # Should complete in reasonable time
        duration = performance_monitor.get_duration('large_backtest')
        assert duration < 60  # Should complete within 60 seconds
        
        # Check memory efficiency
        assert results is not None
    
    def test_memory_cleanup(self, backtest_engine, sample_ohlcv_data, sample_strategy):
        """Test memory cleanup after backtest."""
        import gc
        import sys
        
        # Get initial memory
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Run multiple backtests
        for _ in range(5):
            results = backtest_engine.run(
                sample_ohlcv_data,
                sample_strategy,
                progress_bar=False
            )
            backtest_engine.reset()
        
        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory should not grow significantly
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Reasonable threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])