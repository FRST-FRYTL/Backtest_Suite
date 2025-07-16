"""Working coverage improvements for existing codebase structure."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import actual classes from the codebase
from src.indicators.rsi import RSI
from src.indicators.bollinger import BollingerBands
from src.indicators.vwap import VWAP
from src.indicators.base import Indicator, IndicatorError
from src.backtesting.portfolio import Portfolio
from src.backtesting.order import Order, OrderType, OrderStatus
from src.backtesting.engine import BacktestEngine
from src.strategies.base import TradeAction, Signal, Position as StrategyPosition
from src.strategies.builder import StrategyBuilder


class TestRSIComprehensiveFixed:
    """Comprehensive RSI tests with correct imports."""
    
    def test_rsi_comprehensive_functionality(self, sample_ohlcv_data):
        """Test RSI with comprehensive functionality."""
        rsi = RSI(period=14, overbought=70, oversold=30)
        
        # Test calculation
        result = rsi.calculate(sample_ohlcv_data)
        assert isinstance(result, pd.Series)
        assert result.name == 'rsi'
        assert len(result) == len(sample_ohlcv_data)
        
        # Test bounds
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()
        
        # Test signal generation
        signals = rsi.get_signals(result)
        assert isinstance(signals, pd.DataFrame)
        expected_columns = ['oversold', 'overbought', 'cross_above_oversold', 
                           'cross_below_overbought', 'cross_above_50', 'cross_below_50']
        for col in expected_columns:
            assert col in signals.columns
    
    def test_rsi_divergence_detection(self, sample_ohlcv_data):
        """Test RSI divergence detection."""
        rsi = RSI(period=14)
        rsi_values = rsi.calculate(sample_ohlcv_data)
        prices = sample_ohlcv_data['close']
        
        # Test divergence detection
        divergences = rsi.divergence(prices, rsi_values, window=10)
        assert isinstance(divergences, pd.DataFrame)
        assert 'bearish' in divergences.columns
        assert 'bullish' in divergences.columns
        assert divergences['bearish'].dtype == bool
        assert divergences['bullish'].dtype == bool
    
    def test_rsi_edge_cases(self):
        """Test RSI edge cases."""
        rsi = RSI(period=14)
        
        # Test with constant prices
        constant_data = pd.DataFrame({
            'close': [100] * 20,
            'open': [100] * 20,
            'high': [100] * 20,
            'low': [100] * 20,
            'volume': [1000] * 20
        })
        
        result = rsi.calculate(constant_data)
        # Should fill with 50 (neutral) when no price changes
        assert (result.dropna() == 50).all()
        
        # Test with only gains
        gains_only = pd.DataFrame({
            'close': list(range(100, 120)),
            'open': list(range(100, 120)),
            'high': list(range(100, 120)),
            'low': list(range(100, 120)),
            'volume': [1000] * 20
        })
        
        result_gains = rsi.calculate(gains_only)
        assert isinstance(result_gains, pd.Series)
        assert result_gains.iloc[-1] > 90  # Should be near 100


class TestBollingerBandsComprehensive:
    """Comprehensive Bollinger Bands tests."""
    
    def test_bollinger_bands_calculation(self, sample_ohlcv_data):
        """Test Bollinger Bands calculation."""
        bb = BollingerBands(period=20, std_dev=2.0)
        result = bb.calculate(sample_ohlcv_data)
        
        assert isinstance(result, pd.DataFrame)
        expected_columns = ['middle', 'upper', 'lower', 'bandwidth', 'percent_b']
        for col in expected_columns:
            assert col in result.columns
        
        # Test relationships
        assert (result['upper'] >= result['middle']).all()
        assert (result['lower'] <= result['middle']).all()
        assert (result['bandwidth'] >= 0).all()
    
    def test_bollinger_signals(self, sample_ohlcv_data):
        """Test Bollinger Bands signals."""
        bb = BollingerBands(period=20)
        bands = bb.calculate(sample_ohlcv_data)
        signals = bb.get_signals(sample_ohlcv_data, bands)
        
        assert isinstance(signals, pd.DataFrame)
        expected_signals = ['squeeze', 'expansion', 'upper_breakout', 
                           'lower_breakout', 'mean_reversion_upper', 'mean_reversion_lower']
        for col in expected_signals:
            assert col in signals.columns
            assert signals[col].dtype == bool
    
    def test_bollinger_squeeze_detection(self, sample_ohlcv_data):
        """Test Bollinger Bands squeeze detection."""
        bb = BollingerBands(period=20)
        is_squeeze = bb.detect_squeeze(sample_ohlcv_data)
        
        assert isinstance(is_squeeze, pd.Series)
        assert is_squeeze.dtype == bool


class TestVWAPComprehensive:
    """Comprehensive VWAP tests."""
    
    def test_vwap_calculation(self, sample_ohlcv_data):
        """Test VWAP calculation."""
        vwap = VWAP(period=20)
        result = vwap.calculate(sample_ohlcv_data)
        
        assert isinstance(result, pd.DataFrame)
        expected_columns = ['vwap', 'upper_band', 'lower_band']
        for col in expected_columns:
            assert col in result.columns
        
        # Test relationships
        assert (result['upper_band'] >= result['vwap']).all()
        assert (result['lower_band'] <= result['vwap']).all()
    
    def test_vwap_signals(self, sample_ohlcv_data):
        """Test VWAP signals."""
        vwap = VWAP(period=20)
        vwap_data = vwap.calculate(sample_ohlcv_data)
        signals = vwap.get_signals(sample_ohlcv_data, vwap_data)
        
        assert isinstance(signals, pd.DataFrame)
        expected_columns = ['above_vwap', 'below_vwap', 'cross_above', 'cross_below']
        for col in expected_columns:
            assert col in signals.columns
            assert signals[col].dtype == bool


class TestPortfolioComprehensive:
    """Comprehensive portfolio tests."""
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        portfolio = Portfolio(initial_capital=100000)
        assert portfolio.initial_capital == 100000
        assert portfolio.cash == 100000
        assert portfolio.positions == {}
        assert portfolio.trades == []
    
    def test_portfolio_position_management(self):
        """Test portfolio position management."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Test opening position
        portfolio.open_position(
            symbol='AAPL',
            quantity=100,
            price=150.0,
            side='BUY',
            timestamp=datetime.now()
        )
        
        # Check position creation
        assert 'AAPL' in portfolio.positions
        position = portfolio.positions['AAPL']
        assert position.symbol == 'AAPL'
        assert position.quantity == 100
        assert position.avg_price == 150.0
        
        # Test closing position
        portfolio.close_position(
            symbol='AAPL',
            quantity=100,
            price=160.0,
            timestamp=datetime.now()
        )
        
        # Check position closed
        assert portfolio.positions['AAPL'].quantity == 0
        assert len(portfolio.trades) == 1
        
        # Check trade record
        trade = portfolio.trades[0]
        assert trade['symbol'] == 'AAPL'
        assert trade['entry_price'] == 150.0
        assert trade['exit_price'] == 160.0
        assert trade['pnl'] > 0
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Open positions
        portfolio.open_position('AAPL', 100, 150.0, 'BUY', datetime.now())
        portfolio.open_position('GOOGL', 25, 2000.0, 'BUY', datetime.now())
        
        # Update with market data
        market_data = {'AAPL': 160.0, 'GOOGL': 1950.0}
        portfolio.update_portfolio_value(market_data)
        
        # Check portfolio value
        aapl_value = 100 * 160.0
        googl_value = 25 * 1950.0
        expected_value = portfolio.cash + aapl_value + googl_value
        
        assert abs(portfolio.portfolio_value - expected_value) < 100
    
    def test_portfolio_risk_metrics(self):
        """Test portfolio risk metrics."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Add some trades
        portfolio.trades = [
            {'pnl': 1000, 'duration': 5, 'symbol': 'AAPL'},
            {'pnl': -500, 'duration': 3, 'symbol': 'GOOGL'},
            {'pnl': 750, 'duration': 7, 'symbol': 'MSFT'}
        ]
        
        # Calculate metrics
        metrics = portfolio.calculate_risk_metrics()
        
        # Check metrics exist
        assert isinstance(metrics, dict)
        assert 'win_rate' in metrics
        assert 'total_pnl' in metrics or 'total_return' in metrics
        
        # Check win rate
        wins = sum(1 for trade in portfolio.trades if trade['pnl'] > 0)
        expected_win_rate = wins / len(portfolio.trades)
        assert abs(metrics['win_rate'] - expected_win_rate) < 0.01


class TestOrderManagement:
    """Test order management functionality."""
    
    def test_order_creation(self):
        """Test order creation."""
        order = Order(
            order_id="TEST001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        assert order.order_id == "TEST001"
        assert order.symbol == "AAPL"
        assert order.quantity == 100
        assert order.direction == "BUY"
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING
    
    def test_order_methods(self):
        """Test order methods."""
        order = Order(
            order_id="TEST001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # Test is_buy method
        assert order.is_buy() == True
        
        # Test is_sell method
        assert order.is_sell() == False
        
        # Test sell order
        sell_order = Order(
            order_id="TEST002",
            symbol="AAPL",
            quantity=100,
            direction="SELL",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        assert sell_order.is_buy() == False
        assert sell_order.is_sell() == True
    
    def test_order_filling(self):
        """Test order filling functionality."""
        order = Order(
            order_id="TEST001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # Simulate partial fill
        order.filled_quantity = 50
        order.avg_fill_price = 150.0
        order.status = OrderStatus.PARTIAL
        
        assert order.filled_quantity == 50
        assert order.avg_fill_price == 150.0
        assert order.status == OrderStatus.PARTIAL
        
        # Simulate complete fill
        order.filled_quantity = 100
        order.status = OrderStatus.FILLED
        
        assert order.filled_quantity == 100
        assert order.status == OrderStatus.FILLED


class TestBacktestEngineBasic:
    """Basic backtesting engine tests."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = BacktestEngine()
        assert engine.initial_capital == 100000
        assert engine.commission_rate == 0.001
        assert engine.slippage_rate == 0.0005
        assert engine.portfolio is not None
    
    def test_engine_data_setup(self, sample_ohlcv_data):
        """Test engine data setup."""
        engine = BacktestEngine()
        
        # Test data handler setup
        engine.setup_data_handler(sample_ohlcv_data)
        assert engine.data_handler is not None
        assert hasattr(engine.data_handler, 'get_latest_bar')
    
    def test_engine_strategy_setup(self, sample_ohlcv_data):
        """Test strategy setup."""
        engine = BacktestEngine()
        engine.setup_data_handler(sample_ohlcv_data)
        
        # Create simple strategy
        strategy = StrategyBuilder("Test Strategy")
        strategy.add_entry_rule("rsi < 30")
        strategy.add_exit_rule("rsi > 70")
        built_strategy = strategy.build()
        
        engine.setup_strategy(built_strategy)
        assert engine.strategy is not None
        assert engine.strategy.name == "Test Strategy"


class TestStrategyComponents:
    """Test strategy component functionality."""
    
    def test_signal_creation(self):
        """Test signal creation."""
        signal = Signal(
            timestamp=pd.Timestamp('2023-01-01'),
            action=TradeAction.BUY,
            symbol='AAPL',
            confidence=0.8,
            quantity=100,
            price=150.0
        )
        
        assert signal.timestamp == pd.Timestamp('2023-01-01')
        assert signal.action == TradeAction.BUY
        assert signal.symbol == 'AAPL'
        assert signal.confidence == 0.8
        assert signal.quantity == 100
        assert signal.price == 150.0
        assert signal.metadata == {}
    
    def test_strategy_position(self):
        """Test strategy position."""
        position = StrategyPosition(
            symbol='AAPL',
            quantity=100,
            entry_price=150.0,
            current_price=160.0,
            timestamp=pd.Timestamp('2023-01-01')
        )
        
        assert position.symbol == 'AAPL'
        assert position.quantity == 100
        assert position.entry_price == 150.0
        assert position.current_price == 160.0
        assert position.market_value == 16000.0  # 100 * 160.0
    
    def test_strategy_builder_basic(self):
        """Test basic strategy builder functionality."""
        builder = StrategyBuilder("Test Strategy")
        
        # Test rule addition
        builder.add_entry_rule("rsi < 30")
        builder.add_exit_rule("rsi > 70")
        
        assert len(builder.entry_rules) == 1
        assert len(builder.exit_rules) == 1
        assert builder.entry_rules[0] == "rsi < 30"
        assert builder.exit_rules[0] == "rsi > 70"
        
        # Test risk management
        builder.set_risk_management(
            stop_loss=0.05,
            take_profit=0.10,
            position_size=0.1
        )
        
        assert builder.risk_management['stop_loss'] == 0.05
        assert builder.risk_management['take_profit'] == 0.10
        assert builder.risk_management['position_size'] == 0.1
        
        # Test strategy building
        strategy = builder.build()
        assert strategy is not None
        assert strategy.name == "Test Strategy"


class TestIndicatorIntegration:
    """Integration tests for indicators."""
    
    def test_multiple_indicators_workflow(self, sample_ohlcv_data):
        """Test workflow with multiple indicators."""
        # Initialize indicators
        rsi = RSI(period=14)
        bb = BollingerBands(period=20)
        vwap = VWAP(period=20)
        
        # Calculate indicators
        rsi_values = rsi.calculate(sample_ohlcv_data)
        bb_values = bb.calculate(sample_ohlcv_data)
        vwap_values = vwap.calculate(sample_ohlcv_data)
        
        # Check all calculations completed
        assert isinstance(rsi_values, pd.Series)
        assert isinstance(bb_values, pd.DataFrame)
        assert isinstance(vwap_values, pd.DataFrame)
        
        # Check data lengths match
        assert len(rsi_values) == len(sample_ohlcv_data)
        assert len(bb_values) == len(sample_ohlcv_data)
        assert len(vwap_values) == len(sample_ohlcv_data)
        
        # Generate signals from all indicators
        rsi_signals = rsi.get_signals(rsi_values)
        bb_signals = bb.get_signals(sample_ohlcv_data, bb_values)
        vwap_signals = vwap.get_signals(sample_ohlcv_data, vwap_values)
        
        # Check all signals generated
        assert isinstance(rsi_signals, pd.DataFrame)
        assert isinstance(bb_signals, pd.DataFrame)
        assert isinstance(vwap_signals, pd.DataFrame)
        
        # Create combined signals
        combined_signals = pd.DataFrame(index=sample_ohlcv_data.index)
        combined_signals['rsi_oversold'] = rsi_signals['oversold']
        combined_signals['bb_squeeze'] = bb_signals['squeeze']
        combined_signals['above_vwap'] = vwap_signals['above_vwap']
        
        # Test confluence analysis
        confluence_buy = (
            combined_signals['rsi_oversold'] & 
            combined_signals['above_vwap'] & 
            ~combined_signals['bb_squeeze']
        )
        
        assert isinstance(confluence_buy, pd.Series)
        assert confluence_buy.dtype == bool
        
        # Should have fewer confluence signals than individual signals
        assert confluence_buy.sum() <= combined_signals['rsi_oversold'].sum()
    
    def test_indicator_performance(self, sample_ohlcv_data):
        """Test indicator performance."""
        indicators = [
            RSI(period=14),
            BollingerBands(period=20),
            VWAP(period=20)
        ]
        
        performance_results = {}
        
        for indicator in indicators:
            import time
            start_time = time.time()
            
            result = indicator.calculate(sample_ohlcv_data)
            
            end_time = time.time()
            performance_results[indicator.__class__.__name__] = {
                'calculation_time': end_time - start_time,
                'result_type': type(result).__name__,
                'result_length': len(result)
            }
        
        # All indicators should complete quickly
        for name, perf in performance_results.items():
            assert perf['calculation_time'] < 1.0  # Should complete within 1 second
            assert perf['result_type'] in ['Series', 'DataFrame']
            assert perf['result_length'] > 0
    
    def test_indicator_data_validation(self):
        """Test indicator data validation."""
        rsi = RSI(period=14)
        
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [100, 101, 102],
            'low': [100, 101, 102],
            'volume': [1000, 1000, 1000]
            # Missing 'close' column
        })
        
        with pytest.raises(IndicatorError):
            rsi.calculate(invalid_data)
        
        # Test with empty data
        with pytest.raises(IndicatorError):
            rsi.calculate(pd.DataFrame())


class TestSystemIntegration:
    """System integration tests."""
    
    def test_end_to_end_workflow(self, sample_ohlcv_data):
        """Test end-to-end workflow."""
        # 1. Create strategy
        strategy = StrategyBuilder("E2E Test Strategy")
        strategy.add_entry_rule("rsi < 30")
        strategy.add_exit_rule("rsi > 70")
        strategy.set_risk_management(position_size=0.1)
        built_strategy = strategy.build()
        
        # 2. Initialize backtest engine
        engine = BacktestEngine(initial_capital=100000)
        engine.setup_data_handler(sample_ohlcv_data)
        engine.setup_strategy(built_strategy)
        
        # 3. Run backtest
        results = engine.run_backtest()
        
        # 4. Verify results
        assert isinstance(results, dict)
        assert 'portfolio_value' in results
        assert 'performance_metrics' in results
        
        # Check portfolio value is a time series
        portfolio_value = results['portfolio_value']
        assert isinstance(portfolio_value, pd.Series)
        assert len(portfolio_value) > 0
        
        # Check performance metrics
        metrics = results['performance_metrics']
        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        
        # Portfolio should end with reasonable value
        final_value = portfolio_value.iloc[-1]
        assert final_value > 0
        assert final_value < 1000000  # Reasonable upper bound
    
    def test_multi_asset_workflow(self, multi_symbol_data):
        """Test multi-asset workflow."""
        # Create multi-asset strategy
        strategy = StrategyBuilder("Multi-Asset Strategy")
        strategy.add_entry_rule("rsi < 30")
        strategy.add_exit_rule("rsi > 70")
        strategy.set_risk_management(position_size=0.05)  # 5% per symbol
        built_strategy = strategy.build()
        
        # Initialize engine with multi-symbol data
        engine = BacktestEngine(initial_capital=100000)
        engine.setup_data_handler(multi_symbol_data)
        engine.setup_strategy(built_strategy)
        
        # Run backtest
        results = engine.run_backtest()
        
        # Verify multi-asset results
        assert isinstance(results, dict)
        assert 'portfolio_value' in results
        assert 'trades' in results
        
        # Check that trades might include different symbols
        trades = results['trades']
        if len(trades) > 0:
            symbols = set(trade.get('symbol', 'UNKNOWN') for trade in trades)
            assert len(symbols) >= 1  # At least one symbol traded
        
        # Portfolio should handle multiple assets
        portfolio_value = results['portfolio_value']
        assert isinstance(portfolio_value, pd.Series)
        assert len(portfolio_value) > 0
    
    def test_stress_testing(self, sample_ohlcv_data):
        """Test system under stress conditions."""
        # Create aggressive strategy
        strategy = StrategyBuilder("Stress Test Strategy")
        strategy.add_entry_rule("close > open")  # Very frequent signals
        strategy.add_exit_rule("close < open")
        strategy.set_risk_management(position_size=0.5)  # Large positions
        built_strategy = strategy.build()
        
        # Test with low capital
        engine = BacktestEngine(initial_capital=10000)
        engine.setup_data_handler(sample_ohlcv_data)
        engine.setup_strategy(built_strategy)
        
        # Should handle stress gracefully
        results = engine.run_backtest()
        
        assert isinstance(results, dict)
        assert 'portfolio_value' in results
        
        # System should not crash
        portfolio_value = results['portfolio_value']
        assert isinstance(portfolio_value, pd.Series)
        assert len(portfolio_value) > 0
        
        # Portfolio should remain non-negative
        assert (portfolio_value >= 0).all()


# Run a quick coverage test
if __name__ == "__main__":
    pytest.main([__file__, "-v"])