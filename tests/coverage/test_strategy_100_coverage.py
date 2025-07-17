"""Comprehensive tests for strategy module to achieve 100% coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from src.backtesting.strategy import (
    Strategy, Order, Trade, OrderType, OrderSide
)


class ConcreteStrategy(Strategy):
    """Concrete implementation of Strategy for testing."""
    
    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize strategy."""
        self.initialized = True
        self.init_context = context
    
    def handle_data(self, context: Dict[str, Any], data: pd.DataFrame) -> List[Order]:
        """Handle market data."""
        self.handle_data_called = True
        self.last_context = context
        self.last_data = data
        return self.pending_orders


class TestOrderType:
    """Test OrderType enum."""
    
    def test_order_type_values(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"


class TestOrderSide:
    """Test OrderSide enum."""
    
    def test_order_side_values(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"


class TestOrder:
    """Test Order dataclass."""
    
    def test_order_creation_minimal(self):
        """Test Order creation with minimal fields."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 100
        assert order.price is None
        assert order.stop_price is None
        assert order.order_id is not None
        assert "AAPL" in order.order_id
        assert "buy" in order.order_id
    
    def test_order_creation_full(self):
        """Test Order creation with all fields."""
        timestamp = datetime.now()
        order = Order(
            symbol="GOOGL",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50,
            price=200.0,
            stop_price=195.0,
            timestamp=timestamp,
            order_id="CUSTOM_ID_001"
        )
        
        assert order.symbol == "GOOGL"
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 50
        assert order.price == 200.0
        assert order.stop_price == 195.0
        assert order.timestamp == timestamp
        assert order.order_id == "CUSTOM_ID_001"
    
    def test_order_post_init(self):
        """Test Order __post_init__ generates ID."""
        order = Order(
            symbol="MSFT",
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            quantity=75
        )
        
        # Check auto-generated ID format
        assert order.order_id is not None
        assert "MSFT" in order.order_id
        assert "buy" in order.order_id
        # Should have timestamp format at start
        timestamp_part = order.order_id.split('_')[0]
        assert len(timestamp_part) == 15  # YYYYMMDD_HHMMSS


class TestTrade:
    """Test Trade dataclass."""
    
    def test_trade_creation_minimal(self):
        """Test Trade creation with minimal fields."""
        timestamp = datetime.now()
        trade = Trade(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            timestamp=timestamp
        )
        
        assert trade.symbol == "AAPL"
        assert trade.side == OrderSide.BUY
        assert trade.quantity == 100
        assert trade.price == 150.0
        assert trade.timestamp == timestamp
        assert trade.commission == 0.0
        assert trade.trade_id is not None
    
    def test_trade_creation_full(self):
        """Test Trade creation with all fields."""
        timestamp = datetime.now()
        trade = Trade(
            symbol="GOOGL",
            side=OrderSide.SELL,
            quantity=50,
            price=2000.0,
            timestamp=timestamp,
            commission=10.0,
            trade_id="CUSTOM_TRADE_001"
        )
        
        assert trade.commission == 10.0
        assert trade.trade_id == "CUSTOM_TRADE_001"
    
    def test_trade_value_property(self):
        """Test Trade value property."""
        trade = Trade(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            timestamp=datetime.now()
        )
        
        assert trade.value == 15000.0  # 100 * 150
    
    def test_trade_net_value_property(self):
        """Test Trade net_value property."""
        trade = Trade(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
            commission=10.0
        )
        
        assert trade.net_value == 14990.0  # 15000 - 10
    
    def test_trade_post_init(self):
        """Test Trade __post_init__ generates ID."""
        trade = Trade(
            symbol="TSLA",
            side=OrderSide.SELL,
            quantity=25,
            price=800.0,
            timestamp=datetime.now()
        )
        
        # Check auto-generated ID
        assert trade.trade_id is not None
        assert "TSLA" in trade.trade_id
        assert "sell" in trade.trade_id


class TestStrategy:
    """Test Strategy abstract base class."""
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        strategy = ConcreteStrategy("TestStrategy")
        
        assert strategy.name == "TestStrategy"
        assert strategy.parameters == {}
        assert strategy.current_bar == 0
        assert strategy.bars is None
        assert strategy.positions == {}
        assert strategy.pending_orders == []
        assert strategy.executed_trades == []
        assert strategy.context == {}
    
    def test_strategy_initialization_with_parameters(self):
        """Test strategy initialization with parameters."""
        params = {"fast_period": 10, "slow_period": 20, "threshold": 0.02}
        strategy = ConcreteStrategy("TestStrategy", params)
        
        assert strategy.parameters == params
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            Strategy("AbstractStrategy")
    
    def test_initialize_method(self):
        """Test initialize method."""
        strategy = ConcreteStrategy("TestStrategy")
        context = {"portfolio_value": 100000}
        
        strategy.initialize(context)
        
        assert strategy.initialized == True
        assert strategy.init_context == context
    
    def test_handle_data_method(self):
        """Test handle_data method."""
        strategy = ConcreteStrategy("TestStrategy")
        context = {"current_prices": {"AAPL": 150.0}}
        data = pd.DataFrame({"AAPL": [150.0]})
        
        orders = strategy.handle_data(context, data)
        
        assert strategy.handle_data_called == True
        assert strategy.last_context == context
        assert strategy.last_data.equals(data)
        assert orders == []  # No pending orders initially
    
    def test_before_trading_start(self):
        """Test before_trading_start method."""
        strategy = ConcreteStrategy("TestStrategy")
        context = {"date": "2023-01-01"}
        
        # Should not raise
        strategy.before_trading_start(context)
    
    def test_after_trading_end(self):
        """Test after_trading_end method."""
        strategy = ConcreteStrategy("TestStrategy")
        context = {"date": "2023-01-01"}
        
        # Should not raise
        strategy.after_trading_end(context)
    
    def test_on_order_filled(self):
        """Test on_order_filled method."""
        strategy = ConcreteStrategy("TestStrategy")
        order = Order("AAPL", OrderSide.BUY, OrderType.MARKET, 100)
        trade = Trade("AAPL", OrderSide.BUY, 100, 150.0, datetime.now())
        
        # Should not raise
        strategy.on_order_filled(order, trade)
    
    def test_on_order_canceled(self):
        """Test on_order_canceled method."""
        strategy = ConcreteStrategy("TestStrategy")
        order = Order("AAPL", OrderSide.BUY, OrderType.LIMIT, 100, price=150.0)
        
        # Should not raise
        strategy.on_order_canceled(order)
    
    def test_get_position(self):
        """Test get_position method."""
        strategy = ConcreteStrategy("TestStrategy")
        
        # No position
        assert strategy.get_position("AAPL") == 0
        
        # Add positions
        strategy.positions["AAPL"] = 100
        strategy.positions["GOOGL"] = -50
        
        assert strategy.get_position("AAPL") == 100
        assert strategy.get_position("GOOGL") == -50
        assert strategy.get_position("MSFT") == 0
    
    def test_update_position(self):
        """Test update_position method."""
        strategy = ConcreteStrategy("TestStrategy")
        
        # Add position
        strategy.update_position("AAPL", 100)
        assert strategy.positions["AAPL"] == 100
        
        # Update position
        strategy.update_position("AAPL", 150)
        assert strategy.positions["AAPL"] == 150
        
        # Close position (remove from dict)
        strategy.update_position("AAPL", 0)
        assert "AAPL" not in strategy.positions
        
        # Update non-existent to zero (should not add)
        strategy.update_position("GOOGL", 0)
        assert "GOOGL" not in strategy.positions
    
    def test_order_shares(self):
        """Test order_shares method."""
        strategy = ConcreteStrategy("TestStrategy")
        
        # Buy order
        order = strategy.order_shares("AAPL", 100)
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert len(strategy.pending_orders) == 1
        
        # Sell order
        order = strategy.order_shares("GOOGL", -50, OrderType.LIMIT, limit_price=2000.0)
        assert order.symbol == "GOOGL"
        assert order.side == OrderSide.SELL
        assert order.quantity == 50
        assert order.order_type == OrderType.LIMIT
        assert order.price == 2000.0
        assert len(strategy.pending_orders) == 2
        
        # Stop order
        order = strategy.order_shares("MSFT", 75, OrderType.STOP, stop_price=300.0)
        assert order.stop_price == 300.0
        assert len(strategy.pending_orders) == 3
    
    def test_order_percent(self):
        """Test order_percent method."""
        strategy = ConcreteStrategy("TestStrategy")
        
        # No portfolio value in context
        order = strategy.order_percent("AAPL", 0.1)
        assert order is None
        
        # Add context
        strategy.context = {
            "portfolio_value": 100000,
            "current_prices": {"AAPL": 150.0}
        }
        
        # Order 10% of portfolio
        order = strategy.order_percent("AAPL", 0.1)
        assert order is not None
        assert order.quantity == 66  # 10000 / 150 = 66.67 -> 66
        
        # With existing position
        strategy.positions["AAPL"] = 50
        order = strategy.order_percent("AAPL", 0.1)
        # Target: 10000, Current: 50*150=7500, Need: 2500/150=16.67->16
        assert order.quantity == 16
        
        # No price data
        order = strategy.order_percent("GOOGL", 0.05)
        assert order is None
        
        # Zero quantity (already at target)
        strategy.positions["AAPL"] = 66
        order = strategy.order_percent("AAPL", 0.099)  # Close to current
        assert order is None
    
    def test_order_target_shares(self):
        """Test order_target_shares method."""
        strategy = ConcreteStrategy("TestStrategy")
        
        # No current position
        order = strategy.order_target_shares("AAPL", 100)
        assert order.quantity == 100
        assert order.side == OrderSide.BUY
        
        # With current position
        strategy.positions["AAPL"] = 50
        order = strategy.order_target_shares("AAPL", 100)
        assert order.quantity == 50  # Need 50 more
        
        # Reduce position
        order = strategy.order_target_shares("AAPL", 30)
        assert order.quantity == 20  # Sell 20
        assert order.side == OrderSide.SELL
        
        # Already at target
        strategy.positions["AAPL"] = 100
        order = strategy.order_target_shares("AAPL", 100)
        assert order is None
        
        # Close position
        order = strategy.order_target_shares("AAPL", 0)
        assert order.quantity == 100
        assert order.side == OrderSide.SELL
    
    def test_cancel_order(self):
        """Test cancel_order method."""
        strategy = ConcreteStrategy("TestStrategy")
        
        # Create and add order
        order = strategy.order_shares("AAPL", 100)
        assert len(strategy.pending_orders) == 1
        
        # Cancel order
        result = strategy.cancel_order(order)
        assert result == True
        assert len(strategy.pending_orders) == 0
        
        # Try to cancel non-existent order
        fake_order = Order("GOOGL", OrderSide.BUY, OrderType.MARKET, 50)
        result = strategy.cancel_order(fake_order)
        assert result == False
    
    def test_get_pending_orders(self):
        """Test get_pending_orders method."""
        strategy = ConcreteStrategy("TestStrategy")
        
        # No orders
        assert strategy.get_pending_orders() == []
        
        # Add orders
        order1 = strategy.order_shares("AAPL", 100)
        order2 = strategy.order_shares("GOOGL", 50)
        order3 = strategy.order_shares("AAPL", -50)
        
        # Get all orders
        all_orders = strategy.get_pending_orders()
        assert len(all_orders) == 3
        
        # Filter by symbol
        aapl_orders = strategy.get_pending_orders("AAPL")
        assert len(aapl_orders) == 2
        assert all(order.symbol == "AAPL" for order in aapl_orders)
        
        googl_orders = strategy.get_pending_orders("GOOGL")
        assert len(googl_orders) == 1
        assert googl_orders[0].symbol == "GOOGL"
    
    def test_get_executed_trades(self):
        """Test get_executed_trades method."""
        strategy = ConcreteStrategy("TestStrategy")
        
        # No trades
        assert strategy.get_executed_trades() == []
        
        # Add trades
        trade1 = Trade("AAPL", OrderSide.BUY, 100, 150.0, datetime.now())
        trade2 = Trade("GOOGL", OrderSide.SELL, 50, 2000.0, datetime.now())
        trade3 = Trade("AAPL", OrderSide.SELL, 50, 155.0, datetime.now())
        
        strategy.executed_trades.extend([trade1, trade2, trade3])
        
        # Get all trades
        all_trades = strategy.get_executed_trades()
        assert len(all_trades) == 3
        
        # Filter by symbol
        aapl_trades = strategy.get_executed_trades("AAPL")
        assert len(aapl_trades) == 2
        assert all(trade.symbol == "AAPL" for trade in aapl_trades)
    
    def test_logging_methods(self):
        """Test logging methods."""
        strategy = ConcreteStrategy("TestStrategy")
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            strategy.log_info("Test info message")
            mock_print.assert_called_with("[TestStrategy] INFO: Test info message")
            
            strategy.log_warning("Test warning message")
            mock_print.assert_called_with("[TestStrategy] WARNING: Test warning message")
            
            strategy.log_error("Test error message")
            mock_print.assert_called_with("[TestStrategy] ERROR: Test error message")
    
    def test_parameter_methods(self):
        """Test get_parameter and set_parameter methods."""
        strategy = ConcreteStrategy("TestStrategy", {"param1": 10})
        
        # Get existing parameter
        assert strategy.get_parameter("param1") == 10
        
        # Get non-existent parameter with default
        assert strategy.get_parameter("param2", 20) == 20
        
        # Get non-existent parameter without default
        assert strategy.get_parameter("param3") is None
        
        # Set parameter
        strategy.set_parameter("param2", 30)
        assert strategy.get_parameter("param2") == 30
        
        # Update existing parameter
        strategy.set_parameter("param1", 15)
        assert strategy.get_parameter("param1") == 15
    
    def test_get_state(self):
        """Test get_state method."""
        strategy = ConcreteStrategy("TestStrategy", {"param1": 10})
        strategy.current_bar = 100
        strategy.positions = {"AAPL": 100, "GOOGL": -50}
        strategy.order_shares("MSFT", 75)
        strategy.executed_trades.append(
            Trade("AAPL", OrderSide.BUY, 100, 150.0, datetime.now())
        )
        
        state = strategy.get_state()
        
        assert state["name"] == "TestStrategy"
        assert state["parameters"] == {"param1": 10}
        assert state["current_bar"] == 100
        assert state["positions"] == {"AAPL": 100, "GOOGL": -50}
        assert state["pending_orders_count"] == 1
        assert state["executed_trades_count"] == 1
    
    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""
        strategy = ConcreteStrategy("TestStrategy", {"param1": 10, "param2": 20})
        strategy.positions = {"AAPL": 100}
        
        # Test __str__
        str_repr = str(strategy)
        assert str_repr == "Strategy(name='TestStrategy', parameters={'param1': 10, 'param2': 20})"
        
        # Test __repr__
        repr_str = repr(strategy)
        assert "Strategy(name='TestStrategy'" in repr_str
        assert "parameters={'param1': 10, 'param2': 20}" in repr_str
        assert "positions={'AAPL': 100}" in repr_str
    
    def test_edge_cases(self):
        """Test various edge cases."""
        strategy = ConcreteStrategy("TestStrategy")
        
        # Order with zero amount
        order = strategy.order_shares("AAPL", 0)
        assert order.quantity == 0
        assert order.side == OrderSide.SELL  # Zero is treated as sell
        
        # Very small percent order
        strategy.context = {
            "portfolio_value": 100000,
            "current_prices": {"AAPL": 150.0}
        }
        order = strategy.order_percent("AAPL", 0.00001)  # Very small percent
        # 100000 * 0.00001 = 1, 1/150 = 0.006 -> 0
        assert order is None  # No order created for zero quantity