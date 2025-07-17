"""Comprehensive tests for order module to achieve 100% coverage."""

import pytest
from datetime import datetime

from src.backtesting.order import Order, OrderType, OrderSide, OrderStatus


class TestOrderComplete:
    """Complete test coverage for Order class."""
    
    def test_order_initialization(self):
        """Test order initialization with all parameters."""
        order = Order(
            order_id="TEST001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.LIMIT,
            created_time=datetime.now(),
            status=OrderStatus.PENDING,
            price=150.0,
            stop_price=145.0,
            limit_price=150.0,
            filled_quantity=0,
            avg_fill_price=0.0,
            commission=0.0,
            time_in_force="GTC",
            notes="Test order",
            rejection_reason=None
        )
        
        assert order.order_id == "TEST001"
        assert order.symbol == "AAPL"
        assert order.quantity == 100
        assert order.direction == "BUY"
        assert order.order_type == OrderType.LIMIT
        assert order.status == OrderStatus.PENDING
        assert order.price == 150.0
        assert order.stop_price == 145.0
        assert order.limit_price == 150.0
        assert order.time_in_force == "GTC"
        assert order.notes == "Test order"
    
    def test_is_buy_and_is_sell(self):
        """Test is_buy and is_sell methods."""
        buy_order = Order(
            order_id="BUY001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        sell_order = Order(
            order_id="SELL001",
            symbol="AAPL",
            quantity=100,
            direction="SELL",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # Test is_buy
        assert buy_order.is_buy() == True
        assert sell_order.is_buy() == False
        
        # Test is_sell
        assert buy_order.is_sell() == False
        assert sell_order.is_sell() == True
        
        # Test with lowercase direction
        buy_order.direction = "buy"
        assert buy_order.is_buy() == True
        
        sell_order.direction = "sell"
        assert sell_order.is_sell() == True
    
    def test_is_filled(self):
        """Test is_filled method - covers line 65."""
        order = Order(
            order_id="TEST001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # Test not filled
        order.status = OrderStatus.PENDING
        assert order.is_filled() == False
        
        # Test partially filled
        order.status = OrderStatus.PARTIAL
        assert order.is_filled() == False
        
        # Test filled - This covers line 65
        order.status = OrderStatus.FILLED
        assert order.is_filled() == True
    
    def test_is_active(self):
        """Test is_active method with all statuses."""
        order = Order(
            order_id="TEST001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # Test active statuses
        for status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]:
            order.status = status
            assert order.is_active() == True
        
        # Test inactive statuses
        for status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            order.status = status
            assert order.is_active() == False
    
    def test_remaining_quantity(self):
        """Test remaining_quantity method - covers line 73."""
        order = Order(
            order_id="TEST001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # Test with no fills
        assert order.remaining_quantity() == 100
        
        # Test with partial fill
        order.filled_quantity = 30
        assert order.remaining_quantity() == 70
        
        # Test with complete fill
        order.filled_quantity = 100
        assert order.remaining_quantity() == 0
        
        # Test overfill edge case
        order.filled_quantity = 110
        assert order.remaining_quantity() == -10
    
    def test_fill_method_complete(self):
        """Test fill method with all scenarios."""
        # Test initial fill
        order = Order(
            order_id="TEST001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # First fill
        order.fill(quantity=50, price=150.0, commission=0.5)
        assert order.filled_quantity == 50
        assert order.avg_fill_price == 150.0
        assert order.commission == 0.5
        assert order.status == OrderStatus.PARTIAL
        
        # Second fill with different price
        order.fill(quantity=30, price=151.0, commission=0.3)
        assert order.filled_quantity == 80
        expected_avg = (150.0 * 50 + 151.0 * 30) / 80
        assert abs(order.avg_fill_price - expected_avg) < 0.001
        assert order.commission == 0.8
        assert order.status == OrderStatus.PARTIAL
        
        # Final fill to complete
        order.fill(quantity=20, price=152.0, commission=0.2)
        assert order.filled_quantity == 100
        assert order.status == OrderStatus.FILLED
        assert order.commission == 1.0
    
    def test_fill_partial_status(self):
        """Test fill method sets PARTIAL status - covers line 94."""
        order = Order(
            order_id="TEST001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now(),
            status=OrderStatus.PENDING
        )
        
        # Fill less than total quantity - should set PARTIAL status
        order.fill(quantity=50, price=150.0, commission=0.5)
        assert order.status == OrderStatus.PARTIAL  # This covers line 94
        assert order.filled_quantity == 50
    
    def test_fill_edge_cases(self):
        """Test fill method edge cases."""
        # Test filling from zero
        order = Order(
            order_id="TEST001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now(),
            filled_quantity=0,
            avg_fill_price=0.0
        )
        
        # Single fill that completes order
        order.fill(quantity=100, price=150.0, commission=1.0)
        assert order.filled_quantity == 100
        assert order.avg_fill_price == 150.0
        assert order.status == OrderStatus.FILLED
        
        # Test overfill
        order2 = Order(
            order_id="TEST002",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        order2.fill(quantity=110, price=150.0)  # Overfill
        assert order2.filled_quantity == 110
        assert order2.status == OrderStatus.FILLED
        
        # Test zero quantity fill (edge case)
        order3 = Order(
            order_id="TEST003",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        order3.fill(quantity=0, price=150.0)
        assert order3.filled_quantity == 0
        assert order3.avg_fill_price == 0.0  # No fill, no price
        assert order3.status == OrderStatus.PARTIAL
    
    def test_cancel_method(self):
        """Test cancel method."""
        order = Order(
            order_id="TEST001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.LIMIT,
            created_time=datetime.now()
        )
        
        # Cancel active order
        order.status = OrderStatus.PENDING
        order.cancel()
        assert order.status == OrderStatus.CANCELLED
        
        # Try to cancel already filled order (should not change)
        order.status = OrderStatus.FILLED
        order.cancel()
        assert order.status == OrderStatus.FILLED
        
        # Cancel partially filled order
        order.status = OrderStatus.PARTIAL
        order.cancel()
        assert order.status == OrderStatus.CANCELLED
    
    def test_reject_method(self):
        """Test reject method."""
        order = Order(
            order_id="TEST001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.LIMIT,
            created_time=datetime.now()
        )
        
        # Reject with reason
        order.reject("Insufficient funds")
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == "Insufficient funds"
        assert order.notes == "Rejected: Insufficient funds"
        
        # Reject without reason
        order2 = Order(
            order_id="TEST002",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.LIMIT,
            created_time=datetime.now()
        )
        
        order2.reject()
        assert order2.status == OrderStatus.REJECTED
        assert order2.rejection_reason == ""
        assert order2.notes == ""
    
    def test_property_aliases(self):
        """Test property aliases."""
        order = Order(
            order_id="TEST001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now(),
            avg_fill_price=150.5
        )
        
        # Test id property
        assert order.id == "TEST001"
        assert order.id == order.order_id
        
        # Test fill_price property
        assert order.fill_price == 150.5
        assert order.fill_price == order.avg_fill_price
    
    def test_order_types_and_sides(self):
        """Test OrderType and OrderSide enums."""
        # Test OrderType values
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP.value == "STOP"
        assert OrderType.STOP_LIMIT.value == "STOP_LIMIT"
        assert OrderType.TRAILING_STOP.value == "TRAILING_STOP"
        
        # Test OrderSide values
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"
        
        # Test OrderStatus values
        assert OrderStatus.PENDING.value == "PENDING"
        assert OrderStatus.SUBMITTED.value == "SUBMITTED"
        assert OrderStatus.PARTIAL.value == "PARTIAL"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELLED.value == "CANCELLED"
        assert OrderStatus.REJECTED.value == "REJECTED"
    
    def test_complex_fill_scenarios(self):
        """Test complex fill scenarios for complete coverage."""
        # Test multiple partial fills
        order = Order(
            order_id="TEST001",
            symbol="AAPL",
            quantity=1000,
            direction="BUY",
            order_type=OrderType.LIMIT,
            created_time=datetime.now(),
            price=150.0
        )
        
        # Multiple small fills
        fills = [
            (100, 149.50, 0.10),
            (200, 149.75, 0.20),
            (300, 150.00, 0.30),
            (250, 150.25, 0.25),
            (150, 150.50, 0.15)
        ]
        
        total_value = 0
        total_quantity = 0
        total_commission = 0
        
        for qty, price, comm in fills:
            order.fill(qty, price, comm)
            total_value += qty * price
            total_quantity += qty
            total_commission += comm
            
            if total_quantity < 1000:
                assert order.status == OrderStatus.PARTIAL
            else:
                assert order.status == OrderStatus.FILLED
        
        assert order.filled_quantity == 1000
        assert order.commission == total_commission
        assert abs(order.avg_fill_price - (total_value / total_quantity)) < 0.001