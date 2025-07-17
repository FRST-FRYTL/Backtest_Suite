"""Comprehensive tests for Order class to achieve 100% coverage."""

import pytest
from datetime import datetime, timedelta
from src.backtesting.order import Order, OrderType, OrderSide, OrderStatus


class TestOrderEnums:
    """Test order enumerations."""
    
    def test_order_type_enum(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP.value == "STOP"
        assert OrderType.STOP_LIMIT.value == "STOP_LIMIT"
        assert OrderType.TRAILING_STOP.value == "TRAILING_STOP"
        
        # Test all enum values
        all_types = list(OrderType)
        assert len(all_types) == 5
        
    def test_order_side_enum(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"
        
        # Test all enum values
        all_sides = list(OrderSide)
        assert len(all_sides) == 2
        
    def test_order_status_enum(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "PENDING"
        assert OrderStatus.SUBMITTED.value == "SUBMITTED"
        assert OrderStatus.PARTIAL.value == "PARTIAL"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELLED.value == "CANCELLED"
        assert OrderStatus.REJECTED.value == "REJECTED"
        
        # Test all enum values
        all_statuses = list(OrderStatus)
        assert len(all_statuses) == 6


class TestOrderCreation:
    """Test Order class creation and initialization."""
    
    def test_basic_order_creation(self):
        """Test creating a basic order."""
        now = datetime.now()
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=now
        )
        
        assert order.order_id == "TEST-001"
        assert order.symbol == "AAPL"
        assert order.quantity == 100
        assert order.direction == "BUY"
        assert order.order_type == OrderType.MARKET
        assert order.created_time == now
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0
        assert order.avg_fill_price == 0.0
        assert order.commission == 0.0
        assert order.time_in_force == "DAY"
        assert order.notes == ""
        assert order.rejection_reason is None
        
    def test_limit_order_creation(self):
        """Test creating a limit order with price."""
        order = Order(
            order_id="TEST-002",
            symbol="GOOGL",
            quantity=50,
            direction="SELL",
            order_type=OrderType.LIMIT,
            created_time=datetime.now(),
            price=2500.0,
            limit_price=2500.0  # Test alias
        )
        
        assert order.price == 2500.0
        assert order.limit_price == 2500.0
        
    def test_stop_order_creation(self):
        """Test creating a stop order."""
        order = Order(
            order_id="TEST-003",
            symbol="MSFT",
            quantity=75,
            direction="SELL",
            order_type=OrderType.STOP,
            created_time=datetime.now(),
            stop_price=350.0
        )
        
        assert order.stop_price == 350.0
        
    def test_stop_limit_order_creation(self):
        """Test creating a stop-limit order."""
        order = Order(
            order_id="TEST-004",
            symbol="TSLA",
            quantity=25,
            direction="BUY",
            order_type=OrderType.STOP_LIMIT,
            created_time=datetime.now(),
            stop_price=200.0,
            limit_price=205.0,
            price=205.0  # Test that price and limit_price work
        )
        
        assert order.stop_price == 200.0
        assert order.limit_price == 205.0
        assert order.price == 205.0
        
    def test_trailing_stop_order(self):
        """Test creating a trailing stop order."""
        order = Order(
            order_id="TEST-005",
            symbol="NVDA",
            quantity=40,
            direction="SELL",
            order_type=OrderType.TRAILING_STOP,
            created_time=datetime.now(),
            stop_price=500.0
        )
        
        assert order.order_type == OrderType.TRAILING_STOP
        assert order.stop_price == 500.0
        
    def test_order_with_custom_attributes(self):
        """Test order with all custom attributes."""
        order = Order(
            order_id="TEST-006",
            symbol="AMD",
            quantity=200,
            direction="BUY",
            order_type=OrderType.LIMIT,
            created_time=datetime.now(),
            status=OrderStatus.SUBMITTED,
            price=120.50,
            filled_quantity=50,
            avg_fill_price=120.45,
            commission=2.50,
            time_in_force="GTC",
            notes="Test order with partial fill"
        )
        
        assert order.status == OrderStatus.SUBMITTED
        assert order.filled_quantity == 50
        assert order.avg_fill_price == 120.45
        assert order.commission == 2.50
        assert order.time_in_force == "GTC"
        assert order.notes == "Test order with partial fill"


class TestOrderMethods:
    """Test Order class methods."""
    
    def test_is_buy_method(self):
        """Test is_buy method."""
        buy_order = Order(
            order_id="BUY-001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        assert buy_order.is_buy() is True
        assert buy_order.is_sell() is False
        
        # Test case insensitive
        buy_order.direction = "buy"
        assert buy_order.is_buy() is True
        
        buy_order.direction = "Buy"
        assert buy_order.is_buy() is True
        
    def test_is_sell_method(self):
        """Test is_sell method."""
        sell_order = Order(
            order_id="SELL-001",
            symbol="AAPL",
            quantity=100,
            direction="SELL",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        assert sell_order.is_sell() is True
        assert sell_order.is_buy() is False
        
        # Test case insensitive
        sell_order.direction = "sell"
        assert sell_order.is_sell() is True
        
        sell_order.direction = "Sell"
        assert sell_order.is_sell() is True
        
    def test_is_filled_method(self):
        """Test is_filled method."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # Initially not filled
        assert order.is_filled() is False
        
        # Change status to filled
        order.status = OrderStatus.FILLED
        assert order.is_filled() is True
        
        # Test all other statuses to ensure line 65 is fully covered
        for status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL, 
                      OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            order.status = status
            assert order.is_filled() is False
        
    def test_is_active_method(self):
        """Test is_active method."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # Test active statuses
        order.status = OrderStatus.PENDING
        assert order.is_active() is True
        
        order.status = OrderStatus.SUBMITTED
        assert order.is_active() is True
        
        order.status = OrderStatus.PARTIAL
        assert order.is_active() is True
        
        # Test inactive statuses
        order.status = OrderStatus.FILLED
        assert order.is_active() is False
        
        order.status = OrderStatus.CANCELLED
        assert order.is_active() is False
        
        order.status = OrderStatus.REJECTED
        assert order.is_active() is False
        
    def test_remaining_quantity_method(self):
        """Test remaining_quantity method."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # Initially all remaining - this ensures line 73 executes
        result = order.remaining_quantity()
        assert result == 100
        assert result == order.quantity - order.filled_quantity
        
        # Partially filled
        order.filled_quantity = 30
        result = order.remaining_quantity()
        assert result == 70
        assert result == order.quantity - order.filled_quantity
        
        # Fully filled
        order.filled_quantity = 100
        result = order.remaining_quantity()
        assert result == 0
        assert result == order.quantity - order.filled_quantity
        
        # Over-filled (edge case)
        order.filled_quantity = 120
        result = order.remaining_quantity()
        assert result == -20
        assert result == order.quantity - order.filled_quantity


class TestOrderFillMethod:
    """Test Order fill method in detail."""
    
    def test_fill_simple(self):
        """Test simple order fill."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # Fill the entire order
        order.fill(100, 150.0, 1.0)
        
        assert order.filled_quantity == 100
        assert order.avg_fill_price == 150.0
        assert order.commission == 1.0
        assert order.status == OrderStatus.FILLED
        
    def test_fill_partial(self):
        """Test partial order fills."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # First partial fill
        order.fill(30, 150.0, 0.30)
        
        assert order.filled_quantity == 30
        assert order.avg_fill_price == 150.0
        assert order.commission == 0.30
        assert order.status == OrderStatus.PARTIAL
        
        # Second partial fill at different price
        order.fill(40, 151.0, 0.40)
        
        assert order.filled_quantity == 70
        # Average price: (150*30 + 151*40) / 70 = 150.57...
        assert order.avg_fill_price == pytest.approx(150.571428, rel=1e-5)
        assert order.commission == 0.70
        assert order.status == OrderStatus.PARTIAL
        
        # Final fill
        order.fill(30, 149.0, 0.30)
        
        assert order.filled_quantity == 100
        # Average price: (150*30 + 151*40 + 149*30) / 100 = 150.1
        assert order.avg_fill_price == pytest.approx(150.1, rel=1e-5)
        assert order.commission == 1.0
        assert order.status == OrderStatus.FILLED
        
    def test_fill_partial_with_commission_line_coverage(self):
        """Test partial fill specifically to ensure line 94 (commission addition) is covered."""
        order = Order(
            order_id="TEST-COMM",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # Test partial fill that triggers line 94 - commission addition after status update
        initial_commission = order.commission
        assert initial_commission == 0.0
        
        # Partial fill with commission - this should trigger line 94
        order.fill(50, 100.0, 1.25)
        
        # Verify commission was added (line 94)
        assert order.commission == initial_commission + 1.25
        assert order.status == OrderStatus.PARTIAL
        
        # Another partial fill to ensure commission accumulates
        prev_commission = order.commission
        order.fill(25, 100.0, 0.75)
        
        # Verify commission accumulation
        assert order.commission == prev_commission + 0.75
        assert order.commission == 2.0  # Total: 1.25 + 0.75
        assert order.status == OrderStatus.PARTIAL
        
    def test_fill_zero_commission(self):
        """Test fill with zero commission."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=50,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        order.fill(50, 100.0)  # No commission specified
        
        assert order.commission == 0.0
        assert order.filled_quantity == 50
        assert order.avg_fill_price == 100.0
        
    def test_fill_overfill(self):
        """Test filling more than order quantity."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # Fill more than quantity
        order.fill(120, 150.0, 1.20)
        
        assert order.filled_quantity == 120
        assert order.avg_fill_price == 150.0
        assert order.commission == 1.20
        assert order.status == OrderStatus.FILLED  # Still marked as filled
        
    def test_fill_multiple_times_price_averaging(self):
        """Test complex price averaging with multiple fills."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=1000,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # Multiple fills at different prices
        fills = [
            (100, 100.0, 0.10),
            (200, 101.0, 0.20),
            (300, 99.50, 0.30),
            (150, 102.0, 0.15),
            (250, 100.50, 0.25)
        ]
        
        total_value = 0
        total_quantity = 0
        total_commission = 0
        
        for quantity, price, commission in fills:
            order.fill(quantity, price, commission)
            total_value += quantity * price
            total_quantity += quantity
            total_commission += commission
        
        expected_avg_price = total_value / total_quantity
        
        assert order.filled_quantity == total_quantity
        assert order.avg_fill_price == pytest.approx(expected_avg_price, rel=1e-5)
        assert order.commission == pytest.approx(total_commission, rel=1e-5)
        assert order.status == OrderStatus.FILLED


class TestOrderCancelMethod:
    """Test Order cancel method."""
    
    def test_cancel_active_orders(self):
        """Test cancelling active orders."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.LIMIT,
            created_time=datetime.now(),
            price=150.0
        )
        
        # Cancel pending order
        order.status = OrderStatus.PENDING
        order.cancel()
        assert order.status == OrderStatus.CANCELLED
        
        # Reset and cancel submitted order
        order.status = OrderStatus.SUBMITTED
        order.cancel()
        assert order.status == OrderStatus.CANCELLED
        
        # Reset and cancel partial order
        order.status = OrderStatus.PARTIAL
        order.filled_quantity = 50
        order.cancel()
        assert order.status == OrderStatus.CANCELLED
        
    def test_cancel_inactive_orders(self):
        """Test cancelling inactive orders (should not change status)."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # Try to cancel filled order
        order.status = OrderStatus.FILLED
        order.cancel()
        assert order.status == OrderStatus.FILLED  # Should not change
        
        # Try to cancel rejected order
        order.status = OrderStatus.REJECTED
        order.cancel()
        assert order.status == OrderStatus.REJECTED  # Should not change
        
        # Try to cancel already cancelled order
        order.status = OrderStatus.CANCELLED
        order.cancel()
        assert order.status == OrderStatus.CANCELLED  # Should not change


class TestOrderRejectMethod:
    """Test Order reject method."""
    
    def test_reject_with_reason(self):
        """Test rejecting order with reason."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        rejection_reason = "Insufficient funds"
        order.reject(rejection_reason)
        
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == rejection_reason
        assert order.notes == f"Rejected: {rejection_reason}"
        
    def test_reject_without_reason(self):
        """Test rejecting order without reason."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        order.reject()
        
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == ""
        assert order.notes == ""  # No notes when no reason provided
        
    def test_reject_with_empty_reason(self):
        """Test rejecting order with empty string reason."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        order.reject("")
        
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == ""
        assert order.notes == ""  # No notes for empty reason
        
    def test_reject_multiple_times(self):
        """Test rejecting order multiple times (changing reasons)."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # First rejection
        order.reject("Insufficient funds")
        assert order.rejection_reason == "Insufficient funds"
        assert order.notes == "Rejected: Insufficient funds"
        
        # Second rejection with different reason
        order.reject("Market closed")
        assert order.rejection_reason == "Market closed"
        assert order.notes == "Rejected: Market closed"


class TestOrderProperties:
    """Test Order property methods."""
    
    def test_id_property(self):
        """Test id property (alias for order_id)."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        assert order.id == "TEST-001"
        assert order.id == order.order_id
        
    def test_fill_price_property(self):
        """Test fill_price property (alias for avg_fill_price)."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # Initially zero
        assert order.fill_price == 0.0
        assert order.fill_price == order.avg_fill_price
        
        # After filling
        order.fill(100, 150.0, 1.0)
        assert order.fill_price == 150.0
        assert order.fill_price == order.avg_fill_price


class TestOrderEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_time_in_force_values(self):
        """Test different time in force values."""
        tif_values = ["DAY", "GTC", "IOC", "FOK", "GTD", "EXT"]
        
        for tif in tif_values:
            order = Order(
                order_id=f"TEST-{tif}",
                symbol="AAPL",
                quantity=100,
                direction="BUY",
                order_type=OrderType.LIMIT,
                created_time=datetime.now(),
                price=150.0,
                time_in_force=tif
            )
            assert order.time_in_force == tif
            
    def test_mixed_case_directions(self):
        """Test various case combinations for direction."""
        directions = ["BUY", "buy", "Buy", "SELL", "sell", "Sell"]
        
        for direction in directions:
            order = Order(
                order_id=f"TEST-{direction}",
                symbol="AAPL",
                quantity=100,
                direction=direction,
                order_type=OrderType.MARKET,
                created_time=datetime.now()
            )
            
            if direction.upper() == "BUY":
                assert order.is_buy() is True
                assert order.is_sell() is False
            else:
                assert order.is_buy() is False
                assert order.is_sell() is True
                
    def test_zero_quantity_order(self):
        """Test order with zero quantity."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=0,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        assert order.quantity == 0
        assert order.remaining_quantity() == 0
        
        # Filling zero quantity order
        order.fill(0, 150.0, 0.0)
        assert order.status == OrderStatus.FILLED
        
    def test_negative_quantity_edge_case(self):
        """Test negative quantity (should be allowed for short selling)."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=-100,  # Short sell
            direction="SELL",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        assert order.quantity == -100
        assert order.remaining_quantity() == -100
        
    def test_very_large_numbers(self):
        """Test with very large numbers."""
        order = Order(
            order_id="TEST-001",
            symbol="BRK.A",  # Berkshire Hathaway
            quantity=1,
            direction="BUY",
            order_type=OrderType.LIMIT,
            created_time=datetime.now(),
            price=500000.0  # Very high price
        )
        
        order.fill(1, 500000.0, 10.0)
        
        assert order.avg_fill_price == 500000.0
        assert order.filled_quantity == 1
        assert order.commission == 10.0
        
    def test_precise_decimal_calculations(self):
        """Test precise decimal calculations in fills."""
        order = Order(
            order_id="TEST-001",
            symbol="AAPL",
            quantity=333,  # Odd number
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        # Fill with fractional quantities and prices
        order.fill(111, 150.123, 0.111)
        order.fill(111, 150.456, 0.111)
        order.fill(111, 150.789, 0.111)
        
        expected_avg = (111 * 150.123 + 111 * 150.456 + 111 * 150.789) / 333
        
        assert order.filled_quantity == 333
        assert order.avg_fill_price == pytest.approx(expected_avg, rel=1e-6)
        assert order.commission == pytest.approx(0.333, rel=1e-6)


class TestOrderScenarios:
    """Test realistic order scenarios."""
    
    def test_day_trading_scenario(self):
        """Test a typical day trading scenario."""
        # Morning buy order
        buy_order = Order(
            order_id="DT-001",
            symbol="SPY",
            quantity=100,
            direction="BUY",
            order_type=OrderType.LIMIT,
            created_time=datetime.now().replace(hour=9, minute=30),
            price=440.50,
            time_in_force="DAY"
        )
        
        # Order gets filled in two parts
        buy_order.fill(60, 440.45, 0.60)
        assert buy_order.status == OrderStatus.PARTIAL
        
        buy_order.fill(40, 440.50, 0.40)
        assert buy_order.status == OrderStatus.FILLED
        assert buy_order.avg_fill_price == pytest.approx(440.47, rel=1e-2)
        
        # Afternoon sell order
        sell_order = Order(
            order_id="DT-002",
            symbol="SPY",
            quantity=100,
            direction="SELL",
            order_type=OrderType.LIMIT,
            created_time=datetime.now().replace(hour=14, minute=30),
            price=442.00,
            time_in_force="DAY"
        )
        
        # Quick fill
        sell_order.fill(100, 442.05, 1.00)
        assert sell_order.status == OrderStatus.FILLED
        
        # Calculate profit (simplified)
        buy_cost = buy_order.avg_fill_price * buy_order.quantity + buy_order.commission
        sell_proceeds = sell_order.avg_fill_price * sell_order.quantity - sell_order.commission
        profit = sell_proceeds - buy_cost
        
        assert profit > 0  # Profitable trade
        
    def test_stop_loss_scenario(self):
        """Test stop loss order scenario."""
        # Initial position
        position_order = Order(
            order_id="POS-001",
            symbol="TSLA",
            quantity=50,
            direction="BUY",
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        position_order.fill(50, 200.0, 0.50)
        
        # Place stop loss order
        stop_loss = Order(
            order_id="SL-001",
            symbol="TSLA",
            quantity=50,
            direction="SELL",
            order_type=OrderType.STOP,
            created_time=datetime.now(),
            stop_price=190.0,  # 5% below entry
            notes="Stop loss for position protection"
        )
        
        # Market drops, stop loss triggers
        stop_loss.fill(50, 189.80, 0.50)  # Filled slightly below stop
        
        assert stop_loss.status == OrderStatus.FILLED
        assert stop_loss.avg_fill_price < stop_loss.stop_price  # Slippage
        
    def test_bracket_order_scenario(self):
        """Test bracket order (entry + stop loss + take profit)."""
        # Entry order
        entry = Order(
            order_id="ENTRY-001",
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.LIMIT,
            created_time=datetime.now(),
            price=150.0
        )
        
        # Stop loss order
        stop_loss = Order(
            order_id="SL-001",
            symbol="AAPL",
            quantity=100,
            direction="SELL",
            order_type=OrderType.STOP,
            created_time=datetime.now(),
            stop_price=145.0  # $5 below entry
        )
        
        # Take profit order
        take_profit = Order(
            order_id="TP-001",
            symbol="AAPL",
            quantity=100,
            direction="SELL",
            order_type=OrderType.LIMIT,
            created_time=datetime.now(),
            price=160.0  # $10 above entry
        )
        
        # Entry gets filled
        entry.fill(100, 149.95, 1.00)
        assert entry.status == OrderStatus.FILLED
        
        # Price moves up, take profit hits
        take_profit.fill(100, 160.0, 1.00)
        assert take_profit.status == OrderStatus.FILLED
        
        # Cancel the stop loss since take profit hit
        stop_loss.cancel()
        assert stop_loss.status == OrderStatus.CANCELLED
        
    def test_algorithmic_order_scenario(self):
        """Test algorithmic order execution (VWAP-like)."""
        # Large order to be executed in slices
        algo_order = Order(
            order_id="ALGO-001",
            symbol="MSFT",
            quantity=10000,  # Large order
            direction="BUY",
            order_type=OrderType.LIMIT,
            created_time=datetime.now(),
            price=380.0,
            notes="VWAP execution algorithm"
        )
        
        # Execute in multiple slices throughout the day
        fills = [
            (500, 378.50, 5.00),    # Early morning
            (1500, 379.00, 15.00),  # Morning
            (2000, 379.25, 20.00),  # Mid-morning
            (2500, 379.50, 25.00),  # Late morning
            (1500, 379.75, 15.00),  # Noon
            (1000, 380.00, 10.00),  # Afternoon
            (1000, 379.90, 10.00),  # Late afternoon
        ]
        
        for quantity, price, commission in fills:
            algo_order.fill(quantity, price, commission)
            
        assert algo_order.filled_quantity == 10000
        assert algo_order.status == OrderStatus.FILLED
        assert algo_order.commission == 100.0  # Total commission
        
        # VWAP should be close to the average of fill prices
        avg_price = sum(q * p for q, p, _ in fills) / sum(q for q, _, _ in fills)
        assert algo_order.avg_fill_price == pytest.approx(avg_price, rel=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])