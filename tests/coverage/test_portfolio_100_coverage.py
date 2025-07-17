"""Comprehensive tests for portfolio module to achieve 100% coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.backtesting.portfolio import Portfolio, PortfolioSnapshot
from src.backtesting.position import Position
from src.backtesting.order import Order, OrderType, OrderStatus
from src.backtesting.events import FillEvent


class TestPortfolioSnapshot:
    """Test PortfolioSnapshot dataclass."""
    
    def test_portfolio_snapshot_creation(self):
        """Test PortfolioSnapshot creation with all fields."""
        timestamp = datetime.now()
        positions = {
            'AAPL': {'quantity': 100, 'value': 10000},
            'GOOGL': {'quantity': 50, 'value': 5000}
        }
        
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=50000.0,
            positions_value=15000.0,
            total_value=65000.0,
            unrealized_pnl=1000.0,
            realized_pnl=500.0,
            commission=100.0,
            positions=positions
        )
        
        assert snapshot.timestamp == timestamp
        assert snapshot.cash == 50000.0
        assert snapshot.positions_value == 15000.0
        assert snapshot.total_value == 65000.0
        assert snapshot.unrealized_pnl == 1000.0
        assert snapshot.realized_pnl == 500.0
        assert snapshot.commission == 100.0
        assert snapshot.positions == positions


class TestPortfolio:
    """Test Portfolio class."""
    
    def test_portfolio_initialization_default(self):
        """Test portfolio initialization with default parameters."""
        portfolio = Portfolio()
        
        assert portfolio.initial_capital == 100000.0
        assert portfolio.cash == 100000.0
        assert portfolio.commission_rate == 0.001
        assert portfolio.slippage_rate == 0.0005
        assert portfolio.max_positions == 10
        assert portfolio.min_commission == 0.0
        assert portfolio.commission_structure == {}
        assert portfolio.slippage_model == 'linear'
        assert portfolio.positions == {}
        assert portfolio.closed_positions == []
        assert portfolio.orders == []
        assert portfolio.trades == []
        assert portfolio.history == []
        assert portfolio.total_commission == 0.0
        assert portfolio.total_slippage == 0.0
        assert portfolio.total_trades == 0
        assert portfolio.allow_short == False
    
    def test_portfolio_initialization_custom(self):
        """Test portfolio initialization with custom parameters."""
        commission_structure = {'tier1': 0.001, 'tier2': 0.0008}
        
        portfolio = Portfolio(
            initial_capital=50000.0,
            commission_rate=0.002,
            slippage_rate=0.001,
            max_positions=5,
            min_commission=1.0,
            commission_structure=commission_structure,
            slippage_model='market_impact'
        )
        
        assert portfolio.initial_capital == 50000.0
        assert portfolio.cash == 50000.0
        assert portfolio.commission_rate == 0.002
        assert portfolio.slippage_rate == 0.001
        assert portfolio.max_positions == 5
        assert portfolio.min_commission == 1.0
        assert portfolio.commission_structure == commission_structure
        assert portfolio.slippage_model == 'market_impact'
    
    def test_portfolio_initialization_validation(self):
        """Test portfolio initialization validation."""
        # Test negative initial capital
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            Portfolio(initial_capital=-1000)
        
        # Test zero initial capital
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            Portfolio(initial_capital=0)
        
        # Test negative commission rate
        with pytest.raises(ValueError, match="Commission rate cannot be negative"):
            Portfolio(commission_rate=-0.001)
        
        # Test negative slippage rate
        with pytest.raises(ValueError, match="Slippage rate must be between 0 and 1"):
            Portfolio(slippage_rate=-0.001)
        
        # Test slippage rate >= 1
        with pytest.raises(ValueError, match="Slippage rate must be between 0 and 1"):
            Portfolio(slippage_rate=1.0)
        
        # Test zero max positions
        with pytest.raises(ValueError, match="Max positions must be positive"):
            Portfolio(max_positions=0)
        
        # Test negative max positions
        with pytest.raises(ValueError, match="Max positions must be positive"):
            Portfolio(max_positions=-5)
    
    def test_current_value(self):
        """Test current_value method."""
        portfolio = Portfolio()
        
        # Test with no positions
        assert portfolio.current_value() == 100000.0
        
        # Add positions
        position1 = Position(symbol='AAPL')
        position1.quantity = 100
        position1.entry_price = 100.0
        position1.current_price = 105.0
        portfolio.positions['AAPL'] = position1
        
        position2 = Position(symbol='GOOGL')
        position2.quantity = 50
        position2.entry_price = 200.0
        position2.current_price = 210.0
        portfolio.positions['GOOGL'] = position2
        
        # Current value = cash + (100 * 105) + (50 * 210) = 100000 + 10500 + 10500
        assert portfolio.current_value() == 121000.0
        
        # Test with cash reduced
        portfolio.cash = 80000.0
        assert portfolio.current_value() == 101000.0
    
    def test_get_total_value_with_data(self):
        """Test get_total_value with DataFrame input."""
        portfolio = Portfolio()
        
        # Add positions
        position1 = Position(symbol='AAPL')
        position1.quantity = 100
        position1.entry_price = 100.0
        portfolio.positions['AAPL'] = position1
        
        position2 = Position(symbol='GOOGL')
        position2.quantity = 50
        position2.entry_price = 200.0
        portfolio.positions['GOOGL'] = position2
        
        # Create mock data
        data = pd.DataFrame({
            'AAPL': [110.0],
            'GOOGL': [220.0]
        })
        
        # Mock update_prices
        portfolio.update_prices = Mock()
        
        total_value = portfolio.get_total_value(data)
        
        # Check update_prices was called with correct prices
        portfolio.update_prices.assert_called_once_with({'AAPL': 110.0, 'GOOGL': 220.0})
        
        # Check total value is calculated
        assert total_value == portfolio.current_value()
    
    def test_get_total_value_without_data(self):
        """Test get_total_value without DataFrame input."""
        portfolio = Portfolio()
        portfolio.cash = 90000.0
        
        # Add a position
        position = Position(symbol='AAPL')
        position.quantity = 100
        position.current_price = 150.0
        portfolio.positions['AAPL'] = position
        
        total_value = portfolio.get_total_value()
        assert total_value == 90000.0 + (100 * 150.0)
    
    def test_get_position_weight(self):
        """Test get_position_weight method."""
        portfolio = Portfolio()
        portfolio.cash = 50000.0
        
        # Test with no position
        assert portfolio.get_position_weight('AAPL') == 0.0
        
        # Add position
        position = Position(symbol='AAPL')
        position.quantity = 100
        position.current_price = 500.0  # Position value = 50000
        portfolio.positions['AAPL'] = position
        
        # Weight = 50000 / (50000 + 50000) = 0.5
        assert portfolio.get_position_weight('AAPL') == 0.5
        
        # Test with zero total value edge case
        portfolio.cash = 0.0
        position.quantity = 0
        assert portfolio.get_position_weight('AAPL') == 0.0
    
    def test_unrealized_pnl(self):
        """Test unrealized_pnl calculation."""
        portfolio = Portfolio()
        
        # Test with no positions
        assert portfolio.unrealized_pnl() == 0.0
        
        # Add positions with gains and losses
        position1 = Position(symbol='AAPL')
        position1.quantity = 100
        position1.entry_price = 100.0
        position1.current_price = 110.0  # Gain of 1000
        portfolio.positions['AAPL'] = position1
        
        position2 = Position(symbol='GOOGL')
        position2.quantity = 50
        position2.entry_price = 200.0
        position2.current_price = 190.0  # Loss of 500
        portfolio.positions['GOOGL'] = position2
        
        # Total unrealized = 1000 - 500 = 500
        assert portfolio.unrealized_pnl() == 500.0
    
    def test_realized_pnl(self):
        """Test realized_pnl calculation."""
        portfolio = Portfolio()
        
        # Test with no closed positions
        assert portfolio.realized_pnl() == 0.0
        
        # Add closed positions
        position1 = Position(symbol='AAPL')
        position1.realized_pnl = 1000.0
        portfolio.closed_positions.append(position1)
        
        position2 = Position(symbol='GOOGL')
        position2.realized_pnl = -500.0
        portfolio.closed_positions.append(position2)
        
        # Total realized = 1000 - 500 = 500
        assert portfolio.realized_pnl() == 500.0
    
    def test_total_pnl(self):
        """Test total_pnl calculation."""
        portfolio = Portfolio(initial_capital=100000.0)
        portfolio.cash = 95000.0
        
        # Add position
        position = Position(symbol='AAPL')
        position.quantity = 100
        position.current_price = 100.0
        portfolio.positions['AAPL'] = position
        
        # Total value = 95000 + 10000 = 105000
        # Total PnL = 105000 - 100000 = 5000
        assert portfolio.total_pnl() == 5000.0
    
    def test_return_pct(self):
        """Test return_pct calculation."""
        portfolio = Portfolio(initial_capital=100000.0)
        portfolio.cash = 90000.0
        
        # Add position
        position = Position(symbol='AAPL')
        position.quantity = 100
        position.current_price = 150.0
        portfolio.positions['AAPL'] = position
        
        # Total value = 90000 + 15000 = 105000
        # Return % = (105000 - 100000) / 100000 * 100 = 5%
        assert portfolio.return_pct() == 5.0
    
    def test_position_count(self):
        """Test position_count method."""
        portfolio = Portfolio()
        
        # Test with no positions
        assert portfolio.position_count() == 0
        
        # Add open positions
        position1 = Position(symbol='AAPL')
        position1.quantity = 100
        portfolio.positions['AAPL'] = position1
        
        position2 = Position(symbol='GOOGL')
        position2.quantity = 50
        portfolio.positions['GOOGL'] = position2
        
        assert portfolio.position_count() == 2
        
        # Close one position
        position1.quantity = 0
        assert portfolio.position_count() == 1
    
    def test_can_open_position(self):
        """Test can_open_position method."""
        portfolio = Portfolio(max_positions=2)
        
        # Test with no positions
        assert portfolio.can_open_position() == True
        
        # Add positions up to limit
        position1 = Position(symbol='AAPL')
        position1.quantity = 100
        portfolio.positions['AAPL'] = position1
        
        assert portfolio.can_open_position() == True
        
        position2 = Position(symbol='GOOGL')
        position2.quantity = 50
        portfolio.positions['GOOGL'] = position2
        
        # Now at max positions
        assert portfolio.can_open_position() == False
    
    def test_calculate_position_size_basic(self):
        """Test basic calculate_position_size method."""
        portfolio = Portfolio()
        portfolio.cash = 50000.0
        
        # Test percentage-based sizing
        shares = portfolio.calculate_position_size(
            symbol='AAPL',
            price=100.0,
            position_pct=0.2  # 20% of portfolio
        )
        
        # Position value = 100000 * 0.2 = 20000
        # Shares = 20000 / 100 = 200
        assert shares == 200
        
        # Test risk-based sizing
        shares = portfolio.calculate_position_size(
            symbol='AAPL',
            price=100.0,
            risk_amount=1000.0
        )
        
        # Shares = 1000 / 100 = 10
        assert shares == 10
        
        # Test with insufficient cash
        portfolio.cash = 500.0
        shares = portfolio.calculate_position_size(
            symbol='AAPL',
            price=100.0,
            position_pct=0.5
        )
        
        # Required cash = shares * price * (1 + commission)
        # 500 / (100 * 1.001) ≈ 4.99, so 4 shares
        assert shares == 4
    
    def test_calculate_position_size_enhanced(self):
        """Test enhanced calculate_position_size with multiple methods."""
        portfolio = Portfolio()
        portfolio.cash = 100000.0
        
        # Test Kelly criterion sizing
        shares = portfolio.calculate_position_size(
            symbol='AAPL',
            price=100.0,
            sizing_method='kelly',
            win_rate=0.6,
            avg_win=0.1,
            avg_loss=0.05
        )
        
        # Kelly % = 0.6 - 0.4 / (0.1/0.05) = 0.6 - 0.4/2 = 0.4
        # Shares = 100000 * 0.4 / 100 = 400
        assert shares == 400
        
        # Test risk-based sizing with stop loss
        shares = portfolio.calculate_position_size(
            symbol='AAPL',
            price=100.0,
            sizing_method='risk',
            stop_loss_pct=0.05  # 5% stop loss
        )
        
        # Max risk = 100000 * 0.02 = 2000
        # Risk per share = 100 * 0.05 = 5
        # Shares = 2000 / 5 = 400
        assert shares == 400
        
        # Test with fixed risk amount and stop loss
        shares = portfolio.calculate_position_size(
            symbol='AAPL',
            price=100.0,
            risk_amount=1000.0,
            stop_loss_pct=0.10  # 10% stop loss
        )
        
        # Risk per share = 100 * 0.10 = 10
        # Shares = 1000 / 10 = 100
        assert shares == 100
        
        # Test with max position limit
        shares = portfolio.calculate_position_size(
            symbol='AAPL',
            price=100.0,
            position_pct=0.5,  # 50% position
            max_position_pct=0.25  # Max 25%
        )
        
        # Without limit: 100000 * 0.5 / 100 = 500
        # With limit: 100000 * 0.25 / 100 = 250
        assert shares == 250
        
        # Test with insufficient cash
        portfolio.cash = 1000.0
        shares = portfolio.calculate_position_size(
            symbol='AAPL',
            price=100.0,
            position_pct=0.5
        )
        
        # Available shares considering commission
        # 1000 / (100 * 1.001) ≈ 9.99, so 9 shares
        assert shares == 9
    
    def test_calculate_position_size_edge_cases(self):
        """Test calculate_position_size edge cases."""
        portfolio = Portfolio()
        
        # Test with zero cash
        portfolio.cash = 0.0
        shares = portfolio.calculate_position_size(
            symbol='AAPL',
            price=100.0,
            position_pct=0.1
        )
        assert shares == 0
        
        # Test Kelly with default values
        portfolio.cash = 10000.0
        shares = portfolio.calculate_position_size(
            symbol='AAPL',
            price=100.0,
            sizing_method='kelly'
        )
        # With defaults: win_rate=0.6, avg_win=0.1, avg_loss=0.05
        # Kelly = 0.6 - 0.4/(0.1/0.05) = 0.4
        assert shares == 40  # 10000 * 0.4 / 100
    
    def test_place_order(self):
        """Test place_order method."""
        portfolio = Portfolio()
        
        # Test market order
        order = portfolio.place_order(
            symbol='AAPL',
            quantity=100,
            direction='BUY',
            order_type=OrderType.MARKET
        )
        
        assert order.symbol == 'AAPL'
        assert order.quantity == 100
        assert order.direction == 'BUY'
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.SUBMITTED
        assert len(portfolio.orders) == 1
        assert 'AAPL_' in order.order_id
        
        # Test limit order with price
        order2 = portfolio.place_order(
            symbol='GOOGL',
            quantity=50,
            direction='SELL',
            order_type=OrderType.LIMIT,
            price=150.0
        )
        
        assert order2.order_type == OrderType.LIMIT
        assert order2.price == 150.0
        assert order2.limit_price is None
        
        # Test limit order with limit_price
        order3 = portfolio.place_order(
            symbol='MSFT',
            quantity=75,
            direction='BUY',
            order_type=OrderType.LIMIT,
            limit_price=200.0
        )
        
        assert order3.price == 200.0
        assert order3.limit_price == 200.0
        
        # Test stop order
        order4 = portfolio.place_order(
            symbol='TSLA',
            quantity=25,
            direction='SELL',
            order_type=OrderType.STOP,
            stop_price=500.0
        )
        
        assert order4.order_type == OrderType.STOP
        assert order4.stop_price == 500.0
        assert len(portfolio.orders) == 4
    
    def test_place_order_validation(self):
        """Test place_order validation."""
        portfolio = Portfolio()
        
        # Test zero quantity
        with pytest.raises(ValueError, match="Quantity must be positive"):
            portfolio.place_order('AAPL', 0, 'BUY')
        
        # Test negative quantity
        with pytest.raises(ValueError, match="Quantity must be positive"):
            portfolio.place_order('AAPL', -10, 'BUY')
        
        # Test invalid direction
        with pytest.raises(ValueError, match="Direction must be 'BUY' or 'SELL'"):
            portfolio.place_order('AAPL', 100, 'HOLD')
        
        # Test limit order without price
        with pytest.raises(ValueError, match="Limit orders require a price"):
            portfolio.place_order('AAPL', 100, 'BUY', OrderType.LIMIT)
        
        # Test stop order without stop price
        with pytest.raises(ValueError, match="Stop orders require a stop price"):
            portfolio.place_order('AAPL', 100, 'SELL', OrderType.STOP)
    
    def test_cancel_order(self):
        """Test cancel_order method."""
        portfolio = Portfolio()
        
        # Place orders
        order1 = portfolio.place_order('AAPL', 100, 'BUY')
        order2 = portfolio.place_order('GOOGL', 50, 'SELL')
        
        # Cancel existing order
        assert portfolio.cancel_order(order1.order_id) == True
        assert order1.status == OrderStatus.CANCELLED
        
        # Try to cancel already cancelled order
        assert portfolio.cancel_order(order1.order_id) == False
        
        # Try to cancel non-existent order
        assert portfolio.cancel_order('FAKE_ID') == False
        
        # Fill order2 and try to cancel
        order2.status = OrderStatus.FILLED
        assert portfolio.cancel_order(order2.order_id) == False
    
    def test_execute_order_market_buy(self):
        """Test execute_order for market buy."""
        portfolio = Portfolio()
        portfolio.cash = 20000.0
        
        order = portfolio.place_order('AAPL', 100, 'BUY', OrderType.MARKET)
        
        # Mock _process_fill
        portfolio._process_fill = Mock()
        
        # Execute order
        timestamp = datetime.now()
        result = portfolio.execute_order(order, 100.0, timestamp)
        
        assert result == True
        assert order.status == OrderStatus.FILLED
        
        # Check slippage was applied (buy at higher price)
        expected_fill_price = 100.0 * (1 + portfolio.slippage_rate)
        assert order.avg_fill_price == expected_fill_price
        
        # Check _process_fill was called
        portfolio._process_fill.assert_called_once()
    
    def test_execute_order_market_sell(self):
        """Test execute_order for market sell."""
        portfolio = Portfolio()
        
        # Add position to sell
        position = Position(symbol='AAPL')
        position.quantity = 100
        portfolio.positions['AAPL'] = position
        
        order = portfolio.place_order('AAPL', 50, 'SELL', OrderType.MARKET)
        
        # Mock _process_fill and calculate_commission
        portfolio._process_fill = Mock()
        portfolio.calculate_commission = Mock(return_value=5.0)
        
        # Execute order
        timestamp = datetime.now()
        result = portfolio.execute_order(order, 100.0, timestamp)
        
        assert result == True
        
        # Check slippage was applied (sell at lower price)
        expected_fill_price = 100.0 * (1 - portfolio.slippage_rate)
        assert order.avg_fill_price == expected_fill_price
    
    def test_execute_order_limit_constraints(self):
        """Test execute_order with limit order constraints."""
        portfolio = Portfolio()
        portfolio.cash = 20000.0
        
        # Buy limit order - should not execute above limit
        buy_order = portfolio.place_order(
            'AAPL', 100, 'BUY', OrderType.LIMIT, price=100.0
        )
        
        # Try to execute above limit price
        result = portfolio.execute_order(buy_order, 101.0, datetime.now())
        assert result == False
        assert buy_order.status == OrderStatus.SUBMITTED
        
        # Execute at limit price
        result = portfolio.execute_order(buy_order, 100.0, datetime.now())
        assert result == True
        
        # Sell limit order - should not execute below limit
        position = Position(symbol='GOOGL')
        position.quantity = 50
        portfolio.positions['GOOGL'] = position
        
        sell_order = portfolio.place_order(
            'GOOGL', 50, 'SELL', OrderType.LIMIT, price=200.0
        )
        
        # Try to execute below limit price
        result = portfolio.execute_order(sell_order, 199.0, datetime.now())
        assert result == False
        
        # Execute at limit price
        portfolio._process_fill = Mock()
        result = portfolio.execute_order(sell_order, 200.0, datetime.now())
        assert result == True
    
    def test_execute_order_stop_constraints(self):
        """Test execute_order with stop order constraints."""
        portfolio = Portfolio()
        portfolio.cash = 20000.0
        
        # Stop buy - executes at or above stop price
        stop_buy = portfolio.place_order(
            'AAPL', 100, 'BUY', OrderType.STOP, stop_price=105.0
        )
        
        # Should not execute below stop price
        result = portfolio.execute_order(stop_buy, 104.0, datetime.now())
        assert result == False
        
        # Should execute at stop price
        portfolio._process_fill = Mock()
        result = portfolio.execute_order(stop_buy, 105.0, datetime.now())
        assert result == True
        
        # Stop sell - executes at or below stop price
        position = Position(symbol='GOOGL')
        position.quantity = 50
        portfolio.positions['GOOGL'] = position
        
        stop_sell = portfolio.place_order(
            'GOOGL', 50, 'SELL', OrderType.STOP, stop_price=195.0
        )
        
        # Should not execute above stop price
        result = portfolio.execute_order(stop_sell, 196.0, datetime.now())
        assert result == False
        
        # Should execute at stop price
        portfolio._process_fill = Mock()
        result = portfolio.execute_order(stop_sell, 195.0, datetime.now())
        assert result == True
    
    def test_execute_order_insufficient_cash(self):
        """Test execute_order with insufficient cash."""
        portfolio = Portfolio()
        portfolio.cash = 1000.0
        portfolio.calculate_commission = Mock(return_value=10.0)
        
        order = portfolio.place_order('AAPL', 100, 'BUY', OrderType.MARKET)
        
        # Try to buy 100 shares at $100 = $10,000 + commission
        result = portfolio.execute_order(order, 100.0, datetime.now())
        
        assert result == False
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == "Insufficient cash"
    
    def test_execute_order_insufficient_shares(self):
        """Test execute_order with insufficient shares."""
        portfolio = Portfolio()
        portfolio.allow_short = False
        
        # Try to sell without position
        order = portfolio.place_order('AAPL', 100, 'SELL', OrderType.MARKET)
        result = portfolio.execute_order(order, 100.0, datetime.now())
        
        assert result == False
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == "Insufficient shares"
        
        # Add partial position
        position = Position(symbol='GOOGL')
        position.quantity = 50
        portfolio.positions['GOOGL'] = position
        
        # Try to sell more than we have
        order2 = portfolio.place_order('GOOGL', 100, 'SELL', OrderType.MARKET)
        result = portfolio.execute_order(order2, 200.0, datetime.now())
        
        assert result == False
        assert order2.rejection_reason == "Insufficient shares"
    
    def test_execute_order_short_selling(self):
        """Test execute_order with short selling enabled."""
        portfolio = Portfolio()
        portfolio.allow_short = True
        portfolio._process_fill = Mock()
        portfolio.calculate_commission = Mock(return_value=5.0)
        
        # Should allow selling without position
        order = portfolio.place_order('AAPL', 100, 'SELL', OrderType.MARKET)
        result = portfolio.execute_order(order, 100.0, datetime.now())
        
        assert result == True
        assert order.status == OrderStatus.FILLED
    
    def test_execute_order_inactive(self):
        """Test execute_order with inactive order."""
        portfolio = Portfolio()
        
        order = portfolio.place_order('AAPL', 100, 'BUY')
        order.status = OrderStatus.FILLED  # Already filled
        
        result = portfolio.execute_order(order, 100.0, datetime.now())
        assert result == False
    
    def test_process_fill(self):
        """Test _process_fill method."""
        portfolio = Portfolio()
        portfolio.cash = 10000.0
        
        # Test buy order fill
        buy_order = Order(
            order_id='BUY001',
            symbol='AAPL',
            quantity=100,
            direction='BUY',
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        buy_order.status = OrderStatus.FILLED
        
        timestamp = datetime.now()
        portfolio._process_fill(buy_order, 100.0, 10.0, timestamp)
        
        # Check position was created
        assert 'AAPL' in portfolio.positions
        position = portfolio.positions['AAPL']
        assert position.quantity == 100
        
        # Check cash was reduced
        assert portfolio.cash == 10000.0 - (100 * 100.0 + 10.0)  # 10000 - 10010 = -10
        
        # Check totals
        assert portfolio.total_commission == 10.0
        assert portfolio.total_trades == 1
        
        # Test sell order fill
        sell_order = Order(
            order_id='SELL001',
            symbol='AAPL',
            quantity=50,
            direction='SELL',
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        portfolio._process_fill(sell_order, 105.0, 5.0, timestamp)
        
        # Check position was updated
        assert position.quantity == 50
        
        # Check cash increased
        assert portfolio.cash == -10.0 + (50 * 105.0 - 5.0)  # -10 + 5245 = 5235
        
        # Check totals
        assert portfolio.total_commission == 15.0
        assert portfolio.total_trades == 2
        
        # Test closing position
        sell_order2 = Order(
            order_id='SELL002',
            symbol='AAPL',
            quantity=50,
            direction='SELL',
            order_type=OrderType.MARKET,
            created_time=datetime.now()
        )
        
        portfolio._process_fill(sell_order2, 110.0, 5.0, timestamp)
        
        # Check position is closed
        assert not position.is_open()
        assert len(portfolio.closed_positions) == 1
        assert portfolio.closed_positions[0] == position
    
    def test_update_prices(self):
        """Test update_prices method."""
        portfolio = Portfolio()
        
        # Add positions
        position1 = Position(symbol='AAPL')
        position1.quantity = 100
        position1.current_price = 100.0
        portfolio.positions['AAPL'] = position1
        
        position2 = Position(symbol='GOOGL')
        position2.quantity = 50
        position2.current_price = 200.0
        portfolio.positions['GOOGL'] = position2
        
        # Update prices
        new_prices = {
            'AAPL': 105.0,
            'GOOGL': 210.0,
            'MSFT': 300.0  # Not in portfolio
        }
        
        portfolio.update_prices(new_prices)
        
        # Check prices were updated
        assert position1.current_price == 105.0
        assert position2.current_price == 210.0
    
    def test_check_stops(self):
        """Test check_stops method."""
        portfolio = Portfolio()
        
        # Add position with stop loss
        position = Position(symbol='AAPL')
        position.quantity = 100
        position.entry_price = 100.0
        position.stop_loss = 95.0
        position.current_price = 94.0  # Below stop
        portfolio.positions['AAPL'] = position
        
        # Mock check_stops on position
        position.check_stops = Mock(return_value='stop_loss')
        
        stop_orders = portfolio.check_stops()
        
        assert len(stop_orders) == 1
        assert stop_orders[0].symbol == 'AAPL'
        assert stop_orders[0].quantity == 100
        assert stop_orders[0].direction == 'SELL'
        assert stop_orders[0].order_type == OrderType.MARKET
        assert 'Stop hit: stop_loss' in stop_orders[0].notes
        
        # Test with no stops hit
        position.check_stops = Mock(return_value=None)
        stop_orders = portfolio.check_stops()
        assert len(stop_orders) == 0
    
    def test_take_snapshot(self):
        """Test take_snapshot method."""
        portfolio = Portfolio()
        portfolio.cash = 50000.0
        portfolio.total_commission = 100.0
        
        # Add positions
        position1 = Position(symbol='AAPL')
        position1.quantity = 100
        position1.current_price = 150.0
        position1.to_dict = Mock(return_value={'quantity': 100, 'value': 15000})
        portfolio.positions['AAPL'] = position1
        
        # Add closed position
        closed_pos = Position(symbol='GOOGL')
        closed_pos.realized_pnl = 500.0
        portfolio.closed_positions.append(closed_pos)
        
        # Take snapshot
        timestamp = datetime.now()
        snapshot = portfolio.take_snapshot(timestamp)
        
        assert snapshot.timestamp == timestamp
        assert snapshot.cash == 50000.0
        assert snapshot.positions_value == 15000.0
        assert snapshot.total_value == 65000.0
        assert snapshot.commission == 100.0
        assert len(portfolio.history) == 1
        assert portfolio.history[0] == snapshot
    
    def test_get_performance_summary(self):
        """Test get_performance_summary method."""
        portfolio = Portfolio(initial_capital=100000.0)
        
        # Test with no history
        assert portfolio.get_performance_summary() == {}
        
        # Add history
        timestamps = pd.date_range('2023-01-01', periods=5, freq='D')
        values = [100000, 101000, 99000, 102000, 103000]
        
        for i, (ts, val) in enumerate(zip(timestamps, values)):
            snapshot = PortfolioSnapshot(
                timestamp=ts,
                cash=val * 0.5,
                positions_value=val * 0.5,
                total_value=val,
                unrealized_pnl=0,
                realized_pnl=0,
                commission=0,
                positions={}
            )
            portfolio.history.append(snapshot)
        
        # Add closed positions
        winning_pos = Position(symbol='AAPL')
        winning_pos.realized_pnl = 1000.0
        portfolio.closed_positions.append(winning_pos)
        
        losing_pos = Position(symbol='GOOGL')
        losing_pos.realized_pnl = -500.0
        portfolio.closed_positions.append(losing_pos)
        
        portfolio.total_trades = 2
        portfolio.total_commission = 50.0
        
        # Mock current value
        portfolio.current_value = Mock(return_value=103000.0)
        portfolio.unrealized_pnl = Mock(return_value=500.0)
        portfolio.realized_pnl = Mock(return_value=500.0)
        portfolio.return_pct = Mock(return_value=3.0)
        portfolio.total_pnl = Mock(return_value=3000.0)
        
        summary = portfolio.get_performance_summary()
        
        assert summary['initial_capital'] == 100000.0
        assert summary['final_value'] == 103000.0
        assert summary['total_return'] == 3.0
        assert summary['total_trades'] == 2
        assert summary['winning_trades'] == 1
        assert summary['losing_trades'] == 1
        assert summary['avg_win'] == 1000.0
        assert summary['avg_loss'] == -500.0
        assert 'max_drawdown' in summary
        assert 'sharpe_ratio' in summary
    
    def test_calculate_max_drawdown(self):
        """Test _calculate_max_drawdown method."""
        portfolio = Portfolio()
        
        # Test empty curve
        assert portfolio._calculate_max_drawdown([]) == 0.0
        
        # Test with drawdown
        equity_curve = [100, 110, 105, 95, 100, 90, 95]
        max_dd = portfolio._calculate_max_drawdown(equity_curve)
        
        # Max drawdown from 110 to 90 = (110-90)/110 * 100 = 18.18%
        assert abs(max_dd - 18.18) < 0.01
        
        # Test no drawdown
        equity_curve = [100, 110, 120, 130]
        assert portfolio._calculate_max_drawdown(equity_curve) == 0.0
    
    def test_calculate_commission(self):
        """Test calculate_commission method."""
        # Test simple commission
        portfolio = Portfolio(commission_rate=0.001, min_commission=1.0)
        
        # Test above minimum
        assert portfolio.calculate_commission(10000) == 10.0
        
        # Test minimum commission
        assert portfolio.calculate_commission(500) == 1.0
        
        # Test tiered commission
        portfolio2 = Portfolio(
            commission_structure={1000: 0.002, 5000: 0.0015, 10000: 0.001},
            min_commission=2.0
        )
        
        # Trade value 2000: first 1000 at 0.002, next 1000 at 0.0015
        commission = portfolio2.calculate_commission(2000)
        expected = 1000 * 0.002 + 1000 * 0.0015
        assert commission == max(expected, 2.0)
    
    def test_get_returns(self):
        """Test get_returns method."""
        portfolio = Portfolio(initial_capital=100000.0)
        portfolio.cash = 50000.0
        
        # Add position
        position = Position(symbol='AAPL')
        position.quantity = 100
        position.current_price = 600.0  # Value = 60000
        portfolio.positions['AAPL'] = position
        
        # Add closed position
        closed_pos = Position(symbol='GOOGL')
        closed_pos.realized_pnl = 5000.0
        portfolio.closed_positions.append(closed_pos)
        
        returns = portfolio.get_returns()
        
        assert returns['initial_capital'] == 100000.0
        assert returns['current_value'] == 110000.0  # 50000 + 60000
        assert returns['total_return'] == 10000.0
        assert returns['total_return_pct'] == 10.0
        assert returns['realized_return'] == 5000.0
    
    def test_calculate_portfolio_heat(self):
        """Test calculate_portfolio_heat method."""
        portfolio = Portfolio()
        portfolio.cash = 50000.0
        
        # Add position with stop loss
        position1 = Position(symbol='AAPL')
        position1.quantity = 100
        position1.avg_price = 100.0
        position1.stop_loss = 95.0
        position1.current_price = 100.0
        portfolio.positions['AAPL'] = position1
        
        # Risk = (100 - 95) * 100 = 500
        # Heat = 500 / 60000 = 0.0083
        heat = portfolio.calculate_portfolio_heat()
        assert abs(heat - 0.0083) < 0.0001
        
        # Add position without stop loss
        position2 = Position(symbol='GOOGL')
        position2.quantity = 50
        position2.current_price = 200.0
        portfolio.positions['GOOGL'] = position2
        
        # Still only 500 risk from AAPL
        # Total value = 50000 + 10000 + 10000 = 70000
        heat = portfolio.calculate_portfolio_heat()
        assert abs(heat - 500/70000) < 0.0001
    
    def test_calculate_correlation_risk(self):
        """Test calculate_correlation_risk method."""
        portfolio = Portfolio()
        
        # Simple placeholder implementation
        portfolio.positions = {'AAPL': Mock(), 'GOOGL': Mock()}
        assert portfolio.calculate_correlation_risk() == 0.2  # 2 positions * 0.1
    
    def test_get_trade_statistics(self):
        """Test get_trade_statistics method."""
        portfolio = Portfolio()
        
        # Test with no trades
        assert portfolio.get_trade_statistics() == {}
        
        # Add trades
        for i in range(5):
            pos = Position(symbol=f'STOCK{i}')
            pos.realized_pnl = 100 if i < 3 else -50
            portfolio.closed_positions.append(pos)
        
        stats = portfolio.get_trade_statistics()
        
        assert stats['total_trades'] == 5
        assert stats['winning_trades'] == 3
        assert stats['losing_trades'] == 2
        assert stats['win_rate'] == 60.0
        assert stats['avg_win'] == 100.0
        assert stats['avg_loss'] == -50.0
        assert stats['profit_factor'] == 3.0  # 300 / 100
        assert stats['total_pnl'] == 200.0
    
    def test_calculate_sharpe_ratio(self):
        """Test calculate_sharpe_ratio method."""
        portfolio = Portfolio()
        
        # Test with no history
        assert portfolio.calculate_sharpe_ratio() == 0.0
        
        # Add history with daily returns
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        np.random.seed(42)
        values = 100000 * (1 + np.random.normal(0.001, 0.02, 30)).cumprod()
        
        for date, value in zip(dates, values):
            snapshot = PortfolioSnapshot(
                timestamp=date,
                cash=value * 0.5,
                positions_value=value * 0.5,
                total_value=value,
                unrealized_pnl=0,
                realized_pnl=0,
                commission=0,
                positions={}
            )
            portfolio.history.append(snapshot)
        
        sharpe = portfolio.calculate_sharpe_ratio(risk_free_rate=0.02)
        assert isinstance(sharpe, float)
        assert sharpe != 0.0
    
    def test_calculate_sortino_ratio(self):
        """Test calculate_sortino_ratio method."""
        portfolio = Portfolio()
        
        # Test with no history
        assert portfolio.calculate_sortino_ratio() == 0.0
        
        # Add history
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        values = [100000]
        for i in range(29):
            # Create returns with some negative
            ret = 0.01 if i % 3 != 0 else -0.005
            values.append(values[-1] * (1 + ret))
        
        for date, value in zip(dates, values):
            snapshot = PortfolioSnapshot(
                timestamp=date,
                cash=value,
                positions_value=0,
                total_value=value,
                unrealized_pnl=0,
                realized_pnl=0,
                commission=0,
                positions={}
            )
            portfolio.history.append(snapshot)
        
        sortino = portfolio.calculate_sortino_ratio()
        assert isinstance(sortino, float)
        
        # Test with no downside returns (should return inf)
        portfolio.history = []
        values = [100000 * (1.001 ** i) for i in range(10)]
        for date, value in zip(dates[:10], values):
            snapshot = PortfolioSnapshot(
                timestamp=date,
                cash=value,
                positions_value=0,
                total_value=value,
                unrealized_pnl=0,
                realized_pnl=0,
                commission=0,
                positions={}
            )
            portfolio.history.append(snapshot)
        
        sortino = portfolio.calculate_sortino_ratio()
        assert sortino == float('inf')
    
    def test_get_equity_curve(self):
        """Test get_equity_curve method."""
        portfolio = Portfolio()
        
        # Test with no history
        assert portfolio.get_equity_curve().empty
        
        # Add history
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        values = [100000, 101000, 99000, 102000, 103000]
        
        for date, value in zip(dates, values):
            snapshot = PortfolioSnapshot(
                timestamp=date,
                cash=value,
                positions_value=0,
                total_value=value,
                unrealized_pnl=0,
                realized_pnl=0,
                commission=0,
                positions={}
            )
            portfolio.history.append(snapshot)
        
        equity_curve = portfolio.get_equity_curve()
        
        assert isinstance(equity_curve, pd.Series)
        assert len(equity_curve) == 5
        assert list(equity_curve.values) == values
        assert list(equity_curve.index) == dates
    
    def test_calculate_slippage(self):
        """Test calculate_slippage method."""
        portfolio = Portfolio(slippage_rate=0.001)
        
        slippage = portfolio.calculate_slippage('AAPL', 100, 150.0)
        assert slippage == 100 * 150.0 * 0.001  # 15.0
    
    def test_alias_methods(self):
        """Test alias methods for compatibility."""
        portfolio = Portfolio()
        
        # Add some data
        closed_pos = Position(symbol='AAPL')
        closed_pos.realized_pnl = 1000.0
        portfolio.closed_positions.append(closed_pos)
        
        position = Position(symbol='GOOGL')
        position.quantity = 100
        position.entry_price = 100.0
        position.current_price = 110.0
        portfolio.positions['GOOGL'] = position
        
        # Test aliases
        assert portfolio.get_unrealized_pnl() == portfolio.unrealized_pnl()
        assert portfolio.get_realized_pnl() == portfolio.realized_pnl()