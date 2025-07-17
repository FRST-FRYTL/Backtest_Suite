"""Comprehensive tests for position module to achieve 100% coverage."""

import pytest
from datetime import datetime, timedelta

from src.backtesting.position import Position, Trade


class TestTrade:
    """Test Trade dataclass."""
    
    def test_trade_creation(self):
        """Test Trade creation with all fields."""
        timestamp = datetime.now()
        trade = Trade(
            timestamp=timestamp,
            quantity=100,
            price=150.0,
            commission=1.5,
            trade_type='OPEN'
        )
        
        assert trade.timestamp == timestamp
        assert trade.quantity == 100
        assert trade.price == 150.0
        assert trade.commission == 1.5
        assert trade.trade_type == 'OPEN'


class TestPosition:
    """Test Position class."""
    
    def test_position_initialization(self):
        """Test position initialization with default values."""
        position = Position(symbol='AAPL')
        
        assert position.symbol == 'AAPL'
        assert position.quantity == 0
        assert position.avg_price == 0.0
        assert position.current_price == 0.0
        assert position.opened_at is None
        assert position.closed_at is None
        assert position.trades == []
        assert position.realized_pnl == 0.0
        assert position.total_commission == 0.0
        assert position.stop_loss is None
        assert position.take_profit is None
        assert position.trailing_stop_pct is None
        assert position.highest_price == 0.0
    
    def test_position_with_custom_values(self):
        """Test position initialization with custom values."""
        timestamp = datetime.now()
        trades = [Trade(timestamp, 100, 150.0, 1.5, 'OPEN')]
        
        position = Position(
            symbol='AAPL',
            quantity=100,
            avg_price=150.0,
            current_price=155.0,
            opened_at=timestamp,
            trades=trades,
            realized_pnl=100.0,
            total_commission=1.5,
            stop_loss=145.0,
            take_profit=160.0,
            trailing_stop_pct=0.05,
            highest_price=156.0
        )
        
        assert position.quantity == 100
        assert position.avg_price == 150.0
        assert position.current_price == 155.0
        assert position.opened_at == timestamp
        assert len(position.trades) == 1
        assert position.stop_loss == 145.0
        assert position.take_profit == 160.0
        assert position.trailing_stop_pct == 0.05
        assert position.highest_price == 156.0
    
    def test_is_open(self):
        """Test is_open method."""
        position = Position(symbol='AAPL')
        
        # No quantity - closed
        assert position.is_open() == False
        
        # Long position - open
        position.quantity = 100
        assert position.is_open() == True
        
        # Short position - open
        position.quantity = -100
        assert position.is_open() == True
        
        # Back to zero - closed
        position.quantity = 0
        assert position.is_open() == False
    
    def test_is_long_and_is_short(self):
        """Test is_long and is_short methods."""
        position = Position(symbol='AAPL')
        
        # No position
        position.quantity = 0
        assert position.is_long() == False
        assert position.is_short() == False
        
        # Long position
        position.quantity = 100
        assert position.is_long() == True
        assert position.is_short() == False
        
        # Short position
        position.quantity = -100
        assert position.is_long() == False
        assert position.is_short() == True
    
    def test_market_value(self):
        """Test market_value calculation."""
        position = Position(symbol='AAPL')
        
        # No position
        position.quantity = 0
        position.current_price = 150.0
        assert position.market_value() == 0.0
        
        # Long position
        position.quantity = 100
        position.current_price = 150.0
        assert position.market_value() == 15000.0
        
        # Short position
        position.quantity = -100
        position.current_price = 150.0
        assert position.market_value() == 15000.0  # Absolute value
    
    def test_unrealized_pnl(self):
        """Test unrealized_pnl calculation."""
        position = Position(symbol='AAPL')
        
        # No position
        assert position.unrealized_pnl() == 0.0
        
        # Long position with profit
        position.quantity = 100
        position.avg_price = 150.0
        position.current_price = 155.0
        assert position.unrealized_pnl() == 500.0  # 100 * (155 - 150)
        
        # Long position with loss
        position.current_price = 145.0
        assert position.unrealized_pnl() == -500.0  # 100 * (145 - 150)
        
        # Short position with profit
        position.quantity = -100
        position.avg_price = 150.0
        position.current_price = 145.0
        assert position.unrealized_pnl() == 500.0  # 100 * (150 - 145)
        
        # Short position with loss
        position.current_price = 155.0
        assert position.unrealized_pnl() == -500.0  # 100 * (150 - 155)
    
    def test_total_pnl(self):
        """Test total_pnl calculation."""
        position = Position(symbol='AAPL')
        position.realized_pnl = 200.0
        position.quantity = 100
        position.avg_price = 150.0
        position.current_price = 155.0
        
        # Total = realized + unrealized
        # Unrealized = 100 * (155 - 150) = 500
        assert position.total_pnl() == 700.0
    
    def test_return_pct(self):
        """Test return_pct calculation."""
        position = Position(symbol='AAPL')
        
        # Zero average price
        position.avg_price = 0.0
        assert position.return_pct() == 0.0
        
        # Long position with profit
        position.quantity = 100
        position.avg_price = 150.0
        position.current_price = 165.0
        assert position.return_pct() == 10.0  # (165-150)/150 * 100
        
        # Long position with loss
        position.current_price = 135.0
        assert position.return_pct() == -10.0  # (135-150)/150 * 100
        
        # Short position with profit
        position.quantity = -100
        position.avg_price = 150.0
        position.current_price = 135.0
        assert position.return_pct() == 10.0  # (150-135)/150 * 100
        
        # Short position with loss
        position.current_price = 165.0
        assert position.return_pct() == -10.0  # (150-165)/150 * 100
    
    def test_add_trade_open_position(self):
        """Test add_trade for opening a position."""
        position = Position(symbol='AAPL')
        timestamp = datetime.now()
        
        # Open long position
        position.add_trade(timestamp, 100, 150.0, 1.5)
        
        assert position.quantity == 100
        assert position.avg_price == 150.0
        assert position.opened_at == timestamp
        assert position.total_commission == 1.5
        assert position.realized_pnl == -1.5  # Commission deducted
        assert len(position.trades) == 1
        assert position.trades[0].trade_type == 'OPEN'
    
    def test_add_trade_add_to_position(self):
        """Test add_trade for adding to existing position."""
        position = Position(symbol='AAPL')
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(hours=1)
        
        # Open position
        position.add_trade(timestamp1, 100, 150.0, 1.0)
        
        # Add to position
        position.add_trade(timestamp2, 50, 152.0, 0.5)
        
        assert position.quantity == 150
        # Avg price = (100*150 + 50*152) / 150 = 150.67
        assert abs(position.avg_price - 150.67) < 0.01
        assert position.total_commission == 1.5
        assert position.realized_pnl == -1.5  # Only commission
        assert len(position.trades) == 2
        assert position.trades[1].trade_type == 'ADD'
    
    def test_add_trade_reduce_position(self):
        """Test add_trade for reducing position."""
        position = Position(symbol='AAPL')
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(hours=1)
        
        # Open long position
        position.add_trade(timestamp1, 100, 150.0, 1.0)
        
        # Reduce position by selling 50
        position.add_trade(timestamp2, -50, 155.0, 0.5)
        
        assert position.quantity == 50
        assert position.avg_price == 150.0  # Unchanged
        # Realized P&L = 50 * (155 - 150) - commissions = 250 - 1.5 = 248.5
        assert position.realized_pnl == 248.5
        assert position.total_commission == 1.5
        assert len(position.trades) == 2
        assert position.trades[1].trade_type == 'REDUCE'
    
    def test_add_trade_close_position(self):
        """Test add_trade for closing position."""
        position = Position(symbol='AAPL')
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(hours=1)
        
        # Open long position
        position.add_trade(timestamp1, 100, 150.0, 1.0)
        
        # Close position
        position.add_trade(timestamp2, -100, 160.0, 1.0)
        
        assert position.quantity == 0
        assert position.avg_price == 0.0
        assert position.closed_at == timestamp2
        # Realized P&L = 100 * (160 - 150) - commissions = 1000 - 2 = 998
        assert position.realized_pnl == 998.0
        assert position.total_commission == 2.0
        assert len(position.trades) == 2
        assert position.trades[1].trade_type == 'CLOSE'
    
    def test_add_trade_short_position(self):
        """Test add_trade for short positions."""
        position = Position(symbol='AAPL')
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(hours=1)
        
        # Open short position
        position.add_trade(timestamp1, -100, 150.0, 1.0)
        
        assert position.quantity == -100
        assert position.avg_price == 150.0
        assert position.trades[0].trade_type == 'OPEN'
        
        # Add to short position
        position.add_trade(timestamp2, -50, 148.0, 0.5)
        
        assert position.quantity == -150
        # Avg price = (100*150 + 50*148) / 150 = 149.33
        assert abs(position.avg_price - 149.33) < 0.01
        assert position.trades[1].trade_type == 'ADD'
        
        # Reduce short position
        timestamp3 = timestamp2 + timedelta(hours=1)
        position.add_trade(timestamp3, 50, 145.0, 0.5)
        
        assert position.quantity == -100
        # Realized P&L for short = 50 * (149.33 - 145) - commissions
        expected_pnl = 50 * (position.avg_price - 145.0) - 2.0
        assert abs(position.realized_pnl - expected_pnl) < 0.1
        assert position.trades[2].trade_type == 'REDUCE'
    
    def test_add_trade_reverse_position(self):
        """Test add_trade for reversing position (long to short)."""
        position = Position(symbol='AAPL')
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(hours=1)
        
        # Open long position
        position.add_trade(timestamp1, 100, 150.0, 1.0)
        
        # Reverse to short (sell 200)
        position.add_trade(timestamp2, -200, 155.0, 2.0)
        
        assert position.quantity == -100
        # First 100 closes the long: P&L = 100 * (155 - 150) = 500
        # Next 100 opens short at 155
        assert position.avg_price == 155.0
        # Realized P&L = 500 - 3 (commissions) = 497
        assert position.realized_pnl == 497.0
    
    def test_update_price(self):
        """Test update_price method."""
        position = Position(symbol='AAPL')
        
        # Update price
        position.update_price(150.0)
        assert position.current_price == 150.0
        
        # Test highest price tracking for long position
        position.quantity = 100
        position.update_price(155.0)
        assert position.current_price == 155.0
        assert position.highest_price == 155.0
        
        # Price goes down - highest unchanged
        position.update_price(153.0)
        assert position.current_price == 153.0
        assert position.highest_price == 155.0
        
        # Price goes up - highest updated
        position.update_price(157.0)
        assert position.current_price == 157.0
        assert position.highest_price == 157.0
        
        # Test highest price tracking for short position
        position = Position(symbol='GOOGL')
        position.quantity = -100
        
        # First update sets highest
        position.update_price(200.0)
        assert position.highest_price == 200.0
        
        # Lower price updates highest for short
        position.update_price(195.0)
        assert position.highest_price == 195.0
        
        # Higher price doesn't update highest for short
        position.update_price(198.0)
        assert position.highest_price == 195.0
    
    def test_check_stops_no_position(self):
        """Test check_stops with no position."""
        position = Position(symbol='AAPL')
        position.stop_loss = 145.0
        position.take_profit = 155.0
        position.current_price = 140.0
        
        # No position - no stops triggered
        assert position.check_stops() is None
    
    def test_check_stops_long_position(self):
        """Test check_stops for long position."""
        position = Position(symbol='AAPL')
        position.quantity = 100
        position.avg_price = 150.0
        
        # Test stop loss
        position.stop_loss = 145.0
        position.current_price = 144.0
        assert position.check_stops() == 'stop_loss'
        
        # Price at stop loss
        position.current_price = 145.0
        assert position.check_stops() == 'stop_loss'
        
        # Price above stop loss
        position.current_price = 146.0
        assert position.check_stops() is None
        
        # Test take profit
        position.take_profit = 160.0
        position.current_price = 161.0
        assert position.check_stops() == 'take_profit'
        
        # Price at take profit
        position.current_price = 160.0
        assert position.check_stops() == 'take_profit'
        
        # Test trailing stop
        position.current_price = 155.0
        position.trailing_stop_pct = 0.05  # 5%
        position.highest_price = 160.0
        # Trailing stop = 160 * 0.95 = 152
        position.current_price = 151.0
        assert position.check_stops() == 'trailing_stop'
    
    def test_check_stops_short_position(self):
        """Test check_stops for short position."""
        position = Position(symbol='AAPL')
        position.quantity = -100
        position.avg_price = 150.0
        
        # Test stop loss (higher for short)
        position.stop_loss = 155.0
        position.current_price = 156.0
        assert position.check_stops() == 'stop_loss'
        
        # Price at stop loss
        position.current_price = 155.0
        assert position.check_stops() == 'stop_loss'
        
        # Price below stop loss
        position.current_price = 154.0
        assert position.check_stops() is None
        
        # Test take profit (lower for short)
        position.take_profit = 140.0
        position.current_price = 139.0
        assert position.check_stops() == 'take_profit'
        
        # Price at take profit
        position.current_price = 140.0
        assert position.check_stops() == 'take_profit'
        
        # Test trailing stop for short
        position.current_price = 145.0
        position.trailing_stop_pct = 0.05  # 5%
        position.highest_price = 140.0  # Lowest price for short
        # Trailing stop = 140 * 1.05 = 147
        position.current_price = 148.0
        assert position.check_stops() == 'trailing_stop'
    
    def test_to_dict(self):
        """Test to_dict method."""
        timestamp = datetime.now()
        position = Position(symbol='AAPL')
        position.quantity = 100
        position.avg_price = 150.0
        position.current_price = 155.0
        position.opened_at = timestamp
        position.realized_pnl = 100.0
        position.total_commission = 2.0
        position.stop_loss = 145.0
        position.take_profit = 160.0
        position.trades = [
            Trade(timestamp, 100, 150.0, 1.0, 'OPEN'),
            Trade(timestamp, 50, 152.0, 0.5, 'ADD')
        ]
        
        result = position.to_dict()
        
        assert result['symbol'] == 'AAPL'
        assert result['quantity'] == 100
        assert result['avg_price'] == 150.0
        assert result['current_price'] == 155.0
        assert result['market_value'] == 15500.0
        assert result['unrealized_pnl'] == 500.0
        assert result['realized_pnl'] == 100.0
        assert result['total_pnl'] == 600.0
        assert abs(result['return_pct'] - 3.33) < 0.01
        assert result['commission'] == 2.0
        assert result['opened_at'] == timestamp
        assert result['closed_at'] is None
        assert result['is_open'] == True
        assert result['stop_loss'] == 145.0
        assert result['take_profit'] == 160.0
        assert result['trades_count'] == 2
    
    def test_edge_cases(self):
        """Test various edge cases."""
        position = Position(symbol='AAPL')
        
        # Test with zero highest price for trailing stop
        position.quantity = 100
        position.trailing_stop_pct = 0.05
        position.highest_price = 0
        position.current_price = 150.0
        assert position.check_stops() is None
        
        # Test adding trade with zero quantity
        timestamp = datetime.now()
        position.add_trade(timestamp, 0, 150.0, 0.0)
        # Should still add the trade
        assert len(position.trades) == 1