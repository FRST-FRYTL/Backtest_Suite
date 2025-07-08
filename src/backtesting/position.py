"""Position tracking and management."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Trade:
    """Individual trade within a position."""
    
    timestamp: datetime
    quantity: int
    price: float
    commission: float
    trade_type: str  # 'OPEN', 'ADD', 'REDUCE', 'CLOSE'


@dataclass
class Position:
    """Position representation."""
    
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    current_price: float = 0.0
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    trades: List[Trade] = field(default_factory=list)
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    highest_price: float = 0.0  # For trailing stop
    
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.quantity != 0
        
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
        
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0
        
    def market_value(self) -> float:
        """Calculate current market value."""
        return abs(self.quantity) * self.current_price
        
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.quantity == 0:
            return 0.0
            
        if self.is_long():
            return self.quantity * (self.current_price - self.avg_price)
        else:
            return abs(self.quantity) * (self.avg_price - self.current_price)
            
    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl()
        
    def return_pct(self) -> float:
        """Calculate return percentage."""
        if self.avg_price == 0:
            return 0.0
            
        if self.is_long():
            return ((self.current_price - self.avg_price) / self.avg_price) * 100
        else:
            return ((self.avg_price - self.current_price) / self.avg_price) * 100
            
    def add_trade(
        self,
        timestamp: datetime,
        quantity: int,
        price: float,
        commission: float = 0.0
    ) -> None:
        """
        Add a trade to the position.
        
        Args:
            timestamp: Trade timestamp
            quantity: Trade quantity (positive for buy, negative for sell)
            price: Trade price
            commission: Commission paid
        """
        # Determine trade type
        if self.quantity == 0:
            trade_type = 'OPEN'
            self.opened_at = timestamp
        elif (self.quantity > 0 and quantity > 0) or (self.quantity < 0 and quantity < 0):
            trade_type = 'ADD'
        elif abs(self.quantity + quantity) < abs(self.quantity):
            trade_type = 'REDUCE'
        else:
            trade_type = 'CLOSE'
            
        # Create trade record
        trade = Trade(
            timestamp=timestamp,
            quantity=quantity,
            price=price,
            commission=commission,
            trade_type=trade_type
        )
        self.trades.append(trade)
        
        # Update position
        old_quantity = self.quantity
        old_avg_price = self.avg_price
        
        # Calculate realized P&L for reducing trades
        if old_quantity != 0 and (old_quantity * quantity < 0):
            # Closing or reducing position
            closed_quantity = min(abs(quantity), abs(old_quantity))
            
            if old_quantity > 0:  # Long position
                self.realized_pnl += closed_quantity * (price - old_avg_price)
            else:  # Short position
                self.realized_pnl += closed_quantity * (old_avg_price - price)
                
        # Update quantity and average price
        if self.quantity == 0 or (self.quantity * quantity > 0):
            # Opening or adding to position
            total_value = (abs(self.quantity) * self.avg_price) + (abs(quantity) * price)
            total_quantity = abs(self.quantity) + abs(quantity)
            self.avg_price = total_value / total_quantity if total_quantity > 0 else 0
            
        self.quantity += quantity
        self.total_commission += commission
        self.realized_pnl -= commission  # Subtract commission from P&L
        
        # Close position if quantity is 0
        if self.quantity == 0:
            self.closed_at = timestamp
            self.avg_price = 0.0
            
    def update_price(self, price: float) -> None:
        """Update current price and check stops."""
        self.current_price = price
        
        # Update highest price for trailing stop
        if self.is_long() and price > self.highest_price:
            self.highest_price = price
        elif self.is_short() and (self.highest_price == 0 or price < self.highest_price):
            self.highest_price = price
            
    def check_stops(self) -> Optional[str]:
        """
        Check if any stop conditions are met.
        
        Returns:
            Stop type hit ('stop_loss', 'take_profit', 'trailing_stop') or None
        """
        if not self.is_open():
            return None
            
        # Check stop loss
        if self.stop_loss is not None:
            if self.is_long() and self.current_price <= self.stop_loss:
                return 'stop_loss'
            elif self.is_short() and self.current_price >= self.stop_loss:
                return 'stop_loss'
                
        # Check take profit
        if self.take_profit is not None:
            if self.is_long() and self.current_price >= self.take_profit:
                return 'take_profit'
            elif self.is_short() and self.current_price <= self.take_profit:
                return 'take_profit'
                
        # Check trailing stop
        if self.trailing_stop_pct is not None and self.highest_price > 0:
            if self.is_long():
                trailing_stop = self.highest_price * (1 - self.trailing_stop_pct)
                if self.current_price <= trailing_stop:
                    return 'trailing_stop'
            else:
                trailing_stop = self.highest_price * (1 + self.trailing_stop_pct)
                if self.current_price >= trailing_stop:
                    return 'trailing_stop'
                    
        return None
        
    def to_dict(self) -> dict:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'current_price': self.current_price,
            'market_value': self.market_value(),
            'unrealized_pnl': self.unrealized_pnl(),
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl(),
            'return_pct': self.return_pct(),
            'commission': self.total_commission,
            'opened_at': self.opened_at,
            'closed_at': self.closed_at,
            'is_open': self.is_open(),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trades_count': len(self.trades)
        }