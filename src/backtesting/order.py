"""Order management classes."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderType(Enum):
    """Types of orders."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderStatus(Enum):
    """Order status states."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Order representation."""
    
    order_id: str
    symbol: str
    quantity: int
    direction: str  # 'BUY' or 'SELL'
    order_type: OrderType
    created_time: datetime
    status: OrderStatus = OrderStatus.PENDING
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    limit_price: Optional[float] = None  # Alias for price
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    notes: str = ""
    rejection_reason: Optional[str] = None
    
    def is_buy(self) -> bool:
        """Check if order is a buy order."""
        return self.direction.upper() == "BUY"
        
    def is_sell(self) -> bool:
        """Check if order is a sell order."""
        return self.direction.upper() == "SELL"
        
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
        
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
        
    def remaining_quantity(self) -> int:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity
        
    def fill(self, quantity: int, price: float, commission: float = 0.0) -> None:
        """
        Fill order partially or completely.
        
        Args:
            quantity: Quantity filled
            price: Fill price
            commission: Commission charged
        """
        # Update average fill price
        total_value = (self.avg_fill_price * self.filled_quantity) + (price * quantity)
        self.filled_quantity += quantity
        self.avg_fill_price = total_value / self.filled_quantity
        self.commission += commission
        
        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL
            
    def cancel(self) -> None:
        """Cancel the order."""
        if self.is_active():
            self.status = OrderStatus.CANCELLED
            
    def reject(self, reason: str = "") -> None:
        """Reject the order."""
        self.status = OrderStatus.REJECTED
        self.rejection_reason = reason
        if reason:
            self.notes = f"Rejected: {reason}"
            
    @property
    def id(self) -> str:
        """Get order ID (alias for order_id)."""
        return self.order_id
        
    @property
    def fill_price(self) -> float:
        """Get fill price (alias for avg_fill_price)."""
        return self.avg_fill_price