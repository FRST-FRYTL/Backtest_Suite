"""Strategy interface for backtesting framework."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from datetime import datetime


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Order representation."""
    
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    order_id: Optional[str] = None
    
    def __post_init__(self):
        if self.order_id is None:
            self.order_id = f"{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{self.symbol}_{self.side.value}"


@dataclass
class Trade:
    """Trade representation."""
    
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime
    commission: float = 0.0
    trade_id: Optional[str] = None
    
    def __post_init__(self):
        if self.trade_id is None:
            self.trade_id = f"{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{self.symbol}_{self.side.value}"
    
    @property
    def value(self) -> float:
        """Trade value excluding commission."""
        return self.quantity * self.price
    
    @property
    def net_value(self) -> float:
        """Trade value including commission."""
        return self.value - self.commission


class Strategy(ABC):
    """Abstract base class for backtesting strategies."""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self.current_bar = 0
        self.bars = None
        self.positions = {}
        self.pending_orders = []
        self.executed_trades = []
        self.context = {}
        
    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> None:
        """
        Initialize strategy before backtesting starts.
        
        Args:
            context: Strategy context with market data, portfolio, etc.
        """
        pass
    
    @abstractmethod
    def handle_data(self, context: Dict[str, Any], data: pd.DataFrame) -> List[Order]:
        """
        Handle new market data and generate orders.
        
        Args:
            context: Strategy context
            data: Current market data
            
        Returns:
            List of orders to execute
        """
        pass
    
    def before_trading_start(self, context: Dict[str, Any]) -> None:
        """
        Called before market open each day.
        
        Args:
            context: Strategy context
        """
        pass
    
    def after_trading_end(self, context: Dict[str, Any]) -> None:
        """
        Called after market close each day.
        
        Args:
            context: Strategy context
        """
        pass
    
    def on_order_filled(self, order: Order, trade: Trade) -> None:
        """
        Called when an order is filled.
        
        Args:
            order: Filled order
            trade: Resulting trade
        """
        pass
    
    def on_order_canceled(self, order: Order) -> None:
        """
        Called when an order is canceled.
        
        Args:
            order: Canceled order
        """
        pass
    
    def get_position(self, symbol: str) -> int:
        """
        Get current position size for symbol.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            Position size (positive for long, negative for short)
        """
        return self.positions.get(symbol, 0)
    
    def update_position(self, symbol: str, quantity: int) -> None:
        """
        Update position for symbol.
        
        Args:
            symbol: Symbol to update
            quantity: New position size
        """
        if quantity == 0:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = quantity
    
    def order_shares(
        self,
        symbol: str,
        amount: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """
        Create order for specified number of shares.
        
        Args:
            symbol: Symbol to trade
            amount: Number of shares (positive for buy, negative for sell)
            order_type: Order type
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            
        Returns:
            Created order
        """
        side = OrderSide.BUY if amount > 0 else OrderSide.SELL
        quantity = abs(amount)
        
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=limit_price,
            stop_price=stop_price
        )
        
        self.pending_orders.append(order)
        return order
    
    def order_percent(
        self,
        symbol: str,
        percent: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Optional[Order]:
        """
        Create order for percentage of portfolio.
        
        Args:
            symbol: Symbol to trade
            percent: Percentage of portfolio (0.0 to 1.0)
            order_type: Order type
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            
        Returns:
            Created order or None if insufficient data
        """
        if 'portfolio_value' not in self.context:
            return None
            
        portfolio_value = self.context['portfolio_value']
        current_price = self.context.get('current_prices', {}).get(symbol)
        
        if current_price is None:
            return None
            
        target_value = portfolio_value * percent
        current_position = self.get_position(symbol)
        current_value = current_position * current_price
        
        order_value = target_value - current_value
        quantity = int(order_value / current_price)
        
        if quantity == 0:
            return None
            
        return self.order_shares(symbol, quantity, order_type, limit_price, stop_price)
    
    def order_target_shares(
        self,
        symbol: str,
        target: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Optional[Order]:
        """
        Create order to reach target position.
        
        Args:
            symbol: Symbol to trade
            target: Target position size
            order_type: Order type
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            
        Returns:
            Created order or None if no change needed
        """
        current_position = self.get_position(symbol)
        quantity = target - current_position
        
        if quantity == 0:
            return None
            
        return self.order_shares(symbol, quantity, order_type, limit_price, stop_price)
    
    def cancel_order(self, order: Order) -> bool:
        """
        Cancel pending order.
        
        Args:
            order: Order to cancel
            
        Returns:
            True if order was canceled, False if not found
        """
        if order in self.pending_orders:
            self.pending_orders.remove(order)
            self.on_order_canceled(order)
            return True
        return False
    
    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get pending orders.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of pending orders
        """
        if symbol is None:
            return self.pending_orders.copy()
        return [order for order in self.pending_orders if order.symbol == symbol]
    
    def get_executed_trades(self, symbol: Optional[str] = None) -> List[Trade]:
        """
        Get executed trades.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            List of executed trades
        """
        if symbol is None:
            return self.executed_trades.copy()
        return [trade for trade in self.executed_trades if trade.symbol == symbol]
    
    def log_info(self, message: str) -> None:
        """
        Log info message.
        
        Args:
            message: Message to log
        """
        print(f"[{self.name}] INFO: {message}")
    
    def log_warning(self, message: str) -> None:
        """
        Log warning message.
        
        Args:
            message: Message to log
        """
        print(f"[{self.name}] WARNING: {message}")
    
    def log_error(self, message: str) -> None:
        """
        Log error message.
        
        Args:
            message: Message to log
        """
        print(f"[{self.name}] ERROR: {message}")
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get strategy parameter.
        
        Args:
            name: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value
        """
        return self.parameters.get(name, default)
    
    def set_parameter(self, name: str, value: Any) -> None:
        """
        Set strategy parameter.
        
        Args:
            name: Parameter name
            value: Parameter value
        """
        self.parameters[name] = value
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get strategy state for serialization.
        
        Returns:
            Strategy state dictionary
        """
        return {
            'name': self.name,
            'parameters': self.parameters,
            'current_bar': self.current_bar,
            'positions': self.positions,
            'pending_orders_count': len(self.pending_orders),
            'executed_trades_count': len(self.executed_trades)
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"Strategy(name='{self.name}', parameters={self.parameters})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Strategy(name='{self.name}', parameters={self.parameters}, positions={self.positions})"