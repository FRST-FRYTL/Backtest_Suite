"""Backtesting engine module."""

from .engine import BacktestEngine, BacktestResults
from .portfolio import Portfolio
from .order import Order, OrderType, OrderStatus
from .position import Position
from .events import Event, MarketEvent, SignalEvent, OrderEvent, FillEvent
from .strategy import Strategy, Trade, OrderSide

# Create alias for backward compatibility
BacktestResult = BacktestResults

__all__ = [
    "BacktestEngine",
    "BacktestResults",
    "BacktestResult",
    "Portfolio",
    "Order",
    "OrderType",
    "OrderStatus",
    "Position",
    "Event",
    "MarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "Strategy",
    "Trade",
    "OrderSide"
]