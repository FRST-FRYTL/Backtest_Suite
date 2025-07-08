"""Event-driven architecture components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class EventType(Enum):
    """Types of events in the backtesting system."""
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


@dataclass
class Event(ABC):
    """Base event class."""
    
    timestamp: datetime
    symbol: str
    
    @abstractmethod
    def get_type(self) -> EventType:
        """Return event type."""
        pass


@dataclass
class MarketEvent(Event):
    """Market data update event."""
    
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def get_type(self) -> EventType:
        return EventType.MARKET


@dataclass
class SignalEvent(Event):
    """Trading signal event."""
    
    signal_type: str  # 'LONG', 'SHORT', 'EXIT'
    strength: float  # Signal strength (0-1)
    quantity: Optional[int] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def get_type(self) -> EventType:
        return EventType.SIGNAL


@dataclass
class OrderEvent(Event):
    """Order placement event."""
    
    order_type: str  # 'MARKET', 'LIMIT', 'STOP'
    quantity: int
    direction: str  # 'BUY', 'SELL'
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    
    def get_type(self) -> EventType:
        return EventType.ORDER


@dataclass
class FillEvent(Event):
    """Order fill/execution event."""
    
    quantity: int
    direction: str  # 'BUY', 'SELL'
    fill_price: float
    commission: float
    slippage: float
    
    def get_type(self) -> EventType:
        return EventType.FILL