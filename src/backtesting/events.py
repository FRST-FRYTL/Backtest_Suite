"""Event-driven architecture components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional
import queue


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


class EventQueue:
    """Thread-safe event queue for backtesting."""
    
    def __init__(self):
        """Initialize the event queue."""
        self._queue = queue.Queue()
        self._events = []
        
    def put(self, event: Event):
        """Add event to the queue."""
        self._queue.put(event)
        self._events.append(event)
        
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Event:
        """Get event from the queue."""
        return self._queue.get(block=block, timeout=timeout)
        
    def get_nowait(self) -> Event:
        """Get event without blocking."""
        return self._queue.get_nowait()
        
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
        
    def qsize(self) -> int:
        """Get queue size."""
        return self._queue.qsize()
        
    def clear(self):
        """Clear all events."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._events.clear()
        
    def get_all_events(self) -> List[Event]:
        """Get all events that have been processed."""
        return self._events.copy()