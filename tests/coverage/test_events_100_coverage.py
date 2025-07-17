"""Comprehensive tests for events module to achieve 100% coverage."""

import pytest
from datetime import datetime
import queue

from src.backtesting.events import (
    Event, EventType, MarketEvent, SignalEvent, 
    OrderEvent, FillEvent, EventQueue
)


class TestEventTypes:
    """Test EventType enum."""
    
    def test_event_type_values(self):
        """Test all EventType enum values."""
        assert EventType.MARKET.value == "MARKET"
        assert EventType.SIGNAL.value == "SIGNAL"
        assert EventType.ORDER.value == "ORDER"
        assert EventType.FILL.value == "FILL"
        
        # Test enum membership
        assert EventType.MARKET in EventType
        assert EventType.SIGNAL in EventType
        assert EventType.ORDER in EventType
        assert EventType.FILL in EventType


class TestMarketEvent:
    """Test MarketEvent class."""
    
    def test_market_event_creation(self):
        """Test MarketEvent creation with all fields."""
        timestamp = datetime.now()
        event = MarketEvent(
            timestamp=timestamp,
            symbol="AAPL",
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000.0
        )
        
        assert event.timestamp == timestamp
        assert event.symbol == "AAPL"
        assert event.open == 100.0
        assert event.high == 105.0
        assert event.low == 95.0
        assert event.close == 102.0
        assert event.volume == 1000000.0
        assert event.get_type() == EventType.MARKET


class TestSignalEvent:
    """Test SignalEvent class."""
    
    def test_signal_event_minimal(self):
        """Test SignalEvent with minimal parameters."""
        timestamp = datetime.now()
        event = SignalEvent(
            timestamp=timestamp,
            symbol="AAPL",
            signal_type="LONG",
            strength=0.8
        )
        
        assert event.timestamp == timestamp
        assert event.symbol == "AAPL"
        assert event.signal_type == "LONG"
        assert event.strength == 0.8
        assert event.quantity is None
        assert event.stop_loss is None
        assert event.take_profit is None
        assert event.get_type() == EventType.SIGNAL
    
    def test_signal_event_full(self):
        """Test SignalEvent with all parameters."""
        timestamp = datetime.now()
        event = SignalEvent(
            timestamp=timestamp,
            symbol="AAPL",
            signal_type="SHORT",
            strength=0.95,
            quantity=100,
            stop_loss=105.0,
            take_profit=95.0
        )
        
        assert event.timestamp == timestamp
        assert event.symbol == "AAPL"
        assert event.signal_type == "SHORT"
        assert event.strength == 0.95
        assert event.quantity == 100
        assert event.stop_loss == 105.0
        assert event.take_profit == 95.0
        assert event.get_type() == EventType.SIGNAL
    
    def test_signal_event_exit_type(self):
        """Test SignalEvent with EXIT type."""
        event = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type="EXIT",
            strength=1.0
        )
        
        assert event.signal_type == "EXIT"
        assert event.strength == 1.0
        assert event.get_type() == EventType.SIGNAL


class TestOrderEvent:
    """Test OrderEvent class."""
    
    def test_order_event_market(self):
        """Test OrderEvent for market order."""
        timestamp = datetime.now()
        event = OrderEvent(
            timestamp=timestamp,
            symbol="AAPL",
            order_type="MARKET",
            quantity=100,
            direction="BUY"
        )
        
        assert event.timestamp == timestamp
        assert event.symbol == "AAPL"
        assert event.order_type == "MARKET"
        assert event.quantity == 100
        assert event.direction == "BUY"
        assert event.price is None
        assert event.stop_price is None
        assert event.get_type() == EventType.ORDER
    
    def test_order_event_limit(self):
        """Test OrderEvent for limit order."""
        event = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type="LIMIT",
            quantity=50,
            direction="SELL",
            price=105.0
        )
        
        assert event.order_type == "LIMIT"
        assert event.quantity == 50
        assert event.direction == "SELL"
        assert event.price == 105.0
        assert event.stop_price is None
        assert event.get_type() == EventType.ORDER
    
    def test_order_event_stop(self):
        """Test OrderEvent for stop order."""
        event = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type="STOP",
            quantity=100,
            direction="SELL",
            stop_price=95.0
        )
        
        assert event.order_type == "STOP"
        assert event.quantity == 100
        assert event.direction == "SELL"
        assert event.price is None
        assert event.stop_price == 95.0
        assert event.get_type() == EventType.ORDER


class TestFillEvent:
    """Test FillEvent class."""
    
    def test_fill_event_buy(self):
        """Test FillEvent for buy order."""
        timestamp = datetime.now()
        event = FillEvent(
            timestamp=timestamp,
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            fill_price=102.5,
            commission=1.0,
            slippage=0.1
        )
        
        assert event.timestamp == timestamp
        assert event.symbol == "AAPL"
        assert event.quantity == 100
        assert event.direction == "BUY"
        assert event.fill_price == 102.5
        assert event.commission == 1.0
        assert event.slippage == 0.1
        assert event.get_type() == EventType.FILL
    
    def test_fill_event_sell(self):
        """Test FillEvent for sell order."""
        event = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            quantity=50,
            direction="SELL",
            fill_price=103.0,
            commission=0.5,
            slippage=0.05
        )
        
        assert event.quantity == 50
        assert event.direction == "SELL"
        assert event.fill_price == 103.0
        assert event.commission == 0.5
        assert event.slippage == 0.05
        assert event.get_type() == EventType.FILL


class TestEventQueue:
    """Test EventQueue class."""
    
    def test_event_queue_initialization(self):
        """Test EventQueue initialization."""
        eq = EventQueue()
        assert eq.empty() == True
        assert eq.qsize() == 0
        assert eq.get_all_events() == []
    
    def test_event_queue_put_and_get(self):
        """Test putting and getting events."""
        eq = EventQueue()
        
        # Create test events
        market_event = MarketEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            open=100, high=105, low=95, close=102, volume=1000000
        )
        signal_event = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type="LONG",
            strength=0.8
        )
        
        # Put events
        eq.put(market_event)
        eq.put(signal_event)
        
        assert eq.empty() == False
        assert eq.qsize() == 2
        
        # Get events
        event1 = eq.get(block=False)
        assert event1 == market_event
        assert eq.qsize() == 1
        
        event2 = eq.get(block=False)
        assert event2 == signal_event
        assert eq.qsize() == 0
        assert eq.empty() == True
    
    def test_event_queue_get_nowait(self):
        """Test get_nowait method."""
        eq = EventQueue()
        
        # Test empty queue raises exception
        with pytest.raises(queue.Empty):
            eq.get_nowait()
        
        # Add event and get without waiting
        event = MarketEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            open=100, high=105, low=95, close=102, volume=1000000
        )
        eq.put(event)
        
        retrieved = eq.get_nowait()
        assert retrieved == event
    
    def test_event_queue_blocking_get(self):
        """Test blocking get with timeout."""
        eq = EventQueue()
        
        # Test timeout on empty queue
        with pytest.raises(queue.Empty):
            eq.get(block=True, timeout=0.1)
    
    def test_event_queue_clear(self):
        """Test clearing the queue."""
        eq = EventQueue()
        
        # Add multiple events
        for i in range(5):
            eq.put(MarketEvent(
                timestamp=datetime.now(),
                symbol=f"TEST{i}",
                open=100, high=105, low=95, close=102, volume=1000000
            ))
        
        assert eq.qsize() == 5
        assert len(eq.get_all_events()) == 5
        
        # Clear queue
        eq.clear()
        
        assert eq.empty() == True
        assert eq.qsize() == 0
        assert eq.get_all_events() == []
    
    def test_event_queue_clear_empty_queue(self):
        """Test clearing an already empty queue."""
        eq = EventQueue()
        
        # Clear empty queue (should not raise)
        eq.clear()
        assert eq.empty() == True
    
    def test_event_queue_get_all_events(self):
        """Test get_all_events returns copy."""
        eq = EventQueue()
        
        # Add events
        event1 = MarketEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            open=100, high=105, low=95, close=102, volume=1000000
        )
        event2 = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type="LONG",
            strength=0.8
        )
        
        eq.put(event1)
        eq.put(event2)
        
        # Get all events
        all_events = eq.get_all_events()
        assert len(all_events) == 2
        assert all_events[0] == event1
        assert all_events[1] == event2
        
        # Verify it's a copy (modifying returned list doesn't affect original)
        all_events.clear()
        assert len(eq.get_all_events()) == 2
    
    def test_event_queue_mixed_operations(self):
        """Test mixed queue operations."""
        eq = EventQueue()
        
        # Add some events
        for i in range(3):
            eq.put(MarketEvent(
                timestamp=datetime.now(),
                symbol=f"TEST{i}",
                open=100+i, high=105+i, low=95+i, close=102+i, volume=1000000
            ))
        
        # Get one event
        eq.get(block=False)
        assert eq.qsize() == 2
        
        # Add more events
        eq.put(SignalEvent(
            timestamp=datetime.now(),
            symbol="TEST",
            signal_type="LONG",
            strength=0.9
        ))
        assert eq.qsize() == 3
        
        # Clear and verify
        eq.clear()
        assert eq.empty() == True
        assert eq.qsize() == 0
    
    def test_event_abstract_base_class(self):
        """Test that Event is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            # This should fail because get_type is abstract
            Event(timestamp=datetime.now(), symbol="TEST")
    
    def test_event_inheritance(self):
        """Test that all event classes inherit from Event."""
        market_event = MarketEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            open=100, high=105, low=95, close=102, volume=1000000
        )
        signal_event = SignalEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type="LONG",
            strength=0.8
        )
        order_event = OrderEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            order_type="MARKET",
            quantity=100,
            direction="BUY"
        )
        fill_event = FillEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            fill_price=102.5,
            commission=1.0,
            slippage=0.1
        )
        
        # All should be instances of Event
        assert isinstance(market_event, Event)
        assert isinstance(signal_event, Event)
        assert isinstance(order_event, Event)
        assert isinstance(fill_event, Event)