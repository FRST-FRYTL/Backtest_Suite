"""Base strategy class for all trading strategies."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np


class TradeAction(Enum):
    """Trade action types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    """Trading signal."""
    
    timestamp: pd.Timestamp
    action: TradeAction
    symbol: str
    confidence: float = 1.0
    quantity: Optional[int] = None
    price: Optional[float] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Position:
    """Position information."""
    
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    timestamp: pd.Timestamp
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.quantity * (self.current_price - self.entry_price)
    
    @property
    def unrealized_return(self) -> float:
        """Unrealized return percentage."""
        return (self.current_price - self.entry_price) / self.entry_price


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, name: str, parameters: Dict = None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self.positions: Dict[str, Position] = {}
        self.signals_history: List[Signal] = []
        self.is_initialized = False
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on market data.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            List of trading signals
        """
        pass
    
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Initialize strategy with historical data.
        
        Args:
            data: Historical market data
        """
        self.is_initialized = True
        
    def update_position(self, symbol: str, position: Position) -> None:
        """
        Update position information.
        
        Args:
            symbol: Symbol identifier
            position: Position object
        """
        self.positions[symbol] = position
        
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get current position for symbol.
        
        Args:
            symbol: Symbol identifier
            
        Returns:
            Position object or None
        """
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """
        Check if strategy has position in symbol.
        
        Args:
            symbol: Symbol identifier
            
        Returns:
            True if position exists
        """
        return symbol in self.positions
    
    def close_position(self, symbol: str) -> None:
        """
        Close position for symbol.
        
        Args:
            symbol: Symbol identifier
        """
        if symbol in self.positions:
            del self.positions[symbol]
            
    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value.
        
        Returns:
            Total market value of all positions
        """
        return sum(pos.market_value for pos in self.positions.values())
    
    def get_unrealized_pnl(self) -> float:
        """
        Get total unrealized P&L.
        
        Returns:
            Total unrealized profit/loss
        """
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def add_signal(self, signal: Signal) -> None:
        """
        Add signal to history.
        
        Args:
            signal: Signal object
        """
        self.signals_history.append(signal)
        
    def get_signals_history(self) -> List[Signal]:
        """
        Get signal history.
        
        Returns:
            List of historical signals
        """
        return self.signals_history.copy()
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.positions.clear()
        self.signals_history.clear()
        self.is_initialized = False
        
    def get_parameters(self) -> Dict:
        """
        Get strategy parameters.
        
        Returns:
            Dictionary of parameters
        """
        return self.parameters.copy()
    
    def set_parameters(self, parameters: Dict) -> None:
        """
        Set strategy parameters.
        
        Args:
            parameters: New parameters
        """
        self.parameters.update(parameters)
        
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        
        Returns:
            True if parameters are valid
        """
        return True
    
    def get_state(self) -> Dict:
        """
        Get strategy state for serialization.
        
        Returns:
            Dictionary with strategy state
        """
        return {
            'name': self.name,
            'parameters': self.parameters,
            'positions': {k: {
                'symbol': v.symbol,
                'quantity': v.quantity,
                'entry_price': v.entry_price,
                'current_price': v.current_price,
                'timestamp': v.timestamp.isoformat()
            } for k, v in self.positions.items()},
            'signals_count': len(self.signals_history),
            'is_initialized': self.is_initialized
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name}({self.parameters})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"BaseStrategy(name='{self.name}', parameters={self.parameters})"


class StrategyError(Exception):
    """Exception raised for strategy-related errors."""
    pass


class ParameterError(StrategyError):
    """Exception raised for parameter validation errors."""
    pass


class PositionError(StrategyError):
    """Exception raised for position management errors."""
    pass