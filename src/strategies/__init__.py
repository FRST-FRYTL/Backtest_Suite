"""Trading strategies module."""

from .base import BaseStrategy, TradeAction, Signal, Position, StrategyError
from .builder import StrategyBuilder
from .rules import Rule, Condition, LogicalOperator
from .signals import SignalGenerator

try:
    from .monthly_contribution_strategy import MonthlyContributionStrategy
    MONTHLY_STRATEGY_AVAILABLE = True
except ImportError:
    MONTHLY_STRATEGY_AVAILABLE = False

try:
    from .enhanced_confluence_engine import EnhancedConfluenceEngine
    ENHANCED_CONFLUENCE_AVAILABLE = True
except ImportError:
    ENHANCED_CONFLUENCE_AVAILABLE = False

__all__ = [
    "BaseStrategy",
    "TradeAction", 
    "Signal",
    "Position",
    "StrategyError",
    "StrategyBuilder",
    "Rule",
    "Condition",
    "LogicalOperator",
    "SignalGenerator"
]

if MONTHLY_STRATEGY_AVAILABLE:
    __all__.append("MonthlyContributionStrategy")

if ENHANCED_CONFLUENCE_AVAILABLE:
    __all__.append("EnhancedConfluenceEngine")