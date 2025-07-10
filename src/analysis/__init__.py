"""Analysis module for the backtest suite."""

try:
    from .baseline_comparisons import BaselineComparison, BaselineResults
    BASELINE_AVAILABLE = True
except ImportError:
    BASELINE_AVAILABLE = False

try:
    from .enhanced_trade_tracker import EnhancedTradeTracker, TradeAnalysis, TradeEntry, TradeExit
    ENHANCED_TRACKER_AVAILABLE = True
except ImportError:
    ENHANCED_TRACKER_AVAILABLE = False

__all__ = []

if BASELINE_AVAILABLE:
    __all__.extend(["BaselineComparison", "BaselineResults"])

if ENHANCED_TRACKER_AVAILABLE:
    __all__.extend(["EnhancedTradeTracker", "TradeAnalysis", "TradeEntry", "TradeExit"])