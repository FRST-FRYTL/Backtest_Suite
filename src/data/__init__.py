"""Data fetching and management module."""

from .fetcher import StockDataFetcher
from .cache import DataCache

try:
    from .multi_timeframe_data_manager import MultiTimeframeDataManager, Timeframe, TimeframeConfig
    MULTI_TIMEFRAME_AVAILABLE = True
except ImportError:
    MULTI_TIMEFRAME_AVAILABLE = False

__all__ = ["StockDataFetcher", "DataCache"]

if MULTI_TIMEFRAME_AVAILABLE:
    __all__.extend(["MultiTimeframeDataManager", "Timeframe", "TimeframeConfig"])