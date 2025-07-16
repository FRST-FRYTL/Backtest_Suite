"""Data fetching and management module."""

from .fetcher import StockDataFetcher
from .cache import DataCache

# For backward compatibility
DataFetcher = StockDataFetcher
CacheManager = DataCache
DataSource = StockDataFetcher  # Create alias for DataSource

try:
    from .multi_timeframe_data_manager import MultiTimeframeDataManager, Timeframe, TimeframeConfig
    MULTI_TIMEFRAME_AVAILABLE = True
except ImportError:
    MULTI_TIMEFRAME_AVAILABLE = False

__all__ = ["StockDataFetcher", "DataCache", "DataFetcher", "CacheManager", "DataSource"]

if MULTI_TIMEFRAME_AVAILABLE:
    __all__.extend(["MultiTimeframeDataManager", "Timeframe", "TimeframeConfig"])