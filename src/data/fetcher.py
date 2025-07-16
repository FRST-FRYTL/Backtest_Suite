"""Stock data fetcher with yfinance integration and caching."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf
from aiohttp import ClientSession

from .cache import DataCache

logger = logging.getLogger(__name__)


class DataError(Exception):
    """Base exception for data-related errors."""
    pass


class DataSource:
    """Abstract base class for data sources."""
    
    def __init__(self, name: str):
        """Initialize data source.
        
        Args:
            name: Name of the data source
        """
        self.name = name
        
    def fetch_data(self, symbol: str, start: str, end: str, **kwargs) -> pd.DataFrame:
        """Fetch data from the source.
        
        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with stock data
        """
        raise NotImplementedError("Subclasses must implement fetch_data")
        
    def is_available(self) -> bool:
        """Check if data source is available."""
        return True


class StockDataFetcher:
    """Fetches stock data with caching and retry logic."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize the stock data fetcher.
        
        Args:
            cache_dir: Directory for caching data
        """
        self.cache = DataCache(cache_dir)
        self._session: Optional[ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            
    async def fetch(
        self,
        symbol: str,
        start: Union[str, datetime],
        end: Union[str, datetime] = None,
        interval: str = "1d",
        prepost: bool = False,
        actions: bool = True,
        auto_adjust: bool = True,
        back_adjust: bool = False,
        repair: bool = True,
        keepna: bool = False,
        proxy: Optional[str] = None,
        rounding: bool = False,
        timeout: Optional[float] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch stock data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol
            start: Start date for data
            end: End date for data (default: today)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            prepost: Include pre and post market data
            actions: Download stock dividends and stock splits
            auto_adjust: Adjust all OHLC automatically
            back_adjust: Back-adjusted data to mimic true historical prices
            repair: Repair missing data
            keepna: Keep NaN rows
            proxy: Proxy URL
            rounding: Round values to 2 decimal places
            timeout: Request timeout
            **kwargs: Additional arguments for yfinance
            
        Returns:
            DataFrame with stock data
        """
        # Convert dates to strings if needed
        if isinstance(start, datetime):
            start = start.strftime("%Y-%m-%d")
        if isinstance(end, datetime):
            end = end.strftime("%Y-%m-%d")
        elif end is None:
            end = datetime.now().strftime("%Y-%m-%d")
            
        # Check cache first
        cache_key = f"{symbol}_{start}_{end}_{interval}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.info(f"Using cached data for {symbol}")
            return cached_data
            
        # Fetch from yfinance
        logger.info(f"Fetching data for {symbol} from {start} to {end}")
        
        ticker = yf.Ticker(symbol)
        
        try:
            data = ticker.history(
                start=start,
                end=end,
                interval=interval,
                prepost=prepost,
                actions=actions,
                auto_adjust=auto_adjust,
                back_adjust=back_adjust,
                repair=repair,
                keepna=keepna,
                proxy=proxy,
                rounding=rounding,
                timeout=timeout,
                **kwargs
            )
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
                
            # Cache the data
            self.cache.set(cache_key, data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise
            
    async def fetch_multiple(
        self,
        symbols: List[str],
        start: Union[str, datetime],
        end: Union[str, datetime] = None,
        interval: str = "1d",
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols concurrently.
        
        Args:
            symbols: List of stock ticker symbols
            start: Start date for data
            end: End date for data
            interval: Data interval
            **kwargs: Additional arguments for fetch
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        tasks = [
            self.fetch(symbol, start, end, interval, **kwargs)
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {symbol}: {result}")
            else:
                data[symbol] = result
                
        return data
        
    def fetch_sync(
        self,
        symbol: str,
        start: Union[str, datetime],
        end: Union[str, datetime] = None,
        interval: str = "1d",
        **kwargs
    ) -> pd.DataFrame:
        """
        Synchronous version of fetch for convenience.
        
        Args:
            symbol: Stock ticker symbol
            start: Start date for data
            end: End date for data
            interval: Data interval
            **kwargs: Additional arguments for fetch
            
        Returns:
            DataFrame with stock data
        """
        return asyncio.run(self.fetch(symbol, start, end, interval, **kwargs))
        
    def get_info(self, symbol: str) -> Dict:
        """
        Get stock information.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with stock information
        """
        ticker = yf.Ticker(symbol)
        return ticker.info
        
    def get_options_chain(self, symbol: str, date: Optional[str] = None) -> tuple:
        """
        Get options chain data.
        
        Args:
            symbol: Stock ticker symbol
            date: Option expiration date (default: nearest expiration)
            
        Returns:
            Tuple of (calls DataFrame, puts DataFrame)
        """
        ticker = yf.Ticker(symbol)
        
        if date is None:
            # Get the nearest expiration date
            expirations = ticker.options
            if not expirations:
                raise ValueError(f"No options data available for {symbol}")
            date = expirations[0]
            
        opt = ticker.option_chain(date)
        return opt.calls, opt.puts
        
    def get_dividends(self, symbol: str) -> pd.Series:
        """
        Get dividend history.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Series with dividend data
        """
        ticker = yf.Ticker(symbol)
        return ticker.dividends
        
    def get_splits(self, symbol: str) -> pd.Series:
        """
        Get stock split history.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Series with split data
        """
        ticker = yf.Ticker(symbol)
        return ticker.splits