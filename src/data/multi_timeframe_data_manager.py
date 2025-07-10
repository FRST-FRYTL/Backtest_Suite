"""
Multi-Timeframe Data Manager for Enhanced Confluence Strategy

This module provides comprehensive multi-timeframe data loading, alignment,
and synchronization capabilities for the enhanced confluence strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
import yfinance as yf
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Timeframe(Enum):
    """Supported timeframes for multi-timeframe analysis"""
    HOUR_1 = "1H"
    HOUR_4 = "4H"
    DAY_1 = "1D"
    WEEK_1 = "1W"
    MONTH_1 = "1M"

@dataclass
class TimeframeConfig:
    """Configuration for timeframe hierarchy and weights"""
    timeframe: Timeframe
    weight: float
    description: str
    yfinance_interval: str

class MultiTimeframeDataManager:
    """
    Manages multi-timeframe data loading, alignment, and synchronization
    for enhanced confluence strategy analysis.
    """
    
    # Timeframe configuration with weights for confluence scoring
    TIMEFRAME_CONFIGS = {
        Timeframe.MONTH_1: TimeframeConfig(
            timeframe=Timeframe.MONTH_1,
            weight=0.35,
            description="Monthly macro trend",
            yfinance_interval="1mo"
        ),
        Timeframe.WEEK_1: TimeframeConfig(
            timeframe=Timeframe.WEEK_1,
            weight=0.25,
            description="Weekly structural confirmation",
            yfinance_interval="1wk"
        ),
        Timeframe.DAY_1: TimeframeConfig(
            timeframe=Timeframe.DAY_1,
            weight=0.20,
            description="Daily execution timeframe",
            yfinance_interval="1d"
        ),
        Timeframe.HOUR_4: TimeframeConfig(
            timeframe=Timeframe.HOUR_4,
            weight=0.15,
            description="Short-term momentum",
            yfinance_interval="4h"
        ),
        Timeframe.HOUR_1: TimeframeConfig(
            timeframe=Timeframe.HOUR_1,
            weight=0.05,
            description="Micro-structure timing",
            yfinance_interval="1h"
        )
    }
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the multi-timeframe data manager.
        
        Args:
            data_dir: Directory to store cached data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cached_data: Dict[str, Dict[Timeframe, pd.DataFrame]] = {}
        
    async def load_multi_timeframe_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframes: Optional[List[Timeframe]] = None
    ) -> Dict[str, Dict[Timeframe, pd.DataFrame]]:
        """
        Load multi-timeframe data for specified symbols.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframes: List of timeframes to load (default: all)
            
        Returns:
            Dictionary mapping symbols to timeframe data
        """
        if timeframes is None:
            timeframes = list(self.TIMEFRAME_CONFIGS.keys())
            
        logger.info(f"Loading multi-timeframe data for {len(symbols)} symbols across {len(timeframes)} timeframes")
        
        # Create tasks for parallel data loading
        tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                task = self._load_single_timeframe_data(symbol, timeframe, start_date, end_date)
                tasks.append((symbol, timeframe, task))
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*[task for _, _, task in tasks])
        
        # Organize results by symbol and timeframe
        multi_timeframe_data = {}
        for (symbol, timeframe, _), data in zip(tasks, results):
            if symbol not in multi_timeframe_data:
                multi_timeframe_data[symbol] = {}
            multi_timeframe_data[symbol][timeframe] = data
            
        # Cache the results
        self.cached_data.update(multi_timeframe_data)
        
        return multi_timeframe_data
    
    async def _load_single_timeframe_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Load data for a single symbol and timeframe.
        
        Args:
            symbol: Symbol to load
            timeframe: Timeframe to load
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_file = self.data_dir / f"{symbol}_{timeframe.value}_{start_date}_{end_date}.csv"
        
        # Try to load from cache first
        if cache_file.exists():
            try:
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                logger.info(f"Loaded cached data for {symbol} {timeframe.value}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cached data for {symbol} {timeframe.value}: {e}")
        
        # Download fresh data
        try:
            config = self.TIMEFRAME_CONFIGS[timeframe]
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=config.yfinance_interval,
                auto_adjust=True,
                prepost=True
            )
            
            # Standardize column names
            data.columns = [col.replace(' ', '_').lower() for col in data.columns]
            
            # Add typical price and ensure required columns
            data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
            
            # Cache the data
            data.to_csv(cache_file)
            logger.info(f"Downloaded and cached data for {symbol} {timeframe.value}: {len(data)} rows")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to download data for {symbol} {timeframe.value}: {e}")
            return pd.DataFrame()
    
    def align_timeframes(
        self,
        data_by_timeframe: Dict[Timeframe, pd.DataFrame],
        reference_timeframe: Timeframe = Timeframe.DAY_1
    ) -> Dict[Timeframe, pd.DataFrame]:
        """
        Align multiple timeframes to a reference timeframe for confluence analysis.
        
        Args:
            data_by_timeframe: Dictionary mapping timeframes to DataFrames
            reference_timeframe: Reference timeframe for alignment
            
        Returns:
            Dictionary of aligned timeframe data
        """
        if reference_timeframe not in data_by_timeframe:
            raise ValueError(f"Reference timeframe {reference_timeframe} not found in data")
        
        reference_data = data_by_timeframe[reference_timeframe]
        aligned_data = {reference_timeframe: reference_data.copy()}
        
        for timeframe, data in data_by_timeframe.items():
            if timeframe == reference_timeframe:
                continue
                
            # Align each timeframe to the reference timeframe index
            aligned_data[timeframe] = self._align_single_timeframe(
                data, reference_data.index, timeframe
            )
        
        return aligned_data
    
    def _align_single_timeframe(
        self,
        data: pd.DataFrame,
        reference_index: pd.DatetimeIndex,
        timeframe: Timeframe
    ) -> pd.DataFrame:
        """
        Align a single timeframe to a reference index.
        
        Args:
            data: DataFrame to align
            reference_index: Reference datetime index
            timeframe: Timeframe being aligned
            
        Returns:
            Aligned DataFrame
        """
        # Use forward fill for higher timeframes (monthly, weekly)
        # Use backward fill for lower timeframes (hourly)
        if timeframe in [Timeframe.MONTH_1, Timeframe.WEEK_1]:
            # Forward fill for higher timeframes
            aligned = data.reindex(reference_index, method='ffill')
        else:
            # Backward fill for lower timeframes
            aligned = data.reindex(reference_index, method='bfill')
        
        return aligned
    
    def get_timeframe_weights(self) -> Dict[Timeframe, float]:
        """
        Get the weights for each timeframe in confluence scoring.
        
        Returns:
            Dictionary mapping timeframes to their weights
        """
        return {tf: config.weight for tf, config in self.TIMEFRAME_CONFIGS.items()}
    
    def validate_data_quality(
        self,
        data_by_timeframe: Dict[Timeframe, pd.DataFrame],
        min_data_points: int = 100
    ) -> Dict[Timeframe, Dict[str, Union[bool, int, float]]]:
        """
        Validate data quality across timeframes.
        
        Args:
            data_by_timeframe: Dictionary of timeframe data
            min_data_points: Minimum required data points
            
        Returns:
            Dictionary of validation results for each timeframe
        """
        validation_results = {}
        
        for timeframe, data in data_by_timeframe.items():
            results = {
                'sufficient_data': len(data) >= min_data_points,
                'data_points': len(data),
                'missing_values': data.isnull().sum().sum(),
                'missing_percentage': (data.isnull().sum().sum() / data.size) * 100,
                'date_range': {
                    'start': data.index.min(),
                    'end': data.index.max()
                },
                'columns_present': list(data.columns),
                'price_data_valid': all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']),
                'no_negative_prices': (data[['open', 'high', 'low', 'close']] >= 0).all().all(),
                'high_low_consistency': (data['high'] >= data['low']).all(),
                'ohlc_consistency': (
                    (data['high'] >= data['open']) & 
                    (data['high'] >= data['close']) & 
                    (data['low'] <= data['open']) & 
                    (data['low'] <= data['close'])
                ).all()
            }
            
            validation_results[timeframe] = results
        
        return validation_results
    
    def get_synchronized_data(
        self,
        symbol: str,
        timeframes: Optional[List[Timeframe]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[Timeframe, pd.DataFrame]:
        """
        Get synchronized data for a symbol across specified timeframes.
        
        Args:
            symbol: Symbol to retrieve
            timeframes: List of timeframes (default: all)
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary of synchronized timeframe data
        """
        if symbol not in self.cached_data:
            raise ValueError(f"Data for symbol {symbol} not found. Load data first.")
        
        if timeframes is None:
            timeframes = list(self.TIMEFRAME_CONFIGS.keys())
        
        # Get data for specified timeframes
        data_by_timeframe = {}
        for timeframe in timeframes:
            if timeframe in self.cached_data[symbol]:
                data = self.cached_data[symbol][timeframe].copy()
                
                # Apply date filters if specified
                if start_date:
                    data = data[data.index >= start_date]
                if end_date:
                    data = data[data.index <= end_date]
                
                data_by_timeframe[timeframe] = data
        
        # Align timeframes for confluence analysis
        aligned_data = self.align_timeframes(data_by_timeframe)
        
        return aligned_data
    
    def calculate_timeframe_summary_stats(
        self,
        data_by_timeframe: Dict[Timeframe, pd.DataFrame]
    ) -> Dict[Timeframe, Dict[str, float]]:
        """
        Calculate summary statistics for each timeframe.
        
        Args:
            data_by_timeframe: Dictionary of timeframe data
            
        Returns:
            Dictionary of summary statistics for each timeframe
        """
        summary_stats = {}
        
        for timeframe, data in data_by_timeframe.items():
            if data.empty:
                continue
                
            # Calculate returns
            returns = data['close'].pct_change().dropna()
            
            stats = {
                'total_return': (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100,
                'annualized_return': returns.mean() * 252 * 100,  # Assuming daily data
                'volatility': returns.std() * np.sqrt(252) * 100,
                'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(data['close']),
                'avg_volume': data['volume'].mean(),
                'price_range': {
                    'min': data['low'].min(),
                    'max': data['high'].max(),
                    'current': data['close'].iloc[-1]
                }
            }
            
            summary_stats[timeframe] = stats
        
        return summary_stats
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """
        Calculate maximum drawdown for a price series.
        
        Args:
            prices: Series of prices
            
        Returns:
            Maximum drawdown percentage
        """
        cumulative = (1 + prices.pct_change()).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max - 1) * 100
        return drawdown.min()