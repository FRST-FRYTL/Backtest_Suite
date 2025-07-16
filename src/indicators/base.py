"""Base indicator class for all technical indicators."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import pandas as pd
import numpy as np


class IndicatorError(Exception):
    """Exception raised for indicator calculation errors."""
    pass


class Indicator(ABC):
    """Abstract base class for all indicators."""
    
    def __init__(self, name: str):
        """
        Initialize the indicator.
        
        Args:
            name: Indicator name
        """
        self.name = name
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate the indicator values.
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters
            
        Returns:
            Series or DataFrame with indicator values
        """
        pass
        
    def validate_data(self, data: pd.DataFrame, required_columns: list = None) -> None:
        """
        Validate that required columns exist in the data.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            
        Raises:
            IndicatorError: If required columns are missing or data is invalid
        """
        if data is None or data.empty:
            raise IndicatorError("Data is empty or None")
            
        if required_columns is not None:
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                raise IndicatorError(f"Missing required columns: {missing_columns}")
            
    @staticmethod
    def rolling_window(series: pd.Series, window: int, func=None) -> pd.Series:
        """
        Apply a function over a rolling window.
        
        Args:
            series: Input series
            window: Window size
            func: Function to apply (default: mean)
            
        Returns:
            Series with rolling calculation
        """
        if func is None:
            return series.rolling(window=window).mean()
        return series.rolling(window=window).apply(func)
        
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            series: Input series
            period: EMA period
            
        Returns:
            EMA series
        """
        return series.ewm(span=period, adjust=False).mean()
        
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            series: Input series
            period: SMA period
            
        Returns:
            SMA series
        """
        return series.rolling(window=period).mean()
        
    @staticmethod
    def std_dev(series: pd.Series, period: int) -> pd.Series:
        """
        Calculate rolling standard deviation.
        
        Args:
            series: Input series
            period: Period for calculation
            
        Returns:
            Standard deviation series
        """
        return series.rolling(window=period).std()
        
    @staticmethod
    def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Calculate True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            True Range series
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
    @staticmethod
    def typical_price(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Calculate typical price (HLC/3).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            Typical price series
        """
        return (high + low + close) / 3