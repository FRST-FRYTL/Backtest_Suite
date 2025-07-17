"""Signal filters for strategy framework."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import time


class SignalFilter:
    """Filter for trading signals based on various criteria."""
    
    def __init__(
        self,
        filter_type: str,
        **kwargs
    ):
        """
        Initialize signal filter.
        
        Args:
            filter_type: Type of filter (volume, time, volatility, price, etc.)
            **kwargs: Filter-specific parameters
        """
        self.filter_type = filter_type
        self.params = kwargs
        
        # Validate filter parameters based on type
        self._validate_params()
    
    def _validate_params(self):
        """Validate filter parameters based on filter type."""
        if self.filter_type == "volume":
            if "min_volume" not in self.params and "max_volume" not in self.params:
                raise ValueError("Volume filter requires min_volume or max_volume")
                
        elif self.filter_type == "time":
            if "start_time" not in self.params and "end_time" not in self.params:
                raise ValueError("Time filter requires start_time or end_time")
                
        elif self.filter_type == "volatility":
            if "min_volatility" not in self.params and "max_volatility" not in self.params:
                raise ValueError("Volatility filter requires min_volatility or max_volatility")
                
        elif self.filter_type == "price":
            if "min_price" not in self.params and "max_price" not in self.params:
                raise ValueError("Price filter requires min_price or max_price")
    
    def apply(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Apply filter to signals.
        
        Args:
            data: Market data
            signals: Original signals
            
        Returns:
            Filtered signals
        """
        if self.filter_type == "volume":
            return self._apply_volume_filter(data, signals)
        elif self.filter_type == "time":
            return self._apply_time_filter(data, signals)
        elif self.filter_type == "volatility":
            return self._apply_volatility_filter(data, signals)
        elif self.filter_type == "price":
            return self._apply_price_filter(data, signals)
        else:
            return signals
    
    def _apply_volume_filter(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Apply volume filter."""
        mask = pd.Series(True, index=signals.index)
        
        if "min_volume" in self.params:
            mask &= data["volume"] >= self.params["min_volume"]
        
        if "max_volume" in self.params:
            mask &= data["volume"] <= self.params["max_volume"]
        
        return signals & mask
    
    def _apply_time_filter(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Apply time filter."""
        mask = pd.Series(True, index=signals.index)
        
        # Extract time from index if datetime
        if isinstance(data.index, pd.DatetimeIndex):
            times = data.index.time
            
            if "start_time" in self.params:
                start_time = pd.to_datetime(self.params["start_time"]).time()
                mask &= pd.Series([t >= start_time for t in times], index=data.index)
            
            if "end_time" in self.params:
                end_time = pd.to_datetime(self.params["end_time"]).time()
                mask &= pd.Series([t <= end_time for t in times], index=data.index)
        
        return signals & mask
    
    def _apply_volatility_filter(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Apply volatility filter."""
        mask = pd.Series(True, index=signals.index)
        
        # Calculate volatility if not present
        if "volatility" in data.columns:
            volatility = data["volatility"]
        else:
            # Simple volatility calculation
            returns = data["close"].pct_change()
            volatility = returns.rolling(window=20).std()
        
        if "min_volatility" in self.params:
            mask &= volatility >= self.params["min_volatility"]
        
        if "max_volatility" in self.params:
            mask &= volatility <= self.params["max_volatility"]
        
        return signals & mask
    
    def _apply_price_filter(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Apply price filter."""
        mask = pd.Series(True, index=signals.index)
        
        if "min_price" in self.params:
            mask &= data["close"] >= self.params["min_price"]
        
        if "max_price" in self.params:
            mask &= data["close"] <= self.params["max_price"]
        
        return signals & mask