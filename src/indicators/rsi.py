"""Relative Strength Index (RSI) indicator implementation."""

from typing import Optional

import pandas as pd
import numpy as np

from .base import Indicator


class RSI(Indicator):
    """
    Relative Strength Index (RSI) indicator.
    
    RSI measures momentum - whether a stock is overbought or oversold.
    Values range from 0 to 100, with levels typically at 30 (oversold) and 70 (overbought).
    """
    
    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0
    ):
        """
        Initialize RSI indicator.
        
        Args:
            period: Number of periods for RSI calculation (default: 14)
            overbought: Overbought level (default: 70)
            oversold: Oversold level (default: 30)
        """
        super().__init__(name="RSI")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
    def calculate(
        self,
        data: pd.DataFrame,
        price_column: str = "close"
    ) -> pd.Series:
        """
        Calculate RSI values.
        
        Args:
            data: DataFrame with price data
            price_column: Column to use for calculation (default: 'close')
            
        Returns:
            Series with RSI values
        """
        self.validate_data(data, [price_column])
        
        # Get price series
        prices = data[price_column].copy()
        
        # Calculate price changes
        deltas = prices.diff()
        
        # Separate gains and losses
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = self._calculate_average(gains, self.period)
        avg_losses = self._calculate_average(losses, self.period)
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Handle division by zero, but keep NaN for initial period
        # Only fill with 50 where we have valid data but division by zero
        rsi = rsi.where(avg_losses != 0, 50)  # 50 when no losses
        
        # Set proper series name
        rsi.name = 'rsi'
        
        return rsi
        
    def _calculate_average(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate average using Wilder's smoothing method.
        
        Args:
            series: Input series
            period: Period for averaging
            
        Returns:
            Averaged series
        """
        # First 'period' values should be NaN since we need at least 'period' values
        # for a meaningful RSI calculation
        result = series.ewm(alpha=1/period, adjust=False).mean()
        
        # Set the first 'period' values to NaN to match traditional RSI behavior
        result.iloc[:period] = np.nan
        
        return result
        
    def get_signals(self, rsi: pd.Series) -> pd.DataFrame:
        """
        Generate trading signals based on RSI levels.
        
        Args:
            rsi: RSI values
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=rsi.index)
        
        # Oversold signal (potential buy)
        signals['oversold'] = rsi < self.oversold
        
        # Overbought signal (potential sell)
        signals['overbought'] = rsi > self.overbought
        
        # Crossover signals
        signals['cross_above_oversold'] = (
            (rsi > self.oversold) & 
            (rsi.shift(1) <= self.oversold)
        )
        
        signals['cross_below_overbought'] = (
            (rsi < self.overbought) & 
            (rsi.shift(1) >= self.overbought)
        )
        
        # Centerline crossover
        signals['cross_above_50'] = (
            (rsi > 50) & 
            (rsi.shift(1) <= 50)
        )
        
        signals['cross_below_50'] = (
            (rsi < 50) & 
            (rsi.shift(1) >= 50)
        )
        
        return signals
        
    def divergence(
        self,
        price: pd.Series,
        rsi: pd.Series,
        window: int = 14
    ) -> pd.DataFrame:
        """
        Detect RSI divergences.
        
        Args:
            price: Price series
            rsi: RSI values
            window: Window for peak/trough detection
            
        Returns:
            DataFrame with divergence signals
        """
        divergences = pd.DataFrame(index=price.index)
        
        # Find price peaks and troughs
        price_peaks = self._find_peaks(price, window)
        price_troughs = self._find_troughs(price, window)
        
        # Find RSI peaks and troughs
        rsi_peaks = self._find_peaks(rsi, window)
        rsi_troughs = self._find_troughs(rsi, window)
        
        # Detect bearish divergence (price higher high, RSI lower high)
        divergences['bearish'] = False
        for i in range(1, len(price_peaks)):
            if (price_peaks.iloc[i] > price_peaks.iloc[i-1] and 
                rsi_peaks.iloc[i] < rsi_peaks.iloc[i-1]):
                divergences.loc[price_peaks.index[i], 'bearish'] = True
                
        # Detect bullish divergence (price lower low, RSI higher low)
        divergences['bullish'] = False
        for i in range(1, len(price_troughs)):
            if (price_troughs.iloc[i] < price_troughs.iloc[i-1] and 
                rsi_troughs.iloc[i] > rsi_troughs.iloc[i-1]):
                divergences.loc[price_troughs.index[i], 'bullish'] = True
                
        return divergences
        
    def _find_peaks(self, series: pd.Series, window: int) -> pd.Series:
        """Find local peaks in a series."""
        peaks = series.rolling(window=window*2+1, center=True).apply(
            lambda x: x[window] == x.max() if len(x) == window*2+1 else False
        )
        return series[peaks == 1]
        
    def _find_troughs(self, series: pd.Series, window: int) -> pd.Series:
        """Find local troughs in a series."""
        troughs = series.rolling(window=window*2+1, center=True).apply(
            lambda x: x[window] == x.min() if len(x) == window*2+1 else False
        )
        return series[troughs == 1]