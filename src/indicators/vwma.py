"""Volume Weighted Moving Average (VWMA) bands indicator implementation."""

from typing import Optional, Tuple

import pandas as pd
import numpy as np

from .base import Indicator


class VWMABands(Indicator):
    """
    Volume Weighted Moving Average (VWMA) with bands.
    
    VWMA gives more weight to periods with higher volume, making it more responsive
    to volume-driven price movements than simple moving averages.
    """
    
    def __init__(
        self,
        period: int = 20,
        band_multiplier: float = 2.0,
        price_column: str = "close"
    ):
        """
        Initialize VWMA Bands indicator.
        
        Args:
            period: Number of periods for VWMA calculation (default: 20)
            band_multiplier: Multiplier for standard deviation bands (default: 2.0)
            price_column: Price column to use (default: 'close')
        """
        super().__init__(name="VWMA_Bands")
        self.period = period
        self.band_multiplier = band_multiplier
        self.price_column = price_column
        
    def calculate(
        self,
        data: pd.DataFrame,
        volume_column: str = "volume"
    ) -> pd.DataFrame:
        """
        Calculate VWMA and bands.
        
        Args:
            data: DataFrame with price and volume data
            volume_column: Column name for volume (default: 'volume')
            
        Returns:
            DataFrame with VWMA, upper band, and lower band
        """
        self.validate_data(data, [self.price_column, volume_column])
        
        # Calculate VWMA
        vwma = self._calculate_vwma(
            data[self.price_column],
            data[volume_column]
        )
        
        # Calculate standard deviation of VWMA
        std_dev = self._calculate_rolling_std(
            data[self.price_column],
            data[volume_column],
            vwma
        )
        
        # Create bands
        result = pd.DataFrame(index=data.index)
        result['vwma'] = vwma
        result['vwma_upper'] = vwma + (std_dev * self.band_multiplier)
        result['vwma_lower'] = vwma - (std_dev * self.band_multiplier)
        result['vwma_width'] = result['vwma_upper'] - result['vwma_lower']
        
        # Add simple signal generation
        price = data[self.price_column]
        result['vwma_signal'] = 0  # Default neutral
        result.loc[price > result['vwma_upper'], 'vwma_signal'] = -1  # Sell signal
        result.loc[price < result['vwma_lower'], 'vwma_signal'] = 1   # Buy signal
        
        return result
        
    def _calculate_vwma(
        self,
        price: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate Volume Weighted Moving Average.
        
        Args:
            price: Price series
            volume: Volume series
            
        Returns:
            VWMA series
        """
        # Calculate price * volume
        pv = price * volume
        
        # Rolling sum of price*volume and volume
        pv_sum = pv.rolling(window=self.period, min_periods=1).sum()
        vol_sum = volume.rolling(window=self.period, min_periods=1).sum()
        
        # VWMA = sum(price * volume) / sum(volume)
        vwma = pv_sum / vol_sum
        
        # Handle division by zero (when volume is 0)
        vwma = vwma.fillna(price.rolling(window=self.period, min_periods=1).mean())
        
        return vwma
        
    def _calculate_rolling_std(
        self,
        price: pd.Series,
        volume: pd.Series,
        vwma: pd.Series
    ) -> pd.Series:
        """
        Calculate volume-weighted standard deviation.
        
        Args:
            price: Price series
            volume: Volume series
            vwma: VWMA series
            
        Returns:
            Volume-weighted standard deviation
        """
        # Calculate squared deviations from VWMA
        squared_dev = (price - vwma) ** 2
        
        # Weight by volume
        weighted_squared_dev = squared_dev * volume
        
        # Rolling sum of weighted squared deviations and volume
        wsd_sum = weighted_squared_dev.rolling(window=self.period, min_periods=1).sum()
        vol_sum = volume.rolling(window=self.period, min_periods=1).sum()
        
        # Calculate variance
        variance = wsd_sum / vol_sum
        
        # Standard deviation
        std_dev = np.sqrt(variance)
        
        # Handle NaN values
        std_dev = std_dev.fillna(price.rolling(window=self.period, min_periods=1).std())
        
        return std_dev
        
    def get_signals(
        self,
        data: pd.DataFrame,
        vwma_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate trading signals based on VWMA bands.
        
        Args:
            data: Original price data
            vwma_data: VWMA calculation results
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        price = data[self.price_column]
        
        # Band touch/break signals
        signals['touch_upper'] = price >= vwma_data['vwma_upper']
        signals['touch_lower'] = price <= vwma_data['vwma_lower']
        
        # Band crossover signals
        signals['cross_above_upper'] = (
            (price > vwma_data['vwma_upper']) & 
            (price.shift(1) <= vwma_data['vwma_upper'].shift(1))
        )
        
        signals['cross_below_lower'] = (
            (price < vwma_data['vwma_lower']) & 
            (price.shift(1) >= vwma_data['vwma_lower'].shift(1))
        )
        
        # VWMA crossover signals
        signals['cross_above_vwma'] = (
            (price > vwma_data['vwma']) & 
            (price.shift(1) <= vwma_data['vwma'].shift(1))
        )
        
        signals['cross_below_vwma'] = (
            (price < vwma_data['vwma']) & 
            (price.shift(1) >= vwma_data['vwma'].shift(1))
        )
        
        # Band squeeze signal (volatility contraction)
        band_width_ma = vwma_data['vwma_width'].rolling(window=self.period).mean()
        # Ensure valid comparison by filling NaN values
        band_width_filled = vwma_data['vwma_width'].bfill().ffill()
        band_width_ma_filled = band_width_ma.bfill().ffill()
        signals['band_squeeze'] = band_width_filled < band_width_ma_filled * 0.8
        
        # Band expansion signal (volatility expansion)
        signals['band_expansion'] = band_width_filled > band_width_ma_filled * 1.2
        
        return signals
        
    def calculate_percent_b(
        self,
        price: pd.Series,
        vwma_data: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate %B indicator (position within bands).
        
        Args:
            price: Price series
            vwma_data: VWMA calculation results
            
        Returns:
            %B series (0 = lower band, 1 = upper band)
        """
        return (price - vwma_data['vwma_lower']) / (
            vwma_data['vwma_upper'] - vwma_data['vwma_lower']
        )
        
    def volume_confirmation(
        self,
        data: pd.DataFrame,
        vwma_data: pd.DataFrame,
        volume_column: str = "volume"
    ) -> pd.DataFrame:
        """
        Check for volume confirmation of price movements.
        
        Args:
            data: Original price and volume data
            vwma_data: VWMA calculation results
            volume_column: Volume column name
            
        Returns:
            DataFrame with volume confirmation signals
        """
        signals = pd.DataFrame(index=data.index)
        
        price = data[self.price_column]
        volume = data[volume_column]
        
        # Calculate average volume
        avg_volume = volume.rolling(window=self.period).mean()
        
        # Price above VWMA with high volume
        signals['bullish_volume'] = (
            (price > vwma_data['vwma']) & 
            (volume > avg_volume * 1.5)
        )
        
        # Price below VWMA with high volume
        signals['bearish_volume'] = (
            (price < vwma_data['vwma']) & 
            (volume > avg_volume * 1.5)
        )
        
        # Price movement without volume (potential false move)
        signals['low_volume_move'] = (
            (abs(price.pct_change()) > 0.02) & 
            (volume < avg_volume * 0.7)
        )
        
        return signals