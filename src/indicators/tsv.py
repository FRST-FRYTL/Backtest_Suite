"""Time Segmented Volume (TSV) indicator implementation."""

from typing import Optional

import pandas as pd
import numpy as np

from .base import Indicator


class TSV(Indicator):
    """
    Time Segmented Volume (TSV) indicator.
    
    TSV is a technical indicator that combines price and volume to assess
    the flow of money into and out of a security. It's similar to On Balance Volume
    but uses intraday price changes.
    """
    
    def __init__(
        self,
        period: int = 13,
        signal_period: int = 9
    ):
        """
        Initialize TSV indicator.
        
        Args:
            period: Number of periods for TSV calculation (default: 13)
            signal_period: Period for signal line EMA (default: 9)
        """
        super().__init__(name="TSV")
        self.period = period
        self.signal_period = signal_period
        
    def calculate(
        self,
        data: pd.DataFrame,
        price_column: str = "close",
        volume_column: str = "volume"
    ) -> pd.DataFrame:
        """
        Calculate TSV and signal line.
        
        Args:
            data: DataFrame with price and volume data
            price_column: Column to use for price (default: 'close')
            volume_column: Column to use for volume (default: 'volume')
            
        Returns:
            DataFrame with TSV and signal line
        """
        self.validate_data(data, [price_column, volume_column])
        
        # Calculate raw TSV
        raw_tsv = self._calculate_raw_tsv(
            data[price_column],
            data[volume_column]
        )
        
        # Calculate TSV (moving average of raw TSV)
        tsv = raw_tsv.rolling(window=self.period, min_periods=1).sum()
        
        # Calculate signal line (EMA of TSV)
        signal = self.ema(tsv, self.signal_period)
        
        # Create result DataFrame
        result = pd.DataFrame(index=data.index)
        result['tsv'] = tsv
        result['tsv_signal'] = signal
        result['tsv_histogram'] = tsv - signal
        result['tsv_raw'] = raw_tsv
        
        return result
        
    def _calculate_raw_tsv(
        self,
        price: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate raw TSV values.
        
        Args:
            price: Price series
            volume: Volume series
            
        Returns:
            Raw TSV series
        """
        # Calculate price change
        price_change = price.diff()
        
        # TSV = Volume when price increases, -Volume when price decreases
        raw_tsv = volume.copy()
        raw_tsv[price_change < 0] *= -1
        raw_tsv[price_change == 0] = 0
        
        return raw_tsv
        
    def get_signals(
        self,
        tsv_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate trading signals based on TSV.
        
        Args:
            tsv_data: TSV calculation results
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=tsv_data.index)
        
        # TSV/Signal crossover signals
        signals['tsv_cross_above_signal'] = (
            (tsv_data['tsv'] > tsv_data['tsv_signal']) & 
            (tsv_data['tsv'].shift(1) <= tsv_data['tsv_signal'].shift(1))
        )
        
        signals['tsv_cross_below_signal'] = (
            (tsv_data['tsv'] < tsv_data['tsv_signal']) & 
            (tsv_data['tsv'].shift(1) >= tsv_data['tsv_signal'].shift(1))
        )
        
        # Zero line crossover signals
        signals['tsv_cross_above_zero'] = (
            (tsv_data['tsv'] > 0) & 
            (tsv_data['tsv'].shift(1) <= 0)
        )
        
        signals['tsv_cross_below_zero'] = (
            (tsv_data['tsv'] < 0) & 
            (tsv_data['tsv'].shift(1) >= 0)
        )
        
        # Divergence from histogram
        signals['histogram_expanding_positive'] = (
            (tsv_data['tsv_histogram'] > 0) & 
            (tsv_data['tsv_histogram'] > tsv_data['tsv_histogram'].shift(1))
        )
        
        signals['histogram_expanding_negative'] = (
            (tsv_data['tsv_histogram'] < 0) & 
            (tsv_data['tsv_histogram'] < tsv_data['tsv_histogram'].shift(1))
        )
        
        # Accumulation/Distribution phases
        tsv_ma = tsv_data['tsv'].rolling(window=20).mean()
        signals['accumulation'] = tsv_data['tsv'] > tsv_ma
        signals['distribution'] = tsv_data['tsv'] < tsv_ma
        
        return signals
        
    def detect_divergence(
        self,
        price: pd.Series,
        tsv_data: pd.DataFrame,
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Detect divergences between price and TSV.
        
        Args:
            price: Price series
            tsv_data: TSV calculation results
            lookback: Number of periods to look back
            
        Returns:
            DataFrame with divergence signals
        """
        divergences = pd.DataFrame(index=price.index)
        
        # Calculate rolling highs and lows
        price_high = price.rolling(window=lookback).max()
        price_low = price.rolling(window=lookback).min()
        tsv_high = tsv_data['tsv'].rolling(window=lookback).max()
        tsv_low = tsv_data['tsv'].rolling(window=lookback).min()
        
        # Bearish divergence: price makes higher high, TSV makes lower high
        divergences['bearish_divergence'] = (
            (price == price_high) & 
            (price > price_high.shift(lookback)) &
            (tsv_data['tsv'] < tsv_high.shift(lookback))
        )
        
        # Bullish divergence: price makes lower low, TSV makes higher low
        divergences['bullish_divergence'] = (
            (price == price_low) & 
            (price < price_low.shift(lookback)) &
            (tsv_data['tsv'] > tsv_low.shift(lookback))
        )
        
        # Hidden bearish divergence: price makes lower high, TSV makes higher high
        divergences['hidden_bearish'] = (
            (price == price_high) & 
            (price < price_high.shift(lookback)) &
            (tsv_data['tsv'] > tsv_high.shift(lookback))
        )
        
        # Hidden bullish divergence: price makes higher low, TSV makes lower low
        divergences['hidden_bullish'] = (
            (price == price_low) & 
            (price > price_low.shift(lookback)) &
            (tsv_data['tsv'] < tsv_low.shift(lookback))
        )
        
        return divergences
        
    def calculate_momentum(
        self,
        tsv_data: pd.DataFrame,
        momentum_period: int = 10
    ) -> pd.Series:
        """
        Calculate TSV momentum.
        
        Args:
            tsv_data: TSV calculation results
            momentum_period: Period for momentum calculation
            
        Returns:
            TSV momentum series
        """
        return tsv_data['tsv'].diff(momentum_period)
        
    def volume_analysis(
        self,
        data: pd.DataFrame,
        tsv_data: pd.DataFrame,
        volume_column: str = "volume"
    ) -> pd.DataFrame:
        """
        Analyze volume patterns with TSV.
        
        Args:
            data: Original price and volume data
            tsv_data: TSV calculation results
            volume_column: Volume column name
            
        Returns:
            DataFrame with volume analysis
        """
        analysis = pd.DataFrame(index=data.index)
        
        volume = data[volume_column]
        avg_volume = volume.rolling(window=20).mean()
        
        # High volume with positive TSV (strong buying)
        analysis['strong_buying'] = (
            (volume > avg_volume * 1.5) & 
            (tsv_data['tsv'] > 0) &
            (tsv_data['tsv'] > tsv_data['tsv'].shift(1))
        )
        
        # High volume with negative TSV (strong selling)
        analysis['strong_selling'] = (
            (volume > avg_volume * 1.5) & 
            (tsv_data['tsv'] < 0) &
            (tsv_data['tsv'] < tsv_data['tsv'].shift(1))
        )
        
        # Low volume with TSV movement (weak move)
        analysis['weak_move'] = (
            (volume < avg_volume * 0.7) & 
            (abs(tsv_data['tsv'].pct_change()) > 0.05)
        )
        
        # Volume climax (extreme volume with TSV reversal)
        volume_extreme = volume > volume.rolling(window=50).quantile(0.95)
        tsv_reversal = (
            (tsv_data['tsv'] > 0) & (tsv_data['tsv'].shift(-1) < 0)
        ) | (
            (tsv_data['tsv'] < 0) & (tsv_data['tsv'].shift(-1) > 0)
        )
        
        analysis['volume_climax'] = volume_extreme & tsv_reversal
        
        return analysis