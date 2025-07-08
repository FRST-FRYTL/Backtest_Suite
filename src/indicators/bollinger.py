"""Bollinger Bands indicator implementation."""

from typing import Optional

import pandas as pd
import numpy as np

from .base import Indicator


class BollingerBands(Indicator):
    """
    Bollinger Bands indicator.
    
    Bollinger Bands consist of a middle band (SMA) and two outer bands
    (standard deviations above and below the middle band).
    """
    
    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        ma_type: str = "sma"
    ):
        """
        Initialize Bollinger Bands indicator.
        
        Args:
            period: Number of periods for moving average (default: 20)
            std_dev: Number of standard deviations for bands (default: 2.0)
            ma_type: Type of moving average ('sma' or 'ema', default: 'sma')
        """
        super().__init__(name="Bollinger_Bands")
        self.period = period
        self.std_dev = std_dev
        self.ma_type = ma_type.lower()
        
        if self.ma_type not in ['sma', 'ema']:
            raise ValueError(f"Invalid ma_type: {ma_type}. Must be 'sma' or 'ema'")
            
    def calculate(
        self,
        data: pd.DataFrame,
        price_column: str = "close"
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with price data
            price_column: Column to use for calculation (default: 'close')
            
        Returns:
            DataFrame with middle, upper, and lower bands
        """
        self.validate_data(data, [price_column])
        
        prices = data[price_column]
        
        # Calculate middle band (moving average)
        if self.ma_type == 'sma':
            middle_band = self.sma(prices, self.period)
        else:  # ema
            middle_band = self.ema(prices, self.period)
            
        # Calculate standard deviation
        rolling_std = prices.rolling(window=self.period).std()
        
        # Create bands
        result = pd.DataFrame(index=data.index)
        result['bb_middle'] = middle_band
        result['bb_upper'] = middle_band + (rolling_std * self.std_dev)
        result['bb_lower'] = middle_band - (rolling_std * self.std_dev)
        result['bb_width'] = result['bb_upper'] - result['bb_lower']
        result['bb_percent'] = (prices - result['bb_lower']) / result['bb_width']
        
        return result
        
    def get_signals(
        self,
        data: pd.DataFrame,
        bb_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands.
        
        Args:
            data: Original price data
            bb_data: Bollinger Bands calculation results
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        
        # Get closing prices
        close = data['close'] if 'close' in data.columns else data.iloc[:, 3]
        
        # Band touch signals
        signals['touch_upper'] = close >= bb_data['bb_upper']
        signals['touch_lower'] = close <= bb_data['bb_lower']
        
        # Band breakout signals
        signals['break_above_upper'] = (
            (close > bb_data['bb_upper']) & 
            (close.shift(1) <= bb_data['bb_upper'].shift(1))
        )
        
        signals['break_below_lower'] = (
            (close < bb_data['bb_lower']) & 
            (close.shift(1) >= bb_data['bb_lower'].shift(1))
        )
        
        # Band re-entry signals (after breakout)
        signals['reenter_from_above'] = (
            (close < bb_data['bb_upper']) & 
            (close.shift(1) >= bb_data['bb_upper'].shift(1))
        )
        
        signals['reenter_from_below'] = (
            (close > bb_data['bb_lower']) & 
            (close.shift(1) <= bb_data['bb_lower'].shift(1))
        )
        
        # Middle band crossover signals
        signals['cross_above_middle'] = (
            (close > bb_data['bb_middle']) & 
            (close.shift(1) <= bb_data['bb_middle'].shift(1))
        )
        
        signals['cross_below_middle'] = (
            (close < bb_data['bb_middle']) & 
            (close.shift(1) >= bb_data['bb_middle'].shift(1))
        )
        
        # Squeeze signals (low volatility)
        band_width_ma = bb_data['bb_width'].rolling(window=self.period).mean()
        band_width_std = bb_data['bb_width'].rolling(window=self.period).std()
        
        signals['squeeze'] = bb_data['bb_width'] < (band_width_ma - band_width_std)
        signals['squeeze_release'] = (
            (bb_data['bb_width'] > band_width_ma) & 
            (bb_data['bb_width'].shift(1) <= band_width_ma.shift(1))
        )
        
        return signals
        
    def calculate_bandwidth(self, bb_data: pd.DataFrame) -> pd.Series:
        """
        Calculate Bollinger Band Width indicator.
        
        Args:
            bb_data: Bollinger Bands calculation results
            
        Returns:
            Bandwidth series
        """
        return (bb_data['bb_upper'] - bb_data['bb_lower']) / bb_data['bb_middle']
        
    def detect_patterns(
        self,
        data: pd.DataFrame,
        bb_data: pd.DataFrame,
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Detect common Bollinger Band patterns.
        
        Args:
            data: Original price data
            bb_data: Bollinger Bands calculation results
            lookback: Number of periods to look back for pattern detection
            
        Returns:
            DataFrame with pattern signals
        """
        patterns = pd.DataFrame(index=data.index)
        
        close = data['close'] if 'close' in data.columns else data.iloc[:, 3]
        high = data['high'] if 'high' in data.columns else data.iloc[:, 1]
        low = data['low'] if 'low' in data.columns else data.iloc[:, 2]
        
        # W-Bottom pattern (double bottom at lower band)
        patterns['w_bottom'] = self._detect_w_bottom(
            close, low, bb_data['bb_lower'], lookback
        )
        
        # M-Top pattern (double top at upper band)
        patterns['m_top'] = self._detect_m_top(
            close, high, bb_data['bb_upper'], lookback
        )
        
        # Walking the bands (trend following)
        patterns['walking_upper'] = (
            (close > bb_data['bb_middle']) & 
            (close >= bb_data['bb_upper'] * 0.95)
        ).rolling(window=3).sum() >= 2
        
        patterns['walking_lower'] = (
            (close < bb_data['bb_middle']) & 
            (close <= bb_data['bb_lower'] * 1.05)
        ).rolling(window=3).sum() >= 2
        
        return patterns
        
    def _detect_w_bottom(
        self,
        close: pd.Series,
        low: pd.Series,
        lower_band: pd.Series,
        lookback: int
    ) -> pd.Series:
        """Detect W-Bottom pattern."""
        signal = pd.Series(False, index=close.index)
        
        for i in range(lookback, len(close)):
            window_low = low.iloc[i-lookback:i]
            window_close = close.iloc[i-lookback:i]
            window_bb = lower_band.iloc[i-lookback:i]
            
            # Find two lows near the lower band
            touches = window_low <= window_bb * 1.02
            
            if touches.sum() >= 2:
                # Check if second low is higher than first
                touch_indices = touches[touches].index
                if len(touch_indices) >= 2:
                    first_low = low.loc[touch_indices[0]]
                    second_low = low.loc[touch_indices[-1]]
                    
                    if second_low > first_low and close.iloc[i] > window_bb.iloc[-1]:
                        signal.iloc[i] = True
                        
        return signal
        
    def _detect_m_top(
        self,
        close: pd.Series,
        high: pd.Series,
        upper_band: pd.Series,
        lookback: int
    ) -> pd.Series:
        """Detect M-Top pattern."""
        signal = pd.Series(False, index=close.index)
        
        for i in range(lookback, len(close)):
            window_high = high.iloc[i-lookback:i]
            window_close = close.iloc[i-lookback:i]
            window_bb = upper_band.iloc[i-lookback:i]
            
            # Find two highs near the upper band
            touches = window_high >= window_bb * 0.98
            
            if touches.sum() >= 2:
                # Check if second high is lower than first
                touch_indices = touches[touches].index
                if len(touch_indices) >= 2:
                    first_high = high.loc[touch_indices[0]]
                    second_high = high.loc[touch_indices[-1]]
                    
                    if second_high < first_high and close.iloc[i] < window_bb.iloc[-1]:
                        signal.iloc[i] = True
                        
        return signal