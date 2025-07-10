"""
Technical indicators implementation without ta-lib dependency.
Pure Python/NumPy implementation of key indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculate technical indicators using pandas/numpy."""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.
        
        Args:
            data: Price series
            period: RSI period
            
        Returns:
            RSI values
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def bollinger_bands(
        data: pd.Series,
        period: int = 20,
        std_devs: List[float] = [2.0]
    ) -> Dict[str, pd.Series]:
        """
        Bollinger Bands.
        
        Args:
            data: Price series
            period: MA period
            std_devs: List of standard deviations
            
        Returns:
            Dict with middle, upper, and lower bands
        """
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        bands = {'middle': middle}
        
        for std_dev in std_devs:
            bands[f'upper_{std_dev}'] = middle + (std * std_dev)
            bands[f'lower_{std_dev}'] = middle - (std * std_dev)
            
        return bands
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            
        Returns:
            ATR values
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: Optional[str] = None
    ) -> pd.Series:
        """
        Volume Weighted Average Price.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            period: Reset period ('D' for daily, 'W' for weekly, etc.)
            
        Returns:
            VWAP values
        """
        typical_price = (high + low + close) / 3
        
        if period:
            # Reset VWAP at each period
            grouped = typical_price.groupby(pd.Grouper(freq=period))
            vwap_list = []
            
            for name, group in grouped:
                group_volume = volume.loc[group.index]
                cumulative_tp_volume = (group * group_volume).cumsum()
                cumulative_volume = group_volume.cumsum()
                group_vwap = cumulative_tp_volume / cumulative_volume
                vwap_list.append(group_vwap)
                
            vwap = pd.concat(vwap_list)
        else:
            # Continuous VWAP
            cumulative_tp_volume = (typical_price * volume).cumsum()
            cumulative_volume = volume.cumsum()
            vwap = cumulative_tp_volume / cumulative_volume
            
        return vwap
    
    @staticmethod
    def macd(
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            Dict with MACD line, signal line, and histogram
        """
        ema_fast = data.ewm(span=fast_period, adjust=False).mean()
        ema_slow = data.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3
    ) -> Dict[str, pd.Series]:
        """
        Stochastic Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period
            smooth_k: %K smoothing period
            smooth_d: %D smoothing period
            
        Returns:
            Dict with %K and %D
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_smooth = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_smooth.rolling(window=smooth_d).mean()
        
        return {
            'k': k_smooth,
            'd': d_percent
        }
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume.
        
        Args:
            close: Close prices
            volume: Volume
            
        Returns:
            OBV values
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def rolling_vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int
    ) -> Dict[str, pd.Series]:
        """
        Rolling Volume Weighted Average Price.
        Unlike regular VWAP that resets at specific intervals,
        this calculates a rolling VWAP over the last N periods.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            period: Rolling window period
            
        Returns:
            Dict with VWAP and related calculations
        """
        typical_price = (high + low + close) / 3
        
        # Calculate rolling VWAP
        tp_volume = typical_price * volume
        rolling_tp_volume = tp_volume.rolling(window=period).sum()
        rolling_volume = volume.rolling(window=period).sum()
        vwap = rolling_tp_volume / rolling_volume
        
        # Calculate volume-weighted standard deviation
        price_diff_squared = (typical_price - vwap) ** 2
        weighted_variance = (price_diff_squared * volume).rolling(window=period).sum() / rolling_volume
        vwap_std = np.sqrt(weighted_variance)
        
        return {
            'vwap': vwap,
            'std': vwap_std,
            'typical_price': typical_price
        }
    
    @staticmethod
    def vwap_bands(
        vwap: pd.Series,
        vwap_std: pd.Series,
        std_devs: List[float] = [1, 2, 3]
    ) -> Dict[str, pd.Series]:
        """
        Calculate VWAP bands at different standard deviations.
        
        Args:
            vwap: VWAP values
            vwap_std: VWAP standard deviation
            std_devs: List of standard deviation multipliers
            
        Returns:
            Dict with upper and lower bands
        """
        bands = {}
        
        for std_dev in std_devs:
            bands[f'upper_{std_dev}'] = vwap + (vwap_std * std_dev)
            bands[f'lower_{std_dev}'] = vwap - (vwap_std * std_dev)
            
        return bands
    
    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> Dict[str, pd.Series]:
        """
        Average Directional Index.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period
            
        Returns:
            Dict with ADX, +DI, -DI
        """
        # Calculate directional movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Where both are positive, only keep the larger
        mask = (plus_dm > 0) & (minus_dm > 0)
        plus_dm[mask & (plus_dm <= minus_dm)] = 0
        minus_dm[mask & (minus_dm <= plus_dm)] = 0
        
        # Calculate ATR
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        # Calculate directional indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }