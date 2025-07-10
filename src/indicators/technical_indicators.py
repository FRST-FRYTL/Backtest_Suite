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
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.
        
        Args:
            data: Price series
            period: MA period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
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
    
    @staticmethod
    def true_vwap_with_bands(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        reset_period: str = 'D',
        band_multipliers: List[float] = [1.0, 2.0, 3.0]
    ) -> Dict[str, pd.Series]:
        """
        Enhanced True VWAP calculation with proper reset periods and bands.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            reset_period: Reset frequency ('D', 'W', 'M', or None for no reset)
            band_multipliers: Standard deviation multipliers for bands
            
        Returns:
            Dict with VWAP, bands, and related metrics
        """
        typical_price = (high + low + close) / 3
        vwap_series = pd.Series(index=typical_price.index, dtype=float)
        vwap_std_series = pd.Series(index=typical_price.index, dtype=float)
        
        if reset_period:
            # Ensure we have a proper datetime index
            if not isinstance(typical_price.index, pd.DatetimeIndex):
                logger.warning("Converting index to DatetimeIndex for VWAP calculation")
                typical_price.index = pd.to_datetime(typical_price.index, utc=True).tz_localize(None)
                volume.index = pd.to_datetime(volume.index, utc=True).tz_localize(None)
            
            # Group by reset frequency and calculate VWAP for each group
            grouped = typical_price.groupby(pd.Grouper(freq=reset_period))
            
            for name, group in grouped:
                if len(group) == 0:
                    continue
                    
                group_volume = volume.loc[group.index]
                group_tp = typical_price.loc[group.index]
                
                # Calculate cumulative VWAP
                cumulative_tp_vol = (group_tp * group_volume).cumsum()
                cumulative_vol = group_volume.cumsum()
                
                # Avoid division by zero
                mask = cumulative_vol > 0
                group_vwap = cumulative_tp_vol / cumulative_vol.where(mask, 1)
                group_vwap = group_vwap.where(mask, np.nan)
                
                vwap_series.loc[group.index] = group_vwap
                
                # Calculate volume-weighted standard deviation
                price_diff_sq = (group_tp - group_vwap) ** 2
                cumulative_variance = (price_diff_sq * group_volume).cumsum() / cumulative_vol.where(mask, 1)
                group_std = np.sqrt(cumulative_variance.where(mask, np.nan))
                
                vwap_std_series.loc[group.index] = group_std
        else:
            # Calculate VWAP for entire series without reset
            cumulative_tp_vol = (typical_price * volume).cumsum()
            cumulative_vol = volume.cumsum()
            
            mask = cumulative_vol > 0
            vwap_series = cumulative_tp_vol / cumulative_vol.where(mask, 1)
            vwap_series = vwap_series.where(mask, np.nan)
            
            # Calculate standard deviation
            price_diff_sq = (typical_price - vwap_series) ** 2
            cumulative_variance = (price_diff_sq * volume).cumsum() / cumulative_vol.where(mask, 1)
            vwap_std_series = np.sqrt(cumulative_variance.where(mask, np.nan))
        
        # Create bands
        result = {
            'vwap': vwap_series,
            'vwap_std': vwap_std_series,
            'typical_price': typical_price
        }
        
        for multiplier in band_multipliers:
            result[f'vwap_upper_{multiplier}'] = vwap_series + (vwap_std_series * multiplier)
            result[f'vwap_lower_{multiplier}'] = vwap_series - (vwap_std_series * multiplier)
        
        # Additional metrics
        result['vwap_position'] = (typical_price - vwap_series) / vwap_std_series.where(vwap_std_series > 0, 1)
        result['volume_ratio'] = volume / volume.rolling(window=20).mean()
        
        return result
    
    @staticmethod
    def enhanced_rsi_with_divergence(
        data: pd.Series,
        period: int = 14,
        price_data: Optional[pd.Series] = None
    ) -> Dict[str, pd.Series]:
        """
        Enhanced RSI calculation with divergence detection.
        
        Args:
            data: Price series for RSI calculation
            period: RSI period
            price_data: Price data for divergence analysis (optional)
            
        Returns:
            Dict with RSI, divergence signals, and momentum analysis
        """
        # Calculate standard RSI
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        result = {
            'rsi': rsi,
            'rsi_ema': rsi.ewm(span=9).mean(),  # Smoothed RSI
        }
        
        # RSI momentum
        result['rsi_momentum'] = rsi.diff()
        result['rsi_acceleration'] = result['rsi_momentum'].diff()
        
        # Overbought/Oversold levels with dynamic thresholds
        result['rsi_overbought'] = rsi > 70
        result['rsi_oversold'] = rsi < 30
        result['rsi_extreme_overbought'] = rsi > 80
        result['rsi_extreme_oversold'] = rsi < 20
        
        # RSI divergence detection (if price data provided)
        if price_data is not None:
            result.update(TechnicalIndicators._detect_rsi_divergence(rsi, price_data))
        
        return result
    
    @staticmethod
    def _detect_rsi_divergence(
        rsi: pd.Series,
        price: pd.Series,
        lookback: int = 20
    ) -> Dict[str, pd.Series]:
        """
        Detect RSI divergences with price action.
        
        Args:
            rsi: RSI series
            price: Price series
            lookback: Lookback period for divergence detection
            
        Returns:
            Dict with divergence signals
        """
        # Find local peaks and troughs
        price_peaks = price.rolling(window=lookback, center=True).max() == price
        price_troughs = price.rolling(window=lookback, center=True).min() == price
        
        rsi_peaks = rsi.rolling(window=lookback, center=True).max() == rsi
        rsi_troughs = rsi.rolling(window=lookback, center=True).min() == rsi
        
        # Initialize divergence series
        bullish_divergence = pd.Series(False, index=price.index)
        bearish_divergence = pd.Series(False, index=price.index)
        hidden_bullish_div = pd.Series(False, index=price.index)
        hidden_bearish_div = pd.Series(False, index=price.index)
        
        # Detect regular divergences
        for i in range(lookback, len(price) - lookback):
            current_idx = price.index[i]
            
            # Bullish divergence: price makes lower low, RSI makes higher low
            if price_troughs.iloc[i]:
                prev_trough_indices = price.iloc[:i].where(price_troughs.iloc[:i]).dropna().index
                if len(prev_trough_indices) > 0:
                    prev_idx = prev_trough_indices[-1]
                    if (price.loc[current_idx] < price.loc[prev_idx] and 
                        rsi.loc[current_idx] > rsi.loc[prev_idx]):
                        bullish_divergence.loc[current_idx] = True
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            if price_peaks.iloc[i]:
                prev_peak_indices = price.iloc[:i].where(price_peaks.iloc[:i]).dropna().index
                if len(prev_peak_indices) > 0:
                    prev_idx = prev_peak_indices[-1]
                    if (price.loc[current_idx] > price.loc[prev_idx] and 
                        rsi.loc[current_idx] < rsi.loc[prev_idx]):
                        bearish_divergence.loc[current_idx] = True
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'hidden_bullish_divergence': hidden_bullish_div,
            'hidden_bearish_divergence': hidden_bearish_div
        }
    
    @staticmethod
    def multi_timeframe_sma_alignment(
        data_by_timeframe: Dict[str, pd.Series],
        sma_periods: Dict[str, List[int]],
        reference_timeframe: str = '1D'
    ) -> Dict[str, pd.Series]:
        """
        Calculate SMA alignment across multiple timeframes.
        
        Args:
            data_by_timeframe: Dict mapping timeframe to price series
            sma_periods: Dict mapping timeframe to list of SMA periods
            reference_timeframe: Reference timeframe for alignment
            
        Returns:
            Dict with alignment scores and individual SMAs
        """
        result = {}
        
        # Calculate SMAs for each timeframe
        sma_values = {}
        for tf, data in data_by_timeframe.items():
            sma_values[tf] = {}
            periods = sma_periods.get(tf, [20, 50, 100, 200])
            
            for period in periods:
                sma_key = f'sma_{period}_{tf}'
                sma_values[tf][period] = TechnicalIndicators.sma(data, period)
                result[sma_key] = sma_values[tf][period]
        
        # Calculate alignment scores
        if reference_timeframe in data_by_timeframe:
            ref_data = data_by_timeframe[reference_timeframe]
            ref_index = ref_data.index
            
            # Timeframe alignment score
            tf_alignment = pd.Series(0.0, index=ref_index)
            
            for idx in ref_index:
                alignment_score = 0
                total_weight = 0
                
                for tf, smas in sma_values.items():
                    # Align timeframe data to reference
                    tf_score = 0
                    tf_count = 0
                    
                    for period, sma_series in smas.items():
                        if idx in sma_series.index:
                            price = data_by_timeframe[tf].loc[idx] if idx in data_by_timeframe[tf].index else None
                            sma_val = sma_series.loc[idx]
                            
                            if price is not None and not pd.isna(sma_val) and not pd.isna(price):
                                tf_score += 1 if price > sma_val else 0
                                tf_count += 1
                    
                    if tf_count > 0:
                        # Weight by timeframe importance (could be configurable)
                        tf_weight = {'1M': 0.35, '1W': 0.25, '1D': 0.20, '4H': 0.15, '1H': 0.05}.get(tf, 0.1)
                        alignment_score += (tf_score / tf_count) * tf_weight
                        total_weight += tf_weight
                
                if total_weight > 0:
                    tf_alignment.loc[idx] = alignment_score / total_weight
            
            result['timeframe_alignment'] = tf_alignment
        
        return result