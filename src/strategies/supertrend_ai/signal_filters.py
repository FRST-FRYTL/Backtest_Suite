"""
Signal filtering module for SuperTrend AI strategy.

This module provides various filters to validate and enhance trading signals
based on market conditions, volume, and technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

from ...indicators.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class SignalFilter(ABC):
    """Abstract base class for signal filters."""
    
    @abstractmethod
    def check(self, data: pd.DataFrame, signal: Any, indicator_result: Any) -> bool:
        """
        Check if signal passes the filter.
        
        Args:
            data: Market data
            signal: Trading signal object
            indicator_result: Indicator calculation result
            
        Returns:
            True if signal passes filter, False otherwise
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get filter name."""
        pass


class VolumeFilter(SignalFilter):
    """
    Filter signals based on volume conditions.
    
    Ensures sufficient volume for liquidity and confirms price movements
    with volume support.
    """
    
    def __init__(
        self,
        threshold: float = 1.2,
        lookback: int = 20,
        use_relative: bool = True,
        min_volume: Optional[float] = None
    ):
        """
        Initialize volume filter.
        
        Args:
            threshold: Volume ratio threshold (current/average)
            lookback: Period for average volume calculation
            use_relative: Use relative volume vs absolute
            min_volume: Minimum absolute volume requirement
        """
        self.threshold = threshold
        self.lookback = lookback
        self.use_relative = use_relative
        self.min_volume = min_volume
    
    def check(self, data: pd.DataFrame, signal: Any, indicator_result: Any) -> bool:
        """Check if volume conditions are met."""
        if 'volume' not in data.columns:
            logger.warning("No volume data available, skipping volume filter")
            return True
        
        current_volume = data['volume'].iloc[-1]
        
        # Check minimum volume if specified
        if self.min_volume and current_volume < self.min_volume:
            return False
        
        # Check relative volume
        if self.use_relative:
            avg_volume = data['volume'].rolling(self.lookback).mean().iloc[-1]
            
            if pd.isna(avg_volume) or avg_volume == 0:
                return True  # Pass if can't calculate
            
            volume_ratio = current_volume / avg_volume
            
            # For breakout signals, require higher volume
            if abs(signal.direction) == 1 and hasattr(signal, 'signal_type'):
                if signal.signal_type == 'breakout':
                    return volume_ratio >= self.threshold * 1.5
            
            return volume_ratio >= self.threshold
        
        return True
    
    def get_name(self) -> str:
        return "VolumeFilter"


class TrendStrengthFilter(SignalFilter):
    """
    Filter signals based on trend strength.
    
    Uses ADX and trend consistency metrics to ensure signals
    occur in strong trending conditions.
    """
    
    def __init__(
        self,
        min_strength: float = 0.3,
        adx_threshold: float = 25,
        trend_consistency_period: int = 10,
        use_price_action: bool = True
    ):
        """
        Initialize trend strength filter.
        
        Args:
            min_strength: Minimum trend strength (0-1)
            adx_threshold: Minimum ADX value for trend
            trend_consistency_period: Period to check trend consistency
            use_price_action: Whether to use price action validation
        """
        self.min_strength = min_strength
        self.adx_threshold = adx_threshold
        self.trend_consistency_period = trend_consistency_period
        self.use_price_action = use_price_action
        self.indicators = TechnicalIndicators()
    
    def check(self, data: pd.DataFrame, signal: Any, indicator_result: Any) -> bool:
        """Check if trend strength conditions are met."""
        # Use signal strength if available
        if hasattr(signal, 'strength'):
            if signal.strength < self.min_strength:
                return False
        
        # Calculate ADX
        adx = self._calculate_adx(data)
        if not pd.isna(adx) and adx < self.adx_threshold:
            return False
        
        # Check trend consistency
        if hasattr(indicator_result, 'trend'):
            trend_series = indicator_result.trend
            if len(trend_series) >= self.trend_consistency_period:
                # Check if trend has been consistent
                recent_trend = trend_series.tail(self.trend_consistency_period)
                consistency = (recent_trend == signal.trend).mean()
                
                if consistency < 0.7:  # Require 70% consistency
                    return False
        
        # Price action validation
        if self.use_price_action:
            return self._validate_price_action(data, signal.direction)
        
        return True
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate directional movements
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        pos_dm = pd.Series(0.0, index=data.index)
        neg_dm = pd.Series(0.0, index=data.index)
        
        pos_dm[(up_move > down_move) & (up_move > 0)] = up_move
        neg_dm[(down_move > up_move) & (down_move > 0)] = down_move
        
        # Calculate directional indicators
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window=period).mean()
        
        return adx.iloc[-1] if not adx.empty else np.nan
    
    def _validate_price_action(self, data: pd.DataFrame, direction: int) -> bool:
        """Validate price action supports the signal."""
        close_prices = data['close'].tail(10)
        
        if len(close_prices) < 10:
            return True  # Not enough data
        
        if direction == 1:  # Long signal
            # Check for higher highs and higher lows
            highs = data['high'].tail(10)
            lows = data['low'].tail(10)
            
            higher_highs = (highs.iloc[-1] > highs.iloc[-5]) and (highs.iloc[-5] > highs.iloc[-10])
            higher_lows = (lows.iloc[-1] > lows.iloc[-5]) and (lows.iloc[-5] > lows.iloc[-10])
            
            return higher_highs or higher_lows
        
        else:  # Short signal
            # Check for lower highs and lower lows
            highs = data['high'].tail(10)
            lows = data['low'].tail(10)
            
            lower_highs = (highs.iloc[-1] < highs.iloc[-5]) and (highs.iloc[-5] < highs.iloc[-10])
            lower_lows = (lows.iloc[-1] < lows.iloc[-5]) and (lows.iloc[-5] < lows.iloc[-10])
            
            return lower_highs or lower_lows
    
    def get_name(self) -> str:
        return "TrendStrengthFilter"


class ConfluenceFilter(SignalFilter):
    """
    Filter signals based on multiple indicator confluence.
    
    Requires agreement from multiple technical indicators to
    confirm the signal.
    """
    
    def __init__(
        self,
        min_score: float = 0.6,
        indicators_to_check: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize confluence filter.
        
        Args:
            min_score: Minimum confluence score (0-1)
            indicators_to_check: List of indicators to check
            weights: Weights for each indicator
        """
        self.min_score = min_score
        self.indicators_to_check = indicators_to_check or [
            'rsi', 'macd', 'bollinger', 'stochastic', 'momentum'
        ]
        self.weights = weights or {
            'rsi': 0.2,
            'macd': 0.25,
            'bollinger': 0.2,
            'stochastic': 0.15,
            'momentum': 0.2
        }
        self.indicators = TechnicalIndicators()
    
    def check(self, data: pd.DataFrame, signal: Any, indicator_result: Any) -> bool:
        """Check if confluence conditions are met."""
        # If signal already has confluence score, use it
        if hasattr(signal, 'confluence_score'):
            return signal.confluence_score >= self.min_score
        
        # Calculate confluence from indicators
        scores = {}
        
        if 'rsi' in self.indicators_to_check:
            scores['rsi'] = self._check_rsi(data, signal.direction)
        
        if 'macd' in self.indicators_to_check:
            scores['macd'] = self._check_macd(data, signal.direction)
        
        if 'bollinger' in self.indicators_to_check:
            scores['bollinger'] = self._check_bollinger(data, signal.direction)
        
        if 'stochastic' in self.indicators_to_check:
            scores['stochastic'] = self._check_stochastic(data, signal.direction)
        
        if 'momentum' in self.indicators_to_check:
            scores['momentum'] = self._check_momentum(data, signal.direction)
        
        # Calculate weighted score
        total_score = 0
        total_weight = 0
        
        for indicator, score in scores.items():
            if not pd.isna(score):
                weight = self.weights.get(indicator, 1.0)
                total_score += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return True  # Pass if no indicators available
        
        confluence_score = total_score / total_weight
        
        return confluence_score >= self.min_score
    
    def _check_rsi(self, data: pd.DataFrame, direction: int) -> float:
        """Check RSI confluence."""
        rsi = self.indicators.rsi(data['close'])
        
        if pd.isna(rsi.iloc[-1]):
            return np.nan
        
        current_rsi = rsi.iloc[-1]
        
        if direction == 1:  # Long signal
            if current_rsi < 30:
                return 1.0  # Oversold
            elif current_rsi < 50:
                return 0.7  # Below neutral
            elif current_rsi < 70:
                return 0.3  # Neutral to overbought
            else:
                return 0.0  # Overbought
        
        else:  # Short signal
            if current_rsi > 70:
                return 1.0  # Overbought
            elif current_rsi > 50:
                return 0.7  # Above neutral
            elif current_rsi > 30:
                return 0.3  # Neutral to oversold
            else:
                return 0.0  # Oversold
    
    def _check_macd(self, data: pd.DataFrame, direction: int) -> float:
        """Check MACD confluence."""
        macd_result = self.indicators.macd(data['close'])
        
        if macd_result is None or pd.isna(macd_result['histogram'].iloc[-1]):
            return np.nan
        
        histogram = macd_result['histogram'].iloc[-1]
        histogram_prev = macd_result['histogram'].iloc[-2]
        
        # Check histogram direction and momentum
        if direction == 1:  # Long signal
            if histogram > 0 and histogram > histogram_prev:
                return 1.0  # Strong bullish
            elif histogram > 0:
                return 0.7  # Bullish
            elif histogram > histogram_prev:
                return 0.5  # Improving
            else:
                return 0.0  # Bearish
        
        else:  # Short signal
            if histogram < 0 and histogram < histogram_prev:
                return 1.0  # Strong bearish
            elif histogram < 0:
                return 0.7  # Bearish
            elif histogram < histogram_prev:
                return 0.5  # Deteriorating
            else:
                return 0.0  # Bullish
    
    def _check_bollinger(self, data: pd.DataFrame, direction: int) -> float:
        """Check Bollinger Bands confluence."""
        bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(data['close'])
        
        if pd.isna(bb_upper.iloc[-1]):
            return np.nan
        
        current_price = data['close'].iloc[-1]
        bb_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
        
        if direction == 1:  # Long signal
            if current_price <= bb_lower.iloc[-1]:
                return 1.0  # At lower band
            elif current_price < bb_middle.iloc[-1]:
                return 0.6  # Below middle
            else:
                position = (current_price - bb_middle.iloc[-1]) / (bb_width / 2)
                return max(0, 1 - position)  # Fade as approach upper band
        
        else:  # Short signal
            if current_price >= bb_upper.iloc[-1]:
                return 1.0  # At upper band
            elif current_price > bb_middle.iloc[-1]:
                return 0.6  # Above middle
            else:
                position = (bb_middle.iloc[-1] - current_price) / (bb_width / 2)
                return max(0, 1 - position)  # Fade as approach lower band
    
    def _check_stochastic(self, data: pd.DataFrame, direction: int) -> float:
        """Check Stochastic confluence."""
        # Calculate Stochastic
        high = data['high'].rolling(14).max()
        low = data['low'].rolling(14).min()
        k_percent = 100 * ((data['close'] - low) / (high - low))
        k_percent = k_percent.rolling(3).mean()  # Smooth %K
        d_percent = k_percent.rolling(3).mean()  # %D
        
        if pd.isna(k_percent.iloc[-1]):
            return np.nan
        
        current_k = k_percent.iloc[-1]
        current_d = d_percent.iloc[-1]
        
        if direction == 1:  # Long signal
            if current_k < 20 and current_k > current_d:
                return 1.0  # Oversold with bullish cross
            elif current_k < 20:
                return 0.8  # Oversold
            elif current_k < 50:
                return 0.5  # Below neutral
            else:
                return 0.2  # Less favorable
        
        else:  # Short signal
            if current_k > 80 and current_k < current_d:
                return 1.0  # Overbought with bearish cross
            elif current_k > 80:
                return 0.8  # Overbought
            elif current_k > 50:
                return 0.5  # Above neutral
            else:
                return 0.2  # Less favorable
    
    def _check_momentum(self, data: pd.DataFrame, direction: int, period: int = 10) -> float:
        """Check momentum confluence."""
        momentum = data['close'] - data['close'].shift(period)
        
        if pd.isna(momentum.iloc[-1]):
            return np.nan
        
        current_momentum = momentum.iloc[-1]
        momentum_ma = momentum.rolling(5).mean().iloc[-1]
        
        if direction == 1:  # Long signal
            if current_momentum > 0 and current_momentum > momentum_ma:
                return 1.0  # Strong positive momentum
            elif current_momentum > 0:
                return 0.7  # Positive momentum
            elif current_momentum > momentum_ma:
                return 0.4  # Improving momentum
            else:
                return 0.0  # Negative momentum
        
        else:  # Short signal
            if current_momentum < 0 and current_momentum < momentum_ma:
                return 1.0  # Strong negative momentum
            elif current_momentum < 0:
                return 0.7  # Negative momentum
            elif current_momentum < momentum_ma:
                return 0.4  # Deteriorating momentum
            else:
                return 0.0  # Positive momentum
    
    def get_name(self) -> str:
        return "ConfluenceFilter"


class MarketConditionFilter(SignalFilter):
    """
    Filter signals based on broader market conditions.
    
    Checks market volatility, liquidity, and trading hours.
    """
    
    def __init__(
        self,
        max_volatility: float = 0.5,
        min_liquidity_ratio: float = 0.8,
        allowed_hours: Optional[List[int]] = None,
        check_spreads: bool = True,
        max_spread_percent: float = 0.002
    ):
        """
        Initialize market condition filter.
        
        Args:
            max_volatility: Maximum acceptable volatility
            min_liquidity_ratio: Minimum liquidity ratio
            allowed_hours: Allowed trading hours (24-hour format)
            check_spreads: Whether to check bid-ask spreads
            max_spread_percent: Maximum spread as percentage
        """
        self.max_volatility = max_volatility
        self.min_liquidity_ratio = min_liquidity_ratio
        self.allowed_hours = allowed_hours
        self.check_spreads = check_spreads
        self.max_spread_percent = max_spread_percent
    
    def check(self, data: pd.DataFrame, signal: Any, indicator_result: Any) -> bool:
        """Check if market conditions are favorable."""
        # Check volatility
        returns = data['close'].pct_change()
        current_vol = returns.tail(20).std() * np.sqrt(252)
        
        if current_vol > self.max_volatility:
            logger.info(f"Volatility too high: {current_vol:.2%} > {self.max_volatility:.2%}")
            return False
        
        # Check trading hours if specified
        if self.allowed_hours and hasattr(signal, 'timestamp'):
            hour = signal.timestamp.hour
            if hour not in self.allowed_hours:
                return False
        
        # Check spreads if bid/ask data available
        if self.check_spreads and 'bid' in data.columns and 'ask' in data.columns:
            bid = data['bid'].iloc[-1]
            ask = data['ask'].iloc[-1]
            
            if not pd.isna(bid) and not pd.isna(ask) and bid > 0:
                spread_percent = (ask - bid) / bid
                if spread_percent > self.max_spread_percent:
                    return False
        
        return True
    
    def get_name(self) -> str:
        return "MarketConditionFilter"