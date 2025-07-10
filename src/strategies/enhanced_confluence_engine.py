"""
Enhanced Confluence Strategy Engine

This module implements the enhanced confluence scoring system with proper
multi-timeframe analysis, advanced indicators, and weighted scoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import warnings

try:
    from ..data.multi_timeframe_data_manager import MultiTimeframeDataManager, Timeframe
    from ..indicators.technical_indicators import TechnicalIndicators
except ImportError:
    # Direct imports when run as script
    from data.multi_timeframe_data_manager import MultiTimeframeDataManager, Timeframe
    from indicators.technical_indicators import TechnicalIndicators

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConfluenceSignal:
    """Represents a confluence signal with component breakdown"""
    timestamp: pd.Timestamp
    symbol: str
    confluence_score: float
    signal_strength: str  # 'weak', 'moderate', 'strong'
    timeframe_scores: Dict[Timeframe, float]
    component_scores: Dict[str, float]
    indicator_values: Dict[str, Dict[Timeframe, float]]
    recommendation: str  # 'buy', 'sell', 'hold'
    confidence: float

class SignalStrength(Enum):
    """Signal strength classification"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"

class EnhancedConfluenceEngine:
    """
    Enhanced confluence strategy engine with multi-timeframe analysis
    and weighted scoring system.
    """
    
    def __init__(
        self,
        data_manager: MultiTimeframeDataManager,
        confluence_threshold: float = 0.65,
        min_data_points: int = 50
    ):
        """
        Initialize the enhanced confluence engine.
        
        Args:
            data_manager: Multi-timeframe data manager
            confluence_threshold: Minimum confluence score for signals
            min_data_points: Minimum data points required for analysis
        """
        self.data_manager = data_manager
        self.confluence_threshold = confluence_threshold
        self.min_data_points = min_data_points
        self.indicators = TechnicalIndicators()
        
        # Component weights for confluence scoring
        self.component_weights = {
            'trend': 0.40,      # Trend alignment across timeframes
            'momentum': 0.30,   # Momentum indicators (RSI, etc.)
            'volume': 0.20,     # Volume and VWAP analysis
            'volatility': 0.10  # Volatility indicators
        }
        
        # SMA periods for different timeframes
        self.sma_config = {
            Timeframe.HOUR_1: [20, 50],
            Timeframe.HOUR_4: [20, 50, 100],
            Timeframe.DAY_1: [20, 50, 100, 200],
            Timeframe.WEEK_1: [10, 20, 50],
            Timeframe.MONTH_1: [6, 12, 24]
        }
        
        # RSI periods for different timeframes
        self.rsi_config = {
            Timeframe.HOUR_1: 14,
            Timeframe.HOUR_4: 14,
            Timeframe.DAY_1: 14,
            Timeframe.WEEK_1: 14,
            Timeframe.MONTH_1: 14
        }
    
    def calculate_confluence_scores(
        self,
        symbol: str,
        timeframes: Optional[List[Timeframe]] = None
    ) -> pd.DataFrame:
        """
        Calculate confluence scores for a symbol across timeframes.
        
        Args:
            symbol: Symbol to analyze
            timeframes: List of timeframes to analyze
            
        Returns:
            DataFrame with confluence scores and components
        """
        if timeframes is None:
            timeframes = list(self.data_manager.TIMEFRAME_CONFIGS.keys())
        
        # Get synchronized data
        data_by_timeframe = self.data_manager.get_synchronized_data(symbol, timeframes)
        
        # Calculate indicators for each timeframe
        indicators_by_timeframe = {}
        for timeframe, data in data_by_timeframe.items():
            if len(data) < self.min_data_points:
                logger.warning(f"Insufficient data for {symbol} {timeframe.value}: {len(data)} points")
                continue
                
            indicators_by_timeframe[timeframe] = self._calculate_timeframe_indicators(
                data, timeframe
            )
        
        # Calculate confluence scores
        confluence_data = []
        reference_data = data_by_timeframe[Timeframe.DAY_1]
        
        for idx, row in reference_data.iterrows():
            try:
                confluence_score, component_scores, timeframe_scores = self._calculate_point_confluence(
                    idx, indicators_by_timeframe, data_by_timeframe
                )
                
                confluence_data.append({
                    'timestamp': idx,
                    'symbol': symbol,
                    'confluence_score': confluence_score,
                    'trend_score': component_scores.get('trend', 0),
                    'momentum_score': component_scores.get('momentum', 0),
                    'volume_score': component_scores.get('volume', 0),
                    'volatility_score': component_scores.get('volatility', 0),
                    'timeframe_1H': timeframe_scores.get(Timeframe.HOUR_1, 0),
                    'timeframe_4H': timeframe_scores.get(Timeframe.HOUR_4, 0),
                    'timeframe_1D': timeframe_scores.get(Timeframe.DAY_1, 0),
                    'timeframe_1W': timeframe_scores.get(Timeframe.WEEK_1, 0),
                    'timeframe_1M': timeframe_scores.get(Timeframe.MONTH_1, 0),
                    'signal_strength': self._classify_signal_strength(confluence_score),
                    'close_price': row['close']
                })
                
            except Exception as e:
                logger.debug(f"Error calculating confluence for {symbol} at {idx}: {e}")
                continue
        
        confluence_df = pd.DataFrame(confluence_data)
        if not confluence_df.empty:
            confluence_df.set_index('timestamp', inplace=True)
        
        return confluence_df
    
    def _calculate_timeframe_indicators(
        self,
        data: pd.DataFrame,
        timeframe: Timeframe
    ) -> Dict[str, pd.Series]:
        """
        Calculate all indicators for a specific timeframe.
        
        Args:
            data: OHLCV data for the timeframe
            timeframe: Timeframe being analyzed
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # SMAs
        sma_periods = self.sma_config.get(timeframe, [20, 50])
        for period in sma_periods:
            sma_key = f'sma_{period}'
            indicators[sma_key] = self.indicators.sma(data['close'], period)
        
        # RSI
        rsi_period = self.rsi_config.get(timeframe, 14)
        indicators['rsi'] = self.indicators.rsi(data['close'], rsi_period)
        
        # Bollinger Bands
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = \
            self.indicators.bollinger_bands(data['close'], 20, 2)
        
        # VWAP (proper implementation)
        indicators['vwap'] = self._calculate_true_vwap(data, timeframe)
        
        # VWAP Bands
        vwap_std = self._calculate_vwap_standard_deviation(data, indicators['vwap'])
        indicators['vwap_upper'] = indicators['vwap'] + vwap_std
        indicators['vwap_lower'] = indicators['vwap'] - vwap_std
        
        # ATR
        indicators['atr'] = self.indicators.atr(data['high'], data['low'], data['close'], 14)
        
        # Volume indicators
        indicators['volume_sma'] = self.indicators.sma(data['volume'], 20)
        indicators['volume_ratio'] = data['volume'] / indicators['volume_sma']
        
        return indicators
    
    def _calculate_true_vwap(self, data: pd.DataFrame, timeframe: Timeframe) -> pd.Series:
        """
        Calculate true VWAP with proper reset periods.
        
        Args:
            data: OHLCV data
            timeframe: Timeframe for VWAP calculation
            
        Returns:
            VWAP series
        """
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap_series = pd.Series(index=data.index, dtype=float)
        
        # Determine reset frequency based on timeframe
        if timeframe == Timeframe.HOUR_1:
            # Reset daily for 1H timeframe
            reset_freq = 'D'
        elif timeframe == Timeframe.HOUR_4:
            # Reset daily for 4H timeframe
            reset_freq = 'D'
        elif timeframe == Timeframe.DAY_1:
            # Reset weekly for daily timeframe
            reset_freq = 'W'
        elif timeframe == Timeframe.WEEK_1:
            # Reset monthly for weekly timeframe
            reset_freq = 'M'
        else:
            # No reset for monthly timeframe
            reset_freq = None
        
        if reset_freq:
            # Group by reset frequency and calculate VWAP for each group
            groups = data.groupby(pd.Grouper(freq=reset_freq))
            
            for name, group in groups:
                if len(group) > 0:
                    tp = typical_price.loc[group.index]
                    vol = data['volume'].loc[group.index]
                    
                    cumulative_tp_vol = (tp * vol).cumsum()
                    cumulative_vol = vol.cumsum()
                    
                    group_vwap = cumulative_tp_vol / cumulative_vol
                    vwap_series.loc[group.index] = group_vwap
        else:
            # Calculate VWAP for entire series
            cumulative_tp_vol = (typical_price * data['volume']).cumsum()
            cumulative_vol = data['volume'].cumsum()
            vwap_series = cumulative_tp_vol / cumulative_vol
        
        return vwap_series
    
    def _calculate_vwap_standard_deviation(self, data: pd.DataFrame, vwap: pd.Series) -> pd.Series:
        """
        Calculate VWAP standard deviation for bands.
        
        Args:
            data: OHLCV data
            vwap: VWAP series
            
        Returns:
            VWAP standard deviation series
        """
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        variance = ((typical_price - vwap) ** 2 * data['volume']).rolling(window=20).sum() / \
                   data['volume'].rolling(window=20).sum()
        return np.sqrt(variance)
    
    def _calculate_point_confluence(
        self,
        timestamp: pd.Timestamp,
        indicators_by_timeframe: Dict[Timeframe, Dict[str, pd.Series]],
        data_by_timeframe: Dict[Timeframe, pd.DataFrame]
    ) -> Tuple[float, Dict[str, float], Dict[Timeframe, float]]:
        """
        Calculate confluence score for a specific point in time.
        
        Args:
            timestamp: Timestamp to analyze
            indicators_by_timeframe: Indicators for each timeframe
            data_by_timeframe: Raw data for each timeframe
            
        Returns:
            Tuple of (confluence_score, component_scores, timeframe_scores)
        """
        component_scores = {}
        timeframe_scores = {}
        
        # Calculate component scores for each timeframe
        for timeframe, indicators in indicators_by_timeframe.items():
            if timestamp not in indicators.get('sma_20', pd.Series()).index:
                continue
            
            data = data_by_timeframe[timeframe]
            
            # Get current values
            current_values = {
                'price': data.loc[timestamp, 'close'],
                'volume': data.loc[timestamp, 'volume'],
                'high': data.loc[timestamp, 'high'],
                'low': data.loc[timestamp, 'low']
            }
            
            indicator_values = {}
            for key, series in indicators.items():
                if timestamp in series.index:
                    indicator_values[key] = series.loc[timestamp]
            
            # Calculate component scores
            trend_score = self._calculate_trend_score(current_values, indicator_values)
            momentum_score = self._calculate_momentum_score(current_values, indicator_values)
            volume_score = self._calculate_volume_score(current_values, indicator_values)
            volatility_score = self._calculate_volatility_score(current_values, indicator_values)
            
            # Combine component scores for this timeframe
            tf_score = (
                trend_score * self.component_weights['trend'] +
                momentum_score * self.component_weights['momentum'] +
                volume_score * self.component_weights['volume'] +
                volatility_score * self.component_weights['volatility']
            )
            
            timeframe_scores[timeframe] = tf_score
            
            # Accumulate component scores (weighted by timeframe importance)
            tf_weight = self.data_manager.TIMEFRAME_CONFIGS[timeframe].weight
            
            if 'trend' not in component_scores:
                component_scores['trend'] = 0
                component_scores['momentum'] = 0
                component_scores['volume'] = 0
                component_scores['volatility'] = 0
            
            component_scores['trend'] += trend_score * tf_weight
            component_scores['momentum'] += momentum_score * tf_weight
            component_scores['volume'] += volume_score * tf_weight
            component_scores['volatility'] += volatility_score * tf_weight
        
        # Calculate final confluence score
        confluence_score = 0
        for timeframe, tf_score in timeframe_scores.items():
            tf_weight = self.data_manager.TIMEFRAME_CONFIGS[timeframe].weight
            confluence_score += tf_score * tf_weight
        
        return confluence_score, component_scores, timeframe_scores
    
    def _calculate_trend_score(
        self,
        current_values: Dict[str, float],
        indicator_values: Dict[str, float]
    ) -> float:
        """
        Calculate trend component score.
        
        Args:
            current_values: Current price/volume values
            indicator_values: Current indicator values
            
        Returns:
            Trend score (0-1)
        """
        price = current_values['price']
        score = 0
        count = 0
        
        # SMA alignment
        sma_keys = [k for k in indicator_values.keys() if k.startswith('sma_')]
        if sma_keys:
            sma_scores = []
            for sma_key in sma_keys:
                if sma_key in indicator_values and not pd.isna(indicator_values[sma_key]):
                    sma_value = indicator_values[sma_key]
                    if price > sma_value:
                        sma_scores.append(1.0)
                    else:
                        sma_scores.append(0.0)
            
            if sma_scores:
                score += np.mean(sma_scores) * 0.6
                count += 1
        
        # VWAP position
        if 'vwap' in indicator_values and not pd.isna(indicator_values['vwap']):
            vwap_score = 1.0 if price > indicator_values['vwap'] else 0.0
            score += vwap_score * 0.4
            count += 1
        
        return score / count if count > 0 else 0.5
    
    def _calculate_momentum_score(
        self,
        current_values: Dict[str, float],
        indicator_values: Dict[str, float]
    ) -> float:
        """
        Calculate momentum component score.
        
        Args:
            current_values: Current price/volume values
            indicator_values: Current indicator values
            
        Returns:
            Momentum score (0-1)
        """
        score = 0
        count = 0
        
        # RSI analysis
        if 'rsi' in indicator_values and not pd.isna(indicator_values['rsi']):
            rsi = indicator_values['rsi']
            if rsi < 30:
                rsi_score = 1.0  # Oversold, bullish
            elif rsi > 70:
                rsi_score = 0.0  # Overbought, bearish
            else:
                # Normalize RSI to 0-1 scale
                rsi_score = (rsi - 30) / 40
            
            score += rsi_score
            count += 1
        
        # Bollinger Bands position
        if all(k in indicator_values for k in ['bb_upper', 'bb_lower', 'bb_middle']):
            price = current_values['price']
            bb_upper = indicator_values['bb_upper']
            bb_lower = indicator_values['bb_lower']
            bb_middle = indicator_values['bb_middle']
            
            if not any(pd.isna([bb_upper, bb_lower, bb_middle])):
                # Calculate position within bands
                if bb_upper != bb_lower:
                    bb_position = (price - bb_lower) / (bb_upper - bb_lower)
                    bb_position = max(0, min(1, bb_position))  # Clamp to 0-1
                else:
                    bb_position = 0.5
                
                score += bb_position
                count += 1
        
        return score / count if count > 0 else 0.5
    
    def _calculate_volume_score(
        self,
        current_values: Dict[str, float],
        indicator_values: Dict[str, float]
    ) -> float:
        """
        Calculate volume component score.
        
        Args:
            current_values: Current price/volume values
            indicator_values: Current indicator values
            
        Returns:
            Volume score (0-1)
        """
        score = 0
        count = 0
        
        # Volume ratio
        if 'volume_ratio' in indicator_values and not pd.isna(indicator_values['volume_ratio']):
            volume_ratio = indicator_values['volume_ratio']
            # Higher volume is bullish, cap at 3x average
            volume_score = min(1.0, volume_ratio / 3.0)
            score += volume_score
            count += 1
        
        # VWAP bands
        if all(k in indicator_values for k in ['vwap_upper', 'vwap_lower']):
            price = current_values['price']
            vwap_upper = indicator_values['vwap_upper']
            vwap_lower = indicator_values['vwap_lower']
            
            if not any(pd.isna([vwap_upper, vwap_lower])) and vwap_upper != vwap_lower:
                # Position within VWAP bands
                vwap_position = (price - vwap_lower) / (vwap_upper - vwap_lower)
                vwap_position = max(0, min(1, vwap_position))
                score += vwap_position
                count += 1
        
        return score / count if count > 0 else 0.5
    
    def _calculate_volatility_score(
        self,
        current_values: Dict[str, float],
        indicator_values: Dict[str, float]
    ) -> float:
        """
        Calculate volatility component score.
        
        Args:
            current_values: Current price/volume values
            indicator_values: Current indicator values
            
        Returns:
            Volatility score (0-1)
        """
        score = 0
        count = 0
        
        # ATR analysis
        if 'atr' in indicator_values and not pd.isna(indicator_values['atr']):
            atr = indicator_values['atr']
            price = current_values['price']
            
            if price > 0:
                # ATR as percentage of price
                atr_pct = (atr / price) * 100
                
                # Optimal volatility range: 1-3%
                if atr_pct < 1:
                    volatility_score = atr_pct / 1.0  # Too low volatility
                elif atr_pct > 3:
                    volatility_score = max(0, 1 - (atr_pct - 3) / 3)  # Too high volatility
                else:
                    volatility_score = 1.0  # Optimal volatility
                
                score += volatility_score
                count += 1
        
        return score / count if count > 0 else 0.5
    
    def _classify_signal_strength(self, confluence_score: float) -> str:
        """
        Classify signal strength based on confluence score.
        
        Args:
            confluence_score: Confluence score (0-1)
            
        Returns:
            Signal strength classification
        """
        if confluence_score >= 0.8:
            return SignalStrength.STRONG.value
        elif confluence_score >= 0.65:
            return SignalStrength.MODERATE.value
        else:
            return SignalStrength.WEAK.value
    
    def generate_signals(
        self,
        confluence_df: pd.DataFrame,
        min_strength: SignalStrength = SignalStrength.MODERATE
    ) -> List[ConfluenceSignal]:
        """
        Generate trading signals from confluence analysis.
        
        Args:
            confluence_df: DataFrame with confluence scores
            min_strength: Minimum signal strength required
            
        Returns:
            List of confluence signals
        """
        signals = []
        
        for idx, row in confluence_df.iterrows():
            confluence_score = row['confluence_score']
            signal_strength = row['signal_strength']
            
            # Filter by minimum strength
            if (min_strength == SignalStrength.STRONG and signal_strength != SignalStrength.STRONG.value) or \
               (min_strength == SignalStrength.MODERATE and signal_strength == SignalStrength.WEAK.value):
                continue
            
            # Generate signal
            if confluence_score >= self.confluence_threshold:
                recommendation = 'buy'
                confidence = min(0.95, confluence_score)
            else:
                recommendation = 'hold'
                confidence = 1 - confluence_score
            
            signal = ConfluenceSignal(
                timestamp=idx,
                symbol=row['symbol'],
                confluence_score=confluence_score,
                signal_strength=signal_strength,
                timeframe_scores={
                    Timeframe.HOUR_1: row.get('timeframe_1H', 0),
                    Timeframe.HOUR_4: row.get('timeframe_4H', 0),
                    Timeframe.DAY_1: row.get('timeframe_1D', 0),
                    Timeframe.WEEK_1: row.get('timeframe_1W', 0),
                    Timeframe.MONTH_1: row.get('timeframe_1M', 0)
                },
                component_scores={
                    'trend': row.get('trend_score', 0),
                    'momentum': row.get('momentum_score', 0),
                    'volume': row.get('volume_score', 0),
                    'volatility': row.get('volatility_score', 0)
                },
                indicator_values={},  # Would need to populate from indicators
                recommendation=recommendation,
                confidence=confidence
            )
            
            signals.append(signal)
        
        return signals
    
    def get_confluence_summary(self, confluence_df: pd.DataFrame) -> Dict[str, Union[float, int]]:
        """
        Get summary statistics for confluence analysis.
        
        Args:
            confluence_df: DataFrame with confluence scores
            
        Returns:
            Dictionary with summary statistics
        """
        if confluence_df.empty:
            return {}
        
        summary = {
            'total_periods': len(confluence_df),
            'avg_confluence_score': confluence_df['confluence_score'].mean(),
            'max_confluence_score': confluence_df['confluence_score'].max(),
            'min_confluence_score': confluence_df['confluence_score'].min(),
            'strong_signals': len(confluence_df[confluence_df['signal_strength'] == 'strong']),
            'moderate_signals': len(confluence_df[confluence_df['signal_strength'] == 'moderate']),
            'weak_signals': len(confluence_df[confluence_df['signal_strength'] == 'weak']),
            'above_threshold': len(confluence_df[confluence_df['confluence_score'] >= self.confluence_threshold]),
            'avg_trend_score': confluence_df['trend_score'].mean(),
            'avg_momentum_score': confluence_df['momentum_score'].mean(),
            'avg_volume_score': confluence_df['volume_score'].mean(),
            'avg_volatility_score': confluence_df['volatility_score'].mean()
        }
        
        return summary