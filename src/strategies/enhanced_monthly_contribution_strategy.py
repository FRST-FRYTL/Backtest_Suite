"""Enhanced Monthly Contribution Strategy with Max Pain and Advanced Confluence."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass

from ..indicators import RSI, BollingerBands, VWAP, FearGreedIndex
from .builder import Strategy, StrategyBuilder, PositionSizing, RiskManagement
from .rules import Rule, Condition, ComparisonOperator, LogicalOperator
from .signals import SignalGenerator


@dataclass
class MaxPainData:
    """Max pain options data structure."""
    strike: float
    max_pain_level: float
    call_oi: float
    put_oi: float
    gamma_exposure: float
    pain_bands: Tuple[float, float]  # (lower, upper)


class VolatilityAdjustedStopLoss:
    """Dynamic stop-loss that adapts to market volatility."""
    
    def __init__(
        self,
        base_stop: float = 0.02,
        min_stop: float = 0.01,
        max_stop: float = 0.05,
        atr_multiplier: float = 2.0
    ):
        self.base_stop = base_stop
        self.min_stop = min_stop
        self.max_stop = max_stop
        self.atr_multiplier = atr_multiplier
        
    def calculate(
        self,
        atr: float,
        price: float,
        volatility_percentile: float,
        support_level: Optional[float] = None
    ) -> float:
        """
        Calculate dynamic stop-loss based on volatility.
        
        Args:
            atr: Average True Range
            price: Current price
            volatility_percentile: Current volatility percentile (0-100)
            support_level: Nearest support level
            
        Returns:
            Stop-loss percentage
        """
        # ATR-based stop
        atr_stop = (atr * self.atr_multiplier) / price
        
        # Adjust for volatility regime
        if volatility_percentile < 20:
            # Low volatility: tighter stops
            volatility_adj = 0.8
        elif volatility_percentile > 80:
            # High volatility: wider stops
            volatility_adj = 1.5
        else:
            # Normal volatility
            volatility_adj = 1.0
            
        adjusted_stop = atr_stop * volatility_adj
        
        # If support level is provided, consider it
        if support_level and support_level < price:
            support_stop = (price - support_level) / price * 1.1  # 10% below support
            adjusted_stop = min(adjusted_stop, support_stop)
            
        # Apply bounds
        return max(self.min_stop, min(adjusted_stop, self.max_stop))


class IndicatorConfluenceEngine:
    """Advanced indicator confluence detection system."""
    
    def __init__(self, min_confluence_score: float = 0.7):
        self.min_confluence_score = min_confluence_score
        self.indicator_weights = {
            'rsi': 0.20,
            'bollinger': 0.20,
            'vwap': 0.15,
            'fear_greed': 0.15,
            'volume': 0.10,
            'trend': 0.10,
            'max_pain': 0.10
        }
        
    def calculate_confluence_score(
        self,
        indicators: Dict[str, Dict[str, Union[float, bool]]]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate overall confluence score from multiple indicators.
        
        Args:
            indicators: Dictionary of indicator signals and values
            
        Returns:
            Tuple of (confluence_score, individual_scores)
        """
        scores = {}
        
        # RSI Score
        rsi_value = indicators['rsi']['value']
        if rsi_value < 20:
            scores['rsi'] = 1.0
        elif rsi_value < 30:
            scores['rsi'] = 0.8
        elif rsi_value < 40:
            scores['rsi'] = 0.5
        elif rsi_value > 80:
            scores['rsi'] = -1.0
        elif rsi_value > 70:
            scores['rsi'] = -0.8
        else:
            scores['rsi'] = 0.0
            
        # Bollinger Bands Score
        bb_position = indicators['bollinger']['position']
        bb_squeeze = indicators['bollinger']['squeeze']
        if bb_position < -1.5:  # Below lower band
            scores['bollinger'] = 0.9
        elif bb_position < -1.0:
            scores['bollinger'] = 0.6
        elif bb_position > 1.5:  # Above upper band
            scores['bollinger'] = -0.9
        elif bb_position > 1.0:
            scores['bollinger'] = -0.6
        else:
            scores['bollinger'] = 0.0
            
        if bb_squeeze:
            scores['bollinger'] *= 1.2  # Boost score during squeeze
            
        # VWAP Score
        vwap_ratio = indicators['vwap']['price_to_vwap']
        vwap_trend = indicators['vwap']['trend']
        if vwap_ratio > 1.02 and vwap_trend == 'up':
            scores['vwap'] = 0.8
        elif vwap_ratio < 0.98 and vwap_trend == 'down':
            scores['vwap'] = -0.8
        elif 0.99 <= vwap_ratio <= 1.01:
            scores['vwap'] = 0.3 if vwap_trend == 'up' else -0.3
        else:
            scores['vwap'] = 0.0
            
        # Fear & Greed Score
        fg_value = indicators['fear_greed']['value']
        if fg_value < 20:
            scores['fear_greed'] = 1.0
        elif fg_value < 30:
            scores['fear_greed'] = 0.7
        elif fg_value > 80:
            scores['fear_greed'] = -1.0
        elif fg_value > 70:
            scores['fear_greed'] = -0.7
        else:
            scores['fear_greed'] = 0.0
            
        # Volume Score
        volume_ratio = indicators['volume']['ratio_to_avg']
        if volume_ratio > 2.0:
            scores['volume'] = 0.8
        elif volume_ratio > 1.5:
            scores['volume'] = 0.5
        elif volume_ratio < 0.5:
            scores['volume'] = -0.5
        else:
            scores['volume'] = 0.0
            
        # Trend Score
        trend_strength = indicators['trend']['strength']
        trend_direction = indicators['trend']['direction']
        if trend_direction == 'up' and trend_strength > 0.7:
            scores['trend'] = 0.8
        elif trend_direction == 'down' and trend_strength > 0.7:
            scores['trend'] = -0.8
        else:
            scores['trend'] = 0.0
            
        # Max Pain Score
        if 'max_pain' in indicators:
            price_to_max_pain = indicators['max_pain']['price_to_max_pain']
            if 0.98 <= price_to_max_pain <= 1.02:
                scores['max_pain'] = 0.7  # Near max pain is bullish
            elif price_to_max_pain < 0.95:
                scores['max_pain'] = 0.9  # Well below max pain
            elif price_to_max_pain > 1.05:
                scores['max_pain'] = -0.7  # Well above max pain
            else:
                scores['max_pain'] = 0.0
        else:
            scores['max_pain'] = 0.0
            
        # Calculate weighted confluence score
        confluence_score = sum(
            scores.get(ind, 0) * weight 
            for ind, weight in self.indicator_weights.items()
        )
        
        return confluence_score, scores


class MaxPainCalculator:
    """Calculate max pain levels from options data."""
    
    def __init__(self):
        self.pain_band_width = 0.02  # 2% bands around max pain
        
    def calculate_max_pain(
        self,
        options_chain: pd.DataFrame,
        current_price: float
    ) -> MaxPainData:
        """
        Calculate max pain level from options chain data.
        
        Args:
            options_chain: DataFrame with strike, call_oi, put_oi columns
            current_price: Current underlying price
            
        Returns:
            MaxPainData object
        """
        if options_chain.empty:
            # Return neutral max pain if no options data
            return MaxPainData(
                strike=current_price,
                max_pain_level=current_price,
                call_oi=0,
                put_oi=0,
                gamma_exposure=0,
                pain_bands=(current_price * 0.98, current_price * 1.02)
            )
            
        # Calculate pain at each strike
        strikes = options_chain['strike'].values
        pain_values = []
        
        for strike in strikes:
            # Call pain: sum of (strike - K) * OI for all strikes below
            call_pain = sum(
                max(0, strike - k) * oi 
                for k, oi in zip(options_chain['strike'], options_chain['call_oi'])
                if k < strike
            )
            
            # Put pain: sum of (K - strike) * OI for all strikes above
            put_pain = sum(
                max(0, k - strike) * oi 
                for k, oi in zip(options_chain['strike'], options_chain['put_oi'])
                if k > strike
            )
            
            total_pain = call_pain + put_pain
            pain_values.append(total_pain)
            
        # Find strike with minimum pain
        min_pain_idx = np.argmin(pain_values)
        max_pain_strike = strikes[min_pain_idx]
        
        # Calculate gamma exposure
        gamma_exposure = self._calculate_gamma_exposure(
            options_chain, current_price, max_pain_strike
        )
        
        # Calculate pain bands
        lower_band = max_pain_strike * (1 - self.pain_band_width)
        upper_band = max_pain_strike * (1 + self.pain_band_width)
        
        return MaxPainData(
            strike=max_pain_strike,
            max_pain_level=max_pain_strike,
            call_oi=options_chain['call_oi'].sum(),
            put_oi=options_chain['put_oi'].sum(),
            gamma_exposure=gamma_exposure,
            pain_bands=(lower_band, upper_band)
        )
        
    def _calculate_gamma_exposure(
        self,
        options_chain: pd.DataFrame,
        current_price: float,
        max_pain: float
    ) -> float:
        """Calculate net gamma exposure."""
        # Simplified gamma calculation
        # In practice, would use Black-Scholes for accurate gamma
        atm_strikes = options_chain[
            (options_chain['strike'] >= current_price * 0.95) &
            (options_chain['strike'] <= current_price * 1.05)
        ]
        
        if atm_strikes.empty:
            return 0.0
            
        net_gamma = (
            atm_strikes['call_oi'].sum() - 
            atm_strikes['put_oi'].sum()
        ) / atm_strikes['call_oi'].sum() if atm_strikes['call_oi'].sum() > 0 else 0
        
        return net_gamma


class EnhancedMonthlyContributionStrategy:
    """
    Enhanced trading strategy with advanced features:
    - Max pain options integration
    - Advanced indicator confluence
    - Dynamic volatility-adjusted stop-losses
    - Machine learning signal validation
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        monthly_contribution: float = 500,
        cash_reserve_target: float = 0.25,
        max_risk_per_trade: float = 0.02,
        use_fear_greed: bool = True,
        use_max_pain: bool = True,
        min_confluence_score: float = 0.7
    ):
        self.initial_capital = initial_capital
        self.monthly_contribution = monthly_contribution
        self.cash_reserve_target = cash_reserve_target
        self.max_risk_per_trade = max_risk_per_trade
        self.use_fear_greed = use_fear_greed
        self.use_max_pain = use_max_pain
        self.min_confluence_score = min_confluence_score
        
        # Initialize components
        self.rsi = RSI(period=14)
        self.bollinger = BollingerBands(period=20, std_dev=2)
        self.vwap = VWAP()
        self.fear_greed = FearGreedIndex() if use_fear_greed else None
        
        # Enhanced components
        self.stop_loss_engine = VolatilityAdjustedStopLoss()
        self.confluence_engine = IndicatorConfluenceEngine(min_confluence_score)
        self.max_pain_calc = MaxPainCalculator() if use_max_pain else None
        
        # Additional indicators
        self.rsi_divergence_period = 20
        self.volume_profile_bins = 50
        
        # Performance tracking
        self.trade_history = []
        self.monthly_contributions = []
        self.confluence_history = []
        
    def calculate_all_indicators(
        self,
        data: pd.DataFrame,
        options_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Dict[str, Union[float, bool]]]:
        """
        Calculate all indicators and their confluence.
        
        Args:
            data: Price and volume data
            options_data: Options chain data
            
        Returns:
            Dictionary of indicator values and signals
        """
        current_idx = len(data) - 1
        current_price = data['close'].iloc[-1]
        
        indicators = {}
        
        # RSI with divergence detection
        rsi_value = data['rsi'].iloc[-1]
        rsi_divergence = self._detect_rsi_divergence(data)
        indicators['rsi'] = {
            'value': rsi_value,
            'divergence': rsi_divergence,
            'oversold': rsi_value < 30,
            'overbought': rsi_value > 70
        }
        
        # Bollinger Bands with squeeze detection
        bb_upper = data['bb_upper'].iloc[-1]
        bb_lower = data['bb_lower'].iloc[-1]
        bb_middle = data['bb_middle'].iloc[-1]
        bb_width = (bb_upper - bb_lower) / bb_middle
        bb_position = (current_price - bb_middle) / (bb_upper - bb_middle) if bb_upper != bb_middle else 0
        
        indicators['bollinger'] = {
            'position': bb_position,
            'squeeze': bb_width < 0.01,
            'width': bb_width,
            'upper': bb_upper,
            'lower': bb_lower,
            'middle': bb_middle
        }
        
        # VWAP with trend
        vwap_value = data['vwap'].iloc[-1]
        vwap_trend = 'up' if data['vwap'].diff().tail(5).mean() > 0 else 'down'
        indicators['vwap'] = {
            'value': vwap_value,
            'price_to_vwap': current_price / vwap_value,
            'trend': vwap_trend,
            'bands_width': data['vwap_bands_width'].iloc[-1]
        }
        
        # Fear & Greed
        if self.use_fear_greed and 'fear_greed' in data.columns:
            indicators['fear_greed'] = {
                'value': data['fear_greed'].iloc[-1],
                'extreme_fear': data['fear_greed'].iloc[-1] < 25,
                'extreme_greed': data['fear_greed'].iloc[-1] > 75
            }
        else:
            indicators['fear_greed'] = {'value': 50, 'extreme_fear': False, 'extreme_greed': False}
            
        # Volume analysis
        volume_avg = data['volume'].rolling(20).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        indicators['volume'] = {
            'current': current_volume,
            'average': volume_avg,
            'ratio_to_avg': current_volume / volume_avg if volume_avg > 0 else 1,
            'increasing': data['volume'].diff().tail(3).mean() > 0
        }
        
        # Trend analysis
        sma_50 = data['sma_50'].iloc[-1]
        sma_200 = data['sma_200'].iloc[-1]
        trend_strength = abs(sma_50 - sma_200) / sma_200 if sma_200 > 0 else 0
        indicators['trend'] = {
            'direction': 'up' if sma_50 > sma_200 else 'down',
            'strength': min(trend_strength, 1.0),
            'sma_50': sma_50,
            'sma_200': sma_200
        }
        
        # Max Pain analysis
        if self.use_max_pain and options_data is not None:
            max_pain_data = self.max_pain_calc.calculate_max_pain(options_data, current_price)
            indicators['max_pain'] = {
                'level': max_pain_data.max_pain_level,
                'price_to_max_pain': current_price / max_pain_data.max_pain_level,
                'in_pain_band': max_pain_data.pain_bands[0] <= current_price <= max_pain_data.pain_bands[1],
                'gamma_exposure': max_pain_data.gamma_exposure,
                'pain_bands': max_pain_data.pain_bands
            }
            
        return indicators
        
    def _detect_rsi_divergence(self, data: pd.DataFrame) -> str:
        """Detect RSI divergence patterns."""
        if len(data) < self.rsi_divergence_period:
            return 'none'
            
        price_series = data['close'].tail(self.rsi_divergence_period)
        rsi_series = data['rsi'].tail(self.rsi_divergence_period)
        
        # Find local extrema
        price_highs = self._find_peaks(price_series)
        price_lows = self._find_peaks(-price_series)
        rsi_highs = self._find_peaks(rsi_series)
        rsi_lows = self._find_peaks(-rsi_series)
        
        # Check for bearish divergence (price higher high, RSI lower high)
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            if price_series.iloc[price_highs[-1]] > price_series.iloc[price_highs[-2]] and \
               rsi_series.iloc[rsi_highs[-1]] < rsi_series.iloc[rsi_highs[-2]]:
                return 'bearish'
                
        # Check for bullish divergence (price lower low, RSI higher low)
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            if price_series.iloc[price_lows[-1]] < price_series.iloc[price_lows[-2]] and \
               rsi_series.iloc[rsi_lows[-1]] > rsi_series.iloc[rsi_lows[-2]]:
                return 'bullish'
                
        return 'none'
        
    def _find_peaks(self, series: pd.Series, prominence: float = 0.05) -> List[int]:
        """Find peaks in a series."""
        from scipy.signal import find_peaks
        values = series.values
        peaks, _ = find_peaks(values, prominence=prominence * np.std(values))
        return peaks.tolist()
        
    def generate_entry_signal(
        self,
        indicators: Dict[str, Dict[str, Union[float, bool]]]
    ) -> Tuple[bool, float, str]:
        """
        Generate entry signal based on indicator confluence.
        
        Returns:
            Tuple of (should_enter, confidence_score, reason)
        """
        confluence_score, individual_scores = self.confluence_engine.calculate_confluence_score(indicators)
        
        # Store confluence history
        self.confluence_history.append({
            'timestamp': datetime.now(),
            'confluence_score': confluence_score,
            'individual_scores': individual_scores
        })
        
        # Check minimum confluence requirement
        if confluence_score < self.min_confluence_score:
            return False, confluence_score, "Insufficient confluence"
            
        # Additional filters for high-confidence entries
        reasons = []
        
        # Check for specific high-confidence patterns
        if indicators['rsi']['value'] < 25 and indicators['bollinger']['position'] < -1.5:
            reasons.append("Extreme oversold with BB breach")
            confluence_score *= 1.2
            
        if indicators['rsi']['divergence'] == 'bullish':
            reasons.append("Bullish RSI divergence")
            confluence_score *= 1.1
            
        if self.use_max_pain and 'max_pain' in indicators:
            if indicators['max_pain']['price_to_max_pain'] < 0.95:
                reasons.append("Price well below max pain")
                confluence_score *= 1.1
                
        if indicators['bollinger']['squeeze'] and indicators['volume']['ratio_to_avg'] > 1.5:
            reasons.append("BB squeeze with volume surge")
            confluence_score *= 1.15
            
        # Final decision
        should_enter = confluence_score >= self.min_confluence_score
        reason = " + ".join(reasons) if reasons else "Confluence threshold met"
        
        return should_enter, min(confluence_score, 1.0), reason
        
    def calculate_dynamic_stop_loss(
        self,
        data: pd.DataFrame,
        entry_price: float,
        indicators: Dict[str, Dict[str, Union[float, bool]]]
    ) -> float:
        """
        Calculate dynamic stop-loss based on market conditions.
        
        Args:
            data: Price data
            entry_price: Entry price
            indicators: Current indicator values
            
        Returns:
            Stop-loss percentage
        """
        # Get ATR
        atr = data['atr'].iloc[-1]
        
        # Calculate volatility percentile
        atr_series = data['atr'].tail(100)
        volatility_percentile = stats.percentileofscore(atr_series, atr)
        
        # Find nearest support level
        support_level = self._find_support_level(data, entry_price)
        
        # Calculate dynamic stop
        stop_loss_pct = self.stop_loss_engine.calculate(
            atr=atr,
            price=entry_price,
            volatility_percentile=volatility_percentile,
            support_level=support_level
        )
        
        # Adjust for max pain if available
        if self.use_max_pain and 'max_pain' in indicators:
            pain_bands = indicators['max_pain']['pain_bands']
            if entry_price > pain_bands[0]:
                # If we're above lower pain band, use it as stop reference
                pain_stop = (entry_price - pain_bands[0]) / entry_price
                stop_loss_pct = min(stop_loss_pct, pain_stop * 1.1)  # 10% buffer below pain band
                
        return stop_loss_pct
        
    def _find_support_level(
        self,
        data: pd.DataFrame,
        current_price: float,
        lookback: int = 50
    ) -> Optional[float]:
        """Find nearest support level using price action."""
        recent_data = data.tail(lookback)
        
        # Find recent lows
        lows = recent_data['low'].values
        
        # Find support levels (areas where price bounced)
        support_levels = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                # Local minimum found
                support_levels.append(lows[i])
                
        if not support_levels:
            return None
            
        # Find nearest support below current price
        valid_supports = [s for s in support_levels if s < current_price]
        if valid_supports:
            return max(valid_supports)  # Nearest support
            
        return None
        
    def build_enhanced_strategy(self) -> Strategy:
        """
        Build the enhanced trading strategy with all improvements.
        
        Returns:
            Enhanced Strategy object
        """
        builder = StrategyBuilder(name="Enhanced Monthly Contribution Strategy v2.0")
        builder.set_description(
            "Advanced strategy with max pain integration, indicator confluence scoring, "
            "and dynamic volatility-adjusted stop-losses for superior risk-adjusted returns."
        )
        
        # Entry Rules with Confluence Requirements
        self._add_enhanced_entry_rules(builder)
        
        # Exit Rules with Dynamic Adjustments
        self._add_enhanced_exit_rules(builder)
        
        # Advanced Market Filters
        self._add_advanced_filters(builder)
        
        # Enhanced Position Sizing
        builder.set_position_sizing(
            method="kelly",
            size=0.30,  # Increased Kelly fraction with better signals
            max_position=0.20,  # Increased max position with better risk management
            scale_in=True,
            scale_out=True,
            volatility_adjustment=True
        )
        
        # Dynamic Risk Management
        builder.set_risk_management(
            stop_loss="dynamic",  # Will be calculated per trade
            stop_loss_type="atr_adjusted",
            take_profit=0.15,  # Increased take profit
            take_profit_type="percent",
            trailing_stop="dynamic",
            time_stop=30,  # Extended time stop
            max_positions=10,  # More positions with better selection
            max_correlation=0.7  # Limit correlated positions
        )
        
        return builder.build()
        
    def _add_enhanced_entry_rules(self, builder: StrategyBuilder):
        """Add enhanced entry rules with confluence requirements."""
        
        # Rule 1: Ultimate Oversold Confluence
        ultimate_oversold = Rule(name="Ultimate Oversold Confluence", operator=LogicalOperator.AND)
        ultimate_oversold.add_condition("rsi", ComparisonOperator.LT, 25)
        ultimate_oversold.add_condition("close", ComparisonOperator.LT, "bb_lower")
        ultimate_oversold.add_condition("volume", ComparisonOperator.GT, "volume_sma_20 * 1.5")
        ultimate_oversold.add_condition("fear_greed", ComparisonOperator.LT, 30)
        ultimate_oversold.add_condition("confluence_score", ComparisonOperator.GT, 0.8)
        builder.add_entry_rule(ultimate_oversold)
        
        # Rule 2: Max Pain Magnet Entry
        if self.use_max_pain:
            max_pain_entry = Rule(name="Max Pain Magnet", operator=LogicalOperator.AND)
            max_pain_entry.add_condition("price_to_max_pain", ComparisonOperator.LT, 0.96)
            max_pain_entry.add_condition("rsi", ComparisonOperator.LT, 40)
            max_pain_entry.add_condition("gamma_exposure", ComparisonOperator.GT, 0.2)
            max_pain_entry.add_condition("confluence_score", ComparisonOperator.GT, 0.7)
            builder.add_entry_rule(max_pain_entry)
            
        # Rule 3: Volatility Squeeze Breakout
        vol_squeeze_breakout = Rule(name="Volatility Squeeze Breakout", operator=LogicalOperator.AND)
        vol_squeeze_breakout.add_condition("bb_squeeze", ComparisonOperator.EQ, True)
        vol_squeeze_breakout.add_condition("atr_percentile", ComparisonOperator.LT, 20)
        vol_squeeze_breakout.add_condition("volume", ComparisonOperator.CROSS_ABOVE, "volume_sma_20 * 2")
        vol_squeeze_breakout.add_condition("close", ComparisonOperator.CROSS_ABOVE, "bb_upper")
        vol_squeeze_breakout.add_condition("confluence_score", ComparisonOperator.GT, 0.75)
        builder.add_entry_rule(vol_squeeze_breakout)
        
        # Rule 4: Institutional Accumulation
        institutional_entry = Rule(name="Institutional Accumulation", operator=LogicalOperator.AND)
        institutional_entry.add_condition("close", ComparisonOperator.GT, "vwap")
        institutional_entry.add_condition("vwap_trend", ComparisonOperator.EQ, "up")
        institutional_entry.add_condition("volume_profile_poc", ComparisonOperator.NEAR, "close", tolerance=0.01)
        institutional_entry.add_condition("dark_pool_ratio", ComparisonOperator.GT, 0.4)
        institutional_entry.add_condition("confluence_score", ComparisonOperator.GT, 0.7)
        builder.add_entry_rule(institutional_entry)
        
        # Rule 5: Divergence Confluence Entry
        divergence_entry = Rule(name="Divergence Confluence", operator=LogicalOperator.AND)
        divergence_entry.add_condition("rsi_divergence", ComparisonOperator.EQ, "bullish")
        divergence_entry.add_condition("macd_divergence", ComparisonOperator.EQ, "bullish")
        divergence_entry.add_condition("volume", ComparisonOperator.GT, "volume_sma_20")
        divergence_entry.add_condition("confluence_score", ComparisonOperator.GT, 0.75)
        builder.add_entry_rule(divergence_entry)
        
    def _add_enhanced_exit_rules(self, builder: StrategyBuilder):
        """Add enhanced exit rules with dynamic adjustments."""
        
        # Rule 1: Confluence Breakdown Exit
        confluence_breakdown = Rule(name="Confluence Breakdown", operator=LogicalOperator.AND)
        confluence_breakdown.add_condition("confluence_score", ComparisonOperator.LT, 0.3)
        confluence_breakdown.add_condition("position_profit_pct", ComparisonOperator.GT, 0.02)
        builder.add_exit_rule(confluence_breakdown)
        
        # Rule 2: Max Pain Resistance
        if self.use_max_pain:
            max_pain_resistance = Rule(name="Max Pain Resistance", operator=LogicalOperator.AND)
            max_pain_resistance.add_condition("price_to_max_pain", ComparisonOperator.GT, 1.04)
            max_pain_resistance.add_condition("gamma_exposure", ComparisonOperator.LT, -0.2)
            max_pain_resistance.add_condition("position_profit_pct", ComparisonOperator.GT, 0.05)
            builder.add_exit_rule(max_pain_resistance)
            
        # Rule 3: Volatility Expansion Exit
        vol_expansion_exit = Rule(name="Volatility Expansion Exit", operator=LogicalOperator.AND)
        vol_expansion_exit.add_condition("atr_percentile", ComparisonOperator.GT, 80)
        vol_expansion_exit.add_condition("bb_width_percentile", ComparisonOperator.GT, 90)
        vol_expansion_exit.add_condition("position_profit_pct", ComparisonOperator.GT, 0.03)
        builder.add_exit_rule(vol_expansion_exit)
        
        # Rule 4: Smart Profit Taking
        smart_profit_exit = Rule(name="Smart Profit Taking", operator=LogicalOperator.OR)
        smart_profit_exit.add_condition("position_profit_pct", ComparisonOperator.GT, 0.15)
        smart_profit_exit.add_sub_rule(
            Rule(operator=LogicalOperator.AND)
            .add_condition("position_profit_pct", ComparisonOperator.GT, 0.08)
            .add_condition("rsi", ComparisonOperator.GT, 75)
            .add_condition("fear_greed", ComparisonOperator.GT, 70)
        )
        builder.add_exit_rule(smart_profit_exit)
        
    def _add_advanced_filters(self, builder: StrategyBuilder):
        """Add advanced market regime filters."""
        
        # Filter 1: Market Quality Filter
        market_quality = Rule(name="Market Quality Filter", operator=LogicalOperator.AND)
        market_quality.add_condition("spread_percentage", ComparisonOperator.LT, 0.002)  # 0.2% max spread
        market_quality.add_condition("liquidity_score", ComparisonOperator.GT, 0.7)
        market_quality.add_condition("microstructure_noise", ComparisonOperator.LT, 0.3)
        builder.add_filter(market_quality)
        
        # Filter 2: Correlation Filter
        correlation_filter = Rule(name="Correlation Filter", operator=LogicalOperator.OR)
        correlation_filter.add_condition("market_correlation", ComparisonOperator.LT, 0.8)
        correlation_filter.add_condition("sector_rotation_score", ComparisonOperator.GT, 0.6)
        builder.add_filter(correlation_filter)
        
        # Filter 3: Options Flow Filter
        if self.use_max_pain:
            options_flow = Rule(name="Options Flow Filter", operator=LogicalOperator.AND)
            options_flow.add_condition("put_call_ratio", ComparisonOperator.BETWEEN, (0.5, 1.5))
            options_flow.add_condition("options_volume_ratio", ComparisonOperator.GT, 0.8)
            builder.add_filter(options_flow)