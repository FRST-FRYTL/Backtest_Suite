"""
SuperTrend AI Strategy implementation for the Backtest Suite.

This module implements a sophisticated trading strategy based on the AI-enhanced
SuperTrend indicator with K-means clustering optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..builder import Strategy, PositionSizing, RiskManagement
from ..rules import Rule, Condition
from ..signals import Signal, SignalType
from ...indicators.supertrend_ai import SuperTrendAI, SuperTrendResult
from ...indicators.technical_indicators import TechnicalIndicators
from ...ml.models import MarketRegimeDetector, MarketRegime
from .risk_manager import RiskManager, RiskProfile
from .signal_filters import ConfluenceFilter, VolumeFilter, TrendStrengthFilter

logger = logging.getLogger(__name__)


@dataclass
class SuperTrendSignal:
    """Container for SuperTrend AI signals."""
    timestamp: datetime
    symbol: str
    direction: int  # 1 for long, -1 for short, 0 for neutral
    strength: float  # Signal strength 0-1
    trend: int  # Current trend direction
    support_resistance: float  # Current S/R level
    atr_value: float
    optimal_params: Dict[str, float]
    cluster_id: Optional[int] = None
    confluence_score: float = 0.0
    risk_score: float = 1.0
    filters_passed: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SuperTrendAIStrategy(Strategy):
    """
    AI-enhanced SuperTrend trading strategy with advanced features.
    
    Features:
    - Dynamic parameter optimization using K-means clustering
    - Multi-timeframe analysis
    - Risk-adjusted position sizing
    - Signal filtering with confluence
    - Market regime adaptation
    - Trailing stop-loss based on ATR
    """
    
    def __init__(
        self,
        name: str = "SuperTrend AI Strategy",
        # SuperTrend parameters
        atr_periods: Optional[List[int]] = None,
        multipliers: Optional[List[float]] = None,
        n_clusters: int = 5,
        lookback_window: int = 252,
        adaptive: bool = True,
        # Risk parameters
        risk_per_trade: float = 0.02,
        max_position_size: float = 0.25,
        use_trailing_stop: bool = True,
        trailing_atr_multiplier: float = 2.0,
        # Signal filters
        use_confluence: bool = True,
        min_confluence_score: float = 0.6,
        use_volume_filter: bool = True,
        volume_threshold: float = 1.2,
        use_trend_filter: bool = True,
        min_trend_strength: float = 0.3,
        # Market regime
        use_regime_filter: bool = True,
        allowed_regimes: Optional[List[MarketRegime]] = None,
        # Multi-timeframe
        use_mtf: bool = True,
        mtf_periods: Optional[List[str]] = None,
        # Other parameters
        reoptimize_frequency: int = 63,  # Quarterly
        **kwargs
    ):
        """
        Initialize SuperTrend AI Strategy.
        
        Args:
            name: Strategy name
            atr_periods: ATR periods to test in optimization
            multipliers: Multipliers to test in optimization
            n_clusters: Number of clusters for K-means
            lookback_window: Lookback for parameter optimization
            adaptive: Whether to use adaptive parameters
            risk_per_trade: Risk per trade as fraction of capital
            max_position_size: Maximum position size as fraction of capital
            use_trailing_stop: Whether to use trailing stop-loss
            trailing_atr_multiplier: ATR multiplier for trailing stop
            use_confluence: Whether to use confluence filter
            min_confluence_score: Minimum confluence score for entry
            use_volume_filter: Whether to filter by volume
            volume_threshold: Volume ratio threshold
            use_trend_filter: Whether to filter by trend strength
            min_trend_strength: Minimum trend strength
            use_regime_filter: Whether to filter by market regime
            allowed_regimes: List of regimes to trade in
            use_mtf: Whether to use multi-timeframe analysis
            mtf_periods: Timeframes for MTF analysis
            reoptimize_frequency: How often to reoptimize parameters
            **kwargs: Additional Strategy parameters
        """
        super().__init__(name=name, **kwargs)
        
        # SuperTrend parameters
        self.atr_periods = atr_periods
        self.multipliers = multipliers
        self.n_clusters = n_clusters
        self.lookback_window = lookback_window
        self.adaptive = adaptive
        
        # Risk parameters
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.use_trailing_stop = use_trailing_stop
        self.trailing_atr_multiplier = trailing_atr_multiplier
        
        # Signal filters
        self.use_confluence = use_confluence
        self.min_confluence_score = min_confluence_score
        self.use_volume_filter = use_volume_filter
        self.volume_threshold = volume_threshold
        self.use_trend_filter = use_trend_filter
        self.min_trend_strength = min_trend_strength
        
        # Market regime
        self.use_regime_filter = use_regime_filter
        self.allowed_regimes = allowed_regimes or [
            MarketRegime.BULL,
            MarketRegime.SIDEWAYS,
            MarketRegime.PRE_BREAKOUT
        ]
        
        # Multi-timeframe
        self.use_mtf = use_mtf
        self.mtf_periods = mtf_periods or ['1D', '1W', '1M']
        
        # Other parameters
        self.reoptimize_frequency = reoptimize_frequency
        self.last_optimization = None
        
        # Initialize components
        self.supertrend = SuperTrendAI(
            atr_periods=self.atr_periods,
            multipliers=self.multipliers,
            n_clusters=n_clusters,
            lookback_window=lookback_window,
            adaptive=adaptive
        )
        
        self.risk_manager = RiskManager(
            risk_per_trade=risk_per_trade,
            max_position_size=max_position_size,
            use_volatility_sizing=True
        )
        
        self.regime_detector = MarketRegimeDetector() if use_regime_filter else None
        self.technical_indicators = TechnicalIndicators()
        
        # Initialize filters
        self.filters = []
        if use_confluence:
            self.filters.append(ConfluenceFilter(min_score=min_confluence_score))
        if use_volume_filter:
            self.filters.append(VolumeFilter(threshold=volume_threshold))
        if use_trend_filter:
            self.filters.append(TrendStrengthFilter(min_strength=min_trend_strength))
        
        # State tracking
        self.current_signals = {}
        self.position_metadata = {}
        
    def setup_rules(self):
        """Set up trading rules for the strategy."""
        # Entry rules
        self.add_rule(Rule(
            name="SuperTrend Buy Signal",
            conditions=[
                Condition("supertrend_signal", "==", 1),
                Condition("filters_passed", "==", True),
                Condition("risk_check", "==", True)
            ],
            action="buy",
            priority=1
        ))
        
        self.add_rule(Rule(
            name="SuperTrend Sell Signal",
            conditions=[
                Condition("supertrend_signal", "==", -1),
                Condition("filters_passed", "==", True),
                Condition("risk_check", "==", True)
            ],
            action="sell",
            priority=1
        ))
        
        # Exit rules
        self.add_rule(Rule(
            name="Stop Loss Exit",
            conditions=[
                Condition("stop_loss_hit", "==", True)
            ],
            action="exit",
            priority=0
        ))
        
        self.add_rule(Rule(
            name="Trend Reversal Exit",
            conditions=[
                Condition("trend_reversed", "==", True)
            ],
            action="exit",
            priority=2
        ))
        
        # Position management rules
        self.add_rule(Rule(
            name="Update Trailing Stop",
            conditions=[
                Condition("in_position", "==", True),
                Condition("trailing_stop_update", "==", True)
            ],
            action="update_stop",
            priority=3
        ))
    
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate SuperTrend AI signals.
        
        Args:
            data: Market data
            
        Returns:
            Dictionary of signals and indicators
        """
        # Check if we need to reoptimize
        current_bar = len(data)
        if (self.last_optimization is None or 
            current_bar - self.last_optimization >= self.reoptimize_frequency):
            self.last_optimization = current_bar
            logger.info("Reoptimizing SuperTrend parameters")
        
        # Calculate SuperTrend
        supertrend_result = self.supertrend.calculate(data)
        
        # Get current values
        current_idx = -1
        current_price = data['close'].iloc[current_idx]
        current_trend = supertrend_result.trend.iloc[current_idx]
        current_signal = supertrend_result.signal.iloc[current_idx]
        
        # Calculate signal strength
        signal_strength = self.supertrend.get_signal_strength(
            supertrend_result, current_price
        )
        
        # Create signal object
        signal = SuperTrendSignal(
            timestamp=data.index[current_idx],
            symbol=data.get('symbol', 'UNKNOWN'),
            direction=int(current_signal),
            strength=signal_strength,
            trend=int(current_trend),
            support_resistance=supertrend_result.support_resistance.iloc[current_idx],
            atr_value=supertrend_result.atr_values.iloc[current_idx],
            optimal_params=supertrend_result.optimal_params,
            cluster_id=supertrend_result.cluster_info.get('current_cluster') 
                      if supertrend_result.cluster_info else None
        )
        
        # Apply filters
        filter_results = {}
        all_passed = True
        
        for filter_obj in self.filters:
            passed = filter_obj.check(data, signal, supertrend_result)
            filter_results[filter_obj.__class__.__name__] = passed
            all_passed = all_passed and passed
        
        signal.filters_passed = filter_results
        
        # Calculate confluence score if enabled
        if self.use_confluence:
            confluence_score = self._calculate_confluence(data, supertrend_result)
            signal.confluence_score = confluence_score
        
        # Check market regime if enabled
        regime_allowed = True
        if self.use_regime_filter and self.regime_detector:
            current_regime = self.regime_detector.predict(data)
            regime_allowed = current_regime.regime in self.allowed_regimes
            signal.metadata['market_regime'] = current_regime.regime.name
        
        # Multi-timeframe analysis if enabled
        if self.use_mtf:
            mtf_alignment = self._check_mtf_alignment(data, current_trend)
            signal.metadata['mtf_alignment'] = mtf_alignment
            all_passed = all_passed and mtf_alignment > 0.5
        
        # Risk check
        risk_profile = self.risk_manager.assess_risk(
            data, 
            current_price,
            signal.direction,
            supertrend_result.atr_values.iloc[current_idx]
        )
        signal.risk_score = risk_profile.risk_score
        
        # Store current signal
        self.current_signals[data.get('symbol', 'UNKNOWN')] = signal
        
        return {
            'supertrend_signal': signal.direction,
            'signal_strength': signal.strength,
            'filters_passed': all_passed and regime_allowed,
            'risk_check': risk_profile.acceptable,
            'stop_loss_hit': self._check_stop_loss(data, current_price),
            'trend_reversed': self._check_trend_reversal(supertrend_result),
            'in_position': self._has_position(data.get('symbol', 'UNKNOWN')),
            'trailing_stop_update': self._should_update_trailing_stop(data, supertrend_result),
            'position_size': risk_profile.position_size,
            'stop_loss_price': risk_profile.stop_loss,
            'take_profit_price': risk_profile.take_profit
        }
    
    def _calculate_confluence(self, data: pd.DataFrame, supertrend_result: SuperTrendResult) -> float:
        """Calculate confluence score from multiple indicators."""
        scores = []
        
        # RSI confluence
        rsi = self.technical_indicators.rsi(data['close'])
        if not pd.isna(rsi.iloc[-1]):
            if supertrend_result.trend.iloc[-1] == 1:  # Uptrend
                rsi_score = max(0, (50 - rsi.iloc[-1]) / 50) if rsi.iloc[-1] < 70 else 0
            else:  # Downtrend
                rsi_score = max(0, (rsi.iloc[-1] - 50) / 50) if rsi.iloc[-1] > 30 else 0
            scores.append(rsi_score)
        
        # Moving average confluence
        ma20 = data['close'].rolling(20).mean()
        ma50 = data['close'].rolling(50).mean()
        if not pd.isna(ma20.iloc[-1]) and not pd.isna(ma50.iloc[-1]):
            ma_aligned = (ma20.iloc[-1] > ma50.iloc[-1]) == (supertrend_result.trend.iloc[-1] == 1)
            scores.append(1.0 if ma_aligned else 0.0)
        
        # Volume confluence
        vol_ratio = data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]
        if not pd.isna(vol_ratio):
            vol_score = min(1.0, vol_ratio / 2.0)  # Higher volume is better
            scores.append(vol_score)
        
        # MACD confluence
        macd = self.technical_indicators.macd(data['close'])
        if macd is not None and not pd.isna(macd['histogram'].iloc[-1]):
            macd_aligned = (macd['histogram'].iloc[-1] > 0) == (supertrend_result.trend.iloc[-1] == 1)
            scores.append(1.0 if macd_aligned else 0.0)
        
        return np.mean(scores) if scores else 0.5
    
    def _check_mtf_alignment(self, data: pd.DataFrame, current_trend: int) -> float:
        """Check multi-timeframe trend alignment."""
        # This is a simplified version - in production, you'd resample data
        # For now, we'll use different MA periods as proxy for timeframes
        
        alignment_scores = []
        
        # Short-term (proxy for daily)
        ma10 = data['close'].rolling(10).mean()
        if not pd.isna(ma10.iloc[-1]):
            short_trend = 1 if data['close'].iloc[-1] > ma10.iloc[-1] else -1
            alignment_scores.append(1.0 if short_trend == current_trend else 0.0)
        
        # Medium-term (proxy for weekly)
        ma50 = data['close'].rolling(50).mean()
        if not pd.isna(ma50.iloc[-1]):
            medium_trend = 1 if data['close'].iloc[-1] > ma50.iloc[-1] else -1
            alignment_scores.append(1.0 if medium_trend == current_trend else 0.0)
        
        # Long-term (proxy for monthly)
        ma200 = data['close'].rolling(200).mean()
        if not pd.isna(ma200.iloc[-1]):
            long_trend = 1 if data['close'].iloc[-1] > ma200.iloc[-1] else -1
            alignment_scores.append(1.0 if long_trend == current_trend else 0.0)
        
        return np.mean(alignment_scores) if alignment_scores else 0.5
    
    def _check_stop_loss(self, data: pd.DataFrame, current_price: float) -> bool:
        """Check if stop loss has been hit."""
        symbol = data.get('symbol', 'UNKNOWN')
        if symbol not in self.position_metadata:
            return False
        
        position = self.position_metadata[symbol]
        if 'stop_loss' not in position:
            return False
        
        if position['direction'] == 1:  # Long position
            return current_price <= position['stop_loss']
        else:  # Short position
            return current_price >= position['stop_loss']
    
    def _check_trend_reversal(self, supertrend_result: SuperTrendResult) -> bool:
        """Check if trend has reversed."""
        if len(supertrend_result.trend) < 2:
            return False
        
        return supertrend_result.trend.iloc[-1] != supertrend_result.trend.iloc[-2]
    
    def _has_position(self, symbol: str) -> bool:
        """Check if we have an open position."""
        return symbol in self.position_metadata and self.position_metadata[symbol].get('open', False)
    
    def _should_update_trailing_stop(self, data: pd.DataFrame, supertrend_result: SuperTrendResult) -> bool:
        """Check if trailing stop should be updated."""
        if not self.use_trailing_stop:
            return False
        
        symbol = data.get('symbol', 'UNKNOWN')
        if not self._has_position(symbol):
            return False
        
        position = self.position_metadata[symbol]
        current_sr = supertrend_result.support_resistance.iloc[-1]
        
        if position['direction'] == 1:  # Long position
            # Update if new S/R is higher than current stop
            return current_sr > position.get('stop_loss', 0)
        else:  # Short position
            # Update if new S/R is lower than current stop
            return current_sr < position.get('stop_loss', float('inf'))
    
    def on_signal(self, signal_type: str, data: pd.DataFrame, signals: Dict[str, Any]):
        """Handle trading signals."""
        symbol = data.get('symbol', 'UNKNOWN')
        
        if signal_type == 'buy' or signal_type == 'sell':
            # Record position metadata
            self.position_metadata[symbol] = {
                'open': True,
                'direction': 1 if signal_type == 'buy' else -1,
                'entry_price': data['close'].iloc[-1],
                'stop_loss': signals['stop_loss_price'],
                'take_profit': signals['take_profit_price'],
                'position_size': signals['position_size'],
                'entry_time': data.index[-1],
                'signal': self.current_signals[symbol]
            }
            
            logger.info(f"{signal_type.upper()} signal for {symbol}: "
                       f"Price={data['close'].iloc[-1]:.2f}, "
                       f"Size={signals['position_size']:.2%}, "
                       f"Stop={signals['stop_loss_price']:.2f}")
        
        elif signal_type == 'exit':
            # Close position
            if symbol in self.position_metadata:
                self.position_metadata[symbol]['open'] = False
                self.position_metadata[symbol]['exit_time'] = data.index[-1]
                self.position_metadata[symbol]['exit_price'] = data['close'].iloc[-1]
        
        elif signal_type == 'update_stop' and self.use_trailing_stop:
            # Update trailing stop
            if symbol in self.position_metadata:
                supertrend_result = self.supertrend.calculate(data)
                new_stop = supertrend_result.support_resistance.iloc[-1]
                self.position_metadata[symbol]['stop_loss'] = new_stop
                logger.info(f"Updated trailing stop for {symbol} to {new_stop:.2f}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy-specific performance metrics."""
        metrics = super().get_performance_metrics()
        
        # Add SuperTrend-specific metrics
        if self.current_signals:
            avg_signal_strength = np.mean([s.strength for s in self.current_signals.values()])
            avg_confluence = np.mean([s.confluence_score for s in self.current_signals.values()])
            
            metrics['supertrend_metrics'] = {
                'avg_signal_strength': avg_signal_strength,
                'avg_confluence_score': avg_confluence,
                'total_signals': len(self.current_signals),
                'cluster_distribution': self._get_cluster_distribution()
            }
        
        return metrics
    
    def _get_cluster_distribution(self) -> Dict[int, int]:
        """Get distribution of signals across clusters."""
        distribution = {}
        for signal in self.current_signals.values():
            if signal.cluster_id is not None:
                distribution[signal.cluster_id] = distribution.get(signal.cluster_id, 0) + 1
        return distribution