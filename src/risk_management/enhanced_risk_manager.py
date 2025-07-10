"""
Enhanced Risk Management System

This module provides dynamic position sizing, multiple stop-loss methods,
and comprehensive portfolio risk monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StopLossType(Enum):
    """Stop loss method types"""
    FIXED_PERCENTAGE = "fixed_percentage"
    ATR_BASED = "atr_based"
    TRAILING = "trailing"
    TIME_BASED = "time_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    SUPPORT_LEVEL = "support_level"

class PositionSizingMethod(Enum):
    """Position sizing methods"""
    FIXED_PERCENTAGE = "fixed_percentage"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_BASED = "volatility_based"
    RISK_PARITY = "risk_parity"
    DYNAMIC_CONFIDENCE = "dynamic_confidence"
    MAX_DRAWDOWN_ADJUSTED = "max_drawdown_adjusted"

@dataclass
class RiskParameters:
    """Risk management parameters"""
    max_position_size: float = 0.25          # Maximum 25% per position
    max_portfolio_risk: float = 0.06         # Maximum 6% portfolio risk
    max_correlation_exposure: float = 0.7    # Maximum correlation between positions
    max_sector_exposure: float = 0.4         # Maximum 40% in one sector
    max_drawdown_limit: float = 0.15         # Maximum 15% drawdown
    risk_free_rate: float = 0.02            # Risk-free rate for calculations

@dataclass
class PositionRisk:
    """Risk metrics for a single position"""
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    stop_loss: float
    position_risk_pct: float
    portfolio_risk_contribution: float
    var_95: float
    cvar_95: float
    beta: float
    correlation_risk: float

class EnhancedRiskManager:
    """
    Comprehensive risk management system with dynamic sizing and stops.
    """
    
    def __init__(self, risk_parameters: Optional[RiskParameters] = None):
        """
        Initialize the enhanced risk manager.
        
        Args:
            risk_parameters: Risk management parameters
        """
        self.risk_params = risk_parameters or RiskParameters()
        
        # Portfolio tracking
        self.positions: Dict[str, PositionRisk] = {}
        self.portfolio_history: List[Dict[str, Any]] = []
        self.risk_metrics_history: List[Dict[str, Any]] = []
        
        # Market data cache
        self.volatility_cache: Dict[str, float] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.sector_mapping: Dict[str, str] = {}
        
    def calculate_position_size(
        self,
        symbol: str,
        portfolio_value: float,
        confidence_score: float,
        recent_returns: pd.Series,
        method: PositionSizingMethod = PositionSizingMethod.VOLATILITY_BASED,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> float:
        """
        Calculate optimal position size using specified method.
        
        Args:
            symbol: Symbol to size position for
            portfolio_value: Current portfolio value
            confidence_score: Confluence/confidence score (0-1)
            recent_returns: Recent return series for the asset
            method: Position sizing method
            win_rate: Historical win rate (for Kelly)
            avg_win: Average winning trade return (for Kelly)
            avg_loss: Average losing trade return (for Kelly)
            
        Returns:
            Position size as percentage of portfolio (0-1)
        """
        base_size = 0.0
        
        if method == PositionSizingMethod.FIXED_PERCENTAGE:
            # Simple fixed percentage
            base_size = 0.10  # 10% base
            
        elif method == PositionSizingMethod.KELLY_CRITERION:
            # Kelly Criterion
            if win_rate and avg_win and avg_loss and avg_loss != 0:
                # Kelly percentage = (bp - q) / b
                # where b = avg_win/avg_loss, p = win_rate, q = 1-p
                b = abs(avg_win / avg_loss)
                p = win_rate
                q = 1 - p
                
                kelly_pct = (b * p - q) / b
                # Apply Kelly fraction (typically 0.25 to be conservative)
                base_size = max(0, min(0.25, kelly_pct * 0.25))
            else:
                base_size = 0.05  # Default conservative size
                
        elif method == PositionSizingMethod.VOLATILITY_BASED:
            # Size inversely proportional to volatility
            volatility = recent_returns.std() * np.sqrt(252)
            
            # Target 1% portfolio volatility contribution
            target_vol_contribution = 0.01
            if volatility > 0:
                base_size = target_vol_contribution / volatility
            else:
                base_size = 0.10
                
        elif method == PositionSizingMethod.RISK_PARITY:
            # Equal risk contribution
            volatility = recent_returns.std() * np.sqrt(252)
            
            # Get portfolio volatilities
            all_volatilities = [self._get_volatility(sym) for sym in self.positions.keys()]
            all_volatilities.append(volatility)
            
            if all_volatilities and volatility > 0:
                # Size to equalize risk contribution
                avg_vol = np.mean(all_volatilities)
                base_size = (avg_vol / volatility) * 0.10
            else:
                base_size = 0.10
                
        elif method == PositionSizingMethod.DYNAMIC_CONFIDENCE:
            # Size based on confidence score
            min_size = 0.05
            max_size = 0.20
            base_size = min_size + (max_size - min_size) * confidence_score
            
        elif method == PositionSizingMethod.MAX_DRAWDOWN_ADJUSTED:
            # Adjust size based on recent drawdown
            current_drawdown = self._calculate_current_drawdown()
            
            # Reduce size as drawdown increases
            if current_drawdown < 0.05:
                base_size = 0.15
            elif current_drawdown < 0.10:
                base_size = 0.10
            else:
                base_size = 0.05
        
        # Apply confidence adjustment
        confidence_multiplier = 0.5 + confidence_score * 0.5  # 0.5 to 1.0
        adjusted_size = base_size * confidence_multiplier
        
        # Apply portfolio risk limits
        final_size = self._apply_risk_limits(symbol, adjusted_size, portfolio_value)
        
        logger.info(f"Position size for {symbol}: {final_size:.1%} "
                   f"(method: {method.value}, confidence: {confidence_score:.2f})")
        
        return final_size
    
    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        atr: Optional[float] = None,
        support_level: Optional[float] = None,
        recent_volatility: Optional[float] = None,
        stop_type: StopLossType = StopLossType.ATR_BASED,
        parameters: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate stop loss price using specified method.
        
        Args:
            symbol: Symbol
            entry_price: Entry price
            atr: Average True Range
            support_level: Technical support level
            recent_volatility: Recent volatility
            stop_type: Stop loss calculation method
            parameters: Method-specific parameters
            
        Returns:
            Stop loss price
        """
        params = parameters or {}
        
        if stop_type == StopLossType.FIXED_PERCENTAGE:
            # Fixed percentage stop
            stop_pct = params.get('stop_percentage', 0.05)  # Default 5%
            stop_price = entry_price * (1 - stop_pct)
            
        elif stop_type == StopLossType.ATR_BASED:
            # ATR-based stop
            if atr:
                atr_multiplier = params.get('atr_multiplier', 2.0)
                stop_price = entry_price - (atr * atr_multiplier)
            else:
                # Fallback to fixed percentage
                stop_price = entry_price * 0.95
                
        elif stop_type == StopLossType.TRAILING:
            # Trailing stop (initial setting)
            trail_pct = params.get('trail_percentage', 0.05)
            stop_price = entry_price * (1 - trail_pct)
            
        elif stop_type == StopLossType.TIME_BASED:
            # Time-based stop (wider initially, tightens over time)
            initial_stop_pct = params.get('initial_stop_pct', 0.08)
            stop_price = entry_price * (1 - initial_stop_pct)
            
        elif stop_type == StopLossType.VOLATILITY_ADJUSTED:
            # Volatility-adjusted stop
            if recent_volatility:
                vol_multiplier = params.get('vol_multiplier', 1.5)
                stop_distance = entry_price * recent_volatility * vol_multiplier
                stop_price = entry_price - stop_distance
            else:
                stop_price = entry_price * 0.95
                
        elif stop_type == StopLossType.SUPPORT_LEVEL:
            # Support level based stop
            if support_level and support_level < entry_price:
                buffer = params.get('support_buffer', 0.01)  # 1% below support
                stop_price = support_level * (1 - buffer)
            else:
                # Fallback to fixed percentage
                stop_price = entry_price * 0.95
        
        # Ensure stop is not too tight or too wide
        min_stop_distance = entry_price * 0.02  # Minimum 2%
        max_stop_distance = entry_price * 0.15  # Maximum 15%
        
        stop_distance = entry_price - stop_price
        if stop_distance < min_stop_distance:
            stop_price = entry_price - min_stop_distance
        elif stop_distance > max_stop_distance:
            stop_price = entry_price - max_stop_distance
        
        logger.info(f"Stop loss for {symbol} at {entry_price:.2f}: "
                   f"{stop_price:.2f} ({((entry_price - stop_price) / entry_price * 100):.1f}% distance)")
        
        return stop_price
    
    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        current_stop: float,
        trail_percentage: float = 0.05
    ) -> float:
        """
        Update trailing stop loss.
        
        Args:
            symbol: Symbol
            current_price: Current price
            current_stop: Current stop loss
            trail_percentage: Trailing percentage
            
        Returns:
            Updated stop loss price
        """
        # Calculate new trailing stop
        new_stop = current_price * (1 - trail_percentage)
        
        # Only move stop up, never down
        updated_stop = max(current_stop, new_stop)
        
        if updated_stop > current_stop:
            logger.info(f"Trailing stop for {symbol} updated: "
                       f"{current_stop:.2f} -> {updated_stop:.2f}")
        
        return updated_stop
    
    def calculate_portfolio_risk_metrics(
        self,
        portfolio_value: float,
        market_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            portfolio_value: Current portfolio value
            market_returns: Market return series for beta calculation
            
        Returns:
            Dictionary of risk metrics
        """
        if not self.positions:
            return {
                'total_risk': 0.0,
                'position_count': 0,
                'concentration_risk': 0.0,
                'correlation_risk': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0,
                'max_position_risk': 0.0
            }
        
        # Calculate position-level risks
        position_risks = []
        position_values = []
        
        for symbol, position in self.positions.items():
            position_value = position.position_size * portfolio_value
            position_risk = (position.current_price - position.stop_loss) / position.current_price
            dollar_risk = position_value * position_risk
            
            position_risks.append(dollar_risk / portfolio_value)
            position_values.append(position_value)
        
        # Portfolio-level metrics
        total_risk = sum(position_risks)
        
        # Concentration risk (Herfindahl index)
        if sum(position_values) > 0:
            weights = [v / sum(position_values) for v in position_values]
            concentration_risk = sum(w**2 for w in weights)
        else:
            concentration_risk = 0.0
        
        # Correlation risk
        correlation_risk = self._calculate_correlation_risk()
        
        # VaR and CVaR
        if len(position_risks) > 0:
            var_95 = np.percentile(position_risks, 95)
            cvar_95 = np.mean([r for r in position_risks if r >= var_95])
        else:
            var_95 = 0.0
            cvar_95 = 0.0
        
        metrics = {
            'total_risk': total_risk,
            'position_count': len(self.positions),
            'concentration_risk': concentration_risk,
            'correlation_risk': correlation_risk,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_position_risk': max(position_risks) if position_risks else 0.0,
            'avg_position_risk': np.mean(position_risks) if position_risks else 0.0,
            'risk_utilization': total_risk / self.risk_params.max_portfolio_risk
        }
        
        # Store in history
        self.risk_metrics_history.append({
            'timestamp': pd.Timestamp.now(),
            'portfolio_value': portfolio_value,
            **metrics
        })
        
        return metrics
    
    def check_risk_limits(self, symbol: str, proposed_size: float) -> Tuple[bool, str]:
        """
        Check if proposed position meets risk limits.
        
        Args:
            symbol: Symbol
            proposed_size: Proposed position size (percentage)
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        # Check maximum position size
        if proposed_size > self.risk_params.max_position_size:
            return False, f"Position size {proposed_size:.1%} exceeds limit {self.risk_params.max_position_size:.1%}"
        
        # Check portfolio risk limit
        current_risk = sum(p.portfolio_risk_contribution for p in self.positions.values())
        if current_risk + proposed_size * 0.05 > self.risk_params.max_portfolio_risk:  # Assume 5% risk per position
            return False, f"Would exceed portfolio risk limit {self.risk_params.max_portfolio_risk:.1%}"
        
        # Check correlation exposure
        if symbol in self.positions:
            correlation_exposure = self._calculate_correlation_exposure(symbol)
            if correlation_exposure > self.risk_params.max_correlation_exposure:
                return False, f"Correlation exposure {correlation_exposure:.1%} exceeds limit"
        
        # Check sector exposure
        if symbol in self.sector_mapping:
            sector = self.sector_mapping[symbol]
            sector_exposure = self._calculate_sector_exposure(sector)
            if sector_exposure + proposed_size > self.risk_params.max_sector_exposure:
                return False, f"Would exceed sector exposure limit {self.risk_params.max_sector_exposure:.1%}"
        
        # Check drawdown limit
        current_drawdown = self._calculate_current_drawdown()
        if current_drawdown > self.risk_params.max_drawdown_limit * 0.8:  # 80% of limit
            return False, f"Portfolio near drawdown limit ({current_drawdown:.1%})"
        
        return True, "Risk limits satisfied"
    
    def add_position(
        self,
        symbol: str,
        position_size: float,
        entry_price: float,
        stop_loss: float,
        sector: Optional[str] = None
    ):
        """Add or update position in risk tracking."""
        position_risk = self._calculate_position_risk(
            symbol, position_size, entry_price, stop_loss
        )
        
        self.positions[symbol] = position_risk
        
        if sector:
            self.sector_mapping[symbol] = sector
        
        logger.info(f"Added position {symbol}: size={position_size:.1%}, "
                   f"risk={position_risk.position_risk_pct:.1%}")
    
    def remove_position(self, symbol: str):
        """Remove position from risk tracking."""
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"Removed position {symbol}")
    
    def _apply_risk_limits(
        self,
        symbol: str,
        proposed_size: float,
        portfolio_value: float
    ) -> float:
        """Apply risk limits to proposed position size."""
        # Check if position passes risk limits
        allowed, reason = self.check_risk_limits(symbol, proposed_size)
        
        if not allowed:
            logger.warning(f"Position size reduced for {symbol}: {reason}")
            
            # Try to find maximum allowed size
            max_allowed = proposed_size
            while max_allowed > 0.01 and not allowed:
                max_allowed *= 0.9
                allowed, _ = self.check_risk_limits(symbol, max_allowed)
            
            return max_allowed if allowed else 0.0
        
        return proposed_size
    
    def _calculate_position_risk(
        self,
        symbol: str,
        position_size: float,
        entry_price: float,
        stop_loss: float
    ) -> PositionRisk:
        """Calculate risk metrics for a position."""
        position_risk_pct = (entry_price - stop_loss) / entry_price
        portfolio_risk_contribution = position_size * position_risk_pct
        
        # Simplified VaR and CVaR (would use historical data in practice)
        var_95 = position_size * 0.02  # 2% daily VaR
        cvar_95 = position_size * 0.03  # 3% daily CVaR
        
        return PositionRisk(
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            position_risk_pct=position_risk_pct,
            portfolio_risk_contribution=portfolio_risk_contribution,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=1.0,  # Would calculate from historical data
            correlation_risk=0.0  # Would calculate from correlation matrix
        )
    
    def _get_volatility(self, symbol: str) -> float:
        """Get cached volatility for symbol."""
        return self.volatility_cache.get(symbol, 0.20)  # Default 20% volatility
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current portfolio drawdown."""
        if not self.portfolio_history:
            return 0.0
        
        values = [h['portfolio_value'] for h in self.portfolio_history]
        if not values:
            return 0.0
        
        peak = max(values)
        current = values[-1]
        
        return (peak - current) / peak if peak > 0 else 0.0
    
    def _calculate_correlation_risk(self) -> float:
        """Calculate portfolio correlation risk."""
        if self.correlation_matrix is None or len(self.positions) < 2:
            return 0.0
        
        # Simplified: average pairwise correlation
        symbols = list(self.positions.keys())
        correlations = []
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                if symbols[i] in self.correlation_matrix.index and symbols[j] in self.correlation_matrix.columns:
                    corr = self.correlation_matrix.loc[symbols[i], symbols[j]]
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_correlation_exposure(self, symbol: str) -> float:
        """Calculate correlation exposure for a symbol."""
        if self.correlation_matrix is None:
            return 0.0
        
        if symbol not in self.correlation_matrix.index:
            return 0.0
        
        # Average correlation with existing positions
        correlations = []
        for existing_symbol in self.positions.keys():
            if existing_symbol != symbol and existing_symbol in self.correlation_matrix.columns:
                corr = abs(self.correlation_matrix.loc[symbol, existing_symbol])
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_sector_exposure(self, sector: str) -> float:
        """Calculate current exposure to a sector."""
        sector_exposure = 0.0
        
        for symbol, position in self.positions.items():
            if self.sector_mapping.get(symbol) == sector:
                sector_exposure += position.position_size
        
        return sector_exposure
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        current_metrics = self.calculate_portfolio_risk_metrics(
            portfolio_value=100000  # Dummy value for percentages
        )
        
        report = {
            'current_risk_metrics': current_metrics,
            'position_details': {},
            'risk_limits': {
                'max_position_size': self.risk_params.max_position_size,
                'max_portfolio_risk': self.risk_params.max_portfolio_risk,
                'max_correlation_exposure': self.risk_params.max_correlation_exposure,
                'max_sector_exposure': self.risk_params.max_sector_exposure,
                'max_drawdown_limit': self.risk_params.max_drawdown_limit
            },
            'historical_metrics': {
                'avg_risk_utilization': np.mean([h['risk_utilization'] for h in self.risk_metrics_history[-20:]]) if self.risk_metrics_history else 0,
                'max_historical_risk': max([h['total_risk'] for h in self.risk_metrics_history]) if self.risk_metrics_history else 0,
                'risk_breaches': sum(1 for h in self.risk_metrics_history if h['total_risk'] > self.risk_params.max_portfolio_risk)
            }
        }
        
        # Add position details
        for symbol, position in self.positions.items():
            report['position_details'][symbol] = {
                'size': position.position_size,
                'risk_contribution': position.portfolio_risk_contribution,
                'stop_distance': (position.current_price - position.stop_loss) / position.current_price,
                'var_95': position.var_95
            }
        
        return report