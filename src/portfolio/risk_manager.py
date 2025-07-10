"""
Risk Management Framework

This module implements comprehensive risk management rules including
stop-loss mechanisms, position limits, and risk monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StopLossType(Enum):
    """Types of stop-loss orders."""
    FIXED = "fixed"
    TRAILING = "trailing"
    ATR_BASED = "atr_based"
    TIME_BASED = "time_based"
    VOLATILITY_BASED = "volatility_based"

class RiskMetric(Enum):
    """Risk metrics for monitoring."""
    VAR = "value_at_risk"
    CVAR = "conditional_value_at_risk"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"
    CONCENTRATION = "concentration"

@dataclass
class RiskLimits:
    """Risk limit configuration."""
    max_position_size: float = 0.20  # Max 20% in single position
    max_sector_exposure: float = 0.40  # Max 40% in single sector
    max_correlation: float = 0.80  # Max correlation between positions
    max_leverage: float = 1.0  # No leverage by default
    max_drawdown: float = 0.15  # Max 15% drawdown
    max_var_95: float = 0.05  # Max 5% VaR at 95% confidence
    max_volatility: float = 0.25  # Max 25% annualized volatility
    min_liquidity_ratio: float = 0.20  # Min 20% in liquid assets
    max_concentration_score: float = 0.30  # Max HHI concentration

@dataclass
class StopLossConfig:
    """Stop-loss configuration."""
    stop_type: StopLossType
    initial_stop: float  # Percentage or ATR multiplier
    trailing_stop: Optional[float] = None  # For trailing stops
    time_stop_days: Optional[int] = None  # For time-based stops
    atr_period: int = 14  # For ATR-based stops
    atr_multiplier: float = 2.0  # For ATR-based stops
    
class RiskManager:
    """
    Comprehensive risk management system.
    """
    
    def __init__(
        self,
        risk_limits: Optional[RiskLimits] = None,
        stop_loss_config: Optional[Dict[str, StopLossConfig]] = None,
        update_frequency: str = 'daily'
    ):
        """
        Initialize risk manager.
        
        Args:
            risk_limits: Risk limit configuration
            stop_loss_config: Stop-loss configuration by strategy/asset
            update_frequency: How often to update risk metrics
        """
        self.risk_limits = risk_limits or RiskLimits()
        self.stop_loss_config = stop_loss_config or {}
        self.update_frequency = update_frequency
        
        # Risk monitoring state
        self.risk_metrics_history = pd.DataFrame()
        self.violations_history = []
        self.stop_loss_triggers = {}
        
    def check_position_limits(
        self,
        current_positions: Dict[str, float],
        new_position: Tuple[str, float],
        portfolio_value: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if new position violates limits.
        
        Args:
            current_positions: Current position sizes
            new_position: (symbol, size) of new position
            portfolio_value: Total portfolio value
            
        Returns:
            (is_allowed, rejection_reason)
        """
        symbol, size = new_position
        
        # Check single position limit
        position_pct = abs(size) / portfolio_value
        if position_pct > self.risk_limits.max_position_size:
            return False, f"Position size {position_pct:.1%} exceeds limit {self.risk_limits.max_position_size:.1%}"
        
        # Check total leverage
        total_exposure = sum(abs(pos) for pos in current_positions.values()) + abs(size)
        leverage = total_exposure / portfolio_value
        if leverage > self.risk_limits.max_leverage:
            return False, f"Leverage {leverage:.2f} exceeds limit {self.risk_limits.max_leverage:.2f}"
        
        # Check concentration
        new_positions = current_positions.copy()
        new_positions[symbol] = new_positions.get(symbol, 0) + size
        concentration = self._calculate_concentration(new_positions, portfolio_value)
        if concentration > self.risk_limits.max_concentration_score:
            return False, f"Concentration {concentration:.2f} exceeds limit {self.risk_limits.max_concentration_score:.2f}"
        
        return True, None
    
    def check_risk_limits(
        self,
        portfolio_metrics: Dict[str, float],
        market_data: pd.DataFrame
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Check all risk limits and return violations.
        
        Args:
            portfolio_metrics: Current portfolio metrics
            market_data: Market data for risk calculations
            
        Returns:
            List of limit violations
        """
        violations = []
        timestamp = datetime.now()
        
        # Check drawdown
        if portfolio_metrics.get('max_drawdown', 0) < -self.risk_limits.max_drawdown:
            violations.append({
                'timestamp': timestamp,
                'metric': 'max_drawdown',
                'value': portfolio_metrics['max_drawdown'],
                'limit': -self.risk_limits.max_drawdown,
                'severity': 'high',
                'action': 'reduce_positions'
            })
        
        # Check VaR
        if abs(portfolio_metrics.get('var_95', 0)) > self.risk_limits.max_var_95:
            violations.append({
                'timestamp': timestamp,
                'metric': 'var_95',
                'value': portfolio_metrics['var_95'],
                'limit': self.risk_limits.max_var_95,
                'severity': 'medium',
                'action': 'review_positions'
            })
        
        # Check volatility
        if portfolio_metrics.get('volatility', 0) > self.risk_limits.max_volatility:
            violations.append({
                'timestamp': timestamp,
                'metric': 'volatility',
                'value': portfolio_metrics['volatility'],
                'limit': self.risk_limits.max_volatility,
                'severity': 'medium',
                'action': 'reduce_leverage'
            })
        
        # Store violations
        self.violations_history.extend(violations)
        
        return violations
    
    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        high_price: float,
        market_data: pd.DataFrame,
        entry_date: datetime,
        current_date: datetime
    ) -> Tuple[float, str]:
        """
        Calculate stop-loss price for position.
        
        Args:
            symbol: Asset symbol
            entry_price: Position entry price
            current_price: Current market price
            high_price: Highest price since entry
            market_data: Historical price data
            entry_date: Position entry date
            current_date: Current date
            
        Returns:
            (stop_price, stop_type)
        """
        # Get stop configuration
        config = self.stop_loss_config.get(symbol, StopLossConfig(
            stop_type=StopLossType.FIXED,
            initial_stop=0.05  # Default 5% stop
        ))
        
        if config.stop_type == StopLossType.FIXED:
            stop_price = entry_price * (1 - config.initial_stop)
            return stop_price, "fixed"
            
        elif config.stop_type == StopLossType.TRAILING:
            # Initial stop
            initial_stop = entry_price * (1 - config.initial_stop)
            
            # Trailing stop from highest price
            trailing_stop = high_price * (1 - (config.trailing_stop or config.initial_stop))
            
            # Use higher of the two
            stop_price = max(initial_stop, trailing_stop)
            return stop_price, "trailing"
            
        elif config.stop_type == StopLossType.ATR_BASED:
            # Calculate ATR
            atr = self._calculate_atr(market_data, config.atr_period)
            stop_distance = atr.iloc[-1] * config.atr_multiplier
            
            # For long positions
            if current_price >= entry_price:
                # Trailing ATR stop from high
                stop_price = high_price - stop_distance
            else:
                # Fixed ATR stop from entry
                stop_price = entry_price - stop_distance
                
            return stop_price, "atr_based"
            
        elif config.stop_type == StopLossType.TIME_BASED:
            # Check if time limit exceeded
            days_held = (current_date - entry_date).days
            if days_held >= (config.time_stop_days or 30):
                return current_price * 0.999, "time_based"  # Exit at market
            else:
                # Use initial stop until time limit
                stop_price = entry_price * (1 - config.initial_stop)
                return stop_price, "fixed"
                
        elif config.stop_type == StopLossType.VOLATILITY_BASED:
            # Calculate realized volatility
            volatility = market_data['close'].pct_change().std()
            annualized_vol = volatility * np.sqrt(252)
            
            # Stop distance based on volatility
            stop_distance = current_price * volatility * config.initial_stop * 100
            
            if current_price >= entry_price:
                # Trailing volatility stop
                stop_price = high_price - stop_distance
            else:
                # Fixed volatility stop
                stop_price = entry_price - stop_distance
                
            return stop_price, "volatility_based"
            
        else:
            # Default fixed stop
            stop_price = entry_price * (1 - 0.05)
            return stop_price, "default"
    
    def adjust_position_size(
        self,
        base_size: float,
        symbol: str,
        market_conditions: Dict[str, float],
        portfolio_metrics: Dict[str, float]
    ) -> float:
        """
        Adjust position size based on risk conditions.
        
        Args:
            base_size: Base position size
            symbol: Asset symbol
            market_conditions: Current market conditions
            portfolio_metrics: Current portfolio metrics
            
        Returns:
            Adjusted position size
        """
        adjustment_factor = 1.0
        
        # Reduce size in high volatility
        market_vol = market_conditions.get('market_volatility', 0.15)
        if market_vol > 0.25:
            adjustment_factor *= 0.7
        elif market_vol > 0.20:
            adjustment_factor *= 0.85
        
        # Reduce size if approaching drawdown limit
        current_dd = abs(portfolio_metrics.get('current_drawdown', 0))
        dd_limit = self.risk_limits.max_drawdown
        if current_dd > dd_limit * 0.8:
            adjustment_factor *= 0.5
        elif current_dd > dd_limit * 0.6:
            adjustment_factor *= 0.75
        
        # Reduce size based on correlation
        avg_correlation = portfolio_metrics.get('avg_correlation', 0)
        if avg_correlation > 0.7:
            adjustment_factor *= 0.8
        
        # Apply minimum size
        adjusted_size = base_size * adjustment_factor
        min_size = portfolio_metrics.get('portfolio_value', 10000) * 0.01  # Min 1%
        
        return max(adjusted_size, min_size) if base_size > 0 else adjusted_size
    
    def calculate_portfolio_risk_metrics(
        self,
        positions: Dict[str, float],
        returns_data: pd.DataFrame,
        market_data: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for portfolio.
        
        Args:
            positions: Current positions {symbol: value}
            returns_data: Historical returns data
            market_data: Market benchmark data
            confidence_level: Confidence level for VaR/CVaR
            
        Returns:
            Dictionary of risk metrics
        """
        # Portfolio weights
        total_value = sum(abs(v) for v in positions.values())
        weights = {k: v / total_value for k, v in positions.items() if total_value > 0}
        
        # Filter returns for held assets
        held_assets = [col for col in returns_data.columns if col in weights]
        if not held_assets:
            return {}
            
        asset_returns = returns_data[held_assets]
        asset_weights = np.array([weights.get(asset, 0) for asset in held_assets])
        
        # Portfolio returns
        portfolio_returns = asset_returns @ asset_weights
        
        # Basic metrics
        metrics = {
            'daily_return': portfolio_returns.mean(),
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'skewness': portfolio_returns.skew(),
            'kurtosis': portfolio_returns.kurtosis()
        }
        
        # VaR and CVaR
        var_threshold = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        metrics['var_95'] = -var_threshold * np.sqrt(252)
        metrics['cvar_95'] = -portfolio_returns[portfolio_returns <= var_threshold].mean() * np.sqrt(252)
        
        # Maximum drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        metrics['current_drawdown'] = drawdown.iloc[-1]
        
        # Beta and correlation
        if 'market' in market_data.columns:
            market_returns = market_data['market'].pct_change().dropna()
            aligned_returns = portfolio_returns.align(market_returns, join='inner')
            if len(aligned_returns[0]) > 20:
                metrics['beta'] = np.cov(aligned_returns[0], aligned_returns[1])[0, 1] / np.var(aligned_returns[1])
                metrics['correlation'] = np.corrcoef(aligned_returns[0], aligned_returns[1])[0, 1]
        
        # Concentration metrics
        position_values = np.array(list(positions.values()))
        position_weights = np.abs(position_values) / np.sum(np.abs(position_values))
        metrics['concentration_hhi'] = np.sum(position_weights ** 2)
        metrics['effective_assets'] = 1 / metrics['concentration_hhi'] if metrics['concentration_hhi'] > 0 else 0
        
        # Correlation matrix metrics
        if len(held_assets) > 1:
            corr_matrix = asset_returns.corr()
            metrics['avg_correlation'] = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            metrics['max_correlation'] = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()
        
        # Store metrics history
        self.risk_metrics_history = pd.concat([
            self.risk_metrics_history,
            pd.DataFrame([metrics], index=[datetime.now()])
        ])
        
        return metrics
    
    def generate_risk_report(self) -> Dict[str, Union[pd.DataFrame, List, Dict]]:
        """
        Generate comprehensive risk report.
        
        Returns:
            Dictionary containing risk analysis
        """
        report = {
            'current_metrics': self.risk_metrics_history.iloc[-1].to_dict() if not self.risk_metrics_history.empty else {},
            'metrics_history': self.risk_metrics_history,
            'violations': self.violations_history[-20:],  # Last 20 violations
            'stop_loss_triggers': self.stop_loss_triggers,
            'risk_limits': self.risk_limits.__dict__
        }
        
        # Add summary statistics
        if not self.risk_metrics_history.empty:
            report['summary_stats'] = {
                'avg_volatility': self.risk_metrics_history['volatility'].mean(),
                'max_drawdown_observed': self.risk_metrics_history['max_drawdown'].min(),
                'violation_count': len(self.violations_history),
                'avg_var_95': self.risk_metrics_history['var_95'].mean()
            }
        
        return report
    
    def _calculate_concentration(self, positions: Dict[str, float], portfolio_value: float) -> float:
        """Calculate portfolio concentration (HHI)."""
        weights = [abs(pos) / portfolio_value for pos in positions.values()]
        return sum(w ** 2 for w in weights)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr