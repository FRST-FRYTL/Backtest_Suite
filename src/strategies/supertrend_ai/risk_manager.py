"""
Risk management module for SuperTrend AI strategy.

This module provides sophisticated risk management capabilities including
position sizing, stop-loss calculation, and portfolio risk assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskProfile:
    """Container for risk assessment results."""
    risk_score: float  # 0-1, higher means riskier
    position_size: float  # Fraction of capital
    stop_loss: float  # Stop loss price
    take_profit: Optional[float]  # Take profit price
    risk_amount: float  # Dollar risk
    reward_amount: float  # Potential reward
    risk_reward_ratio: float
    acceptable: bool  # Whether trade meets risk criteria
    volatility_adjusted: bool
    metadata: Dict[str, Any] = None


@dataclass
class PortfolioRisk:
    """Container for portfolio-level risk metrics."""
    total_exposure: float  # Total capital at risk
    correlation_risk: float  # Risk from correlated positions
    concentration_risk: float  # Risk from position concentration
    var_95: float  # 95% Value at Risk
    max_drawdown_risk: float  # Estimated max drawdown
    risk_budget_used: float  # Percentage of risk budget used
    positions_at_risk: List[str]  # Symbols with high risk


class RiskManager:
    """
    Advanced risk management system for trading strategies.
    
    Features:
    - Kelly Criterion position sizing
    - Volatility-adjusted position sizing
    - Dynamic stop-loss calculation
    - Portfolio-level risk assessment
    - Risk budget allocation
    - Correlation-based risk adjustment
    """
    
    def __init__(
        self,
        risk_per_trade: float = 0.02,
        max_position_size: float = 0.25,
        max_portfolio_risk: float = 0.06,
        use_kelly_sizing: bool = True,
        kelly_fraction: float = 0.25,
        use_volatility_sizing: bool = True,
        target_volatility: float = 0.15,
        min_risk_reward: float = 1.5,
        stop_loss_atr_multiplier: float = 2.0,
        take_profit_atr_multiplier: float = 3.0,
        use_correlation_adjustment: bool = True,
        max_correlated_positions: int = 3
    ):
        """
        Initialize Risk Manager.
        
        Args:
            risk_per_trade: Maximum risk per trade as fraction of capital
            max_position_size: Maximum position size as fraction of capital
            max_portfolio_risk: Maximum total portfolio risk
            use_kelly_sizing: Whether to use Kelly Criterion
            kelly_fraction: Fraction of Kelly size to use (for safety)
            use_volatility_sizing: Whether to adjust size by volatility
            target_volatility: Target annualized volatility
            min_risk_reward: Minimum acceptable risk/reward ratio
            stop_loss_atr_multiplier: ATR multiplier for stop loss
            take_profit_atr_multiplier: ATR multiplier for take profit
            use_correlation_adjustment: Whether to adjust for correlations
            max_correlated_positions: Maximum correlated positions allowed
        """
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.use_kelly_sizing = use_kelly_sizing
        self.kelly_fraction = kelly_fraction
        self.use_volatility_sizing = use_volatility_sizing
        self.target_volatility = target_volatility
        self.min_risk_reward = min_risk_reward
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_atr_multiplier = take_profit_atr_multiplier
        self.use_correlation_adjustment = use_correlation_adjustment
        self.max_correlated_positions = max_correlated_positions
        
        # State tracking
        self.open_positions = {}
        self.historical_returns = []
        self.correlation_matrix = None
        
    def assess_risk(
        self,
        data: pd.DataFrame,
        entry_price: float,
        direction: int,
        atr_value: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> RiskProfile:
        """
        Assess risk for a potential trade.
        
        Args:
            data: Market data
            entry_price: Proposed entry price
            direction: 1 for long, -1 for short
            atr_value: Current ATR value
            win_rate: Historical win rate (for Kelly sizing)
            avg_win: Average win amount (for Kelly sizing)
            avg_loss: Average loss amount (for Kelly sizing)
            
        Returns:
            RiskProfile with risk assessment
        """
        # Calculate stop loss
        stop_loss = self._calculate_stop_loss(entry_price, direction, atr_value)
        
        # Calculate take profit
        take_profit = self._calculate_take_profit(entry_price, direction, atr_value)
        
        # Calculate base position size
        base_size = self._calculate_base_position_size(
            entry_price, stop_loss, direction
        )
        
        # Apply Kelly Criterion if enabled
        if self.use_kelly_sizing and win_rate and avg_win and avg_loss:
            kelly_size = self._calculate_kelly_size(win_rate, avg_win, avg_loss)
            base_size = min(base_size, kelly_size)
        
        # Apply volatility adjustment if enabled
        volatility_adjusted = False
        if self.use_volatility_sizing:
            vol_adjustment = self._calculate_volatility_adjustment(data)
            base_size *= vol_adjustment
            volatility_adjusted = True
        
        # Apply correlation adjustment if enabled
        if self.use_correlation_adjustment and self.open_positions:
            corr_adjustment = self._calculate_correlation_adjustment(
                data.get('symbol', 'UNKNOWN')
            )
            base_size *= corr_adjustment
        
        # Ensure within limits
        position_size = min(base_size, self.max_position_size)
        
        # Calculate risk and reward amounts
        risk_per_share = abs(entry_price - stop_loss)
        reward_per_share = abs(take_profit - entry_price) if take_profit else 0
        
        risk_amount = position_size * risk_per_share
        reward_amount = position_size * reward_per_share
        
        # Calculate risk/reward ratio
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        # Calculate risk score (0-1)
        risk_score = self._calculate_risk_score(
            data, position_size, risk_amount, volatility_adjusted
        )
        
        # Determine if trade is acceptable
        acceptable = (
            risk_reward_ratio >= self.min_risk_reward and
            risk_amount <= self.risk_per_trade and
            self._check_portfolio_risk_limit(risk_amount)
        )
        
        return RiskProfile(
            risk_score=risk_score,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            risk_reward_ratio=risk_reward_ratio,
            acceptable=acceptable,
            volatility_adjusted=volatility_adjusted,
            metadata={
                'entry_price': entry_price,
                'direction': direction,
                'atr_value': atr_value,
                'timestamp': datetime.now()
            }
        )
    
    def _calculate_stop_loss(self, entry_price: float, direction: int, atr_value: float) -> float:
        """Calculate stop loss price based on ATR."""
        stop_distance = atr_value * self.stop_loss_atr_multiplier
        
        if direction == 1:  # Long position
            stop_loss = entry_price - stop_distance
        else:  # Short position
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def _calculate_take_profit(self, entry_price: float, direction: int, atr_value: float) -> float:
        """Calculate take profit price based on ATR."""
        profit_distance = atr_value * self.take_profit_atr_multiplier
        
        if direction == 1:  # Long position
            take_profit = entry_price + profit_distance
        else:  # Short position
            take_profit = entry_price - profit_distance
        
        return take_profit
    
    def _calculate_base_position_size(
        self, 
        entry_price: float, 
        stop_loss: float, 
        direction: int
    ) -> float:
        """Calculate base position size using fixed fractional method."""
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        # Calculate number of shares based on risk per trade
        position_value = self.risk_per_trade / (risk_per_share / entry_price)
        
        return position_value
    
    def _calculate_kelly_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate position size using Kelly Criterion."""
        if avg_loss == 0:
            return self.max_position_size
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        q = 1 - win_rate
        b = avg_win / avg_loss
        
        kelly_fraction = (win_rate * b - q) / b
        
        # Apply safety factor and ensure positive
        kelly_size = max(0, kelly_fraction * self.kelly_fraction)
        
        return min(kelly_size, self.max_position_size)
    
    def _calculate_volatility_adjustment(self, data: pd.DataFrame) -> float:
        """Calculate position size adjustment based on volatility."""
        # Calculate recent volatility
        returns = data['close'].pct_change()
        recent_vol = returns.tail(20).std() * np.sqrt(252)  # Annualized
        
        if recent_vol == 0:
            return 1.0
        
        # Adjust size inversely to volatility
        vol_adjustment = self.target_volatility / recent_vol
        
        # Cap adjustment to reasonable range
        return np.clip(vol_adjustment, 0.5, 2.0)
    
    def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """Calculate position size adjustment based on correlation with existing positions."""
        if not self.correlation_matrix:
            return 1.0
        
        # Count correlated positions
        correlated_count = 0
        for existing_symbol in self.open_positions:
            if existing_symbol != symbol:
                correlation = self._get_correlation(symbol, existing_symbol)
                if abs(correlation) > 0.7:  # High correlation threshold
                    correlated_count += 1
        
        # Reduce size based on number of correlated positions
        if correlated_count >= self.max_correlated_positions:
            return 0.0  # No new correlated positions
        
        adjustment = 1.0 - (correlated_count * 0.2)  # 20% reduction per correlated position
        
        return max(0.3, adjustment)  # Minimum 30% of original size
    
    def _calculate_risk_score(
        self,
        data: pd.DataFrame,
        position_size: float,
        risk_amount: float,
        volatility_adjusted: bool
    ) -> float:
        """Calculate overall risk score for the trade."""
        scores = []
        
        # Position size risk
        size_score = position_size / self.max_position_size
        scores.append(size_score)
        
        # Risk amount score
        risk_score = risk_amount / self.risk_per_trade
        scores.append(risk_score)
        
        # Volatility score
        returns = data['close'].pct_change()
        recent_vol = returns.tail(20).std()
        hist_vol = returns.tail(252).std() if len(returns) > 252 else recent_vol
        
        if hist_vol > 0:
            vol_score = recent_vol / hist_vol
            scores.append(min(1.0, vol_score))
        
        # Trend consistency score
        ma20 = data['close'].rolling(20).mean()
        ma50 = data['close'].rolling(50).mean()
        
        if not pd.isna(ma20.iloc[-1]) and not pd.isna(ma50.iloc[-1]):
            trend_aligned = (data['close'].iloc[-1] > ma20.iloc[-1] > ma50.iloc[-1]) or \
                          (data['close'].iloc[-1] < ma20.iloc[-1] < ma50.iloc[-1])
            scores.append(0.5 if trend_aligned else 1.0)
        
        # Calculate weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # Adjust weights as needed
        risk_score = np.average(scores[:len(weights)], weights=weights[:len(scores)])
        
        return float(np.clip(risk_score, 0, 1))
    
    def _check_portfolio_risk_limit(self, new_risk: float) -> bool:
        """Check if new trade would exceed portfolio risk limit."""
        current_risk = sum(pos.get('risk_amount', 0) for pos in self.open_positions.values())
        total_risk = current_risk + new_risk
        
        return total_risk <= self.max_portfolio_risk
    
    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        # Placeholder - in production, this would use actual correlation matrix
        # For now, return a dummy value
        return 0.5
    
    def update_position(self, symbol: str, position_data: Dict[str, Any]):
        """Update open position tracking."""
        self.open_positions[symbol] = position_data
    
    def remove_position(self, symbol: str):
        """Remove closed position from tracking."""
        if symbol in self.open_positions:
            del self.open_positions[symbol]
    
    def get_portfolio_risk(self) -> PortfolioRisk:
        """Calculate current portfolio-level risk metrics."""
        if not self.open_positions:
            return PortfolioRisk(
                total_exposure=0,
                correlation_risk=0,
                concentration_risk=0,
                var_95=0,
                max_drawdown_risk=0,
                risk_budget_used=0,
                positions_at_risk=[]
            )
        
        # Calculate total exposure
        total_exposure = sum(pos.get('position_size', 0) for pos in self.open_positions.values())
        
        # Calculate risk amounts
        total_risk = sum(pos.get('risk_amount', 0) for pos in self.open_positions.values())
        
        # Calculate concentration risk
        position_sizes = [pos.get('position_size', 0) for pos in self.open_positions.values()]
        if position_sizes:
            max_position = max(position_sizes)
            concentration_risk = max_position / total_exposure if total_exposure > 0 else 0
        else:
            concentration_risk = 0
        
        # Calculate correlation risk (simplified)
        correlation_risk = 0
        if len(self.open_positions) > 1:
            # Assume average correlation of 0.3 for now
            correlation_risk = 0.3 * len(self.open_positions) / 10  # Normalized
        
        # Calculate VaR (simplified using normal distribution)
        if self.historical_returns:
            returns_std = np.std(self.historical_returns)
            var_95 = 1.645 * returns_std * total_exposure  # 95% VaR
        else:
            var_95 = 0.05 * total_exposure  # Default 5% VaR
        
        # Estimate maximum drawdown risk
        max_drawdown_risk = min(0.5, total_risk * 2)  # Simplified estimate
        
        # Calculate risk budget usage
        risk_budget_used = total_risk / self.max_portfolio_risk
        
        # Identify high-risk positions
        positions_at_risk = [
            symbol for symbol, pos in self.open_positions.items()
            if pos.get('risk_score', 0) > 0.7
        ]
        
        return PortfolioRisk(
            total_exposure=total_exposure,
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            var_95=var_95,
            max_drawdown_risk=max_drawdown_risk,
            risk_budget_used=risk_budget_used,
            positions_at_risk=positions_at_risk
        )
    
    def add_return(self, return_value: float):
        """Add a return to historical tracking."""
        self.historical_returns.append(return_value)
        
        # Keep only recent history
        if len(self.historical_returns) > 1000:
            self.historical_returns = self.historical_returns[-1000:]
    
    def update_correlation_matrix(self, correlation_matrix: pd.DataFrame):
        """Update correlation matrix for position correlation calculations."""
        self.correlation_matrix = correlation_matrix