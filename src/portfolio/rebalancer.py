"""
Portfolio Rebalancing System

This module implements portfolio rebalancing strategies including
calendar-based, threshold-based, and dynamic rebalancing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RebalanceMethod(Enum):
    """Rebalancing methods."""
    CALENDAR = "calendar"
    THRESHOLD = "threshold"
    DYNAMIC = "dynamic"
    CPPI = "cppi"  # Constant Proportion Portfolio Insurance
    TACTICAL = "tactical"

class RebalanceFrequency(Enum):
    """Rebalancing frequencies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"

class PortfolioRebalancer:
    """
    Comprehensive portfolio rebalancing system.
    """
    
    def __init__(
        self,
        target_weights: Dict[str, float],
        rebalance_method: RebalanceMethod = RebalanceMethod.THRESHOLD,
        rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY,
        threshold: float = 0.05,
        min_trade_size: float = 0.01,
        transaction_cost: float = 0.001
    ):
        """
        Initialize rebalancer.
        
        Args:
            target_weights: Target portfolio weights
            rebalance_method: Rebalancing method
            rebalance_frequency: Frequency for calendar rebalancing
            threshold: Threshold for threshold-based rebalancing (5% default)
            min_trade_size: Minimum trade size as % of portfolio
            transaction_cost: Transaction cost rate
        """
        self.target_weights = target_weights
        self.rebalance_method = rebalance_method
        self.rebalance_frequency = rebalance_frequency
        self.threshold = threshold
        self.min_trade_size = min_trade_size
        self.transaction_cost = transaction_cost
        
        # Rebalancing state
        self.last_rebalance_date = None
        self.rebalance_history = []
        self.drift_history = []
        
    def check_rebalance_needed(
        self,
        current_positions: Dict[str, float],
        portfolio_value: float,
        current_date: datetime,
        market_conditions: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, Dict[str, str]]:
        """
        Check if rebalancing is needed.
        
        Args:
            current_positions: Current position values
            portfolio_value: Total portfolio value
            current_date: Current date
            market_conditions: Optional market conditions
            
        Returns:
            (needs_rebalance, reasons)
        """
        # Calculate current weights
        current_weights = {
            symbol: value / portfolio_value 
            for symbol, value in current_positions.items()
        }
        
        # Calculate drift
        drift = self._calculate_drift(current_weights)
        self.drift_history.append({
            'date': current_date,
            'drift': drift,
            'max_drift': max(drift.values()) if drift else 0
        })
        
        needs_rebalance = False
        reasons = {}
        
        if self.rebalance_method == RebalanceMethod.CALENDAR:
            # Check if enough time has passed
            if self._is_rebalance_due(current_date):
                needs_rebalance = True
                reasons['calendar'] = f"Scheduled {self.rebalance_frequency.value} rebalance"
                
        elif self.rebalance_method == RebalanceMethod.THRESHOLD:
            # Check if any position has drifted beyond threshold
            max_drift = max(drift.values()) if drift else 0
            if max_drift > self.threshold:
                needs_rebalance = True
                reasons['threshold'] = f"Max drift {max_drift:.1%} exceeds threshold {self.threshold:.1%}"
                
        elif self.rebalance_method == RebalanceMethod.DYNAMIC:
            # Dynamic rebalancing based on market conditions
            if market_conditions:
                volatility = market_conditions.get('volatility', 0.15)
                trend_strength = market_conditions.get('trend_strength', 0)
                
                # Rebalance more frequently in high volatility
                if volatility > 0.25 and max(drift.values()) > self.threshold * 0.5:
                    needs_rebalance = True
                    reasons['volatility'] = f"High volatility {volatility:.1%} with drift"
                
                # Rebalance if trend reversal detected
                if abs(trend_strength) > 0.7 and self._is_rebalance_due(current_date, days=7):
                    needs_rebalance = True
                    reasons['trend'] = "Strong trend detected"
                    
        elif self.rebalance_method == RebalanceMethod.CPPI:
            # CPPI rebalancing
            cushion = self._calculate_cppi_cushion(portfolio_value, market_conditions)
            if cushion < 0.1:  # Less than 10% cushion
                needs_rebalance = True
                reasons['cppi'] = f"CPPI cushion {cushion:.1%} below threshold"
                
        elif self.rebalance_method == RebalanceMethod.TACTICAL:
            # Tactical rebalancing based on market regime
            if market_conditions:
                regime = market_conditions.get('regime', 'normal')
                if regime in ['crisis', 'high_volatility'] and max(drift.values()) > self.threshold * 0.3:
                    needs_rebalance = True
                    reasons['tactical'] = f"Tactical rebalance for {regime} regime"
        
        # Always check extreme drift
        if max(drift.values()) > self.threshold * 2:
            needs_rebalance = True
            reasons['extreme_drift'] = f"Extreme drift {max(drift.values()):.1%}"
        
        return needs_rebalance, reasons
    
    def calculate_trades(
        self,
        current_positions: Dict[str, float],
        portfolio_value: float,
        target_weights: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate trades needed to rebalance.
        
        Args:
            current_positions: Current position values
            portfolio_value: Total portfolio value
            target_weights: Override default target weights
            constraints: Trading constraints
            
        Returns:
            Dictionary of trades {symbol: trade_value}
        """
        target_weights = target_weights or self.target_weights
        constraints = constraints or {}
        
        # Calculate target positions
        target_positions = {
            symbol: weight * portfolio_value
            for symbol, weight in target_weights.items()
        }
        
        # Calculate raw trades
        trades = {}
        for symbol in set(list(current_positions.keys()) + list(target_positions.keys())):
            current = current_positions.get(symbol, 0)
            target = target_positions.get(symbol, 0)
            trade = target - current
            
            # Apply minimum trade size
            if abs(trade) / portfolio_value >= self.min_trade_size:
                trades[symbol] = trade
        
        # Apply constraints
        trades = self._apply_constraints(trades, constraints, portfolio_value)
        
        # Optimize for transaction costs
        trades = self._optimize_trades(trades, portfolio_value)
        
        return trades
    
    def execute_rebalance(
        self,
        current_positions: Dict[str, float],
        portfolio_value: float,
        current_date: datetime,
        trades: Optional[Dict[str, float]] = None
    ) -> Dict[str, Union[Dict, float]]:
        """
        Execute rebalancing and record history.
        
        Args:
            current_positions: Current positions
            portfolio_value: Portfolio value
            current_date: Current date
            trades: Pre-calculated trades (optional)
            
        Returns:
            Rebalancing results
        """
        # Calculate trades if not provided
        if trades is None:
            trades = self.calculate_trades(current_positions, portfolio_value)
        
        # Calculate costs
        transaction_costs = sum(abs(trade) * self.transaction_cost for trade in trades.values())
        
        # New positions after trades
        new_positions = current_positions.copy()
        for symbol, trade in trades.items():
            new_positions[symbol] = new_positions.get(symbol, 0) + trade
        
        # Calculate new weights
        new_portfolio_value = portfolio_value - transaction_costs
        new_weights = {
            symbol: value / new_portfolio_value
            for symbol, value in new_positions.items()
        }
        
        # Record rebalancing
        rebalance_record = {
            'date': current_date,
            'trades': trades,
            'transaction_costs': transaction_costs,
            'old_weights': {s: v/portfolio_value for s, v in current_positions.items()},
            'new_weights': new_weights,
            'portfolio_value': portfolio_value
        }
        
        self.rebalance_history.append(rebalance_record)
        self.last_rebalance_date = current_date
        
        return {
            'new_positions': new_positions,
            'trades': trades,
            'transaction_costs': transaction_costs,
            'new_weights': new_weights
        }
    
    def get_rebalance_analytics(self) -> Dict[str, Union[pd.DataFrame, float, Dict]]:
        """
        Get comprehensive rebalancing analytics.
        
        Returns:
            Dictionary with rebalancing statistics
        """
        if not self.rebalance_history:
            return {}
        
        # Convert history to DataFrame
        history_df = pd.DataFrame(self.rebalance_history)
        
        # Calculate statistics
        analytics = {
            'total_rebalances': len(self.rebalance_history),
            'total_costs': history_df['transaction_costs'].sum(),
            'avg_cost_per_rebalance': history_df['transaction_costs'].mean(),
            'rebalance_frequency_days': self._calculate_avg_rebalance_frequency(),
            'avg_drift_at_rebalance': self._calculate_avg_drift_at_rebalance()
        }
        
        # Drift statistics
        if self.drift_history:
            drift_df = pd.DataFrame(self.drift_history)
            analytics['avg_max_drift'] = drift_df['max_drift'].mean()
            analytics['max_observed_drift'] = drift_df['max_drift'].max()
        
        # Trade size analysis
        all_trades = []
        for record in self.rebalance_history:
            for symbol, trade in record['trades'].items():
                all_trades.append({
                    'symbol': symbol,
                    'trade_size': abs(trade),
                    'trade_pct': abs(trade) / record['portfolio_value']
                })
        
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            analytics['avg_trade_size_pct'] = trades_df['trade_pct'].mean()
            analytics['max_trade_size_pct'] = trades_df['trade_pct'].max()
        
        return analytics
    
    def _calculate_drift(self, current_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate drift from target weights."""
        drift = {}
        
        for symbol, target_weight in self.target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            drift[symbol] = abs(current_weight - target_weight)
        
        # Check for positions not in target
        for symbol, current_weight in current_weights.items():
            if symbol not in self.target_weights:
                drift[symbol] = current_weight
        
        return drift
    
    def _is_rebalance_due(self, current_date: datetime, days: Optional[int] = None) -> bool:
        """Check if rebalancing is due based on calendar."""
        if self.last_rebalance_date is None:
            return True
        
        if days:
            # Custom day check
            return (current_date - self.last_rebalance_date).days >= days
        
        # Standard frequency check
        if self.rebalance_frequency == RebalanceFrequency.DAILY:
            return (current_date - self.last_rebalance_date).days >= 1
        elif self.rebalance_frequency == RebalanceFrequency.WEEKLY:
            return (current_date - self.last_rebalance_date).days >= 7
        elif self.rebalance_frequency == RebalanceFrequency.MONTHLY:
            return (current_date - self.last_rebalance_date).days >= 30
        elif self.rebalance_frequency == RebalanceFrequency.QUARTERLY:
            return (current_date - self.last_rebalance_date).days >= 90
        elif self.rebalance_frequency == RebalanceFrequency.ANNUALLY:
            return (current_date - self.last_rebalance_date).days >= 365
        
        return False
    
    def _calculate_cppi_cushion(
        self,
        portfolio_value: float,
        market_conditions: Optional[Dict[str, float]]
    ) -> float:
        """Calculate CPPI cushion."""
        # Floor value (e.g., 80% of initial)
        floor_value = portfolio_value * 0.8
        
        # Cushion
        cushion = (portfolio_value - floor_value) / portfolio_value
        
        # Adjust for market conditions
        if market_conditions:
            volatility = market_conditions.get('volatility', 0.15)
            if volatility > 0.20:
                cushion *= 0.8  # Reduce cushion in high volatility
        
        return cushion
    
    def _apply_constraints(
        self,
        trades: Dict[str, float],
        constraints: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, float]:
        """Apply trading constraints."""
        constrained_trades = trades.copy()
        
        # Max trade size constraint
        max_trade_size = constraints.get('max_trade_size', 0.10) * portfolio_value
        for symbol, trade in constrained_trades.items():
            if abs(trade) > max_trade_size:
                constrained_trades[symbol] = np.sign(trade) * max_trade_size
        
        # Turnover constraint
        max_turnover = constraints.get('max_turnover', 0.50) * portfolio_value
        total_turnover = sum(abs(t) for t in constrained_trades.values())
        
        if total_turnover > max_turnover:
            # Scale down all trades proportionally
            scale_factor = max_turnover / total_turnover
            for symbol in constrained_trades:
                constrained_trades[symbol] *= scale_factor
        
        return constrained_trades
    
    def _optimize_trades(
        self,
        trades: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, float]:
        """Optimize trades to minimize costs."""
        optimized_trades = {}
        
        # Sort trades by size (largest first)
        sorted_trades = sorted(trades.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Keep only trades that significantly improve alignment
        cumulative_improvement = 0
        target_improvement = 0.9  # Capture 90% of rebalancing benefit
        
        total_drift_reduction = sum(abs(t) for _, t in sorted_trades)
        
        for symbol, trade in sorted_trades:
            optimized_trades[symbol] = trade
            cumulative_improvement += abs(trade)
            
            if cumulative_improvement / total_drift_reduction >= target_improvement:
                break
        
        return optimized_trades
    
    def _calculate_avg_rebalance_frequency(self) -> float:
        """Calculate average days between rebalances."""
        if len(self.rebalance_history) < 2:
            return 0
        
        dates = [r['date'] for r in self.rebalance_history]
        intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        
        return np.mean(intervals) if intervals else 0
    
    def _calculate_avg_drift_at_rebalance(self) -> float:
        """Calculate average drift when rebalancing occurs."""
        drift_values = []
        
        for record in self.rebalance_history:
            old_weights = record['old_weights']
            max_drift = max(
                abs(old_weights.get(s, 0) - self.target_weights.get(s, 0))
                for s in set(list(old_weights.keys()) + list(self.target_weights.keys()))
            )
            drift_values.append(max_drift)
        
        return np.mean(drift_values) if drift_values else 0