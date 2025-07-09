"""Monthly Contribution Trading Strategy with Multi-Indicator Approach."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..indicators import RSI, BollingerBands, VWAP, FearGreedIndex
from .builder import Strategy, StrategyBuilder, PositionSizing, RiskManagement
from .rules import Rule, Condition, ComparisonOperator, LogicalOperator
from .signals import SignalGenerator


class MonthlyContributionStrategy:
    """
    A robust trading strategy designed for consistent growth with monthly contributions.
    
    Features:
    - Multi-indicator approach (RSI, Bollinger Bands, VWAP, Fear & Greed)
    - Dollar-cost averaging with $500 monthly contributions
    - Dynamic position sizing based on Kelly Criterion and account value
    - 20-30% cash reserve management
    - Risk management with stop-loss and trailing stops
    - Sector rotation capability
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        monthly_contribution: float = 500,
        cash_reserve_target: float = 0.25,  # 25% cash reserve
        max_risk_per_trade: float = 0.02,   # 2% max risk per trade
        use_fear_greed: bool = True
    ):
        """
        Initialize Monthly Contribution Strategy.
        
        Args:
            initial_capital: Starting capital
            monthly_contribution: Monthly contribution amount
            cash_reserve_target: Target cash reserve percentage (0.2-0.3)
            max_risk_per_trade: Maximum risk per trade as percentage
            use_fear_greed: Whether to use Fear & Greed Index
        """
        self.initial_capital = initial_capital
        self.monthly_contribution = monthly_contribution
        self.cash_reserve_target = cash_reserve_target
        self.max_risk_per_trade = max_risk_per_trade
        self.use_fear_greed = use_fear_greed
        
        # Initialize indicators
        self.rsi = RSI(period=14)
        self.bollinger = BollingerBands(period=20, std_dev=2)
        self.vwap = VWAP()
        self.fear_greed = FearGreedIndex() if use_fear_greed else None
        
        # Trading parameters
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.bb_squeeze_threshold = 0.01  # Bollinger Band squeeze detection
        
        # Performance tracking
        self.trade_history = []
        self.monthly_contributions = []
        
    def build_strategy(self) -> Strategy:
        """
        Build the complete trading strategy with all rules.
        
        Returns:
            Strategy object with entry/exit rules and risk management
        """
        builder = StrategyBuilder(name="Monthly Contribution Multi-Indicator Strategy")
        builder.set_description(
            "Robust strategy combining RSI, Bollinger Bands, VWAP, and Fear & Greed Index "
            "with monthly dollar-cost averaging and dynamic position sizing."
        )
        
        # Entry Rules
        self._add_entry_rules(builder)
        
        # Exit Rules
        self._add_exit_rules(builder)
        
        # Market Regime Filters
        self._add_market_filters(builder)
        
        # Position Sizing
        builder.set_position_sizing(
            method="kelly",  # Use Kelly Criterion
            size=0.25,       # Max 25% Kelly fraction
            max_position=0.15,  # Max 15% per position
            scale_in=True,   # Allow scaling into positions
            scale_out=True   # Allow scaling out of positions
        )
        
        # Risk Management
        builder.set_risk_management(
            stop_loss=0.02,  # 2% stop loss
            stop_loss_type="percent",
            take_profit=0.10,  # 10% take profit
            take_profit_type="percent",
            trailing_stop=0.03,  # 3% trailing stop
            time_stop=20,  # Exit after 20 days if no profit
            max_positions=8  # Maximum 8 concurrent positions
        )
        
        return builder.build()
        
    def _add_entry_rules(self, builder: StrategyBuilder):
        """Add entry rules to the strategy."""
        
        # Rule 1: RSI Oversold with Bollinger Band Support
        oversold_bounce = Rule(name="Oversold Bounce", operator=LogicalOperator.AND)
        oversold_bounce.add_condition("rsi", ComparisonOperator.LT, self.rsi_oversold)
        oversold_bounce.add_condition("close", ComparisonOperator.LT, "bb_lower")
        oversold_bounce.add_condition("volume", ComparisonOperator.GT, "volume_sma_20")
        builder.add_entry_rule(oversold_bounce)
        
        # Rule 2: Bollinger Band Squeeze Breakout
        bb_breakout = Rule(name="BB Squeeze Breakout", operator=LogicalOperator.AND)
        bb_breakout.add_condition("bb_width", ComparisonOperator.LT, self.bb_squeeze_threshold)
        bb_breakout.add_condition("close", ComparisonOperator.CROSS_ABOVE, "bb_upper")
        bb_breakout.add_condition("volume", ComparisonOperator.GT, "volume_sma_20")
        builder.add_entry_rule(bb_breakout)
        
        # Rule 3: VWAP Institutional Support
        vwap_support = Rule(name="VWAP Support", operator=LogicalOperator.AND)
        vwap_support.add_condition("close", ComparisonOperator.GT, "vwap")
        vwap_support.add_condition("close", ComparisonOperator.CROSS_ABOVE, "vwap")
        vwap_support.add_condition("vwap_bands_width", ComparisonOperator.LT, 0.02)
        builder.add_entry_rule(vwap_support)
        
        # Rule 4: Fear & Greed Extreme Fear
        if self.use_fear_greed:
            fear_entry = Rule(name="Extreme Fear Entry", operator=LogicalOperator.AND)
            fear_entry.add_condition("fear_greed", ComparisonOperator.LT, 25)
            fear_entry.add_condition("rsi", ComparisonOperator.LT, 40)
            builder.add_entry_rule(fear_entry)
            
    def _add_exit_rules(self, builder: StrategyBuilder):
        """Add exit rules to the strategy."""
        
        # Rule 1: RSI Overbought
        overbought_exit = Rule(name="Overbought Exit", operator=LogicalOperator.AND)
        overbought_exit.add_condition("rsi", ComparisonOperator.GT, self.rsi_overbought)
        overbought_exit.add_condition("close", ComparisonOperator.GT, "bb_upper")
        builder.add_exit_rule(overbought_exit)
        
        # Rule 2: VWAP Resistance
        vwap_resistance = Rule(name="VWAP Resistance", operator=LogicalOperator.AND)
        vwap_resistance.add_condition("close", ComparisonOperator.LT, "vwap")
        vwap_resistance.add_condition("close", ComparisonOperator.CROSS_BELOW, "vwap")
        builder.add_exit_rule(vwap_resistance)
        
        # Rule 3: Extreme Greed
        if self.use_fear_greed:
            greed_exit = Rule(name="Extreme Greed Exit", operator=LogicalOperator.AND)
            greed_exit.add_condition("fear_greed", ComparisonOperator.GT, 75)
            greed_exit.add_condition("rsi", ComparisonOperator.GT, 60)
            builder.add_exit_rule(greed_exit)
            
        # Rule 4: Bollinger Band Mean Reversion
        bb_mean_reversion = Rule(name="BB Mean Reversion", operator=LogicalOperator.AND)
        bb_mean_reversion.add_condition("close", ComparisonOperator.CROSS_BELOW, "bb_middle")
        bb_mean_reversion.add_condition("position_profit_pct", ComparisonOperator.GT, 0.05)
        builder.add_exit_rule(bb_mean_reversion)
        
    def _add_market_filters(self, builder: StrategyBuilder):
        """Add market regime filters."""
        
        # Filter 1: Avoid extreme volatility
        volatility_filter = Rule(name="Volatility Filter", operator=LogicalOperator.AND)
        volatility_filter.add_condition("atr_pct", ComparisonOperator.LT, 0.05)  # ATR < 5% of price
        builder.add_filter(volatility_filter)
        
        # Filter 2: Minimum liquidity
        liquidity_filter = Rule(name="Liquidity Filter", operator=LogicalOperator.AND)
        liquidity_filter.add_condition("volume", ComparisonOperator.GT, 1000000)  # Min 1M volume
        liquidity_filter.add_condition("dollar_volume", ComparisonOperator.GT, 10000000)  # Min $10M
        builder.add_filter(liquidity_filter)
        
        # Filter 3: Trend alignment
        trend_filter = Rule(name="Trend Filter", operator=LogicalOperator.OR)
        trend_filter.add_condition("sma_50", ComparisonOperator.GT, "sma_200")  # Uptrend
        trend_filter.add_condition("rsi", ComparisonOperator.LT, 30)  # Or oversold
        builder.add_filter(trend_filter)
        
    def calculate_position_size(
        self,
        account_value: float,
        current_price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        volatility: float
    ) -> Tuple[int, float]:
        """
        Calculate position size using Kelly Criterion with safety constraints.
        
        Args:
            account_value: Current account value
            current_price: Current asset price
            win_rate: Historical win rate
            avg_win: Average winning trade return
            avg_loss: Average losing trade return
            volatility: Current volatility (ATR percentage)
            
        Returns:
            Tuple of (shares, position_value)
        """
        # Calculate available capital (respecting cash reserve)
        available_capital = account_value * (1 - self.cash_reserve_target)
        
        # Kelly Criterion calculation
        if win_rate > 0 and avg_loss > 0:
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            kelly_fraction = (b * p - q) / b
            
            # Apply safety constraints
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            # Adjust for volatility
            volatility_adjustment = 1 / (1 + volatility * 10)  # Reduce size in high volatility
            adjusted_kelly = kelly_fraction * volatility_adjustment
        else:
            # Default to conservative sizing
            adjusted_kelly = 0.02
            
        # Calculate position value
        max_position_value = available_capital * 0.15  # Max 15% per position
        kelly_position_value = available_capital * adjusted_kelly
        position_value = min(kelly_position_value, max_position_value)
        
        # Apply risk constraint
        max_risk_value = account_value * self.max_risk_per_trade
        max_shares_by_risk = int(max_risk_value / (current_price * 0.02))  # 2% stop loss
        
        # Calculate final shares
        shares_by_value = int(position_value / current_price)
        shares = min(shares_by_value, max_shares_by_risk)
        
        return shares, shares * current_price
        
    def process_monthly_contribution(
        self,
        current_date: datetime,
        account_value: float,
        cash_balance: float
    ) -> Dict[str, float]:
        """
        Process monthly contribution with intelligent allocation.
        
        Args:
            current_date: Current date
            account_value: Total account value
            cash_balance: Current cash balance
            
        Returns:
            Dictionary with contribution allocation
        """
        contribution = {
            'date': current_date,
            'amount': self.monthly_contribution,
            'cash_allocation': 0,
            'investment_allocation': 0
        }
        
        # Calculate current cash percentage
        cash_percentage = cash_balance / account_value
        
        # Allocate contribution to maintain target cash reserve
        if cash_percentage < self.cash_reserve_target:
            # Need to build cash reserve
            cash_needed = (self.cash_reserve_target * account_value) - cash_balance
            contribution['cash_allocation'] = min(self.monthly_contribution, cash_needed)
            contribution['investment_allocation'] = self.monthly_contribution - contribution['cash_allocation']
        else:
            # Cash reserve adequate, invest full contribution
            contribution['investment_allocation'] = self.monthly_contribution
            
        self.monthly_contributions.append(contribution)
        return contribution
        
    def should_rebalance(
        self,
        positions: List[Dict],
        account_value: float,
        current_date: datetime
    ) -> bool:
        """
        Determine if portfolio should be rebalanced.
        
        Args:
            positions: Current positions
            account_value: Total account value
            current_date: Current date
            
        Returns:
            True if rebalancing needed
        """
        if not positions:
            return False
            
        # Check if any position exceeds max allocation
        for position in positions:
            position_value = position['shares'] * position['current_price']
            position_pct = position_value / account_value
            
            if position_pct > 0.20:  # Position exceeds 20%
                return True
                
        # Check if it's quarterly rebalancing time
        if current_date.month in [3, 6, 9, 12] and current_date.day == 1:
            return True
            
        return False
        
    def get_rebalancing_orders(
        self,
        positions: List[Dict],
        account_value: float,
        target_allocations: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Generate rebalancing orders to maintain target allocations.
        
        Args:
            positions: Current positions
            account_value: Total account value
            target_allocations: Target allocations by symbol
            
        Returns:
            List of rebalancing orders
        """
        orders = []
        
        # Calculate current allocations
        current_allocations = {}
        for position in positions:
            symbol = position['symbol']
            position_value = position['shares'] * position['current_price']
            current_allocations[symbol] = position_value / account_value
            
        # Use equal weight if no target provided
        if not target_allocations:
            num_positions = len(positions)
            target_allocations = {p['symbol']: 1/num_positions for p in positions}
            
        # Calculate rebalancing trades
        for symbol, target_pct in target_allocations.items():
            current_pct = current_allocations.get(symbol, 0)
            diff_pct = target_pct - current_pct
            
            if abs(diff_pct) > 0.02:  # Only rebalance if difference > 2%
                # Find position
                position = next((p for p in positions if p['symbol'] == symbol), None)
                if position:
                    current_shares = position['shares']
                    target_value = account_value * target_pct
                    target_shares = int(target_value / position['current_price'])
                    share_diff = target_shares - current_shares
                    
                    if share_diff != 0:
                        orders.append({
                            'symbol': symbol,
                            'action': 'buy' if share_diff > 0 else 'sell',
                            'shares': abs(share_diff),
                            'price': position['current_price'],
                            'reason': 'rebalancing'
                        })
                        
        return orders
        
    def analyze_performance(self) -> Dict:
        """
        Analyze strategy performance including contribution impact.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trade_history:
            return {}
            
        # Calculate basic metrics
        total_trades = len(self.trade_history)
        winning_trades = [t for t in self.trade_history if t['return'] > 0]
        losing_trades = [t for t in self.trade_history if t['return'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Calculate returns
        returns = [t['return'] for t in self.trade_history]
        avg_return = np.mean(returns) if returns else 0
        
        avg_win = np.mean([t['return'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['return'] for t in losing_trades]) if losing_trades else 0
        
        # Calculate contribution metrics
        total_contributions = sum(c['amount'] for c in self.monthly_contributions)
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown()
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'total_contributions': total_contributions,
            'contribution_count': len(self.monthly_contributions),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_trade_duration': np.mean([t['duration'].days for t in self.trade_history])
        }
        
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from trade history."""
        if not self.trade_history:
            return 0
            
        cumulative_returns = []
        cumulative = 1.0
        
        for trade in self.trade_history:
            cumulative *= (1 + trade['return'])
            cumulative_returns.append(cumulative)
            
        # Calculate drawdowns
        peak = cumulative_returns[0]
        max_drawdown = 0
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown
        
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if not returns:
            return 0
            
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate/252  # Daily risk-free rate
        
        if len(excess_returns) < 2:
            return 0
            
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)