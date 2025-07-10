"""
Position Sizing Module

This module implements various position sizing algorithms including
Kelly Criterion, Volatility-based sizing, and Risk Parity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from enum import Enum
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SizingMethod(Enum):
    """Position sizing methods."""
    FIXED = "fixed"
    KELLY = "kelly"
    VOLATILITY = "volatility"
    RISK_PARITY = "risk_parity"
    ATR_BASED = "atr_based"
    PERCENT_RISK = "percent_risk"
    OPTIMAL_F = "optimal_f"
    DYNAMIC = "dynamic"

class PositionSizer:
    """
    Advanced position sizing algorithms for portfolio management.
    """
    
    def __init__(
        self,
        default_method: SizingMethod = SizingMethod.VOLATILITY,
        risk_per_trade: float = 0.02,
        max_position_size: float = 0.20,
        min_position_size: float = 0.01,
        kelly_fraction: float = 0.25,
        lookback_period: int = 60
    ):
        """
        Initialize position sizer.
        
        Args:
            default_method: Default sizing method
            risk_per_trade: Risk per trade (2% default)
            max_position_size: Maximum position size (20% default)
            min_position_size: Minimum position size (1% default)
            kelly_fraction: Fraction of Kelly criterion to use (25% default)
            lookback_period: Period for calculating statistics
        """
        self.default_method = default_method
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.kelly_fraction = kelly_fraction
        self.lookback_period = lookback_period
        
        # Cache for calculations
        self._kelly_cache = {}
        self._volatility_cache = {}
        
    def calculate_position_size(
        self,
        symbol: str,
        portfolio_value: float,
        signal_strength: float,
        market_data: pd.DataFrame,
        trade_history: Optional[pd.DataFrame] = None,
        method: Optional[SizingMethod] = None,
        stop_loss: Optional[float] = None,
        entry_price: Optional[float] = None
    ) -> Dict[str, Union[float, str]]:
        """
        Calculate position size based on specified method.
        
        Args:
            symbol: Asset symbol
            portfolio_value: Total portfolio value
            signal_strength: Trading signal strength (0-1)
            market_data: Historical market data
            trade_history: Historical trades for Kelly calculation
            method: Sizing method (uses default if None)
            stop_loss: Stop loss price for percent risk method
            entry_price: Entry price for percent risk method
            
        Returns:
            Dictionary with position size and metadata
        """
        method = method or self.default_method
        
        # Base position size
        base_size = portfolio_value * self.risk_per_trade
        
        # Calculate size based on method
        if method == SizingMethod.FIXED:
            size = self._fixed_size(portfolio_value, signal_strength)
            
        elif method == SizingMethod.KELLY:
            size = self._kelly_size(
                symbol, portfolio_value, signal_strength, 
                market_data, trade_history
            )
            
        elif method == SizingMethod.VOLATILITY:
            size = self._volatility_size(
                symbol, portfolio_value, signal_strength, market_data
            )
            
        elif method == SizingMethod.RISK_PARITY:
            size = self._risk_parity_size(
                symbol, portfolio_value, market_data
            )
            
        elif method == SizingMethod.ATR_BASED:
            size = self._atr_size(
                symbol, portfolio_value, signal_strength, market_data
            )
            
        elif method == SizingMethod.PERCENT_RISK:
            size = self._percent_risk_size(
                portfolio_value, stop_loss, entry_price
            )
            
        elif method == SizingMethod.OPTIMAL_F:
            size = self._optimal_f_size(
                symbol, portfolio_value, signal_strength, trade_history
            )
            
        elif method == SizingMethod.DYNAMIC:
            size = self._dynamic_size(
                symbol, portfolio_value, signal_strength, 
                market_data, trade_history
            )
            
        else:
            size = base_size * signal_strength
        
        # Apply position limits
        size = self._apply_limits(size, portfolio_value)
        
        # Calculate additional metrics
        position_pct = size / portfolio_value
        risk_amount = size * self.risk_per_trade
        
        return {
            'size': size,
            'position_pct': position_pct,
            'risk_amount': risk_amount,
            'method': method.value,
            'signal_strength': signal_strength,
            'portfolio_value': portfolio_value
        }
    
    def _fixed_size(self, portfolio_value: float, signal_strength: float) -> float:
        """Fixed position sizing."""
        return portfolio_value * self.risk_per_trade * signal_strength
    
    def _kelly_size(
        self,
        symbol: str,
        portfolio_value: float,
        signal_strength: float,
        market_data: pd.DataFrame,
        trade_history: Optional[pd.DataFrame]
    ) -> float:
        """Kelly Criterion position sizing."""
        # Check cache
        cache_key = f"{symbol}_{len(market_data)}"
        if cache_key in self._kelly_cache:
            kelly_pct = self._kelly_cache[cache_key]
        else:
            # Calculate win rate and win/loss ratio
            if trade_history is not None and len(trade_history) > 10:
                # Use actual trade history
                symbol_trades = trade_history[trade_history['symbol'] == symbol]
                if len(symbol_trades) > 5:
                    wins = symbol_trades[symbol_trades['return'] > 0]
                    losses = symbol_trades[symbol_trades['return'] <= 0]
                    
                    win_rate = len(wins) / len(symbol_trades) if len(symbol_trades) > 0 else 0.5
                    avg_win = wins['return'].mean() if len(wins) > 0 else 0.01
                    avg_loss = abs(losses['return'].mean()) if len(losses) > 0 else 0.01
                    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
                else:
                    # Use market data statistics
                    win_rate, win_loss_ratio = self._estimate_kelly_params(market_data)
            else:
                # Use market data statistics
                win_rate, win_loss_ratio = self._estimate_kelly_params(market_data)
            
            # Kelly formula: f = p - q/b
            # where p = win rate, q = loss rate, b = win/loss ratio
            kelly_pct = win_rate - (1 - win_rate) / win_loss_ratio
            
            # Apply Kelly fraction (for safety)
            kelly_pct = max(0, kelly_pct * self.kelly_fraction)
            
            # Cache result
            self._kelly_cache[cache_key] = kelly_pct
        
        # Calculate position size
        size = portfolio_value * kelly_pct * signal_strength
        
        return size
    
    def _volatility_size(
        self,
        symbol: str,
        portfolio_value: float,
        signal_strength: float,
        market_data: pd.DataFrame
    ) -> float:
        """Volatility-based position sizing."""
        # Check cache
        cache_key = f"{symbol}_{len(market_data)}"
        if cache_key in self._volatility_cache:
            volatility = self._volatility_cache[cache_key]
        else:
            # Calculate volatility
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.rolling(self.lookback_period).std().iloc[-1]
            
            # Annualize
            volatility = volatility * np.sqrt(252)
            
            # Cache result
            self._volatility_cache[cache_key] = volatility
        
        # Target volatility (e.g., 15% annualized)
        target_volatility = 0.15
        
        # Position size inversely proportional to volatility
        if volatility > 0:
            volatility_scalar = target_volatility / volatility
            volatility_scalar = min(2.0, max(0.5, volatility_scalar))  # Limit scaling
        else:
            volatility_scalar = 1.0
        
        size = portfolio_value * self.risk_per_trade * volatility_scalar * signal_strength
        
        return size
    
    def _risk_parity_size(
        self,
        symbol: str,
        portfolio_value: float,
        market_data: pd.DataFrame
    ) -> float:
        """Risk parity position sizing."""
        # Calculate asset volatility
        returns = market_data['close'].pct_change().dropna()
        volatility = returns.rolling(self.lookback_period).std().iloc[-1]
        
        # For risk parity, allocate inversely to volatility
        if volatility > 0:
            # Assume equal risk contribution target
            risk_budget = 1.0  # Equal risk
            size = (portfolio_value * risk_budget) / (volatility * np.sqrt(252))
            
            # Scale to reasonable range
            size = size * self.risk_per_trade
        else:
            size = portfolio_value * self.risk_per_trade
        
        return size
    
    def _atr_size(
        self,
        symbol: str,
        portfolio_value: float,
        signal_strength: float,
        market_data: pd.DataFrame
    ) -> float:
        """ATR-based position sizing."""
        # Calculate ATR
        atr = self._calculate_atr(market_data)
        current_price = market_data['close'].iloc[-1]
        
        # Position size based on ATR
        if atr > 0 and current_price > 0:
            # Risk amount per unit
            risk_per_unit = atr * 2  # 2 ATR stop
            
            # Units to trade
            units = (portfolio_value * self.risk_per_trade) / risk_per_unit
            
            # Position size
            size = units * current_price * signal_strength
        else:
            size = portfolio_value * self.risk_per_trade * signal_strength
        
        return size
    
    def _percent_risk_size(
        self,
        portfolio_value: float,
        stop_loss: Optional[float],
        entry_price: Optional[float]
    ) -> float:
        """Percent risk position sizing."""
        if stop_loss and entry_price and entry_price > stop_loss:
            # Calculate risk per share
            risk_per_share = entry_price - stop_loss
            
            # Position size
            shares = (portfolio_value * self.risk_per_trade) / risk_per_share
            size = shares * entry_price
        else:
            # Fallback to fixed sizing
            size = portfolio_value * self.risk_per_trade
        
        return size
    
    def _optimal_f_size(
        self,
        symbol: str,
        portfolio_value: float,
        signal_strength: float,
        trade_history: Optional[pd.DataFrame]
    ) -> float:
        """Optimal f position sizing."""
        if trade_history is not None and len(trade_history) > 20:
            # Get symbol-specific trades
            symbol_trades = trade_history[trade_history['symbol'] == symbol]
            
            if len(symbol_trades) > 10:
                returns = symbol_trades['return'].values
                
                # Calculate optimal f
                optimal_f = self._calculate_optimal_f(returns)
                
                # Apply fraction for safety
                optimal_f = optimal_f * 0.25
                
                # Position size
                size = portfolio_value * optimal_f * signal_strength
            else:
                # Fallback to volatility sizing
                size = self._volatility_size(symbol, portfolio_value, signal_strength, trade_history)
        else:
            # Fallback to fixed sizing
            size = portfolio_value * self.risk_per_trade * signal_strength
        
        return size
    
    def _dynamic_size(
        self,
        symbol: str,
        portfolio_value: float,
        signal_strength: float,
        market_data: pd.DataFrame,
        trade_history: Optional[pd.DataFrame]
    ) -> float:
        """Dynamic position sizing based on multiple factors."""
        # Start with base size
        base_size = portfolio_value * self.risk_per_trade
        
        # Factor 1: Signal strength
        signal_factor = signal_strength
        
        # Factor 2: Volatility adjustment
        returns = market_data['close'].pct_change().dropna()
        current_vol = returns.rolling(20).std().iloc[-1]
        avg_vol = returns.rolling(self.lookback_period).std().mean()
        vol_factor = avg_vol / current_vol if current_vol > 0 else 1.0
        vol_factor = min(1.5, max(0.5, vol_factor))
        
        # Factor 3: Trend strength
        sma_20 = market_data['close'].rolling(20).mean()
        sma_50 = market_data['close'].rolling(50).mean()
        trend_factor = 1.2 if sma_20.iloc[-1] > sma_50.iloc[-1] else 0.8
        
        # Factor 4: Win rate adjustment
        if trade_history is not None and len(trade_history) > 10:
            recent_trades = trade_history.tail(20)
            win_rate = len(recent_trades[recent_trades['return'] > 0]) / len(recent_trades)
            win_factor = 1.0 + (win_rate - 0.5) * 0.5  # Scale between 0.75 and 1.25
        else:
            win_factor = 1.0
        
        # Combine factors
        total_factor = signal_factor * vol_factor * trend_factor * win_factor
        
        # Calculate final size
        size = base_size * total_factor
        
        return size
    
    def _apply_limits(self, size: float, portfolio_value: float) -> float:
        """Apply position size limits."""
        # Maximum position size
        max_size = portfolio_value * self.max_position_size
        
        # Minimum position size
        min_size = portfolio_value * self.min_position_size
        
        # Apply limits
        size = max(min_size, min(max_size, size))
        
        return size
    
    def _estimate_kelly_params(self, market_data: pd.DataFrame) -> Tuple[float, float]:
        """Estimate Kelly parameters from market data."""
        returns = market_data['close'].pct_change().dropna()
        
        # Use recent period
        recent_returns = returns.tail(self.lookback_period)
        
        # Calculate win rate
        win_rate = len(recent_returns[recent_returns > 0]) / len(recent_returns)
        
        # Calculate win/loss ratio
        wins = recent_returns[recent_returns > 0]
        losses = recent_returns[recent_returns <= 0]
        
        avg_win = wins.mean() if len(wins) > 0 else 0.01
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.01
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        return win_rate, win_loss_ratio
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1] if not atr.empty else 0
    
    def _calculate_optimal_f(self, returns: np.ndarray) -> float:
        """Calculate optimal f for position sizing."""
        # Convert returns to profit/loss amounts
        pnl = returns
        
        # Find the largest loss
        max_loss = abs(min(pnl)) if min(pnl) < 0 else 0.01
        
        # Test different f values
        f_values = np.linspace(0.01, 0.99, 99)
        twr_values = []
        
        for f in f_values:
            twr = 1.0
            for ret in pnl:
                twr *= (1 + f * ret / max_loss)
                if twr <= 0:
                    twr = 0
                    break
            twr_values.append(twr)
        
        # Find optimal f
        optimal_idx = np.argmax(twr_values)
        optimal_f = f_values[optimal_idx]
        
        return optimal_f
    
    def get_position_summary(
        self,
        positions: Dict[str, float],
        portfolio_value: float
    ) -> pd.DataFrame:
        """
        Get summary of current positions and sizing.
        
        Args:
            positions: Current positions {symbol: value}
            portfolio_value: Total portfolio value
            
        Returns:
            DataFrame with position summary
        """
        summary_data = []
        
        for symbol, value in positions.items():
            position_pct = value / portfolio_value
            
            summary_data.append({
                'symbol': symbol,
                'value': value,
                'position_pct': position_pct,
                'vs_max': position_pct / self.max_position_size,
                'risk_amount': abs(value) * self.risk_per_trade
            })
        
        return pd.DataFrame(summary_data)