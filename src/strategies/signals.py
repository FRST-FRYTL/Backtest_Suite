"""Signal generation from strategy rules."""

from typing import List, Optional

import pandas as pd
import numpy as np

from .rules import Rule


class SignalGenerator:
    """Generates trading signals from strategy rules."""
    
    def generate(self, strategy, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all signals for a strategy.
        
        Args:
            strategy: Strategy object
            data: Market data DataFrame
            
        Returns:
            DataFrame with signal columns
        """
        signals = pd.DataFrame(index=data.index)
        
        # Generate entry signals
        signals['entry'] = self._generate_entry_signals(
            strategy.entry_rules,
            strategy.filters,
            data
        )
        
        # Generate exit signals
        signals['exit'] = self._generate_exit_signals(
            strategy.exit_rules,
            data
        )
        
        # Generate position sizing
        signals['position_size'] = self._calculate_position_sizes(
            strategy.position_sizing,
            signals['entry'],
            data
        )
        
        # Generate stop levels
        stop_levels = self._calculate_stop_levels(
            strategy.risk_management,
            signals['entry'],
            data
        )
        signals['stop_loss'] = stop_levels['stop_loss']
        signals['take_profit'] = stop_levels['take_profit']
        
        # Add signal strength and confidence
        signals['signal_strength'] = self._calculate_signal_strength(
            strategy.entry_rules,
            data,
            signals['entry']
        )
        
        return signals
        
    def _generate_entry_signals(
        self,
        entry_rules: List[Rule],
        filters: List[Rule],
        data: pd.DataFrame
    ) -> pd.Series:
        """Generate entry signals from rules and filters."""
        if not entry_rules:
            return pd.Series(False, index=data.index)
            
        # Evaluate filters first
        filter_pass = pd.Series(True, index=data.index)
        for filter_rule in filters:
            filter_pass &= filter_rule.evaluate_series(data)
            
        # Evaluate entry rules
        entry_signal = pd.Series(False, index=data.index)
        for rule in entry_rules:
            rule_signal = rule.evaluate_series(data)
            entry_signal |= rule_signal  # OR logic between rules
            
        # Apply filters
        return entry_signal & filter_pass
        
    def _generate_exit_signals(
        self,
        exit_rules: List[Rule],
        data: pd.DataFrame
    ) -> pd.Series:
        """Generate exit signals from rules."""
        if not exit_rules:
            return pd.Series(False, index=data.index)
            
        exit_signal = pd.Series(False, index=data.index)
        for rule in exit_rules:
            rule_signal = rule.evaluate_series(data)
            exit_signal |= rule_signal  # OR logic between rules
            
        return exit_signal
        
    def _calculate_position_sizes(
        self,
        position_sizing,
        entry_signals: pd.Series,
        data: pd.DataFrame
    ) -> pd.Series:
        """Calculate position sizes for entry signals."""
        sizes = pd.Series(0.0, index=data.index)
        
        # Simple implementation - in practice, this would consider
        # account balance, volatility, etc.
        if position_sizing.method == "fixed":
            sizes[entry_signals] = position_sizing.size
        elif position_sizing.method == "percent":
            # Percentage of account (simplified)
            sizes[entry_signals] = position_sizing.size
            
        return sizes
        
    def _calculate_stop_levels(
        self,
        risk_management,
        entry_signals: pd.Series,
        data: pd.DataFrame
    ) -> dict:
        """Calculate stop loss and take profit levels."""
        stop_loss = pd.Series(np.nan, index=data.index)
        take_profit = pd.Series(np.nan, index=data.index)
        
        # Get entry prices
        entry_prices = data['close'].where(entry_signals)
        
        # Calculate ATR if needed
        atr = None
        if risk_management.stop_loss_type == "atr" or risk_management.take_profit_type == "atr":
            if 'atr' in data.columns:
                atr = data['atr']
            else:
                # Simple ATR calculation
                high_low = data['high'] - data['low']
                high_close = abs(data['high'] - data['close'].shift(1))
                low_close = abs(data['low'] - data['close'].shift(1))
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(window=14).mean()
                
        # Calculate stops for each entry
        for idx in entry_signals[entry_signals].index:
            entry_price = entry_prices.loc[idx]
            
            if pd.notna(entry_price):
                stops = risk_management.calculate_stops(
                    entry_price,
                    atr.loc[idx] if atr is not None else None
                )
                
                if stops['stop_loss'] is not None:
                    stop_loss.loc[idx] = stops['stop_loss']
                if stops['take_profit'] is not None:
                    take_profit.loc[idx] = stops['take_profit']
                    
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        
    def _calculate_signal_strength(
        self,
        entry_rules: List[Rule],
        data: pd.DataFrame,
        entry_signals: pd.Series
    ) -> pd.Series:
        """
        Calculate signal strength based on how many rules are triggered.
        
        Returns:
            Series with signal strength (0-1)
        """
        strength = pd.Series(0.0, index=data.index)
        
        if not entry_rules:
            return strength
            
        # Count how many rules are satisfied at each point
        for idx in entry_signals[entry_signals].index:
            satisfied_count = 0
            for rule in entry_rules:
                if rule.evaluate(data, idx):
                    satisfied_count += 1
                    
            strength.loc[idx] = satisfied_count / len(entry_rules)
            
        return strength
        
    def generate_trade_list(
        self,
        signals: pd.DataFrame,
        data: pd.DataFrame
    ) -> List[dict]:
        """
        Generate list of trades from signals.
        
        Args:
            signals: DataFrame with signals
            data: Market data
            
        Returns:
            List of trade dictionaries
        """
        trades = []
        position_open = False
        current_trade = None
        
        for idx, row in signals.iterrows():
            if row['entry'] and not position_open:
                # Open new position
                current_trade = {
                    'entry_time': idx,
                    'entry_price': data.loc[idx, 'close'],
                    'position_size': row['position_size'],
                    'stop_loss': row['stop_loss'],
                    'take_profit': row['take_profit'],
                    'signal_strength': row['signal_strength']
                }
                position_open = True
                
            elif position_open and (row['exit'] or idx == signals.index[-1]):
                # Close position
                if current_trade:
                    current_trade['exit_time'] = idx
                    current_trade['exit_price'] = data.loc[idx, 'close']
                    current_trade['duration'] = idx - current_trade['entry_time']
                    current_trade['return'] = (
                        (current_trade['exit_price'] - current_trade['entry_price']) /
                        current_trade['entry_price']
                    )
                    trades.append(current_trade)
                    
                position_open = False
                current_trade = None
                
        return trades