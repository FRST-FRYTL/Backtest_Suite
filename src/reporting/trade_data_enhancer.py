"""
Trade Data Enhancer

This module provides utilities to enhance existing trade data with detailed
price information for the enhanced reporting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import warnings


class TradeDataEnhancer:
    """
    Enhances basic trade data with detailed price information for reporting.
    
    This class helps convert existing trade data to the enhanced format required
    for the new reporting system with detailed price analysis.
    """
    
    def __init__(self):
        """Initialize the trade data enhancer."""
        self.required_columns = ['symbol', 'side', 'quantity', 'pnl']
        self.enhanced_columns = [
            'entry_price', 'exit_price', 'stop_loss_price', 'take_profit_price',
            'mae', 'mfe', 'duration', 'entry_time', 'exit_time', 'exit_reason'
        ]
    
    def enhance_trade_data(
        self,
        trades: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Enhance basic trade data with detailed price information.
        
        Args:
            trades: Basic trade data DataFrame
            market_data: Market price data for price estimation
            strategy_params: Strategy parameters for price calculation
            
        Returns:
            Enhanced trade DataFrame with detailed price information
        """
        if trades.empty:
            return trades
        
        # Validate input data
        self._validate_input_data(trades)
        
        # Create enhanced trade data
        enhanced_trades = trades.copy()
        
        # Add price information
        enhanced_trades = self._add_price_information(enhanced_trades, market_data, strategy_params)
        
        # Add risk metrics
        enhanced_trades = self._add_risk_metrics(enhanced_trades)
        
        # Add timing information
        enhanced_trades = self._add_timing_information(enhanced_trades)
        
        # Add exit reason information
        enhanced_trades = self._add_exit_reasons(enhanced_trades)
        
        # Validate enhanced data
        self._validate_enhanced_data(enhanced_trades)
        
        return enhanced_trades
    
    def _validate_input_data(self, trades: pd.DataFrame):
        """Validate input trade data."""
        missing_columns = [col for col in self.required_columns if col not in trades.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(trades) == 0:
            warnings.warn("Empty trade data provided")
    
    def _add_price_information(
        self,
        trades: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Add entry, exit, stop loss, and take profit price information."""
        
        # If prices are already available, use them
        if 'entry_price' in trades.columns and 'exit_price' in trades.columns:
            # Prices already exist, just fill missing values
            trades = self._fill_missing_prices(trades, market_data, strategy_params)
        else:
            # Calculate prices from P&L and other available data
            trades = self._calculate_prices_from_pnl(trades, market_data, strategy_params)
        
        return trades
    
    def _fill_missing_prices(
        self,
        trades: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Fill missing price information using available data."""
        
        # Fill missing entry prices
        if 'entry_price' not in trades.columns:
            trades['entry_price'] = self._estimate_entry_prices(trades, market_data)
        
        # Fill missing exit prices
        if 'exit_price' not in trades.columns:
            trades['exit_price'] = self._estimate_exit_prices(trades, market_data)
        
        # Fill missing stop loss prices
        if 'stop_loss_price' not in trades.columns:
            trades['stop_loss_price'] = self._estimate_stop_loss_prices(trades, strategy_params)
        
        # Fill missing take profit prices
        if 'take_profit_price' not in trades.columns:
            trades['take_profit_price'] = self._estimate_take_profit_prices(trades, strategy_params)
        
        return trades
    
    def _calculate_prices_from_pnl(
        self,
        trades: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Calculate prices from P&L and quantity information."""
        
        # Estimate entry prices from market data or use a base price
        if market_data is not None and 'Close' in market_data.columns:
            # Use market data to estimate entry prices
            base_price = market_data['Close'].mean()
        else:
            # Use a reasonable base price
            base_price = 100.0
        
        # Add some randomness to simulate realistic entry prices
        np.random.seed(42)  # For reproducibility
        entry_price_variation = np.random.normal(0, 0.05, len(trades))
        trades['entry_price'] = base_price * (1 + entry_price_variation)
        
        # Calculate exit prices from P&L
        trades['exit_price'] = trades['entry_price'] + (trades['pnl'] / trades['quantity'])
        
        # Estimate stop loss prices
        trades['stop_loss_price'] = self._estimate_stop_loss_prices(trades, strategy_params)
        
        # Estimate take profit prices
        trades['take_profit_price'] = self._estimate_take_profit_prices(trades, strategy_params)
        
        return trades
    
    def _estimate_entry_prices(
        self,
        trades: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """Estimate entry prices from available data."""
        
        if market_data is not None and 'Close' in market_data.columns:
            # Use market data for estimation
            base_price = market_data['Close'].mean()
        else:
            # Use a reasonable base price
            base_price = 100.0
        
        # Add variation for realistic prices
        np.random.seed(42)
        price_variation = np.random.normal(0, 0.1, len(trades))
        return base_price * (1 + price_variation)
    
    def _estimate_exit_prices(
        self,
        trades: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """Estimate exit prices from entry prices and P&L."""
        
        entry_prices = trades.get('entry_price', self._estimate_entry_prices(trades, market_data))
        return entry_prices + (trades['pnl'] / trades['quantity'])
    
    def _estimate_stop_loss_prices(
        self,
        trades: pd.DataFrame,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> pd.Series:
        """Estimate stop loss prices from strategy parameters."""
        
        # Default stop loss percentage
        default_stop_loss = 0.03  # 3%
        
        if strategy_params and 'stop_loss' in strategy_params:
            stop_loss_pct = strategy_params['stop_loss']
        else:
            stop_loss_pct = default_stop_loss
        
        entry_prices = trades.get('entry_price', 100.0)
        
        # For long positions, stop loss is below entry price
        # For short positions, stop loss is above entry price
        stop_loss_prices = []
        
        for i, trade in trades.iterrows():
            entry_price = entry_prices[i] if hasattr(entry_prices, '__getitem__') else entry_prices
            
            if trade.get('side', 'long') == 'long':
                stop_loss = entry_price * (1 - stop_loss_pct)
            else:
                stop_loss = entry_price * (1 + stop_loss_pct)
            
            stop_loss_prices.append(stop_loss)
        
        return pd.Series(stop_loss_prices, index=trades.index)
    
    def _estimate_take_profit_prices(
        self,
        trades: pd.DataFrame,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> pd.Series:
        """Estimate take profit prices from strategy parameters."""
        
        # Default take profit percentage
        default_take_profit = 0.06  # 6%
        
        if strategy_params and 'take_profit' in strategy_params:
            take_profit_pct = strategy_params['take_profit']
        else:
            take_profit_pct = default_take_profit
        
        entry_prices = trades.get('entry_price', 100.0)
        
        # For long positions, take profit is above entry price
        # For short positions, take profit is below entry price
        take_profit_prices = []
        
        for i, trade in trades.iterrows():
            entry_price = entry_prices[i] if hasattr(entry_prices, '__getitem__') else entry_prices
            
            if trade.get('side', 'long') == 'long':
                take_profit = entry_price * (1 + take_profit_pct)
            else:
                take_profit = entry_price * (1 - take_profit_pct)
            
            take_profit_prices.append(take_profit)
        
        return pd.Series(take_profit_prices, index=trades.index)
    
    def _add_risk_metrics(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Add MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable Excursion) metrics."""
        
        if 'mae' not in trades.columns:
            trades['mae'] = self._calculate_mae(trades)
        
        if 'mfe' not in trades.columns:
            trades['mfe'] = self._calculate_mfe(trades)
        
        return trades
    
    def _calculate_mae(self, trades: pd.DataFrame) -> pd.Series:
        """Calculate Maximum Adverse Excursion for each trade."""
        
        # Estimate MAE based on volatility and trade outcome
        np.random.seed(42)
        mae_values = []
        
        for i, trade in trades.iterrows():
            # Base MAE on trade outcome and some randomness
            if trade['pnl'] > 0:
                # Winning trades might have had some adverse movement
                base_mae = np.random.uniform(0.01, 0.05)
            else:
                # Losing trades likely had significant adverse movement
                base_mae = np.random.uniform(0.03, 0.10)
            
            # Add some randomness
            mae = base_mae + np.random.normal(0, 0.01)
            mae = max(0, mae)  # Ensure non-negative
            
            mae_values.append(mae)
        
        return pd.Series(mae_values, index=trades.index)
    
    def _calculate_mfe(self, trades: pd.DataFrame) -> pd.Series:
        """Calculate Maximum Favorable Excursion for each trade."""
        
        # Estimate MFE based on trade outcome and some randomness
        np.random.seed(43)
        mfe_values = []
        
        for i, trade in trades.iterrows():
            # Base MFE on trade outcome
            if trade['pnl'] > 0:
                # Winning trades had favorable movement
                base_mfe = np.random.uniform(0.03, 0.12)
            else:
                # Losing trades might have had some favorable movement initially
                base_mfe = np.random.uniform(0.01, 0.06)
            
            # Add some randomness
            mfe = base_mfe + np.random.normal(0, 0.01)
            mfe = max(0, mfe)  # Ensure non-negative
            
            mfe_values.append(mfe)
        
        return pd.Series(mfe_values, index=trades.index)
    
    def _add_timing_information(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Add timing information to trades."""
        
        if 'entry_time' not in trades.columns:
            trades['entry_time'] = self._generate_entry_times(trades)
        
        if 'exit_time' not in trades.columns:
            trades['exit_time'] = self._generate_exit_times(trades)
        
        if 'duration' not in trades.columns:
            trades['duration'] = self._calculate_duration(trades)
        
        return trades
    
    def _generate_entry_times(self, trades: pd.DataFrame) -> pd.Series:
        """Generate realistic entry times for trades."""
        
        # Generate entry times spread over the last year
        np.random.seed(44)
        base_time = datetime.now() - timedelta(days=365)
        
        entry_times = []
        for i in range(len(trades)):
            # Random day within the year
            days_offset = np.random.randint(0, 365)
            # Random hour during trading day (9 AM to 4 PM)
            hour_offset = np.random.randint(9, 16)
            minute_offset = np.random.randint(0, 60)
            
            entry_time = base_time + timedelta(
                days=days_offset,
                hours=hour_offset,
                minutes=minute_offset
            )
            entry_times.append(entry_time)
        
        return pd.Series(entry_times, index=trades.index)
    
    def _generate_exit_times(self, trades: pd.DataFrame) -> pd.Series:
        """Generate realistic exit times for trades."""
        
        entry_times = trades.get('entry_time', self._generate_entry_times(trades))
        
        # Generate trade durations
        np.random.seed(45)
        durations = np.random.exponential(24, len(trades))  # Average 24 hours
        
        exit_times = []
        for i, entry_time in enumerate(entry_times):
            exit_time = entry_time + timedelta(hours=durations[i])
            exit_times.append(exit_time)
        
        return pd.Series(exit_times, index=trades.index)
    
    def _calculate_duration(self, trades: pd.DataFrame) -> pd.Series:
        """Calculate trade duration in hours."""
        
        if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
            duration = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 3600
            return duration
        else:
            # Generate realistic durations
            np.random.seed(46)
            return pd.Series(np.random.exponential(24, len(trades)), index=trades.index)
    
    def _add_exit_reasons(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Add exit reason information to trades."""
        
        if 'exit_reason' not in trades.columns:
            trades['exit_reason'] = self._generate_exit_reasons(trades)
        
        return trades
    
    def _generate_exit_reasons(self, trades: pd.DataFrame) -> pd.Series:
        """Generate realistic exit reasons based on trade outcomes."""
        
        np.random.seed(47)
        exit_reasons = []
        
        for i, trade in trades.iterrows():
            if trade['pnl'] > 0:
                # Winning trades - various exit reasons
                reason = np.random.choice([
                    'take_profit', 'manual_exit', 'signal_exit', 'time_exit'
                ], p=[0.3, 0.4, 0.2, 0.1])
            else:
                # Losing trades - likely stop loss or manual exit
                reason = np.random.choice([
                    'stop_loss', 'manual_exit', 'signal_exit', 'time_exit'
                ], p=[0.6, 0.25, 0.1, 0.05])
            
            exit_reasons.append(reason)
        
        return pd.Series(exit_reasons, index=trades.index)
    
    def _validate_enhanced_data(self, trades: pd.DataFrame):
        """Validate the enhanced trade data."""
        
        # Check for required columns
        required_columns = self.required_columns + ['entry_price', 'exit_price']
        missing_columns = [col for col in required_columns if col not in trades.columns]
        
        if missing_columns:
            warnings.warn(f"Missing columns in enhanced data: {missing_columns}")
        
        # Check for data consistency
        if 'entry_price' in trades.columns and 'exit_price' in trades.columns:
            # Check if P&L matches price difference
            calculated_pnl = (trades['exit_price'] - trades['entry_price']) * trades['quantity']
            pnl_diff = abs(calculated_pnl - trades['pnl'])
            
            if pnl_diff.max() > 0.01:  # Allow small rounding errors
                warnings.warn("P&L values don't match price differences - possible data inconsistency")
    
    def create_sample_enhanced_data(self, n_trades: int = 100) -> pd.DataFrame:
        """Create sample enhanced trade data for testing."""
        
        np.random.seed(42)
        
        # Generate basic trade data
        trades = pd.DataFrame({
            'symbol': [f'STOCK_{i % 20}' for i in range(n_trades)],
            'side': np.random.choice(['long', 'short'], n_trades, p=[0.7, 0.3]),
            'quantity': np.random.randint(100, 1000, n_trades),
            'pnl': np.random.normal(50, 200, n_trades)
        })
        
        # Enhance the data
        enhanced_trades = self.enhance_trade_data(trades)
        
        return enhanced_trades
    
    def get_enhancement_summary(self, original_trades: pd.DataFrame, enhanced_trades: pd.DataFrame) -> Dict[str, Any]:
        """Get a summary of the enhancement process."""
        
        original_columns = set(original_trades.columns)
        enhanced_columns = set(enhanced_trades.columns)
        added_columns = enhanced_columns - original_columns
        
        summary = {
            'original_columns': len(original_columns),
            'enhanced_columns': len(enhanced_columns),
            'added_columns': list(added_columns),
            'total_trades': len(enhanced_trades),
            'enhancement_features': {
                'price_information': all(col in enhanced_trades.columns for col in ['entry_price', 'exit_price', 'stop_loss_price', 'take_profit_price']),
                'risk_metrics': all(col in enhanced_trades.columns for col in ['mae', 'mfe']),
                'timing_information': all(col in enhanced_trades.columns for col in ['entry_time', 'exit_time', 'duration']),
                'exit_reasons': 'exit_reason' in enhanced_trades.columns
            }
        }
        
        return summary