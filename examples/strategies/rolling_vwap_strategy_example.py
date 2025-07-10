"""
Example strategy using Rolling VWAP indicators.

This example demonstrates how to use the rolling VWAP implementation
in a trading strategy, showing:
- Multiple rolling VWAP periods
- Standard deviation bands
- Cross detection
- Position relative to VWAP
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class RollingVWAPStrategy:
    """
    Trading strategy based on Rolling VWAP with multiple timeframes.
    
    Key concepts:
    - Short-term VWAP (5-20 periods) for entry/exit timing
    - Medium-term VWAP (50 periods) for trend direction
    - Long-term VWAP (200 periods) for overall market regime
    - Standard deviation bands for volatility-adjusted levels
    """
    
    def __init__(self, config: dict):
        """Initialize strategy with configuration."""
        self.config = config
        self.positions = {}
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Rolling VWAP indicators.
        
        Signal logic:
        1. Buy when price crosses above short-term VWAP with medium-term trend up
        2. Sell when price crosses below short-term VWAP or hits upper band
        3. Use position relative to bands for position sizing
        """
        df = df.copy()
        
        # Initialize signal columns
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # Skip if not enough data
        if len(df) < 200:
            return df
            
        # Define VWAP columns
        short_vwap = 'Rolling_VWAP_20'
        medium_vwap = 'Rolling_VWAP_50'
        long_vwap = 'Rolling_VWAP_200'
        
        # Check if required columns exist
        required_cols = [short_vwap, medium_vwap, long_vwap]
        if not all(col in df.columns for col in required_cols):
            print("Missing required VWAP columns")
            return df
            
        # Calculate trend conditions
        df['short_trend_up'] = df['Close'] > df[short_vwap]
        df['medium_trend_up'] = df[medium_vwap] > df[medium_vwap].shift(5)
        df['long_trend_up'] = df[long_vwap] > df[long_vwap].shift(20)
        
        # Entry signals
        entry_conditions = (
            # Price crosses above short-term VWAP
            (df[f'{short_vwap}_Cross_Above'] == True) &
            # Medium-term trend is up
            (df['medium_trend_up'] == True) &
            # Not at extreme levels (not above 2-sigma band)
            (df['Close'] < df[f'{short_vwap}_Upper_2']) &
            # Volume confirmation (above average)
            (df['Volume'] > df['Volume'].rolling(20).mean())
        )
        
        # Exit signals
        exit_conditions = (
            # Price crosses below short-term VWAP
            (df[f'{short_vwap}_Cross_Below'] == True) |
            # Price hits upper 2-sigma band
            (df['Close'] >= df[f'{short_vwap}_Upper_2']) |
            # Price falls below lower 2-sigma band (stop loss)
            (df['Close'] <= df[f'{short_vwap}_Lower_2'])
        )
        
        # Set signals
        df.loc[entry_conditions, 'signal'] = 1
        df.loc[exit_conditions, 'signal'] = -1
        
        # Calculate signal strength based on multiple factors
        for idx in df[entry_conditions].index:
            strength = 0.0
            
            # Factor 1: Position within bands (0-1 scale)
            position = df.loc[idx, f'{short_vwap}_Position']
            if 0.3 <= position <= 0.7:  # Prefer middle of bands
                strength += 0.3
                
            # Factor 2: Alignment of multiple VWAPs
            if df.loc[idx, short_vwap] > df.loc[idx, medium_vwap]:
                strength += 0.2
            if df.loc[idx, medium_vwap] > df.loc[idx, long_vwap]:
                strength += 0.2
                
            # Factor 3: Distance from VWAP (prefer close to VWAP)
            distance = abs(df.loc[idx, f'{short_vwap}_Distance'])
            if distance < 0.5:  # Within 0.5% of VWAP
                strength += 0.3
                
            df.loc[idx, 'signal_strength'] = strength
            
        return df
        
    def backtest_simple(self, df: pd.DataFrame) -> Dict:
        """
        Simple backtest implementation for demonstration.
        
        Returns:
            Dictionary with performance metrics
        """
        df = self.generate_signals(df)
        
        # Initialize tracking variables
        position = 0
        entry_price = 0
        trades = []
        
        for idx in range(1, len(df)):
            current_signal = df.iloc[idx]['signal']
            current_price = df.iloc[idx]['Close']
            
            # Entry
            if current_signal == 1 and position == 0:
                position = 1
                entry_price = current_price
                trades.append({
                    'entry_date': df.index[idx],
                    'entry_price': entry_price,
                    'signal_strength': df.iloc[idx]['signal_strength']
                })
                
            # Exit
            elif current_signal == -1 and position == 1:
                position = 0
                exit_price = current_price
                
                if trades:
                    trades[-1]['exit_date'] = df.index[idx]
                    trades[-1]['exit_price'] = exit_price
                    trades[-1]['return'] = (exit_price - entry_price) / entry_price
                    
        # Calculate metrics
        completed_trades = [t for t in trades if 'return' in t]
        
        if completed_trades:
            returns = [t['return'] for t in completed_trades]
            winning_trades = [t for t in completed_trades if t['return'] > 0]
            
            metrics = {
                'total_trades': len(completed_trades),
                'winning_trades': len(winning_trades),
                'win_rate': len(winning_trades) / len(completed_trades),
                'avg_return': np.mean(returns),
                'total_return': np.prod([1 + r for r in returns]) - 1,
                'max_drawdown': self._calculate_max_drawdown(returns),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns)
            }
        else:
            metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
            
        return metrics
        
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns."""
        if not returns:
            return 0
            
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min())
        
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0
            
        excess_returns = np.array(returns) - risk_free_rate / 252  # Daily risk-free rate
        
        if excess_returns.std() == 0:
            return 0
            
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def example_usage():
    """
    Example of how to use the Rolling VWAP strategy.
    """
    print("Rolling VWAP Strategy Example")
    print("=" * 50)
    
    # Configuration
    config = {
        'short_vwap_period': 20,
        'medium_vwap_period': 50,
        'long_vwap_period': 200,
        'std_devs': [1, 2, 3],
        'volume_threshold': 1.2  # 20% above average
    }
    
    # In practice, you would load data with indicators already calculated
    # For this example, we'll show the expected DataFrame structure
    
    print("\nExpected DataFrame columns after indicator calculation:")
    print("- OHLCV columns: Open, High, Low, Close, Volume")
    print("- Rolling VWAP columns:")
    print("  - Rolling_VWAP_5, Rolling_VWAP_10, Rolling_VWAP_20, etc.")
    print("  - Rolling_VWAP_20_Upper_1, Rolling_VWAP_20_Upper_2, etc.")
    print("  - Rolling_VWAP_20_Lower_1, Rolling_VWAP_20_Lower_2, etc.")
    print("  - Rolling_VWAP_20_Position (0-1 scale)")
    print("  - Rolling_VWAP_20_Distance (% from VWAP)")
    print("  - Rolling_VWAP_20_Cross_Above (boolean)")
    print("  - Rolling_VWAP_20_Cross_Below (boolean)")
    
    print("\nStrategy Signal Logic:")
    print("1. BUY when:")
    print("   - Price crosses above 20-period VWAP")
    print("   - 50-period VWAP is trending up")
    print("   - Price is below upper 2-sigma band")
    print("   - Volume is above 20-period average")
    
    print("\n2. SELL when:")
    print("   - Price crosses below 20-period VWAP")
    print("   - Price hits upper 2-sigma band (profit target)")
    print("   - Price falls below lower 2-sigma band (stop loss)")
    
    print("\n3. Position Sizing:")
    print("   - Based on signal strength (0-1)")
    print("   - Considers VWAP alignment and position within bands")
    
    # Create strategy instance
    strategy = RollingVWAPStrategy(config)
    
    print("\nStrategy initialized successfully!")
    print("\nTo use with real data:")
    print("1. Load data using MultiTimeframeIndicators")
    print("2. Calculate rolling VWAP indicators")
    print("3. Pass DataFrame to strategy.generate_signals()")
    print("4. Use signals for backtesting or live trading")


if __name__ == "__main__":
    example_usage()