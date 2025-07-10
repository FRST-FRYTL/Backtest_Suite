"""
Generate backtest results for various strategies
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.backtesting.engine import BacktestEngine
from src.strategies.base import BaseStrategy, Signal
from src.strategies.technical_strategy import TechnicalStrategy
from src.data.data_loader import DataLoader
from src.risk.risk_manager import RiskManager, RiskConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Create reports directory
os.makedirs('reports/backtest', exist_ok=True)
os.makedirs('reports/summary', exist_ok=True)

# Simple Buy and Hold Strategy
class BuyAndHoldStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="BuyAndHold")
        self.bought = False
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        if not self.bought:
            signals.iloc[0] = 1  # Buy on first day
            self.bought = True
        return signals

# Simple Moving Average Crossover Strategy
class SMAStrategy(BaseStrategy):
    def __init__(self, short_window=20, long_window=50):
        super().__init__(name=f"SMA_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[short_ma > long_ma] = 1
        signals[short_ma < long_ma] = -1
        
        return signals

# RSI Strategy
class RSIStrategy(BaseStrategy):
    def __init__(self, period=14, oversold=30, overbought=70):
        super().__init__(name=f"RSI_{period}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=data.index)
        signals[rsi < self.oversold] = 1  # Buy signal
        signals[rsi > self.overbought] = -1  # Sell signal
        
        return signals

# Mean Reversion Strategy
class MeanReversionStrategy(BaseStrategy):
    def __init__(self, lookback=20, z_threshold=2):
        super().__init__(name=f"MeanReversion_{lookback}")
        self.lookback = lookback
        self.z_threshold = z_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate z-score
        returns = data['close'].pct_change()
        mean = returns.rolling(window=self.lookback).mean()
        std = returns.rolling(window=self.lookback).std()
        z_score = (returns - mean) / std
        
        signals = pd.Series(0, index=data.index)
        signals[z_score < -self.z_threshold] = 1  # Buy when oversold
        signals[z_score > self.z_threshold] = -1  # Sell when overbought
        
        return signals

def generate_synthetic_data(symbol='TEST', days=252):
    """Generate synthetic market data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, size=days)  # Daily returns
    price = 100 * np.exp(np.cumsum(returns))
    
    # Add some volume
    volume = np.random.randint(1000000, 5000000, size=days)
    
    # Create OHLC data
    data = pd.DataFrame({
        'open': price * (1 + np.random.uniform(-0.005, 0.005, size=days)),
        'high': price * (1 + np.random.uniform(0, 0.01, size=days)),
        'low': price * (1 + np.random.uniform(-0.01, 0, size=days)),
        'close': price,
        'volume': volume
    }, index=dates)
    
    # Ensure high >= close and low <= close
    data['high'] = data[['high', 'close']].max(axis=1)
    data['low'] = data[['low', 'close']].min(axis=1)
    
    return data

def run_backtest(strategy, data, initial_capital=100000):
    """Run backtest for a single strategy"""
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=0.001,  # 0.1% commission
        slippage=0.001     # 0.1% slippage
    )
    
    # Run backtest
    results = engine.run(strategy, data)
    
    return results

def main():
    # Generate synthetic data
    logger.info("Generating synthetic market data...")
    data = generate_synthetic_data(days=504)  # 2 years of data
    
    # Initialize strategies
    strategies = [
        BuyAndHoldStrategy(),
        SMAStrategy(short_window=10, long_window=30),
        SMAStrategy(short_window=20, long_window=50),
        RSIStrategy(period=14),
        MeanReversionStrategy(lookback=20, z_threshold=2)
    ]
    
    # Run backtests
    all_results = {}
    for strategy in strategies:
        logger.info(f"Running backtest for {strategy.name}...")
        try:
            results = run_backtest(strategy, data)
            all_results[strategy.name] = results
            
            # Save individual results
            results_file = f"reports/backtest/{strategy.name}_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'strategy': strategy.name,
                    'total_return': float(results.get('total_return', 0)),
                    'sharpe_ratio': float(results.get('sharpe_ratio', 0)),
                    'max_drawdown': float(results.get('max_drawdown', 0)),
                    'win_rate': float(results.get('win_rate', 0)),
                    'total_trades': int(results.get('total_trades', 0)),
                    'final_capital': float(results.get('final_capital', 100000))
                }, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Error running backtest for {strategy.name}: {e}")
            all_results[strategy.name] = {
                'error': str(e),
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0
            }
    
    # Save summary
    summary_file = "reports/summary/backtest_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Summary saved to {summary_file}")
    logger.info("Backtest generation complete!")
    
    return all_results

if __name__ == "__main__":
    results = main()
    
    # Print summary
    print("\n=== Backtest Summary ===")
    for strategy, metrics in results.items():
        print(f"\n{strategy}:")
        if 'error' not in metrics:
            print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"  Total Trades: {metrics.get('total_trades', 0)}")
        else:
            print(f"  Error: {metrics['error']}")