"""
Simple test script to verify ML integration with real data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.data.download_historical_data import load_cached_data
from src.strategies.ml_strategy import MLStrategy
from src.backtesting.engine import BacktestEngine


def test_ml_integration():
    """Test ML integration with real data."""
    print("=" * 60)
    print("Testing ML Integration with Real Market Data")
    print("=" * 60)
    
    # Step 1: Load real data
    print("\n1. Loading market data...")
    
    # Try to load SPY data
    data = load_cached_data("SPY")
    
    if data is None:
        print("No data found. Please run download_data.py first.")
        return
    
    print(f"Loaded {len(data)} days of SPY data")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Add symbol attribute
    data.attrs['symbol'] = 'SPY'
    
    # Step 2: Create ML strategy (without ensemble to avoid model dependencies)
    print("\n2. Creating ML strategy...")
    
    ml_strategy = MLStrategy(
        name="Simple ML Strategy",
        use_ensemble=False,  # Use individual models
        direction_threshold=0.6,
        confidence_threshold=0.65,
        regime_filter=False,  # Disable regime filter for simplicity
        volatility_scaling=True,
        risk_per_trade=0.02,
        feature_lookback=30,  # Reduced lookback
        retrain_frequency=252
    )
    
    # Configure simple position sizing
    ml_strategy.position_sizing.method = "fixed"
    ml_strategy.position_sizing.size = 100  # 100 shares
    
    # Configure risk management
    ml_strategy.risk_management.stop_loss = 0.02
    ml_strategy.risk_management.take_profit = 0.04
    ml_strategy.risk_management.max_positions = 1
    
    # Step 3: Prepare test data
    print("\n3. Preparing test data...")
    
    # Use last 500 days for testing
    test_data = data.iloc[-500:]
    
    # Step 4: Test feature preparation
    print("\n4. Testing feature preparation...")
    
    try:
        features = ml_strategy.prepare_features(test_data)
        print(f"Generated {len(features.columns)} features")
        print(f"Sample features: {list(features.columns[:10])}")
    except Exception as e:
        print(f"Error preparing features: {e}")
        return
    
    # Step 5: Run simple backtest
    print("\n5. Running simple backtest...")
    
    engine = BacktestEngine(
        initial_capital=10000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        max_positions=1
    )
    
    try:
        # Run backtest on last 100 days
        backtest_data = test_data.iloc[-100:]
        results = engine.run(
            data=backtest_data,
            strategy=ml_strategy,
            progress_bar=False
        )
        
        print("\nBacktest Results:")
        print(f"Total Return: {results['performance']['total_return']:.2f}%")
        print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['performance']['max_drawdown']:.2%}")
        print(f"Total Trades: {results['performance']['total_trades']}")
        
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("ML Integration Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_ml_integration()