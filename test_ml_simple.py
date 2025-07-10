"""
Simple demonstration of ML integration concept without heavy dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Import basic modules
from src.data.download_historical_data import load_cached_data
from src.backtesting.engine import BacktestEngine
from src.strategies.builder import Strategy, StrategyBuilder

# Import feature engineering directly
import importlib.util
spec = importlib.util.spec_from_file_location("feature_engineering", "src/ml/features/feature_engineering.py")
fe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fe_module)
FeatureEngineer = fe_module.FeatureEngineer


def create_simple_ml_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple ML-like signals for demonstration.
    This simulates what ML models would output.
    """
    # Feature engineering
    fe = FeatureEngineer()
    features = fe.engineer_features(data)
    
    # Simulate ML predictions (in reality, these would come from trained models)
    # Simple momentum-based "ML" signal
    features['momentum'] = features['close'].pct_change(20)
    features['volatility'] = features['returns'].rolling(20).std()
    
    # Simulate direction prediction (1 = long, -1 = short, 0 = neutral)
    features['ml_direction'] = 0
    features.loc[features['momentum'] > 0.02, 'ml_direction'] = 1
    features.loc[features['momentum'] < -0.02, 'ml_direction'] = -1
    
    # Simulate confidence score
    features['ml_confidence'] = abs(features['momentum']) * 10
    features['ml_confidence'] = features['ml_confidence'].clip(0, 1)
    
    # Simulate volatility forecast
    features['ml_volatility_forecast'] = features['volatility'] * (1 + np.random.normal(0, 0.1, len(features)))
    
    return features


def run_ml_backtest_demo():
    """Run a demonstration of ML-enhanced backtesting."""
    print("=" * 60)
    print("ML-Enhanced Backtesting Demonstration")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading market data...")
    data = load_cached_data("SPY")
    
    if data is None:
        print("No data found. Please run download_data.py first.")
        return
    
    # Use recent data
    data = data.iloc[-500:]
    data.attrs['symbol'] = 'SPY'
    
    print(f"Loaded {len(data)} days of SPY data")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Generate ML features and signals
    print("\n2. Generating ML features and signals...")
    ml_data = create_simple_ml_signals(data)
    
    print(f"Generated {len([col for col in ml_data.columns if col not in data.columns])} ML features")
    
    # Create ML-based strategy
    print("\n3. Creating ML-based strategy...")
    
    builder = StrategyBuilder("ML Demo Strategy")
    
    # Entry rules based on ML signals
    builder.add_entry_rule("ml_direction == 1 and ml_confidence > 0.7")
    
    # Exit rules
    builder.add_exit_rule("ml_direction == -1 or ml_confidence < 0.3")
    
    # Risk management with ML volatility
    builder.set_risk_management(
        stop_loss=0.02,
        take_profit=0.04,
        max_positions=3
    )
    
    strategy = builder.build()
    
    # Run backtest
    print("\n4. Running backtest...")
    
    engine = BacktestEngine(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        max_positions=3
    )
    
    # Add ML signals to data
    backtest_data = ml_data[['open', 'high', 'low', 'close', 'volume', 
                            'ml_direction', 'ml_confidence', 'ml_volatility_forecast']].copy()
    backtest_data.attrs = data.attrs
    
    results = engine.run(
        data=backtest_data,
        strategy=strategy,
        progress_bar=True
    )
    
    # Display results
    print("\n5. Backtest Results:")
    print("-" * 40)
    
    perf = results['performance']
    print(f"Total Return: {perf['total_return']:.2f}%")
    print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {perf['max_drawdown']:.2%}")
    print(f"Total Trades: {perf['total_trades']}")
    print(f"Win Rate: {perf['win_rate']:.2%}")
    print(f"Profit Factor: {perf['profit_factor']:.2f}")
    
    # Plot results
    print("\n6. Creating visualizations...")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Equity curve
    ax1 = axes[0]
    equity = results['equity_curve']
    equity.plot(ax=ax1, label='ML Strategy', color='blue', linewidth=2)
    ax1.set_title('Equity Curve')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # ML signals
    ax2 = axes[1]
    ml_data['close'].plot(ax=ax2, label='Price', color='black', alpha=0.7)
    
    # Mark buy signals
    buy_signals = ml_data[ml_data['ml_direction'] == 1]
    ax2.scatter(buy_signals.index, buy_signals['close'], 
               color='green', marker='^', s=100, label='Buy Signal', alpha=0.7)
    
    # Mark sell signals
    sell_signals = ml_data[ml_data['ml_direction'] == -1]
    ax2.scatter(sell_signals.index, sell_signals['close'], 
               color='red', marker='v', s=100, label='Sell Signal', alpha=0.7)
    
    ax2.set_title('Price with ML Signals')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ML confidence over time
    ax3 = axes[2]
    ml_data['ml_confidence'].plot(ax=ax3, color='purple', alpha=0.7)
    ax3.axhline(y=0.7, color='green', linestyle='--', label='Entry Threshold')
    ax3.axhline(y=0.3, color='red', linestyle='--', label='Exit Threshold')
    ax3.set_title('ML Model Confidence')
    ax3.set_ylabel('Confidence Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'examples/reports/ml_demo_results.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Show ML feature statistics
    print("\n7. ML Feature Statistics:")
    print("-" * 40)
    
    print(f"Average Confidence: {ml_data['ml_confidence'].mean():.3f}")
    print(f"Signal Distribution:")
    print(f"  Long signals: {(ml_data['ml_direction'] == 1).sum()}")
    print(f"  Short signals: {(ml_data['ml_direction'] == -1).sum()}")
    print(f"  Neutral: {(ml_data['ml_direction'] == 0).sum()}")
    
    # Feature importance (simulated)
    print("\nTop Features (Simulated Importance):")
    features = ['momentum', 'rsi', 'bb_position', 'volume_ratio', 'atr_percent']
    importances = [0.25, 0.20, 0.15, 0.12, 0.10]
    
    for feat, imp in zip(features, importances):
        print(f"  {feat}: {imp:.3f}")
    
    print("\n" + "=" * 60)
    print("ML Integration Demonstration Complete!")
    print("=" * 60)
    print("\nNOTE: This is a simplified demonstration.")
    print("Full ML integration includes:")
    print("- XGBoost for direction prediction")
    print("- LSTM for volatility forecasting")
    print("- Market regime detection")
    print("- Ensemble models")
    print("- Walk-forward analysis")
    print("- Automatic retraining")
    
    plt.show()
    
    return results


if __name__ == "__main__":
    results = run_ml_backtest_demo()