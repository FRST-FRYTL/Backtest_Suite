"""
Comprehensive example demonstrating ML-enhanced backtesting with real market data.

This example shows:
1. Loading real market data
2. Setting up ML strategy with ensemble models
3. Running walk-forward ML backtest
4. Analyzing ML-specific metrics
5. Visualizing results
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from src.backtesting.ml_integration import MLBacktestEngine, MLBacktestConfig
from src.strategies.ml_strategy import MLStrategy
from src.data.download_historical_data import download_data, load_cached_data
from src.visualization.charts import create_performance_dashboard


def setup_ml_strategy():
    """Configure ML trading strategy."""
    ml_strategy = MLStrategy(
        name="Ensemble ML Strategy",
        use_ensemble=True,
        direction_threshold=0.65,
        confidence_threshold=0.7,
        regime_filter=True,
        volatility_scaling=True,
        risk_per_trade=0.02,
        feature_lookback=50,
        retrain_frequency=63  # Quarterly retraining
    )
    
    # Configure position sizing
    ml_strategy.position_sizing.method = "volatility"
    ml_strategy.position_sizing.max_position = 0.25  # Max 25% per position
    
    # Configure risk management
    ml_strategy.risk_management.stop_loss = 0.02  # 2% stop loss
    ml_strategy.risk_management.stop_loss_type = "atr"
    ml_strategy.risk_management.stop_loss = 2.0  # 2 ATR stop
    ml_strategy.risk_management.take_profit = 4.0  # 4 ATR profit target
    ml_strategy.risk_management.take_profit_type = "atr"
    ml_strategy.risk_management.max_positions = 3
    ml_strategy.risk_management.trailing_stop = 0.015  # 1.5% trailing stop
    
    return ml_strategy


def run_ml_backtest_example():
    """Run complete ML backtest example."""
    print("=" * 80)
    print("ML-Enhanced Backtesting Example")
    print("=" * 80)
    
    # Step 1: Load real market data
    print("\n1. Loading market data...")
    
    # Try to load cached data first
    symbol = "SPY"
    data = load_cached_data(symbol)
    
    if data is None:
        print(f"No cached data found. Downloading {symbol} data...")
        # Download if not cached
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)  # 5 years
        
        data = download_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
    
    print(f"Loaded {len(data)} days of {symbol} data")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Step 2: Configure ML backtest
    print("\n2. Configuring ML backtest...")
    
    ml_config = MLBacktestConfig(
        use_walk_forward=True,
        walk_forward_window=252,  # 1 year training window
        retrain_frequency=63,     # Retrain quarterly
        validation_split=0.2,
        min_training_samples=500,
        feature_selection=True,
        feature_importance_threshold=0.01,
        ensemble_voting='soft',
        risk_parity=True
    )
    
    # Step 3: Initialize ML backtest engine
    print("\n3. Initializing ML backtest engine...")
    
    ml_engine = MLBacktestEngine(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        max_positions=3,
        ml_config=ml_config
    )
    
    # Step 4: Setup ML strategy
    print("\n4. Setting up ML strategy...")
    ml_strategy = setup_ml_strategy()
    
    # Step 5: Run ML backtest
    print("\n5. Running ML backtest with walk-forward analysis...")
    print("This may take a few minutes as models are trained on each window...\n")
    
    # Run backtest on last 3 years (to have 2 years for initial training)
    backtest_start = data.index[-756]  # 3 years
    
    results = ml_engine.run_ml_backtest(
        data=data,
        ml_strategy=ml_strategy,
        start_date=backtest_start,
        progress_bar=True
    )
    
    # Step 6: Analyze results
    print("\n6. Analyzing results...")
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    
    # Performance metrics
    perf = results['performance']
    print(f"\nPerformance Metrics:")
    print(f"Total Return: {perf['total_return']:.2f}%")
    print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {perf['max_drawdown']:.2%}")
    print(f"Total Trades: {perf['total_trades']}")
    print(f"Walk-Forward Windows: {perf['walk_forward_windows']}")
    
    # ML metrics
    ml_metrics = results['ml_metrics']
    print(f"\nML Metrics:")
    print(f"Total Predictions: {ml_metrics['total_predictions']}")
    print(f"Average Confidence: {ml_metrics['avg_confidence']:.3f}")
    print(f"Average Probability: {ml_metrics['avg_probability']:.3f}")
    print(f"High Confidence Ratio: {ml_metrics.get('high_confidence_ratio', 0):.2%}")
    
    # Direction distribution
    print(f"\nDirection Distribution:")
    for direction, count in ml_metrics['direction_distribution'].items():
        direction_name = "Long" if direction == 1 else "Short" if direction == -1 else "Neutral"
        print(f"  {direction_name}: {count}")
    
    # Regime analysis
    regime_analysis = results['regime_analysis']
    print(f"\nMarket Regime Distribution:")
    for regime, count in regime_analysis['regime_counts'].items():
        print(f"  {regime}: {count}")
    
    # Feature importance
    feature_importance = results['feature_importance']
    print(f"\nTop 10 Most Important Features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    # Step 7: Visualize results
    print("\n7. Creating visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('ML-Enhanced Backtest Results', fontsize=16)
    
    # 1. Equity curve
    ax1 = axes[0, 0]
    equity_curve = results['equity_curve']
    equity_curve.plot(ax=ax1, label='ML Strategy', color='blue', linewidth=2)
    
    # Add benchmark (buy and hold)
    benchmark_returns = data.loc[equity_curve.index, 'close'].pct_change().fillna(0)
    benchmark_equity = 100000 * (1 + benchmark_returns).cumprod()
    benchmark_equity.plot(ax=ax1, label='Buy & Hold', color='gray', alpha=0.7, linestyle='--')
    
    ax1.set_title('Equity Curve Comparison')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = axes[0, 1]
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100
    drawdown.plot(ax=ax2, color='red', linewidth=2)
    ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    ax2.set_title('Drawdown')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    
    # 3. ML Confidence over time
    ax3 = axes[1, 0]
    if ml_engine.ml_predictions:
        pred_df = pd.DataFrame(ml_engine.ml_predictions)
        pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
        pred_df.set_index('timestamp', inplace=True)
        
        # Plot confidence over time
        pred_df['confidence'].plot(ax=ax3, color='green', alpha=0.6, label='Confidence')
        pred_df['confidence'].rolling(20).mean().plot(ax=ax3, color='darkgreen', linewidth=2, label='20-day MA')
        ax3.axhline(y=ml_strategy.confidence_threshold, color='red', linestyle='--', label='Threshold')
        ax3.set_title('ML Model Confidence Over Time')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Confidence')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Regime distribution
    ax4 = axes[1, 1]
    if regime_analysis['regime_counts']:
        regimes = list(regime_analysis['regime_counts'].keys())
        counts = list(regime_analysis['regime_counts'].values())
        
        colors = ['green', 'yellow', 'red', 'blue', 'purple'][:len(regimes)]
        ax4.bar(regimes, counts, color=colors, alpha=0.7)
        ax4.set_title('Market Regime Distribution')
        ax4.set_xlabel('Market Regime')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Feature importance
    ax5 = axes[2, 0]
    if feature_importance:
        top_features = list(feature_importance.items())[:15]
        features = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        
        y_pos = np.arange(len(features))
        ax5.barh(y_pos, importances, color='teal', alpha=0.7)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(features)
        ax5.set_title('Top 15 Feature Importances')
        ax5.set_xlabel('Importance Score')
        ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. Walk-forward performance
    ax6 = axes[2, 1]
    if 'walk_forward_details' in results:
        oos_periods = results['walk_forward_details']['out_of_sample_periods']
        if oos_periods:
            window_returns = [p['performance']['total_return'] for p in oos_periods]
            window_labels = [f"W{i+1}" for i in range(len(window_returns))]
            
            colors = ['green' if r > 0 else 'red' for r in window_returns]
            ax6.bar(window_labels, window_returns, color=colors, alpha=0.7)
            ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax6.set_title('Walk-Forward Window Returns')
            ax6.set_xlabel('Window')
            ax6.set_ylabel('Return (%)')
            ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'examples/reports/ml_backtest_results.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualizations saved to: {output_path}")
    
    # Step 8: Generate detailed ML report
    print("\n8. Generating detailed ML report...")
    
    # Create ML performance summary
    ml_summary = {
        'Strategy': ml_strategy.name,
        'Ensemble Model': ml_strategy.use_ensemble,
        'Direction Threshold': ml_strategy.direction_threshold,
        'Confidence Threshold': ml_strategy.confidence_threshold,
        'Initial Capital': ml_engine.initial_capital,
        'Final Capital': equity_curve.iloc[-1],
        'Total Return (%)': perf['total_return'],
        'Sharpe Ratio': perf['sharpe_ratio'],
        'Max Drawdown (%)': perf['max_drawdown'] * 100,
        'Total Trades': perf['total_trades'],
        'Walk-Forward Windows': perf['walk_forward_windows'],
        'Avg Confidence': ml_metrics['avg_confidence'],
        'High Confidence Ratio (%)': ml_metrics.get('high_confidence_ratio', 0) * 100,
        'Prediction Frequency (%)': ml_metrics.get('prediction_frequency', 0) * 100
    }
    
    # Save summary to file
    summary_df = pd.DataFrame([ml_summary]).T
    summary_df.columns = ['Value']
    summary_path = 'examples/reports/ml_backtest_summary.csv'
    summary_df.to_csv(summary_path)
    print(f"Summary saved to: {summary_path}")
    
    # Save detailed results
    detailed_results = {
        'equity_curve': equity_curve,
        'ml_predictions': pd.DataFrame(ml_engine.ml_predictions) if ml_engine.ml_predictions else pd.DataFrame(),
        'feature_importance': pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance']),
        'regime_history': pd.DataFrame(ml_engine.regime_history) if ml_engine.regime_history else pd.DataFrame()
    }
    
    # Save to Excel with multiple sheets
    excel_path = 'examples/reports/ml_backtest_detailed.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for sheet_name, df in detailed_results.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=True)
    
    print(f"Detailed results saved to: {excel_path}")
    
    print("\n" + "=" * 80)
    print("ML Backtest Example Complete!")
    print("=" * 80)
    
    return results


def analyze_ml_predictions(results):
    """Perform additional analysis on ML predictions."""
    print("\n" + "=" * 50)
    print("ADVANCED ML PREDICTION ANALYSIS")
    print("=" * 50)
    
    if 'ml_metrics' not in results:
        print("No ML predictions found.")
        return
    
    # Analyze prediction patterns
    pred_analysis = results.get('prediction_analysis', {})
    
    if 'hourly_patterns' in pred_analysis:
        print("\nPrediction Patterns by Hour:")
        hourly = pred_analysis['hourly_patterns']
        if 'confidence' in hourly:
            for hour, conf in sorted(hourly['confidence'].items()):
                print(f"  Hour {hour:02d}: Confidence={conf:.3f}")
    
    if 'daily_patterns' in pred_analysis:
        print("\nPrediction Patterns by Day of Week:")
        daily = pred_analysis['daily_patterns']
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        if 'confidence' in daily:
            for day_num, conf in sorted(daily['confidence'].items()):
                if day_num < len(days):
                    print(f"  {days[day_num]}: Confidence={conf:.3f}")
    
    # Analyze regime transitions
    regime_analysis = results.get('regime_analysis', {})
    if 'regime_transitions' in regime_analysis:
        print("\nRegime Transition Matrix:")
        transitions = regime_analysis['regime_transitions']
        for from_regime, to_regimes in transitions.items():
            print(f"\n  From {from_regime}:")
            for to_regime, count in to_regimes.items():
                print(f"    To {to_regime}: {count} times")


if __name__ == "__main__":
    # Run the ML backtest example
    results = run_ml_backtest_example()
    
    # Perform additional analysis
    analyze_ml_predictions(results)
    
    # Show plots
    plt.show()