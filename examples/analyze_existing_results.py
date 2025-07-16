"""
Analyze Existing Backtest Results

This script analyzes existing backtest results and generates performance reports.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.timeframe_performance_analyzer import (
    TimeframePerformanceAnalyzer,
    TimeframeResult,
    PerformanceMetrics as AnalysisMetrics
)


def load_mock_results() -> list:
    """Load or create mock results for analysis demonstration."""
    
    # Check if we have real results
    results_files = list(Path("backtest_results").glob("*.json"))
    if results_files:
        # Load the most recent results file
        latest_file = max(results_files, key=lambda f: f.stat().st_mtime)
        print(f"Loading results from: {latest_file}")
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    # Otherwise, create mock results for demonstration
    print("No existing results found. Creating mock results for demonstration...")
    
    timeframes = ['1D', '1W', '1M']
    symbols = ['SPY']
    
    # Parameter combinations
    param_sets = [
        {'rsi_period': 14, 'bb_period': 20, 'stop_loss_atr': 2.0},
        {'rsi_period': 10, 'bb_period': 15, 'stop_loss_atr': 1.5},
        {'rsi_period': 20, 'bb_period': 25, 'stop_loss_atr': 2.5},
        {'rsi_period': 14, 'bb_period': 20, 'stop_loss_atr': 2.0, 'use_supertrend': True},
        {'rsi_period': 14, 'bb_period': 30, 'stop_loss_atr': 3.0},
    ]
    
    results = []
    np.random.seed(42)  # For reproducibility
    
    for timeframe in timeframes:
        for symbol in symbols:
            for params in param_sets:
                # Generate realistic mock metrics based on timeframe
                base_return = np.random.uniform(0.05, 0.25)
                base_sharpe = np.random.uniform(0.5, 2.0)
                
                # Adjust metrics based on timeframe
                if timeframe == '1D':
                    # Daily - more trades, moderate performance
                    return_mult = 1.0
                    sharpe_mult = 1.0
                    trades = np.random.randint(50, 200)
                elif timeframe == '1W':
                    # Weekly - fewer trades, potentially better risk-adjusted returns
                    return_mult = 0.9
                    sharpe_mult = 1.1
                    trades = np.random.randint(20, 60)
                else:  # Monthly
                    # Monthly - very few trades, highest risk-adjusted returns
                    return_mult = 0.8
                    sharpe_mult = 1.2
                    trades = np.random.randint(5, 20)
                
                # Add parameter influence
                if params.get('use_supertrend', False):
                    sharpe_mult *= 1.15
                    return_mult *= 1.1
                
                # Generate metrics
                total_return = base_return * return_mult
                sharpe_ratio = base_sharpe * sharpe_mult
                
                # Ensure reasonable relationships between metrics
                volatility = total_return / (sharpe_ratio * np.sqrt(252))
                max_drawdown = -np.random.uniform(0.05, 0.20) * (2 - sharpe_ratio/2)
                win_rate = 0.4 + (sharpe_ratio / 10) + np.random.uniform(-0.1, 0.1)
                win_rate = np.clip(win_rate, 0.3, 0.7)
                
                result = {
                    'timeframe': timeframe,
                    'symbol': symbol,
                    'parameters': params,
                    'metrics': {
                        'total_return': total_return,
                        'annualized_return': total_return,
                        'sharpe_ratio': sharpe_ratio,
                        'sortino_ratio': sharpe_ratio * 1.2,
                        'max_drawdown': max_drawdown,
                        'calmar_ratio': total_return / abs(max_drawdown),
                        'win_rate': win_rate,
                        'profit_factor': 1 + sharpe_ratio / 2,
                        'volatility': volatility,
                        'var_95': -volatility * 1.645,
                        'cvar_95': -volatility * 2.06,
                        'total_trades': trades,
                        'avg_trade_duration': None,
                        'beta': 0.8 + np.random.uniform(-0.2, 0.2),
                        'alpha': total_return * 0.1
                    },
                    'start_date': '2020-01-01',
                    'end_date': '2024-01-01'
                }
                
                results.append(result)
    
    return results


def main():
    """Main analysis function."""
    print("=" * 80)
    print("Multi-Timeframe Performance Analysis")
    print("=" * 80)
    
    # Create directories
    Path("backtest_results").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    
    # Load or create results
    results = load_mock_results()
    print(f"\nLoaded {len(results)} backtest results")
    
    # Initialize analyzer
    analyzer = TimeframePerformanceAnalyzer()
    
    # Convert results to TimeframeResult objects
    for result in results:
        try:
            metrics = AnalysisMetrics(
                total_return=result['metrics']['total_return'] * 100,  # Convert to percentage
                annualized_return=result['metrics']['annualized_return'],
                sharpe_ratio=result['metrics']['sharpe_ratio'],
                sortino_ratio=result['metrics']['sortino_ratio'],
                max_drawdown=result['metrics']['max_drawdown'],
                calmar_ratio=result['metrics']['calmar_ratio'],
                win_rate=result['metrics']['win_rate'],
                profit_factor=result['metrics']['profit_factor'],
                volatility=result['metrics']['volatility'],
                var_95=result['metrics']['var_95'],
                cvar_95=result['metrics']['cvar_95'],
                total_trades=result['metrics']['total_trades'],
                avg_trade_duration=result['metrics'].get('avg_trade_duration'),
                beta=result['metrics'].get('beta'),
                alpha=result['metrics'].get('alpha')
            )
            
            tf_result = TimeframeResult(
                timeframe=result['timeframe'],
                symbol=result['symbol'],
                parameters=result['parameters'],
                metrics=metrics,
                start_date=result['start_date'],
                end_date=result['end_date']
            )
            
            analyzer.results.append(tf_result)
            
        except Exception as e:
            print(f"Error processing result: {str(e)}")
    
    print(f"\nSuccessfully processed {len(analyzer.results)} results")
    
    # Run analyses
    print("\nRunning analyses...")
    timeframe_analysis = analyzer.analyze_by_timeframe()
    parameter_sensitivity = analyzer.analyze_parameter_sensitivity()
    robust_configs = analyzer.find_robust_configurations(min_sharpe=1.0, max_drawdown=-0.25)
    
    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    print("\nPerformance by Timeframe:")
    print("-" * 60)
    print(f"{'Timeframe':<10} {'Configs':<8} {'Avg Sharpe':<12} {'Avg Return':<12} {'Best Sharpe':<12}")
    print("-" * 60)
    
    for tf, analysis in sorted(timeframe_analysis.items()):
        print(f"{tf:<10} {analysis['count']:<8} "
              f"{analysis['avg_sharpe']:<12.3f} {analysis['avg_return']:<12.1f}% "
              f"{analysis['best_sharpe']:<12.3f}")
    
    print("\n\nTop 10 Most Robust Configurations:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Avg Sharpe':<12} {'Avg Return':<12} {'Worst DD':<12} {'Timeframes':<12} {'Parameters'}")
    print("-" * 80)
    
    for i, config in enumerate(robust_configs[:10]):
        # Handle both dict and string parameters
        if isinstance(config['parameters'], dict):
            params_str = ', '.join([f"{k}={v}" for k, v in sorted(config['parameters'].items())])
        else:
            params_str = str(config['parameters'])
        print(f"{i+1:<5} {config['avg_sharpe']:<12.3f} "
              f"{config['avg_return']:<12.1f}% {config['worst_drawdown']:<12.1%} "
              f"{config['timeframe_count']:<12} {params_str}")
    
    print("\n\nParameter Sensitivity (Correlation with Sharpe Ratio):")
    print("-" * 60)
    
    for param, correlations in parameter_sensitivity.items():
        param_clean = param.replace('param_', '')
        sharpe_corr = correlations.get('sharpe_ratio', 0)
        if abs(sharpe_corr) > 0.1:  # Only show significant correlations
            print(f"{param_clean:<20}: {sharpe_corr:>6.3f} {'(positive impact)' if sharpe_corr > 0 else '(negative impact)'}")
    
    # Generate HTML report
    print("\n\nGenerating HTML report...")
    report_path = Path("reports/spx_timeframe_analysis.html")
    analyzer.generate_html_report(report_path)
    
    print(f"\nAnalysis complete! Report saved to: {report_path}")
    print("\nKey Insights:")
    print("-" * 40)
    
    # Generate insights
    insights = []
    
    # Best timeframe insight
    best_tf = max(timeframe_analysis.items(), key=lambda x: x[1]['avg_sharpe'])
    insights.append(f"• {best_tf[0]} timeframe shows the best average Sharpe ratio ({best_tf[1]['avg_sharpe']:.3f})")
    
    # Consistency insight
    if robust_configs:
        most_robust = robust_configs[0]
        insights.append(f"• Most robust configuration achieves {most_robust['avg_sharpe']:.3f} Sharpe across {most_robust['timeframe_count']} timeframes")
    
    # Parameter insights
    high_impact_params = []
    for param, corrs in parameter_sensitivity.items():
        if abs(corrs.get('sharpe_ratio', 0)) > 0.2:
            high_impact_params.append(param.replace('param_', ''))
    
    if high_impact_params:
        insights.append(f"• High-impact parameters: {', '.join(high_impact_params)}")
    
    # Risk management insight
    avg_drawdowns = [analysis['avg_drawdown'] for analysis in timeframe_analysis.values()]
    if avg_drawdowns:
        insights.append(f"• Average maximum drawdown across timeframes: {np.mean(avg_drawdowns):.1%}")
    
    for insight in insights:
        print(insight)
    
    print("\n" + "=" * 80)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()