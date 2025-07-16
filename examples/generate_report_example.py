#!/usr/bin/env python3
"""
Example script demonstrating how to use the StandardReportGenerator.

This example shows various ways to generate reports from backtest results,
including customization options and different data formats.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.reporting import StandardReportGenerator, ReportConfig
from src.reporting.report_config import ReportSection


def generate_sample_backtest_results():
    """Generate comprehensive sample backtest results for demonstration."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate date range
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Generate realistic equity curve with trends
    trend = np.linspace(0, 0.3, len(dates))  # Upward trend
    noise = np.random.normal(0, 0.015, len(dates))
    seasonal = 0.05 * np.sin(2 * np.pi * np.arange(len(dates)) / 252)
    
    returns = 0.0003 + trend/len(dates) + noise + seasonal/len(dates)
    equity_curve = pd.Series(10000 * (1 + returns).cumprod(), index=dates)
    
    # Generate trades with realistic patterns
    n_trades = 150
    trade_indices = np.sort(np.random.choice(len(dates)-10, n_trades, replace=False))
    
    trades = []
    for i, idx in enumerate(trade_indices):
        entry_date = dates[idx]
        hold_period = np.random.randint(1, 15)
        exit_date = dates[min(idx + hold_period, len(dates)-1)]
        
        # Generate correlated entry/exit prices
        entry_price = 100 + np.random.normal(0, 10)
        price_change = np.random.normal(0.002, 0.02)  # Slight positive bias
        exit_price = entry_price * (1 + price_change)
        
        # Determine trade direction
        side = np.random.choice(['long', 'short'], p=[0.6, 0.4])
        
        # Calculate P&L
        if side == 'long':
            pnl = (exit_price - entry_price) * 100  # 100 shares
        else:
            pnl = (entry_price - exit_price) * 100
            
        trades.append({
            'trade_id': i + 1,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': 100,
            'pnl': pnl,
            'commission': 2.0,
            'net_pnl': pnl - 2.0
        })
    
    trades_df = pd.DataFrame(trades)
    
    # Calculate comprehensive metrics
    daily_returns = equity_curve.pct_change().dropna()
    
    # Basic metrics
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    annual_return = (1 + total_return) ** (252/len(daily_returns)) - 1
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - 0.02) / volatility  # 2% risk-free rate
    
    # Drawdown calculation
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sortino ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - 0.02) / downside_std if downside_std > 0 else 0
    
    # Trade statistics
    winning_trades = trades_df[trades_df['net_pnl'] > 0]
    losing_trades = trades_df[trades_df['net_pnl'] < 0]
    
    metrics = {
        # Returns
        'total_return': total_return,
        'annual_return': annual_return,
        'monthly_return': annual_return / 12,
        'daily_return': daily_returns.mean(),
        
        # Risk metrics
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': annual_return / abs(max_drawdown) if max_drawdown != 0 else 0,
        'downside_deviation': downside_std,
        
        # Trade metrics
        'total_trades': len(trades_df),
        'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
        'avg_win': winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0,
        'avg_loss': losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0,
        'profit_factor': (winning_trades['net_pnl'].sum() / 
                         abs(losing_trades['net_pnl'].sum()) 
                         if len(losing_trades) > 0 and losing_trades['net_pnl'].sum() != 0 else 0),
        'win_loss_ratio': (abs(winning_trades['net_pnl'].mean() / losing_trades['net_pnl'].mean()) 
                          if len(losing_trades) > 0 and losing_trades['net_pnl'].mean() != 0 else 0),
        
        # Additional metrics
        'avg_trade_duration': (trades_df['exit_date'] - trades_df['entry_date']).dt.days.mean(),
        'expectancy': trades_df['net_pnl'].mean(),
        'total_commission': trades_df['commission'].sum(),
        'net_profit': trades_df['net_pnl'].sum()
    }
    
    # Generate optimization results
    optimization_results = []
    for atr_length in range(10, 21, 2):
        for factor in np.arange(1.5, 4.1, 0.5):
            for signal_strength in range(3, 7):
                # Generate correlated metrics
                base_sharpe = np.random.uniform(0.5, 2.5)
                opt_result = {
                    'atr_length': atr_length,
                    'factor': factor,
                    'signal_strength': signal_strength,
                    'sharpe_ratio': base_sharpe,
                    'total_return': base_sharpe * 0.15 + np.random.normal(0, 0.05),
                    'max_drawdown': -0.1 - (2.5 - base_sharpe) * 0.1 + np.random.normal(0, 0.02),
                    'win_rate': 0.4 + base_sharpe * 0.1 + np.random.normal(0, 0.05),
                    'profit_factor': 0.8 + base_sharpe * 0.4 + np.random.normal(0, 0.1),
                    'total_trades': np.random.randint(50, 200)
                }
                optimization_results.append(opt_result)
    
    optimization_df = pd.DataFrame(optimization_results)
    
    return {
        'equity_curve': equity_curve,
        'trades': trades_df,
        'metrics': metrics,
        'optimization_results': optimization_df,
        'config': {
            'strategy_name': 'Advanced Momentum Strategy',
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'initial_capital': 10000,
            'commission': 2.0,
            'data_source': 'Yahoo Finance',
            'timeframe': 'Daily',
            'universe': 'S&P 500 Components'
        },
        'additional_info': {
            'market_regime': 'Bull Market',
            'correlation_sp500': 0.75,
            'beta': 1.2,
            'alpha': annual_return - 0.12 * 1.2  # CAPM alpha
        }
    }


def example_basic_report():
    """Generate a basic report with default settings."""
    print("=== Example 1: Basic Report Generation ===\n")
    
    # Generate sample data
    backtest_results = generate_sample_backtest_results()
    
    # Create report generator with default config
    generator = StandardReportGenerator()
    
    # Generate reports in all formats
    output_paths = generator.generate_report(
        backtest_results=backtest_results,
        strategy_name="Advanced Momentum Strategy",
        output_dir="reports/examples/basic",
        formats=['markdown', 'html', 'json']
    )
    
    print("Basic report generated successfully!")
    for format_type, path in output_paths.items():
        print(f"  - {format_type}: {path}")


def example_custom_sections():
    """Generate a report with custom sections only."""
    print("\n=== Example 2: Custom Sections Report ===\n")
    
    # Generate sample data
    backtest_results = generate_sample_backtest_results()
    
    # Create custom config
    config = ReportConfig()
    
    # Disable all sections first
    for section in ReportSection:
        config.disable_section(section)
    
    # Enable only specific sections
    config.enable_section(ReportSection.EXECUTIVE_SUMMARY)
    config.enable_section(ReportSection.PERFORMANCE_METRICS)
    config.enable_section(ReportSection.RISK_ANALYSIS)
    config.enable_section(ReportSection.RECOMMENDATIONS)
    
    # Create generator with custom config
    generator = StandardReportGenerator(config)
    
    # Generate report
    output_paths = generator.generate_report(
        backtest_results=backtest_results,
        strategy_name="Focused Analysis Strategy",
        output_dir="reports/examples/custom_sections",
        formats=['markdown', 'html']
    )
    
    print("Custom sections report generated!")
    for format_type, path in output_paths.items():
        print(f"  - {format_type}: {path}")


def example_styled_report():
    """Generate a report with custom styling."""
    print("\n=== Example 3: Styled Report ===\n")
    
    # Generate sample data
    backtest_results = generate_sample_backtest_results()
    
    # Create config with custom styling
    config = ReportConfig()
    
    # Customize style
    config.customize_style(
        primary_color='#FF6B6B',
        secondary_color='#4ECDC4',
        success_color='#45B7D1',
        warning_color='#FFA07A',
        danger_color='#DC143C',
        chart_template='plotly_dark',
        font_family='Helvetica, Arial, sans-serif',
        company_name='Quantum Trading Systems'
    )
    
    # Customize thresholds
    config.set_threshold('sharpe_ratio', 
                        excellent=2.5, 
                        good=1.8, 
                        acceptable=1.2, 
                        poor=0.8)
    
    # Create generator
    generator = StandardReportGenerator(config)
    
    # Generate report
    output_paths = generator.generate_report(
        backtest_results=backtest_results,
        strategy_name="Quantum Momentum Pro",
        output_dir="reports/examples/styled",
        formats=['html']  # HTML best shows styling
    )
    
    print("Styled report generated!")
    for format_type, path in output_paths.items():
        print(f"  - {format_type}: {path}")


def example_performance_comparison():
    """Generate a report comparing multiple strategies."""
    print("\n=== Example 4: Multi-Strategy Comparison Report ===\n")
    
    # Generate data for multiple strategies
    strategies = ['Conservative', 'Moderate', 'Aggressive']
    all_results = {}
    
    for i, strategy in enumerate(strategies):
        # Modify parameters for each strategy
        results = generate_sample_backtest_results()
        
        # Adjust metrics based on strategy type
        risk_multiplier = 0.5 + i * 0.5
        results['metrics']['volatility'] *= risk_multiplier
        results['metrics']['annual_return'] *= (0.7 + i * 0.3)
        results['metrics']['max_drawdown'] *= risk_multiplier
        results['metrics']['sharpe_ratio'] = (
            results['metrics']['annual_return'] / results['metrics']['volatility']
        )
        
        all_results[strategy] = results
    
    # Create comparison report
    config = ReportConfig()
    generator = StandardReportGenerator(config)
    
    # Generate individual reports and combine metrics
    comparison_data = {
        'strategies': {},
        'summary': {}
    }
    
    for strategy, results in all_results.items():
        comparison_data['strategies'][strategy] = results['metrics']
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data['strategies']).T
    
    # Generate comparison report
    combined_results = {
        'metrics': comparison_df.to_dict(),
        'comparison_table': comparison_df,
        'config': {
            'report_type': 'Multi-Strategy Comparison',
            'strategies': strategies,
            'comparison_date': datetime.now().strftime('%Y-%m-%d')
        }
    }
    
    output_paths = generator.generate_report(
        backtest_results=combined_results,
        strategy_name="Strategy Comparison Analysis",
        output_dir="reports/examples/comparison",
        formats=['markdown', 'html']
    )
    
    print("Comparison report generated!")
    for format_type, path in output_paths.items():
        print(f"  - {format_type}: {path}")


def main():
    """Run all examples."""
    print("Standard Report Generator Examples")
    print("=" * 50)
    
    # Create output directories
    import os
    for dir_path in ['reports/examples/basic', 
                     'reports/examples/custom_sections',
                     'reports/examples/styled', 
                     'reports/examples/comparison']:
        os.makedirs(dir_path, exist_ok=True)
    
    # Run examples
    example_basic_report()
    example_custom_sections()
    example_styled_report()
    example_performance_comparison()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("\nCheck the 'reports/examples/' directory for generated reports.")


if __name__ == '__main__':
    main()