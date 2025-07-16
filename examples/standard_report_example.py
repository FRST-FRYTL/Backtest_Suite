"""
Standard Report Example

This example demonstrates how to use the StandardReportGenerator
to create professional backtest reports.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reporting import StandardReportGenerator, ReportConfig


def generate_sample_data():
    """Generate sample backtest data for demonstration"""
    
    # Create sample equity curve
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Generate realistic equity curve
    returns = np.random.normal(0.0003, 0.01, len(dates))  # Daily returns
    equity_curve = pd.Series(
        (1 + returns).cumprod() * 100000,  # Starting with $100k
        index=dates,
        name='equity'
    )
    
    # Generate sample trades
    n_trades = 150
    trade_dates = np.random.choice(dates[:-5], n_trades, replace=False)
    trade_dates.sort()
    
    trades_data = []
    for i, entry_date in enumerate(trade_dates):
        # Random trade duration (1-10 days)
        duration_days = np.random.randint(1, 10)
        exit_date = entry_date + timedelta(days=duration_days)
        
        # Generate P&L with realistic distribution
        win = np.random.random() < 0.55  # 55% win rate
        if win:
            pnl = np.random.lognormal(6, 0.5)  # Winners
        else:
            pnl = -np.random.lognormal(5.8, 0.4)  # Losers
            
        trades_data.append({
            'entry_time': entry_date,
            'exit_time': exit_date,
            'side': np.random.choice(['long', 'short']),
            'size': np.random.uniform(5000, 20000),
            'pnl': pnl,
            'duration': duration_days * 24,  # Convert to hours
            'entry_reason': np.random.choice(['breakout', 'reversal', 'momentum']),
            'exit_reason': np.random.choice(['target', 'stop', 'time', 'signal'])
        })
    
    trades = pd.DataFrame(trades_data)
    
    # Calculate metrics
    returns_series = equity_curve.pct_change().dropna()
    
    # Calculate drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    
    metrics = {
        # Returns
        'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1),
        'annual_return': returns_series.mean() * 252,
        'monthly_return': returns_series.mean() * 21,
        
        # Risk
        'volatility': returns_series.std() * np.sqrt(252),
        'max_drawdown': drawdown.min(),
        'downside_deviation': returns_series[returns_series < 0].std() * np.sqrt(252),
        
        # Risk-adjusted
        'sharpe_ratio': (returns_series.mean() * 252) / (returns_series.std() * np.sqrt(252)),
        'sortino_ratio': (returns_series.mean() * 252) / (returns_series[returns_series < 0].std() * np.sqrt(252)),
        'calmar_ratio': (returns_series.mean() * 252) / abs(drawdown.min()),
        
        # Trading
        'win_rate': (trades['pnl'] > 0).mean(),
        'profit_factor': trades[trades['pnl'] > 0]['pnl'].sum() / abs(trades[trades['pnl'] < 0]['pnl'].sum()),
        'avg_win_loss_ratio': abs(trades[trades['pnl'] > 0]['pnl'].mean() / trades[trades['pnl'] < 0]['pnl'].mean()),
        
        # VaR
        'var_95': np.percentile(returns_series, 5),
        'cvar_95': returns_series[returns_series <= np.percentile(returns_series, 5)].mean(),
        
        # Additional
        'total_trades': len(trades),
        'avg_trade_duration': trades['duration'].mean(),
        'consistency_score': 0.75  # Mock consistency score
    }
    
    # Strategy parameters
    strategy_params = {
        'strategy_name': 'Momentum Breakout',
        'lookback_period': 20,
        'entry_threshold': 2.0,
        'exit_threshold': 0.5,
        'stop_loss': 0.02,
        'take_profit': 0.05,
        'position_sizing': 'risk_parity',
        'max_positions': 5,
        'risk_per_trade': 0.01
    }
    
    # Additional data for comprehensive report
    execution_stats = {
        'total_orders': len(trades) * 2,
        'filled_orders': len(trades) * 2,
        'rejected_orders': 3,
        'avg_fill_time': 125,  # milliseconds
        'avg_slippage': 0.0001,
        'max_slippage': 0.0005,
        'slippage_cost': 250,
        'total_commission': 1500,
        'total_spread_cost': 800,
        'avg_cost_per_trade': 15.33
    }
    
    performance_stats = {
        'total_time': 45.2,  # seconds
        'avg_time_per_bar': 0.062,
        'bars_processed': len(dates),
        'peak_memory': 128.5,  # MB
        'avg_memory': 95.2,
        'optimization_iterations': 0,
        'convergence_time': 0
    }
    
    data_statistics = {
        'start_date': dates[0],
        'end_date': dates[-1],
        'total_bars': len(dates),
        'missing_data_pct': 0.001,
        'outliers': 2,
        'gaps': 0,
        'suspicious_values': 0,
        'splits_adjusted': True,
        'dividends_adjusted': True,
        'currency_normalized': False
    }
    
    # Create sample market data for regime analysis
    market_data = pd.DataFrame({
        'close': equity_curve * np.random.uniform(0.95, 1.05, len(equity_curve)),
        'volume': np.random.randint(1000000, 5000000, len(equity_curve))
    }, index=dates)
    
    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'metrics': metrics,
        'returns': returns_series,
        'strategy_params': strategy_params,
        'execution_stats': execution_stats,
        'performance_stats': performance_stats,
        'data_statistics': data_statistics,
        'market_data': market_data
    }


def example_1_basic_report():
    """Example 1: Generate a basic report with default settings"""
    print("Example 1: Basic Report Generation")
    print("-" * 50)
    
    # Generate sample data
    backtest_results = generate_sample_data()
    
    # Create report generator with default config
    generator = StandardReportGenerator()
    
    # Generate report
    output_files = generator.generate_report(
        backtest_results=backtest_results,
        output_dir="reports/examples/",
        report_name="basic_example"
    )
    
    print(f"✓ HTML Report: {output_files['html']}")
    print(f"✓ JSON Report: {output_files.get('json', 'Not generated')}")
    print()


def example_2_custom_config():
    """Example 2: Custom configuration with specific thresholds"""
    print("Example 2: Custom Configuration")
    print("-" * 50)
    
    # Generate sample data
    backtest_results = generate_sample_data()
    
    # Create custom configuration
    config = ReportConfig(
        title="Advanced Momentum Strategy Analysis",
        subtitle="Q4 2023 Performance Review",
        author="Quantitative Research Team",
        
        # Custom thresholds
        min_sharpe_ratio=1.5,
        max_drawdown_limit=0.15,
        min_win_rate=0.55,
        
        # Select specific output formats
        output_formats=["html"],
        
        # Custom styling
        chart_style="professional",
        figure_dpi=300,
        
        # Custom colors
        color_scheme={
            "primary": "#1E88E5",
            "secondary": "#FFA726",
            "success": "#43A047",
            "warning": "#FB8C00",
            "danger": "#E53935",
            "info": "#00ACC1",
            "background": "#FFFFFF",
            "text": "#212121"
        }
    )
    
    # Create generator with custom config
    generator = StandardReportGenerator(config)
    
    # Generate report
    output_files = generator.generate_report(
        backtest_results=backtest_results,
        output_dir="reports/examples/",
        report_name="custom_config_example"
    )
    
    print(f"✓ Custom Report: {output_files['html']}")
    print()


def example_3_selective_sections():
    """Example 3: Generate report with only specific sections"""
    print("Example 3: Selective Sections")
    print("-" * 50)
    
    # Generate sample data
    backtest_results = generate_sample_data()
    
    # Config with only performance and risk sections
    config = ReportConfig(
        title="Performance & Risk Focus Report",
        
        # Disable some sections
        include_executive_summary=True,
        include_performance_analysis=True,
        include_risk_analysis=True,
        include_trade_analysis=False,  # Disabled
        include_market_regime_analysis=False,  # Disabled
        include_technical_details=False,  # Disabled
        
        output_formats=["html"]
    )
    
    generator = StandardReportGenerator(config)
    
    output_files = generator.generate_report(
        backtest_results=backtest_results,
        output_dir="reports/examples/",
        report_name="selective_sections_example"
    )
    
    print(f"✓ Focused Report: {output_files['html']}")
    print()


def example_4_custom_theme():
    """Example 4: Apply different visual themes"""
    print("Example 4: Custom Themes")
    print("-" * 50)
    
    # Generate sample data
    backtest_results = generate_sample_data()
    
    themes = ["professional", "minimal", "colorful"]
    
    for theme in themes:
        config = ReportConfig(
            title=f"{theme.title()} Theme Report",
            chart_style=theme,
            output_formats=["html"]
        )
        
        generator = StandardReportGenerator(config)
        generator.set_theme(theme)
        
        output_files = generator.generate_report(
            backtest_results=backtest_results,
            output_dir="reports/examples/",
            report_name=f"{theme}_theme_example"
        )
        
        print(f"✓ {theme.title()} Theme: {output_files['html']}")
    
    print()


def example_5_custom_section():
    """Example 5: Add custom sections to the report"""
    print("Example 5: Custom Sections")
    print("-" * 50)
    
    # Generate sample data
    backtest_results = generate_sample_data()
    
    # Create generator
    generator = StandardReportGenerator()
    
    # Add custom section with additional analysis
    custom_analysis = {
        "market_conditions": {
            "trend_strength": "Strong Uptrend",
            "volatility_regime": "Normal",
            "correlation_stability": "High"
        },
        "strategy_edge": {
            "alpha_generation": "2.3% annually",
            "information_ratio": 1.45,
            "consistency": "85% positive months"
        },
        "recommendations": [
            "Consider increasing position sizes in strong trends",
            "Add volatility filters during high VIX periods",
            "Implement regime detection for adaptive sizing"
        ]
    }
    
    generator.add_custom_section(
        section_name="Enhanced Analysis",
        section_content=custom_analysis
    )
    
    # Generate report
    output_files = generator.generate_report(
        backtest_results=backtest_results,
        output_dir="reports/examples/",
        report_name="custom_section_example"
    )
    
    print(f"✓ Report with Custom Section: {output_files['html']}")
    print()


def example_6_batch_processing():
    """Example 6: Generate reports for multiple strategies"""
    print("Example 6: Batch Processing")
    print("-" * 50)
    
    # Simulate multiple strategy results
    strategies = {
        "momentum": {
            "sharpe": 1.45,
            "drawdown": -0.18,
            "win_rate": 0.52
        },
        "mean_reversion": {
            "sharpe": 1.82,
            "drawdown": -0.12,
            "win_rate": 0.65
        },
        "pairs_trading": {
            "sharpe": 2.15,
            "drawdown": -0.08,
            "win_rate": 0.71
        }
    }
    
    for strategy_name, adjustments in strategies.items():
        # Generate base data
        backtest_results = generate_sample_data()
        
        # Adjust metrics for each strategy
        backtest_results['metrics']['sharpe_ratio'] = adjustments['sharpe']
        backtest_results['metrics']['max_drawdown'] = adjustments['drawdown']
        backtest_results['metrics']['win_rate'] = adjustments['win_rate']
        backtest_results['strategy_params']['strategy_name'] = strategy_name.replace('_', ' ').title()
        
        # Create config for each strategy
        config = ReportConfig(
            title=f"{strategy_name.replace('_', ' ').title()} Strategy Analysis",
            output_formats=["html", "json"]
        )
        
        generator = StandardReportGenerator(config)
        
        output_files = generator.generate_report(
            backtest_results=backtest_results,
            output_dir=f"reports/examples/{strategy_name}/",
            report_name=f"{strategy_name}_report"
        )
        
        print(f"✓ {strategy_name}: {output_files['html']}")
    
    print()


def example_7_minimal_data():
    """Example 7: Handle minimal data gracefully"""
    print("Example 7: Minimal Data Handling")
    print("-" * 50)
    
    # Create minimal required data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    equity_curve = pd.Series(
        np.linspace(100000, 110000, len(dates)),
        index=dates
    )
    
    # Minimal backtest results
    backtest_results = {
        'equity_curve': equity_curve,
        'trades': pd.DataFrame(),  # Empty trades
        'metrics': {
            'total_return': 0.10,
            'sharpe_ratio': 0.85
        },
        'strategy_params': {
            'strategy_name': 'Buy and Hold'
        }
    }
    
    # Generate report with minimal data
    generator = StandardReportGenerator()
    
    output_files = generator.generate_report(
        backtest_results=backtest_results,
        output_dir="reports/examples/",
        report_name="minimal_data_example"
    )
    
    print(f"✓ Minimal Data Report: {output_files['html']}")
    print()


def main():
    """Run all examples"""
    print("=" * 70)
    print("STANDARD REPORT GENERATOR EXAMPLES")
    print("=" * 70)
    print()
    
    # Create output directory
    os.makedirs("reports/examples", exist_ok=True)
    
    # Run examples
    example_1_basic_report()
    example_2_custom_config()
    example_3_selective_sections()
    example_4_custom_theme()
    example_5_custom_section()
    example_6_batch_processing()
    example_7_minimal_data()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("Check the reports/examples/ directory for generated reports.")
    print("=" * 70)


if __name__ == "__main__":
    main()