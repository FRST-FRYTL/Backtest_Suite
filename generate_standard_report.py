#!/usr/bin/env python3
"""
Generate standardized reports from backtest results.

This script provides a command-line interface for generating professional
reports from backtest results in multiple formats (Markdown, HTML, JSON).
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.reporting import StandardReportGenerator, ReportConfig
from src.reporting.report_config import ReportSection


def load_backtest_results(file_path: str) -> Dict[str, Any]:
    """
    Load backtest results from various file formats.
    
    Args:
        file_path: Path to the backtest results file
        
    Returns:
        Dictionary containing backtest results
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine file type and load accordingly
    if path.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
    elif path.suffix in ['.csv', '.tsv']:
        # Assume it's trade data
        data = {
            'trades': pd.read_csv(path)
        }
    elif path.suffix in ['.pkl', '.pickle']:
        data = pd.read_pickle(path)
        if isinstance(data, pd.DataFrame):
            data = {'trades': data}
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    return data


def prepare_sample_data() -> Dict[str, Any]:
    """
    Generate sample backtest data for demonstration.
    
    Returns:
        Dictionary containing sample backtest results
    """
    # Generate sample equity curve
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))
    equity = pd.Series((1 + returns).cumprod() * 10000, index=dates)
    
    # Generate sample trades
    n_trades = 100
    trade_dates = pd.to_datetime(np.random.choice(dates[:-1], n_trades, replace=False))
    trades = pd.DataFrame({
        'entry_date': trade_dates,
        'exit_date': trade_dates + pd.Timedelta(days=np.random.randint(1, 10)),
        'pnl': np.random.normal(50, 200, n_trades),
        'entry_price': np.random.uniform(100, 200, n_trades),
        'exit_price': np.random.uniform(100, 200, n_trades),
        'side': np.random.choice(['long', 'short'], n_trades)
    })
    
    # Calculate metrics
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(equity)) - 1
    daily_returns = equity.pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    
    # Calculate drawdown
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Trade metrics
    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] < 0]
    
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'monthly_return': annual_return / 12,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sharpe_ratio * 1.2,  # Simplified
        'max_drawdown': max_drawdown,
        'volatility': daily_returns.std() * np.sqrt(252),
        'total_trades': len(trades),
        'win_rate': len(winning_trades) / len(trades),
        'profit_factor': winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum()),
        'avg_win': winning_trades['pnl'].mean(),
        'avg_loss': losing_trades['pnl'].mean(),
        'win_loss_ratio': abs(winning_trades['pnl'].mean() / losing_trades['pnl'].mean())
    }
    
    # Generate optimization results
    param_grid = []
    for atr_length in [10, 12, 14, 16]:
        for factor in [2.0, 2.5, 3.0, 3.5]:
            param_grid.append({
                'atr_length': atr_length,
                'factor': factor,
                'sharpe_ratio': np.random.uniform(0.5, 2.5),
                'total_return': np.random.uniform(0.1, 0.5),
                'max_drawdown': np.random.uniform(-0.3, -0.1)
            })
    
    optimization_df = pd.DataFrame(param_grid)
    
    return {
        'equity_curve': equity,
        'trades': trades,
        'metrics': metrics,
        'optimization_results': optimization_df,
        'config': {
            'start_date': str(dates[0].date()),
            'end_date': str(dates[-1].date()),
            'initial_capital': 10000,
            'commission': 0.001
        }
    }


def main():
    """Main function to generate standardized reports."""
    parser = argparse.ArgumentParser(
        description='Generate standardized backtest reports'
    )
    
    parser.add_argument(
        'input',
        nargs='?',
        help='Path to backtest results file (JSON, CSV, or pickle). If not provided, uses sample data.'
    )
    
    parser.add_argument(
        '--strategy',
        default='Sample Strategy',
        help='Strategy name for the report'
    )
    
    parser.add_argument(
        '--output-dir',
        default='reports/standardized',
        help='Output directory for reports (default: reports/standardized)'
    )
    
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['markdown', 'html', 'json', 'all'],
        default=['all'],
        help='Output formats (default: all)'
    )
    
    parser.add_argument(
        '--sections',
        nargs='+',
        choices=['executive_summary', 'methodology', 'performance', 'risk', 
                'trades', 'optimization', 'timeframe', 'recommendations', 
                'technical', 'appendix'],
        help='Specific sections to include (default: all)'
    )
    
    parser.add_argument(
        '--style',
        choices=['default', 'dark', 'minimal', 'professional'],
        default='default',
        help='Report style theme'
    )
    
    parser.add_argument(
        '--company',
        help='Company name for branding'
    )
    
    parser.add_argument(
        '--logo',
        help='Path to company logo'
    )
    
    args = parser.parse_args()
    
    # Load or generate data
    if args.input:
        print(f"Loading backtest results from: {args.input}")
        try:
            backtest_data = load_backtest_results(args.input)
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)
    else:
        print("No input file provided. Generating sample data for demonstration...")
        backtest_data = prepare_sample_data()
    
    # Configure report
    config = ReportConfig()
    
    # Customize sections if specified
    if args.sections:
        # Disable all sections first
        for section in ReportSection:
            config.disable_section(section)
        
        # Enable only specified sections
        section_map = {
            'executive_summary': ReportSection.EXECUTIVE_SUMMARY,
            'methodology': ReportSection.METHODOLOGY,
            'performance': ReportSection.PERFORMANCE_METRICS,
            'risk': ReportSection.RISK_ANALYSIS,
            'trades': ReportSection.TRADE_ANALYSIS,
            'optimization': ReportSection.PARAMETER_OPTIMIZATION,
            'timeframe': ReportSection.TIMEFRAME_ANALYSIS,
            'recommendations': ReportSection.RECOMMENDATIONS,
            'technical': ReportSection.TECHNICAL_DETAILS,
            'appendix': ReportSection.APPENDIX
        }
        
        for section_name in args.sections:
            if section_name in section_map:
                config.enable_section(section_map[section_name])
    
    # Apply style theme
    if args.style == 'dark':
        config.customize_style(
            primary_color='#1E88E5',
            secondary_color='#FFA726',
            chart_template='plotly_dark'
        )
    elif args.style == 'minimal':
        config.customize_style(
            primary_color='#333333',
            secondary_color='#666666',
            chart_template='simple_white'
        )
    elif args.style == 'professional':
        config.customize_style(
            primary_color='#003366',
            secondary_color='#336699',
            chart_template='seaborn'
        )
    
    # Add company branding
    if args.company:
        config.customize_style(company_name=args.company)
    if args.logo:
        config.customize_style(logo_path=args.logo)
    
    # Determine output formats
    if 'all' in args.formats:
        formats = ['markdown', 'html', 'json']
    else:
        formats = args.formats
    
    # Create report generator
    generator = StandardReportGenerator(config)
    
    # Generate reports
    print(f"\nGenerating {', '.join(formats)} reports for '{args.strategy}'...")
    
    try:
        output_paths = generator.generate_report(
            backtest_results=backtest_data,
            strategy_name=args.strategy,
            output_dir=args.output_dir,
            formats=formats
        )
        
        print("\nReports generated successfully!")
        print("\nOutput files:")
        for format_type, path in output_paths.items():
            if format_type == 'charts':
                print(f"  - Charts: {path}")
            else:
                print(f"  - {format_type.capitalize()}: {path}")
        
        # Display summary metrics
        if 'metrics' in backtest_data:
            metrics = backtest_data['metrics']
            print("\nPerformance Summary:")
            print(f"  - Total Return: {metrics.get('total_return', 0)*100:.2f}%")
            print(f"  - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  - Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"  - Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
        
    except Exception as e:
        print(f"\nError generating reports: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()