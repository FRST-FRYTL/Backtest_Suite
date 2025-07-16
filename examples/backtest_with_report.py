"""
Example: Backtest with Automatic Report Generation

This example demonstrates how the backtest engine automatically generates
standardized reports after completing a backtest.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.engine import BacktestEngine
from src.strategies.builder import StrategyBuilder
from src.reporting import ReportConfig


def create_sample_data():
    """Create sample market data for demonstration"""
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    price = 100
    prices = []
    
    for _ in range(len(dates)):
        # Random walk with slight upward bias
        change = np.random.normal(0.0005, 0.02)
        price *= (1 + change)
        prices.append(price)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'Close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Ensure high >= low
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    data.attrs['symbol'] = 'TEST'
    
    return data


def create_simple_strategy():
    """Create a simple moving average crossover strategy"""
    builder = StrategyBuilder(name="MA Crossover Strategy")
    
    # Add indicators
    builder.add_indicator('SMA', 'fast_ma', period=20)
    builder.add_indicator('SMA', 'slow_ma', period=50)
    
    # Entry rule: Fast MA crosses above Slow MA
    builder.add_entry_rule(
        lambda data: (
            (data['fast_ma'] > data['slow_ma']) & 
            (data['fast_ma'].shift(1) <= data['slow_ma'].shift(1))
        ),
        name="MA Crossover Entry"
    )
    
    # Exit rule: Fast MA crosses below Slow MA
    builder.add_exit_rule(
        lambda data: (
            (data['fast_ma'] < data['slow_ma']) & 
            (data['fast_ma'].shift(1) >= data['slow_ma'].shift(1))
        ),
        name="MA Crossover Exit"
    )
    
    # Risk management
    builder.set_stop_loss(percent=0.02)  # 2% stop loss
    builder.set_take_profit(percent=0.05)  # 5% take profit
    
    return builder.build()


def example_1_default_report():
    """Example 1: Backtest with default report configuration"""
    print("Example 1: Default Report Generation")
    print("-" * 50)
    
    # Create sample data
    data = create_sample_data()
    
    # Create strategy
    strategy = create_simple_strategy()
    
    # Initialize backtest engine with default report settings
    engine = BacktestEngine(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        generate_report=True  # Enable automatic report generation
    )
    
    # Run backtest
    results = engine.run(
        data=data,
        strategy=strategy,
        start_date=datetime(2022, 6, 1),
        end_date=datetime(2023, 6, 1)
    )
    
    print("\nBacktest completed!")
    print(f"Total Return: {results['metrics']['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
    print()


def example_2_custom_report_config():
    """Example 2: Backtest with custom report configuration"""
    print("Example 2: Custom Report Configuration")
    print("-" * 50)
    
    # Create sample data
    data = create_sample_data()
    
    # Create strategy
    strategy = create_simple_strategy()
    
    # Create custom report configuration
    report_config = ReportConfig(
        title="Moving Average Strategy Performance Analysis",
        subtitle="Advanced Configuration Example",
        author="Quantitative Research Team",
        
        # Custom thresholds for evaluation
        min_sharpe_ratio=1.0,
        max_drawdown_limit=0.15,
        min_win_rate=0.50,
        
        # Select output formats
        output_formats=["html", "json"],
        
        # Custom styling
        chart_style="professional",
        figure_dpi=300,
        
        # Enable/disable sections
        include_market_regime_analysis=False  # Skip for simplicity
    )
    
    # Initialize backtest engine with custom report config
    engine = BacktestEngine(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        generate_report=True,
        report_config=report_config,
        report_dir="reports/custom"
    )
    
    # Run backtest
    results = engine.run(
        data=data,
        strategy=strategy
    )
    
    print("\nBacktest completed with custom report!")
    print()


def example_3_disable_report():
    """Example 3: Backtest without automatic report generation"""
    print("Example 3: Disable Automatic Report")
    print("-" * 50)
    
    # Create sample data
    data = create_sample_data()
    
    # Create strategy
    strategy = create_simple_strategy()
    
    # Initialize backtest engine without report generation
    engine = BacktestEngine(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        generate_report=False  # Disable automatic report
    )
    
    # Run backtest
    results = engine.run(
        data=data,
        strategy=strategy
    )
    
    print("\nBacktest completed without report generation")
    print(f"Results available in 'results' dictionary")
    
    # You can still generate a report manually if needed
    from src.reporting import StandardReportGenerator
    
    print("\nGenerating manual report...")
    generator = StandardReportGenerator()
    output_files = generator.generate_report(
        backtest_results=results,
        output_dir="reports/manual",
        report_name="manual_report"
    )
    
    print("Manual report generated!")
    print()


def example_4_multiple_strategies():
    """Example 4: Compare multiple strategies with reports"""
    print("Example 4: Multiple Strategy Comparison")
    print("-" * 50)
    
    # Create sample data
    data = create_sample_data()
    
    # Define multiple strategies
    strategies = []
    
    # Strategy 1: Fast MA Crossover
    builder1 = StrategyBuilder(name="Fast MA Strategy")
    builder1.add_indicator('SMA', 'fast_ma', period=10)
    builder1.add_indicator('SMA', 'slow_ma', period=30)
    builder1.add_entry_rule(
        lambda d: (d['fast_ma'] > d['slow_ma']) & (d['fast_ma'].shift(1) <= d['slow_ma'].shift(1))
    )
    builder1.add_exit_rule(
        lambda d: (d['fast_ma'] < d['slow_ma']) & (d['fast_ma'].shift(1) >= d['slow_ma'].shift(1))
    )
    strategies.append(builder1.build())
    
    # Strategy 2: Slow MA Crossover
    builder2 = StrategyBuilder(name="Slow MA Strategy")
    builder2.add_indicator('SMA', 'fast_ma', period=20)
    builder2.add_indicator('SMA', 'slow_ma', period=50)
    builder2.add_entry_rule(
        lambda d: (d['fast_ma'] > d['slow_ma']) & (d['fast_ma'].shift(1) <= d['slow_ma'].shift(1))
    )
    builder2.add_exit_rule(
        lambda d: (d['fast_ma'] < d['slow_ma']) & (d['fast_ma'].shift(1) >= d['slow_ma'].shift(1))
    )
    strategies.append(builder2.build())
    
    # Run backtests for each strategy
    for i, strategy in enumerate(strategies):
        print(f"\nTesting {strategy.name}...")
        
        # Custom config for each strategy
        config = ReportConfig(
            title=f"{strategy.name} Analysis",
            output_formats=["html"]
        )
        
        engine = BacktestEngine(
            initial_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005,
            generate_report=True,
            report_config=config,
            report_dir=f"reports/comparison"
        )
        
        results = engine.run(data=data, strategy=strategy)
        
        print(f"  Return: {results['metrics']['total_return']:.2%}")
        print(f"  Sharpe: {results['metrics']['sharpe_ratio']:.2f}")
        print(f"  Max DD: {results['metrics']['max_drawdown']:.2%}")
    
    print("\nAll strategy reports generated in reports/comparison/")
    print()


def main():
    """Run all examples"""
    print("=" * 70)
    print("BACKTEST WITH AUTOMATIC REPORT GENERATION EXAMPLES")
    print("=" * 70)
    print()
    
    # Run examples
    example_1_default_report()
    example_2_custom_report_config()
    example_3_disable_report()
    example_4_multiple_strategies()
    
    print("=" * 70)
    print("All examples completed!")
    print("Check the reports/ directory for generated reports.")
    print("=" * 70)


if __name__ == "__main__":
    main()