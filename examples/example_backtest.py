"""Example backtest script showing how to use the Backtest Suite."""

import asyncio
from datetime import datetime, timedelta

from src.data import StockDataFetcher
from src.indicators import RSI, BollingerBands, VWAP
from src.strategies import StrategyBuilder
from src.backtesting import BacktestEngine
from src.utils import PerformanceMetrics
from src.visualization import Dashboard, ChartGenerator
from src.optimization import StrategyOptimizer


async def main():
    """Run example backtest."""
    
    # 1. Fetch data
    print("Fetching data...")
    fetcher = StockDataFetcher()
    
    # Fetch 1 year of daily data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data = await fetcher.fetch(
        symbol="AAPL",
        start=start_date,
        end=end_date,
        interval="1d"
    )
    
    print(f"Fetched {len(data)} bars of data")
    
    # 2. Calculate indicators
    print("\nCalculating indicators...")
    
    # RSI
    rsi = RSI(period=14)
    data['rsi'] = rsi.calculate(data)
    
    # Bollinger Bands
    bb = BollingerBands(period=20, std_dev=2)
    bb_data = bb.calculate(data)
    data = data.join(bb_data)
    
    # VWAP
    vwap = VWAP(window=None)  # Session VWAP
    vwap_data = vwap.calculate(data)
    data = data.join(vwap_data[['vwap']])
    
    # 3. Create strategy
    print("\nBuilding strategy...")
    
    builder = StrategyBuilder("Example RSI Strategy")
    builder.set_description("Simple RSI mean reversion with Bollinger Bands")
    
    # Entry rules
    entry_rule = builder.add_entry_rule("rsi < 30 and Close < bb_lower")
    
    # Exit rules
    exit_rule = builder.add_exit_rule("rsi > 70 or Close > bb_upper")
    
    # Risk management
    builder.set_risk_management(
        stop_loss=0.05,  # 5% stop loss
        take_profit=0.10,  # 10% take profit
        max_positions=3
    )
    
    # Position sizing
    builder.set_position_sizing(
        method="percent",
        size=0.1  # 10% of portfolio per position
    )
    
    strategy = builder.build()
    
    # 4. Run backtest
    print("\nRunning backtest...")
    
    engine = BacktestEngine(
        initial_capital=100000,
        commission_rate=0.001,  # 0.1%
        slippage_rate=0.0005    # 0.05%
    )
    
    results = engine.run(data, strategy)
    
    # 5. Calculate performance metrics
    print("\nCalculating performance metrics...")
    
    metrics = PerformanceMetrics.calculate(
        results['equity_curve'],
        results['trades']
    )
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(metrics.generate_report())
    
    # 6. Generate visualizations
    print("\nGenerating visualizations...")
    
    # Create dashboard
    dashboard = Dashboard()
    dashboard_path = dashboard.create_dashboard(
        results,
        output_path="backtest_results.html"
    )
    print(f"Dashboard saved to: {dashboard_path}")
    
    # Create individual charts
    chart_gen = ChartGenerator(style="plotly")
    
    # Equity curve
    equity_fig = chart_gen.plot_equity_curve(results['equity_curve'])
    equity_fig.write_html("equity_curve.html")
    
    # Returns distribution
    returns = results['equity_curve']['total_value'].pct_change().dropna()
    returns_fig = chart_gen.plot_returns_distribution(returns)
    returns_fig.write_html("returns_distribution.html")
    
    # Trade chart
    trades_fig = chart_gen.plot_trades(
        data,
        results['trades'],
        "AAPL",
        indicators={'RSI': data['rsi'], 'VWAP': data['vwap']}
    )
    trades_fig.write_html("trades_chart.html")
    
    print("\nVisualization files created!")
    
    # 7. Parameter optimization example
    print("\n" + "="*50)
    print("PARAMETER OPTIMIZATION")
    print("="*50)
    
    # Define parameter grid
    param_grid = {
        'rsi_period': [10, 14, 20],
        'rsi_oversold': [25, 30, 35],
        'rsi_overbought': [65, 70, 75],
        'stop_loss': [0.03, 0.05, 0.07]
    }
    
    # Note: For demonstration, we're using a simple custom strategy builder
    # In practice, you'd extend the builder to accept these parameters
    
    print(f"Testing {len(param_grid['rsi_period']) * len(param_grid['rsi_oversold']) * len(param_grid['rsi_overbought']) * len(param_grid['stop_loss'])} parameter combinations...")
    
    # optimizer = StrategyOptimizer(
    #     data=data,
    #     strategy_builder=builder,
    #     optimization_metric="sharpe_ratio"
    # )
    
    # opt_results = optimizer.grid_search(param_grid)
    # print(f"\nBest parameters: {opt_results.best_params}")
    # print(f"Best Sharpe ratio: {opt_results.best_score:.3f}")
    
    print("\nBacktest complete!")


if __name__ == "__main__":
    asyncio.run(main())