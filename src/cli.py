"""Command-line interface for the backtesting suite."""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import track

from .data import StockDataFetcher
from .indicators import RSI, BollingerBands, VWAP, TSV
from .strategies import StrategyBuilder
from .backtesting import BacktestEngine
from .utils import PerformanceMetrics
from .visualization import Dashboard, ChartGenerator
from .optimization import StrategyOptimizer, WalkForwardAnalysis


console = Console()


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """Backtest Suite - Professional backtesting framework for trading strategies."""
    pass


@cli.command()
@click.option('--symbol', '-s', required=True, help='Stock symbol to fetch')
@click.option('--start', '-S', help='Start date (YYYY-MM-DD)')
@click.option('--end', '-E', help='End date (YYYY-MM-DD)')
@click.option('--interval', '-i', default='1d', help='Data interval (1m, 5m, 1h, 1d)')
@click.option('--output', '-o', help='Output file path')
def fetch(symbol: str, start: str, end: str, interval: str, output: Optional[str]):
    """Fetch stock data from Yahoo Finance."""
    console.print(f"[bold blue]Fetching data for {symbol}...[/bold blue]")
    
    # Set default dates if not provided
    if not end:
        end = datetime.now().strftime('%Y-%m-%d')
    if not start:
        start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
    # Fetch data
    fetcher = StockDataFetcher()
    
    try:
        data = fetcher.fetch_sync(symbol, start, end, interval)
        console.print(f"[green]✓ Fetched {len(data)} bars of data[/green]")
        
        # Display summary
        table = Table(title=f"{symbol} Data Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Start Date", str(data.index[0]))
        table.add_row("End Date", str(data.index[-1]))
        table.add_row("Total Bars", str(len(data)))
        table.add_row("Open", f"${data['Open'].iloc[-1]:.2f}")
        table.add_row("High", f"${data['High'].iloc[-1]:.2f}")
        table.add_row("Low", f"${data['Low'].iloc[-1]:.2f}")
        table.add_row("Close", f"${data['Close'].iloc[-1]:.2f}")
        table.add_row("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
        
        console.print(table)
        
        # Save if output specified
        if output:
            data.to_csv(output)
            console.print(f"[green]✓ Saved data to {output}[/green]")
            
    except Exception as e:
        console.print(f"[red]✗ Error fetching data: {e}[/red]")
        raise click.Exit(1)


@cli.command()
@click.option('--data', '-d', required=True, help='Path to data file (CSV)')
@click.option('--strategy', '-s', required=True, help='Strategy config file (YAML/JSON)')
@click.option('--capital', '-c', default=100000, help='Initial capital')
@click.option('--commission', default=0.001, help='Commission rate')
@click.option('--output', '-o', help='Output directory for results')
@click.option('--no-chart', is_flag=True, help='Skip chart generation')
def run(data: str, strategy: str, capital: float, commission: float, 
        output: Optional[str], no_chart: bool):
    """Run a backtest with specified strategy."""
    console.print("[bold blue]Running backtest...[/bold blue]")
    
    # Load data
    try:
        df = pd.read_csv(data, index_col=0, parse_dates=True)
        console.print(f"[green]✓ Loaded {len(df)} bars of data[/green]")
    except Exception as e:
        console.print(f"[red]✗ Error loading data: {e}[/red]")
        raise click.Exit(1)
        
    # Load strategy
    try:
        strategy_path = Path(strategy)
        if strategy_path.suffix == '.yaml':
            strategy_builder = StrategyBuilder.from_yaml(strategy)
        elif strategy_path.suffix == '.json':
            strategy_builder = StrategyBuilder.from_json(strategy)
        else:
            console.print("[red]✗ Strategy file must be YAML or JSON[/red]")
            raise click.Exit(1)
            
        strategy_obj = strategy_builder.build()
        console.print(f"[green]✓ Loaded strategy: {strategy_obj.name}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Error loading strategy: {e}[/red]")
        raise click.Exit(1)
        
    # Run backtest
    engine = BacktestEngine(
        initial_capital=capital,
        commission_rate=commission
    )
    
    with console.status("[bold green]Running backtest..."):
        results = engine.run(df, strategy_obj, progress_bar=True)
        
    # Calculate metrics
    metrics = PerformanceMetrics.calculate(
        results['equity_curve'],
        results['trades']
    )
    
    # Display results
    console.print("\n[bold cyan]Performance Summary[/bold cyan]")
    console.print(metrics.generate_report())
    
    # Save results
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
            
        # Save trades
        results['trades'].to_csv(output_dir / 'trades.csv')
        
        # Save equity curve
        results['equity_curve'].to_csv(output_dir / 'equity_curve.csv')
        
        console.print(f"[green]✓ Saved results to {output_dir}[/green]")
        
        # Generate dashboard
        if not no_chart:
            dashboard = Dashboard()
            dashboard_path = dashboard.create_dashboard(
                results,
                output_path=str(output_dir / 'dashboard.html')
            )
            console.print(f"[green]✓ Generated dashboard: {dashboard_path}[/green]")


@cli.command()
@click.option('--data', '-d', required=True, help='Path to data file')
@click.option('--strategy', '-s', required=True, help='Base strategy config')
@click.option('--params', '-p', required=True, help='Parameter search config (YAML)')
@click.option('--method', '-m', default='grid', 
              type=click.Choice(['grid', 'random', 'differential']))
@click.option('--metric', default='sharpe_ratio', help='Metric to optimize')
@click.option('--output', '-o', help='Output directory')
def optimize(data: str, strategy: str, params: str, method: str, 
            metric: str, output: Optional[str]):
    """Optimize strategy parameters."""
    console.print("[bold blue]Running strategy optimization...[/bold blue]")
    
    # Load data
    df = pd.read_csv(data, index_col=0, parse_dates=True)
    
    # Load base strategy
    if Path(strategy).suffix == '.yaml':
        strategy_builder = StrategyBuilder.from_yaml(strategy)
    else:
        strategy_builder = StrategyBuilder.from_json(strategy)
        
    # Load parameter config
    with open(params, 'r') as f:
        param_config = yaml.safe_load(f)
        
    # Create optimizer
    optimizer = StrategyOptimizer(
        data=df,
        strategy_builder=strategy_builder,
        optimization_metric=metric
    )
    
    # Run optimization
    with console.status(f"[bold green]Running {method} search..."):
        if method == 'grid':
            results = optimizer.grid_search(param_config['parameters'])
        elif method == 'random':
            results = optimizer.random_search(
                param_config['parameters'],
                n_iter=param_config.get('n_iter', 100)
            )
        else:  # differential
            results = optimizer.differential_evolution(param_config['parameters'])
            
    # Display results
    console.print(f"\n[bold cyan]Optimization Results[/bold cyan]")
    console.print(f"Best Score ({metric}): {results.best_score:.4f}")
    console.print(f"Total Iterations: {results.total_iterations}")
    
    # Display best parameters
    table = Table(title="Best Parameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    
    for param, value in results.best_params.items():
        table.add_row(param, str(value))
        
    console.print(table)
    
    # Display top results
    console.print("\n[bold cyan]Top 5 Results[/bold cyan]")
    top_results = results.get_top_results(5)
    
    for i, result in enumerate(top_results, 1):
        console.print(f"\n{i}. Score: {result['score']:.4f}")
        console.print(f"   Params: {result['params']}")
        
    # Save results
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all results
        results_df = results.to_dataframe()
        results_df.to_csv(output_dir / 'optimization_results.csv')
        
        # Save best parameters
        with open(output_dir / 'best_params.yaml', 'w') as f:
            yaml.dump(results.best_params, f)
            
        console.print(f"[green]✓ Saved results to {output_dir}[/green]")


@cli.command()
@click.option('--symbols', '-s', required=True, help='Comma-separated symbols')
@click.option('--strategies', '-S', required=True, help='Directory with strategies')
@click.option('--start', help='Start date')
@click.option('--end', help='End date')
@click.option('--capital', '-c', default=100000, help='Initial capital')
@click.option('--output', '-o', required=True, help='Output directory')
def batch(symbols: str, strategies: str, start: str, end: str, 
          capital: float, output: str):
    """Run batch backtests for multiple symbols and strategies."""
    console.print("[bold blue]Running batch backtests...[/bold blue]")
    
    # Parse symbols
    symbol_list = [s.strip() for s in symbols.split(',')]
    
    # Find strategy files
    strategy_dir = Path(strategies)
    strategy_files = list(strategy_dir.glob('*.yaml')) + list(strategy_dir.glob('*.json'))
    
    if not strategy_files:
        console.print("[red]✗ No strategy files found[/red]")
        raise click.Exit(1)
        
    console.print(f"Found {len(strategy_files)} strategies and {len(symbol_list)} symbols")
    console.print(f"Total backtests: {len(strategy_files) * len(symbol_list)}")
    
    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run backtests
    results_summary = []
    
    for symbol in track(symbol_list, description="Processing symbols..."):
        # Fetch data
        fetcher = StockDataFetcher()
        try:
            data = fetcher.fetch_sync(symbol, start, end)
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to fetch {symbol}: {e}[/yellow]")
            continue
            
        for strategy_file in strategy_files:
            try:
                # Load strategy
                if strategy_file.suffix == '.yaml':
                    strategy_builder = StrategyBuilder.from_yaml(str(strategy_file))
                else:
                    strategy_builder = StrategyBuilder.from_json(str(strategy_file))
                    
                strategy_obj = strategy_builder.build()
                
                # Run backtest
                engine = BacktestEngine(initial_capital=capital)
                results = engine.run(data, strategy_obj, progress_bar=False)
                
                # Calculate metrics
                metrics = PerformanceMetrics.calculate(
                    results['equity_curve'],
                    results['trades']
                )
                
                # Store summary
                results_summary.append({
                    'symbol': symbol,
                    'strategy': strategy_file.stem,
                    'total_return': metrics.total_return,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'win_rate': metrics.win_rate,
                    'total_trades': metrics.total_trades
                })
                
                # Save individual results
                result_dir = output_dir / f"{symbol}_{strategy_file.stem}"
                result_dir.mkdir(exist_ok=True)
                
                results['equity_curve'].to_csv(result_dir / 'equity_curve.csv')
                results['trades'].to_csv(result_dir / 'trades.csv')
                
            except Exception as e:
                console.print(f"[yellow]⚠ Failed {symbol} + {strategy_file.stem}: {e}[/yellow]")
                
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(output_dir / 'batch_summary.csv', index=False)
    
    # Display summary
    console.print("\n[bold cyan]Batch Results Summary[/bold cyan]")
    
    # Best performing combinations
    if not summary_df.empty:
        best_return = summary_df.nlargest(5, 'total_return')
        
        table = Table(title="Top 5 by Total Return")
        table.add_column("Symbol")
        table.add_column("Strategy")
        table.add_column("Return %")
        table.add_column("Sharpe")
        
        for _, row in best_return.iterrows():
            table.add_row(
                row['symbol'],
                row['strategy'],
                f"{row['total_return']:.2f}%",
                f"{row['sharpe_ratio']:.2f}"
            )
            
        console.print(table)
        
    console.print(f"[green]✓ Batch complete. Results saved to {output_dir}[/green]")


@cli.command()
def indicators():
    """List available indicators."""
    console.print("[bold cyan]Available Technical Indicators[/bold cyan]\n")
    
    indicators_list = [
        ("RSI", "Relative Strength Index", "Momentum oscillator (0-100)"),
        ("VWMA Bands", "Volume Weighted MA Bands", "Price bands weighted by volume"),
        ("Bollinger Bands", "Bollinger Bands", "Volatility bands around SMA"),
        ("TSV", "Time Segmented Volume", "Money flow indicator"),
        ("VWAP", "Volume Weighted Average Price", "Intraday price benchmark"),
        ("Anchored VWAP", "Anchored VWAP", "VWAP from specific date"),
    ]
    
    meta_indicators = [
        ("Fear & Greed", "Fear and Greed Index", "Market sentiment (0-100)"),
        ("Insider Trading", "Insider Trading Data", "Executive buy/sell activity"),
        ("Max Pain", "Options Max Pain", "Strike with maximum option holder loss"),
    ]
    
    # Technical indicators table
    table = Table(title="Technical Indicators")
    table.add_column("Indicator", style="cyan")
    table.add_column("Full Name", style="magenta")
    table.add_column("Description")
    
    for ind in indicators_list:
        table.add_row(*ind)
        
    console.print(table)
    
    # Meta indicators table
    table2 = Table(title="\nMeta Indicators")
    table2.add_column("Indicator", style="cyan")
    table2.add_column("Full Name", style="magenta")
    table2.add_column("Description")
    
    for ind in meta_indicators:
        table2.add_row(*ind)
        
    console.print(table2)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()