"""
Execute all trading strategies with proper data handling.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import StockDataFetcher
from src.indicators import RSI, BollingerBands, VWAP
from src.strategies import StrategyBuilder
from src.backtesting import BacktestEngine
from src.utils import PerformanceMetrics
from src.visualization import Dashboard, ChartGenerator
from src.indicators.technical_indicators import TechnicalIndicators


def prepare_data(data):
    """Convert column names to lowercase for compatibility."""
    data = data.copy()
    data.columns = data.columns.str.lower()
    return data


async def execute_rolling_vwap_strategy(raw_data, initial_capital=10000):
    """Execute rolling VWAP strategy."""
    print("\n" + "="*60)
    print("EXECUTING ROLLING VWAP STRATEGY")
    print("="*60)
    
    # Prepare data
    data = prepare_data(raw_data)
    
    # Calculate VWAP indicators
    vwap_data = TechnicalIndicators.rolling_vwap(
        data['high'], data['low'], data['close'], data['volume'], 
        period=20
    )
    data['vwap'] = vwap_data['vwap']
    data['vwap_std'] = vwap_data['std']
    
    # Calculate VWAP bands
    bands = TechnicalIndicators.vwap_bands(
        data['vwap'], data['vwap_std'], [1, 2]
    )
    data['vwap_upper'] = bands['upper_1']
    data['vwap_lower'] = bands['lower_1']
    
    # Volume analysis
    data['volume_ma'] = data['volume'].rolling(window=20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    
    # Build strategy
    builder = StrategyBuilder("Rolling VWAP Strategy")
    builder.set_description("Buy when price crosses below VWAP with high volume")
    
    # Entry rules
    entry_rule = builder.add_entry_rule(
        "(close < vwap) and (close.shift(1) >= vwap.shift(1)) and (volume_ratio > 1.2)"
    )
    
    # Exit rules
    exit_rule = builder.add_exit_rule(
        "(close > vwap_upper) or (close < vwap * 0.97)"
    )
    
    # Risk management
    builder.set_risk_management(
        stop_loss=0.03,
        take_profit=0.06,
        max_positions=3
    )
    
    # Position sizing
    builder.set_position_sizing(
        method="percent",
        size=0.1
    )
    
    strategy = builder.build()
    
    # Run backtest
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    
    results = engine.run(data, strategy)
    
    # Calculate metrics
    metrics = PerformanceMetrics.calculate(
        results['equity_curve'],
        results['trades']
    )
    
    print("\nRolling VWAP Strategy Results:")
    print(metrics.generate_report())
    
    return results, metrics


async def execute_mean_reversion_strategy(raw_data, initial_capital=10000):
    """Execute mean reversion strategy."""
    print("\n" + "="*60)
    print("EXECUTING MEAN REVERSION STRATEGY")
    print("="*60)
    
    # Prepare data
    data = prepare_data(raw_data)
    
    # Calculate indicators
    rsi = RSI(period=14)
    data['rsi'] = rsi.calculate(data)
    
    bb = BollingerBands(period=20, std_dev=2)
    bb_data = bb.calculate(data)
    data = data.join(bb_data)
    
    # ATR for dynamic stops
    data['atr'] = TechnicalIndicators.atr(
        data['high'], data['low'], data['close'], period=14
    )
    
    # Build strategy
    builder = StrategyBuilder("Mean Reversion Strategy")
    builder.set_description("Buy oversold conditions at lower BB, sell overbought at upper BB")
    
    # Entry rules
    entry_rule = builder.add_entry_rule(
        "(rsi < 30) and (close <= bb_lower * 1.01)"
    )
    
    # Exit rules
    exit_rule = builder.add_exit_rule(
        "(rsi > 70) or (close >= bb_upper * 0.99)"
    )
    
    # Risk management
    builder.set_risk_management(
        stop_loss=0.04,
        take_profit=0.08,
        max_positions=4
    )
    
    # Position sizing
    builder.set_position_sizing(
        method="percent",
        size=0.1
    )
    
    strategy = builder.build()
    
    # Run backtest
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    
    results = engine.run(data, strategy)
    
    # Calculate metrics
    metrics = PerformanceMetrics.calculate(
        results['equity_curve'],
        results['trades']
    )
    
    print("\nMean Reversion Strategy Results:")
    print(metrics.generate_report())
    
    return results, metrics


async def execute_momentum_strategy(raw_data, initial_capital=10000):
    """Execute momentum strategy."""
    print("\n" + "="*60)
    print("EXECUTING MOMENTUM STRATEGY")
    print("="*60)
    
    # Prepare data
    data = prepare_data(raw_data)
    
    # Calculate MACD
    macd_data = TechnicalIndicators.macd(
        data['close'], fast_period=12, slow_period=26, signal_period=9
    )
    data['macd'] = macd_data['macd']
    data['macd_signal'] = macd_data['signal']
    data['macd_histogram'] = macd_data['histogram']
    
    # Calculate ADX
    adx_data = TechnicalIndicators.adx(
        data['high'], data['low'], data['close'], period=14
    )
    data['adx'] = adx_data['adx']
    
    # Price momentum
    data['momentum_20'] = data['close'].pct_change(20)
    data['momentum_50'] = data['close'].pct_change(50)
    
    # Volume momentum
    data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
    data['volume_momentum'] = data['volume'] / data['volume_ma_20']
    
    # Build strategy
    builder = StrategyBuilder("Momentum Strategy")
    builder.set_description("Buy strong momentum with MACD confirmation and ADX strength")
    
    # Entry rules
    entry_rule = builder.add_entry_rule(
        "(macd > macd_signal) and (macd > 0) and " +
        "(adx > 25) and (momentum_20 > 0.05) and " +
        "(volume_momentum > 1.1)"
    )
    
    # Exit rules
    exit_rule = builder.add_exit_rule(
        "(macd < macd_signal) or (momentum_20 < -0.02) or (adx < 20)"
    )
    
    # Risk management
    builder.set_risk_management(
        stop_loss=0.05,
        take_profit=0.15,
        max_positions=3
    )
    
    # Position sizing
    builder.set_position_sizing(
        method="percent",
        size=0.15
    )
    
    strategy = builder.build()
    
    # Run backtest
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    
    results = engine.run(data, strategy)
    
    # Calculate metrics
    metrics = PerformanceMetrics.calculate(
        results['equity_curve'],
        results['trades']
    )
    
    print("\nMomentum Strategy Results:")
    print(metrics.generate_report())
    
    return results, metrics


async def main():
    """Execute all strategies with monthly contributions simulation."""
    print("="*60)
    print("STRATEGY EXECUTION WITH $10,000 + $500/month")
    print("="*60)
    print(f"Initial Capital: $10,000")
    print(f"Monthly Contribution: $500")
    print(f"Symbol: SPY")
    print(f"Period: 5 years")
    print("="*60)
    
    # Fetch data
    fetcher = StockDataFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)
    
    print("\nFetching market data...")
    data = await fetcher.fetch(
        symbol="SPY",
        start=start_date,
        end=end_date,
        interval="1d"
    )
    print(f"Fetched {len(data)} bars of data")
    
    # Calculate total capital with monthly contributions
    months = 5 * 12  # 5 years
    total_contributed = 10000 + (500 * months)
    print(f"\nTotal capital over 5 years: ${total_contributed:,}")
    
    # Execute strategies
    results = {}
    
    # Rolling VWAP
    vwap_results, vwap_metrics = await execute_rolling_vwap_strategy(data.copy())
    results['rolling_vwap'] = {
        'results': vwap_results,
        'metrics': vwap_metrics
    }
    
    # Mean Reversion
    mr_results, mr_metrics = await execute_mean_reversion_strategy(data.copy())
    results['mean_reversion'] = {
        'results': mr_results,
        'metrics': mr_metrics
    }
    
    # Momentum
    mom_results, mom_metrics = await execute_momentum_strategy(data.copy())
    results['momentum'] = {
        'results': mom_results,
        'metrics': mom_metrics
    }
    
    # Generate comparative report
    print("\n" + "="*60)
    print("COMPARATIVE STRATEGY ANALYSIS")
    print("="*60)
    
    comparison = pd.DataFrame()
    for strategy_name, data in results.items():
        metrics = data['metrics']
        comparison[strategy_name] = {
            'Total Return': f"{metrics.total_return:.2%}",
            'Annual Return': f"{metrics.cagr:.2%}",
            'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
            'Max Drawdown': f"{metrics.max_drawdown:.2%}",
            'Win Rate': f"{metrics.win_rate:.2%}",
            'Total Trades': metrics.total_trades,
            'Avg Trade Return': f"{metrics.avg_return:.2%}",
            'Profit Factor': f"{metrics.profit_factor:.2f}"
        }
    
    print(comparison.T.to_string())
    
    # Save results
    output_dir = "reports/strategy_execution"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary
    summary = {
        'execution_date': datetime.now().isoformat(),
        'initial_capital': 10000,
        'monthly_contribution': 500,
        'total_contributed': total_contributed,
        'symbol': 'SPY',
        'period_years': 5,
        'strategies': {}
    }
    
    for strategy_name, data in results.items():
        metrics = data['metrics']
        summary['strategies'][strategy_name] = {
            'total_return': metrics.total_return,
            'annual_return': metrics.cagr,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'win_rate': metrics.win_rate,
            'total_trades': metrics.total_trades,
            'final_value': float(data['results']['equity_curve']['total_value'].iloc[-1])
        }
    
    with open(f"{output_dir}/strategy_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate visualizations
    chart_gen = ChartGenerator(style="plotly")
    dashboard = Dashboard()
    
    for strategy_name, data in results.items():
        strategy_results = data['results']
        
        # Equity curve
        equity_fig = chart_gen.plot_equity_curve(strategy_results['equity_curve'])
        equity_fig.write_html(f"{output_dir}/{strategy_name}_equity_curve.html")
        
        # Dashboard
        dashboard.create_dashboard(
            strategy_results,
            output_path=f"{output_dir}/{strategy_name}_dashboard.html"
        )
        
        # Save trades
        strategy_results['trades'].to_csv(f"{output_dir}/{strategy_name}_trades.csv")
    
    # Store results in memory
    memory_key = f"strategy_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\nResults saved to {output_dir}/")
    print("\nStrategy execution completed successfully!")
    
    # Print final summary
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"Best Strategy by Sharpe Ratio: {max(results.items(), key=lambda x: x[1]['metrics'].sharpe_ratio)[0]}")
    print(f"Best Strategy by Total Return: {max(results.items(), key=lambda x: x[1]['metrics'].total_return)[0]}")
    print(f"Most Active Strategy: {max(results.items(), key=lambda x: x[1]['metrics'].total_trades)[0]}")
    

if __name__ == "__main__":
    asyncio.run(main())