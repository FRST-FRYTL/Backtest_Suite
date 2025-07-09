#!/usr/bin/env python3
"""
Example: Monthly Contribution Strategy Backtest

This example demonstrates how to use the Monthly Contribution Strategy
with $10,000 initial capital and $500 monthly contributions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml

from src.strategies import MonthlyContributionStrategy
from src.backtesting.engine import BacktestEngine
from src.data.fetcher import DataFetcher
from src.indicators import RSI, BollingerBands, VWAP, FearGreedIndex
from src.utils.metrics import calculate_metrics
from src.visualization.charts import plot_backtest_results


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def prepare_data_with_indicators(symbol: str, start_date: str, end_date: str):
    """
    Fetch data and calculate all required indicators.
    
    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with price data and indicators
    """
    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    
    # Fetch price data
    fetcher = DataFetcher()
    data = await fetcher.fetch_data(symbol, start_date, end_date)
    
    # Calculate RSI
    rsi = RSI(period=14)
    data['rsi'] = rsi.calculate(data)
    
    # Calculate Bollinger Bands
    bb = BollingerBands(period=20, std_dev=2)
    bb_data = bb.calculate(data)
    data['bb_upper'] = bb_data['upper']
    data['bb_middle'] = bb_data['middle']
    data['bb_lower'] = bb_data['lower']
    data['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
    
    # Calculate VWAP
    vwap = VWAP()
    vwap_data = vwap.calculate(data)
    data['vwap'] = vwap_data['vwap']
    data['vwap_upper'] = vwap_data['upper_band']
    data['vwap_lower'] = vwap_data['lower_band']
    data['vwap_bands_width'] = (vwap_data['upper_band'] - vwap_data['lower_band']) / vwap_data['vwap']
    
    # Calculate moving averages
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['sma_50'] = data['close'].rolling(window=50).mean()
    data['sma_200'] = data['close'].rolling(window=200).mean()
    data['volume_sma_20'] = data['volume'].rolling(window=20).mean()
    
    # Calculate ATR for volatility
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift(1))
    low_close = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['atr'] = true_range.rolling(window=14).mean()
    data['atr_pct'] = data['atr'] / data['close']
    
    # Dollar volume
    data['dollar_volume'] = data['close'] * data['volume']
    
    # Fetch Fear & Greed Index
    fg = FearGreedIndex()
    try:
        fg_data = await fg.fetch_historical(limit=365*5)  # 5 years of data
        # Align with price data
        data = data.join(fg_data[['value']], how='left')
        data.rename(columns={'value': 'fear_greed'}, inplace=True)
        data['fear_greed'] = data['fear_greed'].fillna(method='ffill')
    except Exception as e:
        logger.warning(f"Could not fetch Fear & Greed Index: {e}")
        # Use neutral value if unavailable
        data['fear_greed'] = 50
    
    return data


def run_backtest_with_contributions(data: pd.DataFrame, strategy_config: dict):
    """
    Run backtest with monthly contributions.
    
    Args:
        data: Price data with indicators
        strategy_config: Strategy configuration
        
    Returns:
        Backtest results
    """
    # Initialize strategy
    strategy = MonthlyContributionStrategy(
        initial_capital=strategy_config['account']['initial_capital'],
        monthly_contribution=strategy_config['account']['monthly_contribution'],
        cash_reserve_target=strategy_config['account']['cash_reserve_target'],
        use_fear_greed=True
    )
    
    # Build strategy rules
    trading_strategy = strategy.build_strategy()
    
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_capital=strategy_config['account']['initial_capital'],
        commission=strategy_config['backtest']['commission'],
        slippage=strategy_config['backtest']['slippage']
    )
    
    # Track monthly contributions
    contribution_day = strategy_config['account']['contribution_day']
    last_contribution_month = None
    
    # Run backtest day by day
    for i, (date, row) in enumerate(data.iterrows()):
        # Process monthly contribution
        if date.day == contribution_day and date.month != last_contribution_month:
            contribution = strategy.process_monthly_contribution(
                date,
                engine.get_account_value(),
                engine.get_cash_balance()
            )
            engine.add_cash(contribution['investment_allocation'])
            last_contribution_month = date.month
            logger.info(f"Monthly contribution on {date}: ${contribution['amount']}")
        
        # Generate signals
        signals = trading_strategy.generate_signals(data.iloc[:i+1])
        
        # Process signals
        if signals.iloc[-1]['entry']:
            # Calculate position size
            win_rate = engine.get_win_rate()
            avg_win, avg_loss = engine.get_average_win_loss()
            volatility = row['atr_pct']
            
            shares, position_value = strategy.calculate_position_size(
                engine.get_account_value(),
                row['close'],
                win_rate,
                avg_win,
                avg_loss,
                volatility
            )
            
            if shares > 0:
                engine.enter_position(
                    date=date,
                    symbol=data.attrs.get('symbol', 'UNKNOWN'),
                    shares=shares,
                    price=row['close'],
                    stop_loss=signals.iloc[-1]['stop_loss'],
                    take_profit=signals.iloc[-1]['take_profit']
                )
        
        elif signals.iloc[-1]['exit']:
            engine.exit_all_positions(date, row['close'])
        
        # Update trailing stops
        engine.update_trailing_stops(row['high'])
        
        # Check for rebalancing
        if i % 63 == 0:  # Quarterly (approximately 63 trading days)
            positions = engine.get_open_positions()
            if strategy.should_rebalance(positions, engine.get_account_value(), date):
                rebalance_orders = strategy.get_rebalancing_orders(
                    positions,
                    engine.get_account_value()
                )
                for order in rebalance_orders:
                    engine.process_rebalance_order(order)
    
    # Get final results
    results = engine.get_results()
    results['strategy_metrics'] = strategy.analyze_performance()
    
    return results


def analyze_results(results: dict, benchmark_data: pd.DataFrame = None):
    """
    Analyze and display backtest results.
    
    Args:
        results: Backtest results
        benchmark_data: Optional benchmark data for comparison
    """
    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS - Monthly Contribution Strategy")
    logger.info("="*60)
    
    # Account summary
    logger.info(f"\nAccount Summary:")
    logger.info(f"Initial Capital: ${results['initial_capital']:,.2f}")
    logger.info(f"Total Contributions: ${results['strategy_metrics']['total_contributions']:,.2f}")
    logger.info(f"Final Value: ${results['final_value']:,.2f}")
    logger.info(f"Total Return: {results['total_return']:.2%}")
    logger.info(f"Annualized Return: {results['annualized_return']:.2%}")
    
    # Risk metrics
    logger.info(f"\nRisk Metrics:")
    logger.info(f"Max Drawdown: {results['strategy_metrics']['max_drawdown']:.2%}")
    logger.info(f"Sharpe Ratio: {results['strategy_metrics']['sharpe_ratio']:.2f}")
    logger.info(f"Win Rate: {results['strategy_metrics']['win_rate']:.2%}")
    logger.info(f"Profit Factor: {results['strategy_metrics']['profit_factor']:.2f}")
    
    # Trading statistics
    logger.info(f"\nTrading Statistics:")
    logger.info(f"Total Trades: {results['strategy_metrics']['total_trades']}")
    logger.info(f"Average Return per Trade: {results['strategy_metrics']['avg_return']:.2%}")
    logger.info(f"Average Win: {results['strategy_metrics']['avg_win']:.2%}")
    logger.info(f"Average Loss: {results['strategy_metrics']['avg_loss']:.2%}")
    logger.info(f"Average Trade Duration: {results['strategy_metrics']['avg_trade_duration']:.1f} days")
    
    # Contribution analysis
    logger.info(f"\nContribution Analysis:")
    logger.info(f"Number of Contributions: {results['strategy_metrics']['contribution_count']}")
    logger.info(f"Average Monthly Contribution: ${results['strategy_metrics']['total_contributions']/results['strategy_metrics']['contribution_count']:,.2f}")
    
    # Compare with buy-and-hold
    if benchmark_data is not None:
        benchmark_return = (benchmark_data['close'].iloc[-1] / benchmark_data['close'].iloc[0]) - 1
        logger.info(f"\nBenchmark Comparison (Buy & Hold):")
        logger.info(f"Strategy Return: {results['total_return']:.2%}")
        logger.info(f"Benchmark Return: {benchmark_return:.2%}")
        logger.info(f"Excess Return: {results['total_return'] - benchmark_return:.2%}")


async def main():
    """Main function to run the example."""
    # Load configuration
    config_path = "examples/strategies/monthly_contribution_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set parameters
    symbol = "SPY"  # S&P 500 ETF
    start_date = config['backtest']['start_date']
    end_date = config['backtest']['end_date']
    
    # Prepare data
    data = await prepare_data_with_indicators(symbol, start_date, end_date)
    data.attrs['symbol'] = symbol
    
    # Run backtest
    logger.info("Starting backtest...")
    results = run_backtest_with_contributions(data, config)
    
    # Analyze results
    analyze_results(results, data)
    
    # Plot results
    logger.info("\nGenerating performance charts...")
    plot_backtest_results(results, save_path="examples/monthly_contribution_results.png")
    
    # Monte Carlo simulation
    if config['backtest']['monte_carlo']['enabled']:
        logger.info("\nRunning Monte Carlo simulation...")
        mc_results = run_monte_carlo_simulation(
            data,
            config,
            simulations=config['backtest']['monte_carlo']['simulations']
        )
        logger.info(f"95% Confidence Interval for Annual Return: {mc_results['ci_low']:.2%} - {mc_results['ci_high']:.2%}")


def run_monte_carlo_simulation(data: pd.DataFrame, config: dict, simulations: int = 1000):
    """
    Run Monte Carlo simulation for strategy robustness testing.
    
    Args:
        data: Historical data
        config: Strategy configuration
        simulations: Number of simulations
        
    Returns:
        Monte Carlo results
    """
    returns = []
    
    for i in range(simulations):
        # Randomly sample returns with replacement
        sampled_data = data.sample(n=len(data), replace=True).sort_index()
        
        # Run backtest on sampled data
        results = run_backtest_with_contributions(sampled_data, config)
        returns.append(results['annualized_return'])
    
    # Calculate statistics
    returns_array = np.array(returns)
    
    return {
        'mean_return': np.mean(returns_array),
        'std_return': np.std(returns_array),
        'ci_low': np.percentile(returns_array, 2.5),
        'ci_high': np.percentile(returns_array, 97.5),
        'returns': returns_array
    }


if __name__ == "__main__":
    asyncio.run(main())