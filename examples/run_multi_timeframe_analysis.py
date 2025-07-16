"""
Multi-Timeframe SPX Strategy Analysis Script

This script runs backtests across multiple timeframes (1D, 1W, 1M) with various
parameter configurations and generates a comprehensive performance analysis report.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import StockDataFetcher
from src.indicators import RSI, BollingerBands, VWAP, VWMABands
from src.indicators.technical_indicators import MACD, ATR, StochasticOscillator
from src.indicators.supertrend_ai import SuperTrendAI
from src.backtesting import BacktestEngine
from src.strategies import BaseStrategy
from src.utils import PerformanceMetrics
from src.analysis.timeframe_performance_analyzer import (
    TimeframePerformanceAnalyzer, 
    TimeframeResult, 
    PerformanceMetrics as AnalysisMetrics
)


class MultiTimeframeStrategy(BaseStrategy):
    """Strategy that adapts parameters based on timeframe."""
    
    def __init__(self, timeframe: str, config: Dict[str, Any]):
        """Initialize strategy with timeframe-specific configuration."""
        super().__init__()
        self.timeframe = timeframe
        self.config = config
        
        # Adapt parameters based on timeframe
        self._adapt_parameters_for_timeframe()
        
        # Initialize indicators
        self.rsi = RSI(period=self.config['rsi_period'])
        self.bb = BollingerBands(
            period=self.config['bb_period'],
            num_std=self.config['bb_std']
        )
        self.macd = MACD(
            fast_period=self.config['macd_fast'],
            slow_period=self.config['macd_slow'],
            signal_period=self.config['macd_signal']
        )
        self.atr = ATR(period=self.config['atr_period'])
        
        # Initialize SuperTrend AI if enabled
        if self.config.get('use_supertrend', False):
            self.supertrend = SuperTrendAI(
                atr_length=self.config['st_atr_length'],
                factor_min=self.config['st_factor_min'],
                factor_max=self.config['st_factor_max']
            )
    
    def _adapt_parameters_for_timeframe(self):
        """Adapt strategy parameters based on timeframe."""
        # Base parameters
        base_config = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bb_period': 20,
            'bb_std': 2.0,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_period': 14,
            'stop_loss_atr': 2.0,
            'take_profit_atr': 3.0,
            'use_supertrend': False,
            'st_atr_length': 10,
            'st_factor_min': 1.0,
            'st_factor_max': 3.0
        }
        
        # Timeframe-specific adjustments
        if self.timeframe == '1M':
            # Monthly timeframe - longer periods, less sensitive
            multiplier = 2.5
            base_config['rsi_period'] = int(14 * multiplier)
            base_config['bb_period'] = int(20 * multiplier)
            base_config['macd_fast'] = int(12 * multiplier)
            base_config['macd_slow'] = int(26 * multiplier)
            base_config['atr_period'] = int(14 * multiplier)
            base_config['stop_loss_atr'] = 1.5  # Tighter stops for monthly
            base_config['take_profit_atr'] = 4.0  # Larger targets
            
        elif self.timeframe == '1W':
            # Weekly timeframe - moderate adjustments
            multiplier = 1.5
            base_config['rsi_period'] = int(14 * multiplier)
            base_config['bb_period'] = int(20 * multiplier)
            base_config['macd_fast'] = int(12 * multiplier)
            base_config['macd_slow'] = int(26 * multiplier)
            base_config['atr_period'] = int(14 * multiplier)
            base_config['stop_loss_atr'] = 1.75
            base_config['take_profit_atr'] = 3.5
            
        # Update config with base + user overrides
        base_config.update(self.config)
        self.config = base_config
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with indicators."""
        # Calculate indicators
        data['rsi'] = self.rsi.calculate(data)
        
        bb_data = self.bb.calculate(data)
        for col in bb_data.columns:
            data[col] = bb_data[col]
        
        macd_data = self.macd.calculate(data)
        for col in macd_data.columns:
            data[col] = macd_data[col]
        
        data['atr'] = self.atr.calculate(data)
        
        # Calculate SuperTrend if enabled
        if self.config.get('use_supertrend', False) and hasattr(self, 'supertrend'):
            st_data = self.supertrend.calculate(data)
            for col in st_data.columns:
                data[f'st_{col}'] = st_data[col]
        
        # Add price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on multiple indicators."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # RSI signals
        rsi_long = (data['rsi'] < self.config['rsi_oversold']).astype(int)
        rsi_short = (data['rsi'] > self.config['rsi_overbought']).astype(int)
        
        # Bollinger Bands signals
        bb_long = (data['close'] < data['lower_band']).astype(int)
        bb_short = (data['close'] > data['upper_band']).astype(int)
        
        # MACD signals
        macd_long = (
            (data['macd'] > data['signal_line']) & 
            (data['macd'].shift(1) <= data['signal_line'].shift(1))
        ).astype(int)
        
        macd_short = (
            (data['macd'] < data['signal_line']) & 
            (data['macd'].shift(1) >= data['signal_line'].shift(1))
        ).astype(int)
        
        # Combine signals with confluence
        long_signals = rsi_long + bb_long + macd_long
        short_signals = rsi_short + bb_short + macd_short
        
        # SuperTrend signals if available
        if 'st_long_entry' in data.columns:
            long_signals += data['st_long_entry'].astype(int)
            short_signals += data['st_short_entry'].astype(int)
        
        # Generate final signals based on confluence
        min_confluence = 2
        signals.loc[long_signals >= min_confluence, 'signal'] = 1
        signals.loc[short_signals >= min_confluence, 'signal'] = -1
        
        # Add stop loss and take profit levels
        signals['stop_loss'] = np.nan
        signals['take_profit'] = np.nan
        
        # Calculate levels for long positions
        long_mask = signals['signal'] == 1
        signals.loc[long_mask, 'stop_loss'] = (
            data.loc[long_mask, 'close'] - 
            self.config['stop_loss_atr'] * data.loc[long_mask, 'atr']
        )
        signals.loc[long_mask, 'take_profit'] = (
            data.loc[long_mask, 'close'] + 
            self.config['take_profit_atr'] * data.loc[long_mask, 'atr']
        )
        
        # Calculate levels for short positions  
        short_mask = signals['signal'] == -1
        signals.loc[short_mask, 'stop_loss'] = (
            data.loc[short_mask, 'close'] + 
            self.config['stop_loss_atr'] * data.loc[short_mask, 'atr']
        )
        signals.loc[short_mask, 'take_profit'] = (
            data.loc[short_mask, 'close'] - 
            self.config['take_profit_atr'] * data.loc[short_mask, 'atr']
        )
        
        return signals


def run_single_backtest(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single backtest with given parameters."""
    timeframe = params['timeframe']
    symbol = params['symbol']
    config = params['config']
    data_path = params['data_path']
    
    try:
        # Load data
        data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        
        # Initialize strategy
        strategy = MultiTimeframeStrategy(timeframe, config)
        
        # Prepare data
        data = strategy.prepare_data(data)
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Initialize backtest engine
        engine = BacktestEngine(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005
        )
        
        # Run backtest
        results = engine.run(data, signals)
        
        # Calculate performance metrics
        metrics = PerformanceMetrics(results['equity_curve'])
        
        # Create result object
        result = {
            'timeframe': timeframe,
            'symbol': symbol,
            'parameters': config,
            'metrics': {
                'total_return': metrics.total_return(),
                'annualized_return': metrics.annualized_return(),
                'sharpe_ratio': metrics.sharpe_ratio(),
                'sortino_ratio': metrics.sortino_ratio(),
                'max_drawdown': metrics.max_drawdown(),
                'calmar_ratio': metrics.calmar_ratio(),
                'win_rate': metrics.win_rate(),
                'profit_factor': metrics.profit_factor(),
                'volatility': metrics.annualized_volatility(),
                'var_95': metrics.value_at_risk(0.95),
                'cvar_95': metrics.conditional_value_at_risk(0.95),
                'total_trades': len(results.get('trades', [])),
                'avg_trade_duration': metrics.average_trade_duration() if hasattr(metrics, 'average_trade_duration') else None
            },
            'start_date': data.index[0].strftime('%Y-%m-%d'),
            'end_date': data.index[-1].strftime('%Y-%m-%d'),
            'equity_curve': results['equity_curve'].to_dict() if 'equity_curve' in results else None
        }
        
        return result
        
    except Exception as e:
        print(f"Error in backtest for {symbol} {timeframe}: {str(e)}")
        return None


def generate_parameter_grid() -> List[Dict[str, Any]]:
    """Generate parameter combinations to test."""
    param_grid = {
        'rsi_period': [10, 14, 20],
        'rsi_oversold': [20, 30],
        'rsi_overbought': [70, 80],
        'bb_period': [15, 20, 25],
        'bb_std': [1.5, 2.0, 2.5],
        'stop_loss_atr': [1.5, 2.0, 2.5],
        'take_profit_atr': [2.0, 3.0, 4.0],
        'use_supertrend': [False, True],
        'st_factor_min': [1.0, 1.5],
        'st_factor_max': [3.0, 4.0]
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    
    param_combinations = []
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        param_combinations.append(param_dict)
    
    # Limit to reasonable number for testing
    max_combinations = 50
    if len(param_combinations) > max_combinations:
        # Sample randomly
        indices = np.random.choice(len(param_combinations), max_combinations, replace=False)
        param_combinations = [param_combinations[i] for i in indices]
    
    return param_combinations


async def fetch_spx_data(timeframes: List[str], start_date: str, end_date: str) -> Dict[str, str]:
    """Fetch SPX (SPY) data for multiple timeframes."""
    fetcher = StockDataFetcher()
    data_paths = {}
    
    # SPX is tracked by SPY ETF
    symbol = 'SPY'
    
    for timeframe in timeframes:
        print(f"Fetching {symbol} data for {timeframe} timeframe...")
        
        # Check if data already exists
        data_path = Path(f"data/{symbol}_{timeframe}_{start_date}_{end_date}.csv")
        
        if data_path.exists():
            print(f"Data already exists: {data_path}")
            data_paths[timeframe] = str(data_path)
            continue
        
        try:
            # Map timeframe to yfinance interval
            interval_map = {
                '1D': '1d',
                '1W': '1wk', 
                '1M': '1mo'
            }
            
            interval = interval_map.get(timeframe, '1d')
            
            # Fetch data
            data = await fetcher.fetch_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            
            # Save data
            data.to_csv(data_path)
            data_paths[timeframe] = str(data_path)
            
            print(f"Saved {symbol} {timeframe} data to {data_path}")
            
        except Exception as e:
            print(f"Error fetching {symbol} {timeframe} data: {str(e)}")
    
    return data_paths


def main():
    """Main function to run multi-timeframe analysis."""
    # Configuration
    timeframes = ['1D', '1W', '1M']
    symbol = 'SPY'  # S&P 500 ETF
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    print("=" * 80)
    print("Multi-Timeframe SPX Strategy Analysis")
    print("=" * 80)
    
    # Create results directory
    results_dir = Path("backtest_results")
    results_dir.mkdir(exist_ok=True)
    
    # Fetch data for all timeframes
    print("\n1. Fetching historical data...")
    data_paths = asyncio.run(fetch_spx_data(timeframes, start_date, end_date))
    
    if not data_paths:
        print("Error: No data available for analysis")
        return
    
    # Generate parameter combinations
    print("\n2. Generating parameter combinations...")
    param_combinations = generate_parameter_grid()
    print(f"Testing {len(param_combinations)} parameter combinations across {len(timeframes)} timeframes")
    
    # Prepare backtest tasks
    backtest_tasks = []
    for timeframe, data_path in data_paths.items():
        for config in param_combinations:
            task = {
                'timeframe': timeframe,
                'symbol': symbol,
                'config': config.copy(),
                'data_path': data_path
            }
            backtest_tasks.append(task)
    
    print(f"\nTotal backtests to run: {len(backtest_tasks)}")
    
    # Run backtests in parallel
    print("\n3. Running backtests...")
    results = []
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_single_backtest, task): task 
            for task in backtest_tasks
        }
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            task = future_to_task[future]
            
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"Progress: {completed}/{len(backtest_tasks)} - "
                          f"{task['timeframe']} Sharpe: {result['metrics']['sharpe_ratio']:.3f}")
            except Exception as e:
                print(f"Error in task {task['timeframe']}: {str(e)}")
            
            # Print progress every 10 tasks
            if completed % 10 == 0:
                print(f"Completed {completed}/{len(backtest_tasks)} backtests...")
    
    print(f"\nSuccessfully completed {len(results)} backtests")
    
    # Save raw results
    results_file = results_dir / f"spx_multi_timeframe_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n4. Saved results to {results_file}")
    
    # Initialize analyzer
    print("\n5. Analyzing results...")
    analyzer = TimeframePerformanceAnalyzer()
    
    # Convert results to TimeframeResult objects
    for result in results:
        try:
            metrics = AnalysisMetrics(
                total_return=result['metrics']['total_return'],
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
                avg_trade_duration=result['metrics'].get('avg_trade_duration')
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
    
    # Run analyses
    timeframe_analysis = analyzer.analyze_by_timeframe()
    parameter_sensitivity = analyzer.analyze_parameter_sensitivity()
    robust_configs = analyzer.find_robust_configurations()
    
    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    print("\nBest Performance by Timeframe:")
    for tf, analysis in timeframe_analysis.items():
        print(f"\n{tf}:")
        print(f"  - Average Sharpe: {analysis['avg_sharpe']:.3f}")
        print(f"  - Average Return: {analysis['avg_return']:.2%}")
        print(f"  - Best Sharpe: {analysis['best_sharpe']:.3f}")
        if 'best_config' in analysis:
            print(f"  - Best Config: {analysis['best_config']['parameters']}")
    
    print(f"\nTop 5 Most Robust Configurations:")
    for i, config in enumerate(robust_configs[:5]):
        print(f"\n{i+1}. Average Sharpe: {config['avg_sharpe']:.3f}")
        print(f"   Parameters: {config['parameters']}")
        print(f"   Timeframes: {config['timeframe_count']}")
        print(f"   Worst Drawdown: {config['worst_drawdown']:.2%}")
    
    # Generate HTML report
    print("\n6. Generating HTML report...")
    report_path = Path("reports/spx_timeframe_analysis.html")
    analyzer.generate_html_report(report_path)
    
    print(f"\nAnalysis complete! Report saved to: {report_path}")
    print("\nTo view the report:")
    print(f"  - Open {report_path} in a web browser")
    print("  - Or run: python -m http.server 8000 --directory reports")
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()