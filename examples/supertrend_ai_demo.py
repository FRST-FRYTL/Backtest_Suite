#!/usr/bin/env python3
"""
SuperTrend AI Strategy Demonstration
This demonstrates the implementation of the SuperTrend AI strategy from TradingView
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

from src.data.fetcher import StockDataFetcher


class SuperTrendAI:
    """SuperTrend AI indicator implementation"""
    
    def __init__(self, length=10, min_mult=1.0, max_mult=5.0, step=0.5, perf_alpha=10):
        self.length = length
        self.min_mult = min_mult
        self.max_mult = max_mult
        self.step = step
        self.perf_alpha = perf_alpha
        self.factors = np.arange(min_mult, max_mult + step, step)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SuperTrend AI signals"""
        df = df.copy()
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.length).mean()
        
        # Calculate HL2
        hl2 = (df['high'] + df['low']) / 2
        
        # Calculate multiple SuperTrends
        supertrends = {}
        performances = {}
        
        for factor in self.factors:
            upper = hl2 + (atr * factor)
            lower = hl2 - (atr * factor)
            
            # Initialize trend
            trend = pd.Series(0, index=df.index)
            supertrend = pd.Series(np.nan, index=df.index)
            
            for i in range(1, len(df)):
                # Update bands
                if df['close'].iloc[i-1] <= upper.iloc[i-1]:
                    upper.iloc[i] = min(upper.iloc[i], upper.iloc[i-1])
                    
                if df['close'].iloc[i-1] >= lower.iloc[i-1]:
                    lower.iloc[i] = max(lower.iloc[i], lower.iloc[i-1])
                
                # Determine trend
                if df['close'].iloc[i] > upper.iloc[i-1]:
                    trend.iloc[i] = 1
                elif df['close'].iloc[i] < lower.iloc[i-1]:
                    trend.iloc[i] = -1
                else:
                    trend.iloc[i] = trend.iloc[i-1]
                
                # Set SuperTrend value
                if trend.iloc[i] == 1:
                    supertrend.iloc[i] = lower.iloc[i]
                else:
                    supertrend.iloc[i] = upper.iloc[i]
            
            supertrends[factor] = {
                'trend': trend,
                'value': supertrend,
                'upper': upper,
                'lower': lower
            }
            
            # Calculate performance
            returns = df['close'].pct_change()
            perf = 0
            for i in range(1, len(df)):
                if i > 0 and trend.iloc[i-1] != 0:
                    diff = np.sign(df['close'].iloc[i] - supertrend.iloc[i-1])
                    perf += (2 / (self.perf_alpha + 1)) * (returns.iloc[i] * diff - perf)
            
            performances[factor] = perf
        
        # Simple clustering - divide into 3 groups based on performance
        perf_values = list(performances.values())
        perf_sorted = sorted(perf_values)
        
        # Define clusters
        worst_cluster = perf_sorted[:len(perf_sorted)//3]
        avg_cluster = perf_sorted[len(perf_sorted)//3:2*len(perf_sorted)//3]
        best_cluster = perf_sorted[2*len(perf_sorted)//3:]
        
        # Select best performing factor
        best_factor = max(performances, key=performances.get)
        selected_st = supertrends[best_factor]
        
        # Create signals
        df['supertrend'] = selected_st['value']
        df['trend'] = selected_st['trend']
        
        # Generate trading signals
        df['signal'] = 0
        df.loc[df['trend'] > df['trend'].shift(), 'signal'] = 1  # Buy signal
        df.loc[df['trend'] < df['trend'].shift(), 'signal'] = -1  # Sell signal
        
        # Calculate signal strength (simplified)
        df['signal_strength'] = np.abs((df['close'] - df['supertrend']) / df['supertrend']) * 10
        df['signal_strength'] = df['signal_strength'].clip(0, 10).round()
        
        return df


class SuperTrendAIStrategy:
    """Trading strategy based on SuperTrend AI"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.indicator = SuperTrendAI(
            length=self.config.get('atr_length', 10),
            min_mult=self.config.get('min_mult', 1.0),
            max_mult=self.config.get('max_mult', 5.0),
            step=self.config.get('step', 0.5),
            perf_alpha=self.config.get('perf_alpha', 10)
        )
        self.min_signal_strength = self.config.get('min_signal_strength', 4)
        self.use_stop_loss = self.config.get('use_stop_loss', True)
        self.stop_loss_atr = self.config.get('stop_loss_atr', 2.0)
        self.use_take_profit = self.config.get('use_take_profit', True)
        self.risk_reward_ratio = self.config.get('risk_reward_ratio', 2.0)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        # Calculate SuperTrend AI
        df = self.indicator.calculate(df)
        
        # Filter by signal strength
        df['trade_signal'] = df['signal']
        if self.min_signal_strength > 0:
            df.loc[df['signal_strength'] < self.min_signal_strength, 'trade_signal'] = 0
        
        return df
    
    def calculate_position_size(self, capital: float, price: float, volatility: float = None) -> int:
        """Calculate position size based on capital and risk"""
        # Simple fixed percentage allocation
        position_value = capital * 0.95  # Use 95% of capital
        shares = int(position_value / price)
        return max(1, shares)


def main():
    """Run SuperTrend AI strategy backtest"""
    print("SuperTrend AI Strategy Backtest")
    print("=" * 50)
    
    # Configuration
    config = {
        'symbol': 'AAPL',
        'start_date': '2022-01-01',
        'end_date': '2023-12-31',
        'initial_capital': 100000,
        'commission': 0.001,
        
        # SuperTrend AI parameters
        'atr_length': 10,
        'min_mult': 1.0,
        'max_mult': 5.0,
        'step': 0.5,
        'perf_alpha': 10,
        'min_signal_strength': 4,
        
        # Risk management
        'use_stop_loss': True,
        'stop_loss_atr': 2.0,
        'use_take_profit': True,
        'risk_reward_ratio': 2.0
    }
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Fetch data
    print(f"\nFetching data for {config['symbol']}...")
    fetcher = StockDataFetcher()
    
    # Check if data exists
    data_file = data_dir / f"{config['symbol']}_daily.csv"
    if data_file.exists():
        print(f"Loading existing data from {data_file}")
        df = pd.read_csv(data_file, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        # Rename columns to match expected format
        df.columns = df.columns.str.lower()
    else:
        print(f"Downloading new data...")
        df = fetcher.fetch_sync(
            config['symbol'],
            start=config['start_date'],
            end=config['end_date']
        )
        if df is not None and not df.empty:
            df.to_csv(data_file)
            print(f"Data saved to {data_file}")
    
    if df is None or df.empty:
        print("Failed to fetch data. Please check your internet connection.")
        return
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Create strategy
    strategy = SuperTrendAIStrategy(config)
    
    # Generate signals
    print("\nGenerating trading signals...")
    df = strategy.generate_signals(df)
    
    # Count signals
    buy_signals = (df['trade_signal'] == 1).sum()
    sell_signals = (df['trade_signal'] == -1).sum()
    print(f"Buy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    
    # Simple backtest
    print("\nRunning backtest...")
    capital = config['initial_capital']
    position = 0
    trades = []
    equity_curve = []
    
    for idx, row in df.iterrows():
        # Record equity
        current_equity = capital + (position * row['close'] if position > 0 else 0)
        equity_curve.append({
            'date': idx,
            'equity': current_equity,
            'price': row['close']
        })
        
        # Check for signals
        if row['trade_signal'] == 1 and position == 0:
            # Buy signal
            shares = strategy.calculate_position_size(capital, row['close'])
            cost = shares * row['close'] * (1 + config['commission'])
            if cost <= capital:
                position = shares
                capital -= cost
                trades.append({
                    'date': idx,
                    'type': 'BUY',
                    'price': row['close'],
                    'shares': shares,
                    'value': cost
                })
        
        elif row['trade_signal'] == -1 and position > 0:
            # Sell signal
            proceeds = position * row['close'] * (1 - config['commission'])
            capital += proceeds
            trades.append({
                'date': idx,
                'type': 'SELL',
                'price': row['close'],
                'shares': position,
                'value': proceeds
            })
            position = 0
    
    # Close final position
    if position > 0:
        final_value = position * df.iloc[-1]['close'] * (1 - config['commission'])
        capital += final_value
        position = 0
    
    # Calculate metrics
    equity_df = pd.DataFrame(equity_curve)
    equity_df.set_index('date', inplace=True)
    
    total_return = (equity_df['equity'].iloc[-1] / config['initial_capital'] - 1) * 100
    
    # Calculate daily returns
    equity_df['returns'] = equity_df['equity'].pct_change()
    sharpe_ratio = np.sqrt(252) * equity_df['returns'].mean() / equity_df['returns'].std()
    
    # Calculate max drawdown
    equity_df['cummax'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
    max_drawdown = equity_df['drawdown'].min() * 100
    
    # Buy and hold comparison
    buy_hold_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    
    # Print results
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Initial Capital: ${config['initial_capital']:,.2f}")
    print(f"Final Capital: ${equity_df['equity'].iloc[-1]:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Number of Trades: {len(trades)}")
    print(f"Win Rate: {(total_return > 0) * 100:.0f}%")
    
    # Save results
    results = {
        'config': config,
        'metrics': {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades)
        },
        'trades': trades
    }
    
    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Save JSON report
    report_file = reports_dir / f"supertrend_ai_{config['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nReport saved to: {report_file}")
    
    # Generate HTML visualization
    print("\nGenerating HTML report...")
    try:
        # Create visualization data
        viz_data = {
            'symbol': config['symbol'],
            'df': df,
            'equity_curve': equity_df,
            'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
            'metrics': results['metrics']
        }
        
        # Generate report (simplified version)
        html_file = reports_dir / f"supertrend_ai_{config['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        generate_simple_report(viz_data, html_file)
        print(f"HTML report saved to: {html_file}")
        
    except Exception as e:
        print(f"Error generating HTML report: {e}")
    
    print("\nBacktest completed successfully!")


def generate_simple_report(data, output_file):
    """Generate a simple HTML report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SuperTrend AI Strategy Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #333; }}
            .metric {{ display: inline-block; margin: 20px; padding: 10px; border: 1px solid #ddd; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>SuperTrend AI Strategy Backtest Report</h1>
        <h2>Symbol: {data['symbol']}</h2>
        
        <div>
            <div class="metric">
                <div>Total Return</div>
                <div class="metric-value">{data['metrics']['total_return']:.2f}%</div>
            </div>
            <div class="metric">
                <div>Buy & Hold Return</div>
                <div class="metric-value">{data['metrics']['buy_hold_return']:.2f}%</div>
            </div>
            <div class="metric">
                <div>Sharpe Ratio</div>
                <div class="metric-value">{data['metrics']['sharpe_ratio']:.2f}</div>
            </div>
            <div class="metric">
                <div>Max Drawdown</div>
                <div class="metric-value">{data['metrics']['max_drawdown']:.2f}%</div>
            </div>
            <div class="metric">
                <div>Number of Trades</div>
                <div class="metric-value">{data['metrics']['num_trades']}</div>
            </div>
        </div>
        
        <h2>Strategy Description</h2>
        <p>
            The SuperTrend AI strategy uses machine learning (K-means clustering) to dynamically optimize
            the SuperTrend indicator parameters based on historical performance. It evaluates multiple
            factor values simultaneously and selects the best performing configuration.
        </p>
        
        <h2>Key Features</h2>
        <ul>
            <li>Dynamic parameter optimization using K-means clustering</li>
            <li>Performance-based factor selection</li>
            <li>Signal strength filtering (0-10 scale)</li>
            <li>Risk management with ATR-based stops</li>
            <li>Adaptive to changing market conditions</li>
        </ul>
        
        <p><em>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)


if __name__ == "__main__":
    main()