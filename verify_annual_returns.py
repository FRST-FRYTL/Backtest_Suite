#!/usr/bin/env python3
"""
Verify SuperTrend AI Annual Returns on Real SPX Data
Tests the claimed 18.5% annual return across multiple timeframes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

from src.strategies.supertrend_ai_strategy import SuperTrendAIStrategy, SuperTrendConfig
from src.indicators.technical_indicators import TechnicalIndicators

def load_spx_data(timeframe: str) -> pd.DataFrame:
    """Load SPX data for a specific timeframe"""
    spx_dir = Path("data/SPX")
    
    # Map timeframe to directory
    timeframe_map = {
        '1min': '1min',
        '5min': '5min', 
        '15min': '15min',
        '30min': '30min',
        '1H': '1H',
        '4H': '4H',
        'Daily': '1D'
    }
    
    tf_dir = spx_dir / timeframe_map[timeframe]
    
    # Find the latest file
    files = list(tf_dir.glob("SPY_*_latest.csv"))
    if not files:
        files = list(tf_dir.glob("SPY_*.csv"))
        files = [f for f in files if 'cache' not in str(f)]
    
    if not files:
        print(f"No data files found for {timeframe}")
        return None
        
    # Load the most recent file
    data_file = sorted(files)[-1]
    print(f"Loading {timeframe} data from: {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Rename columns to match expected format
    column_map = {
        'timestamp': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    
    # Try different column name variations
    for old, new in column_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)
        elif old.capitalize() in df.columns:
            df.rename(columns={old.capitalize(): new}, inplace=True)
    
    # Convert date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    elif 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
    
    # Sort by date
    df.sort_index(inplace=True)
    
    return df

def run_backtest(df: pd.DataFrame, timeframe: str, config: SuperTrendConfig) -> dict:
    """Run backtest on data and calculate annual return"""
    
    # Create strategy instance
    strategy = SuperTrendAIStrategy(config)
    
    # Generate signals
    print(f"Generating signals for {timeframe}...")
    signals_df = strategy.calculate(df)
    
    # Run simple backtest
    initial_capital = 100000
    capital = initial_capital
    position = 0
    position_price = 0
    trades = []
    equity_curve = []
    
    for i in range(len(signals_df)):
        current_price = signals_df['Close'].iloc[i]
        signal = signals_df['signal'].iloc[i]
        
        # Record equity
        if position > 0:
            current_equity = capital + (position * current_price)
        else:
            current_equity = capital
            
        equity_curve.append({
            'timestamp': signals_df.index[i],
            'equity': current_equity,
            'price': current_price
        })
        
        # Execute trades
        if signal == 1 and position == 0:
            # Buy signal
            shares = int(capital * 0.95 / current_price)  # Use 95% of capital
            if shares > 0:
                position = shares
                position_price = current_price
                capital -= shares * current_price
                trades.append({
                    'timestamp': signals_df.index[i],
                    'type': 'BUY',
                    'price': current_price,
                    'shares': shares
                })
                
        elif signal == -1 and position > 0:
            # Sell signal
            capital += position * current_price
            trades.append({
                'timestamp': signals_df.index[i],
                'type': 'SELL',
                'price': current_price,
                'shares': position,
                'profit': (current_price - position_price) * position
            })
            position = 0
            position_price = 0
    
    # Close final position if open
    if position > 0:
        final_price = signals_df['Close'].iloc[-1]
        capital += position * final_price
        trades.append({
            'timestamp': signals_df.index[-1],
            'type': 'SELL',
            'price': final_price,
            'shares': position,
            'profit': (final_price - position_price) * position
        })
    
    # Calculate metrics
    final_equity = capital
    total_return = (final_equity / initial_capital - 1) * 100
    
    # Calculate annualized return
    start_date = signals_df.index[0]
    end_date = signals_df.index[-1]
    years = (end_date - start_date).total_seconds() / (365.25 * 24 * 3600)
    
    if years > 0:
        annual_return = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
    else:
        annual_return = 0
    
    # Calculate other metrics
    equity_df = pd.DataFrame(equity_curve)
    equity_df.set_index('timestamp', inplace=True)
    
    # Daily returns
    daily_returns = equity_df['equity'].pct_change().dropna()
    
    # Sharpe ratio (assuming 252 trading days)
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    else:
        sharpe_ratio = 0
    
    # Max drawdown
    equity_df['cummax'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
    max_drawdown = equity_df['drawdown'].min() * 100
    
    # Buy and hold comparison
    buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    buy_hold_annual = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    return {
        'timeframe': timeframe,
        'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        'years': years,
        'initial_capital': initial_capital,
        'final_equity': final_equity,
        'total_return': total_return,
        'annual_return': annual_return,
        'buy_hold_return': buy_hold_return,
        'buy_hold_annual': buy_hold_annual,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': len(trades),
        'win_rate': sum(1 for t in trades if t.get('profit', 0) > 0) / len(trades) * 100 if trades else 0
    }

def main():
    """Main verification function"""
    print("=" * 80)
    print("SuperTrend AI Annual Return Verification")
    print("Testing claimed 18.5% annual return on real SPX data")
    print("=" * 80)
    
    # Configuration (using default parameters from the strategy)
    config = SuperTrendConfig(
        atr_length=10,
        min_factor=1.0,
        max_factor=5.0,
        factor_step=0.5,
        performance_alpha=10.0,
        cluster_selection='Best',
        use_signal_strength=True,
        min_signal_strength=4,
        use_stop_loss=True,
        stop_loss_type='ATR',
        stop_loss_atr_mult=2.0,
        use_take_profit=True,
        take_profit_type='Risk/Reward',
        risk_reward_ratio=2.0
    )
    
    # Timeframes to test
    timeframes = ['1min', '5min', '15min', '30min', '1H', '4H', 'Daily']
    
    results = []
    
    for timeframe in timeframes:
        print(f"\n{'='*60}")
        print(f"Testing {timeframe} timeframe...")
        print(f"{'='*60}")
        
        # Load data
        df = load_spx_data(timeframe)
        if df is None or df.empty:
            print(f"Skipping {timeframe} - no data available")
            continue
            
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        try:
            # Run backtest
            result = run_backtest(df, timeframe, config)
            results.append(result)
            
            # Print results
            print(f"\nResults for {timeframe}:")
            print(f"  Annual Return: {result['annual_return']:.2f}%")
            print(f"  Total Return: {result['total_return']:.2f}%")
            print(f"  Buy & Hold Annual: {result['buy_hold_annual']:.2f}%")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {result['max_drawdown']:.2f}%")
            print(f"  Number of Trades: {result['num_trades']}")
            print(f"  Win Rate: {result['win_rate']:.1f}%")
            
        except Exception as e:
            print(f"Error testing {timeframe}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary report
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Claimed Annual Return: 18.5%")
    print(f"\nActual Results by Timeframe:")
    print(f"{'Timeframe':<15} {'Annual Return':<15} {'vs Claimed':<15} {'vs Buy&Hold':<15}")
    print("-" * 60)
    
    for result in results:
        diff_from_claimed = result['annual_return'] - 18.5
        diff_from_bh = result['annual_return'] - result['buy_hold_annual']
        print(f"{result['timeframe']:<15} {result['annual_return']:>12.2f}% {diff_from_claimed:>12.2f}% {diff_from_bh:>12.2f}%")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"reports/supertrend_ai_verification_{timestamp}.json"
    
    verification_report = {
        'timestamp': timestamp,
        'claimed_annual_return': 18.5,
        'configuration': config.__dict__,
        'results': results,
        'summary': {
            'average_annual_return': np.mean([r['annual_return'] for r in results]),
            'median_annual_return': np.median([r['annual_return'] for r in results]),
            'best_timeframe': max(results, key=lambda x: x['annual_return'])['timeframe'],
            'worst_timeframe': min(results, key=lambda x: x['annual_return'])['timeframe']
        }
    }
    
    # Save JSON report
    os.makedirs('reports', exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(verification_report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Store results in memory for other agents
    print("\nStoring verification results in memory...")
    # This would be where we'd call the memory storage hook
    
    print("\nVerification completed!")

if __name__ == "__main__":
    main()