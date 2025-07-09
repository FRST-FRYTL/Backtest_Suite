#!/usr/bin/env python3
"""
Monthly Contribution Backtest
============================

Tests a dollar-cost averaging strategy with monthly contributions
across different market periods.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from src.backtesting.engine import BacktestEngine
from src.strategies.builder import StrategyBuilder
from src.utils.metrics import calculate_sharpe, calculate_max_drawdown
from src.data.fetcher import DataFetcher
from src.monitoring.collectors import MetricsCollector
from src.monitoring.dashboard import Dashboard


class MonthlyContributionStrategy:
    """Dollar-cost averaging strategy with monthly contributions."""
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 monthly_contribution: float = 500,
                 rebalance_threshold: float = 0.10):
        """
        Initialize monthly contribution strategy.
        
        Args:
            initial_capital: Starting portfolio value
            monthly_contribution: Amount to invest each month
            rebalance_threshold: Rebalance when allocation deviates by this %
        """
        self.initial_capital = initial_capital
        self.monthly_contribution = monthly_contribution
        self.rebalance_threshold = rebalance_threshold
        self.target_weights = {
            'SPY': 0.30,  # S&P 500
            'QQQ': 0.25,  # NASDAQ 100
            'IWM': 0.15,  # Russell 2000
            'GLD': 0.15,  # Gold
            'TLT': 0.15   # Long-term Treasuries
        }
        self.last_contribution_date = None
        
    def generate_signals(self, data: pd.DataFrame, portfolio) -> Dict[str, float]:
        """Generate buy/sell signals based on monthly contributions and rebalancing."""
        signals = {}
        current_date = data.index[-1]
        
        # Check if it's time for monthly contribution
        if self._is_contribution_day(current_date):
            print(f"📅 Monthly contribution on {current_date.strftime('%Y-%m-%d')}")
            signals = self._calculate_contribution_signals(data, portfolio)
            self.last_contribution_date = current_date
            
        # Check if rebalancing is needed
        elif self._needs_rebalancing(portfolio):
            print(f"⚖️ Rebalancing portfolio on {current_date.strftime('%Y-%m-%d')}")
            signals = self._calculate_rebalancing_signals(data, portfolio)
            
        return signals
    
    def _is_contribution_day(self, current_date: pd.Timestamp) -> bool:
        """Check if it's the first trading day of the month."""
        if self.last_contribution_date is None:
            return True
            
        return (current_date.month != self.last_contribution_date.month or
                current_date.year != self.last_contribution_date.year)
    
    def _calculate_contribution_signals(self, data: pd.DataFrame, portfolio) -> Dict[str, float]:
        """Calculate how to allocate monthly contribution."""
        signals = {}
        total_value = portfolio.get_total_value(data)
        contribution_amount = self.monthly_contribution
        
        # Add contribution to cash
        portfolio.cash += contribution_amount
        
        # Calculate target dollar amounts
        new_total = total_value + contribution_amount
        
        for symbol, target_weight in self.target_weights.items():
            if symbol in data.columns:
                current_price = data[symbol].iloc[-1]
                current_position = portfolio.positions.get(symbol, 0)
                current_value = current_position * current_price
                
                target_value = new_total * target_weight
                value_difference = target_value - current_value
                
                if value_difference > 0:
                    shares_to_buy = int(value_difference / current_price)
                    if shares_to_buy > 0:
                        signals[symbol] = shares_to_buy
                        
        return signals
    
    def _needs_rebalancing(self, portfolio) -> bool:
        """Check if portfolio needs rebalancing."""
        if not portfolio.positions:
            return False
            
        total_value = portfolio.get_total_value()
        if total_value == 0:
            return False
            
        for symbol, target_weight in self.target_weights.items():
            current_weight = portfolio.get_position_weight(symbol)
            if abs(current_weight - target_weight) > self.rebalance_threshold:
                return True
                
        return False
    
    def _calculate_rebalancing_signals(self, data: pd.DataFrame, portfolio) -> Dict[str, float]:
        """Calculate rebalancing trades."""
        signals = {}
        total_value = portfolio.get_total_value(data)
        
        for symbol, target_weight in self.target_weights.items():
            if symbol in data.columns:
                current_price = data[symbol].iloc[-1]
                current_position = portfolio.positions.get(symbol, 0)
                current_value = current_position * current_price
                
                target_value = total_value * target_weight
                value_difference = target_value - current_value
                
                shares_difference = int(value_difference / current_price)
                if shares_difference != 0:
                    signals[symbol] = shares_difference
                    
        return signals


def run_backtest_period(symbols: List[str], 
                       start_date: str, 
                       end_date: str,
                       period_name: str) -> Dict:
    """Run backtest for a specific time period."""
    print(f"\n{'='*60}")
    print(f"Running backtest for {period_name}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}")
    
    # Fetch data
    fetcher = DataFetcher()
    data = {}
    
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        df = fetcher.fetch_data(symbol, start_date, end_date)
        if df is not None and not df.empty:
            data[symbol] = df
    
    if not data:
        print("❌ No data available for the specified period")
        return {}
    
    # Combine data into single DataFrame
    combined_data = pd.DataFrame()
    for symbol, df in data.items():
        combined_data[symbol] = df['Close']
    
    combined_data = combined_data.dropna()
    
    # Initialize strategy and engine
    strategy = MonthlyContributionStrategy(
        initial_capital=10000,
        monthly_contribution=500,
        rebalance_threshold=0.10
    )
    
    engine = BacktestEngine(
        data=combined_data,
        strategy=strategy,
        initial_capital=10000,
        commission=0.001  # 0.1% commission
    )
    
    # Run backtest
    results = engine.run()
    
    # Calculate additional metrics
    portfolio_values = results['portfolio_value']
    returns = portfolio_values.pct_change().dropna()
    
    # Calculate metrics
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
    
    # Calculate total contributions
    months = len(portfolio_values) / 21  # Approximate trading days per month
    total_contributions = 10000 + (500 * months)
    
    # Calculate CAGR
    years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
    cagr = (((portfolio_values.iloc[-1] / 10000) ** (1/years)) - 1) * 100 if years > 0 else 0
    
    # Other metrics
    sharpe = calculate_sharpe(returns, periods_per_year=252)
    max_dd, max_dd_duration = calculate_max_drawdown(portfolio_values)
    
    # Win rate
    winning_days = (returns > 0).sum()
    total_days = len(returns)
    win_rate = (winning_days / total_days * 100) if total_days > 0 else 0
    
    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else np.inf
    
    metrics = {
        'period': period_name,
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': 10000,
        'final_value': portfolio_values.iloc[-1],
        'total_contributions': total_contributions,
        'total_return': total_return,
        'cagr': cagr,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'max_dd_days': max_dd_duration,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'portfolio_values': portfolio_values,
        'returns': returns,
        'trades': results.get('trades', [])
    }
    
    # Print summary
    print(f"\n📊 {period_name} Results:")
    print(f"Initial Capital: ${10000:,.2f}")
    print(f"Total Contributions: ${total_contributions:,.2f}")
    print(f"Final Value: ${portfolio_values.iloc[-1]:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"CAGR: {cagr:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    return metrics


def create_visualizations(all_results: Dict[str, Dict], output_dir: str):
    """Create comprehensive visualizations for backtest results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_results)))
    
    # 1. Equity Curves Comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for (period_name, results), color in zip(all_results.items(), colors):
        portfolio_values = results['portfolio_values']
        ax.plot(portfolio_values.index, portfolio_values.values, 
                label=period_name, linewidth=2, color=color)
        
        # Mark contribution points (monthly)
        contribution_dates = pd.date_range(
            start=portfolio_values.index[0],
            end=portfolio_values.index[-1],
            freq='MS'
        )
        
        # Filter to only show contribution marks that are in the data
        contribution_dates = [d for d in contribution_dates if d in portfolio_values.index]
        if contribution_dates:
            contribution_values = portfolio_values.loc[contribution_dates]
            ax.scatter(contribution_dates, contribution_values, 
                      alpha=0.3, s=20, color=color, marker='o')
    
    ax.set_title('Portfolio Value Over Time - All Periods', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_curves_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Drawdown Chart
    fig, axes = plt.subplots(len(all_results), 1, figsize=(14, 4*len(all_results)), sharex=True)
    if len(all_results) == 1:
        axes = [axes]
    
    for idx, ((period_name, results), ax) in enumerate(zip(all_results.items(), axes)):
        portfolio_values = results['portfolio_values']
        
        # Calculate drawdown
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max * 100
        
        ax.fill_between(drawdown.index, drawdown.values, 0, 
                       color='red', alpha=0.3, label='Drawdown')
        ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
        
        ax.set_title(f'Drawdown - {period_name}', fontsize=14)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(top=0)
        
        # Mark max drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        ax.annotate(f'Max DD: {max_dd_value:.1f}%',
                   xy=(max_dd_idx, max_dd_value),
                   xytext=(10, -20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    axes[-1].set_xlabel('Date', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drawdown_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Monthly Returns Heatmap
    for period_name, results in all_results.items():
        returns = results['returns']
        portfolio_values = results['portfolio_values']
        
        # Calculate monthly returns
        monthly_returns = portfolio_values.resample('M').last().pct_change().dropna()
        
        # Create pivot table for heatmap
        monthly_returns_df = pd.DataFrame(monthly_returns)
        monthly_returns_df['Year'] = monthly_returns_df.index.year
        monthly_returns_df['Month'] = monthly_returns_df.index.month
        
        pivot_table = monthly_returns_df.pivot_table(
            values=0, index='Year', columns='Month', aggfunc='mean'
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Replace month numbers with names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table.columns = [month_names[m-1] for m in pivot_table.columns]
        
        sns.heatmap(pivot_table * 100, annot=True, fmt='.1f', 
                   cmap='RdYlGn', center=0, 
                   cbar_kws={'label': 'Monthly Return (%)'},
                   ax=ax)
        
        ax.set_title(f'Monthly Returns Heatmap - {period_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        plt.tight_layout()
        safe_period_name = period_name.replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(os.path.join(output_dir, f'monthly_returns_{safe_period_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Performance Metrics Comparison
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    metrics_to_plot = [
        ('total_return', 'Total Return (%)', 'green'),
        ('cagr', 'CAGR (%)', 'blue'),
        ('sharpe_ratio', 'Sharpe Ratio', 'orange'),
        ('max_drawdown', 'Max Drawdown (%)', 'red'),
        ('win_rate', 'Win Rate (%)', 'purple'),
        ('profit_factor', 'Profit Factor', 'brown')
    ]
    
    for idx, (metric, label, color) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        periods = list(all_results.keys())
        values = [results[metric] for results in all_results.values()]
        
        bars = ax.bar(periods, values, color=color, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}',
                   ha='center', va='bottom')
        
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x labels if needed
        if len(periods) > 3:
            ax.set_xticklabels(periods, rotation=45, ha='right')
    
    plt.suptitle('Performance Metrics Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Visualizations saved to {output_dir}")


def main():
    """Run comprehensive backtests across multiple time periods."""
    symbols = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
    
    # Define test periods
    test_periods = [
        ('2010-01-01', '2015-12-31', 'Recovery Period (2010-2015)'),
        ('2015-01-01', '2020-12-31', 'Bull Market (2015-2020)'),
        ('2020-01-01', '2023-12-31', 'Volatility Period (2020-2023)'),
        ('2018-01-01', '2023-12-31', 'Full Cycle (2018-2023)')
    ]
    
    all_results = {}
    
    # Run backtests for each period
    for start_date, end_date, period_name in test_periods:
        results = run_backtest_period(symbols, start_date, end_date, period_name)
        if results:
            all_results[period_name] = results
    
    # Create visualizations
    if all_results:
        output_dir = '/workspaces/Backtest_Suite/examples/backtests/results'
        create_visualizations(all_results, output_dir)
        
        # Summary report
        print("\n" + "="*60)
        print("BACKTEST SUMMARY REPORT")
        print("="*60)
        
        for period_name, results in all_results.items():
            print(f"\n{period_name}:")
            print(f"  Total Return: {results['total_return']:.2f}%")
            print(f"  CAGR: {results['cagr']:.2f}%")
            print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
            print(f"  Final Value: ${results['final_value']:,.2f}")
    
    else:
        print("\n❌ No backtest results to display")


if __name__ == "__main__":
    main()