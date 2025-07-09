#!/usr/bin/env python3
"""
Stress Test Scenarios
====================

Tests the monthly contribution strategy under various stress conditions
including market crashes, high volatility, and different contribution amounts.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

from src.backtesting.engine import BacktestEngine
from src.strategies.builder import StrategyBuilder
from src.utils.metrics import calculate_sharpe, calculate_max_drawdown
from src.data.fetcher import DataFetcher
from src.monitoring.collectors import MetricsCollector


class StressTestStrategy:
    """Monthly contribution strategy with stress test variations."""
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 monthly_contribution: float = 500,
                 contribution_multiplier_on_crash: float = 2.0,
                 crash_threshold: float = -0.10,
                 volatility_adjustment: bool = True):
        """
        Initialize stress test strategy.
        
        Args:
            initial_capital: Starting portfolio value
            monthly_contribution: Base monthly contribution amount
            contribution_multiplier_on_crash: Multiply contribution during crashes
            crash_threshold: Market drop % to trigger increased contributions
            volatility_adjustment: Adjust position sizing based on volatility
        """
        self.initial_capital = initial_capital
        self.base_contribution = monthly_contribution
        self.monthly_contribution = monthly_contribution
        self.contribution_multiplier = contribution_multiplier_on_crash
        self.crash_threshold = crash_threshold
        self.volatility_adjustment = volatility_adjustment
        
        self.target_weights = {
            'SPY': 0.30,
            'QQQ': 0.25,
            'IWM': 0.15,
            'GLD': 0.15,
            'TLT': 0.15
        }
        
        self.last_contribution_date = None
        self.market_conditions = {'volatility': 'normal', 'trend': 'neutral'}
        
    def generate_signals(self, data: pd.DataFrame, portfolio) -> Dict[str, float]:
        """Generate signals with stress test adjustments."""
        signals = {}
        current_date = data.index[-1]
        
        # Analyze market conditions
        self._analyze_market_conditions(data)
        
        # Adjust contribution based on market stress
        self._adjust_contribution_amount(data)
        
        # Check if it's time for monthly contribution
        if self._is_contribution_day(current_date):
            print(f"📅 Contribution on {current_date.strftime('%Y-%m-%d')}: ${self.monthly_contribution:.2f}")
            signals = self._calculate_contribution_signals(data, portfolio)
            self.last_contribution_date = current_date
            
        return signals
    
    def _analyze_market_conditions(self, data: pd.DataFrame):
        """Analyze current market conditions for stress indicators."""
        if len(data) < 20:
            return
            
        # Calculate recent market performance
        spy_returns = data['SPY'].pct_change().dropna()
        recent_return = (data['SPY'].iloc[-1] / data['SPY'].iloc[-20] - 1)
        
        # Calculate volatility
        recent_volatility = spy_returns.tail(20).std() * np.sqrt(252)
        
        # Detect market crash
        if recent_return < self.crash_threshold:
            self.market_conditions['trend'] = 'crash'
            print(f"🔴 Market crash detected! Recent return: {recent_return:.2%}")
        elif recent_return < -0.05:
            self.market_conditions['trend'] = 'correction'
        elif recent_return > 0.10:
            self.market_conditions['trend'] = 'rally'
        else:
            self.market_conditions['trend'] = 'normal'
            
        # Categorize volatility
        if recent_volatility > 0.30:
            self.market_conditions['volatility'] = 'high'
        elif recent_volatility > 0.20:
            self.market_conditions['volatility'] = 'elevated'
        else:
            self.market_conditions['volatility'] = 'normal'
    
    def _adjust_contribution_amount(self, data: pd.DataFrame):
        """Adjust contribution amount based on market stress."""
        # Reset to base amount
        self.monthly_contribution = self.base_contribution
        
        # Increase contributions during crashes
        if self.market_conditions['trend'] == 'crash':
            self.monthly_contribution *= self.contribution_multiplier
            print(f"💰 Increased contribution to ${self.monthly_contribution:.2f} due to crash")
        elif self.market_conditions['trend'] == 'correction':
            self.monthly_contribution *= 1.5
            
        # Adjust for volatility if enabled
        if self.volatility_adjustment:
            if self.market_conditions['volatility'] == 'high':
                # Reduce equity allocation during high volatility
                self.target_weights['SPY'] = 0.20
                self.target_weights['QQQ'] = 0.15
                self.target_weights['IWM'] = 0.10
                self.target_weights['GLD'] = 0.25
                self.target_weights['TLT'] = 0.30
            else:
                # Reset to normal allocation
                self.target_weights = {
                    'SPY': 0.30,
                    'QQQ': 0.25,
                    'IWM': 0.15,
                    'GLD': 0.15,
                    'TLT': 0.15
                }
    
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
        
        # Add contribution to cash
        portfolio.cash += self.monthly_contribution
        
        # Calculate target dollar amounts
        new_total = total_value + self.monthly_contribution
        
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


def run_stress_test(scenario_name: str,
                   start_date: str,
                   end_date: str,
                   monthly_contribution: float = 500,
                   crash_multiplier: float = 2.0,
                   volatility_adjustment: bool = True) -> Dict:
    """Run a specific stress test scenario."""
    print(f"\n{'='*60}")
    print(f"Running Stress Test: {scenario_name}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Base Contribution: ${monthly_contribution}/month")
    print(f"Crash Multiplier: {crash_multiplier}x")
    print(f"Volatility Adjustment: {volatility_adjustment}")
    print(f"{'='*60}")
    
    # Fetch data
    symbols = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
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
    
    # Combine data
    combined_data = pd.DataFrame()
    for symbol, df in data.items():
        combined_data[symbol] = df['Close']
    combined_data = combined_data.dropna()
    
    # Initialize strategy
    strategy = StressTestStrategy(
        initial_capital=10000,
        monthly_contribution=monthly_contribution,
        contribution_multiplier_on_crash=crash_multiplier,
        crash_threshold=-0.10,
        volatility_adjustment=volatility_adjustment
    )
    
    # Run backtest
    engine = BacktestEngine(
        data=combined_data,
        strategy=strategy,
        initial_capital=10000,
        commission=0.001
    )
    
    results = engine.run()
    
    # Calculate metrics
    portfolio_values = results['portfolio_value']
    returns = portfolio_values.pct_change().dropna()
    
    # Identify crash periods
    spy_returns = combined_data['SPY'].pct_change()
    rolling_returns = spy_returns.rolling(20).sum()
    crash_periods = rolling_returns[rolling_returns < -0.10].index
    
    # Calculate recovery metrics
    drawdowns = []
    running_max = portfolio_values.expanding().max()
    dd = (portfolio_values - running_max) / running_max
    
    # Find all drawdown periods
    in_drawdown = False
    current_dd = {'start': None, 'trough': None, 'end': None, 'depth': 0}
    
    for date, dd_value in dd.items():
        if dd_value < -0.05 and not in_drawdown:
            in_drawdown = True
            current_dd = {'start': date, 'trough': date, 'depth': dd_value}
        elif in_drawdown:
            if dd_value < current_dd['depth']:
                current_dd['trough'] = date
                current_dd['depth'] = dd_value
            elif dd_value >= -0.01:  # Recovered
                current_dd['end'] = date
                drawdowns.append(current_dd)
                in_drawdown = False
    
    # Calculate recovery times
    recovery_times = []
    for dd_info in drawdowns:
        if dd_info['end'] and dd_info['trough']:
            recovery_days = (dd_info['end'] - dd_info['trough']).days
            recovery_times.append(recovery_days)
    
    avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
    
    # Calculate metrics
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
    years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
    cagr = (((portfolio_values.iloc[-1] / 10000) ** (1/years)) - 1) * 100 if years > 0 else 0
    sharpe = calculate_sharpe(returns, periods_per_year=252)
    max_dd, max_dd_duration = calculate_max_drawdown(portfolio_values)
    
    # Calculate crash period performance
    crash_performance = {}
    if len(crash_periods) > 0:
        for period in crash_periods:
            if period in portfolio_values.index:
                # Find performance during and after crash
                crash_start = period
                crash_end_idx = min(portfolio_values.index.get_loc(period) + 60, 
                                  len(portfolio_values) - 1)
                crash_end = portfolio_values.index[crash_end_idx]
                
                crash_return = (portfolio_values.loc[crash_end] / 
                              portfolio_values.loc[crash_start] - 1) * 100
                crash_performance[crash_start] = crash_return
    
    results_dict = {
        'scenario': scenario_name,
        'portfolio_values': portfolio_values,
        'returns': returns,
        'total_return': total_return,
        'cagr': cagr,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'drawdown_periods': len(drawdowns),
        'avg_recovery_days': avg_recovery_time,
        'crash_periods': len(crash_periods),
        'crash_performance': crash_performance,
        'final_value': portfolio_values.iloc[-1]
    }
    
    # Print summary
    print(f"\n📊 {scenario_name} Results:")
    print(f"Final Value: ${portfolio_values.iloc[-1]:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"CAGR: {cagr:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Number of Drawdowns (>5%): {len(drawdowns)}")
    print(f"Avg Recovery Time: {avg_recovery_time:.0f} days")
    print(f"Crash Periods Detected: {len(crash_periods)}")
    
    return results_dict


def create_stress_test_visualizations(all_results: Dict[str, Dict], output_dir: str):
    """Create visualizations for stress test results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Strategy Comparison During Crashes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot portfolio values
    for scenario_name, results in all_results.items():
        portfolio_values = results['portfolio_values']
        ax1.plot(portfolio_values.index, portfolio_values.values, 
                label=scenario_name, linewidth=2)
    
    ax1.set_title('Stress Test Strategy Comparison', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot drawdowns
    for scenario_name, results in all_results.items():
        portfolio_values = results['portfolio_values']
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max * 100
        ax2.plot(drawdown.index, drawdown.values, label=scenario_name, linewidth=2)
    
    ax2.set_title('Drawdown Comparison', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(top=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stress_test_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Recovery Analysis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scenarios = []
    avg_recovery_times = []
    max_drawdowns = []
    
    for scenario_name, results in all_results.items():
        scenarios.append(scenario_name)
        avg_recovery_times.append(results['avg_recovery_days'])
        max_drawdowns.append(abs(results['max_drawdown']))
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x - width/2, avg_recovery_times, width, 
                   label='Avg Recovery Days', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, max_drawdowns, width, 
                    label='Max Drawdown %', color='red', alpha=0.7)
    
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Recovery Days', fontsize=12, color='blue')
    ax2.set_ylabel('Max Drawdown (%)', fontsize=12, color='red')
    ax.set_title('Recovery Analysis by Scenario', fontsize=16, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add value labels
    for bar, value in zip(bars1, avg_recovery_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.0f}',
               ha='center', va='bottom')
    
    for bar, value in zip(bars2, max_drawdowns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.1f}%',
               ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recovery_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Contribution Strategy Effectiveness
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate metrics for bar chart
    scenarios = []
    total_returns = []
    sharpe_ratios = []
    
    for scenario_name, results in all_results.items():
        scenarios.append(scenario_name.replace(' Contribution', '\nContribution'))
        total_returns.append(results['total_return'])
        sharpe_ratios.append(results['sharpe_ratio'])
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x - width/2, total_returns, width, 
                   label='Total Return %', color='green', alpha=0.7)
    bars2 = ax2.bar(x + width/2, sharpe_ratios, width, 
                    label='Sharpe Ratio', color='orange', alpha=0.7)
    
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Total Return (%)', fontsize=12, color='green')
    ax2.set_ylabel('Sharpe Ratio', fontsize=12, color='orange')
    ax.set_title('Strategy Performance Comparison', fontsize=16, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.tick_params(axis='y', labelcolor='green')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Add value labels
    for bar, value in zip(bars1, total_returns):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.1f}%',
               ha='center', va='bottom')
    
    for bar, value in zip(bars2, sharpe_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.2f}',
               ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'contribution_effectiveness.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Stress test visualizations saved to {output_dir}")


def main():
    """Run comprehensive stress test scenarios."""
    
    # Test period with known market stress events
    test_period = ('2018-01-01', '2023-12-31')
    
    # Define stress test scenarios
    scenarios = [
        {
            'name': 'Base Strategy ($500/month)',
            'monthly_contribution': 500,
            'crash_multiplier': 1.0,  # No increase during crashes
            'volatility_adjustment': False
        },
        {
            'name': 'Crash Buyer ($500 base, 2x on crash)',
            'monthly_contribution': 500,
            'crash_multiplier': 2.0,  # Double contributions during crashes
            'volatility_adjustment': False
        },
        {
            'name': 'Volatility Adjusted ($500/month)',
            'monthly_contribution': 500,
            'crash_multiplier': 1.0,
            'volatility_adjustment': True  # Adjust allocation based on volatility
        },
        {
            'name': 'Aggressive ($1000 base, 3x on crash)',
            'monthly_contribution': 1000,
            'crash_multiplier': 3.0,
            'volatility_adjustment': True
        },
        {
            'name': 'Conservative ($250/month)',
            'monthly_contribution': 250,
            'crash_multiplier': 1.5,
            'volatility_adjustment': True
        }
    ]
    
    all_results = {}
    
    # Run each scenario
    for scenario in scenarios:
        results = run_stress_test(
            scenario_name=scenario['name'],
            start_date=test_period[0],
            end_date=test_period[1],
            monthly_contribution=scenario['monthly_contribution'],
            crash_multiplier=scenario['crash_multiplier'],
            volatility_adjustment=scenario['volatility_adjustment']
        )
        
        if results:
            all_results[scenario['name']] = results
    
    # Create visualizations
    if all_results:
        output_dir = '/workspaces/Backtest_Suite/examples/backtests/results'
        create_stress_test_visualizations(all_results, output_dir)
        
        # Summary report
        print("\n" + "="*60)
        print("STRESS TEST SUMMARY REPORT")
        print("="*60)
        
        print("\nScenario Performance Comparison:")
        print(f"{'Scenario':<35} {'Return':>10} {'CAGR':>10} {'Sharpe':>10} {'Max DD':>10} {'Recovery':>10}")
        print("-" * 85)
        
        for scenario_name, results in all_results.items():
            print(f"{scenario_name:<35} "
                  f"{results['total_return']:>9.1f}% "
                  f"{results['cagr']:>9.1f}% "
                  f"{results['sharpe_ratio']:>10.2f} "
                  f"{results['max_drawdown']:>9.1f}% "
                  f"{results['avg_recovery_days']:>9.0f}d")
        
        # Find best performers
        best_return = max(all_results.items(), key=lambda x: x[1]['total_return'])
        best_sharpe = max(all_results.items(), key=lambda x: x[1]['sharpe_ratio'])
        best_drawdown = min(all_results.items(), key=lambda x: abs(x[1]['max_drawdown']))
        
        print(f"\n🏆 Best Total Return: {best_return[0]} ({best_return[1]['total_return']:.1f}%)")
        print(f"🏆 Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.2f})")
        print(f"🏆 Smallest Drawdown: {best_drawdown[0]} ({best_drawdown[1]['max_drawdown']:.1f}%)")
    
    else:
        print("\n❌ No stress test results to display")


if __name__ == "__main__":
    main()