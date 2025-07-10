"""
Execute all trading strategies with specified capital parameters.

This script runs:
1. Rolling VWAP strategy
2. Mean reversion strategy  
3. Momentum strategy

All with $10,000 initial capital and $500 monthly contributions.
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


class StrategyExecutor:
    """Execute multiple strategies with consistent parameters."""
    
    def __init__(self, initial_capital=10000, monthly_contribution=500):
        self.initial_capital = initial_capital
        self.monthly_contribution = monthly_contribution
        self.results = {}
        
    async def execute_rolling_vwap_strategy(self, data):
        """Execute rolling VWAP strategy."""
        print("\n" + "="*60)
        print("EXECUTING ROLLING VWAP STRATEGY")
        print("="*60)
        
        # Calculate indicators
        vwap = VWAP(window=20)  # 20-period rolling VWAP
        vwap_data = vwap.calculate(data)
        data = data.join(vwap_data[['vwap', 'vwap_upper', 'vwap_lower']])
        
        # Volume analysis
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Build strategy
        builder = StrategyBuilder("Rolling VWAP Strategy")
        builder.set_description("Buy when price crosses below VWAP with high volume, sell when crosses above")
        
        # Entry rules - Buy when price crosses below VWAP with high volume
        entry_rule = builder.add_entry_rule(
            "(close < vwap) and (close[1] >= vwap[1]) and (volume_ratio > 1.2)"
        )
        
        # Exit rules - Sell when price crosses above VWAP upper band or stop loss
        exit_rule = builder.add_exit_rule(
            "(close > vwap_upper) or (close < vwap * 0.97)"  # 3% stop loss
        )
        
        # Risk management
        builder.set_risk_management(
            stop_loss=0.03,  # 3% stop loss
            take_profit=0.06,  # 6% take profit
            max_positions=3
        )
        
        # Position sizing
        builder.set_position_sizing(
            method="kelly",
            size=0.25,  # 25% Kelly fraction for conservative sizing
            max_size=0.2  # Max 20% per position
        )
        
        strategy = builder.build()
        
        # Run backtest with monthly contributions
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission_rate=0.001,
            slippage_rate=0.0005,
            monthly_contribution=self.monthly_contribution
        )
        
        results = engine.run(data, strategy)
        
        # Calculate metrics
        metrics = PerformanceMetrics.calculate(
            results['equity_curve'],
            results['trades']
        )
        
        print("\nRolling VWAP Strategy Results:")
        print(metrics.generate_report())
        
        self.results['rolling_vwap'] = {
            'results': results,
            'metrics': metrics,
            'strategy_params': {
                'vwap_period': 20,
                'volume_threshold': 1.2,
                'stop_loss': 0.03,
                'take_profit': 0.06
            }
        }
        
        return results, metrics
        
    async def execute_mean_reversion_strategy(self, data):
        """Execute mean reversion strategy using RSI and Bollinger Bands."""
        print("\n" + "="*60)
        print("EXECUTING MEAN REVERSION STRATEGY")
        print("="*60)
        
        # Calculate indicators
        rsi = RSI(period=14)
        data['rsi'] = rsi.calculate(data)
        
        bb = BollingerBands(period=20, std_dev=2)
        bb_data = bb.calculate(data)
        data = data.join(bb_data)
        
        # ATR for dynamic stop loss
        data['atr'] = TechnicalIndicators.atr(
            data['high'], data['low'], data['close'], period=14
        )
        
        # Build strategy
        builder = StrategyBuilder("Mean Reversion Strategy")
        builder.set_description("Buy oversold conditions at lower BB, sell overbought at upper BB")
        
        # Entry rules - Buy when RSI oversold and price at lower BB
        entry_rule = builder.add_entry_rule(
            "(rsi < 30) and (close <= bb_lower * 1.01)"
        )
        
        # Exit rules - Sell when RSI overbought or price at upper BB
        exit_rule = builder.add_exit_rule(
            "(rsi > 70) or (close >= bb_upper * 0.99)"
        )
        
        # Risk management with dynamic ATR-based stops
        builder.set_risk_management(
            stop_loss=0.04,  # 4% stop loss
            take_profit=0.08,  # 8% take profit
            max_positions=4,
            use_atr_stops=True,
            atr_multiplier=2.0
        )
        
        # Position sizing
        builder.set_position_sizing(
            method="volatility",
            size=0.02,  # 2% risk per trade
            volatility_window=20
        )
        
        strategy = builder.build()
        
        # Run backtest
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission_rate=0.001,
            slippage_rate=0.0005,
            monthly_contribution=self.monthly_contribution
        )
        
        results = engine.run(data, strategy)
        
        # Calculate metrics
        metrics = PerformanceMetrics.calculate(
            results['equity_curve'],
            results['trades']
        )
        
        print("\nMean Reversion Strategy Results:")
        print(metrics.generate_report())
        
        self.results['mean_reversion'] = {
            'results': results,
            'metrics': metrics,
            'strategy_params': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'bb_period': 20,
                'bb_std': 2,
                'stop_loss': 0.04,
                'take_profit': 0.08
            }
        }
        
        return results, metrics
        
    async def execute_momentum_strategy(self, data):
        """Execute momentum strategy using MACD and ADX."""
        print("\n" + "="*60)
        print("EXECUTING MOMENTUM STRATEGY")
        print("="*60)
        
        # Calculate indicators
        macd_data = TechnicalIndicators.macd(
            data['close'], fast_period=12, slow_period=26, signal_period=9
        )
        data['macd'] = macd_data['macd']
        data['macd_signal'] = macd_data['signal']
        data['macd_histogram'] = macd_data['histogram']
        
        adx_data = TechnicalIndicators.adx(
            data['high'], data['low'], data['close'], period=14
        )
        data['adx'] = adx_data['adx']
        
        # Price momentum
        data['momentum_20'] = data['close'].pct_change(20)  # 20-day momentum
        data['momentum_50'] = data['close'].pct_change(50)  # 50-day momentum
        
        # Volume momentum
        data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
        data['volume_momentum'] = data['volume'] / data['volume_ma_20']
        
        # Build strategy
        builder = StrategyBuilder("Momentum Strategy")
        builder.set_description("Buy strong momentum with MACD confirmation and ADX strength")
        
        # Entry rules - Buy when strong momentum confirmed by MACD and ADX
        entry_rule = builder.add_entry_rule(
            "(macd > macd_signal) and (macd > 0) and " +
            "(adx > 25) and (momentum_20 > 0.05) and " + 
            "(volume_momentum > 1.1)"
        )
        
        # Exit rules - Exit when momentum weakens
        exit_rule = builder.add_exit_rule(
            "(macd < macd_signal) or (momentum_20 < -0.02) or (adx < 20)"
        )
        
        # Risk management
        builder.set_risk_management(
            stop_loss=0.05,  # 5% trailing stop
            take_profit=0.15,  # 15% take profit
            max_positions=3,
            trailing_stop=True,
            trailing_stop_distance=0.05
        )
        
        # Position sizing - larger positions for stronger momentum
        builder.set_position_sizing(
            method="momentum",
            size=0.15,  # Base 15% position
            momentum_factor=1.5  # Scale up to 1.5x for strong momentum
        )
        
        strategy = builder.build()
        
        # Run backtest
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission_rate=0.001,
            slippage_rate=0.0005,
            monthly_contribution=self.monthly_contribution
        )
        
        results = engine.run(data, strategy)
        
        # Calculate metrics
        metrics = PerformanceMetrics.calculate(
            results['equity_curve'],
            results['trades']
        )
        
        print("\nMomentum Strategy Results:")
        print(metrics.generate_report())
        
        self.results['momentum'] = {
            'results': results,
            'metrics': metrics,
            'strategy_params': {
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'adx_period': 14,
                'adx_threshold': 25,
                'momentum_period': 20,
                'momentum_threshold': 0.05,
                'stop_loss': 0.05,
                'take_profit': 0.15
            }
        }
        
        return results, metrics
        
    async def execute_all_strategies(self, symbol="SPY", period_years=5):
        """Execute all strategies and generate comprehensive report."""
        
        # Fetch data
        print(f"\nFetching {period_years} years of data for {symbol}...")
        fetcher = StockDataFetcher()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * period_years)
        
        data = await fetcher.fetch(
            symbol=symbol,
            start=start_date,
            end=end_date,
            interval="1d"
        )
        
        print(f"Fetched {len(data)} bars of data")
        
        # Execute strategies
        await self.execute_rolling_vwap_strategy(data.copy())
        await self.execute_mean_reversion_strategy(data.copy())
        await self.execute_momentum_strategy(data.copy())
        
        # Generate comparative report
        self.generate_comparative_report()
        
        # Save results
        self.save_results()
        
        return self.results
        
    def generate_comparative_report(self):
        """Generate comparative analysis of all strategies."""
        print("\n" + "="*60)
        print("COMPARATIVE STRATEGY ANALYSIS")
        print("="*60)
        
        comparison = pd.DataFrame()
        
        for strategy_name, data in self.results.items():
            metrics = data['metrics']
            comparison[strategy_name] = {
                'Total Return': f"{metrics.total_return:.2%}",
                'Annual Return': f"{metrics.cagr:.2%}",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'Win Rate': f"{metrics.win_rate:.2%}",
                'Profit Factor': f"{metrics.profit_factor:.2f}",
                'Total Trades': metrics.total_trades,
                'Avg Trade Return': f"{metrics.avg_return:.2%}",
                'Best Trade': f"{metrics.best_trade:.2%}",
                'Worst Trade': f"{metrics.worst_trade:.2%}",
                'Risk-Adjusted Return': f"{metrics.calmar_ratio:.2f}"
            }
            
        print(comparison.T.to_string())
        
        # Identify best strategy by different metrics
        print("\n" + "-"*40)
        print("BEST STRATEGY BY METRIC:")
        print("-"*40)
        
        # Convert percentage strings back to floats for comparison
        metrics_to_compare = {
            'Sharpe Ratio': lambda x: float(x.replace('%', '')),
            'Total Return': lambda x: float(x.replace('%', '')),
            'Win Rate': lambda x: float(x.replace('%', '')),
            'Max Drawdown': lambda x: -float(x.replace('%', ''))  # Lower is better
        }
        
        for metric, converter in metrics_to_compare.items():
            values = {}
            for strategy in comparison.columns:
                try:
                    values[strategy] = converter(comparison.loc[metric, strategy])
                except:
                    values[strategy] = 0
                    
            best_strategy = max(values, key=values.get)
            print(f"{metric}: {best_strategy} ({comparison.loc[metric, best_strategy]})")
            
    def save_results(self):
        """Save all results to files."""
        output_dir = "reports/strategy_execution"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics summary
        summary = {}
        for strategy_name, data in self.results.items():
            metrics = data['metrics']
            summary[strategy_name] = {
                'total_return': metrics.total_return,
                'annual_return': metrics.cagr,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'win_rate': metrics.win_rate,
                'total_trades': metrics.total_trades,
                'strategy_params': data['strategy_params']
            }
            
        with open(f"{output_dir}/strategy_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Generate visualizations
        chart_gen = ChartGenerator(style="plotly")
        dashboard = Dashboard()
        
        for strategy_name, data in self.results.items():
            # Save individual strategy results
            results = data['results']
            
            # Equity curve
            equity_fig = chart_gen.plot_equity_curve(results['equity_curve'])
            equity_fig.write_html(f"{output_dir}/{strategy_name}_equity_curve.html")
            
            # Dashboard
            dashboard_path = dashboard.create_dashboard(
                results,
                output_path=f"{output_dir}/{strategy_name}_dashboard.html"
            )
            
            # Save trades
            results['trades'].to_csv(f"{output_dir}/{strategy_name}_trades.csv")
            
        print(f"\nResults saved to {output_dir}/")
        

async def main():
    """Main execution function."""
    executor = StrategyExecutor(
        initial_capital=10000,
        monthly_contribution=500
    )
    
    # Execute all strategies
    results = await executor.execute_all_strategies(
        symbol="SPY",
        period_years=5
    )
    
    print("\n" + "="*60)
    print("ALL STRATEGIES EXECUTED SUCCESSFULLY")
    print("="*60)
    print(f"Initial Capital: ${executor.initial_capital:,}")
    print(f"Monthly Contribution: ${executor.monthly_contribution:,}")
    print(f"Total Strategies Tested: {len(results)}")
    

if __name__ == "__main__":
    asyncio.run(main())