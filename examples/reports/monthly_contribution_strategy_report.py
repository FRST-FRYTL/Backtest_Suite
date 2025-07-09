"""
Monthly Contribution Strategy Report Generator

Creates comprehensive report and visualizations for a $10,000 initial investment
with $500 monthly contributions using optimized RSI and Bollinger Bands strategy.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from src.data import StockDataFetcher
from src.indicators import RSI, BollingerBands, VWAP, TSV
from src.strategies import StrategyBuilder
from src.backtesting import BacktestEngine
from src.utils import PerformanceMetrics
from src.visualization import Dashboard, ChartGenerator


class MonthlyContributionReport:
    """Generate comprehensive report for monthly contribution strategy."""
    
    def __init__(self, initial_capital=10000, monthly_contribution=500):
        self.initial_capital = initial_capital
        self.monthly_contribution = monthly_contribution
        self.results = None
        self.metrics = None
        self.data = None
        
    async def run_backtest(self, symbol="SPY", years=5):
        """Run backtest with optimized parameters."""
        print(f"Running backtest for {symbol} with ${self.initial_capital:,} initial capital")
        print(f"Monthly contributions: ${self.monthly_contribution:,}")
        
        # Fetch data
        fetcher = StockDataFetcher()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        self.data = await fetcher.fetch(
            symbol=symbol,
            start=start_date,
            end=end_date,
            interval="1d"
        )
        
        # Calculate indicators with optimized parameters
        rsi = RSI(period=14)
        self.data['rsi'] = rsi.calculate(self.data)
        
        bb = BollingerBands(period=20, std_dev=2.0)
        bb_data = bb.calculate(self.data)
        self.data = self.data.join(bb_data)
        
        vwap = VWAP(window=None)
        vwap_data = vwap.calculate(self.data)
        self.data = self.data.join(vwap_data[['vwap']])
        
        tsv = TSV(period=13)
        self.data['tsv'] = tsv.calculate(self.data)
        
        # Build optimized strategy
        builder = StrategyBuilder("Monthly Contribution Strategy")
        builder.set_description("Optimized RSI + Bollinger Bands with monthly DCA")
        
        # Entry rules (optimized)
        builder.add_entry_rule("(rsi < 35 and Close < bb_lower) or (rsi < 30 and Close < vwap)")
        
        # Exit rules (optimized)
        builder.add_exit_rule("(rsi > 65 and Close > bb_upper) or (rsi > 70)")
        
        # Risk management (optimized)
        builder.set_risk_management(
            stop_loss=0.08,  # 8% stop loss
            take_profit=0.15,  # 15% take profit
            max_positions=5
        )
        
        # Position sizing (optimized for monthly contributions)
        builder.set_position_sizing(
            method="percent",
            size=0.2  # 20% of portfolio per position
        )
        
        strategy = builder.build()
        
        # Run backtest with monthly contributions
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission_rate=0.0005,  # $0.50 per trade
            slippage_rate=0.0001,    # 0.01% slippage
            monthly_contribution=self.monthly_contribution
        )
        
        self.results = engine.run(self.data, strategy)
        
        # Calculate metrics
        self.metrics = PerformanceMetrics.calculate(
            self.results['equity_curve'],
            self.results['trades']
        )
        
        return self.results, self.metrics
    
    def generate_executive_summary(self):
        """Generate executive summary of strategy performance."""
        total_invested = self.initial_capital + (self.monthly_contribution * 
                        len(self.results['equity_curve'].resample('M').last()))
        final_value = self.results['equity_curve']['total_value'].iloc[-1]
        total_return = (final_value - total_invested) / total_invested * 100
        
        summary = f"""
# MONTHLY CONTRIBUTION STRATEGY REPORT
## Executive Summary

### Investment Overview
- **Initial Capital**: ${self.initial_capital:,.2f}
- **Monthly Contribution**: ${self.monthly_contribution:,.2f}
- **Total Invested**: ${total_invested:,.2f}
- **Final Portfolio Value**: ${final_value:,.2f}
- **Total Return**: {total_return:.2f}%

### Key Performance Metrics
- **Annual Return**: {self.metrics.annual_return:.2f}%
- **Sharpe Ratio**: {self.metrics.sharpe_ratio:.2f}
- **Maximum Drawdown**: {self.metrics.max_drawdown:.2f}%
- **Win Rate**: {self.metrics.win_rate:.2f}%
- **Profit Factor**: {self.metrics.profit_factor:.2f}

### Trading Activity
- **Total Trades**: {self.metrics.total_trades}
- **Winning Trades**: {self.metrics.winning_trades}
- **Losing Trades**: {self.metrics.losing_trades}
- **Average Trade Duration**: {self.metrics.avg_trade_duration:.1f} days

### Risk Metrics
- **Value at Risk (95%)**: {self.metrics.var_95:.2f}%
- **Conditional VaR (95%)**: {self.metrics.cvar_95:.2f}%
- **Beta**: {self.metrics.beta:.2f}
- **Alpha**: {self.metrics.alpha:.2f}%
"""
        return summary
    
    def create_performance_visualization(self):
        """Create comprehensive performance visualization."""
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Portfolio Value Over Time', 'Monthly Returns Distribution',
                'Drawdown Analysis', 'Trade P&L Distribution',
                'Rolling Sharpe Ratio', 'Contribution vs Growth',
                'Win Rate by Month', 'Risk-Return Scatter'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Portfolio value with contributions
        equity = self.results['equity_curve']
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity['total_value'],
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add contribution line
        contributions = pd.Series(
            index=equity.index,
            data=self.initial_capital + np.arange(len(equity)) * 
                 (self.monthly_contribution / 20)  # Approximate daily contribution
        )
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=contributions,
                name='Total Contributions',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        # 2. Monthly returns distribution
        monthly_returns = equity['total_value'].resample('M').last().pct_change().dropna()
        fig.add_trace(
            go.Histogram(
                x=monthly_returns,
                nbinsx=30,
                name='Monthly Returns',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # 3. Drawdown analysis
        drawdown = (equity['total_value'] / equity['total_value'].cummax() - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        # 4. Trade P&L distribution
        if len(self.results['trades']) > 0:
            trade_pnl = [trade['pnl'] for trade in self.results['trades']]
            fig.add_trace(
                go.Histogram(
                    x=trade_pnl,
                    nbinsx=30,
                    name='Trade P&L',
                    marker_color='green'
                ),
                row=2, col=2
            )
        
        # 5. Rolling Sharpe ratio (252-day)
        returns = equity['total_value'].pct_change()
        rolling_sharpe = (returns.rolling(252).mean() / returns.rolling(252).std()) * np.sqrt(252)
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                name='Rolling Sharpe',
                line=dict(color='purple', width=2)
            ),
            row=3, col=1
        )
        
        # 6. Contribution vs Growth breakdown
        growth = equity['total_value'] - contributions
        contribution_pct = (contributions / equity['total_value']) * 100
        growth_pct = (growth / equity['total_value']) * 100
        
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=contribution_pct,
                name='Contribution %',
                stackgroup='one',
                line=dict(color='lightgray')
            ),
            row=3, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=growth_pct,
                name='Growth %',
                stackgroup='one',
                line=dict(color='green')
            ),
            row=3, col=2
        )
        
        # 7. Win rate by month
        if len(self.results['trades']) > 0:
            trades_df = pd.DataFrame(self.results['trades'])
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            trades_df['month'] = trades_df['exit_date'].dt.to_period('M')
            monthly_stats = trades_df.groupby('month').agg({
                'pnl': ['count', lambda x: (x > 0).sum()]
            })
            monthly_stats.columns = ['total', 'wins']
            monthly_stats['win_rate'] = (monthly_stats['wins'] / monthly_stats['total']) * 100
            
            fig.add_trace(
                go.Bar(
                    x=monthly_stats.index.astype(str),
                    y=monthly_stats['win_rate'],
                    name='Win Rate %',
                    marker_color='orange'
                ),
                row=4, col=1
            )
        
        # 8. Risk-Return scatter
        monthly_risk = monthly_returns.std() * np.sqrt(12) * 100
        annual_return = monthly_returns.mean() * 12 * 100
        
        fig.add_trace(
            go.Scatter(
                x=[monthly_risk],
                y=[annual_return],
                mode='markers+text',
                name='Strategy',
                marker=dict(size=20, color='red'),
                text=['Strategy'],
                textposition='top center'
            ),
            row=4, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1600,
            showlegend=False,
            title_text="Monthly Contribution Strategy Performance Analysis",
            title_font_size=20
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_xaxes(title_text="Monthly Return (%)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title_text="P&L ($)", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=2)
        fig.update_yaxes(title_text="Percentage (%)", row=3, col=2)
        fig.update_xaxes(title_text="Month", row=4, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=4, col=1)
        fig.update_xaxes(title_text="Annual Volatility (%)", row=4, col=2)
        fig.update_yaxes(title_text="Annual Return (%)", row=4, col=2)
        
        return fig
    
    def create_strategy_analysis(self):
        """Create detailed strategy analysis visualization."""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Entry/Exit Points on Price', 'RSI Strategy Signals',
                'Bollinger Bands Analysis', 'Volume Analysis',
                'Trade Duration Distribution', 'P&L by Entry Signal'
            ),
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Price with entry/exit points
        price_data = self.data.iloc[-252:]  # Last year
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data['Open'],
                high=price_data['High'],
                low=price_data['Low'],
                close=price_data['Close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Add trade markers
        if len(self.results['trades']) > 0:
            recent_trades = [t for t in self.results['trades'] 
                           if pd.to_datetime(t['entry_date']) >= price_data.index[0]]
            
            entry_dates = [pd.to_datetime(t['entry_date']) for t in recent_trades]
            entry_prices = [t['entry_price'] for t in recent_trades]
            exit_dates = [pd.to_datetime(t['exit_date']) for t in recent_trades]
            exit_prices = [t['exit_price'] for t in recent_trades]
            
            fig.add_trace(
                go.Scatter(
                    x=entry_dates,
                    y=entry_prices,
                    mode='markers',
                    name='Buy',
                    marker=dict(symbol='triangle-up', size=10, color='green')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=exit_dates,
                    y=exit_prices,
                    mode='markers',
                    name='Sell',
                    marker=dict(symbol='triangle-down', size=10, color='red')
                ),
                row=1, col=1
            )
        
        # 2. RSI with signals
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['rsi'],
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=1, col=2
        )
        
        # Add RSI levels
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=1, col=2)
        
        # 3. Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['Close'],
                name='Close',
                line=dict(color='black', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['bb_upper'],
                name='BB Upper',
                line=dict(color='red', width=1, dash='dash')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['bb_middle'],
                name='BB Middle',
                line=dict(color='blue', width=1, dash='dot')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['bb_lower'],
                name='BB Lower',
                line=dict(color='green', width=1, dash='dash')
            ),
            row=2, col=1
        )
        
        # 4. Volume analysis
        fig.add_trace(
            go.Bar(
                x=price_data.index,
                y=price_data['Volume'],
                name='Volume',
                marker_color='lightblue'
            ),
            row=2, col=2, secondary_y=False
        )
        
        # Add TSV on secondary axis
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['tsv'],
                name='TSV',
                line=dict(color='orange', width=2)
            ),
            row=2, col=2, secondary_y=True
        )
        
        # 5. Trade duration distribution
        if len(self.results['trades']) > 0:
            durations = [(pd.to_datetime(t['exit_date']) - 
                         pd.to_datetime(t['entry_date'])).days 
                        for t in self.results['trades']]
            
            fig.add_trace(
                go.Histogram(
                    x=durations,
                    nbinsx=20,
                    name='Trade Duration',
                    marker_color='purple'
                ),
                row=3, col=1
            )
        
        # 6. P&L by entry signal type
        if len(self.results['trades']) > 0:
            # Categorize trades by entry condition
            trades_df = pd.DataFrame(self.results['trades'])
            
            # Simple categorization based on RSI level at entry
            def categorize_trade(trade):
                entry_idx = self.data.index.get_loc(pd.to_datetime(trade['entry_date']), method='nearest')
                rsi_val = self.data.iloc[entry_idx]['rsi']
                if rsi_val < 25:
                    return 'Extreme Oversold'
                elif rsi_val < 30:
                    return 'Oversold'
                elif rsi_val < 35:
                    return 'Mild Oversold'
                else:
                    return 'Other'
            
            trades_df['signal_type'] = trades_df.apply(categorize_trade, axis=1)
            signal_stats = trades_df.groupby('signal_type')['pnl'].agg(['mean', 'count'])
            
            fig.add_trace(
                go.Bar(
                    x=signal_stats.index,
                    y=signal_stats['mean'],
                    name='Avg P&L by Signal',
                    marker_color='green',
                    text=signal_stats['count'],
                    textposition='outside'
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=False,
            title_text="Strategy Analysis - Entry/Exit Signals and Indicators",
            title_font_size=20
        )
        
        return fig
    
    def save_report(self, output_dir="examples/reports/output"):
        """Save complete report with all visualizations."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save executive summary
        summary = self.generate_executive_summary()
        with open(f"{output_dir}/executive_summary.md", 'w') as f:
            f.write(summary)
        
        # Save performance metrics
        metrics_df = pd.DataFrame([{
            'Metric': 'Total Return (%)',
            'Value': f"{((self.results['equity_curve']['total_value'].iloc[-1] - self.initial_capital) / self.initial_capital * 100):.2f}"
        }, {
            'Metric': 'Annual Return (%)',
            'Value': f"{self.metrics.annual_return:.2f}"
        }, {
            'Metric': 'Sharpe Ratio',
            'Value': f"{self.metrics.sharpe_ratio:.2f}"
        }, {
            'Metric': 'Max Drawdown (%)',
            'Value': f"{self.metrics.max_drawdown:.2f}"
        }, {
            'Metric': 'Win Rate (%)',
            'Value': f"{self.metrics.win_rate:.2f}"
        }, {
            'Metric': 'Profit Factor',
            'Value': f"{self.metrics.profit_factor:.2f}"
        }, {
            'Metric': 'Total Trades',
            'Value': f"{self.metrics.total_trades}"
        }])
        metrics_df.to_csv(f"{output_dir}/performance_metrics.csv", index=False)
        
        # Save visualizations
        perf_fig = self.create_performance_visualization()
        perf_fig.write_html(f"{output_dir}/performance_analysis.html")
        
        strategy_fig = self.create_strategy_analysis()
        strategy_fig.write_html(f"{output_dir}/strategy_analysis.html")
        
        # Create main dashboard
        dashboard = Dashboard()
        dashboard_path = dashboard.create_dashboard(
            self.results,
            output_path=f"{output_dir}/main_dashboard.html"
        )
        
        print(f"\nReport saved to {output_dir}/")
        print("Files created:")
        print("- executive_summary.md")
        print("- performance_metrics.csv")
        print("- performance_analysis.html")
        print("- strategy_analysis.html")
        print("- main_dashboard.html")
        
        return output_dir


async def main():
    """Generate comprehensive strategy report."""
    print("="*60)
    print("MONTHLY CONTRIBUTION STRATEGY REPORT GENERATOR")
    print("="*60)
    
    # Create report generator
    report = MonthlyContributionReport(
        initial_capital=10000,
        monthly_contribution=500
    )
    
    # Run backtest
    print("\nRunning 5-year backtest on SPY...")
    results, metrics = await report.run_backtest(symbol="SPY", years=5)
    
    # Generate and save report
    print("\nGenerating comprehensive report...")
    output_dir = report.save_report()
    
    print("\n" + "="*60)
    print("REPORT GENERATION COMPLETE")
    print("="*60)
    
    # Print quick summary
    print(report.generate_executive_summary())


if __name__ == "__main__":
    asyncio.run(main())