"""Chart generation for backtesting visualization."""

from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns


class ChartGenerator:
    """Generate various charts for backtesting results."""
    
    def __init__(self, style: str = "plotly"):
        """
        Initialize chart generator.
        
        Args:
            style: Chart library to use ('plotly' or 'matplotlib')
        """
        self.style = style
        
        if style == "matplotlib":
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")
            
    def plot_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
        title: str = "Portfolio Equity Curve"
    ) -> Union[go.Figure, plt.Figure]:
        """
        Plot portfolio equity curve.
        
        Args:
            equity_curve: DataFrame with portfolio values
            benchmark: Optional benchmark series
            title: Chart title
            
        Returns:
            Plotly figure or matplotlib figure
        """
        if self.style == "plotly":
            return self._plot_equity_curve_plotly(equity_curve, benchmark, title)
        else:
            return self._plot_equity_curve_matplotlib(equity_curve, benchmark, title)
            
    def _plot_equity_curve_plotly(
        self,
        equity_curve: pd.DataFrame,
        benchmark: Optional[pd.Series],
        title: str
    ) -> go.Figure:
        """Create equity curve with Plotly."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=('Equity Curve', 'Drawdown %')
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve['total_value'],
                mode='lines',
                name='Portfolio',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add benchmark if provided
        if benchmark is not None:
            # Normalize to start at same value
            benchmark_normalized = benchmark / benchmark.iloc[0] * equity_curve['total_value'].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=benchmark.index,
                    y=benchmark_normalized,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='gray', width=1, dash='dash')
                ),
                row=1, col=1
            )
            
        # Calculate and plot drawdown
        running_max = equity_curve['total_value'].expanding().max()
        drawdown = (equity_curve['total_value'] - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            hovermode='x unified',
            showlegend=True,
            height=700
        )
        
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        
        return fig
        
    def _plot_equity_curve_matplotlib(
        self,
        equity_curve: pd.DataFrame,
        benchmark: Optional[pd.Series],
        title: str
    ) -> plt.Figure:
        """Create equity curve with matplotlib."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Equity curve
        ax1.plot(equity_curve.index, equity_curve['total_value'], 
                label='Portfolio', linewidth=2, color='blue')
        
        if benchmark is not None:
            benchmark_normalized = benchmark / benchmark.iloc[0] * equity_curve['total_value'].iloc[0]
            ax1.plot(benchmark.index, benchmark_normalized,
                    label='Benchmark', linewidth=1, linestyle='--', color='gray')
            
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        running_max = equity_curve['total_value'].expanding().max()
        drawdown = (equity_curve['total_value'] - running_max) / running_max * 100
        
        ax2.fill_between(equity_curve.index, 0, drawdown, 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.plot(equity_curve.index, drawdown, color='red', linewidth=1)
        ax2.set_ylabel('Drawdown %')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    def plot_returns_distribution(
        self,
        returns: pd.Series,
        title: str = "Returns Distribution"
    ) -> Union[go.Figure, plt.Figure]:
        """
        Plot returns distribution histogram.
        
        Args:
            returns: Series of returns
            title: Chart title
            
        Returns:
            Figure object
        """
        if self.style == "plotly":
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=returns * 100,  # Convert to percentage
                nbinsx=50,
                name='Returns',
                marker_color='blue',
                opacity=0.7
            ))
            
            # Add normal distribution overlay
            mean = returns.mean() * 100
            std = returns.std() * 100
            x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
            y = stats.norm.pdf(x, mean, std) * len(returns) * (returns.max() - returns.min()) * 100 / 50
            
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Returns (%)",
                yaxis_title="Frequency",
                showlegend=True
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Histogram
            ax.hist(returns * 100, bins=50, alpha=0.7, color='blue', 
                   density=True, label='Returns')
            
            # Normal distribution overlay
            mean = returns.mean() * 100
            std = returns.std() * 100
            x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
            ax.plot(x, stats.norm.pdf(x, mean, std), 'r-', 
                   linewidth=2, label='Normal Distribution')
            
            ax.set_xlabel('Returns (%)')
            ax.set_ylabel('Density')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
            
    def plot_trades(
        self,
        data: pd.DataFrame,
        trades: pd.DataFrame,
        symbol: str,
        indicators: Optional[Dict[str, pd.Series]] = None
    ) -> Union[go.Figure, plt.Figure]:
        """
        Plot price chart with trade markers.
        
        Args:
            data: OHLCV data
            trades: Trade history
            symbol: Symbol name
            indicators: Optional indicators to plot
            
        Returns:
            Figure object
        """
        if self.style == "plotly":
            # Create candlestick chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3]
            )
            
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add trade markers
            buy_trades = trades[trades['type'] == 'OPEN']
            sell_trades = trades[trades['type'] == 'CLOSE']
            
            if not buy_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_trades['timestamp'],
                        y=buy_trades['price'],
                        mode='markers',
                        name='Buy',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='green'
                        )
                    ),
                    row=1, col=1
                )
                
            if not sell_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_trades['timestamp'],
                        y=sell_trades['price'],
                        mode='markers',
                        name='Sell',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='red'
                        )
                    ),
                    row=1, col=1
                )
                
            # Add indicators
            if indicators:
                for name, indicator in indicators.items():
                    fig.add_trace(
                        go.Scatter(
                            x=indicator.index,
                            y=indicator,
                            mode='lines',
                            name=name,
                            line=dict(width=1)
                        ),
                        row=1, col=1
                    )
                    
            # Volume
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name='Volume',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f'{symbol} - Trade Analysis',
                xaxis_title='Date',
                yaxis_title='Price',
                hovermode='x unified',
                height=800
            )
            
            fig.update_xaxes(rangeslider_visible=False)
            
            return fig
        else:
            # Matplotlib version
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            
            # Price plot
            ax1.plot(data.index, data['close'], label='Close', linewidth=1, color='black')
            
            # Trade markers
            buy_trades = trades[trades['type'] == 'OPEN']
            sell_trades = trades[trades['type'] == 'CLOSE']
            
            if not buy_trades.empty:
                ax1.scatter(buy_trades['timestamp'], buy_trades['price'],
                          marker='^', s=100, color='green', label='Buy', zorder=5)
                          
            if not sell_trades.empty:
                ax1.scatter(sell_trades['timestamp'], sell_trades['price'],
                          marker='v', s=100, color='red', label='Sell', zorder=5)
                          
            # Indicators
            if indicators:
                for name, indicator in indicators.items():
                    ax1.plot(indicator.index, indicator, label=name, linewidth=1)
                    
            ax1.set_ylabel('Price')
            ax1.set_title(f'{symbol} - Trade Analysis')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Volume
            ax2.bar(data.index, data['volume'], alpha=0.3, color='blue')
            ax2.set_ylabel('Volume')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
    def plot_performance_metrics(
        self,
        metrics: Dict,
        title: str = "Performance Metrics"
    ) -> Union[go.Figure, plt.Figure]:
        """
        Plot performance metrics dashboard.
        
        Args:
            metrics: Dictionary of metrics
            title: Chart title
            
        Returns:
            Figure object
        """
        if self.style == "plotly":
            # Create subplots for different metric categories
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Returns', 'Risk Metrics', 
                    'Trade Statistics', 'Risk-Adjusted Returns'
                ),
                specs=[[{'type': 'bar'}, {'type': 'bar'}],
                      [{'type': 'bar'}, {'type': 'bar'}]]
            )
            
            # Returns
            returns_data = {
                'Total Return': float(metrics.get('total_return', '0%').strip('%')),
                'Annual Return': float(metrics.get('annualized_return', '0%').strip('%'))
            }
            fig.add_trace(
                go.Bar(x=list(returns_data.keys()), y=list(returns_data.values()),
                      marker_color='green'),
                row=1, col=1
            )
            
            # Risk metrics
            risk_data = {
                'Volatility': float(metrics.get('volatility', '0%').strip('%')),
                'Max Drawdown': -float(metrics.get('max_drawdown', '0%').strip('%'))
            }
            fig.add_trace(
                go.Bar(x=list(risk_data.keys()), y=list(risk_data.values()),
                      marker_color='red'),
                row=1, col=2
            )
            
            # Trade stats
            trade_data = {
                'Win Rate': float(metrics.get('win_rate', '0%').strip('%')),
                'Profit Factor': float(metrics.get('profit_factor', '0'))
            }
            fig.add_trace(
                go.Bar(x=list(trade_data.keys()), y=list(trade_data.values()),
                      marker_color='blue'),
                row=2, col=1
            )
            
            # Risk-adjusted
            risk_adj_data = {
                'Sharpe': float(metrics.get('sharpe_ratio', '0')),
                'Sortino': float(metrics.get('sortino_ratio', '0')),
                'Calmar': float(metrics.get('calmar_ratio', '0'))
            }
            fig.add_trace(
                go.Bar(x=list(risk_adj_data.keys()), y=list(risk_adj_data.values()),
                      marker_color='purple'),
                row=2, col=2
            )
            
            fig.update_layout(
                title=title,
                showlegend=False,
                height=600
            )
            
            return fig
        else:
            # Matplotlib version
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Parse metrics
            total_return = float(metrics.get('total_return', '0%').strip('%'))
            annual_return = float(metrics.get('annualized_return', '0%').strip('%'))
            volatility = float(metrics.get('volatility', '0%').strip('%'))
            max_dd = float(metrics.get('max_drawdown', '0%').strip('%'))
            win_rate = float(metrics.get('win_rate', '0%').strip('%'))
            profit_factor = float(metrics.get('profit_factor', '0'))
            sharpe = float(metrics.get('sharpe_ratio', '0'))
            sortino = float(metrics.get('sortino_ratio', '0'))
            
            # Returns
            ax1.bar(['Total Return', 'Annual Return'], [total_return, annual_return], 
                   color='green', alpha=0.7)
            ax1.set_ylabel('Return (%)')
            ax1.set_title('Returns')
            
            # Risk
            ax2.bar(['Volatility', 'Max Drawdown'], [volatility, -max_dd],
                   color='red', alpha=0.7)
            ax2.set_ylabel('Risk (%)')
            ax2.set_title('Risk Metrics')
            
            # Trade stats
            ax3.bar(['Win Rate', 'Profit Factor'], [win_rate, profit_factor],
                   color='blue', alpha=0.7)
            ax3.set_ylabel('Value')
            ax3.set_title('Trade Statistics')
            
            # Risk-adjusted
            ax4.bar(['Sharpe', 'Sortino'], [sharpe, sortino],
                   color='purple', alpha=0.7)
            ax4.set_ylabel('Ratio')
            ax4.set_title('Risk-Adjusted Returns')
            
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            
            return fig