"""
Detailed Performance Analysis Reports

This module creates comprehensive performance analysis reports with
detailed metrics, statistics, and visualizations.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import calendar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceAnalysisReport:
    """
    Creates detailed performance analysis reports.
    """
    
    def __init__(self):
        """Initialize report generator."""
        self.colors = {
            'positive': '#00ff88',
            'negative': '#ff3366',
            'neutral': '#4287f5',
            'heatmap': 'RdYlGn'
        }
    
    def create_performance_report(
        self,
        trades: List[Dict],
        equity_curve: pd.Series,
        strategy_metrics: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """
        Create complete performance analysis report.
        
        Args:
            trades: List of trade dictionaries
            equity_curve: Portfolio equity over time
            strategy_metrics: Calculated strategy metrics
            
        Returns:
            Dictionary of figure name to Plotly figure
        """
        figures = {}
        
        # Trade statistics table
        figures['trade_stats'] = self._create_trade_statistics_table(trades)
        
        # Monthly/Yearly returns heatmap
        figures['returns_heatmap'] = self._create_returns_heatmap(trades)
        
        # Drawdown analysis
        figures['drawdown_analysis'] = self._create_drawdown_analysis(equity_curve)
        
        # Win/Loss distribution
        figures['win_loss_dist'] = self._create_win_loss_distribution(trades)
        
        # Rolling performance metrics
        figures['rolling_metrics'] = self._create_rolling_metrics(equity_curve, trades)
        
        # Trade duration analysis
        figures['duration_analysis'] = self._create_duration_analysis(trades)
        
        # Performance by market conditions
        figures['market_conditions'] = self._create_market_condition_analysis(trades)
        
        return figures
    
    def _create_trade_statistics_table(self, trades: List[Dict]) -> go.Figure:
        """Create comprehensive trade statistics table."""
        trade_df = pd.DataFrame(trades)
        
        # Calculate statistics
        stats = {
            'Total Trades': len(trades),
            'Winning Trades': sum(1 for t in trades if t['return'] > 0),
            'Losing Trades': sum(1 for t in trades if t['return'] <= 0),
            'Win Rate': f"{(trade_df['return'] > 0).mean()*100:.1f}%",
            'Average Win': f"{trade_df[trade_df['return'] > 0]['return'].mean()*100:.2f}%",
            'Average Loss': f"{trade_df[trade_df['return'] <= 0]['return'].mean()*100:.2f}%",
            'Largest Win': f"{trade_df['return'].max()*100:.2f}%",
            'Largest Loss': f"{trade_df['return'].min()*100:.2f}%",
            'Average Return': f"{trade_df['return'].mean()*100:.2f}%",
            'Return Std Dev': f"{trade_df['return'].std()*100:.2f}%",
            'Profit Factor': self._calculate_profit_factor(trade_df),
            'Average Hold Days': f"{trade_df['hold_days'].mean():.1f}",
            'Max Hold Days': f"{trade_df['hold_days'].max():.0f}",
            'Min Hold Days': f"{trade_df['hold_days'].min():.0f}",
            'Average Confluence': f"{trade_df['confluence_score'].mean():.3f}",
            'Trades per Month': f"{len(trades) / ((pd.to_datetime(trade_df['exit_time'].max()) - pd.to_datetime(trade_df['entry_time'].min())).days / 30):.1f}"
        }
        
        # Split into categories
        categories = {
            'Overview': ['Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate'],
            'Returns': ['Average Win', 'Average Loss', 'Largest Win', 'Largest Loss', 
                       'Average Return', 'Return Std Dev', 'Profit Factor'],
            'Duration': ['Average Hold Days', 'Max Hold Days', 'Min Hold Days'],
            'Other': ['Average Confluence', 'Trades per Month']
        }
        
        # Create subplots for each category
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'table'}, {'type': 'table'}],
                   [{'type': 'table'}, {'type': 'table'}]],
            subplot_titles=list(categories.keys())
        )
        
        # Add tables
        for idx, (category, metrics) in enumerate(categories.items()):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            values = [[m, stats[m]] for m in metrics]
            
            fig.add_trace(
                go.Table(
                    cells=dict(
                        values=[list(x) for x in zip(*values)],
                        fill_color=['#1a1f2e', '#0a0e1a'],
                        align='left',
                        font=dict(size=12)
                    )
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Comprehensive Trade Statistics',
            template='plotly_dark',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def _create_returns_heatmap(self, trades: List[Dict]) -> go.Figure:
        """Create monthly and yearly returns heatmap."""
        trade_df = pd.DataFrame(trades)
        trade_df['exit_date'] = pd.to_datetime(trade_df['exit_time'])
        
        # Calculate monthly returns
        monthly_returns = trade_df.groupby([
            trade_df['exit_date'].dt.year,
            trade_df['exit_date'].dt.month
        ])['return'].sum()
        
        # Create matrix for heatmap
        years = sorted(trade_df['exit_date'].dt.year.unique())
        months = list(range(1, 13))
        
        z_data = []
        for year in years:
            year_data = []
            for month in months:
                if (year, month) in monthly_returns.index:
                    year_data.append(monthly_returns[(year, month)] * 100)
                else:
                    year_data.append(0)
            z_data.append(year_data)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[calendar.month_abbr[m] for m in months],
            y=[str(y) for y in years],
            colorscale=self.colors['heatmap'],
            zmid=0,
            text=[[f'{val:.1f}%' for val in row] for row in z_data],
            texttemplate='%{text}',
            textfont=dict(size=10),
            colorbar=dict(title='Return (%)')
        ))
        
        # Add yearly totals
        yearly_returns = trade_df.groupby(trade_df['exit_date'].dt.year)['return'].sum() * 100
        
        # Add annotations for yearly totals
        for i, (year, total) in enumerate(yearly_returns.items()):
            fig.add_annotation(
                x=12.5,
                y=i,
                text=f'<b>{total:.1f}%</b>',
                showarrow=False,
                font=dict(size=12, color='white')
            )
        
        fig.update_layout(
            title='Monthly Returns Heatmap',
            xaxis_title='Month',
            yaxis_title='Year',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def _create_drawdown_analysis(self, equity_curve: pd.Series) -> go.Figure:
        """Create detailed drawdown analysis."""
        # Calculate drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        
        # Find drawdown periods
        drawdown_periods = self._find_drawdown_periods(drawdown)
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Equity Curve with Drawdown Periods', 'Drawdown Chart'),
            row_heights=[0.6, 0.4]
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                name='Portfolio Value',
                line=dict(color=self.colors['neutral'], width=2)
            ),
            row=1, col=1
        )
        
        # Mark drawdown periods
        for period in drawdown_periods:
            if period['duration'] > 5:  # Only show significant drawdowns
                fig.add_vrect(
                    x0=period['start'],
                    x1=period['end'],
                    fillcolor='rgba(255,51,102,0.2)',
                    layer='below',
                    line_width=0,
                    row=1, col=1
                )
        
        # Drawdown chart
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                name='Drawdown',
                fill='tozeroy',
                fillcolor='rgba(255,51,102,0.3)',
                line=dict(color=self.colors['negative'], width=1)
            ),
            row=2, col=1
        )
        
        # Add max drawdown line
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        
        fig.add_annotation(
            x=max_dd_idx,
            y=max_dd_value * 100,
            text=f'Max DD: {max_dd_value*100:.1f}%',
            showarrow=True,
            arrowhead=2,
            row=2, col=1
        )
        
        fig.update_yaxes(title_text='Portfolio Value', row=1, col=1)
        fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=2, col=1)
        
        fig.update_layout(
            title='Drawdown Analysis',
            template='plotly_dark',
            height=700,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_win_loss_distribution(self, trades: List[Dict]) -> go.Figure:
        """Create win/loss distribution analysis."""
        trade_df = pd.DataFrame(trades)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Return Distribution',
                'Win/Loss Count by Size',
                'Cumulative Win Rate',
                'Return vs Confluence Score'
            )
        )
        
        # Return distribution histogram
        fig.add_trace(
            go.Histogram(
                x=trade_df['return'] * 100,
                nbinsx=50,
                name='Returns',
                marker_color=self.colors['neutral'],
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Win/Loss count by size
        bins = [-100, -5, -2, -1, 0, 1, 2, 5, 100]
        labels = ['< -5%', '-5 to -2%', '-2 to -1%', '-1 to 0%', 
                 '0 to 1%', '1 to 2%', '2 to 5%', '> 5%']
        
        trade_df['return_bin'] = pd.cut(trade_df['return'] * 100, bins=bins, labels=labels)
        bin_counts = trade_df['return_bin'].value_counts().sort_index()
        
        colors = ['darkred', 'red', 'lightcoral', 'pink', 
                 'lightgreen', 'green', 'darkgreen', 'darkgreen']
        
        fig.add_trace(
            go.Bar(
                x=bin_counts.index,
                y=bin_counts.values,
                marker_color=colors,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Cumulative win rate
        trade_df_sorted = trade_df.sort_values('exit_time')
        trade_df_sorted['cumulative_wins'] = (trade_df_sorted['return'] > 0).cumsum()
        trade_df_sorted['cumulative_total'] = range(1, len(trade_df_sorted) + 1)
        trade_df_sorted['cumulative_win_rate'] = (
            trade_df_sorted['cumulative_wins'] / trade_df_sorted['cumulative_total'] * 100
        )
        
        fig.add_trace(
            go.Scatter(
                x=trade_df_sorted.index,
                y=trade_df_sorted['cumulative_win_rate'],
                mode='lines',
                line=dict(color=self.colors['positive'], width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add target line
        fig.add_hline(y=60, line_dash="dash", line_color="yellow", 
                     annotation_text="Target: 60%", row=2, col=1)
        
        # Return vs Confluence scatter
        colors = [self.colors['positive'] if r > 0 else self.colors['negative'] 
                 for r in trade_df['return']]
        
        fig.add_trace(
            go.Scatter(
                x=trade_df['confluence_score'],
                y=trade_df['return'] * 100,
                mode='markers',
                marker=dict(color=colors, size=8),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Add trend line
        z = np.polyfit(trade_df['confluence_score'], trade_df['return'] * 100, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(trade_df['confluence_score'].min(), 
                            trade_df['confluence_score'].max(), 100)
        
        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                line=dict(color='white', width=1, dash='dash'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text='Return (%)', row=1, col=1)
        fig.update_xaxes(title_text='Return Range', row=1, col=2)
        fig.update_xaxes(title_text='Trade Number', row=2, col=1)
        fig.update_xaxes(title_text='Confluence Score', row=2, col=2)
        
        fig.update_yaxes(title_text='Frequency', row=1, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=2)
        fig.update_yaxes(title_text='Win Rate (%)', row=2, col=1)
        fig.update_yaxes(title_text='Return (%)', row=2, col=2)
        
        fig.update_layout(
            title='Win/Loss Distribution Analysis',
            template='plotly_dark',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def _create_rolling_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Dict]
    ) -> go.Figure:
        """Create rolling performance metrics chart."""
        trade_df = pd.DataFrame(trades)
        trade_df['exit_time'] = pd.to_datetime(trade_df['exit_time'])
        trade_df.set_index('exit_time', inplace=True)
        
        # Calculate rolling metrics
        window = 20  # 20 trades
        
        rolling_metrics = pd.DataFrame(index=trade_df.index)
        rolling_metrics['win_rate'] = (trade_df['return'] > 0).rolling(window).mean() * 100
        rolling_metrics['avg_return'] = trade_df['return'].rolling(window).mean() * 100
        rolling_metrics['sharpe'] = (
            trade_df['return'].rolling(window).mean() / 
            trade_df['return'].rolling(window).std() * np.sqrt(252)
        )
        rolling_metrics['avg_confluence'] = trade_df['confluence_score'].rolling(window).mean()
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'Rolling Win Rate ({window} trades)',
                f'Rolling Average Return ({window} trades)',
                f'Rolling Sharpe Ratio ({window} trades)',
                f'Rolling Average Confluence ({window} trades)'
            )
        )
        
        # Add traces
        metrics = ['win_rate', 'avg_return', 'sharpe', 'avg_confluence']
        colors = [self.colors['positive'], self.colors['neutral'], 
                 self.colors['positive'], self.colors['neutral']]
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            fig.add_trace(
                go.Scatter(
                    x=rolling_metrics.index,
                    y=rolling_metrics[metric],
                    mode='lines',
                    line=dict(color=color, width=2),
                    showlegend=False
                ),
                row=i+1, col=1
            )
        
        # Add reference lines
        fig.add_hline(y=60, line_dash="dash", line_color="yellow", row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_hline(y=1.5, line_dash="dash", line_color="yellow", row=3, col=1)
        fig.add_hline(y=0.65, line_dash="dash", line_color="yellow", row=4, col=1)
        
        # Update axes
        fig.update_yaxes(title_text='Win Rate (%)', row=1, col=1)
        fig.update_yaxes(title_text='Return (%)', row=2, col=1)
        fig.update_yaxes(title_text='Sharpe Ratio', row=3, col=1)
        fig.update_yaxes(title_text='Confluence', row=4, col=1)
        fig.update_xaxes(title_text='Date', row=4, col=1)
        
        fig.update_layout(
            title='Rolling Performance Metrics',
            template='plotly_dark',
            height=800,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_duration_analysis(self, trades: List[Dict]) -> go.Figure:
        """Create trade duration analysis."""
        trade_df = pd.DataFrame(trades)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Hold Days Distribution',
                'Return vs Hold Days',
                'Average Return by Duration',
                'Win Rate by Duration'
            )
        )
        
        # Hold days distribution
        fig.add_trace(
            go.Histogram(
                x=trade_df['hold_days'],
                nbinsx=30,
                marker_color=self.colors['neutral'],
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Return vs Hold days scatter
        colors = [self.colors['positive'] if r > 0 else self.colors['negative'] 
                 for r in trade_df['return']]
        
        fig.add_trace(
            go.Scatter(
                x=trade_df['hold_days'],
                y=trade_df['return'] * 100,
                mode='markers',
                marker=dict(color=colors, size=6),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Average return by duration buckets
        duration_bins = [0, 5, 10, 20, 30, 50, 100]
        duration_labels = ['0-5', '5-10', '10-20', '20-30', '30-50', '50+']
        
        trade_df['duration_bin'] = pd.cut(
            trade_df['hold_days'], 
            bins=duration_bins, 
            labels=duration_labels
        )
        
        avg_by_duration = trade_df.groupby('duration_bin')['return'].mean() * 100
        
        fig.add_trace(
            go.Bar(
                x=avg_by_duration.index,
                y=avg_by_duration.values,
                marker_color=[self.colors['positive'] if v > 0 else self.colors['negative'] 
                            for v in avg_by_duration.values],
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Win rate by duration
        win_rate_by_duration = trade_df.groupby('duration_bin').apply(
            lambda x: (x['return'] > 0).mean() * 100
        )
        
        fig.add_trace(
            go.Bar(
                x=win_rate_by_duration.index,
                y=win_rate_by_duration.values,
                marker_color=self.colors['positive'],
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text='Hold Days', row=1, col=1)
        fig.update_xaxes(title_text='Hold Days', row=1, col=2)
        fig.update_xaxes(title_text='Duration Range', row=2, col=1)
        fig.update_xaxes(title_text='Duration Range', row=2, col=2)
        
        fig.update_yaxes(title_text='Frequency', row=1, col=1)
        fig.update_yaxes(title_text='Return (%)', row=1, col=2)
        fig.update_yaxes(title_text='Avg Return (%)', row=2, col=1)
        fig.update_yaxes(title_text='Win Rate (%)', row=2, col=2)
        
        fig.update_layout(
            title='Trade Duration Analysis',
            template='plotly_dark',
            height=700,
            showlegend=False
        )
        
        return fig
    
    def _create_market_condition_analysis(self, trades: List[Dict]) -> go.Figure:
        """Create performance analysis by market conditions."""
        # This is a placeholder - in real implementation would use actual market data
        trade_df = pd.DataFrame(trades)
        
        # Simulate market conditions based on time periods
        trade_df['market_condition'] = pd.cut(
            range(len(trade_df)),
            bins=5,
            labels=['Bull Market', 'Bear Market', 'High Volatility', 
                   'Low Volatility', 'Sideways']
        )
        
        # Calculate metrics by condition
        metrics_by_condition = trade_df.groupby('market_condition').agg({
            'return': ['mean', 'count'],
            'confluence_score': 'mean'
        })
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                'Average Return by Market Condition',
                'Trade Count by Market Condition',
                'Average Confluence by Market Condition'
            )
        )
        
        # Average return
        avg_returns = metrics_by_condition['return']['mean'] * 100
        fig.add_trace(
            go.Bar(
                x=avg_returns.index,
                y=avg_returns.values,
                marker_color=[self.colors['positive'] if v > 0 else self.colors['negative'] 
                            for v in avg_returns.values],
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Trade count
        trade_counts = metrics_by_condition['return']['count']
        fig.add_trace(
            go.Bar(
                x=trade_counts.index,
                y=trade_counts.values,
                marker_color=self.colors['neutral'],
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Average confluence
        avg_confluence = metrics_by_condition['confluence_score']['mean']
        fig.add_trace(
            go.Bar(
                x=avg_confluence.index,
                y=avg_confluence.values,
                marker_color=self.colors['neutral'],
                showlegend=False
            ),
            row=1, col=3
        )
        
        # Update axes
        fig.update_xaxes(tickangle=-45)
        fig.update_yaxes(title_text='Return (%)', row=1, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=2)
        fig.update_yaxes(title_text='Confluence Score', row=1, col=3)
        
        fig.update_layout(
            title='Performance by Market Conditions',
            template='plotly_dark',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def _calculate_profit_factor(self, trade_df: pd.DataFrame) -> str:
        """Calculate profit factor."""
        profits = trade_df[trade_df['return'] > 0]['return'].sum()
        losses = abs(trade_df[trade_df['return'] <= 0]['return'].sum())
        
        if losses > 0:
            return f"{profits/losses:.2f}"
        else:
            return "∞"
    
    def _find_drawdown_periods(
        self,
        drawdown: pd.Series,
        threshold: float = -0.02
    ) -> List[Dict[str, Any]]:
        """Find significant drawdown periods."""
        periods = []
        in_drawdown = False
        start = None
        
        for date, dd in drawdown.items():
            if dd < threshold and not in_drawdown:
                in_drawdown = True
                start = date
            elif dd >= -0.001 and in_drawdown:
                in_drawdown = False
                if start:
                    periods.append({
                        'start': start,
                        'end': date,
                        'duration': (date - start).days,
                        'max_drawdown': drawdown[start:date].min()
                    })
        
        return periods
    
    def save_all_figures(
        self,
        figures: Dict[str, go.Figure],
        output_dir: str = 'reports/performance_analysis'
    ):
        """Save all performance analysis figures."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in figures.items():
            filepath = os.path.join(output_dir, f"{name}.html")
            fig.write_html(
                filepath,
                include_plotlyjs='cdn',
                config={'displayModeBar': True, 'displaylogo': False}
            )
            logger.info(f"Saved {name} to {filepath}")
        
        return output_dir