"""
Enhanced Interactive Visualization Framework

This module provides comprehensive interactive visualizations for the enhanced
confluence strategy including multi-timeframe charts, confluence heatmaps,
and trade analysis dashboards.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
from datetime import datetime
import json

try:
    from ..data.multi_timeframe_data_manager import Timeframe
    from ..analysis.enhanced_trade_tracker import TradeAnalysis, EnhancedTradeTracker
    from ..analysis.baseline_comparisons import BaselineResults
except ImportError:
    from data.multi_timeframe_data_manager import Timeframe
    from analysis.enhanced_trade_tracker import TradeAnalysis, EnhancedTradeTracker
    from analysis.baseline_comparisons import BaselineResults

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedInteractiveCharts:
    """
    Enhanced interactive visualization framework for confluence strategy analysis.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the enhanced charts framework.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'confluence_strong': '#2ca02c',
            'confluence_moderate': '#ff7f0e',
            'confluence_weak': '#d62728',
            'timeframes': {
                '1H': '#e377c2',
                '4H': '#7f7f7f',
                '1D': '#1f77b4',
                '1W': '#ff7f0e',
                '1M': '#2ca02c'
            }
        }
    
    def create_master_trading_view(
        self,
        data_by_timeframe: Dict[Timeframe, pd.DataFrame],
        confluence_scores: pd.DataFrame,
        trades: List[TradeAnalysis],
        indicators: Dict[Timeframe, Dict[str, pd.Series]]
    ) -> go.Figure:
        """
        Create comprehensive multi-timeframe trading view.
        
        Args:
            data_by_timeframe: Data for each timeframe
            confluence_scores: Confluence score DataFrame
            trades: List of completed trades
            indicators: Indicators for each timeframe
            
        Returns:
            Plotly figure with master trading view
        """
        # Create subplot structure
        fig = make_subplots(
            rows=6, cols=2,
            specs=[
                [{"colspan": 2}, None],           # Price action (main)
                [{"colspan": 2}, None],           # Volume
                [{}, {}],                         # RSI 1D, RSI 1W  
                [{}, {}],                         # VWAP 1D, VWAP 1W
                [{"colspan": 2}, None],           # Confluence score
                [{"colspan": 2}, None]            # Trade P&L
            ],
            subplot_titles=[
                'Price Action with Multi-Timeframe Indicators',
                'Volume Profile', 
                'RSI Daily', 'RSI Weekly',
                'VWAP Daily', 'VWAP Weekly', 
                'Confluence Score Timeline',
                'Cumulative Trade P&L'
            ],
            vertical_spacing=0.02,
            row_heights=[0.4, 0.1, 0.1, 0.1, 0.15, 0.15]
        )
        
        # Main price chart with candlesticks
        if Timeframe.DAY_1 in data_by_timeframe:
            daily_data = data_by_timeframe[Timeframe.DAY_1]
            
            fig.add_trace(
                go.Candlestick(
                    x=daily_data.index,
                    open=daily_data['open'],
                    high=daily_data['high'],
                    low=daily_data['low'],
                    close=daily_data['close'],
                    name='Price',
                    increasing_line_color=self.colors['success'],
                    decreasing_line_color=self.colors['danger']
                ),
                row=1, col=1
            )
            
            # Add multi-timeframe SMAs
            self._add_sma_lines(fig, indicators, row=1, col=1)
            
            # Add VWAP with bands
            self._add_vwap_with_bands(fig, indicators, Timeframe.DAY_1, row=1, col=1)
            
            # Add trade markers
            self._add_trade_markers(fig, trades, row=1, col=1)
            
            # Volume chart
            fig.add_trace(
                go.Bar(
                    x=daily_data.index,
                    y=daily_data['volume'],
                    name='Volume',
                    marker_color=self.colors['info'],
                    opacity=0.6
                ),
                row=2, col=1
            )
        
        # RSI panels
        self._add_rsi_chart(fig, indicators, Timeframe.DAY_1, row=3, col=1, title='Daily RSI')
        self._add_rsi_chart(fig, indicators, Timeframe.WEEK_1, row=3, col=2, title='Weekly RSI')
        
        # VWAP panels  
        self._add_vwap_chart(fig, indicators, Timeframe.DAY_1, row=4, col=1, title='Daily VWAP')
        self._add_vwap_chart(fig, indicators, Timeframe.WEEK_1, row=4, col=2, title='Weekly VWAP')
        
        # Confluence score timeline
        if not confluence_scores.empty:
            fig.add_trace(
                go.Scatter(
                    x=confluence_scores.index,
                    y=confluence_scores['confluence_score'],
                    mode='lines',
                    name='Confluence Score',
                    line=dict(color=self.colors['primary'], width=2),
                    fill='tonexty'
                ),
                row=5, col=1
            )
            
            # Add confluence threshold line
            fig.add_hline(
                y=0.65, line_dash="dash", line_color=self.colors['warning'],
                annotation_text="Confluence Threshold", row=5, col=1
            )
        
        # Cumulative P&L
        if trades:
            cumulative_pnl = self._calculate_cumulative_pnl(trades)
            fig.add_trace(
                go.Scatter(
                    x=cumulative_pnl.index,
                    y=cumulative_pnl.values,
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color=self.colors['success'], width=2),
                    fill='tozeroy'
                ),
                row=6, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Enhanced Confluence Strategy - Master Trading View',
            height=1200,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        return fig
    
    def create_confluence_heatmap(
        self,
        confluence_scores: pd.DataFrame,
        trades: List[TradeAnalysis]
    ) -> go.Figure:
        """
        Create confluence score heatmap showing performance by score and time.
        
        Args:
            confluence_scores: Confluence score DataFrame
            trades: List of completed trades
            
        Returns:
            Plotly heatmap figure
        """
        if confluence_scores.empty:
            return go.Figure()
        
        # Create time periods and confluence ranges
        confluence_scores['date'] = confluence_scores.index.date
        confluence_scores['month'] = confluence_scores.index.to_period('M')
        
        # Define confluence score ranges
        score_ranges = [
            (0.0, 0.5, '0.0-0.5'),
            (0.5, 0.6, '0.5-0.6'),
            (0.6, 0.7, '0.6-0.7'),
            (0.7, 0.8, '0.7-0.8'),
            (0.8, 1.0, '0.8-1.0')
        ]
        
        # Create trade performance lookup
        trade_performance = {}
        for trade in trades:
            entry_date = trade.entry.timestamp.date()
            score = trade.entry.confluence_score
            performance = trade.total_return
            
            if entry_date not in trade_performance:
                trade_performance[entry_date] = []
            trade_performance[entry_date].append((score, performance))
        
        # Build heatmap data
        months = sorted(confluence_scores['month'].unique())
        heatmap_data = []
        
        for month in months:
            month_data = confluence_scores[confluence_scores['month'] == month]
            month_trades = []
            
            # Get trades for this month
            for date in month_data['date'].unique():
                if date in trade_performance:
                    month_trades.extend(trade_performance[date])
            
            row_data = []
            for min_score, max_score, range_label in score_ranges:
                # Filter trades in this score range
                range_trades = [(s, p) for s, p in month_trades if min_score <= s < max_score]
                
                if range_trades:
                    avg_return = np.mean([p for s, p in range_trades])
                    trade_count = len(range_trades)
                else:
                    avg_return = 0
                    trade_count = 0
                
                row_data.append({
                    'month': str(month),
                    'range': range_label,
                    'avg_return': avg_return,
                    'trade_count': trade_count,
                    'text': f"{trade_count} trades<br>{avg_return:.1f}% avg return"
                })
            
            heatmap_data.extend(row_data)
        
        # Convert to pivot format
        heatmap_df = pd.DataFrame(heatmap_data)
        if heatmap_df.empty:
            return go.Figure()
        
        pivot_returns = heatmap_df.pivot(index='range', columns='month', values='avg_return')
        pivot_text = heatmap_df.pivot(index='range', columns='month', values='text')
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_returns.values,
            x=pivot_returns.columns,
            y=pivot_returns.index,
            text=pivot_text.values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale='RdYlGn',
            zmid=0,
            hovertemplate='Month: %{x}<br>Confluence: %{y}<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Performance by Confluence Score and Time Period',
            xaxis_title='Month',
            yaxis_title='Confluence Score Range',
            height=400
        )
        
        return fig
    
    def create_timeframe_participation_radar(
        self,
        trades: List[TradeAnalysis]
    ) -> go.Figure:
        """
        Create radar chart showing timeframe contribution to performance.
        
        Args:
            trades: List of completed trades
            
        Returns:
            Plotly radar chart figure
        """
        if not trades:
            return go.Figure()
        
        # Separate winning and losing trades
        winning_trades = [t for t in trades if t.total_return > 0]
        losing_trades = [t for t in trades if t.total_return <= 0]
        
        # Calculate average timeframe scores
        timeframes = ['1H', '4H', '1D', '1W', '1M']
        
        def get_avg_scores(trade_list):
            if not trade_list:
                return [0] * len(timeframes)
            
            avg_scores = []
            for tf in timeframes:
                scores = []
                for trade in trade_list:
                    if tf in trade.entry.timeframe_scores:
                        scores.append(trade.entry.timeframe_scores[tf])
                avg_scores.append(np.mean(scores) if scores else 0)
            return avg_scores
        
        winning_scores = get_avg_scores(winning_trades)
        losing_scores = get_avg_scores(losing_trades)
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=winning_scores,
            theta=timeframes,
            fill='toself',
            name=f'Winning Trades ({len(winning_trades)})',
            line_color=self.colors['success']
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=losing_scores,
            theta=timeframes,
            fill='toself',
            name=f'Losing Trades ({len(losing_trades)})',
            line_color=self.colors['danger']
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title='Timeframe Participation: Winning vs Losing Trades'
        )
        
        return fig
    
    def create_trade_performance_dashboard(
        self,
        trade_tracker: EnhancedTradeTracker
    ) -> go.Figure:
        """
        Create comprehensive trade performance dashboard.
        
        Args:
            trade_tracker: Enhanced trade tracker instance
            
        Returns:
            Plotly dashboard figure
        """
        summary_stats = trade_tracker.get_trade_summary_statistics()
        
        if not summary_stats:
            return go.Figure().add_annotation(
                text="No trade data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Win Rate & Trade Count',
                'Return Distribution',
                'Risk Metrics',
                'Confluence Performance',
                'Hold Time Analysis',
                'Performance Trends'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "histogram"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "box"}, {"type": "scatter"}]
            ]
        )
        
        # Win rate indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=summary_stats.get('win_rate', 0),
                title={'text': f"Win Rate<br>{summary_stats.get('total_trades', 0)} trades"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.colors['success']},
                    'steps': [
                        {'range': [0, 50], 'color': self.colors['danger']},
                        {'range': [50, 70], 'color': self.colors['warning']},
                        {'range': [70, 100], 'color': self.colors['success']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 60
                    }
                }
            ),
            row=1, col=1
        )
        
        # Return distribution
        if trade_tracker.completed_trades:
            returns = [trade.total_return for trade in trade_tracker.completed_trades]
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=20,
                    name='Return Distribution',
                    marker_color=self.colors['info']
                ),
                row=1, col=2
            )
        
        # Risk metrics bar chart
        risk_metrics = {
            'Avg Return': summary_stats.get('avg_return', 0),
            'Best Trade': summary_stats.get('best_trade', 0),
            'Worst Trade': summary_stats.get('worst_trade', 0),
            'Profit Factor': summary_stats.get('profit_factor', 0)
        }
        
        fig.add_trace(
            go.Bar(
                x=list(risk_metrics.keys()),
                y=list(risk_metrics.values()),
                name='Risk Metrics',
                marker_color=self.colors['primary']
            ),
            row=1, col=3
        )
        
        # Add more charts for confluence, hold time, and trends
        self._add_confluence_performance_chart(fig, trade_tracker, row=2, col=1)
        self._add_hold_time_analysis(fig, trade_tracker, row=2, col=2)
        self._add_performance_trends(fig, trade_tracker, row=2, col=3)
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Trade Performance Dashboard"
        )
        
        return fig
    
    def create_baseline_comparison_chart(
        self,
        strategy_results: Dict[str, Any],
        baseline_results: List[BaselineResults]
    ) -> go.Figure:
        """
        Create comprehensive baseline comparison chart.
        
        Args:
            strategy_results: Strategy performance results
            baseline_results: List of baseline results
            
        Returns:
            Plotly comparison chart
        """
        # Create comparison data
        strategies = ['Enhanced Confluence Strategy']
        strategies.extend([baseline.strategy_name for baseline in baseline_results])
        
        total_returns = [strategy_results.get('total_return', 0)]
        total_returns.extend([baseline.total_return for baseline in baseline_results])
        
        sharpe_ratios = [strategy_results.get('sharpe_ratio', 0)]
        sharpe_ratios.extend([baseline.sharpe_ratio for baseline in baseline_results])
        
        max_drawdowns = [strategy_results.get('max_drawdown', 0)]
        max_drawdowns.extend([baseline.max_drawdown for baseline in baseline_results])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Total Return Comparison',
                'Sharpe Ratio Comparison',
                'Maximum Drawdown Comparison',
                'Risk-Return Scatter'
            ]
        )
        
        # Total return comparison
        colors = [self.colors['primary']] + [self.colors['secondary']] * len(baseline_results)
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=total_returns,
                name='Total Return (%)',
                marker_color=colors
            ),
            row=1, col=1
        )
        
        # Sharpe ratio comparison
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=sharpe_ratios,
                name='Sharpe Ratio',
                marker_color=colors
            ),
            row=1, col=2
        )
        
        # Max drawdown comparison
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=max_drawdowns,
                name='Max Drawdown (%)',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # Risk-return scatter
        volatilities = [strategy_results.get('volatility', 0)]
        volatilities.extend([baseline.volatility for baseline in baseline_results])
        
        fig.add_trace(
            go.Scatter(
                x=volatilities,
                y=total_returns,
                mode='markers+text',
                text=strategies,
                textposition="top center",
                name='Risk vs Return',
                marker=dict(size=12, color=colors)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Strategy vs Baseline Performance Comparison"
        )
        
        return fig
    
    def _add_sma_lines(self, fig, indicators, row, col):
        """Add SMA lines to chart."""
        if Timeframe.DAY_1 in indicators:
            daily_indicators = indicators[Timeframe.DAY_1]
            sma_keys = [k for k in daily_indicators.keys() if k.startswith('sma_')]
            
            for sma_key in sma_keys:
                period = sma_key.split('_')[1]
                fig.add_trace(
                    go.Scatter(
                        x=daily_indicators[sma_key].index,
                        y=daily_indicators[sma_key].values,
                        mode='lines',
                        name=f'SMA {period}',
                        line=dict(width=1, dash='dash'),
                        opacity=0.7
                    ),
                    row=row, col=col
                )
    
    def _add_vwap_with_bands(self, fig, indicators, timeframe, row, col):
        """Add VWAP with bands."""
        if timeframe in indicators and 'vwap' in indicators[timeframe]:
            vwap_data = indicators[timeframe]
            
            # VWAP line
            fig.add_trace(
                go.Scatter(
                    x=vwap_data['vwap'].index,
                    y=vwap_data['vwap'].values,
                    mode='lines',
                    name='VWAP',
                    line=dict(color=self.colors['warning'], width=2)
                ),
                row=row, col=col
            )
            
            # VWAP bands
            if 'vwap_upper_1.0' in vwap_data and 'vwap_lower_1.0' in vwap_data:
                fig.add_trace(
                    go.Scatter(
                        x=vwap_data['vwap_upper_1.0'].index,
                        y=vwap_data['vwap_upper_1.0'].values,
                        mode='lines',
                        name='VWAP Upper',
                        line=dict(color=self.colors['warning'], width=1, dash='dot'),
                        opacity=0.5
                    ),
                    row=row, col=col
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=vwap_data['vwap_lower_1.0'].index,
                        y=vwap_data['vwap_lower_1.0'].values,
                        mode='lines',
                        name='VWAP Lower',
                        line=dict(color=self.colors['warning'], width=1, dash='dot'),
                        opacity=0.5,
                        fill='tonexty',
                        fillcolor='rgba(255, 127, 14, 0.1)'
                    ),
                    row=row, col=col
                )
    
    def _add_trade_markers(self, fig, trades, row, col):
        """Add trade entry/exit markers."""
        if not trades:
            return
        
        entry_dates = [trade.entry.timestamp for trade in trades]
        entry_prices = [trade.entry.price for trade in trades]
        exit_dates = [trade.exit.timestamp for trade in trades]
        exit_prices = [trade.exit.price for trade in trades]
        
        # Entry markers
        fig.add_trace(
            go.Scatter(
                x=entry_dates,
                y=entry_prices,
                mode='markers',
                name='Trade Entries',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color=self.colors['success']
                )
            ),
            row=row, col=col
        )
        
        # Exit markers
        fig.add_trace(
            go.Scatter(
                x=exit_dates,
                y=exit_prices,
                mode='markers',
                name='Trade Exits',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color=self.colors['danger']
                )
            ),
            row=row, col=col
        )
    
    def _add_rsi_chart(self, fig, indicators, timeframe, row, col, title):
        """Add RSI chart."""
        if timeframe in indicators and 'rsi' in indicators[timeframe]:
            rsi_data = indicators[timeframe]['rsi']
            
            fig.add_trace(
                go.Scatter(
                    x=rsi_data.index,
                    y=rsi_data.values,
                    mode='lines',
                    name=f'RSI {timeframe.value}',
                    line=dict(color=self.colors['primary'])
                ),
                row=row, col=col
            )
            
            # Add overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color=self.colors['danger'], row=row, col=col)
            fig.add_hline(y=30, line_dash="dash", line_color=self.colors['success'], row=row, col=col)
    
    def _add_vwap_chart(self, fig, indicators, timeframe, row, col, title):
        """Add VWAP deviation chart."""
        if timeframe in indicators and 'vwap_position' in indicators[timeframe]:
            vwap_pos = indicators[timeframe]['vwap_position']
            
            fig.add_trace(
                go.Scatter(
                    x=vwap_pos.index,
                    y=vwap_pos.values,
                    mode='lines',
                    name=f'VWAP Position {timeframe.value}',
                    line=dict(color=self.colors['info'])
                ),
                row=row, col=col
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color='gray', row=row, col=col)
    
    def _calculate_cumulative_pnl(self, trades):
        """Calculate cumulative P&L from trades."""
        if not trades:
            return pd.Series(dtype=float)
        
        pnl_data = []
        cumulative = 0
        
        for trade in trades:
            cumulative += trade.total_return
            pnl_data.append({
                'date': trade.exit.timestamp,
                'cumulative_pnl': cumulative
            })
        
        df = pd.DataFrame(pnl_data)
        return df.set_index('date')['cumulative_pnl']
    
    def _add_confluence_performance_chart(self, fig, trade_tracker, row, col):
        """Add confluence performance analysis."""
        confluence_analysis = trade_tracker.analyze_performance_by_confluence()
        
        if not confluence_analysis.empty:
            fig.add_trace(
                go.Scatter(
                    x=confluence_analysis['confluence_range'],
                    y=confluence_analysis['avg_return'],
                    mode='markers+lines',
                    name='Avg Return by Confluence',
                    marker=dict(size=confluence_analysis['trade_count']*2)
                ),
                row=row, col=col
            )
    
    def _add_hold_time_analysis(self, fig, trade_tracker, row, col):
        """Add hold time analysis."""
        if trade_tracker.completed_trades:
            hold_times = [trade.exit.hold_days for trade in trade_tracker.completed_trades]
            returns = [trade.total_return for trade in trade_tracker.completed_trades]
            
            fig.add_trace(
                go.Box(
                    y=hold_times,
                    name='Hold Times (Days)',
                    boxpoints='all'
                ),
                row=row, col=col
            )
    
    def _add_performance_trends(self, fig, trade_tracker, row, col):
        """Add performance trend analysis."""
        if trade_tracker.completed_trades:
            # Create monthly performance aggregation
            monthly_data = {}
            
            for trade in trade_tracker.completed_trades:
                month = trade.exit.timestamp.to_period('M')
                if month not in monthly_data:
                    monthly_data[month] = []
                monthly_data[month].append(trade.total_return)
            
            months = sorted(monthly_data.keys())
            avg_returns = [np.mean(monthly_data[month]) for month in months]
            
            fig.add_trace(
                go.Scatter(
                    x=[str(month) for month in months],
                    y=avg_returns,
                    mode='lines+markers',
                    name='Monthly Avg Return'
                ),
                row=row, col=col
            )
    
    def save_chart(self, fig: go.Figure, filename: str, include_plotlyjs: bool = True) -> str:
        """
        Save chart to HTML file.
        
        Args:
            fig: Plotly figure
            filename: Output filename
            include_plotlyjs: Whether to include Plotly.js
            
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / filename
        fig.write_html(str(filepath), include_plotlyjs=include_plotlyjs)
        logger.info(f"Saved chart to {filepath}")
        return str(filepath)