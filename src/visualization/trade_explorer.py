"""
Interactive Trade Explorer

This module creates an interactive table and visualization system for exploring
individual trades with filtering and detailed analysis capabilities.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractiveTradeExplorer:
    """
    Creates interactive trade exploration interface.
    """
    
    def __init__(self):
        """Initialize trade explorer."""
        self.profit_color = '#00ff88'
        self.loss_color = '#ff3366'
        
    def create_trade_table(
        self,
        trades: List[Dict],
        include_columns: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create interactive trade table with sorting and filtering.
        
        Args:
            trades: List of trade dictionaries
            include_columns: Columns to include (None for all)
            
        Returns:
            Plotly table figure
        """
        # Convert to DataFrame
        trade_df = pd.DataFrame(trades)
        
        # Default columns if not specified
        if include_columns is None:
            include_columns = [
                'trade_id', 'symbol', 'entry_time', 'exit_time',
                'entry_price', 'exit_price', 'position_size',
                'return', 'pnl', 'confluence_score', 'hold_days',
                'exit_reason'
            ]
        
        # Filter columns that exist
        available_columns = [col for col in include_columns if col in trade_df.columns]
        display_df = trade_df[available_columns].copy()
        
        # Format numeric columns
        if 'return' in display_df.columns:
            display_df['return'] = display_df['return'].apply(lambda x: f"{x*100:.2f}%")
        if 'pnl' in display_df.columns:
            display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:,.2f}")
        if 'confluence_score' in display_df.columns:
            display_df['confluence_score'] = display_df['confluence_score'].apply(lambda x: f"{x:.3f}")
        if 'position_size' in display_df.columns:
            display_df['position_size'] = display_df['position_size'].apply(lambda x: f"{x:.1%}")
        
        # Format dates
        for col in ['entry_time', 'exit_time']:
            if col in display_df.columns:
                display_df[col] = pd.to_datetime(display_df[col]).dt.strftime('%Y-%m-%d %H:%M')
        
        # Create color coding for returns
        cell_colors = []
        for col in display_df.columns:
            if col == 'return':
                colors = []
                for val in trade_df['return']:
                    if val > 0:
                        colors.append(self.profit_color)
                    else:
                        colors.append(self.loss_color)
                cell_colors.append(colors)
            else:
                cell_colors.append(['white'] * len(display_df))
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(display_df.columns),
                fill_color='#1a1f2e',
                align='left',
                font=dict(size=12, color='white')
            ),
            cells=dict(
                values=[display_df[col] for col in display_df.columns],
                fill_color=cell_colors,
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title="Interactive Trade Explorer",
            template='plotly_dark',
            height=600
        )
        
        return fig
    
    def create_trade_details_view(
        self,
        trade: Dict,
        price_data: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Create detailed view for a single trade.
        
        Args:
            trade: Trade dictionary
            price_data: Optional price data for context
            
        Returns:
            Plotly figure with trade details
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Trade Price Action',
                'Confluence Breakdown',
                'Risk/Reward Analysis',
                'Trade Metrics'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'table'}]
            ]
        )
        
        # Price action chart
        if price_data is not None:
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            
            # Filter data for trade period
            mask = (price_data.index >= entry_time) & (price_data.index <= exit_time)
            trade_data = price_data[mask]
            
            if not trade_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=trade_data.index,
                        y=trade_data['close'],
                        mode='lines',
                        name='Price',
                        line=dict(color='#4287f5', width=2)
                    ),
                    row=1, col=1
                )
                
                # Add entry/exit markers
                fig.add_trace(
                    go.Scatter(
                        x=[entry_time],
                        y=[trade['entry_price']],
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=15, color=self.profit_color),
                        name='Entry'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[exit_time],
                        y=[trade['exit_price']],
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=15, color=self.loss_color),
                        name='Exit'
                    ),
                    row=1, col=1
                )
                
                # Add stop loss line if available
                if 'stop_loss' in trade:
                    fig.add_hline(
                        y=trade['stop_loss'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Stop Loss",
                        row=1, col=1
                    )
        
        # Confluence breakdown
        if 'component_scores' in trade:
            components = list(trade['component_scores'].keys())
            scores = list(trade['component_scores'].values())
            
            fig.add_trace(
                go.Bar(
                    x=components,
                    y=scores,
                    marker_color=['#4287f5', '#f5a442', '#9933ff', '#ff3366'],
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Risk/Reward analysis
        risk_reward_data = {
            'Risk': abs(trade.get('max_risk', 0)),
            'Reward': trade.get('max_reward', 0),
            'Actual': trade.get('pnl', 0)
        }
        
        colors = ['red', 'green', 'green' if risk_reward_data['Actual'] > 0 else 'red']
        
        fig.add_trace(
            go.Bar(
                x=list(risk_reward_data.keys()),
                y=list(risk_reward_data.values()),
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Trade metrics table
        metrics = {
            'Trade ID': trade.get('trade_id', 'N/A'),
            'Symbol': trade.get('symbol', 'N/A'),
            'Duration': f"{trade.get('hold_days', 0)} days",
            'Return': f"{trade.get('return', 0)*100:.2f}%",
            'Confluence': f"{trade.get('confluence_score', 0):.3f}",
            'Exit Reason': trade.get('exit_reason', 'N/A'),
            'Position Size': f"{trade.get('position_size', 0):.1%}",
            'Risk/Reward': f"{trade.get('risk_reward_ratio', 0):.2f}"
        }
        
        fig.add_trace(
            go.Table(
                cells=dict(
                    values=[list(metrics.keys()), list(metrics.values())],
                    fill_color=['#1a1f2e', '#0a0e1a'],
                    align='left'
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Trade Details - {trade.get('trade_id', 'N/A')}",
            template='plotly_dark',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_trade_scatter_matrix(
        self,
        trades: List[Dict],
        dimensions: List[str] = None
    ) -> go.Figure:
        """
        Create scatter matrix for trade analysis.
        
        Args:
            trades: List of trade dictionaries
            dimensions: Dimensions to include in scatter matrix
            
        Returns:
            Plotly scatter matrix figure
        """
        trade_df = pd.DataFrame(trades)
        
        # Default dimensions
        if dimensions is None:
            dimensions = ['confluence_score', 'return', 'hold_days', 'position_size']
        
        # Filter available dimensions
        available_dims = [dim for dim in dimensions if dim in trade_df.columns]
        
        # Create color based on profit/loss
        colors = ['green' if r > 0 else 'red' for r in trade_df['return']]
        
        fig = px.scatter_matrix(
            trade_df,
            dimensions=available_dims,
            color=trade_df['return'] > 0,
            color_discrete_map={True: self.profit_color, False: self.loss_color},
            labels={col: col.replace('_', ' ').title() for col in available_dims},
            title="Trade Scatter Matrix Analysis"
        )
        
        fig.update_traces(diagonal_visible=False)
        
        fig.update_layout(
            template='plotly_dark',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_trade_performance_summary(
        self,
        trades: List[Dict]
    ) -> go.Figure:
        """
        Create summary statistics visualization.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Plotly figure with summary stats
        """
        trade_df = pd.DataFrame(trades)
        
        # Calculate statistics
        stats = {
            'Total Trades': len(trades),
            'Win Rate': f"{(trade_df['return'] > 0).mean()*100:.1f}%",
            'Avg Return': f"{trade_df['return'].mean()*100:.2f}%",
            'Best Trade': f"{trade_df['return'].max()*100:.2f}%",
            'Worst Trade': f"{trade_df['return'].min()*100:.2f}%",
            'Avg Hold Days': f"{trade_df['hold_days'].mean():.1f}",
            'Profit Factor': self._calculate_profit_factor(trade_df),
            'Avg Confluence': f"{trade_df['confluence_score'].mean():.3f}"
        }
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Key Metrics',
                'Return Distribution',
                'Trade Outcomes by Exit Reason',
                'Monthly Performance'
            ),
            specs=[
                [{'type': 'table'}, {'type': 'histogram'}],
                [{'type': 'bar'}, {'type': 'bar'}]
            ]
        )
        
        # Metrics table
        fig.add_trace(
            go.Table(
                cells=dict(
                    values=[list(stats.keys()), list(stats.values())],
                    fill_color=['#1a1f2e', '#0a0e1a'],
                    align='left',
                    font=dict(size=12)
                )
            ),
            row=1, col=1
        )
        
        # Return distribution
        fig.add_trace(
            go.Histogram(
                x=trade_df['return'] * 100,
                nbinsx=30,
                marker_color='#4287f5',
                name='Returns',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Exit reason analysis
        if 'exit_reason' in trade_df.columns:
            exit_stats = trade_df.groupby('exit_reason')['return'].agg(['count', 'mean'])
            
            fig.add_trace(
                go.Bar(
                    x=exit_stats.index,
                    y=exit_stats['count'],
                    name='Count',
                    marker_color='#4287f5',
                    yaxis='y3'
                ),
                row=2, col=1
            )
        
        # Monthly performance
        trade_df['month'] = pd.to_datetime(trade_df['exit_time']).dt.to_period('M')
        monthly_returns = trade_df.groupby('month')['return'].sum() * 100
        
        fig.add_trace(
            go.Bar(
                x=monthly_returns.index.astype(str),
                y=monthly_returns.values,
                marker_color=['green' if r > 0 else 'red' for r in monthly_returns.values],
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_xaxes(title_text="Return (%)", row=1, col=2)
        fig.update_xaxes(title_text="Exit Reason", row=2, col=1)
        fig.update_xaxes(title_text="Month", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=2)
        
        fig.update_layout(
            title="Trade Performance Summary",
            template='plotly_dark',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def _calculate_profit_factor(self, trade_df: pd.DataFrame) -> str:
        """Calculate profit factor from trades."""
        profits = trade_df[trade_df['return'] > 0]['return'].sum()
        losses = abs(trade_df[trade_df['return'] <= 0]['return'].sum())
        
        if losses > 0:
            return f"{profits/losses:.2f}"
        else:
            return "∞"
    
    def save_explorer(self, fig: go.Figure, filename: str, output_dir: str = 'reports'):
        """Save trade explorer to HTML file."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        fig.write_html(
            filepath,
            include_plotlyjs='cdn',
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']
            }
        )
        
        logger.info(f"Trade explorer saved to {filepath}")
        return filepath