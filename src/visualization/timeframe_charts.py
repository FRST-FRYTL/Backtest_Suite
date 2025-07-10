"""
Timeframe Participation Visualizations

This module creates visualizations showing how different timeframes contribute
to trading decisions and performance.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeframeCharts:
    """
    Creates timeframe participation and analysis visualizations.
    """
    
    def __init__(self):
        """Initialize timeframe chart creator."""
        self.timeframe_colors = {
            '1H': '#FF6B6B',
            '4H': '#4ECDC4',
            '1D': '#45B7D1',
            '1W': '#96CEB4',
            '1M': '#FECA57'
        }
    
    def create_timeframe_participation_chart(
        self,
        trades: List[Dict]
    ) -> go.Figure:
        """
        Create stacked bar chart showing timeframe participation in each trade.
        
        Args:
            trades: List of trade dictionaries with timeframe scores
            
        Returns:
            Plotly stacked bar chart
        """
        # Extract timeframe participation data
        trade_ids = []
        timeframe_data = {tf: [] for tf in self.timeframe_colors.keys()}
        
        for i, trade in enumerate(trades):
            trade_ids.append(f"Trade {i+1}")
            
            # Get timeframe scores
            tf_scores = trade.get('timeframe_scores', {})
            total_score = sum(tf_scores.values()) if tf_scores else 1
            
            # Normalize to percentages
            for tf in self.timeframe_colors.keys():
                score = tf_scores.get(tf, 0)
                percentage = (score / total_score * 100) if total_score > 0 else 0
                timeframe_data[tf].append(percentage)
        
        # Create stacked bar chart
        fig = go.Figure()
        
        for timeframe, values in timeframe_data.items():
            fig.add_trace(go.Bar(
                name=timeframe,
                x=trade_ids,
                y=values,
                marker_color=self.timeframe_colors[timeframe],
                text=[f'{v:.1f}%' if v > 5 else '' for v in values],
                textposition='inside',
                textfont=dict(size=10)
            ))
        
        # Update layout
        fig.update_layout(
            barmode='stack',
            title='Timeframe Participation by Trade',
            xaxis_title='Trade',
            yaxis_title='Participation (%)',
            template='plotly_dark',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_timeframe_importance_pie(
        self,
        trades: List[Dict]
    ) -> go.Figure:
        """
        Create pie chart showing overall timeframe importance.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Plotly pie chart
        """
        # Calculate average contribution of each timeframe
        timeframe_totals = {tf: 0 for tf in self.timeframe_colors.keys()}
        
        for trade in trades:
            tf_scores = trade.get('timeframe_scores', {})
            for tf, score in tf_scores.items():
                if tf in timeframe_totals:
                    timeframe_totals[tf] += score
        
        # Create pie chart
        labels = list(timeframe_totals.keys())
        values = list(timeframe_totals.values())
        colors = [self.timeframe_colors[tf] for tf in labels]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker_colors=colors,
            textfont=dict(size=14),
            textposition='inside',
            textinfo='label+percent'
        )])
        
        # Update layout
        fig.update_layout(
            title='Overall Timeframe Importance',
            template='plotly_dark',
            height=500,
            annotations=[dict(
                text='Timeframe<br>Distribution',
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False
            )]
        )
        
        return fig
    
    def create_participation_timeline(
        self,
        trades: List[Dict]
    ) -> go.Figure:
        """
        Create timeline showing how timeframe participation evolves.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Plotly line chart
        """
        # Prepare data
        dates = []
        timeframe_series = {tf: [] for tf in self.timeframe_colors.keys()}
        
        for trade in trades:
            entry_time = pd.to_datetime(trade['entry_time'])
            dates.append(entry_time)
            
            tf_scores = trade.get('timeframe_scores', {})
            total_score = sum(tf_scores.values()) if tf_scores else 1
            
            for tf in self.timeframe_colors.keys():
                score = tf_scores.get(tf, 0)
                percentage = (score / total_score * 100) if total_score > 0 else 0
                timeframe_series[tf].append(percentage)
        
        # Create line chart
        fig = go.Figure()
        
        for timeframe, values in timeframe_series.items():
            # Add rolling average
            rolling_values = pd.Series(values).rolling(10, min_periods=1).mean()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=rolling_values,
                name=timeframe,
                mode='lines',
                line=dict(color=self.timeframe_colors[timeframe], width=2)
            ))
        
        # Update layout
        fig.update_layout(
            title='Timeframe Participation Evolution (10-trade rolling average)',
            xaxis_title='Date',
            yaxis_title='Participation (%)',
            template='plotly_dark',
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_timeframe_performance_analysis(
        self,
        trades: List[Dict]
    ) -> go.Figure:
        """
        Analyze performance by dominant timeframe.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Plotly figure with performance analysis
        """
        # Classify trades by dominant timeframe
        timeframe_trades = {tf: [] for tf in self.timeframe_colors.keys()}
        
        for trade in trades:
            tf_scores = trade.get('timeframe_scores', {})
            if tf_scores:
                dominant_tf = max(tf_scores.items(), key=lambda x: x[1])[0]
                if dominant_tf in timeframe_trades:
                    timeframe_trades[dominant_tf].append(trade['return'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Average Return by Dominant Timeframe',
                'Win Rate by Dominant Timeframe',
                'Trade Count by Dominant Timeframe',
                'Return Distribution'
            )
        )
        
        # Calculate statistics
        timeframes = []
        avg_returns = []
        win_rates = []
        trade_counts = []
        
        for tf, returns in timeframe_trades.items():
            if returns:
                timeframes.append(tf)
                avg_returns.append(np.mean(returns) * 100)
                win_rates.append(sum(1 for r in returns if r > 0) / len(returns) * 100)
                trade_counts.append(len(returns))
        
        # Average return bar chart
        colors1 = ['green' if r > 0 else 'red' for r in avg_returns]
        fig.add_trace(
            go.Bar(
                x=timeframes,
                y=avg_returns,
                marker_color=colors1,
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Win rate bar chart
        fig.add_trace(
            go.Bar(
                x=timeframes,
                y=win_rates,
                marker_color='#4287f5',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Trade count bar chart
        fig.add_trace(
            go.Bar(
                x=timeframes,
                y=trade_counts,
                marker_color='#f5a442',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Return distribution box plot
        for tf, returns in timeframe_trades.items():
            if returns:
                fig.add_trace(
                    go.Box(
                        y=[r * 100 for r in returns],
                        name=tf,
                        marker_color=self.timeframe_colors[tf],
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                     annotation_text="50%", row=1, col=2)
        
        # Update axes
        fig.update_xaxes(title_text="Timeframe", row=1, col=1)
        fig.update_xaxes(title_text="Timeframe", row=1, col=2)
        fig.update_xaxes(title_text="Timeframe", row=2, col=1)
        fig.update_xaxes(title_text="Timeframe", row=2, col=2)
        
        fig.update_yaxes(title_text="Avg Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=1, col=2)
        fig.update_yaxes(title_text="Trade Count", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title='Performance Analysis by Dominant Timeframe',
            template='plotly_dark',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_timeframe_correlation_matrix(
        self,
        trades: List[Dict]
    ) -> go.Figure:
        """
        Create correlation matrix between timeframe scores.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Plotly heatmap
        """
        # Extract timeframe scores into DataFrame
        data = []
        for trade in trades:
            tf_scores = trade.get('timeframe_scores', {})
            if tf_scores:
                data.append(tf_scores)
        
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data).fillna(0)
        
        # Calculate correlation
        correlation = df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation.round(2).values,
            texttemplate='%{text}',
            textfont=dict(size=12),
            colorbar=dict(title='Correlation')
        ))
        
        # Update layout
        fig.update_layout(
            title='Timeframe Score Correlation Matrix',
            template='plotly_dark',
            height=500,
            xaxis=dict(tickangle=-45)
        )
        
        return fig
    
    def save_all_charts(
        self,
        charts: Dict[str, go.Figure],
        output_dir: str = 'reports/timeframe_analysis'
    ):
        """Save all timeframe charts to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in charts.items():
            filepath = os.path.join(output_dir, f"{name}.html")
            fig.write_html(
                filepath,
                include_plotlyjs='cdn',
                config={'displayModeBar': True, 'displaylogo': False}
            )
            logger.info(f"Saved {name} to {filepath}")
        
        return output_dir