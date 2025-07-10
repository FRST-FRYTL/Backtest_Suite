"""
Confluence Analysis Charts and Heatmaps

This module creates visualizations for confluence score analysis including
heatmaps, component breakdowns, and signal quality visualizations.
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

class ConfluenceCharts:
    """
    Creates confluence analysis visualizations.
    """
    
    def __init__(self):
        """Initialize confluence chart creator."""
        self.component_colors = {
            'trend': '#4287f5',
            'momentum': '#f5a442',
            'volume': '#9933ff',
            'volatility': '#ff3366'
        }
        
        self.heatmap_colorscale = [
            [0.0, '#0a0e1a'],  # Dark blue (low confluence)
            [0.3, '#1a1f2e'],  # Dark gray
            [0.5, '#4287f5'],  # Blue
            [0.7, '#f5a442'],  # Orange
            [0.85, '#00ff88'], # Green
            [1.0, '#ff0000']   # Red (high confluence)
        ]
    
    def create_confluence_heatmap(
        self,
        confluence_history: pd.DataFrame,
        timeframe_weights: Dict[str, float]
    ) -> go.Figure:
        """
        Create confluence score heatmap over time.
        
        Args:
            confluence_history: DataFrame with confluence scores by timeframe
            timeframe_weights: Weight of each timeframe
            
        Returns:
            Plotly heatmap figure
        """
        # Prepare data for heatmap
        timeframes = list(confluence_history.columns)
        z_data = confluence_history.T.values
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=confluence_history.index,
            y=timeframes,
            colorscale=self.heatmap_colorscale,
            colorbar=dict(
                title="Confluence Score",
                tickmode="linear",
                tick0=0,
                dtick=0.1
            ),
            hovertemplate='Timeframe: %{y}<br>Time: %{x}<br>Score: %{z:.3f}<extra></extra>'
        ))
        
        # Add weight annotations
        for i, (tf, weight) in enumerate(timeframe_weights.items()):
            if tf in timeframes:
                fig.add_annotation(
                    x=confluence_history.index[0],
                    y=tf,
                    text=f"{weight:.0%}",
                    showarrow=False,
                    xanchor='right',
                    xshift=-10,
                    font=dict(color='white', size=10)
                )
        
        # Update layout
        fig.update_layout(
            title="Confluence Score Heatmap by Timeframe",
            xaxis_title="Date",
            yaxis_title="Timeframe",
            template='plotly_dark',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_component_breakdown(
        self,
        component_scores: Dict[str, pd.Series],
        weights: Dict[str, float]
    ) -> go.Figure:
        """
        Create stacked area chart showing component contributions.
        
        Args:
            component_scores: Score series for each component
            weights: Component weights
            
        Returns:
            Plotly stacked area chart
        """
        fig = go.Figure()
        
        # Calculate weighted contributions
        weighted_scores = {}
        for component, scores in component_scores.items():
            weight = weights.get(component, 0.25)
            weighted_scores[component] = scores * weight
        
        # Add traces for each component
        for component, scores in weighted_scores.items():
            fig.add_trace(go.Scatter(
                x=scores.index,
                y=scores.values,
                name=f"{component.capitalize()} ({weights.get(component, 0.25):.0%})",
                mode='lines',
                stackgroup='one',
                fillcolor=self.component_colors.get(component, '#888888'),
                line=dict(width=0.5, color=self.component_colors.get(component, '#888888'))
            ))
        
        # Add total confluence line
        total_confluence = sum(weighted_scores.values())
        fig.add_trace(go.Scatter(
            x=total_confluence.index,
            y=total_confluence.values,
            name='Total Confluence',
            mode='lines',
            line=dict(color='white', width=2, dash='dash'),
            yaxis='y2'
        ))
        
        # Update layout
        fig.update_layout(
            title="Confluence Component Breakdown",
            xaxis_title="Date",
            yaxis_title="Component Contribution",
            yaxis2=dict(
                title="Total Confluence",
                overlaying='y',
                side='right',
                range=[0, 1]
            ),
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
    
    def create_confluence_distribution(
        self,
        confluence_scores: pd.Series,
        trade_outcomes: Optional[pd.Series] = None
    ) -> go.Figure:
        """
        Create histogram showing confluence score distribution.
        
        Args:
            confluence_scores: Series of confluence scores
            trade_outcomes: Optional series of trade returns
            
        Returns:
            Plotly histogram figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Confluence Score Distribution', 'Average Return by Confluence Level')
        )
        
        # Histogram of confluence scores
        fig.add_trace(
            go.Histogram(
                x=confluence_scores,
                nbinsx=50,
                name='Frequency',
                marker_color='#4287f5',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # If trade outcomes provided, show return by confluence level
        if trade_outcomes is not None and len(trade_outcomes) > 0:
            # Bin confluence scores
            bins = np.linspace(0, 1, 21)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Calculate average return for each bin
            avg_returns = []
            for i in range(len(bins) - 1):
                mask = (confluence_scores >= bins[i]) & (confluence_scores < bins[i + 1])
                if mask.any():
                    avg_return = trade_outcomes[mask].mean()
                    avg_returns.append(avg_return)
                else:
                    avg_returns.append(0)
            
            # Add bar chart
            colors = ['green' if r > 0 else 'red' for r in avg_returns]
            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=avg_returns,
                    name='Avg Return',
                    marker_color=colors,
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Update layout
        fig.update_xaxes(title_text="Confluence Score", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Average Return (%)", row=2, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_signal_quality_timeline(
        self,
        trades: List[Dict],
        lookback_periods: List[int] = [20, 50, 100]
    ) -> go.Figure:
        """
        Create timeline showing signal quality evolution.
        
        Args:
            trades: List of trade dictionaries
            lookback_periods: Periods for rolling metrics
            
        Returns:
            Plotly figure with signal quality metrics
        """
        # Extract trade data
        trade_df = pd.DataFrame(trades)
        trade_df['entry_time'] = pd.to_datetime(trade_df['entry_time'])
        trade_df.set_index('entry_time', inplace=True)
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Rolling Win Rate', 'Rolling Sharpe Ratio', 'Average Confluence Score')
        )
        
        # Calculate rolling metrics
        colors = ['#4287f5', '#f5a442', '#00ff88']
        
        for i, period in enumerate(lookback_periods):
            # Rolling win rate
            win_rate = (trade_df['return'] > 0).rolling(period).mean()
            fig.add_trace(
                go.Scatter(
                    x=win_rate.index,
                    y=win_rate.values,
                    name=f'{period} trades',
                    line=dict(color=colors[i % len(colors)], width=2)
                ),
                row=1, col=1
            )
            
            # Rolling Sharpe ratio
            rolling_sharpe = (trade_df['return'].rolling(period).mean() / 
                            trade_df['return'].rolling(period).std() * np.sqrt(252))
            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    name=f'{period} trades',
                    line=dict(color=colors[i % len(colors)], width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Rolling average confluence
            avg_confluence = trade_df['confluence_score'].rolling(period).mean()
            fig.add_trace(
                go.Scatter(
                    x=avg_confluence.index,
                    y=avg_confluence.values,
                    name=f'{period} trades',
                    line=dict(color=colors[i % len(colors)], width=2),
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # Add reference lines
        fig.add_hline(y=0.6, line_dash="dash", line_color="yellow", 
                     annotation_text="Target: 60%", row=1, col=1)
        fig.add_hline(y=1.5, line_dash="dash", line_color="yellow", 
                     annotation_text="Target: 1.5", row=2, col=1)
        fig.add_hline(y=0.65, line_dash="dash", line_color="yellow", 
                     annotation_text="Entry Threshold", row=3, col=1)
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Win Rate", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
        fig.update_yaxes(title_text="Confluence Score", row=3, col=1)
        
        fig.update_layout(
            title="Signal Quality Evolution",
            template='plotly_dark',
            height=800,
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
    
    def create_confluence_radar_chart(
        self,
        average_scores: Dict[str, Dict[str, float]]
    ) -> go.Figure:
        """
        Create radar chart showing average component scores by timeframe.
        
        Args:
            average_scores: Nested dict of timeframe -> component -> score
            
        Returns:
            Plotly radar chart
        """
        fig = go.Figure()
        
        components = ['trend', 'momentum', 'volume', 'volatility']
        
        for timeframe, scores in average_scores.items():
            values = [scores.get(comp, 0) for comp in components]
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=components + [components[0]],
                fill='toself',
                name=timeframe
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Average Component Scores by Timeframe",
            template='plotly_dark',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def save_all_charts(
        self,
        charts: Dict[str, go.Figure],
        output_dir: str = 'reports/confluence_analysis'
    ):
        """Save all confluence charts to files."""
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