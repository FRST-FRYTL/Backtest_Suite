"""
Multi-Timeframe Master Chart Visualization

This module creates comprehensive multi-timeframe charts showing price action,
indicators, and trade signals across all timeframes.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from data.multi_timeframe_data_manager import Timeframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiTimeframeMasterChart:
    """
    Creates master chart with synchronized multi-timeframe view.
    """
    
    def __init__(self):
        """Initialize the master chart creator."""
        self.colors = {
            'bullish': '#00ff88',
            'bearish': '#ff3366',
            'sma_20': '#4287f5',
            'sma_50': '#f5a442',
            'sma_200': '#f54242',
            'volume': '#9933ff',
            'buy_signal': '#00ff00',
            'sell_signal': '#ff0000',
            'background': '#0a0e1a',
            'grid': '#1a1f2e'
        }
        
    def create_master_chart(
        self,
        data_by_timeframe: Dict[Timeframe, pd.DataFrame],
        indicators_by_timeframe: Dict[Timeframe, Dict[str, pd.Series]],
        trades: Optional[List[Dict]] = None,
        symbol: str = "Asset"
    ) -> go.Figure:
        """
        Create comprehensive multi-timeframe chart.
        
        Args:
            data_by_timeframe: OHLCV data for each timeframe
            indicators_by_timeframe: Indicators for each timeframe
            trades: List of trade dictionaries
            symbol: Symbol name
            
        Returns:
            Plotly figure with multi-timeframe visualization
        """
        # Create subplots for each timeframe
        timeframes = sorted(data_by_timeframe.keys(), key=lambda x: x.value)
        n_timeframes = len(timeframes)
        
        # Create subplot layout
        fig = make_subplots(
            rows=n_timeframes + 2,  # +2 for volume and confluence
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.3] * n_timeframes + [0.15, 0.15],
            subplot_titles=[f"{tf.value} Timeframe" for tf in timeframes] + 
                          ['Volume', 'Confluence Score']
        )
        
        # Plot each timeframe
        for idx, timeframe in enumerate(timeframes):
            row = idx + 1
            data = data_by_timeframe[timeframe]
            indicators = indicators_by_timeframe.get(timeframe, {})
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name=f"{timeframe.value} Price",
                    showlegend=idx == 0,
                    increasing_line_color=self.colors['bullish'],
                    decreasing_line_color=self.colors['bearish']
                ),
                row=row, col=1
            )
            
            # Add SMAs
            for sma_period in [20, 50, 200]:
                sma_key = f'sma_{sma_period}'
                if sma_key in indicators:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=indicators[sma_key],
                            name=f"SMA {sma_period}",
                            line=dict(color=self.colors[sma_key], width=1),
                            showlegend=idx == 0
                        ),
                        row=row, col=1
                    )
            
            # Add Bollinger Bands if available
            if 'bb_upper' in indicators and 'bb_lower' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['bb_upper'],
                        name='BB Upper',
                        line=dict(color='rgba(128,128,128,0.5)', width=1),
                        showlegend=False
                    ),
                    row=row, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=indicators['bb_lower'],
                        name='BB Lower',
                        line=dict(color='rgba(128,128,128,0.5)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        showlegend=False
                    ),
                    row=row, col=1
                )
        
        # Add volume chart
        volume_row = n_timeframes + 1
        main_data = data_by_timeframe[timeframes[0]]  # Use finest timeframe for volume
        
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(main_data['close'], main_data['open'])]
        
        fig.add_trace(
            go.Bar(
                x=main_data.index,
                y=main_data['volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=volume_row, col=1
        )
        
        # Add confluence score if available
        confluence_row = n_timeframes + 2
        if trades:
            # Extract confluence scores from trades
            confluence_data = self._extract_confluence_scores(trades)
            if confluence_data:
                fig.add_trace(
                    go.Scatter(
                        x=confluence_data['timestamp'],
                        y=confluence_data['score'],
                        name='Confluence Score',
                        line=dict(color=self.colors['volume'], width=2),
                        fill='tozeroy',
                        fillcolor='rgba(153,51,255,0.2)'
                    ),
                    row=confluence_row, col=1
                )
                
                # Add threshold line
                fig.add_hline(
                    y=0.65,
                    line_dash="dash",
                    line_color="yellow",
                    annotation_text="Entry Threshold",
                    row=confluence_row, col=1
                )
        
        # Add trade markers
        if trades:
            self._add_trade_markers(fig, trades, timeframes, n_timeframes)
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Multi-Timeframe Analysis",
            template='plotly_dark',
            height=300 * (n_timeframes + 2),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        # Update axes
        for i in range(1, n_timeframes + 3):
            fig.update_xaxes(showgrid=True, gridcolor=self.colors['grid'], row=i, col=1)
            fig.update_yaxes(showgrid=True, gridcolor=self.colors['grid'], row=i, col=1)
        
        return fig
    
    def _add_trade_markers(
        self,
        fig: go.Figure,
        trades: List[Dict],
        timeframes: List[Timeframe],
        n_timeframes: int
    ):
        """Add trade entry/exit markers to all timeframe charts."""
        for trade in trades:
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            
            # Add markers to each timeframe chart
            for idx in range(n_timeframes):
                row = idx + 1
                
                # Entry marker
                fig.add_trace(
                    go.Scatter(
                        x=[entry_time],
                        y=[trade['entry_price']],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color=self.colors['buy_signal']
                        ),
                        name='Buy',
                        showlegend=idx == 0,
                        hovertext=f"Buy @ {trade['entry_price']:.2f}<br>"
                                 f"Confluence: {trade.get('confluence_score', 0):.2f}"
                    ),
                    row=row, col=1
                )
                
                # Exit marker
                fig.add_trace(
                    go.Scatter(
                        x=[exit_time],
                        y=[trade['exit_price']],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color=self.colors['sell_signal']
                        ),
                        name='Sell',
                        showlegend=idx == 0,
                        hovertext=f"Sell @ {trade['exit_price']:.2f}<br>"
                                 f"Return: {trade.get('return', 0)*100:.1f}%"
                    ),
                    row=row, col=1
                )
    
    def _extract_confluence_scores(self, trades: List[Dict]) -> Dict[str, List]:
        """Extract confluence score time series from trades."""
        timestamps = []
        scores = []
        
        for trade in trades:
            if 'confluence_history' in trade:
                for ts, score in trade['confluence_history']:
                    timestamps.append(pd.to_datetime(ts))
                    scores.append(score)
        
        if timestamps:
            return {'timestamp': timestamps, 'score': scores}
        return {}
    
    def create_synchronized_chart(
        self,
        primary_timeframe: Timeframe,
        data_by_timeframe: Dict[Timeframe, pd.DataFrame],
        sync_period: str = '1D'
    ) -> go.Figure:
        """
        Create synchronized chart showing alignment across timeframes.
        
        Args:
            primary_timeframe: Main timeframe to display
            data_by_timeframe: Data for all timeframes
            sync_period: Period to highlight synchronization
            
        Returns:
            Plotly figure with synchronized view
        """
        fig = go.Figure()
        
        # Get primary data
        primary_data = data_by_timeframe[primary_timeframe]
        
        # Add primary candlestick
        fig.add_trace(
            go.Candlestick(
                x=primary_data.index,
                open=primary_data['open'],
                high=primary_data['high'],
                low=primary_data['low'],
                close=primary_data['close'],
                name=f"{primary_timeframe.value} Price"
            )
        )
        
        # Add alignment indicators from other timeframes
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        color_idx = 0
        
        for timeframe, data in data_by_timeframe.items():
            if timeframe != primary_timeframe:
                # Resample to primary timeframe for alignment
                resampled = data['close'].resample(primary_data.index.freq).last()
                
                fig.add_trace(
                    go.Scatter(
                        x=resampled.index,
                        y=resampled,
                        name=f"{timeframe.value} Close",
                        line=dict(
                            color=colors[color_idx % len(colors)],
                            width=2,
                            dash='dash'
                        )
                    )
                )
                color_idx += 1
        
        # Update layout
        fig.update_layout(
            title=f"Synchronized Multi-Timeframe View - {primary_timeframe.value}",
            template='plotly_dark',
            height=600,
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode='x unified'
        )
        
        return fig
    
    def save_chart(self, fig: go.Figure, filename: str, output_dir: str = 'reports'):
        """Save chart to HTML file."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        fig.write_html(
            filepath,
            include_plotlyjs='cdn',
            config={'displayModeBar': True, 'displaylogo': False}
        )
        
        logger.info(f"Chart saved to {filepath}")
        return filepath