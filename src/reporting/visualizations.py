"""
Visualization utilities for standardized reports.

This module provides reusable chart functions with consistent styling
for generating professional visualizations in backtest reports.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ReportVisualizations:
    """Standardized visualization functions for reports."""
    
    def __init__(self, style_config: Optional[Dict[str, Any]] = None):
        """Initialize with optional style configuration."""
        self.style = self._default_style()
        if style_config:
            self.style.update(style_config)
    
    def _default_style(self) -> Dict[str, Any]:
        """Get default styling configuration."""
        return {
            'template': 'plotly_white',
            'color_scheme': {
                'primary': '#2E86AB',
                'secondary': '#A23B72',
                'success': '#27AE60',
                'warning': '#F39C12',
                'danger': '#E74C3C',
                'neutral': '#95A5A6'
            },
            'font': {
                'family': 'Arial, sans-serif',
                'size': 12
            },
            'chart_height': 500,
            'chart_width': 800
        }
    
    def performance_summary_chart(self, metrics: Dict[str, float], 
                                 thresholds: Optional[Dict[str, Dict[str, float]]] = None) -> go.Figure:
        """Create a performance summary chart with key metrics."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Returns', 'Risk Metrics', 'Trade Statistics', 'Risk-Adjusted Returns'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Returns subplot
        returns_data = {
            'Total Return': metrics.get('total_return', 0) * 100,
            'Annual Return': metrics.get('annual_return', 0) * 100,
            'Monthly Return': metrics.get('monthly_return', 0) * 100
        }
        fig.add_trace(
            go.Bar(x=list(returns_data.keys()), y=list(returns_data.values()),
                   marker_color=self.style['color_scheme']['primary'],
                   text=[f"{v:.1f}%" for v in returns_data.values()],
                   textposition='outside'),
            row=1, col=1
        )
        
        # Risk metrics subplot
        risk_data = {
            'Max Drawdown': abs(metrics.get('max_drawdown', 0)) * 100,
            'Volatility': metrics.get('volatility', 0) * 100,
            'Downside Dev': metrics.get('downside_deviation', 0) * 100
        }
        fig.add_trace(
            go.Bar(x=list(risk_data.keys()), y=list(risk_data.values()),
                   marker_color=self.style['color_scheme']['danger'],
                   text=[f"{v:.1f}%" for v in risk_data.values()],
                   textposition='outside'),
            row=1, col=2
        )
        
        # Trade statistics subplot
        trade_data = {
            'Win Rate': metrics.get('win_rate', 0) * 100,
            'Profit Factor': metrics.get('profit_factor', 0) * 10,  # Scale for visibility
            'Avg Trades/Month': metrics.get('avg_trades_per_month', 0)
        }
        fig.add_trace(
            go.Bar(x=list(trade_data.keys()), y=list(trade_data.values()),
                   marker_color=self.style['color_scheme']['success'],
                   text=[f"{v:.1f}" for v in trade_data.values()],
                   textposition='outside'),
            row=2, col=1
        )
        
        # Risk-adjusted returns (radar chart simulation)
        risk_adj_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
        risk_adj_values = [
            metrics.get('sharpe_ratio', 0),
            metrics.get('sortino_ratio', 0),
            metrics.get('calmar_ratio', 0)
        ]
        fig.add_trace(
            go.Scatter(x=risk_adj_metrics, y=risk_adj_values,
                      mode='markers+lines',
                      marker=dict(size=12, color=self.style['color_scheme']['secondary']),
                      line=dict(width=3)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Performance Summary Dashboard',
            template=self.style['template'],
            height=self.style['chart_height'] * 1.5,
            showlegend=False,
            font=self.style['font']
        )
        
        # Update axes
        fig.update_yaxes(title_text="Percentage (%)", row=1, col=1)
        fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        fig.update_yaxes(title_text="Ratio", row=2, col=2)
        
        return fig
    
    def cumulative_returns(self, returns: pd.Series, benchmark: Optional[pd.Series] = None) -> go.Figure:
        """Create cumulative returns chart."""
        fig = go.Figure()
        
        # Strategy cumulative returns
        cum_returns = (1 + returns).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns.values * 100,
            mode='lines',
            name='Strategy',
            line=dict(color=self.style['color_scheme']['primary'], width=2.5)
        ))
        
        # Benchmark cumulative returns if provided
        if benchmark is not None:
            bench_cum_returns = (1 + benchmark).cumprod() - 1
            fig.add_trace(go.Scatter(
                x=bench_cum_returns.index,
                y=bench_cum_returns.values * 100,
                mode='lines',
                name='Benchmark',
                line=dict(color=self.style['color_scheme']['neutral'], width=2, dash='dash')
            ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title='Cumulative Returns Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            template=self.style['template'],
            height=self.style['chart_height'],
            hovermode='x unified',
            font=self.style['font']
        )
        
        return fig
    
    def drawdown_chart(self, returns: pd.Series) -> go.Figure:
        """Create drawdown visualization."""
        # Calculate drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        fig = go.Figure()
        
        # Drawdown area
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color=self.style['color_scheme']['danger'], width=0),
            fillcolor=self.style['color_scheme']['danger'],
            opacity=0.3
        ))
        
        # Highlight maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        
        fig.add_trace(go.Scatter(
            x=[max_dd_idx],
            y=[max_dd_value * 100],
            mode='markers+text',
            name='Max Drawdown',
            marker=dict(size=12, color=self.style['color_scheme']['danger']),
            text=[f"Max DD: {max_dd_value*100:.1f}%"],
            textposition='top center'
        ))
        
        fig.update_layout(
            title='Drawdown Analysis',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template=self.style['template'],
            height=self.style['chart_height'],
            hovermode='x unified',
            font=self.style['font']
        )
        
        return fig
    
    def monthly_returns_heatmap(self, returns: pd.Series) -> go.Figure:
        """Create monthly returns heatmap."""
        # Prepare monthly returns data
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table for heatmap
        monthly_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values * 100
        })
        
        pivot_table = monthly_df.pivot(index='Year', columns='Month', values='Return')
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=pivot_table.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot_table.values, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Return %")
        ))
        
        fig.update_layout(
            title='Monthly Returns Heatmap',
            xaxis_title='Month',
            yaxis_title='Year',
            template=self.style['template'],
            height=self.style['chart_height'],
            font=self.style['font']
        )
        
        return fig
    
    def trade_distribution(self, trades: pd.DataFrame) -> go.Figure:
        """Create trade profit/loss distribution chart."""
        if 'pnl' not in trades.columns:
            return go.Figure()  # Return empty figure if no P&L data
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Trade P&L Distribution', 'Win/Loss Breakdown'),
            specs=[[{'type': 'histogram'}, {'type': 'pie'}]]
        )
        
        # P&L histogram
        fig.add_trace(
            go.Histogram(
                x=trades['pnl'],
                nbinsx=30,
                name='P&L Distribution',
                marker_color=self.style['color_scheme']['primary'],
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Win/Loss pie chart
        wins = len(trades[trades['pnl'] > 0])
        losses = len(trades[trades['pnl'] <= 0])
        
        fig.add_trace(
            go.Pie(
                labels=['Wins', 'Losses'],
                values=[wins, losses],
                marker_colors=[self.style['color_scheme']['success'], 
                              self.style['color_scheme']['danger']],
                textinfo='label+percent',
                textposition='inside'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Trade Analysis',
            template=self.style['template'],
            height=self.style['chart_height'],
            showlegend=False,
            font=self.style['font']
        )
        
        fig.update_xaxes(title_text="Profit/Loss", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        
        return fig
    
    def parameter_heatmap(self, optimization_results: pd.DataFrame,
                         param1: str, param2: str, metric: str = 'sharpe_ratio') -> go.Figure:
        """Create parameter optimization heatmap."""
        # Pivot data for heatmap
        pivot_data = optimization_results.pivot(
            index=param2,
            columns=param1,
            values=metric
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Viridis',
            text=np.round(pivot_data.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title=metric.replace('_', ' ').title())
        ))
        
        # Mark best parameter combination
        best_idx = np.unravel_index(np.argmax(pivot_data.values), pivot_data.shape)
        fig.add_trace(go.Scatter(
            x=[pivot_data.columns[best_idx[1]]],
            y=[pivot_data.index[best_idx[0]]],
            mode='markers',
            marker=dict(
                size=20,
                color='red',
                symbol='star',
                line=dict(color='white', width=2)
            ),
            name='Best',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f'Parameter Optimization: {metric.replace("_", " ").title()}',
            xaxis_title=param1.replace('_', ' ').title(),
            yaxis_title=param2.replace('_', ' ').title(),
            template=self.style['template'],
            height=self.style['chart_height'],
            font=self.style['font']
        )
        
        return fig
    
    def rolling_metrics(self, returns: pd.Series, window: int = 252) -> go.Figure:
        """Create rolling metrics visualization."""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Rolling Returns', 'Rolling Sharpe Ratio', 'Rolling Volatility'),
            shared_xaxes=True,
            vertical_spacing=0.08
        )
        
        # Rolling returns
        rolling_returns = returns.rolling(window).mean() * 252  # Annualized
        fig.add_trace(
            go.Scatter(
                x=rolling_returns.index,
                y=rolling_returns.values * 100,
                mode='lines',
                name='Rolling Return',
                line=dict(color=self.style['color_scheme']['primary'])
            ),
            row=1, col=1
        )
        
        # Rolling Sharpe
        rolling_sharpe = (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color=self.style['color_scheme']['secondary'])
            ),
            row=2, col=1
        )
        
        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values * 100,
                mode='lines',
                name='Rolling Volatility',
                line=dict(color=self.style['color_scheme']['warning'])
            ),
            row=3, col=1
        )
        
        # Add reference lines
        fig.add_hline(y=0, row=1, col=1, line_dash="dot", line_color="gray", opacity=0.5)
        fig.add_hline(y=1, row=2, col=1, line_dash="dot", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=f'Rolling {window}-Day Metrics',
            template=self.style['template'],
            height=self.style['chart_height'] * 1.5,
            showlegend=False,
            font=self.style['font']
        )
        
        fig.update_yaxes(title_text="Annual Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
        fig.update_yaxes(title_text="Annual Volatility (%)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig
    
    def create_performance_table(self, metrics: Dict[str, float],
                               thresholds: Optional[Dict[str, Dict[str, float]]] = None) -> str:
        """Create an HTML table with performance metrics."""
        html = """
        <table style="width:100%; border-collapse: collapse; font-family: Arial, sans-serif;">
            <thead>
                <tr style="background-color: #2E86AB; color: white;">
                    <th style="padding: 12px; text-align: left;">Metric</th>
                    <th style="padding: 12px; text-align: right;">Value</th>
                    <th style="padding: 12px; text-align: center;">Rating</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Define metric formatting
        formatters = {
            'percentage': lambda x: f"{x*100:.2f}%",
            'decimal': lambda x: f"{x:.3f}",
            'integer': lambda x: f"{int(x):,}",
            'currency': lambda x: f"${x:,.2f}"
        }
        
        # Add rows for each metric
        for i, (key, value) in enumerate(metrics.items()):
            # Determine formatting
            if 'return' in key or 'rate' in key or 'drawdown' in key:
                formatted_value = formatters['percentage'](value)
            elif 'ratio' in key or 'factor' in key:
                formatted_value = formatters['decimal'](value)
            elif 'trades' in key or 'count' in key:
                formatted_value = formatters['integer'](value)
            else:
                formatted_value = formatters['decimal'](value)
            
            # Determine rating if thresholds available
            rating = "N/A"
            rating_color = "#95A5A6"
            if thresholds and key in thresholds:
                rating = self._get_rating(value, thresholds[key])
                rating_color = self._get_rating_color(rating)
            
            # Add row
            row_bg = "#f8f9fa" if i % 2 == 0 else "white"
            html += f"""
                <tr style="background-color: {row_bg};">
                    <td style="padding: 10px; border-bottom: 1px solid #ddd;">
                        {key.replace('_', ' ').title()}
                    </td>
                    <td style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">
                        <strong>{formatted_value}</strong>
                    </td>
                    <td style="padding: 10px; text-align: center; border-bottom: 1px solid #ddd;">
                        <span style="color: {rating_color}; font-weight: bold;">
                            {rating}
                        </span>
                    </td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    def _get_rating(self, value: float, threshold: Dict[str, float]) -> str:
        """Get rating based on thresholds."""
        # This is a simplified version - you'd want to use the actual
        # MetricThresholds class here
        if 'excellent' in threshold and value >= threshold['excellent']:
            return "Excellent"
        elif 'good' in threshold and value >= threshold['good']:
            return "Good"
        elif 'acceptable' in threshold and value >= threshold['acceptable']:
            return "Acceptable"
        else:
            return "Poor"
    
    def _get_rating_color(self, rating: str) -> str:
        """Get color for rating."""
        colors = {
            'Excellent': '#27AE60',
            'Good': '#3498DB',
            'Acceptable': '#F39C12',
            'Poor': '#E74C3C',
            'N/A': '#95A5A6'
        }
        return colors.get(rating, '#95A5A6')
    
    def create_trade_price_chart(self, trades: pd.DataFrame, price_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """Create a comprehensive trade price chart showing entry/exit/stop levels."""
        if trades.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # Add price data if available
        if price_data is not None and 'close' in price_data.columns:
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['close'],
                mode='lines',
                name='Price',
                line=dict(color='black', width=1),
                opacity=0.7
            ))
        
        # Add entry points
        if 'entry_price' in trades.columns and 'entry_time' in trades.columns:
            entry_data = trades.dropna(subset=['entry_price', 'entry_time'])
            if not entry_data.empty:
                fig.add_trace(go.Scatter(
                    x=entry_data['entry_time'],
                    y=entry_data['entry_price'],
                    mode='markers',
                    name='Entry',
                    marker=dict(
                        size=8,
                        color=self.style['color_scheme']['primary'],
                        symbol='triangle-up',
                        line=dict(width=2, color='white')
                    )
                ))
        
        # Add exit points
        if 'exit_price' in trades.columns and 'exit_time' in trades.columns:
            exit_data = trades.dropna(subset=['exit_price', 'exit_time'])
            if not exit_data.empty:
                # Color exits based on profitability
                colors = []
                for _, trade in exit_data.iterrows():
                    if 'pnl' in trade and pd.notna(trade['pnl']):
                        colors.append(self.style['color_scheme']['success'] if trade['pnl'] > 0 else self.style['color_scheme']['danger'])
                    else:
                        colors.append(self.style['color_scheme']['secondary'])
                
                fig.add_trace(go.Scatter(
                    x=exit_data['exit_time'],
                    y=exit_data['exit_price'],
                    mode='markers',
                    name='Exit',
                    marker=dict(
                        size=8,
                        color=colors,
                        symbol='triangle-down',
                        line=dict(width=2, color='white')
                    )
                ))
        
        # Add stop loss levels
        if 'stop_loss' in trades.columns and 'entry_time' in trades.columns:
            stop_data = trades.dropna(subset=['stop_loss', 'entry_time'])
            if not stop_data.empty:
                fig.add_trace(go.Scatter(
                    x=stop_data['entry_time'],
                    y=stop_data['stop_loss'],
                    mode='markers',
                    name='Stop Loss',
                    marker=dict(
                        size=6,
                        color=self.style['color_scheme']['danger'],
                        symbol='x',
                        line=dict(width=2)
                    )
                ))
        
        # Add take profit levels
        if 'take_profit' in trades.columns and 'entry_time' in trades.columns:
            tp_data = trades.dropna(subset=['take_profit', 'entry_time'])
            if not tp_data.empty:
                fig.add_trace(go.Scatter(
                    x=tp_data['entry_time'],
                    y=tp_data['take_profit'],
                    mode='markers',
                    name='Take Profit',
                    marker=dict(
                        size=6,
                        color=self.style['color_scheme']['success'],
                        symbol='star',
                        line=dict(width=2)
                    )
                ))
        
        fig.update_layout(
            title='Trade Price Analysis',
            xaxis_title='Date',
            yaxis_title='Price',
            template=self.style['template'],
            height=self.style['chart_height'] * 1.2,
            hovermode='x unified',
            font=self.style['font'],
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_stop_loss_analysis(self, trades: pd.DataFrame) -> go.Figure:
        """Create stop loss effectiveness analysis chart."""
        if trades.empty or 'stop_loss' not in trades.columns:
            return go.Figure()
        
        # Filter trades with stop loss data
        stop_trades = trades.dropna(subset=['stop_loss']).copy()
        if stop_trades.empty:
            return go.Figure()
        
        # Calculate stop loss metrics
        stop_trades['stop_hit'] = False
        if 'exit_reason' in stop_trades.columns:
            stop_trades['stop_hit'] = stop_trades['exit_reason'].str.contains('stop', case=False, na=False)
        elif 'exit_price' in stop_trades.columns and 'entry_price' in stop_trades.columns:
            # Estimate stop hits based on price movement
            for idx, trade in stop_trades.iterrows():
                if 'side' in trade:
                    if trade['side'] == 'long':
                        stop_trades.loc[idx, 'stop_hit'] = trade['exit_price'] <= trade['stop_loss']
                    else:
                        stop_trades.loc[idx, 'stop_hit'] = trade['exit_price'] >= trade['stop_loss']
        
        # Calculate stop distance as percentage
        if 'entry_price' in stop_trades.columns:
            stop_trades['stop_distance'] = abs(stop_trades['stop_loss'] - stop_trades['entry_price']) / stop_trades['entry_price'] * 100
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Stop Loss Hit Rate', 'Stop Distance Distribution', 'Stop vs No-Stop P&L', 'Stop Distance vs P&L'),
            specs=[[{'type': 'pie'}, {'type': 'histogram'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Stop hit rate pie chart
        stop_hit_counts = stop_trades['stop_hit'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=['No Stop Hit', 'Stop Hit'],
                values=[stop_hit_counts.get(False, 0), stop_hit_counts.get(True, 0)],
                marker_colors=[self.style['color_scheme']['success'], self.style['color_scheme']['danger']]
            ),
            row=1, col=1
        )
        
        # Stop distance histogram
        if 'stop_distance' in stop_trades.columns:
            fig.add_trace(
                go.Histogram(
                    x=stop_trades['stop_distance'],
                    nbinsx=20,
                    name='Stop Distance',
                    marker_color=self.style['color_scheme']['warning']
                ),
                row=1, col=2
            )
        
        # P&L comparison
        if 'pnl' in stop_trades.columns:
            stop_hit_pnl = stop_trades[stop_trades['stop_hit']]['pnl'].mean()
            no_stop_pnl = stop_trades[~stop_trades['stop_hit']]['pnl'].mean()
            
            fig.add_trace(
                go.Bar(
                    x=['Stop Hit', 'No Stop Hit'],
                    y=[stop_hit_pnl, no_stop_pnl],
                    marker_color=[self.style['color_scheme']['danger'], self.style['color_scheme']['success']]
                ),
                row=2, col=1
            )
            
            # Stop distance vs P&L scatter
            if 'stop_distance' in stop_trades.columns:
                fig.add_trace(
                    go.Scatter(
                        x=stop_trades['stop_distance'],
                        y=stop_trades['pnl'],
                        mode='markers',
                        marker=dict(
                            color=stop_trades['stop_hit'].map({True: self.style['color_scheme']['danger'], False: self.style['color_scheme']['success']}),
                            size=6
                        ),
                        name='Trades'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title='Stop Loss Analysis',
            template=self.style['template'],
            height=self.style['chart_height'] * 1.5,
            showlegend=False,
            font=self.style['font']
        )
        
        fig.update_xaxes(title_text="Stop Distance (%)", row=1, col=2)
        fig.update_xaxes(title_text="Stop Distance (%)", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Average P&L", row=2, col=1)
        fig.update_yaxes(title_text="P&L", row=2, col=2)
        
        return fig
    
    def create_trade_risk_chart(self, trades: pd.DataFrame) -> go.Figure:
        """Create risk per trade analysis chart."""
        if trades.empty:
            return go.Figure()
        
        trades_copy = trades.copy()
        
        # Calculate risk per trade
        if 'stop_loss' in trades_copy.columns and 'entry_price' in trades_copy.columns and 'size' in trades_copy.columns:
            trades_copy['risk_per_trade'] = abs(trades_copy['stop_loss'] - trades_copy['entry_price']) * trades_copy['size']
            trades_copy['risk_pct'] = trades_copy['risk_per_trade'] / (trades_copy['entry_price'] * trades_copy['size']) * 100
        elif 'pnl' in trades_copy.columns:
            # Use actual loss as proxy for risk
            trades_copy['risk_per_trade'] = trades_copy['pnl'].where(trades_copy['pnl'] < 0, 0).abs()
            trades_copy['risk_pct'] = trades_copy['risk_per_trade'] / trades_copy.get('size', 1) * 100
        else:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk per Trade Distribution', 'Risk vs P&L', 'Risk Over Time', 'Risk by Trade Size'),
            specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Risk distribution histogram
        fig.add_trace(
            go.Histogram(
                x=trades_copy['risk_pct'],
                nbinsx=25,
                name='Risk Distribution',
                marker_color=self.style['color_scheme']['warning']
            ),
            row=1, col=1
        )
        
        # Risk vs P&L scatter
        if 'pnl' in trades_copy.columns:
            fig.add_trace(
                go.Scatter(
                    x=trades_copy['risk_pct'],
                    y=trades_copy['pnl'],
                    mode='markers',
                    marker=dict(
                        color=trades_copy['pnl'].apply(lambda x: self.style['color_scheme']['success'] if x > 0 else self.style['color_scheme']['danger']),
                        size=6
                    ),
                    name='Risk vs P&L'
                ),
                row=1, col=2
            )
        
        # Risk over time
        if 'entry_time' in trades_copy.columns:
            fig.add_trace(
                go.Scatter(
                    x=trades_copy['entry_time'],
                    y=trades_copy['risk_pct'],
                    mode='markers+lines',
                    marker=dict(color=self.style['color_scheme']['primary'], size=4),
                    line=dict(color=self.style['color_scheme']['primary'], width=1),
                    name='Risk Over Time'
                ),
                row=2, col=1
            )
        
        # Risk by trade size
        if 'size' in trades_copy.columns:
            fig.add_trace(
                go.Scatter(
                    x=trades_copy['size'],
                    y=trades_copy['risk_pct'],
                    mode='markers',
                    marker=dict(
                        color=self.style['color_scheme']['secondary'],
                        size=6
                    ),
                    name='Risk by Size'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Trade Risk Analysis',
            template=self.style['template'],
            height=self.style['chart_height'] * 1.5,
            showlegend=False,
            font=self.style['font']
        )
        
        fig.update_xaxes(title_text="Risk (%)", row=1, col=1)
        fig.update_xaxes(title_text="Risk (%)", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Trade Size", row=2, col=2)
        
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="P&L", row=1, col=2)
        fig.update_yaxes(title_text="Risk (%)", row=2, col=1)
        fig.update_yaxes(title_text="Risk (%)", row=2, col=2)
        
        return fig
    
    def save_all_charts(self, figures: Dict[str, go.Figure], output_dir: str) -> Dict[str, str]:
        """Save all charts to files and return paths."""
        import os
        
        paths = {}
        for name, fig in figures.items():
            # Save as HTML
            html_path = os.path.join(output_dir, f"{name}.html")
            fig.write_html(html_path)
            
            # Save as static image (requires kaleido)
            try:
                png_path = os.path.join(output_dir, f"{name}.png")
                fig.write_image(png_path, width=1200, height=800)
                paths[name] = {'html': html_path, 'png': png_path}
            except:
                paths[name] = {'html': html_path}
        
        return paths