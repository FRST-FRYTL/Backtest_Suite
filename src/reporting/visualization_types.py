"""
Visualization Types for Standardized Reporting

This module provides consistent, professional visualizations
for all report types.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from enum import Enum
from dataclasses import dataclass

warnings.filterwarnings('ignore')


class ChartType(Enum):
    """Enumeration of available chart types."""
    EQUITY_CURVE = "equity_curve"
    DRAWDOWN = "drawdown"
    RETURNS_DISTRIBUTION = "returns_distribution"
    TRADE_SCATTER = "trade_scatter"
    ROLLING_METRICS = "rolling_metrics"
    HEATMAP = "heatmap"
    TRADE_PRICE = "trade_price"
    TRADE_RISK = "trade_risk"


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    figure_size: Tuple[int, int] = (12, 8)
    figure_dpi: int = 300
    color_scheme: Dict[str, str] = None
    
    def __post_init__(self):
        if self.color_scheme is None:
            self.color_scheme = {
                "primary": "#1f77b4",
                "secondary": "#ff7f0e",
                "success": "#2ca02c",
                "warning": "#ff9800",
                "danger": "#d62728",
                "info": "#17a2b8",
                "background": "#ffffff",
                "text": "#333333"
            }


class BaseVisualization:
    """Base class for all visualizations"""
    
    def __init__(self, config):
        self.config = config
        self.setup_style()
        
    def setup_style(self):
        """Setup consistent visual style"""
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Configure seaborn
        sns.set_palette("husl")
        
        # Set default colors from config
        self.colors = self.config.color_scheme
        
        # Plotly template
        self.plotly_template = self._create_plotly_template()
        
    def _create_plotly_template(self) -> dict:
        """Create custom Plotly template"""
        return {
            "layout": {
                "font": {"family": "Arial, sans-serif", "size": 12, "color": self.colors["text"]},
                "paper_bgcolor": self.colors["background"],
                "plot_bgcolor": self.colors["background"],
                "title": {"font": {"size": 16, "color": self.colors["text"]}},
                "xaxis": {"gridcolor": "#E5E5E5", "linecolor": "#E5E5E5"},
                "yaxis": {"gridcolor": "#E5E5E5", "linecolor": "#E5E5E5"},
                "colorway": [
                    self.colors["primary"],
                    self.colors["secondary"],
                    self.colors["success"],
                    self.colors["warning"],
                    self.colors["danger"]
                ]
            }
        }
    
    def save_figure(self, fig, save_path: Optional[Path] = None, format: str = "png"):
        """Save figure to file"""
        if save_path:
            if isinstance(fig, go.Figure):
                # Plotly figure
                if format == "png":
                    fig.write_image(str(save_path), scale=2)
                elif format == "html":
                    fig.write_html(str(save_path))
            else:
                # Matplotlib figure
                fig.savefig(str(save_path), dpi=self.config.figure_dpi, bbox_inches='tight')
                plt.close(fig)


class EquityCurveChart(BaseVisualization):
    """Create professional equity curve visualizations"""
    
    def create(
        self,
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Create equity curve chart"""
        # Create figure
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Strategy',
            line=dict(color=self.colors["primary"], width=2),
            fill='tonexty',
            fillcolor=f"rgba{tuple(int(self.colors['primary'][i:i+2], 16) for i in (1, 3, 5)) + (0.1,)}"
        ))
        
        # Add benchmark if provided
        if benchmark is not None:
            fig.add_trace(go.Scatter(
                x=benchmark.index,
                y=benchmark.values,
                mode='lines',
                name='Benchmark',
                line=dict(color=self.colors["secondary"], width=2, dash='dash')
            ))
        
        # Add drawdown shading
        drawdown = self._calculate_drawdown(equity_curve)
        self._add_drawdown_shading(fig, drawdown)
        
        # Update layout
        fig.update_layout(
            title='Portfolio Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value',
            template=self.plotly_template,
            hovermode='x unified',
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            height=500
        )
        
        # Add annotations for key events
        self._add_performance_annotations(fig, equity_curve)
        
        # Save if requested
        self.save_figure(fig, save_path)
        
        return {
            "figure": fig,
            "data": {
                "final_value": equity_curve.iloc[-1],
                "peak_value": equity_curve.max(),
                "total_return": (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
            }
        }
    
    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return drawdown
    
    def _add_drawdown_shading(self, fig: go.Figure, drawdown: pd.Series):
        """Add drawdown shading to the chart"""
        # Find drawdown periods
        is_drawdown = drawdown < 0
        drawdown_starts = (~is_drawdown).shift(1) & is_drawdown
        drawdown_ends = is_drawdown.shift(1) & (~is_drawdown)
        
        # Add shading for each drawdown period
        start_idx = None
        for idx in drawdown.index:
            if drawdown_starts.loc[idx]:
                start_idx = idx
            elif drawdown_ends.loc[idx] and start_idx is not None:
                fig.add_vrect(
                    x0=start_idx,
                    x1=idx,
                    fillcolor=self.colors["danger"],
                    opacity=0.1,
                    layer="below",
                    line_width=0
                )
                start_idx = None
    
    def _add_performance_annotations(self, fig: go.Figure, equity_curve: pd.Series):
        """Add annotations for key performance events"""
        # Mark maximum value
        max_idx = equity_curve.idxmax()
        fig.add_annotation(
            x=max_idx,
            y=equity_curve.loc[max_idx],
            text="Peak",
            showarrow=True,
            arrowhead=2,
            arrowcolor=self.colors["success"],
            ax=0,
            ay=-40
        )
        
        # Mark maximum drawdown
        drawdown = self._calculate_drawdown(equity_curve)
        min_dd_idx = drawdown.idxmin()
        fig.add_annotation(
            x=min_dd_idx,
            y=equity_curve.loc[min_dd_idx],
            text=f"Max DD: {drawdown.loc[min_dd_idx]:.1%}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=self.colors["danger"],
            ax=0,
            ay=40
        )


class DrawdownChart(BaseVisualization):
    """Create drawdown analysis visualizations"""
    
    def create(
        self,
        equity_curve: pd.Series,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Create drawdown chart"""
        # Calculate drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Underwater Plot', 'Drawdown Duration'),
            row_heights=[0.7, 0.3]
        )
        
        # Underwater plot
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                mode='lines',
                name='Drawdown',
                line=dict(color=self.colors["danger"], width=1),
                fill='tozeroy',
                fillcolor=f"rgba{tuple(int(self.colors['danger'][i:i+2], 16) for i in (1, 3, 5)) + (0.3,)}"
            ),
            row=1, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        # Duration plot
        duration = self._calculate_drawdown_duration(drawdown)
        fig.add_trace(
            go.Bar(
                x=duration.index,
                y=duration.values,
                name='Duration (days)',
                marker_color=self.colors["warning"]
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Drawdown Analysis',
            template=self.plotly_template,
            showlegend=False,
            height=600
        )
        
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=1)
        fig.update_yaxes(title_text="Days", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        # Add statistics annotations
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        avg_dd = drawdown[drawdown < 0].mean() if any(drawdown < 0) else 0
        
        stats_text = f"Max Drawdown: {max_dd:.1%}<br>"
        stats_text += f"Date: {max_dd_idx.strftime('%Y-%m-%d')}<br>"
        stats_text += f"Average Drawdown: {avg_dd:.1%}"
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.95,
            text=stats_text,
            showarrow=False,
            font=dict(size=10),
            bgcolor=self.colors["background"],
            bordercolor=self.colors["text"],
            borderwidth=1
        )
        
        # Save if requested
        self.save_figure(fig, save_path)
        
        return {
            "figure": fig,
            "data": {
                "max_drawdown": max_dd,
                "max_drawdown_date": max_dd_idx,
                "avg_drawdown": avg_dd,
                "current_drawdown": drawdown.iloc[-1]
            }
        }
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> pd.Series:
        """Calculate duration of drawdowns"""
        # Create a series to track duration
        duration = pd.Series(0, index=drawdown.index)
        current_duration = 0
        
        for i in range(len(drawdown)):
            if drawdown.iloc[i] < 0:
                current_duration += 1
                duration.iloc[i] = current_duration
            else:
                current_duration = 0
                
        return duration


class ReturnsDistribution(BaseVisualization):
    """Create returns distribution visualizations"""
    
    def create(
        self,
        returns: pd.Series,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Create returns distribution chart"""
        # Create figure with subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'Returns Distribution',
                'Q-Q Plot',
                'Returns Over Time',
                'Rolling Volatility'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Histogram with KDE
        hist_data = returns.dropna()
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=hist_data.values * 100,
                name='Returns',
                nbinsx=50,
                marker_color=self.colors["primary"],
                opacity=0.7,
                histnorm='probability'
            ),
            row=1, col=1
        )
        
        # Add normal distribution overlay
        x_range = np.linspace(hist_data.min() * 100, hist_data.max() * 100, 100)
        normal_dist = self._normal_distribution(x_range, hist_data.mean() * 100, hist_data.std() * 100)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_dist,
                mode='lines',
                name='Normal',
                line=dict(color=self.colors["secondary"], width=2)
            ),
            row=1, col=1
        )
        
        # 2. Q-Q Plot
        qq_data = self._calculate_qq_data(hist_data)
        fig.add_trace(
            go.Scatter(
                x=qq_data['theoretical'],
                y=qq_data['sample'] * 100,
                mode='markers',
                name='Q-Q',
                marker=dict(color=self.colors["primary"], size=5)
            ),
            row=1, col=2
        )
        
        # Add diagonal line
        diag_range = [qq_data['theoretical'].min(), qq_data['theoretical'].max()]
        fig.add_trace(
            go.Scatter(
                x=diag_range,
                y=diag_range,
                mode='lines',
                name='Normal Line',
                line=dict(color=self.colors["secondary"], dash='dash')
            ),
            row=1, col=2
        )
        
        # 3. Returns over time
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=returns.values * 100,
                mode='lines',
                name='Daily Returns',
                line=dict(color=self.colors["primary"], width=1)
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # 4. Rolling volatility
        rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='30-day Vol',
                line=dict(color=self.colors["warning"], width=2)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Returns Analysis',
            template=self.plotly_template,
            showlegend=False,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Returns (%)", row=1, col=1)
        fig.update_yaxes(title_text="Probability", row=1, col=1)
        
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles (%)", row=1, col=2)
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=2)
        
        # Add statistics box
        stats_text = self._create_returns_statistics(returns)
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            text=stats_text,
            showarrow=False,
            font=dict(size=10),
            bgcolor=self.colors["background"],
            bordercolor=self.colors["text"],
            borderwidth=1
        )
        
        # Save if requested
        self.save_figure(fig, save_path)
        
        return {
            "figure": fig,
            "data": {
                "mean_return": returns.mean(),
                "std_return": returns.std(),
                "skewness": returns.skew(),
                "kurtosis": returns.kurtosis(),
                "var_95": returns.quantile(0.05),
                "cvar_95": returns[returns <= returns.quantile(0.05)].mean()
            }
        }
    
    def _normal_distribution(self, x: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Calculate normal distribution values"""
        from scipy import stats
        return stats.norm.pdf(x, mean, std) * (x[1] - x[0])
    
    def _calculate_qq_data(self, returns: pd.Series) -> Dict[str, np.ndarray]:
        """Calculate Q-Q plot data"""
        from scipy import stats
        
        # Sort returns
        sorted_returns = np.sort(returns.dropna())
        
        # Calculate theoretical quantiles
        n = len(sorted_returns)
        theoretical_quantiles = stats.norm.ppf((np.arange(n) + 0.5) / n)
        
        return {
            'theoretical': theoretical_quantiles,
            'sample': sorted_returns
        }
    
    def _create_returns_statistics(self, returns: pd.Series) -> str:
        """Create returns statistics text"""
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        stats_text = "<b>Returns Statistics</b><br>"
        stats_text += f"Annual Return: {annual_return:.1%}<br>"
        stats_text += f"Annual Volatility: {annual_vol:.1%}<br>"
        stats_text += f"Sharpe Ratio: {sharpe:.2f}<br>"
        stats_text += f"Skewness: {returns.skew():.2f}<br>"
        stats_text += f"Kurtosis: {returns.kurtosis():.2f}"
        
        return stats_text


class TradeScatterPlot(BaseVisualization):
    """Create trade analysis scatter plots"""
    
    def create(
        self,
        trades: pd.DataFrame,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Create trade scatter plot"""
        if trades.empty:
            return {"message": "No trades to visualize"}
        
        # Create figure
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'Trade P&L Distribution',
                'Trade Duration vs P&L',
                'Entry Time Analysis',
                'Trade Sequence'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. P&L Distribution
        winners = trades[trades['pnl'] > 0]
        losers = trades[trades['pnl'] <= 0]
        
        fig.add_trace(
            go.Bar(
                x=['Winners', 'Losers'],
                y=[len(winners), len(losers)],
                marker_color=[self.colors["success"], self.colors["danger"]],
                text=[f'{len(winners)}', f'{len(losers)}'],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Duration vs P&L
        if 'duration' in trades.columns:
            fig.add_trace(
                go.Scatter(
                    x=trades['duration'],
                    y=trades['pnl'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=trades['pnl'],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="P&L")
                    ),
                    text=[f"Trade {i+1}" for i in range(len(trades))],
                    hovertemplate='Duration: %{x:.1f}h<br>P&L: %{y:.2f}<br>%{text}'
                ),
                row=1, col=2
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        
        # 3. Entry time analysis
        if 'entry_time' in trades.columns:
            trades['hour'] = pd.to_datetime(trades['entry_time']).dt.hour
            hourly_pnl = trades.groupby('hour')['pnl'].agg(['sum', 'count', 'mean'])
            
            fig.add_trace(
                go.Bar(
                    x=hourly_pnl.index,
                    y=hourly_pnl['sum'],
                    name='Total P&L',
                    marker_color=self.colors["primary"],
                    yaxis='y',
                    offsetgroup=1
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=hourly_pnl.index,
                    y=hourly_pnl['count'],
                    name='Trade Count',
                    line=dict(color=self.colors["secondary"], width=2),
                    yaxis='y2',
                    mode='lines+markers'
                ),
                row=2, col=1
            )
        
        # 4. Trade sequence
        trades['cumulative_pnl'] = trades['pnl'].cumsum()
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(trades))),
                y=trades['cumulative_pnl'],
                mode='lines+markers',
                line=dict(color=self.colors["primary"], width=2),
                marker=dict(
                    size=6,
                    color=['green' if pnl > 0 else 'red' for pnl in trades['pnl']]
                ),
                name='Cumulative P&L'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Trade Analysis',
            template=self.plotly_template,
            showlegend=False,
            height=800
        )
        
        # Update axes
        fig.update_xaxes(title_text="Trade Type", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        
        if 'duration' in trades.columns:
            fig.update_xaxes(title_text="Duration (hours)", row=1, col=2)
            fig.update_yaxes(title_text="P&L", row=1, col=2)
        
        fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
        fig.update_yaxes(title_text="Total P&L", row=2, col=1)
        
        fig.update_xaxes(title_text="Trade Number", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative P&L", row=2, col=2)
        
        # Save if requested
        self.save_figure(fig, save_path)
        
        return {
            "figure": fig,
            "data": {
                "total_trades": len(trades),
                "winning_trades": len(winners),
                "losing_trades": len(losers),
                "win_rate": len(winners) / len(trades) if len(trades) > 0 else 0,
                "avg_winner": winners['pnl'].mean() if not winners.empty else 0,
                "avg_loser": losers['pnl'].mean() if not losers.empty else 0
            }
        }


class RollingMetricsChart(BaseVisualization):
    """Create rolling metrics visualizations"""
    
    def create(
        self,
        equity_curve: pd.Series,
        window: int = 252,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Create rolling metrics chart"""
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Calculate rolling metrics
        rolling_return = returns.rolling(window).mean() * 252
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol
        
        # Calculate rolling drawdown
        rolling_dd = pd.Series(index=equity_curve.index, dtype=float)
        for i in range(window, len(equity_curve)):
            window_data = equity_curve.iloc[i-window:i]
            dd = (window_data.iloc[-1] - window_data.max()) / window_data.max()
            rolling_dd.iloc[i] = dd
        
        # Create figure
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'{window}-Day Rolling Return',
                f'{window}-Day Rolling Volatility',
                f'{window}-Day Rolling Sharpe Ratio',
                f'{window}-Day Rolling Maximum Drawdown'
            ),
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )
        
        # 1. Rolling Return
        fig.add_trace(
            go.Scatter(
                x=rolling_return.index,
                y=rolling_return.values * 100,
                mode='lines',
                name='Rolling Return',
                line=dict(color=self.colors["primary"], width=2)
            ),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        # 2. Rolling Volatility
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values * 100,
                mode='lines',
                name='Rolling Volatility',
                line=dict(color=self.colors["warning"], width=2)
            ),
            row=2, col=1
        )
        
        # Add volatility bands
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()
        fig.add_hline(y=vol_mean * 100, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_hline(y=(vol_mean + vol_std) * 100, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=(vol_mean - vol_std) * 100, line_dash="dot", line_color="green", row=2, col=1)
        
        # 3. Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color=self.colors["info"], width=2)
            ),
            row=3, col=1
        )
        
        # Add Sharpe threshold lines
        fig.add_hline(y=1, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        
        # 4. Rolling Drawdown
        fig.add_trace(
            go.Scatter(
                x=rolling_dd.index,
                y=rolling_dd.values * 100,
                mode='lines',
                name='Rolling Max DD',
                line=dict(color=self.colors["danger"], width=2),
                fill='tozeroy',
                fillcolor=f"rgba{tuple(int(self.colors['danger'][i:i+2], 16) for i in (1, 3, 5)) + (0.2,)}"
            ),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'Rolling {window}-Day Metrics',
            template=self.plotly_template,
            showlegend=False,
            height=1000
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=4, col=1)
        fig.update_xaxes(title_text="Date", row=4, col=1)
        
        # Save if requested
        self.save_figure(fig, save_path)
        
        return {
            "figure": fig,
            "data": {
                "current_rolling_return": rolling_return.iloc[-1] if not rolling_return.empty else 0,
                "current_rolling_vol": rolling_vol.iloc[-1] if not rolling_vol.empty else 0,
                "current_rolling_sharpe": rolling_sharpe.iloc[-1] if not rolling_sharpe.empty else 0,
                "avg_rolling_sharpe": rolling_sharpe.mean() if not rolling_sharpe.empty else 0,
                "sharpe_consistency": (rolling_sharpe > 1).mean() if not rolling_sharpe.empty else 0
            }
        }


class HeatmapVisualization(BaseVisualization):
    """Create heatmap visualizations for correlation and performance matrices"""
    
    def create(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        chart_type: str = "correlation",
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Create heatmap visualization"""
        
        if chart_type == "correlation":
            fig = self._create_correlation_heatmap(data)
        elif chart_type == "monthly_returns":
            fig = self._create_monthly_returns_heatmap(data)
        elif chart_type == "parameter_sensitivity":
            fig = self._create_parameter_sensitivity_heatmap(data)
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")
        
        # Save if requested
        self.save_figure(fig, save_path)
        
        return {"figure": fig, "type": chart_type}
    
    def _create_correlation_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Create correlation matrix heatmap"""
        if isinstance(data, dict):
            # Convert dict to DataFrame
            data = pd.DataFrame(data)
        
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        # Update layout
        fig.update_layout(
            title='Correlation Matrix',
            template=self.plotly_template,
            width=800,
            height=800
        )
        
        return fig
    
    def _create_monthly_returns_heatmap(self, returns: pd.Series) -> go.Figure:
        """Create monthly returns heatmap"""
        if isinstance(returns, dict):
            returns = pd.Series(returns)
        
        # Convert to monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create year-month matrix
        monthly_returns_df = pd.DataFrame(monthly_returns)
        monthly_returns_df['Year'] = monthly_returns_df.index.year
        monthly_returns_df['Month'] = monthly_returns_df.index.month
        
        # Pivot to create heatmap data
        heatmap_data = monthly_returns_df.pivot(
            index='Year',
            columns='Month',
            values=0
        )
        
        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values * 100,
            x=month_names,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            zmid=0,
            text=(heatmap_data * 100).round(1).values,
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Return (%)")
        ))
        
        # Update layout
        fig.update_layout(
            title='Monthly Returns Heatmap',
            xaxis_title='Month',
            yaxis_title='Year',
            template=self.plotly_template,
            width=900,
            height=600
        )
        
        return fig
    
    def _create_parameter_sensitivity_heatmap(self, data: Dict[str, Any]) -> go.Figure:
        """Create parameter sensitivity heatmap"""
        # Extract parameter results
        param1_values = data.get('param1_values', [])
        param2_values = data.get('param2_values', [])
        results_matrix = data.get('results', [[]])
        metric_name = data.get('metric', 'Sharpe Ratio')
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=results_matrix,
            x=param1_values,
            y=param2_values,
            colorscale='Viridis',
            text=np.round(results_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 8},
            colorbar=dict(title=metric_name)
        ))
        
        # Find optimal point
        max_idx = np.unravel_index(np.argmax(results_matrix), results_matrix.shape)
        optimal_x = param1_values[max_idx[1]]
        optimal_y = param2_values[max_idx[0]]
        
        # Add marker for optimal point
        fig.add_trace(go.Scatter(
            x=[optimal_x],
            y=[optimal_y],
            mode='markers',
            marker=dict(
                size=20,
                color='red',
                symbol='star',
                line=dict(color='white', width=2)
            ),
            showlegend=False,
            hovertext=f'Optimal: {metric_name}={results_matrix[max_idx]:.2f}'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Parameter Sensitivity: {metric_name}',
            xaxis_title=data.get('param1_name', 'Parameter 1'),
            yaxis_title=data.get('param2_name', 'Parameter 2'),
            template=self.plotly_template,
            width=800,
            height=600
        )
        
        return fig


class TradePriceChart(BaseVisualization):
    """Create trade price analysis visualizations"""
    
    def create(
        self,
        trades: pd.DataFrame,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Create trade price chart"""
        if trades.empty:
            return {"message": "No trades to visualize"}
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'Entry vs Exit Prices',
                'Stop Loss Analysis',
                'Risk-Reward Distribution',
                'Price Movement Analysis'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Entry vs Exit Prices scatter plot
        if 'entry_price' in trades.columns and 'exit_price' in trades.columns:
            winners = trades[trades['pnl'] > 0]
            losers = trades[trades['pnl'] <= 0]
            
            # Add winning trades
            if not winners.empty:
                fig.add_trace(
                    go.Scatter(
                        x=winners['entry_price'],
                        y=winners['exit_price'],
                        mode='markers',
                        name='Winners',
                        marker=dict(
                            size=8,
                            color=self.colors["success"],
                            symbol='circle'
                        ),
                        hovertemplate='Entry: $%{x:.2f}<br>Exit: $%{y:.2f}<br>P&L: %{customdata:.2f}<extra></extra>',
                        customdata=winners['pnl']
                    ),
                    row=1, col=1
                )
            
            # Add losing trades
            if not losers.empty:
                fig.add_trace(
                    go.Scatter(
                        x=losers['entry_price'],
                        y=losers['exit_price'],
                        mode='markers',
                        name='Losers',
                        marker=dict(
                            size=8,
                            color=self.colors["danger"],
                            symbol='circle'
                        ),
                        hovertemplate='Entry: $%{x:.2f}<br>Exit: $%{y:.2f}<br>P&L: %{customdata:.2f}<extra></extra>',
                        customdata=losers['pnl']
                    ),
                    row=1, col=1
                )
            
            # Add diagonal line (break-even)
            min_price = min(trades['entry_price'].min(), trades['exit_price'].min())
            max_price = max(trades['entry_price'].max(), trades['exit_price'].max())
            fig.add_trace(
                go.Scatter(
                    x=[min_price, max_price],
                    y=[min_price, max_price],
                    mode='lines',
                    name='Break-even',
                    line=dict(color='gray', dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # 2. Stop Loss Analysis
        if 'entry_price' in trades.columns and 'stop_loss_price' in trades.columns:
            trades_with_sl = trades.dropna(subset=['stop_loss_price'])
            
            if not trades_with_sl.empty:
                sl_distance = (trades_with_sl['entry_price'] - trades_with_sl['stop_loss_price']).abs() / trades_with_sl['entry_price'] * 100
                
                fig.add_trace(
                    go.Histogram(
                        x=sl_distance,
                        nbinsx=20,
                        name='Stop Loss Distance',
                        marker_color=self.colors["warning"],
                        opacity=0.7
                    ),
                    row=1, col=2
                )
        
        # 3. Risk-Reward Distribution
        if all(col in trades.columns for col in ['entry_price', 'exit_price', 'stop_loss_price']):
            trades_complete = trades.dropna(subset=['entry_price', 'exit_price', 'stop_loss_price'])
            
            if not trades_complete.empty:
                # Calculate risk and reward
                risk = (trades_complete['entry_price'] - trades_complete['stop_loss_price']).abs() / trades_complete['entry_price'] * 100
                reward = (trades_complete['exit_price'] - trades_complete['entry_price']).abs() / trades_complete['entry_price'] * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=risk,
                        y=reward,
                        mode='markers',
                        name='Risk vs Reward',
                        marker=dict(
                            size=8,
                            color=trades_complete['pnl'],
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="P&L")
                        ),
                        hovertemplate='Risk: %{x:.2f}%<br>Reward: %{y:.2f}%<br>P&L: %{customdata:.2f}<extra></extra>',
                        customdata=trades_complete['pnl']
                    ),
                    row=2, col=1
                )
                
                # Add 1:1 risk-reward line
                max_val = max(risk.max(), reward.max())
                fig.add_trace(
                    go.Scatter(
                        x=[0, max_val],
                        y=[0, max_val],
                        mode='lines',
                        name='1:1 Risk-Reward',
                        line=dict(color='gray', dash='dash'),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # 4. Price Movement Analysis
        if 'entry_price' in trades.columns and 'exit_price' in trades.columns:
            price_change = (trades['exit_price'] - trades['entry_price']) / trades['entry_price'] * 100
            
            # Create histogram of price changes
            fig.add_trace(
                go.Histogram(
                    x=price_change,
                    nbinsx=30,
                    name='Price Change Distribution',
                    marker_color=self.colors["info"],
                    opacity=0.7
                ),
                row=2, col=2
            )
            
            # Add zero line
            fig.add_vline(x=0, line_dash="dash", line_color="gray", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title='Trade Price Analysis',
            template=self.plotly_template,
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Entry Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Exit Price ($)", row=1, col=1)
        
        fig.update_xaxes(title_text="Stop Loss Distance (%)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        fig.update_xaxes(title_text="Risk (%)", row=2, col=1)
        fig.update_yaxes(title_text="Reward (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="Price Change (%)", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        # Save if requested
        self.save_figure(fig, save_path)
        
        return {
            "figure": fig,
            "data": {
                "avg_entry_price": trades.get('entry_price', pd.Series()).mean(),
                "avg_exit_price": trades.get('exit_price', pd.Series()).mean(),
                "avg_price_change": price_change.mean() if 'entry_price' in trades.columns and 'exit_price' in trades.columns else 0,
                "price_change_volatility": price_change.std() if 'entry_price' in trades.columns and 'exit_price' in trades.columns else 0
            }
        }


class TradeRiskChart(BaseVisualization):
    """Create trade risk analysis visualizations"""
    
    def create(
        self,
        trades: pd.DataFrame,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Create trade risk analysis chart"""
        if trades.empty:
            return {"message": "No trades to visualize"}
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'MAE vs MFE Analysis',
                'Stop Loss Effectiveness',
                'Risk per Trade',
                'Trade Risk-Reward Ratio'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. MAE vs MFE Analysis
        if 'mae' in trades.columns and 'mfe' in trades.columns:
            mae_mfe_trades = trades.dropna(subset=['mae', 'mfe'])
            
            if not mae_mfe_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=mae_mfe_trades['mae'] * 100,
                        y=mae_mfe_trades['mfe'] * 100,
                        mode='markers',
                        name='MAE vs MFE',
                        marker=dict(
                            size=8,
                            color=mae_mfe_trades['pnl'],
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="P&L")
                        ),
                        hovertemplate='MAE: %{x:.2f}%<br>MFE: %{y:.2f}%<br>P&L: %{customdata:.2f}<extra></extra>',
                        customdata=mae_mfe_trades['pnl']
                    ),
                    row=1, col=1
                )
        
        # 2. Stop Loss Effectiveness
        if 'exit_reason' in trades.columns and 'pnl' in trades.columns:
            exit_reasons = trades['exit_reason'].value_counts()
            
            # Create bar chart of exit reasons
            fig.add_trace(
                go.Bar(
                    x=exit_reasons.index,
                    y=exit_reasons.values,
                    name='Exit Reasons',
                    marker_color=self.colors["primary"],
                    text=exit_reasons.values,
                    textposition='auto'
                ),
                row=1, col=2
            )
        
        # 3. Risk per Trade
        if all(col in trades.columns for col in ['entry_price', 'stop_loss_price']):
            trades_with_risk = trades.dropna(subset=['entry_price', 'stop_loss_price'])
            
            if not trades_with_risk.empty:
                risk_per_trade = (trades_with_risk['entry_price'] - trades_with_risk['stop_loss_price']).abs() / trades_with_risk['entry_price'] * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(risk_per_trade))),
                        y=risk_per_trade,
                        mode='lines+markers',
                        name='Risk per Trade',
                        line=dict(color=self.colors["warning"], width=2),
                        marker=dict(size=6)
                    ),
                    row=2, col=1
                )
                
                # Add average risk line
                avg_risk = risk_per_trade.mean()
                fig.add_hline(
                    y=avg_risk,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Avg Risk: {avg_risk:.2f}%",
                    row=2, col=1
                )
        
        # 4. Risk-Reward Ratio
        if all(col in trades.columns for col in ['entry_price', 'exit_price', 'stop_loss_price']):
            complete_trades = trades.dropna(subset=['entry_price', 'exit_price', 'stop_loss_price'])
            
            if not complete_trades.empty:
                # Calculate risk-reward ratio
                reward = (complete_trades['exit_price'] - complete_trades['entry_price']).abs() / complete_trades['entry_price']
                risk = (complete_trades['entry_price'] - complete_trades['stop_loss_price']).abs() / complete_trades['entry_price']
                
                risk_reward_ratio = reward / risk
                risk_reward_ratio = risk_reward_ratio.replace([np.inf, -np.inf], np.nan).dropna()
                
                if not risk_reward_ratio.empty:
                    fig.add_trace(
                        go.Histogram(
                            x=risk_reward_ratio,
                            nbinsx=20,
                            name='Risk-Reward Ratio',
                            marker_color=self.colors["info"],
                            opacity=0.7
                        ),
                        row=2, col=2
                    )
                    
                    # Add 1:1 ratio line
                    fig.add_vline(x=1, line_dash="dash", line_color="gray", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title='Trade Risk Analysis',
            template=self.plotly_template,
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="MAE (%)", row=1, col=1)
        fig.update_yaxes(title_text="MFE (%)", row=1, col=1)
        
        fig.update_xaxes(title_text="Exit Reason", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        fig.update_xaxes(title_text="Trade Number", row=2, col=1)
        fig.update_yaxes(title_text="Risk (%)", row=2, col=1)
        
        fig.update_xaxes(title_text="Risk-Reward Ratio", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        # Save if requested
        self.save_figure(fig, save_path)
        
        return {
            "figure": fig,
            "data": {
                "avg_mae": trades.get('mae', pd.Series()).mean(),
                "avg_mfe": trades.get('mfe', pd.Series()).mean(),
                "avg_risk_reward": risk_reward_ratio.mean() if 'risk_reward_ratio' in locals() and not risk_reward_ratio.empty else 0
            }
        }