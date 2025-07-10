"""
Simplified Risk Dashboard

A simpler version of the risk dashboard that avoids subplot compatibility issues.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRiskDashboard:
    """
    Simplified risk monitoring dashboard.
    """
    
    def __init__(self):
        """Initialize dashboard."""
        self.colors = {
            'safe': '#00ff88',
            'warning': '#f5a442',
            'danger': '#ff3366',
            'neutral': '#4287f5'
        }
    
    def create_risk_overview(
        self,
        risk_metrics: Dict[str, float],
        risk_limits: Dict[str, float],
        portfolio_value: float,
        positions: Dict[str, float]
    ) -> go.Figure:
        """Create simplified risk overview."""
        # Create subplots with compatible types only
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Risk Metrics', 'Position Allocation',
                'Risk Limit Usage', 'Key Statistics'
            )
        )
        
        # 1. Risk Metrics Bar Chart
        metrics = ['VaR 95%', 'Volatility', 'Max DD', 'Correlation']
        values = [
            abs(risk_metrics.get('var_95', 0)) * 100,
            risk_metrics.get('volatility', 0) * 100,
            abs(risk_metrics.get('max_drawdown', 0)) * 100,
            risk_metrics.get('avg_correlation', 0) * 100
        ]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                marker_color=[self._get_metric_color(m, v) for m, v in zip(metrics, values)],
                text=[f"{v:.1f}%" for v in values],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Position Allocation
        symbols = list(positions.keys())
        position_values = list(positions.values())
        
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=position_values,
                marker_color=self.colors['neutral'],
                text=[f"${v/1000:.1f}k" for v in position_values],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. Risk Limit Usage
        limit_names = ['VaR', 'Vol', 'DD', 'Leverage']
        usages = [
            abs(risk_metrics.get('var_95', 0)) / risk_limits.get('max_var_95', 0.05) * 100,
            risk_metrics.get('volatility', 0) / risk_limits.get('max_volatility', 0.25) * 100,
            abs(risk_metrics.get('current_drawdown', 0)) / risk_limits.get('max_drawdown', 0.15) * 100,
            risk_metrics.get('leverage', 1.0) / risk_limits.get('max_leverage', 1.0) * 100
        ]
        
        colors = [self.colors['danger'] if u > 80 else self.colors['warning'] if u > 60 else self.colors['safe'] 
                 for u in usages]
        
        fig.add_trace(
            go.Bar(
                x=limit_names,
                y=usages,
                marker_color=colors,
                text=[f"{u:.0f}%" for u in usages],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # 4. Key Statistics
        stats_names = ['Annual Return', 'Sharpe Ratio', 'Win Rate', 'Max DD']
        stats_values = [
            risk_metrics.get('daily_return', 0) * 252 * 100,
            risk_metrics.get('sharpe_ratio', 0),
            risk_metrics.get('win_rate', 0.5) * 100,
            abs(risk_metrics.get('max_drawdown', 0)) * 100
        ]
        
        fig.add_trace(
            go.Bar(
                x=stats_names,
                y=stats_values,
                marker_color=[
                    self.colors['safe'] if stats_values[0] > 0 else self.colors['danger'],
                    self.colors['safe'] if stats_values[1] > 1 else self.colors['warning'],
                    self.colors['safe'] if stats_values[2] > 50 else self.colors['warning'],
                    self.colors['danger'] if stats_values[3] > 20 else self.colors['warning']
                ],
                text=[
                    f"{stats_values[0]:.1f}%",
                    f"{stats_values[1]:.2f}",
                    f"{stats_values[2]:.0f}%",
                    f"{stats_values[3]:.1f}%"
                ],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Portfolio Risk Overview',
            template='plotly_dark',
            height=800,
            showlegend=False
        )
        
        # Add reference lines
        fig.add_hline(y=100, line_dash="dash", line_color="red", 
                     annotation_text="Limit", row=2, col=1)
        
        return fig
    
    def create_stress_test_summary(
        self,
        stress_results: pd.DataFrame
    ) -> go.Figure:
        """Create stress test summary visualization."""
        fig = go.Figure()
        
        # Scenario impact bar chart
        scenarios = stress_results['scenario'].tolist()
        impacts = stress_results['portfolio_return'].tolist()
        
        colors = [self.colors['danger'] if impact < -0.1 else self.colors['warning'] 
                 for impact in impacts]
        
        fig.add_trace(
            go.Bar(
                x=scenarios,
                y=[i * 100 for i in impacts],
                marker_color=colors,
                text=[f"{i*100:.1f}%" for i in impacts],
                textposition='auto'
            )
        )
        
        fig.update_layout(
            title='Stress Test Scenario Impacts',
            xaxis_title='Scenario',
            yaxis_title='Portfolio Impact (%)',
            template='plotly_dark',
            height=500
        )
        
        return fig
    
    def _get_metric_color(self, metric: str, value: float) -> str:
        """Get color based on metric and value."""
        thresholds = {
            'VaR 95%': (3, 5),
            'Volatility': (15, 25),
            'Max DD': (10, 20),
            'Correlation': (60, 80)
        }
        
        low, high = thresholds.get(metric, (50, 80))
        
        if value < low:
            return self.colors['safe']
        elif value < high:
            return self.colors['warning']
        else:
            return self.colors['danger']
    
    def save_dashboard(self, fig: go.Figure, filepath: str):
        """Save dashboard to HTML file."""
        fig.write_html(
            filepath,
            include_plotlyjs='cdn',
            config={'displayModeBar': True, 'displaylogo': False}
        )
        logger.info(f"Dashboard saved to {filepath}")