"""
Risk Monitoring Dashboard

This module creates an interactive risk monitoring dashboard for
real-time portfolio risk tracking and alerts.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskDashboard:
    """
    Interactive risk monitoring dashboard.
    """
    
    def __init__(self, update_frequency: str = 'real-time'):
        """
        Initialize risk dashboard.
        
        Args:
            update_frequency: Dashboard update frequency
        """
        self.update_frequency = update_frequency
        self.colors = {
            'safe': '#00ff88',
            'warning': '#f5a442',
            'danger': '#ff3366',
            'neutral': '#4287f5',
            'background': '#0a0e1a',
            'grid': '#1a1f2e'
        }
        
    def create_risk_overview_dashboard(
        self,
        risk_metrics: Dict[str, float],
        risk_limits: Dict[str, float],
        portfolio_value: float,
        positions: Dict[str, float]
    ) -> go.Figure:
        """
        Create main risk overview dashboard.
        
        Args:
            risk_metrics: Current risk metrics
            risk_limits: Risk limit thresholds
            portfolio_value: Total portfolio value
            positions: Current positions
            
        Returns:
            Plotly figure with risk dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Risk Metrics Status', 'VaR & CVaR', 'Volatility Trend',
                'Position Concentration', 'Correlation Matrix', 'Drawdown Analysis',
                'Risk Limits Usage', 'Stop-Loss Monitor', 'Risk Alerts'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'pie'}, {'type': 'heatmap'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'table'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Risk Metrics Status (Gauges)
        self._add_risk_gauges(fig, risk_metrics, risk_limits, row=1, col=1)
        
        # 2. VaR & CVaR
        self._add_var_cvar_chart(fig, risk_metrics, row=1, col=2)
        
        # 3. Volatility Trend
        self._add_volatility_trend(fig, risk_metrics, row=1, col=3)
        
        # 4. Position Concentration
        self._add_concentration_pie(fig, positions, portfolio_value, row=2, col=1)
        
        # 5. Correlation Matrix
        self._add_correlation_heatmap(fig, positions, row=2, col=2)
        
        # 6. Drawdown Analysis
        self._add_drawdown_chart(fig, risk_metrics, row=2, col=3)
        
        # 7. Risk Limits Usage
        self._add_risk_limits_bars(fig, risk_metrics, risk_limits, row=3, col=1)
        
        # 8. Stop-Loss Monitor
        self._add_stop_loss_table(fig, positions, row=3, col=2)
        
        # 9. Risk Alerts
        self._add_risk_alerts(fig, risk_metrics, risk_limits, row=3, col=3)
        
        # Update layout
        fig.update_layout(
            title='Portfolio Risk Monitoring Dashboard',
            template='plotly_dark',
            height=1200,
            showlegend=False,
            margin=dict(t=80, b=40, l=40, r=40)
        )
        
        return fig
    
    def create_real_time_monitor(
        self,
        risk_history: pd.DataFrame,
        current_metrics: Dict[str, float],
        alerts: List[Dict]
    ) -> go.Figure:
        """
        Create real-time risk monitoring view.
        
        Args:
            risk_history: Historical risk metrics
            current_metrics: Current risk values
            alerts: Active risk alerts
            
        Returns:
            Plotly figure with real-time monitor
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Real-Time Risk Metrics', 'Active Alerts',
                'Risk Factor Contributions', 'Intraday VaR'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'table'}],
                [{'type': 'bar'}, {'type': 'scatter'}]
            ]
        )
        
        # Real-time metrics
        for i, (metric, values) in enumerate(risk_history.items()):
            if metric in ['volatility', 'var_95', 'correlation']:
                fig.add_trace(
                    go.Scatter(
                        x=risk_history.index,
                        y=values,
                        name=metric,
                        mode='lines',
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
        
        # Active alerts table
        if alerts:
            alert_data = {
                'Time': [a['timestamp'] for a in alerts],
                'Metric': [a['metric'] for a in alerts],
                'Value': [f"{a['value']:.2%}" for a in alerts],
                'Limit': [f"{a['limit']:.2%}" for a in alerts],
                'Severity': [a['severity'] for a in alerts]
            }
            
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=list(alert_data.keys()),
                        fill_color='#1a1f2e',
                        align='left'
                    ),
                    cells=dict(
                        values=list(alert_data.values()),
                        fill_color='#0a0e1a',
                        align='left',
                        font=dict(
                            color=['white', 'white', 'white', 'white',
                                  [self._get_severity_color(s) for s in alert_data['Severity']]]
                        )
                    )
                ),
                row=1, col=2
            )
        
        # Risk factor contributions
        contributions = current_metrics.get('risk_contributions', {})
        if contributions:
            fig.add_trace(
                go.Bar(
                    x=list(contributions.keys()),
                    y=list(contributions.values()),
                    marker_color=self.colors['neutral']
                ),
                row=2, col=1
            )
        
        # Intraday VaR
        if 'intraday_var' in risk_history.columns:
            fig.add_trace(
                go.Scatter(
                    x=risk_history.index,
                    y=risk_history['intraday_var'],
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color=self.colors['danger'])
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Real-Time Risk Monitor',
            template='plotly_dark',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_stress_test_dashboard(
        self,
        stress_results: pd.DataFrame,
        scenario_impacts: Dict[str, float]
    ) -> go.Figure:
        """
        Create stress test results dashboard.
        
        Args:
            stress_results: Stress test results
            scenario_impacts: Impact by scenario
            
        Returns:
            Plotly figure with stress test dashboard
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Scenario Impacts', 'Loss Distribution',
                'Recovery Times', 'Risk Factor Sensitivity'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'histogram'}],
                [{'type': 'scatter'}, {'type': 'heatmap'}]
            ]
        )
        
        # Scenario impacts
        scenarios = stress_results['scenario'].tolist()
        impacts = stress_results['portfolio_return'].tolist()
        
        colors = [self.colors['danger'] if impact < -0.1 else self.colors['warning'] 
                 if impact < -0.05 else self.colors['safe'] for impact in impacts]
        
        fig.add_trace(
            go.Bar(
                x=scenarios,
                y=[i * 100 for i in impacts],
                marker_color=colors,
                text=[f"{i*100:.1f}%" for i in impacts],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Loss distribution
        if 'simulated_returns' in stress_results:
            fig.add_trace(
                go.Histogram(
                    x=stress_results['simulated_returns'] * 100,
                    nbinsx=50,
                    marker_color=self.colors['neutral'],
                    name='Return Distribution'
                ),
                row=1, col=2
            )
        
        # Recovery times
        fig.add_trace(
            go.Scatter(
                x=stress_results['portfolio_return'] * 100,
                y=stress_results['recovery_time'],
                mode='markers+text',
                text=scenarios,
                textposition='top center',
                marker=dict(
                    size=10,
                    color=stress_results['recovery_time'],
                    colorscale='Viridis',
                    showscale=True
                )
            ),
            row=2, col=1
        )
        
        # Risk factor sensitivity heatmap
        if 'sensitivity_matrix' in stress_results:
            fig.add_trace(
                go.Heatmap(
                    z=stress_results['sensitivity_matrix'],
                    colorscale='RdBu',
                    zmid=0
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Stress Test Results Dashboard',
            template='plotly_dark',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_position_risk_dashboard(
        self,
        positions: Dict[str, float],
        position_risks: Dict[str, Dict[str, float]],
        correlations: pd.DataFrame
    ) -> go.Figure:
        """
        Create position-level risk dashboard.
        
        Args:
            positions: Current positions
            position_risks: Risk metrics by position
            correlations: Correlation matrix
            
        Returns:
            Plotly figure with position risk dashboard
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Position Sizes', 'Individual VaRs', 'Risk Contributions',
                'Position Correlations', 'Stop-Loss Distances', 'Position P&L'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'heatmap'}, {'type': 'bar'}, {'type': 'waterfall'}]
            ]
        )
        
        symbols = list(positions.keys())
        
        # Position sizes
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=list(positions.values()),
                marker_color=self.colors['neutral'],
                name='Position Value'
            ),
            row=1, col=1
        )
        
        # Individual VaRs
        vars = [position_risks.get(s, {}).get('var_95', 0) for s in symbols]
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=vars,
                marker_color=self.colors['warning'],
                name='VaR 95%'
            ),
            row=1, col=2
        )
        
        # Risk contributions
        contributions = [position_risks.get(s, {}).get('risk_contribution', 0) for s in symbols]
        fig.add_trace(
            go.Pie(
                labels=symbols,
                values=contributions,
                hole=0.3
            ),
            row=1, col=3
        )
        
        # Correlation heatmap
        if not correlations.empty:
            fig.add_trace(
                go.Heatmap(
                    z=correlations.values,
                    x=correlations.columns,
                    y=correlations.index,
                    colorscale='RdBu',
                    zmid=0
                ),
                row=2, col=1
            )
        
        # Stop-loss distances
        sl_distances = [position_risks.get(s, {}).get('stop_distance', 0) for s in symbols]
        colors = [self.colors['danger'] if d < 0.02 else self.colors['warning'] 
                 if d < 0.05 else self.colors['safe'] for d in sl_distances]
        
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=[d * 100 for d in sl_distances],
                marker_color=colors,
                name='Stop Distance %'
            ),
            row=2, col=2
        )
        
        # Position P&L waterfall
        pnls = [position_risks.get(s, {}).get('unrealized_pnl', 0) for s in symbols]
        fig.add_trace(
            go.Waterfall(
                x=symbols + ['Total'],
                y=pnls + [sum(pnls)],
                connector={'line': {'color': 'rgb(63, 63, 63)'}}
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title='Position Risk Analysis Dashboard',
            template='plotly_dark',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def _add_risk_gauges(self, fig, risk_metrics, risk_limits, row, col):
        """Add risk metric gauges."""
        # Overall risk score (0-100)
        risk_score = self._calculate_risk_score(risk_metrics, risk_limits)
        
        # Create a simple bar chart instead of gauge for subplot compatibility
        fig.add_trace(
            go.Bar(
                x=['Risk Score'],
                y=[risk_score],
                text=[f"{risk_score:.0f}"],
                textposition='auto',
                marker_color=self._get_risk_color(risk_score),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add reference line at 50
        fig.add_hline(
            y=50,
            line_dash="dash",
            line_color="yellow",
            annotation_text="Target",
            row=row, col=col
        )
    
    def _add_var_cvar_chart(self, fig, risk_metrics, row, col):
        """Add VaR and CVaR chart."""
        metrics = ['VaR 95%', 'CVaR 95%', 'VaR 99%', 'CVaR 99%']
        values = [
            abs(risk_metrics.get('var_95', 0)) * 100,
            abs(risk_metrics.get('cvar_95', 0)) * 100,
            abs(risk_metrics.get('var_99', 0)) * 100,
            abs(risk_metrics.get('cvar_99', 0)) * 100
        ]
        
        colors = [self.colors['warning'] if v > 5 else self.colors['safe'] for v in values]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                marker_color=colors,
                text=[f"{v:.1f}%" for v in values],
                textposition='auto'
            ),
            row=row, col=col
        )
    
    def _add_volatility_trend(self, fig, risk_metrics, row, col):
        """Add volatility trend chart."""
        # Mock historical data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        current_vol = risk_metrics.get('volatility', 0.15)
        vols = np.random.normal(current_vol, current_vol * 0.1, 30)
        vols = np.maximum(0.05, vols)  # Floor at 5%
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=vols * 100,
                mode='lines',
                line=dict(color=self.colors['neutral'], width=2),
                fill='tozeroy',
                fillcolor='rgba(66, 135, 245, 0.2)'
            ),
            row=row, col=col
        )
        
        # Add current level
        fig.add_hline(
            y=current_vol * 100,
            line_dash="dash",
            line_color="yellow",
            annotation_text=f"Current: {current_vol*100:.1f}%",
            row=row, col=col
        )
    
    def _add_concentration_pie(self, fig, positions, portfolio_value, row, col):
        """Add position concentration pie chart."""
        symbols = list(positions.keys())
        values = [abs(v) for v in positions.values()]
        
        # Add "Cash" if positions don't sum to portfolio value
        position_sum = sum(values)
        if position_sum < portfolio_value * 0.95:
            symbols.append('Cash')
            values.append(portfolio_value - position_sum)
        
        fig.add_trace(
            go.Pie(
                labels=symbols,
                values=values,
                hole=0.3,
                marker=dict(
                    colors=px.colors.qualitative.Set3[:len(symbols)]
                )
            ),
            row=row, col=col
        )
    
    def _add_correlation_heatmap(self, fig, positions, row, col):
        """Add correlation heatmap."""
        # Mock correlation matrix
        n_assets = len(positions)
        corr_matrix = np.random.rand(n_assets, n_assets)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1)
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix,
                x=list(positions.keys()),
                y=list(positions.keys()),
                colorscale='RdBu',
                zmid=0,
                zmin=-1,
                zmax=1
            ),
            row=row, col=col
        )
    
    def _add_drawdown_chart(self, fig, risk_metrics, row, col):
        """Add drawdown analysis chart."""
        # Mock drawdown data
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        current_dd = risk_metrics.get('current_drawdown', 0)
        max_dd = risk_metrics.get('max_drawdown', -0.10)
        
        # Generate realistic drawdown path
        dd_values = []
        dd = 0
        for _ in range(90):
            dd = dd * 0.95 + np.random.normal(current_dd/90, 0.01)
            dd = max(dd, max_dd)
            dd = min(dd, 0)
            dd_values.append(dd)
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=[d * 100 for d in dd_values],
                mode='lines',
                fill='tozeroy',
                line=dict(color=self.colors['danger']),
                fillcolor='rgba(255, 51, 102, 0.2)'
            ),
            row=row, col=col
        )
        
        # Add max drawdown line
        fig.add_hline(
            y=max_dd * 100,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Max DD: {max_dd*100:.1f}%",
            row=row, col=col
        )
    
    def _add_risk_limits_bars(self, fig, risk_metrics, risk_limits, row, col):
        """Add risk limits usage bars."""
        limits = []
        usages = []
        colors = []
        
        # Check each limit
        limit_checks = {
            'Position Size': (
                risk_metrics.get('max_position_pct', 0),
                risk_limits.get('max_position_size', 0.20)
            ),
            'Volatility': (
                risk_metrics.get('volatility', 0),
                risk_limits.get('max_volatility', 0.25)
            ),
            'VaR 95%': (
                abs(risk_metrics.get('var_95', 0)),
                risk_limits.get('max_var_95', 0.05)
            ),
            'Drawdown': (
                abs(risk_metrics.get('current_drawdown', 0)),
                risk_limits.get('max_drawdown', 0.15)
            ),
            'Leverage': (
                risk_metrics.get('leverage', 1.0),
                risk_limits.get('max_leverage', 1.0)
            )
        }
        
        for limit_name, (usage, limit) in limit_checks.items():
            limits.append(limit_name)
            usage_pct = (usage / limit * 100) if limit > 0 else 0
            usages.append(usage_pct)
            
            if usage_pct > 90:
                colors.append(self.colors['danger'])
            elif usage_pct > 70:
                colors.append(self.colors['warning'])
            else:
                colors.append(self.colors['safe'])
        
        fig.add_trace(
            go.Bar(
                x=limits,
                y=usages,
                marker_color=colors,
                text=[f"{u:.0f}%" for u in usages],
                textposition='auto'
            ),
            row=row, col=col
        )
        
        # Add 100% line
        fig.add_hline(
            y=100,
            line_dash="dash",
            line_color="red",
            annotation_text="Limit",
            row=row, col=col
        )
    
    def _add_stop_loss_table(self, fig, positions, row, col):
        """Add stop-loss monitoring table."""
        # Mock stop-loss data
        sl_data = {
            'Symbol': list(positions.keys()),
            'Entry': ['$100', '$150', '$200', '$50'][:len(positions)],
            'Current': ['$105', '$145', '$195', '$48'][:len(positions)],
            'Stop': ['$95', '$142', '$190', '$47'][:len(positions)],
            'Distance': ['4.8%', '2.1%', '2.6%', '2.1%'][:len(positions)],
            'Status': ['Safe', 'Warning', 'Safe', 'Warning'][:len(positions)]
        }
        
        # Adjust data length
        for key in sl_data:
            sl_data[key] = sl_data[key][:len(positions)]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(sl_data.keys()),
                    fill_color='#1a1f2e',
                    align='left',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=list(sl_data.values()),
                    fill_color='#0a0e1a',
                    align='left',
                    font=dict(
                        color=['white'] * (len(sl_data) - 1) + 
                              [[self.colors['safe'] if s == 'Safe' else self.colors['warning'] 
                                for s in sl_data['Status']]]
                    )
                )
            ),
            row=row, col=col
        )
    
    def _add_risk_alerts(self, fig, risk_metrics, risk_limits, row, col):
        """Add risk alerts indicator."""
        # Count active alerts
        alerts = 0
        
        if risk_metrics.get('volatility', 0) > risk_limits.get('max_volatility', 0.25):
            alerts += 1
        if abs(risk_metrics.get('var_95', 0)) > risk_limits.get('max_var_95', 0.05):
            alerts += 1
        if abs(risk_metrics.get('current_drawdown', 0)) > risk_limits.get('max_drawdown', 0.15) * 0.8:
            alerts += 1
        
        alert_color = self.colors['safe'] if alerts == 0 else \
                     self.colors['warning'] if alerts <= 2 else \
                     self.colors['danger']
        
        # Create a simple text display for alerts
        fig.add_trace(
            go.Bar(
                x=['Active Alerts'],
                y=[alerts],
                text=[f"{alerts}"],
                textposition='outside',
                marker_color=alert_color,
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add annotations for context
        fig.add_annotation(
            text=f"<b>{alerts}</b><br>Alerts",
            xref=f"x{(row-1)*3+col}",
            yref=f"y{(row-1)*3+col}",
            x=0,
            y=alerts + 0.5,
            showarrow=False,
            font=dict(size=30, color=alert_color)
        )
    
    def _calculate_risk_score(self, risk_metrics: Dict, risk_limits: Dict) -> float:
        """Calculate overall risk score (0-100)."""
        score = 0
        weights = {
            'volatility': 0.25,
            'var_95': 0.25,
            'drawdown': 0.25,
            'correlation': 0.15,
            'concentration': 0.10
        }
        
        # Volatility component
        vol_usage = risk_metrics.get('volatility', 0) / risk_limits.get('max_volatility', 0.25)
        score += vol_usage * weights['volatility'] * 100
        
        # VaR component
        var_usage = abs(risk_metrics.get('var_95', 0)) / risk_limits.get('max_var_95', 0.05)
        score += var_usage * weights['var_95'] * 100
        
        # Drawdown component
        dd_usage = abs(risk_metrics.get('current_drawdown', 0)) / risk_limits.get('max_drawdown', 0.15)
        score += dd_usage * weights['drawdown'] * 100
        
        # Correlation component
        corr_usage = risk_metrics.get('avg_correlation', 0) / risk_limits.get('max_correlation', 0.80)
        score += corr_usage * weights['correlation'] * 100
        
        # Concentration component
        conc_usage = risk_metrics.get('concentration_hhi', 0) / risk_limits.get('max_concentration_score', 0.30)
        score += conc_usage * weights['concentration'] * 100
        
        return min(100, max(0, score))
    
    def _get_risk_color(self, score: float) -> str:
        """Get color based on risk score."""
        if score < 30:
            return self.colors['safe']
        elif score < 70:
            return self.colors['warning']
        else:
            return self.colors['danger']
    
    def _get_severity_color(self, severity: str) -> str:
        """Get color based on severity."""
        severity_colors = {
            'low': self.colors['safe'],
            'medium': self.colors['warning'],
            'high': self.colors['danger']
        }
        return severity_colors.get(severity, 'white')
    
    def save_dashboard(self, fig: go.Figure, filepath: str):
        """Save dashboard to HTML file."""
        fig.write_html(
            filepath,
            include_plotlyjs='cdn',
            config={'displayModeBar': True, 'displaylogo': False}
        )
        logger.info(f"Risk dashboard saved to {filepath}")