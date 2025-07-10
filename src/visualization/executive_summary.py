"""
Executive Summary Dashboard

This module creates a comprehensive executive summary with key performance
metrics and insights for the enhanced confluence strategy.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExecutiveSummaryDashboard:
    """
    Creates executive summary dashboard with key metrics and visualizations.
    """
    
    def __init__(self):
        """Initialize dashboard creator."""
        self.metric_colors = {
            'positive': '#00ff88',
            'negative': '#ff3366',
            'neutral': '#4287f5',
            'benchmark': '#f5a442'
        }
    
    def create_dashboard(
        self,
        strategy_results: Dict[str, Any],
        benchmark_results: Dict[str, Any],
        trades: List[Dict],
        equity_curve: pd.Series
    ) -> str:
        """
        Create complete executive summary dashboard.
        
        Args:
            strategy_results: Strategy performance metrics
            benchmark_results: Benchmark performance metrics
            trades: List of trades
            equity_curve: Portfolio equity over time
            
        Returns:
            HTML string with complete dashboard
        """
        # Create all dashboard components
        metrics_fig = self._create_key_metrics_panel(strategy_results, benchmark_results)
        equity_fig = self._create_equity_curve_chart(equity_curve, benchmark_results)
        comparison_fig = self._create_comparison_table(strategy_results, benchmark_results)
        risk_fig = self._create_risk_metrics_panel(strategy_results, benchmark_results)
        
        # Generate HTML
        html_content = self._generate_dashboard_html(
            metrics_fig, equity_fig, comparison_fig, risk_fig,
            strategy_results, benchmark_results
        )
        
        return html_content
    
    def _create_key_metrics_panel(
        self,
        strategy_results: Dict[str, Any],
        benchmark_results: Dict[str, Any]
    ) -> go.Figure:
        """Create key metrics panel with gauge charts."""
        fig = make_subplots(
            rows=2, cols=3,
            specs=[
                [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]
            ],
            subplot_titles=('Total Return', 'Sharpe Ratio', 'Win Rate',
                          'Max Drawdown', 'Profit Factor', 'Alpha vs Benchmark')
        )
        
        # Total Return
        total_return = strategy_results.get('total_return', 0) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=total_return,
                delta={'reference': benchmark_results.get('total_return', 0) * 100},
                gauge={'axis': {'range': [-50, 100]},
                      'bar': {'color': self.metric_colors['positive'] if total_return > 0 else self.metric_colors['negative']},
                      'threshold': {'value': 0}},
                title={'text': "Total Return (%)"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=1
        )
        
        # Sharpe Ratio
        sharpe = strategy_results.get('sharpe_ratio', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=sharpe,
                gauge={'axis': {'range': [0, 3]},
                      'bar': {'color': self.metric_colors['positive'] if sharpe > 1.5 else self.metric_colors['neutral']},
                      'threshold': {'value': 1.5}},
                title={'text': "Sharpe Ratio"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=2
        )
        
        # Win Rate
        win_rate = strategy_results.get('win_rate', 0) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=win_rate,
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': self.metric_colors['positive'] if win_rate > 60 else self.metric_colors['neutral']},
                      'threshold': {'value': 60}},
                title={'text': "Win Rate (%)"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=3
        )
        
        # Max Drawdown
        max_dd = abs(strategy_results.get('max_drawdown', 0)) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=max_dd,
                gauge={'axis': {'range': [0, 50]},
                      'bar': {'color': self.metric_colors['positive'] if max_dd < 10 else self.metric_colors['negative']},
                      'threshold': {'value': 10}},
                title={'text': "Max Drawdown (%)"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=2, col=1
        )
        
        # Profit Factor
        profit_factor = strategy_results.get('profit_factor', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=profit_factor,
                gauge={'axis': {'range': [0, 4]},
                      'bar': {'color': self.metric_colors['positive'] if profit_factor > 1.5 else self.metric_colors['neutral']},
                      'threshold': {'value': 1.5}},
                title={'text': "Profit Factor"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=2, col=2
        )
        
        # Alpha
        alpha = (strategy_results.get('total_return', 0) - 
                benchmark_results.get('total_return', 0)) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=alpha,
                gauge={'axis': {'range': [-20, 50]},
                      'bar': {'color': self.metric_colors['positive'] if alpha > 0 else self.metric_colors['negative']},
                      'threshold': {'value': 0}},
                title={'text': "Alpha (%)"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=600,
            showlegend=False,
            title_text="Key Performance Metrics"
        )
        
        return fig
    
    def _create_equity_curve_chart(
        self,
        equity_curve: pd.Series,
        benchmark_results: Dict[str, Any]
    ) -> go.Figure:
        """Create equity curve comparison chart."""
        fig = go.Figure()
        
        # Strategy equity curve
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            name='Strategy',
            line=dict(color=self.metric_colors['positive'], width=2)
        ))
        
        # Benchmark equity curve (simulated)
        if 'equity_curve' in benchmark_results:
            fig.add_trace(go.Scatter(
                x=benchmark_results['equity_curve'].index,
                y=benchmark_results['equity_curve'].values,
                name='Benchmark',
                line=dict(color=self.metric_colors['benchmark'], width=2, dash='dash')
            ))
        
        # Add drawdown area
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            name='Drawdown',
            yaxis='y2',
            line=dict(color=self.metric_colors['negative'], width=1),
            fill='tozeroy',
            fillcolor='rgba(255,51,102,0.2)'
        ))
        
        fig.update_layout(
            title='Portfolio Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value',
            yaxis2=dict(
                title='Drawdown (%)',
                overlaying='y',
                side='right',
                range=[-30, 5]
            ),
            template='plotly_dark',
            height=400,
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
    
    def _create_comparison_table(
        self,
        strategy_results: Dict[str, Any],
        benchmark_results: Dict[str, Any]
    ) -> go.Figure:
        """Create strategy vs benchmark comparison table."""
        metrics = [
            'Total Return',
            'Annual Return',
            'Sharpe Ratio',
            'Max Drawdown',
            'Volatility',
            'Win Rate',
            'Total Trades',
            'Profit Factor',
            'Calmar Ratio',
            'Sortino Ratio'
        ]
        
        strategy_values = []
        benchmark_values = []
        differences = []
        
        # Map metric names to result keys
        metric_map = {
            'Total Return': 'total_return',
            'Annual Return': 'annual_return',
            'Sharpe Ratio': 'sharpe_ratio',
            'Max Drawdown': 'max_drawdown',
            'Volatility': 'volatility',
            'Win Rate': 'win_rate',
            'Total Trades': 'total_trades',
            'Profit Factor': 'profit_factor',
            'Calmar Ratio': 'calmar_ratio',
            'Sortino Ratio': 'sortino_ratio'
        }
        
        for metric in metrics:
            key = metric_map.get(metric, metric.lower().replace(' ', '_'))
            
            # Get values
            strat_val = strategy_results.get(key, 0)
            bench_val = benchmark_results.get(key, 0)
            
            # Format based on metric type
            if metric in ['Total Return', 'Annual Return', 'Max Drawdown', 'Volatility', 'Win Rate']:
                strategy_values.append(f"{strat_val*100:.2f}%")
                benchmark_values.append(f"{bench_val*100:.2f}%")
                diff = (strat_val - bench_val) * 100
                differences.append(f"{diff:+.2f}%")
            elif metric == 'Total Trades':
                strategy_values.append(f"{int(strat_val)}")
                benchmark_values.append(f"{int(bench_val)}")
                differences.append(f"{int(strat_val - bench_val):+d}")
            else:
                strategy_values.append(f"{strat_val:.2f}")
                benchmark_values.append(f"{bench_val:.2f}")
                differences.append(f"{strat_val - bench_val:+.2f}")
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Metric', 'Strategy', 'Benchmark', 'Difference'],
                fill_color='#1a1f2e',
                align='left',
                font=dict(size=14, color='white')
            ),
            cells=dict(
                values=[metrics, strategy_values, benchmark_values, differences],
                fill_color=['#0a0e1a'] * 4,
                align='left',
                font=dict(size=12),
                height=30
            )
        )])
        
        fig.update_layout(
            title='Strategy vs Benchmark Comparison',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def _create_risk_metrics_panel(
        self,
        strategy_results: Dict[str, Any],
        benchmark_results: Dict[str, Any]
    ) -> go.Figure:
        """Create risk metrics visualization panel."""
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'bar'}, {'type': 'scatter'}, {'type': 'pie'}]],
            subplot_titles=('Risk Metrics', 'Risk-Return Profile', 'Risk Allocation')
        )
        
        # Risk metrics bar chart
        risk_metrics = {
            'VaR 95%': abs(strategy_results.get('var_95', 0)) * 100,
            'CVaR 95%': abs(strategy_results.get('cvar_95', 0)) * 100,
            'Downside Dev': strategy_results.get('downside_deviation', 0) * 100,
            'Beta': strategy_results.get('beta', 1),
            'Correlation': strategy_results.get('correlation', 0)
        }
        
        fig.add_trace(
            go.Bar(
                x=list(risk_metrics.keys()),
                y=list(risk_metrics.values()),
                marker_color=self.metric_colors['neutral'],
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Risk-return scatter (simulated for multiple strategies)
        returns = [strategy_results.get('annual_return', 0) * 100]
        risks = [strategy_results.get('volatility', 0) * 100]
        labels = ['Current Strategy']
        
        # Add benchmark
        if benchmark_results:
            returns.append(benchmark_results.get('annual_return', 0) * 100)
            risks.append(benchmark_results.get('volatility', 0) * 100)
            labels.append('Benchmark')
        
        fig.add_trace(
            go.Scatter(
                x=risks,
                y=returns,
                mode='markers+text',
                text=labels,
                textposition='top center',
                marker=dict(size=15, color=[self.metric_colors['positive'], 
                                           self.metric_colors['benchmark']]),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Risk allocation pie (example)
        risk_allocation = {
            'Market Risk': 40,
            'Timing Risk': 25,
            'Selection Risk': 20,
            'Concentration Risk': 15
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(risk_allocation.keys()),
                values=list(risk_allocation.values()),
                hole=0.3,
                showlegend=True
            ),
            row=1, col=3
        )
        
        # Update axes
        fig.update_xaxes(title_text="Metric", row=1, col=1)
        fig.update_xaxes(title_text="Risk (Volatility %)", row=1, col=2)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=1, col=2)
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def _generate_dashboard_html(
        self,
        metrics_fig: go.Figure,
        equity_fig: go.Figure,
        comparison_fig: go.Figure,
        risk_fig: go.Figure,
        strategy_results: Dict[str, Any],
        benchmark_results: Dict[str, Any]
    ) -> str:
        """Generate complete HTML dashboard."""
        # Convert figures to HTML
        metrics_html = metrics_fig.to_html(include_plotlyjs=False, div_id="metrics-panel")
        equity_html = equity_fig.to_html(include_plotlyjs=False, div_id="equity-curve")
        comparison_html = comparison_fig.to_html(include_plotlyjs=False, div_id="comparison-table")
        risk_html = risk_fig.to_html(include_plotlyjs=False, div_id="risk-panel")
        
        # Generate insights
        insights = self._generate_insights(strategy_results, benchmark_results)
        
        # HTML template
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Confluence Strategy - Executive Summary</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #0a0e1a;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            padding: 20px;
            background-color: #1a1f2e;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .insights {{
            background-color: #1a1f2e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .insights h3 {{
            color: #4287f5;
        }}
        .insight-item {{
            margin: 10px 0;
            padding: 10px;
            background-color: #0a0e1a;
            border-radius: 5px;
        }}
        .metric-positive {{
            color: #00ff88;
        }}
        .metric-negative {{
            color: #ff3366;
        }}
        .chart-container {{
            margin-bottom: 20px;
            background-color: #1a1f2e;
            padding: 10px;
            border-radius: 10px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Enhanced Confluence Strategy - Executive Summary</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="insights">
        <h3>Key Insights</h3>
        {insights}
    </div>
    
    <div class="chart-container">
        {metrics_html}
    </div>
    
    <div class="chart-container">
        {equity_html}
    </div>
    
    <div class="chart-container">
        {comparison_html}
    </div>
    
    <div class="chart-container">
        {risk_html}
    </div>
    
    <div class="footer">
        <p>Enhanced Confluence Strategy Report - Phase 3 Implementation</p>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def _generate_insights(
        self,
        strategy_results: Dict[str, Any],
        benchmark_results: Dict[str, Any]
    ) -> str:
        """Generate key insights from results."""
        insights = []
        
        # Performance insight
        total_return = strategy_results.get('total_return', 0)
        bench_return = benchmark_results.get('total_return', 0)
        alpha = total_return - bench_return
        
        if alpha > 0:
            insights.append(
                f'<div class="insight-item">✅ Strategy outperformed benchmark by '
                f'<span class="metric-positive">{alpha*100:.1f}%</span></div>'
            )
        else:
            insights.append(
                f'<div class="insight-item">❌ Strategy underperformed benchmark by '
                f'<span class="metric-negative">{abs(alpha)*100:.1f}%</span></div>'
            )
        
        # Risk insight
        sharpe = strategy_results.get('sharpe_ratio', 0)
        if sharpe > 1.5:
            insights.append(
                f'<div class="insight-item">✅ Excellent risk-adjusted returns with '
                f'<span class="metric-positive">Sharpe ratio of {sharpe:.2f}</span></div>'
            )
        
        # Consistency insight
        win_rate = strategy_results.get('win_rate', 0)
        if win_rate > 0.6:
            insights.append(
                f'<div class="insight-item">✅ High consistency with '
                f'<span class="metric-positive">{win_rate*100:.1f}% win rate</span></div>'
            )
        
        # Drawdown insight
        max_dd = abs(strategy_results.get('max_drawdown', 0))
        if max_dd < 0.1:
            insights.append(
                f'<div class="insight-item">✅ Well-controlled risk with max drawdown of '
                f'<span class="metric-positive">{max_dd*100:.1f}%</span></div>'
            )
        
        return '\n'.join(insights)
    
    def save_dashboard(
        self,
        html_content: str,
        filename: str = 'executive_summary.html',
        output_dir: str = 'reports'
    ):
        """Save dashboard to HTML file."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Executive summary saved to {filepath}")
        return filepath