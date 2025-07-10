"""
Benchmark Comparison System

This module creates visualizations comparing strategy performance against
various benchmarks including buy-and-hold, market indices, and other strategies.
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

class BenchmarkComparison:
    """
    Creates benchmark comparison visualizations.
    """
    
    def __init__(self):
        """Initialize benchmark comparison system."""
        self.strategy_color = '#00ff88'
        self.benchmark_colors = {
            'buy_hold': '#4287f5',
            'spy': '#f5a442',
            'equal_weight': '#9933ff',
            '60_40': '#ff3366'
        }
    
    def create_benchmark_comparison(
        self,
        strategy_results: Dict[str, Any],
        benchmark_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, go.Figure]:
        """
        Create complete benchmark comparison analysis.
        
        Args:
            strategy_results: Strategy performance metrics and equity curve
            benchmark_results: Dictionary of benchmark name to results
            
        Returns:
            Dictionary of figure name to Plotly figure
        """
        figures = {}
        
        # Side-by-side metrics comparison
        figures['metrics_comparison'] = self._create_metrics_comparison(
            strategy_results, benchmark_results
        )
        
        # Relative performance chart
        figures['relative_performance'] = self._create_relative_performance_chart(
            strategy_results, benchmark_results
        )
        
        # Alpha generation chart
        figures['alpha_chart'] = self._create_alpha_chart(
            strategy_results, benchmark_results
        )
        
        # Risk-adjusted returns comparison
        figures['risk_adjusted'] = self._create_risk_adjusted_comparison(
            strategy_results, benchmark_results
        )
        
        # Rolling correlation analysis
        figures['correlation'] = self._create_correlation_analysis(
            strategy_results, benchmark_results
        )
        
        # Performance attribution vs benchmarks
        figures['attribution'] = self._create_performance_attribution(
            strategy_results, benchmark_results
        )
        
        return figures
    
    def _create_metrics_comparison(
        self,
        strategy_results: Dict[str, Any],
        benchmark_results: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        """Create side-by-side metrics comparison."""
        # Metrics to compare
        metrics = [
            ('total_return', 'Total Return', True),
            ('annual_return', 'Annual Return', True),
            ('sharpe_ratio', 'Sharpe Ratio', False),
            ('max_drawdown', 'Max Drawdown', True),
            ('volatility', 'Volatility', True),
            ('calmar_ratio', 'Calmar Ratio', False),
            ('sortino_ratio', 'Sortino Ratio', False),
            ('win_rate', 'Win Rate', True)
        ]
        
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=[m[1] for m in metrics]
        )
        
        # Prepare data
        strategies = ['Strategy'] + list(benchmark_results.keys())
        
        for idx, (metric_key, metric_name, is_percentage) in enumerate(metrics):
            row = idx // 4 + 1
            col = idx % 4 + 1
            
            values = []
            colors = []
            
            # Strategy value
            strategy_val = strategy_results.get(metric_key, 0)
            values.append(strategy_val * 100 if is_percentage else strategy_val)
            colors.append(self.strategy_color)
            
            # Benchmark values
            for bench_name, bench_results in benchmark_results.items():
                bench_val = bench_results.get(metric_key, 0)
                values.append(bench_val * 100 if is_percentage else bench_val)
                colors.append(self.benchmark_colors.get(bench_name, '#888888'))
            
            # Create bar chart
            fig.add_trace(
                go.Bar(
                    x=strategies,
                    y=values,
                    marker_color=colors,
                    showlegend=False,
                    text=[f'{v:.1f}{"%" if is_percentage else ""}' for v in values],
                    textposition='auto'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Strategy vs Benchmarks - Key Metrics Comparison',
            template='plotly_dark',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def _create_relative_performance_chart(
        self,
        strategy_results: Dict[str, Any],
        benchmark_results: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        """Create relative performance chart."""
        fig = go.Figure()
        
        # Get equity curves
        strategy_equity = strategy_results.get('equity_curve')
        
        if strategy_equity is None:
            return fig
        
        # Normalize all equity curves to start at 100
        strategy_norm = strategy_equity / strategy_equity.iloc[0] * 100
        
        # Add strategy line
        fig.add_trace(go.Scatter(
            x=strategy_norm.index,
            y=strategy_norm.values,
            name='Strategy',
            line=dict(color=self.strategy_color, width=2)
        ))
        
        # Add benchmark lines
        for bench_name, bench_results in benchmark_results.items():
            bench_equity = bench_results.get('equity_curve')
            if bench_equity is not None:
                bench_norm = bench_equity / bench_equity.iloc[0] * 100
                
                fig.add_trace(go.Scatter(
                    x=bench_norm.index,
                    y=bench_norm.values,
                    name=bench_name.replace('_', ' ').title(),
                    line=dict(
                        color=self.benchmark_colors.get(bench_name, '#888888'),
                        width=2,
                        dash='dash' if bench_name != 'buy_hold' else None
                    )
                ))
        
        # Add annotations for final values
        final_date = strategy_norm.index[-1]
        fig.add_annotation(
            x=final_date,
            y=strategy_norm.iloc[-1],
            text=f"{strategy_norm.iloc[-1]:.0f}",
            showarrow=False,
            xshift=10,
            font=dict(color=self.strategy_color)
        )
        
        fig.update_layout(
            title='Relative Performance (Normalized to 100)',
            xaxis_title='Date',
            yaxis_title='Normalized Value',
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
    
    def _create_alpha_chart(
        self,
        strategy_results: Dict[str, Any],
        benchmark_results: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        """Create alpha generation chart."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Cumulative Alpha Generation', 'Rolling 12-Month Alpha')
        )
        
        # Get equity curves
        strategy_equity = strategy_results.get('equity_curve')
        
        if strategy_equity is None:
            return fig
        
        # Calculate alpha vs each benchmark
        for bench_name, bench_results in benchmark_results.items():
            bench_equity = bench_results.get('equity_curve')
            if bench_equity is not None:
                # Calculate cumulative alpha
                strategy_returns = strategy_equity.pct_change()
                bench_returns = bench_equity.pct_change()
                
                # Align dates
                common_dates = strategy_returns.index.intersection(bench_returns.index)
                alpha_returns = strategy_returns[common_dates] - bench_returns[common_dates]
                cumulative_alpha = (1 + alpha_returns).cumprod() - 1
                
                # Cumulative alpha
                fig.add_trace(
                    go.Scatter(
                        x=cumulative_alpha.index,
                        y=cumulative_alpha.values * 100,
                        name=f'vs {bench_name.replace("_", " ").title()}',
                        line=dict(
                            color=self.benchmark_colors.get(bench_name, '#888888'),
                            width=2
                        )
                    ),
                    row=1, col=1
                )
                
                # Rolling 12-month alpha
                rolling_alpha = alpha_returns.rolling(252).sum() * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=rolling_alpha.index,
                        y=rolling_alpha.values,
                        name=f'vs {bench_name.replace("_", " ").title()}',
                        line=dict(
                            color=self.benchmark_colors.get(bench_name, '#888888'),
                            width=2
                        ),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # Add zero lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Update axes
        fig.update_yaxes(title_text='Cumulative Alpha (%)', row=1, col=1)
        fig.update_yaxes(title_text='12-Month Alpha (%)', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=2, col=1)
        
        fig.update_layout(
            title='Alpha Generation Analysis',
            template='plotly_dark',
            height=700,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_risk_adjusted_comparison(
        self,
        strategy_results: Dict[str, Any],
        benchmark_results: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        """Create risk-adjusted returns comparison."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Risk-Return Scatter', 'Risk-Adjusted Metrics')
        )
        
        # Risk-return scatter
        returns = []
        volatilities = []
        names = []
        colors = []
        
        # Add strategy
        returns.append(strategy_results.get('annual_return', 0) * 100)
        volatilities.append(strategy_results.get('volatility', 0) * 100)
        names.append('Strategy')
        colors.append(self.strategy_color)
        
        # Add benchmarks
        for bench_name, bench_results in benchmark_results.items():
            returns.append(bench_results.get('annual_return', 0) * 100)
            volatilities.append(bench_results.get('volatility', 0) * 100)
            names.append(bench_name.replace('_', ' ').title())
            colors.append(self.benchmark_colors.get(bench_name, '#888888'))
        
        fig.add_trace(
            go.Scatter(
                x=volatilities,
                y=returns,
                mode='markers+text',
                text=names,
                textposition='top center',
                marker=dict(size=20, color=colors),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add efficient frontier line (simplified)
        x_range = np.linspace(min(volatilities), max(volatilities), 100)
        y_range = x_range * strategy_results.get('sharpe_ratio', 0.5)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_range,
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Risk-adjusted metrics bar chart
        metrics = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
        metric_names = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
        
        x_data = []
        y_data = []
        colors_data = []
        
        for metric in metrics:
            # Strategy
            x_data.extend([metric_names[metrics.index(metric)]])
            y_data.extend([strategy_results.get(metric, 0)])
            colors_data.extend([self.strategy_color])
            
            # Benchmarks
            for bench_name in benchmark_results.keys():
                x_data.extend([metric_names[metrics.index(metric)]])
                y_data.extend([benchmark_results[bench_name].get(metric, 0)])
                colors_data.extend([self.benchmark_colors.get(bench_name, '#888888')])
        
        fig.add_trace(
            go.Bar(
                x=x_data,
                y=y_data,
                marker_color=colors_data,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text='Volatility (%)', row=1, col=1)
        fig.update_yaxes(title_text='Annual Return (%)', row=1, col=1)
        fig.update_xaxes(title_text='Metric', row=1, col=2)
        fig.update_yaxes(title_text='Value', row=1, col=2)
        
        fig.update_layout(
            title='Risk-Adjusted Performance Comparison',
            template='plotly_dark',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def _create_correlation_analysis(
        self,
        strategy_results: Dict[str, Any],
        benchmark_results: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        """Create rolling correlation analysis."""
        fig = go.Figure()
        
        # Get returns
        strategy_equity = strategy_results.get('equity_curve')
        if strategy_equity is None:
            return fig
        
        strategy_returns = strategy_equity.pct_change()
        
        # Calculate rolling correlation with each benchmark
        window = 60  # 60-day rolling correlation
        
        for bench_name, bench_results in benchmark_results.items():
            bench_equity = bench_results.get('equity_curve')
            if bench_equity is not None:
                bench_returns = bench_equity.pct_change()
                
                # Align dates
                common_dates = strategy_returns.index.intersection(bench_returns.index)
                
                # Calculate rolling correlation
                rolling_corr = strategy_returns[common_dates].rolling(window).corr(
                    bench_returns[common_dates]
                )
                
                fig.add_trace(go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    name=f'vs {bench_name.replace("_", " ").title()}',
                    line=dict(
                        color=self.benchmark_colors.get(bench_name, '#888888'),
                        width=2
                    )
                ))
        
        # Add reference lines
        fig.add_hline(y=0.8, line_dash="dash", line_color="yellow", 
                     annotation_text="High Correlation")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title=f'Rolling {window}-Day Correlation Analysis',
            xaxis_title='Date',
            yaxis_title='Correlation',
            template='plotly_dark',
            height=400,
            hovermode='x unified',
            yaxis=dict(range=[-1, 1])
        )
        
        return fig
    
    def _create_performance_attribution(
        self,
        strategy_results: Dict[str, Any],
        benchmark_results: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        """Create performance attribution vs benchmarks."""
        # Calculate outperformance sources
        fig = go.Figure()
        
        # Simple attribution breakdown
        categories = ['Base Return', 'Timing', 'Selection', 'Risk Management', 'Total Alpha']
        
        # Create waterfall chart for primary benchmark
        primary_bench = 'buy_hold'
        if primary_bench in benchmark_results:
            bench_return = benchmark_results[primary_bench].get('total_return', 0) * 100
            strategy_return = strategy_results.get('total_return', 0) * 100
            
            # Simplified attribution
            values = [
                bench_return,
                (strategy_return - bench_return) * 0.3,  # Timing
                (strategy_return - bench_return) * 0.5,  # Selection
                (strategy_return - bench_return) * 0.2,  # Risk Management
                0  # Total (calculated)
            ]
            
            # Create waterfall
            fig.add_trace(go.Waterfall(
                name="Performance Attribution",
                orientation="v",
                measure=["relative", "relative", "relative", "relative", "total"],
                x=categories,
                textposition="outside",
                text=[f"{v:.1f}%" for v in values],
                y=values,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": self.strategy_color}},
                decreasing={"marker": {"color": "#ff3366"}}
            ))
        
        fig.update_layout(
            title='Performance Attribution vs Buy & Hold',
            xaxis_title='Component',
            yaxis_title='Return Contribution (%)',
            template='plotly_dark',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def save_all_comparisons(
        self,
        figures: Dict[str, go.Figure],
        output_dir: str = 'reports/benchmark_comparison'
    ):
        """Save all benchmark comparison figures."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in figures.items():
            filepath = os.path.join(output_dir, f"{name}.html")
            fig.write_html(
                filepath,
                include_plotlyjs='cdn',
                config={'displayModeBar': True, 'displaylogo': False}
            )
            logger.info(f"Saved {name} to {filepath}")
        
        return output_dir