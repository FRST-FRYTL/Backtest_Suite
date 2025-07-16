"""
Timeframe Performance Analyzer

Analyzes backtesting results across multiple timeframes and parameter configurations
to identify optimal settings and performance patterns.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    """Container for strategy performance metrics"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    volatility: float
    var_95: float
    cvar_95: float
    total_trades: int
    avg_trade_duration: Optional[float] = None
    best_trade: Optional[float] = None
    worst_trade: Optional[float] = None
    recovery_time: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None
    information_ratio: Optional[float] = None
    
    @property
    def risk_adjusted_return(self) -> float:
        """Calculate risk-adjusted return score"""
        if self.volatility > 0:
            return self.annualized_return / self.volatility
        return 0.0
    
    @property
    def consistency_score(self) -> float:
        """Calculate consistency score based on multiple factors"""
        scores = []
        
        # Win rate contribution (0-40 points)
        scores.append(min(self.win_rate * 40, 40))
        
        # Sharpe ratio contribution (0-30 points)
        if self.sharpe_ratio > 0:
            scores.append(min(self.sharpe_ratio * 10, 30))
        else:
            scores.append(0)
        
        # Drawdown contribution (0-30 points)
        dd_score = max(0, 30 * (1 - abs(self.max_drawdown) / 0.5))
        scores.append(dd_score)
        
        return sum(scores)


@dataclass
class TimeframeResult:
    """Container for timeframe-specific backtest results"""
    timeframe: str
    symbol: str
    parameters: Dict[str, Any]
    metrics: PerformanceMetrics
    start_date: str
    end_date: str
    
    @property
    def parameter_hash(self) -> str:
        """Create unique hash for parameter configuration"""
        param_str = "_".join([f"{k}={v}" for k, v in sorted(self.parameters.items())])
        return f"{self.symbol}_{self.timeframe}_{param_str}"


class TimeframePerformanceAnalyzer:
    """
    Analyzes performance across multiple timeframes and parameter configurations
    """
    
    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = results_dir or Path("backtest_results")
        self.results: List[TimeframeResult] = []
        self.analysis_results: Dict[str, Any] = {}
        
    def load_results(self, results_file: Optional[Path] = None) -> None:
        """Load backtest results from file or directory"""
        if results_file and results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                self._parse_results(data)
        else:
            # Load all result files from directory
            for file_path in self.results_dir.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        self._parse_results(data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    def _parse_results(self, data: Dict[str, Any]) -> None:
        """Parse results data into TimeframeResult objects"""
        # Implementation will depend on actual data structure
        pass
    
    def analyze_by_timeframe(self) -> Dict[str, Dict[str, Any]]:
        """Analyze performance metrics grouped by timeframe"""
        timeframe_analysis = {}
        
        # Group results by timeframe
        df = self._results_to_dataframe()
        
        for timeframe in df['timeframe'].unique():
            tf_data = df[df['timeframe'] == timeframe]
            
            analysis = {
                'count': len(tf_data),
                'avg_return': tf_data['total_return'].mean(),
                'avg_sharpe': tf_data['sharpe_ratio'].mean(),
                'avg_drawdown': tf_data['max_drawdown'].mean(),
                'best_sharpe': tf_data['sharpe_ratio'].max(),
                'best_return': tf_data['total_return'].max(),
                'worst_drawdown': tf_data['max_drawdown'].min(),
                'avg_trades': tf_data['total_trades'].mean(),
                'consistency': tf_data['consistency_score'].mean(),
                'risk_adjusted_return': tf_data['risk_adjusted_return'].mean()
            }
            
            # Find best configuration
            best_idx = tf_data['sharpe_ratio'].idxmax()
            if pd.notna(best_idx):
                best_config = tf_data.loc[best_idx]
                analysis['best_config'] = {
                    'parameters': best_config['parameters'],
                    'sharpe_ratio': best_config['sharpe_ratio'],
                    'total_return': best_config['total_return'],
                    'max_drawdown': best_config['max_drawdown']
                }
            
            timeframe_analysis[timeframe] = analysis
            
        self.analysis_results['timeframe_analysis'] = timeframe_analysis
        return timeframe_analysis
    
    def analyze_parameter_sensitivity(self) -> Dict[str, Dict[str, float]]:
        """Analyze how sensitive performance is to parameter changes"""
        df = self._results_to_dataframe()
        param_sensitivity = {}
        
        # Extract parameter columns
        param_cols = [col for col in df.columns if col.startswith('param_')]
        
        for param in param_cols:
            correlations = {}
            for metric in ['sharpe_ratio', 'total_return', 'max_drawdown', 'consistency_score']:
                if metric in df.columns and pd.api.types.is_numeric_dtype(df[param]):
                    corr = df[param].corr(df[metric])
                    correlations[metric] = corr if pd.notna(corr) else 0.0
            
            param_sensitivity[param] = correlations
        
        self.analysis_results['parameter_sensitivity'] = param_sensitivity
        return param_sensitivity
    
    def find_robust_configurations(self, min_sharpe: float = 1.0, 
                                 max_drawdown: float = -0.20) -> List[Dict[str, Any]]:
        """Find configurations that perform well across multiple timeframes"""
        df = self._results_to_dataframe()
        
        # Group by parameter configuration
        config_performance = {}
        
        for _, row in df.iterrows():
            config_key = str(row['parameters'])
            if config_key not in config_performance:
                config_performance[config_key] = {
                    'timeframes': [],
                    'sharpe_ratios': [],
                    'returns': [],
                    'drawdowns': [],
                    'parameters': row['parameters']
                }
            
            config_performance[config_key]['timeframes'].append(row['timeframe'])
            config_performance[config_key]['sharpe_ratios'].append(row['sharpe_ratio'])
            config_performance[config_key]['returns'].append(row['total_return'])
            config_performance[config_key]['drawdowns'].append(row['max_drawdown'])
        
        # Find robust configurations
        robust_configs = []
        
        for config_key, perf in config_performance.items():
            avg_sharpe = np.mean(perf['sharpe_ratios'])
            avg_return = np.mean(perf['returns'])
            worst_drawdown = min(perf['drawdowns'])
            consistency = len(perf['timeframes'])
            
            if avg_sharpe >= min_sharpe and worst_drawdown >= max_drawdown:
                robust_configs.append({
                    'parameters': perf['parameters'],
                    'avg_sharpe': avg_sharpe,
                    'avg_return': avg_return,
                    'worst_drawdown': worst_drawdown,
                    'timeframe_count': consistency,
                    'sharpe_std': np.std(perf['sharpe_ratios']),
                    'return_std': np.std(perf['returns'])
                })
        
        # Sort by average Sharpe ratio
        robust_configs.sort(key=lambda x: x['avg_sharpe'], reverse=True)
        
        self.analysis_results['robust_configurations'] = robust_configs
        return robust_configs
    
    def create_performance_heatmap(self) -> go.Figure:
        """Create heatmap showing performance across timeframes and parameters"""
        df = self._results_to_dataframe()
        
        # Create pivot table for heatmap
        # This is a simplified version - actual implementation depends on data structure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sharpe Ratio', 'Total Return', 
                          'Max Drawdown', 'Consistency Score'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'consistency_score']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, (row, col) in zip(metrics, positions):
            # Create pivot table
            pivot = df.pivot_table(
                values=metric,
                index='timeframe',
                columns='parameter_hash',
                aggfunc='mean'
            )
            
            # Create heatmap
            heatmap = go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='RdYlGn' if metric != 'max_drawdown' else 'RdYlGn_r',
                showscale=True,
                name=metric
            )
            
            fig.add_trace(heatmap, row=row, col=col)
        
        fig.update_layout(
            title="Performance Heatmap: Timeframes vs Parameters",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_timeframe_comparison_chart(self) -> go.Figure:
        """Create comparison chart for different timeframes"""
        analysis = self.analysis_results.get('timeframe_analysis', {})
        
        if not analysis:
            self.analyze_by_timeframe()
            analysis = self.analysis_results.get('timeframe_analysis', {})
        
        timeframes = list(analysis.keys())
        metrics = ['avg_return', 'avg_sharpe', 'avg_drawdown', 'consistency']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Return', 'Average Sharpe Ratio', 
                          'Average Max Drawdown', 'Consistency Score'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1
            
            values = [analysis[tf].get(metric, 0) for tf in timeframes]
            
            bar = go.Bar(
                x=timeframes,
                y=values,
                name=metric,
                marker_color=colors[i],
                text=[f"{v:.2f}" for v in values],
                textposition='outside'
            )
            
            fig.add_trace(bar, row=row, col=col)
        
        fig.update_layout(
            title="Timeframe Performance Comparison",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_drawdown_analysis(self) -> go.Figure:
        """Create drawdown analysis visualization"""
        df = self._results_to_dataframe()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Drawdown Distribution by Timeframe',
                'Drawdown vs Return Scatter',
                'Recovery Time Analysis',
                'Risk-Return Profile'
            )
        )
        
        # 1. Drawdown distribution
        for i, timeframe in enumerate(df['timeframe'].unique()):
            tf_data = df[df['timeframe'] == timeframe]
            
            hist = go.Histogram(
                x=tf_data['max_drawdown'],
                name=timeframe,
                opacity=0.7,
                nbinsx=20
            )
            fig.add_trace(hist, row=1, col=1)
        
        # 2. Drawdown vs Return scatter
        scatter = go.Scatter(
            x=df['max_drawdown'],
            y=df['total_return'],
            mode='markers',
            marker=dict(
                size=10,
                color=df['sharpe_ratio'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            text=df['timeframe'],
            name='Risk-Return'
        )
        fig.add_trace(scatter, row=1, col=2)
        
        # 3. Recovery time (if available)
        if 'recovery_time' in df.columns:
            for timeframe in df['timeframe'].unique():
                tf_data = df[df['timeframe'] == timeframe]
                box = go.Box(
                    y=tf_data['recovery_time'],
                    name=timeframe
                )
                fig.add_trace(box, row=2, col=1)
        
        # 4. Risk-Return profile
        for timeframe in df['timeframe'].unique():
            tf_data = df[df['timeframe'] == timeframe]
            scatter = go.Scatter(
                x=tf_data['volatility'],
                y=tf_data['annualized_return'],
                mode='markers',
                name=timeframe,
                marker=dict(size=12)
            )
            fig.add_trace(scatter, row=2, col=2)
        
        fig.update_layout(
            title="Comprehensive Drawdown Analysis",
            height=900,
            showlegend=True
        )
        
        return fig
    
    def create_return_distribution_plots(self) -> go.Figure:
        """Create return distribution analysis"""
        df = self._results_to_dataframe()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Return Distribution by Timeframe',
                'Q-Q Plot: Return Normality',
                'Return Density Comparison',
                'Cumulative Return Distribution'
            )
        )
        
        timeframes = df['timeframe'].unique()
        colors = px.colors.qualitative.Set2
        
        # 1. Return distribution histograms
        for i, timeframe in enumerate(timeframes):
            tf_data = df[df['timeframe'] == timeframe]
            hist = go.Histogram(
                x=tf_data['total_return'],
                name=timeframe,
                opacity=0.7,
                marker_color=colors[i % len(colors)]
            )
            fig.add_trace(hist, row=1, col=1)
        
        # 2. Q-Q plot for normality check
        for i, timeframe in enumerate(timeframes):
            tf_data = df[df['timeframe'] == timeframe]
            returns = tf_data['total_return'].dropna()
            
            if len(returns) > 3:
                # Calculate theoretical quantiles
                sorted_returns = np.sort(returns)
                n = len(sorted_returns)
                theoretical_q = stats.norm.ppf((np.arange(n) + 0.5) / n)
                
                qq_scatter = go.Scatter(
                    x=theoretical_q,
                    y=sorted_returns,
                    mode='markers',
                    name=timeframe,
                    marker=dict(size=6)
                )
                fig.add_trace(qq_scatter, row=1, col=2)
        
        # Add reference line for Q-Q plot
        fig.add_trace(
            go.Scatter(
                x=[-3, 3],
                y=[-3, 3],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Density plots
        for i, timeframe in enumerate(timeframes):
            tf_data = df[df['timeframe'] == timeframe]
            returns = tf_data['total_return'].dropna()
            
            if len(returns) > 3:
                kde_x = np.linspace(returns.min(), returns.max(), 100)
                kde = stats.gaussian_kde(returns)
                kde_y = kde(kde_x)
                
                density = go.Scatter(
                    x=kde_x,
                    y=kde_y,
                    mode='lines',
                    name=timeframe,
                    line=dict(width=2)
                )
                fig.add_trace(density, row=2, col=1)
        
        # 4. Cumulative distribution
        for i, timeframe in enumerate(timeframes):
            tf_data = df[df['timeframe'] == timeframe]
            returns = tf_data['total_return'].dropna().sort_values()
            
            if len(returns) > 0:
                ecdf_y = np.arange(1, len(returns) + 1) / len(returns)
                
                ecdf = go.Scatter(
                    x=returns,
                    y=ecdf_y,
                    mode='lines',
                    name=timeframe,
                    line=dict(width=2)
                )
                fig.add_trace(ecdf, row=2, col=2)
        
        fig.update_layout(
            title="Return Distribution Analysis",
            height=900,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Total Return", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_xaxes(title_text="Total Return", row=2, col=1)
        fig.update_xaxes(title_text="Total Return", row=2, col=2)
        
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Density", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Probability", row=2, col=2)
        
        return fig
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis"""
        data = []
        
        for result in self.results:
            row = {
                'timeframe': result.timeframe,
                'symbol': result.symbol,
                'parameters': str(result.parameters),
                'parameter_hash': result.parameter_hash,
                'start_date': result.start_date,
                'end_date': result.end_date,
                'total_return': result.metrics.total_return,
                'annualized_return': result.metrics.annualized_return,
                'sharpe_ratio': result.metrics.sharpe_ratio,
                'sortino_ratio': result.metrics.sortino_ratio,
                'max_drawdown': result.metrics.max_drawdown,
                'calmar_ratio': result.metrics.calmar_ratio,
                'win_rate': result.metrics.win_rate,
                'profit_factor': result.metrics.profit_factor,
                'volatility': result.metrics.volatility,
                'var_95': result.metrics.var_95,
                'cvar_95': result.metrics.cvar_95,
                'total_trades': result.metrics.total_trades,
                'risk_adjusted_return': result.metrics.risk_adjusted_return,
                'consistency_score': result.metrics.consistency_score
            }
            
            # Add individual parameters as columns
            for param_name, param_value in result.parameters.items():
                row[f'param_{param_name}'] = param_value
            
            # Add optional metrics if available
            for field in ['avg_trade_duration', 'best_trade', 'worst_trade', 
                         'recovery_time', 'beta', 'alpha', 'information_ratio']:
                value = getattr(result.metrics, field, None)
                if value is not None:
                    row[field] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_html_report(self, output_path: Path) -> None:
        """Generate comprehensive HTML report"""
        # Perform all analyses
        self.analyze_by_timeframe()
        self.analyze_parameter_sensitivity()
        robust_configs = self.find_robust_configurations()
        
        # Create visualizations
        heatmap_fig = self.create_performance_heatmap()
        comparison_fig = self.create_timeframe_comparison_chart()
        drawdown_fig = self.create_drawdown_analysis()
        distribution_fig = self.create_return_distribution_plots()
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multi-Timeframe Performance Analysis</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                h2 {{
                    color: #555;
                    margin-top: 40px;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 10px;
                }}
                .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .summary-card {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    border: 1px solid #dee2e6;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #007bff;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #6c757d;
                    margin-bottom: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    text-align: left;
                    padding: 12px;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f8f9fa;
                    font-weight: bold;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .plot-container {{
                    margin: 30px 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                }}
                .timestamp {{
                    text-align: center;
                    color: #6c757d;
                    margin-top: 40px;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Multi-Timeframe Performance Analysis Report</h1>
                
                <h2>Executive Summary</h2>
                <div class="summary-grid">
                    {self._generate_summary_cards()}
                </div>
                
                <h2>Best Configurations by Timeframe</h2>
                {self._generate_timeframe_table()}
                
                <h2>Most Robust Configurations</h2>
                {self._generate_robust_configs_table(robust_configs[:10])}
                
                <h2>Performance Visualizations</h2>
                
                <div class="plot-container">
                    <h3>Timeframe Comparison</h3>
                    <div id="comparison-plot"></div>
                </div>
                
                <div class="plot-container">
                    <h3>Performance Heatmap</h3>
                    <div id="heatmap-plot"></div>
                </div>
                
                <div class="plot-container">
                    <h3>Drawdown Analysis</h3>
                    <div id="drawdown-plot"></div>
                </div>
                
                <div class="plot-container">
                    <h3>Return Distribution Analysis</h3>
                    <div id="distribution-plot"></div>
                </div>
                
                <h2>Parameter Sensitivity Analysis</h2>
                {self._generate_sensitivity_table()}
                
                <h2>Key Findings and Recommendations</h2>
                {self._generate_recommendations()}
                
                <div class="timestamp">
                    Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
            
            <script>
                Plotly.newPlot('comparison-plot', {comparison_fig.to_json()});
                Plotly.newPlot('heatmap-plot', {heatmap_fig.to_json()});
                Plotly.newPlot('drawdown-plot', {drawdown_fig.to_json()});
                Plotly.newPlot('distribution-plot', {distribution_fig.to_json()});
            </script>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_summary_cards(self) -> str:
        """Generate summary cards HTML"""
        df = self._results_to_dataframe()
        
        cards = []
        
        # Overall best Sharpe
        best_sharpe_idx = df['sharpe_ratio'].idxmax()
        if pd.notna(best_sharpe_idx):
            best_sharpe = df.loc[best_sharpe_idx]
            cards.append(f"""
                <div class="summary-card">
                    <div class="metric-label">Best Sharpe Ratio</div>
                    <div class="metric-value">{best_sharpe['sharpe_ratio']:.3f}</div>
                    <div>{best_sharpe['timeframe']} timeframe</div>
                </div>
            """)
        
        # Overall best return
        best_return_idx = df['total_return'].idxmax()
        if pd.notna(best_return_idx):
            best_return = df.loc[best_return_idx]
            cards.append(f"""
                <div class="summary-card">
                    <div class="metric-label">Best Total Return</div>
                    <div class="metric-value">{best_return['total_return']:.1f}%</div>
                    <div>{best_return['timeframe']} timeframe</div>
                </div>
            """)
        
        # Best risk-adjusted return
        best_rar_idx = df['risk_adjusted_return'].idxmax()
        if pd.notna(best_rar_idx):
            best_rar = df.loc[best_rar_idx]
            cards.append(f"""
                <div class="summary-card">
                    <div class="metric-label">Best Risk-Adjusted Return</div>
                    <div class="metric-value">{best_rar['risk_adjusted_return']:.3f}</div>
                    <div>{best_rar['timeframe']} timeframe</div>
                </div>
            """)
        
        # Most consistent
        best_consistency_idx = df['consistency_score'].idxmax()
        if pd.notna(best_consistency_idx):
            best_consistency = df.loc[best_consistency_idx]
            cards.append(f"""
                <div class="summary-card">
                    <div class="metric-label">Most Consistent Strategy</div>
                    <div class="metric-value">{best_consistency['consistency_score']:.1f}</div>
                    <div>{best_consistency['timeframe']} timeframe</div>
                </div>
            """)
        
        return '\n'.join(cards)
    
    def _generate_timeframe_table(self) -> str:
        """Generate timeframe analysis table"""
        analysis = self.analysis_results.get('timeframe_analysis', {})
        
        if not analysis:
            return "<p>No timeframe analysis available</p>"
        
        rows = []
        for timeframe, data in sorted(analysis.items()):
            best_config = data.get('best_config', {})
            rows.append(f"""
                <tr>
                    <td>{timeframe}</td>
                    <td>{data['count']}</td>
                    <td>{data['avg_return']:.2f}%</td>
                    <td>{data['avg_sharpe']:.3f}</td>
                    <td>{data['avg_drawdown']:.2f}%</td>
                    <td>{data['avg_trades']:.0f}</td>
                    <td>{f"{best_config.get('sharpe_ratio'):.3f}" if isinstance(best_config.get('sharpe_ratio'), (int, float)) else 'N/A'}</td>
                </tr>
            """)
        
        return f"""
            <table>
                <thead>
                    <tr>
                        <th>Timeframe</th>
                        <th>Configurations Tested</th>
                        <th>Avg Return</th>
                        <th>Avg Sharpe</th>
                        <th>Avg Drawdown</th>
                        <th>Avg Trades</th>
                        <th>Best Sharpe</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """
    
    def _generate_robust_configs_table(self, configs: List[Dict]) -> str:
        """Generate robust configurations table"""
        if not configs:
            return "<p>No robust configurations found</p>"
        
        rows = []
        for i, config in enumerate(configs):
            # Handle both dict and string parameters
            if isinstance(config['parameters'], dict):
                params_str = ', '.join([f"{k}={v}" for k, v in config['parameters'].items()])
            else:
                params_str = str(config['parameters'])
            rows.append(f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{params_str}</td>
                    <td>{config['avg_sharpe']:.3f}</td>
                    <td>{config['avg_return']:.2f}%</td>
                    <td>{config['worst_drawdown']:.2f}%</td>
                    <td>{config['timeframe_count']}</td>
                    <td>{config['sharpe_std']:.3f}</td>
                </tr>
            """)
        
        return f"""
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Parameters</th>
                        <th>Avg Sharpe</th>
                        <th>Avg Return</th>
                        <th>Worst Drawdown</th>
                        <th>Timeframes</th>
                        <th>Sharpe Std Dev</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """
    
    def _generate_sensitivity_table(self) -> str:
        """Generate parameter sensitivity table"""
        sensitivity = self.analysis_results.get('parameter_sensitivity', {})
        
        if not sensitivity:
            return "<p>No parameter sensitivity analysis available</p>"
        
        rows = []
        for param, correlations in sensitivity.items():
            param_clean = param.replace('param_', '')
            rows.append(f"""
                <tr>
                    <td>{param_clean}</td>
                    <td>{correlations.get('sharpe_ratio', 0):.3f}</td>
                    <td>{correlations.get('total_return', 0):.3f}</td>
                    <td>{correlations.get('max_drawdown', 0):.3f}</td>
                    <td>{correlations.get('consistency_score', 0):.3f}</td>
                </tr>
            """)
        
        return f"""
            <table>
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Sharpe Correlation</th>
                        <th>Return Correlation</th>
                        <th>Drawdown Correlation</th>
                        <th>Consistency Correlation</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """
    
    def _generate_recommendations(self) -> str:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Analyze timeframe performance
        tf_analysis = self.analysis_results.get('timeframe_analysis', {})
        if tf_analysis:
            best_tf_sharpe = max(tf_analysis.items(), 
                               key=lambda x: x[1].get('avg_sharpe', 0))
            recommendations.append(
                f"<li><strong>Best Average Performance:</strong> {best_tf_sharpe[0]} "
                f"timeframe shows the highest average Sharpe ratio ({best_tf_sharpe[1]['avg_sharpe']:.3f})</li>"
            )
        
        # Analyze robust configurations
        robust_configs = self.analysis_results.get('robust_configurations', [])
        if robust_configs:
            best_robust = robust_configs[0]
            # Handle both dict and string parameters
            if isinstance(best_robust['parameters'], dict):
                params_str = ', '.join([f'{k}={v}' for k, v in best_robust['parameters'].items()])
            else:
                params_str = str(best_robust['parameters'])
            recommendations.append(
                f"<li><strong>Most Robust Configuration:</strong> "
                f"{params_str} "
                f"performs consistently across {best_robust['timeframe_count']} timeframes</li>"
            )
        
        # Parameter sensitivity insights
        sensitivity = self.analysis_results.get('parameter_sensitivity', {})
        if sensitivity:
            high_impact_params = []
            for param, corrs in sensitivity.items():
                max_corr = max(abs(corrs.get('sharpe_ratio', 0)), 
                             abs(corrs.get('total_return', 0)))
                if max_corr > 0.3:
                    high_impact_params.append(param.replace('param_', ''))
            
            if high_impact_params:
                recommendations.append(
                    f"<li><strong>High Impact Parameters:</strong> "
                    f"{', '.join(high_impact_params)} show strong correlation with performance</li>"
                )
        
        return f"""
            <ul>
                {''.join(recommendations)}
                <li><strong>Risk Management:</strong> Consider position sizing based on timeframe volatility characteristics</li>
                <li><strong>Diversification:</strong> Running strategies across multiple timeframes may improve risk-adjusted returns</li>
            </ul>
        """


def main():
    """Example usage of the TimeframePerformanceAnalyzer"""
    analyzer = TimeframePerformanceAnalyzer()
    
    # Load results (implementation depends on actual data structure)
    # analyzer.load_results()
    
    # Generate report
    # analyzer.generate_html_report(Path("reports/spx_timeframe_analysis.html"))
    
    print("Timeframe Performance Analyzer created successfully!")


if __name__ == "__main__":
    main()