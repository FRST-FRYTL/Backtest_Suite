"""
Visualization Agent for ML Pipeline

Creates comprehensive visualizations for data analysis, model performance,
and trading strategy results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent


class VisualizationAgent(BaseAgent):
    """
    Agent responsible for creating visualizations including:
    - Data exploration plots
    - Feature analysis visualizations
    - Model performance charts
    - Trading strategy dashboards
    - Interactive reports
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("VisualizationAgent", config)
        self.visualizations = {}
        self.color_palette = None
        self.plot_style = None
        
    def initialize(self) -> bool:
        """Initialize visualization resources."""
        try:
            self.logger.info("Initializing Visualization Agent")
            
            # Validate required configuration
            required_keys = ["plot_types", "output_format", "style"]
            if not self.validate_config(required_keys):
                return False
            
            # Initialize visualization settings
            self.plot_types = self.config.get("plot_types", [
                "performance", "risk", "features", "predictions"
            ])
            self.output_format = self.config.get("output_format", "static")  # static or interactive
            self.style = self.config.get("style", "seaborn")
            
            # Set plot style
            if self.style == "seaborn":
                sns.set_style("whitegrid")
                self.color_palette = sns.color_palette("husl", 8)
            else:
                plt.style.use(self.style)
                
            # Initialize color schemes
            self.colors = {
                "primary": "#1f77b4",
                "secondary": "#ff7f0e", 
                "success": "#2ca02c",
                "danger": "#d62728",
                "warning": "#ff9800",
                "info": "#17a2b8"
            }
            
            self.logger.info("Visualization Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False
    
    def execute(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute visualization creation.
        
        Args:
            data: Dictionary containing data to visualize
            
        Returns:
            Dict containing visualization paths and metadata
        """
        try:
            viz_results = {}
            
            # Create data exploration visualizations
            if "raw_data" in data:
                viz_results["data_exploration"] = self._create_data_exploration_viz(
                    data["raw_data"]
                )
            
            # Create feature analysis visualizations
            if "features" in data:
                viz_results["feature_analysis"] = self._create_feature_analysis_viz(
                    data["features"], 
                    data.get("feature_importance")
                )
            
            # Create model performance visualizations
            if "predictions" in data and "actual" in data:
                viz_results["model_performance"] = self._create_model_performance_viz(
                    data["predictions"],
                    data["actual"],
                    data.get("model_metrics")
                )
            
            # Create trading strategy visualizations
            if "returns" in data:
                viz_results["trading_performance"] = self._create_trading_performance_viz(
                    data["returns"],
                    data.get("positions"),
                    data.get("benchmark_returns")
                )
            
            # Create risk analysis visualizations
            if "risk_metrics" in data:
                viz_results["risk_analysis"] = self._create_risk_analysis_viz(
                    data["risk_metrics"]
                )
            
            # Create comprehensive dashboard
            if self.output_format == "interactive":
                viz_results["dashboard"] = self._create_interactive_dashboard(data)
            else:
                viz_results["dashboard"] = self._create_static_dashboard(data)
            
            # Generate report
            viz_results["report"] = self._generate_visual_report(viz_results)
            
            self.visualizations = viz_results
            
            return {
                "visualizations": viz_results,
                "summary": self._generate_visualization_summary(),
                "metadata": {
                    "total_plots": self._count_total_plots(viz_results),
                    "format": self.output_format,
                    "style": self.style
                }
            }
            
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            raise
    
    def _create_data_exploration_viz(self, data: pd.DataFrame) -> Dict[str, str]:
        """Create data exploration visualizations."""
        self.logger.info("Creating data exploration visualizations")
        
        viz_paths = {}
        
        # Distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Price distribution
        if 'close' in data.columns:
            ax = axes[0, 0]
            data['close'].hist(bins=50, ax=ax, alpha=0.7, color=self.colors["primary"])
            ax.set_xlabel('Price')
            ax.set_ylabel('Frequency')
            ax.set_title('Price Distribution')
            
        # Returns distribution
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            ax = axes[0, 1]
            returns.hist(bins=50, ax=ax, alpha=0.7, color=self.colors["secondary"])
            ax.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
            ax.set_xlabel('Returns')
            ax.set_ylabel('Frequency')
            ax.set_title('Returns Distribution')
            ax.legend()
            
        # Volume analysis
        if 'volume' in data.columns:
            ax = axes[1, 0]
            data['volume'].rolling(20).mean().plot(ax=ax, label='20-day MA', color=self.colors["info"])
            ax.set_xlabel('Date')
            ax.set_ylabel('Volume')
            ax.set_title('Volume Trend')
            ax.legend()
            
        # Correlation heatmap
        ax = axes[1, 1]
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:5]
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Feature Correlations')
        
        plt.tight_layout()
        plt.savefig('/tmp/data_exploration.png', dpi=300, bbox_inches='tight')
        viz_paths["overview"] = '/tmp/data_exploration.png'
        plt.close()
        
        # Time series plot
        if 'close' in data.columns and isinstance(data.index, pd.DatetimeIndex):
            plt.figure(figsize=(14, 8))
            
            plt.subplot(3, 1, 1)
            data['close'].plot(color=self.colors["primary"], linewidth=1)
            plt.title('Price Time Series')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(3, 1, 2)
            returns.plot(color=self.colors["secondary"], linewidth=0.5)
            plt.title('Returns Time Series')
            plt.ylabel('Returns')
            plt.grid(True, alpha=0.3)
            
            if 'volume' in data.columns:
                plt.subplot(3, 1, 3)
                data['volume'].plot(kind='bar', color=self.colors["info"], width=1)
                plt.title('Volume')
                plt.ylabel('Volume')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('/tmp/timeseries_analysis.png', dpi=300, bbox_inches='tight')
            viz_paths["timeseries"] = '/tmp/timeseries_analysis.png'
            plt.close()
        
        return viz_paths
    
    def _create_feature_analysis_viz(self, features: pd.DataFrame,
                                   importance: Optional[Dict[str, float]]) -> Dict[str, str]:
        """Create feature analysis visualizations."""
        self.logger.info("Creating feature analysis visualizations")
        
        viz_paths = {}
        
        # Feature importance plot
        if importance:
            plt.figure(figsize=(10, 8))
            
            # Sort features by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
            feature_names = [f[0] for f in sorted_features]
            importance_values = [f[1] for f in sorted_features]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(feature_names))
            plt.barh(y_pos, importance_values, color=self.colors["primary"])
            plt.yticks(y_pos, feature_names)
            plt.xlabel('Importance Score')
            plt.title('Top 20 Feature Importance')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('/tmp/feature_importance.png', dpi=300, bbox_inches='tight')
            viz_paths["importance"] = '/tmp/feature_importance.png'
            plt.close()
        
        # Feature relationships
        if len(features.columns) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Select top features for visualization
            top_features = features.columns[:4] if len(features.columns) >= 4 else features.columns
            
            for idx, (i, j) in enumerate([(0, 1), (0, 2), (1, 2), (1, 3)]):
                ax = axes[idx // 2, idx % 2]
                if i < len(top_features) and j < len(top_features):
                    ax.scatter(features[top_features[i]], features[top_features[j]], 
                             alpha=0.5, s=10, color=self.colors["secondary"])
                    ax.set_xlabel(top_features[i])
                    ax.set_ylabel(top_features[j])
                    ax.set_title(f'{top_features[i]} vs {top_features[j]}')
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('/tmp/feature_relationships.png', dpi=300, bbox_inches='tight')
            viz_paths["relationships"] = '/tmp/feature_relationships.png'
            plt.close()
        
        return viz_paths
    
    def _create_model_performance_viz(self, predictions: Union[pd.Series, np.ndarray],
                                    actual: Union[pd.Series, np.ndarray],
                                    metrics: Optional[Dict[str, float]]) -> Dict[str, str]:
        """Create model performance visualizations."""
        self.logger.info("Creating model performance visualizations")
        
        viz_paths = {}
        
        # Performance analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Predictions vs Actual
        ax = axes[0, 0]
        ax.scatter(actual, predictions, alpha=0.5, s=20, color=self.colors["primary"])
        
        # Add perfect prediction line
        min_val = min(actual.min(), predictions.min())
        max_val = max(actual.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Add regression line
        z = np.polyfit(actual, predictions, 1)
        p = np.poly1d(z)
        ax.plot([min_val, max_val], p([min_val, max_val]), 
               color=self.colors["success"], label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Predictions vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Residuals plot
        ax = axes[0, 1]
        residuals = actual - predictions
        ax.scatter(predictions, residuals, alpha=0.5, s=20, color=self.colors["secondary"])
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)
        
        # Residual distribution
        ax = axes[1, 0]
        ax.hist(residuals, bins=50, alpha=0.7, color=self.colors["info"], density=True)
        
        # Fit normal distribution
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x, (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2/(2*sigma**2)), 
               'r-', label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
        
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Density')
        ax.set_title('Residual Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax = axes[1, 1]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/tmp/model_performance.png', dpi=300, bbox_inches='tight')
        viz_paths["performance"] = '/tmp/model_performance.png'
        plt.close()
        
        # Time series of predictions if index is datetime
        if isinstance(actual, pd.Series) and isinstance(actual.index, pd.DatetimeIndex):
            plt.figure(figsize=(14, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(actual.index, actual.values, label='Actual', 
                    color=self.colors["primary"], alpha=0.7)
            plt.plot(actual.index, predictions, label='Predicted', 
                    color=self.colors["danger"], alpha=0.7)
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.title('Predictions vs Actual Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.plot(actual.index, residuals, color=self.colors["warning"], alpha=0.7)
            plt.axhline(y=0, color='red', linestyle='--')
            plt.xlabel('Date')
            plt.ylabel('Residuals')
            plt.title('Residuals Over Time')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('/tmp/predictions_timeseries.png', dpi=300, bbox_inches='tight')
            viz_paths["timeseries"] = '/tmp/predictions_timeseries.png'
            plt.close()
        
        return viz_paths
    
    def _create_trading_performance_viz(self, returns: pd.Series,
                                      positions: Optional[pd.Series],
                                      benchmark: Optional[pd.Series]) -> Dict[str, str]:
        """Create trading performance visualizations."""
        self.logger.info("Creating trading performance visualizations")
        
        viz_paths = {}
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Performance dashboard
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        # Cumulative returns
        ax = axes[0, 0]
        cum_returns.plot(ax=ax, label='Strategy', color=self.colors["primary"], linewidth=2)
        if benchmark is not None:
            cum_benchmark = (1 + benchmark).cumprod()
            cum_benchmark.plot(ax=ax, label='Benchmark', 
                             color=self.colors["secondary"], linewidth=2, alpha=0.7)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns')
        ax.set_title('Cumulative Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Drawdown
        ax = axes[0, 1]
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        drawdown.plot(ax=ax, color=self.colors["danger"], linewidth=1)
        ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color=self.colors["danger"])
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.set_title('Drawdown Chart')
        ax.grid(True, alpha=0.3)
        
        # Returns distribution
        ax = axes[1, 0]
        returns.hist(bins=50, ax=ax, alpha=0.7, color=self.colors["info"], density=True)
        ax.axvline(returns.mean(), color='red', linestyle='--', 
                  label=f'Mean: {returns.mean():.4f}')
        ax.set_xlabel('Returns')
        ax.set_ylabel('Density')
        ax.set_title('Returns Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rolling volatility
        ax = axes[1, 1]
        rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)
        rolling_vol.plot(ax=ax, color=self.colors["warning"], linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Annualized Volatility')
        ax.set_title('21-Day Rolling Volatility')
        ax.grid(True, alpha=0.3)
        
        # Monthly returns heatmap
        ax = axes[2, 0]
        if isinstance(returns.index, pd.DatetimeIndex):
            monthly_returns = returns.resample('M').sum()
            monthly_pivot = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values
            }).pivot(index='Month', columns='Year', values='Return')
            
            sns.heatmap(monthly_pivot, annot=True, fmt='.2%', 
                       cmap='RdYlGn', center=0, ax=ax)
            ax.set_title('Monthly Returns Heatmap')
        
        # Rolling Sharpe ratio
        ax = axes[2, 1]
        rolling_sharpe = returns.rolling(window=252).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        rolling_sharpe.plot(ax=ax, color=self.colors["success"], linewidth=2)
        ax.axhline(y=1, color='red', linestyle='--', label='Sharpe = 1')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Rolling Sharpe Ratio (252-day)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/tmp/trading_performance.png', dpi=300, bbox_inches='tight')
        viz_paths["performance"] = '/tmp/trading_performance.png'
        plt.close()
        
        # Position analysis if available
        if positions is not None:
            plt.figure(figsize=(14, 6))
            
            plt.subplot(2, 1, 1)
            positions.plot(drawstyle='steps-post', color=self.colors["primary"], linewidth=1)
            plt.ylabel('Position')
            plt.title('Trading Positions Over Time')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            trade_count = positions.diff().abs().rolling(window=252).sum() / 2
            trade_count.plot(color=self.colors["secondary"], linewidth=2)
            plt.xlabel('Date')
            plt.ylabel('Number of Trades')
            plt.title('Rolling Trade Count (252-day)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('/tmp/position_analysis.png', dpi=300, bbox_inches='tight')
            viz_paths["positions"] = '/tmp/position_analysis.png'
            plt.close()
        
        return viz_paths
    
    def _create_risk_analysis_viz(self, risk_metrics: Dict[str, Any]) -> Dict[str, str]:
        """Create risk analysis visualizations."""
        self.logger.info("Creating risk analysis visualizations")
        
        viz_paths = {}
        
        # Risk metrics dashboard
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # VaR visualization
        ax = axes[0, 0]
        if "var_analysis" in risk_metrics:
            var_data = risk_metrics["var_analysis"]
            confidence_levels = list(var_data.keys())
            var_values = [var_data[cl]["historical"]["var_daily"] for cl in confidence_levels]
            cvar_values = [var_data[cl]["historical"]["cvar_daily"] for cl in confidence_levels]
            
            x = np.arange(len(confidence_levels))
            width = 0.35
            
            ax.bar(x - width/2, var_values, width, label='VaR', color=self.colors["primary"])
            ax.bar(x + width/2, cvar_values, width, label='CVaR', color=self.colors["danger"])
            
            ax.set_xlabel('Confidence Level')
            ax.set_ylabel('Daily Risk')
            ax.set_title('Value at Risk Analysis')
            ax.set_xticks(x)
            ax.set_xticklabels(confidence_levels)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Risk factor analysis
        ax = axes[0, 1]
        if "factor_analysis" in risk_metrics:
            factors = risk_metrics["factor_analysis"]
            if "explained_variance" in factors:
                variance_ratios = factors["explained_variance"][:5]
                factor_names = [f'Factor {i+1}' for i in range(len(variance_ratios))]
                
                ax.bar(factor_names, variance_ratios, color=self.colors["secondary"])
                ax.set_xlabel('Risk Factor')
                ax.set_ylabel('Explained Variance Ratio')
                ax.set_title('Risk Factor Contribution')
                ax.grid(True, alpha=0.3)
        
        # Stress test results
        ax = axes[1, 0]
        if "stress_testing" in risk_metrics:
            stress_results = risk_metrics["stress_testing"]
            scenarios = list(stress_results.keys())
            impacts = [stress_results[s]["impact"]["return_impact"] for s in scenarios]
            
            colors = [self.colors["danger"] if impact < 0 else self.colors["success"] 
                     for impact in impacts]
            ax.bar(scenarios, impacts, color=colors)
            ax.set_xlabel('Stress Scenario')
            ax.set_ylabel('Return Impact')
            ax.set_title('Stress Test Results')
            ax.grid(True, alpha=0.3)
        
        # Risk attribution
        ax = axes[1, 1]
        if "risk_attribution" in risk_metrics:
            attribution = risk_metrics["risk_attribution"]
            if "risk_contributions" in attribution:
                contributions = attribution["risk_contributions"]
                assets = list(contributions.keys())[:5]  # Top 5
                values = [contributions[a] for a in assets]
                
                ax.pie(values, labels=assets, autopct='%1.1f%%', 
                      colors=self.color_palette[:len(assets)])
                ax.set_title('Risk Attribution')
        
        plt.tight_layout()
        plt.savefig('/tmp/risk_analysis.png', dpi=300, bbox_inches='tight')
        viz_paths["risk_metrics"] = '/tmp/risk_analysis.png'
        plt.close()
        
        return viz_paths
    
    def _create_interactive_dashboard(self, data: Dict[str, Any]) -> str:
        """Create interactive dashboard using Plotly."""
        self.logger.info("Creating interactive dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Cumulative Returns', 'Drawdown', 
                          'Returns Distribution', 'Feature Importance',
                          'Model Performance', 'Risk Metrics'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Add traces based on available data
        if "returns" in data:
            returns = data["returns"]
            cum_returns = (1 + returns).cumprod()
            
            # Cumulative returns
            fig.add_trace(
                go.Scatter(x=cum_returns.index, y=cum_returns.values,
                          name='Strategy', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Drawdown
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            fig.add_trace(
                go.Scatter(x=drawdown.index, y=drawdown.values,
                          name='Drawdown', fill='tozeroy',
                          line=dict(color='red')),
                row=1, col=2
            )
            
            # Returns distribution
            fig.add_trace(
                go.Histogram(x=returns.values, name='Returns',
                            nbinsx=50, histnorm='probability density'),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title_text="Trading Strategy Dashboard",
            showlegend=True,
            height=1200,
            width=1400
        )
        
        # Save interactive dashboard
        fig.write_html('/tmp/interactive_dashboard.html')
        return '/tmp/interactive_dashboard.html'
    
    def _create_static_dashboard(self, data: Dict[str, Any]) -> str:
        """Create static dashboard summary."""
        self.logger.info("Creating static dashboard")
        
        # Create figure with GridSpec for complex layout
        fig = plt.figure(figsize=(16, 20))
        gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 0.5])
        
        # Title
        fig.suptitle('ML Trading Strategy Dashboard', fontsize=16, fontweight='bold')
        
        # Add various plots based on available data
        plot_idx = 0
        
        # Performance metrics summary
        if "performance_metrics" in data:
            ax = fig.add_subplot(gs[4, :])
            ax.axis('tight')
            ax.axis('off')
            
            metrics = data["performance_metrics"]
            summary_data = []
            
            if "sharpe_ratio" in metrics:
                summary_data.append(["Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"])
            if "annual_return" in metrics:
                summary_data.append(["Annual Return", f"{metrics['annual_return']:.2%}"])
            if "max_drawdown" in metrics:
                summary_data.append(["Max Drawdown", f"{metrics['max_drawdown']:.2%}"])
            if "win_rate" in metrics:
                summary_data.append(["Win Rate", f"{metrics['win_rate']:.2%}"])
            
            if summary_data:
                table = ax.table(cellText=summary_data,
                               colLabels=['Metric', 'Value'],
                               cellLoc='center',
                               loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
        
        plt.tight_layout()
        plt.savefig('/tmp/static_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return '/tmp/static_dashboard.png'
    
    def _generate_visual_report(self, visualizations: Dict[str, Any]) -> str:
        """Generate HTML report with all visualizations."""
        self.logger.info("Generating visual report")
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Trading Strategy Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #666; }
                .section { margin-bottom: 30px; }
                img { max-width: 100%; height: auto; margin: 10px 0; }
                .metrics { background-color: #f0f0f0; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>ML Trading Strategy Analysis Report</h1>
        """
        
        # Add sections for each visualization category
        for category, viz_data in visualizations.items():
            if isinstance(viz_data, dict) and viz_data:
                html_content += f'<div class="section"><h2>{category.replace("_", " ").title()}</h2>'
                
                for viz_name, viz_path in viz_data.items():
                    if viz_path and viz_path.endswith('.png'):
                        # Convert to base64 for embedding
                        import base64
                        with open(viz_path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode()
                        
                        html_content += f'''
                        <h3>{viz_name.replace("_", " ").title()}</h3>
                        <img src="data:image/png;base64,{img_data}" alt="{viz_name}">
                        '''
                
                html_content += '</div>'
        
        html_content += """
        </body>
        </html>
        """
        
        # Save report
        report_path = '/tmp/visual_report.html'
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    def _generate_visualization_summary(self) -> Dict[str, Any]:
        """Generate visualization summary."""
        summary = {
            "total_visualizations": self._count_total_plots(self.visualizations),
            "categories": list(self.visualizations.keys()),
            "format": self.output_format,
            "recommendations": []
        }
        
        # Add recommendations based on what was visualized
        if "model_performance" in self.visualizations:
            summary["recommendations"].append(
                "Review residual plots for model bias detection"
            )
        
        if "trading_performance" in self.visualizations:
            summary["recommendations"].append(
                "Monitor drawdown charts for risk management"
            )
        
        if "risk_analysis" in self.visualizations:
            summary["recommendations"].append(
                "Use stress test visualizations for scenario planning"
            )
        
        return summary
    
    def _count_total_plots(self, viz_dict: Dict[str, Any]) -> int:
        """Count total number of plots created."""
        count = 0
        for value in viz_dict.values():
            if isinstance(value, dict):
                count += len(value)
            elif isinstance(value, str):
                count += 1
        return count
    
    def create_custom_plot(self, plot_type: str, data: Any, **kwargs) -> str:
        """Create custom plot based on type."""
        self.logger.info(f"Creating custom {plot_type} plot")
        
        plt.figure(figsize=kwargs.get("figsize", (10, 6)))
        
        if plot_type == "candlestick" and isinstance(data, pd.DataFrame):
            # Simple candlestick representation
            up = data[data.close >= data.open]
            down = data[data.close < data.open]
            
            plt.bar(up.index, up.close - up.open, bottom=up.open, 
                   color='green', width=0.8)
            plt.bar(down.index, down.close - down.open, bottom=down.open, 
                   color='red', width=0.8)
            
            plt.bar(up.index, up.high - up.close, bottom=up.close, 
                   color='green', width=0.1)
            plt.bar(up.index, up.low - up.open, bottom=up.low, 
                   color='green', width=0.1)
            
            plt.bar(down.index, down.high - down.open, bottom=down.open, 
                   color='red', width=0.1)
            plt.bar(down.index, down.low - down.close, bottom=down.low, 
                   color='red', width=0.1)
            
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title('Candlestick Chart')
            
        elif plot_type == "correlation_network":
            # Placeholder for network visualization
            plt.text(0.5, 0.5, 'Correlation Network Visualization', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            
        plt.tight_layout()
        output_path = f'/tmp/custom_{plot_type}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path