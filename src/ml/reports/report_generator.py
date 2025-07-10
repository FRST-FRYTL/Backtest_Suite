"""
Machine Learning Report Generator

Generates comprehensive HTML reports with interactive visualizations for ML results.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from jinja2 import Environment, FileSystemLoader
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64


class MLReportGenerator:
    """Generates HTML reports for machine learning results."""
    
    def __init__(self, template_dir: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize the report generator.
        
        Args:
            template_dir: Directory containing HTML templates
            output_dir: Directory for output reports
        """
        self.template_dir = template_dir or os.path.join(os.path.dirname(__file__), '../../../reports/templates')
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), '../../../reports/output')
        
        # Create directories if they don't exist
        Path(self.template_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
    
    def generate_feature_analysis_report(
        self,
        feature_importance: pd.DataFrame,
        correlation_matrix: pd.DataFrame,
        feature_distributions: Dict[str, pd.Series],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Generate feature analysis report.
        
        Args:
            feature_importance: DataFrame with feature importance scores
            correlation_matrix: Feature correlation matrix
            feature_distributions: Dictionary of feature distributions
            metadata: Additional metadata
            
        Returns:
            Path to generated report
        """
        # Create feature importance chart
        importance_fig = go.Figure()
        
        # Sort features by importance
        sorted_features = feature_importance.sort_values('importance', ascending=True)
        
        importance_fig.add_trace(go.Bar(
            x=sorted_features['importance'],
            y=sorted_features.index,
            orientation='h',
            marker_color='lightblue',
            text=sorted_features['importance'].round(3),
            textposition='auto'
        ))
        
        importance_fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=max(400, len(sorted_features) * 25)
        )
        
        # Create correlation heatmap
        correlation_fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        correlation_fig.update_layout(
            title='Feature Correlation Matrix',
            height=600,
            width=800
        )
        
        # Create feature distribution plots
        distribution_figs = []
        for feature_name, distribution in list(feature_distributions.items())[:10]:  # Top 10 features
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=distribution,
                name='Distribution',
                nbinsx=30,
                marker_color='lightgreen',
                opacity=0.7
            ))
            
            # Add KDE overlay
            from scipy import stats
            kde = stats.gaussian_kde(distribution.dropna())
            x_range = np.linspace(distribution.min(), distribution.max(), 100)
            kde_values = kde(x_range)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=kde_values * len(distribution) * (distribution.max() - distribution.min()) / 30,
                mode='lines',
                name='KDE',
                line=dict(color='red', width=2),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title=f'Distribution of {feature_name}',
                xaxis_title=feature_name,
                yaxis_title='Count',
                yaxis2=dict(
                    overlaying='y',
                    side='right',
                    title='Density'
                ),
                height=400
            )
            
            distribution_figs.append({
                'name': feature_name,
                'plot': fig.to_html(include_plotlyjs=False, div_id=f"dist_{feature_name}")
            })
        
        # Prepare template data
        template_data = {
            'report_title': 'ML Feature Analysis Report',
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': metadata or {},
            'feature_importance_plot': importance_fig.to_html(include_plotlyjs=False),
            'correlation_heatmap': correlation_fig.to_html(include_plotlyjs=False),
            'distribution_plots': distribution_figs,
            'feature_stats': self._calculate_feature_stats(feature_distributions)
        }
        
        # Generate report
        template = self._create_feature_analysis_template()
        html_content = template.render(**template_data)
        
        # Save report
        output_path = os.path.join(self.output_dir, f'feature_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def generate_performance_dashboard(
        self,
        model_metrics: Dict[str, Dict[str, float]],
        confusion_matrices: Dict[str, np.ndarray],
        roc_data: Dict[str, Dict[str, np.ndarray]],
        profit_curves: Dict[str, pd.DataFrame],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Generate model performance dashboard.
        
        Args:
            model_metrics: Dictionary of model metrics
            confusion_matrices: Confusion matrices by model
            roc_data: ROC curve data (fpr, tpr, thresholds)
            profit_curves: Profit curves by model
            metadata: Additional metadata
            
        Returns:
            Path to generated report
        """
        # Create metrics comparison chart
        metrics_df = pd.DataFrame(model_metrics).T
        
        metrics_fig = go.Figure()
        
        for metric in metrics_df.columns:
            metrics_fig.add_trace(go.Bar(
                name=metric,
                x=metrics_df.index,
                y=metrics_df[metric],
                text=metrics_df[metric].round(3),
                textposition='auto'
            ))
        
        metrics_fig.update_layout(
            title='Model Performance Metrics',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=500
        )
        
        # Create confusion matrix plots
        confusion_plots = []
        for model_name, cm in confusion_matrices.items():
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Negative', 'Predicted Positive'],
                y=['Actual Negative', 'Actual Positive'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 14},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=f'Confusion Matrix - {model_name}',
                height=400,
                width=400
            )
            
            confusion_plots.append({
                'name': model_name,
                'plot': fig.to_html(include_plotlyjs=False, div_id=f"cm_{model_name}")
            })
        
        # Create ROC curves
        roc_fig = go.Figure()
        
        for model_name, roc_info in roc_data.items():
            roc_fig.add_trace(go.Scatter(
                x=roc_info['fpr'],
                y=roc_info['tpr'],
                mode='lines',
                name=f"{model_name} (AUC={roc_info.get('auc', 0):.3f})",
                line=dict(width=2)
            ))
        
        # Add diagonal reference line
        roc_fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', dash='dash')
        ))
        
        roc_fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            width=600
        )
        
        # Create profit curves
        profit_fig = go.Figure()
        
        for model_name, profit_df in profit_curves.items():
            profit_fig.add_trace(go.Scatter(
                x=profit_df.index,
                y=profit_df['cumulative_profit'],
                mode='lines',
                name=model_name,
                line=dict(width=2)
            ))
        
        profit_fig.update_layout(
            title='Cumulative Profit Curves',
            xaxis_title='Time',
            yaxis_title='Cumulative Profit',
            height=500
        )
        
        # Prepare template data
        template_data = {
            'report_title': 'ML Performance Dashboard',
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': metadata or {},
            'metrics_plot': metrics_fig.to_html(include_plotlyjs=False),
            'confusion_matrices': confusion_plots,
            'roc_curves': roc_fig.to_html(include_plotlyjs=False),
            'profit_curves': profit_fig.to_html(include_plotlyjs=False),
            'model_summaries': self._create_model_summaries(model_metrics)
        }
        
        # Generate report
        template = self._create_performance_dashboard_template()
        html_content = template.render(**template_data)
        
        # Save report
        output_path = os.path.join(self.output_dir, f'performance_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def generate_optimization_results_report(
        self,
        optimization_history: List[Dict],
        parameter_evolution: pd.DataFrame,
        best_configurations: List[Dict],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Generate optimization results report.
        
        Args:
            optimization_history: History of optimization iterations
            parameter_evolution: Evolution of parameters over iterations
            best_configurations: Best parameter configurations found
            metadata: Additional metadata
            
        Returns:
            Path to generated report
        """
        # Create performance improvement chart
        history_df = pd.DataFrame(optimization_history)
        
        improvement_fig = go.Figure()
        
        improvement_fig.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df['best_score'],
            mode='lines+markers',
            name='Best Score',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        improvement_fig.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df['current_score'],
            mode='lines',
            name='Current Score',
            line=dict(color='lightblue', width=1),
            opacity=0.7
        ))
        
        improvement_fig.update_layout(
            title='Optimization Performance Over Iterations',
            xaxis_title='Iteration',
            yaxis_title='Score',
            height=500
        )
        
        # Create parameter evolution plots
        param_figs = []
        
        for param in parameter_evolution.columns[:6]:  # Top 6 parameters
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=parameter_evolution.index,
                y=parameter_evolution[param],
                mode='lines+markers',
                name=param,
                line=dict(width=2)
            ))
            
            fig.update_layout(
                title=f'Evolution of {param}',
                xaxis_title='Iteration',
                yaxis_title='Value',
                height=300
            )
            
            param_figs.append({
                'name': param,
                'plot': fig.to_html(include_plotlyjs=False, div_id=f"param_{param}")
            })
        
        # Create best configurations table
        best_configs_df = pd.DataFrame(best_configurations)
        
        # Prepare template data
        template_data = {
            'report_title': 'Optimization Results',
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': metadata or {},
            'improvement_plot': improvement_fig.to_html(include_plotlyjs=False),
            'parameter_plots': param_figs,
            'best_configurations': best_configs_df.to_html(classes='table table-striped', index=False),
            'optimization_summary': self._create_optimization_summary(optimization_history)
        }
        
        # Generate report
        template = self._create_optimization_results_template()
        html_content = template.render(**template_data)
        
        # Save report
        output_path = os.path.join(self.output_dir, f'optimization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def generate_regime_analysis_report(
        self,
        regime_data: pd.DataFrame,
        transition_matrix: pd.DataFrame,
        performance_by_regime: Dict[str, Dict[str, float]],
        detection_accuracy: Dict[str, float],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Generate regime analysis report.
        
        Args:
            regime_data: DataFrame with regime predictions and actual values
            transition_matrix: Regime transition probabilities
            performance_by_regime: Performance metrics by regime
            detection_accuracy: Regime detection accuracy metrics
            metadata: Additional metadata
            
        Returns:
            Path to generated report
        """
        # Create regime timeline
        timeline_fig = go.Figure()
        
        # Plot regime states
        timeline_fig.add_trace(go.Scatter(
            x=regime_data.index,
            y=regime_data['regime'],
            mode='lines',
            name='Detected Regime',
            line=dict(color='blue', width=2),
            fill='tozeroy'
        ))
        
        if 'actual_regime' in regime_data.columns:
            timeline_fig.add_trace(go.Scatter(
                x=regime_data.index,
                y=regime_data['actual_regime'],
                mode='lines',
                name='Actual Regime',
                line=dict(color='red', width=1, dash='dash'),
                opacity=0.7
            ))
        
        timeline_fig.update_layout(
            title='Regime Timeline',
            xaxis_title='Date',
            yaxis_title='Regime',
            height=400
        )
        
        # Create transition diagram
        transition_fig = go.Figure(data=go.Heatmap(
            z=transition_matrix.values,
            x=transition_matrix.columns,
            y=transition_matrix.index,
            colorscale='Viridis',
            text=transition_matrix.values.round(3),
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        transition_fig.update_layout(
            title='Regime Transition Probabilities',
            xaxis_title='To Regime',
            yaxis_title='From Regime',
            height=500,
            width=600
        )
        
        # Create performance by regime chart
        perf_df = pd.DataFrame(performance_by_regime).T
        
        perf_fig = go.Figure()
        
        for metric in perf_df.columns:
            perf_fig.add_trace(go.Bar(
                name=metric,
                x=perf_df.index,
                y=perf_df[metric],
                text=perf_df[metric].round(3),
                textposition='auto'
            ))
        
        perf_fig.update_layout(
            title='Performance by Regime',
            xaxis_title='Regime',
            yaxis_title='Value',
            barmode='group',
            height=500
        )
        
        # Create accuracy metrics chart
        accuracy_fig = go.Figure()
        
        accuracy_fig.add_trace(go.Bar(
            x=list(detection_accuracy.keys()),
            y=list(detection_accuracy.values()),
            text=[f"{v:.3f}" for v in detection_accuracy.values()],
            textposition='auto',
            marker_color='lightcoral'
        ))
        
        accuracy_fig.update_layout(
            title='Regime Detection Accuracy',
            xaxis_title='Metric',
            yaxis_title='Score',
            height=400
        )
        
        # Prepare template data
        template_data = {
            'report_title': 'Regime Analysis Report',
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': metadata or {},
            'timeline_plot': timeline_fig.to_html(include_plotlyjs=False),
            'transition_diagram': transition_fig.to_html(include_plotlyjs=False),
            'performance_plot': perf_fig.to_html(include_plotlyjs=False),
            'accuracy_plot': accuracy_fig.to_html(include_plotlyjs=False),
            'regime_statistics': self._calculate_regime_statistics(regime_data)
        }
        
        # Generate report
        template = self._create_regime_analysis_template()
        html_content = template.render(**template_data)
        
        # Save report
        output_path = os.path.join(self.output_dir, f'regime_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def generate_strategy_comparison_report(
        self,
        ml_performance: pd.DataFrame,
        baseline_performance: pd.DataFrame,
        comparison_metrics: Dict[str, Dict[str, float]],
        trade_analysis: Dict[str, pd.DataFrame],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Generate ML strategy comparison report.
        
        Args:
            ml_performance: ML strategy performance data
            baseline_performance: Baseline strategy performance data
            comparison_metrics: Comparison metrics between strategies
            trade_analysis: Trade-level analysis for each strategy
            metadata: Additional metadata
            
        Returns:
            Path to generated report
        """
        # Create cumulative returns comparison
        returns_fig = go.Figure()
        
        returns_fig.add_trace(go.Scatter(
            x=ml_performance.index,
            y=ml_performance['cumulative_returns'],
            mode='lines',
            name='ML Strategy',
            line=dict(color='blue', width=2)
        ))
        
        returns_fig.add_trace(go.Scatter(
            x=baseline_performance.index,
            y=baseline_performance['cumulative_returns'],
            mode='lines',
            name='Baseline Strategy',
            line=dict(color='gray', width=2)
        ))
        
        returns_fig.update_layout(
            title='Cumulative Returns Comparison',
            xaxis_title='Date',
            yaxis_title='Cumulative Returns',
            height=500
        )
        
        # Create metrics comparison
        metrics_df = pd.DataFrame(comparison_metrics)
        
        metrics_fig = go.Figure()
        
        for strategy in metrics_df.columns:
            metrics_fig.add_trace(go.Bar(
                name=strategy,
                x=metrics_df.index,
                y=metrics_df[strategy],
                text=metrics_df[strategy].round(3),
                textposition='auto'
            ))
        
        metrics_fig.update_layout(
            title='Strategy Metrics Comparison',
            xaxis_title='Metric',
            yaxis_title='Value',
            barmode='group',
            height=500
        )
        
        # Create drawdown comparison
        drawdown_fig = go.Figure()
        
        drawdown_fig.add_trace(go.Scatter(
            x=ml_performance.index,
            y=ml_performance['drawdown'] * 100,
            mode='lines',
            name='ML Strategy',
            line=dict(color='red', width=1),
            fill='tozeroy'
        ))
        
        drawdown_fig.add_trace(go.Scatter(
            x=baseline_performance.index,
            y=baseline_performance['drawdown'] * 100,
            mode='lines',
            name='Baseline Strategy',
            line=dict(color='orange', width=1),
            fill='tozeroy',
            opacity=0.5
        ))
        
        drawdown_fig.update_layout(
            title='Drawdown Comparison (%)',
            xaxis_title='Date',
            yaxis_title='Drawdown %',
            height=400
        )
        
        # Create trade analysis plots
        trade_plots = []
        
        for strategy_name, trades_df in trade_analysis.items():
            # Win rate by month
            monthly_wins = trades_df.groupby(pd.Grouper(freq='ME')).agg({
                'profit': ['count', lambda x: (x > 0).sum() / len(x) * 100]
            })
            monthly_wins.columns = ['trade_count', 'win_rate']
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=monthly_wins.index,
                y=monthly_wins['win_rate'],
                name='Win Rate %',
                marker_color='lightgreen'
            ))
            
            fig.update_layout(
                title=f'Monthly Win Rate - {strategy_name}',
                xaxis_title='Month',
                yaxis_title='Win Rate %',
                height=300
            )
            
            trade_plots.append({
                'name': strategy_name,
                'plot': fig.to_html(include_plotlyjs=False, div_id=f"trades_{strategy_name}")
            })
        
        # Prepare template data
        template_data = {
            'report_title': 'ML Strategy Comparison',
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': metadata or {},
            'returns_plot': returns_fig.to_html(include_plotlyjs=False),
            'metrics_plot': metrics_fig.to_html(include_plotlyjs=False),
            'drawdown_plot': drawdown_fig.to_html(include_plotlyjs=False),
            'trade_analysis': trade_plots,
            'strategy_summary': self._create_strategy_summary(comparison_metrics)
        }
        
        # Generate report
        template = self._create_strategy_comparison_template()
        html_content = template.render(**template_data)
        
        # Save report
        output_path = os.path.join(self.output_dir, f'strategy_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def _calculate_feature_stats(self, feature_distributions: Dict[str, pd.Series]) -> pd.DataFrame:
        """Calculate statistics for features."""
        stats = []
        
        for feature_name, distribution in feature_distributions.items():
            stats.append({
                'Feature': feature_name,
                'Mean': distribution.mean(),
                'Std': distribution.std(),
                'Min': distribution.min(),
                'Max': distribution.max(),
                'Skew': distribution.skew(),
                'Kurtosis': distribution.kurtosis()
            })
        
        return pd.DataFrame(stats).round(3).to_html(classes='table table-striped', index=False)
    
    def _create_model_summaries(self, model_metrics: Dict[str, Dict[str, float]]) -> List[Dict]:
        """Create model summary cards."""
        summaries = []
        
        for model_name, metrics in model_metrics.items():
            summary = {
                'name': model_name,
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'auc': metrics.get('auc', 0)
            }
            summaries.append(summary)
        
        return summaries
    
    def _create_optimization_summary(self, optimization_history: List[Dict]) -> Dict:
        """Create optimization summary statistics."""
        history_df = pd.DataFrame(optimization_history)
        
        return {
            'total_iterations': len(optimization_history),
            'best_score': history_df['best_score'].max(),
            'improvement': (history_df['best_score'].iloc[-1] - history_df['best_score'].iloc[0]) / history_df['best_score'].iloc[0] * 100,
            'convergence_iteration': history_df[history_df['best_score'] == history_df['best_score'].max()].index[0]
        }
    
    def _calculate_regime_statistics(self, regime_data: pd.DataFrame) -> Dict:
        """Calculate regime statistics."""
        regime_counts = regime_data['regime'].value_counts()
        regime_durations = []
        
        current_regime = regime_data['regime'].iloc[0]
        duration = 0
        
        for regime in regime_data['regime']:
            if regime == current_regime:
                duration += 1
            else:
                regime_durations.append(duration)
                current_regime = regime
                duration = 1
        
        regime_durations.append(duration)
        
        return {
            'regime_counts': regime_counts.to_dict(),
            'avg_duration': np.mean(regime_durations),
            'max_duration': np.max(regime_durations),
            'min_duration': np.min(regime_durations)
        }
    
    def _create_strategy_summary(self, comparison_metrics: Dict[str, Dict[str, float]]) -> Dict:
        """Create strategy comparison summary."""
        ml_metrics = comparison_metrics.get('ML Strategy', {})
        baseline_metrics = comparison_metrics.get('Baseline Strategy', {})
        
        return {
            'return_improvement': (ml_metrics.get('total_return', 0) - baseline_metrics.get('total_return', 0)) / baseline_metrics.get('total_return', 1) * 100,
            'sharpe_improvement': ml_metrics.get('sharpe_ratio', 0) - baseline_metrics.get('sharpe_ratio', 0),
            'max_dd_improvement': baseline_metrics.get('max_drawdown', 0) - ml_metrics.get('max_drawdown', 0),
            'win_rate_improvement': ml_metrics.get('win_rate', 0) - baseline_metrics.get('win_rate', 0)
        }
    
    def _create_feature_analysis_template(self):
        """Create feature analysis HTML template."""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; background-color: #f5f5f5; }
        .card { margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .metric-card { text-align: center; padding: 20px; }
        .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .metric-label { color: #6c757d; }
        .section-header { margin: 30px 0 20px 0; color: #333; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1 class="text-center mb-4">{{ report_title }}</h1>
        <p class="text-center text-muted">Generated on {{ generation_time }}</p>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3>Feature Importance Analysis</h3>
                    </div>
                    <div class="card-body">
                        {{ feature_importance_plot }}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3>Feature Correlation Matrix</h3>
                    </div>
                    <div class="card-body">
                        {{ correlation_heatmap }}
                    </div>
                </div>
            </div>
        </div>
        
        <h2 class="section-header">Feature Distributions</h2>
        <div class="row">
            {% for dist in distribution_plots %}
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        {{ dist.plot }}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3>Feature Statistics</h3>
                    </div>
                    <div class="card-body">
                        {{ feature_stats }}
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        return self.env.from_string(template_str)
    
    def _create_performance_dashboard_template(self):
        """Create performance dashboard HTML template."""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; background-color: #f5f5f5; }
        .card { margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .metric-card { text-align: center; padding: 20px; }
        .metric-value { font-size: 2em; font-weight: bold; }
        .metric-label { color: #6c757d; }
        .model-card { border-left: 4px solid #007bff; }
        .section-header { margin: 30px 0 20px 0; color: #333; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1 class="text-center mb-4">{{ report_title }}</h1>
        <p class="text-center text-muted">Generated on {{ generation_time }}</p>
        
        <div class="row">
            {% for model in model_summaries %}
            <div class="col-md-3">
                <div class="card metric-card model-card">
                    <h4>{{ model.name }}</h4>
                    <div class="metric-value" style="color: {% if model.accuracy > 0.8 %}green{% elif model.accuracy > 0.6 %}orange{% else %}red{% endif %}">
                        {{ "{:.1%}".format(model.accuracy) }}
                    </div>
                    <div class="metric-label">Accuracy</div>
                    <hr>
                    <small>
                        F1: {{ "{:.3f}".format(model.f1_score) }}<br>
                        AUC: {{ "{:.3f}".format(model.auc) }}
                    </small>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3>Model Performance Comparison</h3>
                    </div>
                    <div class="card-body">
                        {{ metrics_plot }}
                    </div>
                </div>
            </div>
        </div>
        
        <h2 class="section-header">Confusion Matrices</h2>
        <div class="row">
            {% for cm in confusion_matrices %}
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        {{ cm.plot }}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h3>ROC Curves</h3>
                    </div>
                    <div class="card-body">
                        {{ roc_curves }}
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h3>Profit Curves</h3>
                    </div>
                    <div class="card-body">
                        {{ profit_curves }}
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        return self.env.from_string(template_str)
    
    def _create_optimization_results_template(self):
        """Create optimization results HTML template."""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; background-color: #f5f5f5; }
        .card { margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .summary-card { background-color: #e3f2fd; }
        .metric-value { font-size: 2em; font-weight: bold; color: #1976d2; }
        .metric-label { color: #6c757d; }
        .section-header { margin: 30px 0 20px 0; color: #333; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1 class="text-center mb-4">{{ report_title }}</h1>
        <p class="text-center text-muted">Generated on {{ generation_time }}</p>
        
        <div class="row">
            <div class="col-md-3">
                <div class="card metric-card summary-card">
                    <div class="card-body text-center">
                        <div class="metric-value">{{ optimization_summary.total_iterations }}</div>
                        <div class="metric-label">Total Iterations</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card summary-card">
                    <div class="card-body text-center">
                        <div class="metric-value">{{ "{:.3f}".format(optimization_summary.best_score) }}</div>
                        <div class="metric-label">Best Score</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card summary-card">
                    <div class="card-body text-center">
                        <div class="metric-value">{{ "{:.1f}%".format(optimization_summary.improvement) }}</div>
                        <div class="metric-label">Improvement</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card summary-card">
                    <div class="card-body text-center">
                        <div class="metric-value">{{ optimization_summary.convergence_iteration }}</div>
                        <div class="metric-label">Convergence Iteration</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3>Optimization Progress</h3>
                    </div>
                    <div class="card-body">
                        {{ improvement_plot }}
                    </div>
                </div>
            </div>
        </div>
        
        <h2 class="section-header">Parameter Evolution</h2>
        <div class="row">
            {% for param in parameter_plots %}
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        {{ param.plot }}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3>Best Configurations</h3>
                    </div>
                    <div class="card-body">
                        {{ best_configurations }}
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        return self.env.from_string(template_str)
    
    def _create_regime_analysis_template(self):
        """Create regime analysis HTML template."""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; background-color: #f5f5f5; }
        .card { margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .regime-stat { text-align: center; padding: 15px; }
        .regime-value { font-size: 1.8em; font-weight: bold; color: #00796b; }
        .regime-label { color: #6c757d; font-size: 0.9em; }
        .section-header { margin: 30px 0 20px 0; color: #333; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1 class="text-center mb-4">{{ report_title }}</h1>
        <p class="text-center text-muted">Generated on {{ generation_time }}</p>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3>Regime Timeline</h3>
                    </div>
                    <div class="card-body">
                        {{ timeline_plot }}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            {% for regime, count in regime_statistics.regime_counts.items() %}
            <div class="col-md-3">
                <div class="card regime-stat">
                    <div class="regime-value">{{ count }}</div>
                    <div class="regime-label">{{ regime }} Periods</div>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h3>Regime Transitions</h3>
                    </div>
                    <div class="card-body">
                        {{ transition_diagram }}
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h3>Detection Accuracy</h3>
                    </div>
                    <div class="card-body">
                        {{ accuracy_plot }}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3>Performance by Regime</h3>
                    </div>
                    <div class="card-body">
                        {{ performance_plot }}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3>Regime Duration Statistics</h3>
                    </div>
                    <div class="card-body">
                        <table class="table">
                            <tr>
                                <td><strong>Average Duration:</strong></td>
                                <td>{{ "{:.1f}".format(regime_statistics.avg_duration) }} periods</td>
                            </tr>
                            <tr>
                                <td><strong>Maximum Duration:</strong></td>
                                <td>{{ regime_statistics.max_duration }} periods</td>
                            </tr>
                            <tr>
                                <td><strong>Minimum Duration:</strong></td>
                                <td>{{ regime_statistics.min_duration }} periods</td>
                            </tr>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        return self.env.from_string(template_str)
    
    def _create_strategy_comparison_template(self):
        """Create strategy comparison HTML template."""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; background-color: #f5f5f5; }
        .card { margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .improvement-card { text-align: center; padding: 20px; }
        .improvement-value { font-size: 2em; font-weight: bold; }
        .improvement-label { color: #6c757d; }
        .positive { color: #4caf50; }
        .negative { color: #f44336; }
        .section-header { margin: 30px 0 20px 0; color: #333; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1 class="text-center mb-4">{{ report_title }}</h1>
        <p class="text-center text-muted">Generated on {{ generation_time }}</p>
        
        <div class="row">
            <div class="col-md-3">
                <div class="card improvement-card">
                    <div class="improvement-value {% if strategy_summary.return_improvement > 0 %}positive{% else %}negative{% endif %}">
                        {{ "{:+.1f}%".format(strategy_summary.return_improvement) }}
                    </div>
                    <div class="improvement-label">Return Improvement</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card improvement-card">
                    <div class="improvement-value {% if strategy_summary.sharpe_improvement > 0 %}positive{% else %}negative{% endif %}">
                        {{ "{:+.3f}".format(strategy_summary.sharpe_improvement) }}
                    </div>
                    <div class="improvement-label">Sharpe Improvement</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card improvement-card">
                    <div class="improvement-value {% if strategy_summary.max_dd_improvement > 0 %}positive{% else %}negative{% endif %}">
                        {{ "{:+.1f}%".format(strategy_summary.max_dd_improvement * 100) }}
                    </div>
                    <div class="improvement-label">Max DD Improvement</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card improvement-card">
                    <div class="improvement-value {% if strategy_summary.win_rate_improvement > 0 %}positive{% else %}negative{% endif %}">
                        {{ "{:+.1f}%".format(strategy_summary.win_rate_improvement * 100) }}
                    </div>
                    <div class="improvement-label">Win Rate Improvement</div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3>Cumulative Returns</h3>
                    </div>
                    <div class="card-body">
                        {{ returns_plot }}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h3>Performance Metrics</h3>
                    </div>
                    <div class="card-body">
                        {{ metrics_plot }}
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h3>Drawdown Analysis</h3>
                    </div>
                    <div class="card-body">
                        {{ drawdown_plot }}
                    </div>
                </div>
            </div>
        </div>
        
        <h2 class="section-header">Trade Analysis</h2>
        <div class="row">
            {% for trade in trade_analysis %}
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h4>{{ trade.name }}</h4>
                    </div>
                    <div class="card-body">
                        {{ trade.plot }}
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""
        return self.env.from_string(template_str)