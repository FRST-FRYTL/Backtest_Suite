"""
Standard Report Generator for Backtest Suite.

This module provides the main class for generating standardized reports
from backtest results in multiple formats (Markdown, HTML, JSON).
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .report_config import ReportConfig, ReportSection
from .visualizations import ReportVisualizations


class StandardReportGenerator:
    """Generate standardized reports from backtest results."""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.
        
        Args:
            config: Report configuration object. If None, uses default config.
        """
        self.config = config or ReportConfig()
        self.visualizations = ReportVisualizations(
            {'template': self.config.style.chart_template}
        )
        self.report_data = {}
        self.charts = {}
        
    def generate_report(self, 
                       backtest_results: Union[Dict[str, Any], pd.DataFrame],
                       strategy_name: str,
                       output_dir: str,
                       formats: List[str] = ['markdown', 'html', 'json'],
                       additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Generate reports in specified formats.
        
        Args:
            backtest_results: Backtest results data
            strategy_name: Name of the strategy
            output_dir: Directory to save reports
            formats: List of output formats
            additional_data: Additional data to include in report
            
        Returns:
            Dictionary mapping format to file path
        """
        # Prepare output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Process backtest results
        self._process_backtest_results(backtest_results, additional_data)
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Generate reports in each format
        output_paths = {}
        
        if 'markdown' in formats:
            md_path = os.path.join(output_dir, f"{strategy_name}_report_{timestamp}.md")
            self._generate_markdown_report(md_path, strategy_name)
            output_paths['markdown'] = md_path
            
        if 'html' in formats:
            html_path = os.path.join(output_dir, f"{strategy_name}_report_{timestamp}.html")
            self._generate_html_report(html_path, strategy_name)
            output_paths['html'] = html_path
            
        if 'json' in formats:
            json_path = os.path.join(output_dir, f"{strategy_name}_report_{timestamp}.json")
            self._generate_json_report(json_path, strategy_name)
            output_paths['json'] = json_path
            
        # Save charts
        if self.charts:
            charts_dir = os.path.join(output_dir, f"charts_{timestamp}")
            os.makedirs(charts_dir, exist_ok=True)
            chart_paths = self.visualizations.save_all_charts(self.charts, charts_dir)
            output_paths['charts'] = chart_paths
            
        return output_paths
    
    def _process_backtest_results(self, results: Union[Dict, pd.DataFrame], 
                                 additional_data: Optional[Dict] = None) -> None:
        """Process and standardize backtest results."""
        # Handle different input formats
        if isinstance(results, pd.DataFrame):
            self.report_data['trades'] = results
            self.report_data['metrics'] = self._calculate_metrics_from_trades(results)
        else:
            self.report_data = results.copy()
            
        # Add additional data
        if additional_data:
            self.report_data.update(additional_data)
            
        # Ensure required metrics exist
        self._ensure_required_metrics()
        
        # Calculate additional metrics if needed
        self._calculate_additional_metrics()
        
    def _calculate_metrics_from_trades(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics from trades DataFrame."""
        metrics = {}
        
        # Basic trade statistics
        metrics['total_trades'] = len(trades)
        
        if 'pnl' in trades.columns:
            metrics['total_pnl'] = trades['pnl'].sum()
            metrics['avg_pnl'] = trades['pnl'].mean()
            metrics['win_rate'] = len(trades[trades['pnl'] > 0]) / len(trades) if len(trades) > 0 else 0
            
            # Win/loss statistics
            winning_trades = trades[trades['pnl'] > 0]
            losing_trades = trades[trades['pnl'] < 0]
            
            metrics['avg_win'] = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            metrics['avg_loss'] = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
            
            # Profit factor
            gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
            
        return metrics
    
    def _ensure_required_metrics(self) -> None:
        """Ensure all required metrics exist with default values."""
        default_metrics = {
            'total_return': 0.0,
            'annual_return': 0.0,
            'monthly_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_loss_ratio': 0.0
        }
        
        if 'metrics' not in self.report_data:
            self.report_data['metrics'] = {}
            
        for metric, default_value in default_metrics.items():
            if metric not in self.report_data['metrics']:
                self.report_data['metrics'][metric] = default_value
                
    def _calculate_additional_metrics(self) -> None:
        """Calculate additional metrics from available data."""
        metrics = self.report_data['metrics']
        
        # Calculate Calmar ratio if we have annual return and max drawdown
        if metrics.get('annual_return') and metrics.get('max_drawdown'):
            metrics['calmar_ratio'] = abs(metrics['annual_return'] / metrics['max_drawdown'])
            
        # Calculate average trades per month if we have trade data
        if 'trades' in self.report_data and 'entry_date' in self.report_data['trades'].columns:
            trades_df = self.report_data['trades']
            date_range = pd.to_datetime(trades_df['entry_date'])
            months = (date_range.max() - date_range.min()).days / 30.44
            metrics['avg_trades_per_month'] = len(trades_df) / months if months > 0 else 0
            
    def _generate_visualizations(self) -> None:
        """Generate all visualizations for the report."""
        metrics = self.report_data.get('metrics', {})
        
        # Performance summary chart
        self.charts['performance_summary'] = self.visualizations.performance_summary_chart(
            metrics, self.config.thresholds
        )
        
        # Generate other charts based on available data
        if 'equity_curve' in self.report_data:
            equity = self.report_data['equity_curve']
            if isinstance(equity, pd.Series):
                returns = equity.pct_change().dropna()
                
                # Cumulative returns
                self.charts['cumulative_returns'] = self.visualizations.cumulative_returns(returns)
                
                # Drawdown chart
                self.charts['drawdown'] = self.visualizations.drawdown_chart(returns)
                
                # Monthly heatmap
                if len(returns) > 30:  # Only if we have enough data
                    self.charts['monthly_heatmap'] = self.visualizations.monthly_returns_heatmap(returns)
                    
                # Rolling metrics
                if len(returns) > 252:  # Only if we have enough data
                    self.charts['rolling_metrics'] = self.visualizations.rolling_metrics(returns)
                    
        # Trade distribution
        if 'trades' in self.report_data:
            self.charts['trade_distribution'] = self.visualizations.trade_distribution(
                self.report_data['trades']
            )
            
        # Parameter heatmap if optimization results available
        if 'optimization_results' in self.report_data:
            opt_results = self.report_data['optimization_results']
            if isinstance(opt_results, pd.DataFrame) and len(opt_results) > 1:
                # Find two parameters to plot
                param_cols = [col for col in opt_results.columns 
                             if col not in ['sharpe_ratio', 'total_return', 'max_drawdown']]
                if len(param_cols) >= 2:
                    self.charts['parameter_heatmap'] = self.visualizations.parameter_heatmap(
                        opt_results, param_cols[0], param_cols[1]
                    )
                    
    def _generate_markdown_report(self, output_path: str, strategy_name: str) -> None:
        """Generate markdown format report."""
        sections = []
        
        # Add header
        sections.append(f"# {strategy_name} - Standardized Backtest Report")
        sections.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Generate each enabled section
        for section in self.config.get_section_order():
            section_content = self._generate_section_content(section, 'markdown')
            if section_content:
                sections.append(section_content)
                
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n\n'.join(sections))
            
    def _generate_html_report(self, output_path: str, strategy_name: str) -> None:
        """Generate HTML format report."""
        html_template = self._get_html_template()
        
        # Generate sections
        sections_html = []
        for section in self.config.get_section_order():
            section_content = self._generate_section_content(section, 'html')
            if section_content:
                sections_html.append(f'<div class="section">{section_content}</div>')
                
        # Embed charts
        charts_html = []
        for chart_name, fig in self.charts.items():
            chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            charts_html.append(f'<div class="chart">{chart_html}</div>')
            
        # Fill template
        html_content = html_template.format(
            title=f"{strategy_name} - Standardized Backtest Report",
            strategy_name=strategy_name,
            generated_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            sections='\n'.join(sections_html),
            charts='\n'.join(charts_html),
            primary_color=self.config.style.primary_color,
            font_family=self.config.style.font_family
        )
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
            
    def _generate_json_report(self, output_path: str, strategy_name: str) -> None:
        """Generate JSON format report."""
        json_data = {
            'strategy_name': strategy_name,
            'generated_at': datetime.now().isoformat(),
            'config': {
                'sections': [s.value for s in self.config.get_section_order()],
                'metrics': list(self.config.metrics.keys()),
                'thresholds': {k: vars(v) for k, v in self.config.thresholds.items()}
            },
            'data': {}
        }
        
        # Add metrics
        json_data['data']['metrics'] = self.report_data.get('metrics', {})
        
        # Add trade data if available
        if 'trades' in self.report_data:
            trades_df = self.report_data['trades']
            json_data['data']['trades'] = trades_df.to_dict('records')
            
        # Add optimization results if available
        if 'optimization_results' in self.report_data:
            opt_df = self.report_data['optimization_results']
            json_data['data']['optimization'] = opt_df.to_dict('records')
            
        # Add any additional data
        for key, value in self.report_data.items():
            if key not in ['metrics', 'trades', 'optimization_results', 'equity_curve']:
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    json_data['data'][key] = value.to_dict()
                else:
                    json_data['data'][key] = value
                    
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
            
    def _generate_section_content(self, section: ReportSection, format: str) -> str:
        """Generate content for a specific section."""
        section_config = self.config.sections[section]
        content = []
        
        # Section title
        if format == 'markdown':
            content.append(f"## {section_config.title}")
        else:  # HTML
            content.append(f"<h2>{section_config.title}</h2>")
            
        # Generate content based on section type
        if section == ReportSection.EXECUTIVE_SUMMARY:
            content.append(self._generate_executive_summary(format))
        elif section == ReportSection.PERFORMANCE_METRICS:
            content.append(self._generate_performance_metrics(format))
        elif section == ReportSection.RISK_ANALYSIS:
            content.append(self._generate_risk_analysis(format))
        elif section == ReportSection.TRADE_ANALYSIS:
            content.append(self._generate_trade_analysis(format))
        elif section == ReportSection.METHODOLOGY:
            content.append(self._generate_methodology(format))
        elif section == ReportSection.RECOMMENDATIONS:
            content.append(self._generate_recommendations(format))
            
        # Add custom content if provided
        if section_config.custom_content:
            content.append(section_config.custom_content)
            
        return '\n\n'.join(content)
    
    def _generate_executive_summary(self, format: str) -> str:
        """Generate executive summary section."""
        metrics = self.report_data.get('metrics', {})
        
        if format == 'markdown':
            summary = []
            
            # Key metrics
            summary.append("### Key Performance Metrics")
            summary.append(f"- **Total Return**: {metrics.get('total_return', 0)*100:.2f}%")
            summary.append(f"- **Sharpe Ratio**: {metrics.get('sharpe_ratio', 0):.3f}")
            summary.append(f"- **Maximum Drawdown**: {metrics.get('max_drawdown', 0)*100:.2f}%")
            summary.append(f"- **Win Rate**: {metrics.get('win_rate', 0)*100:.1f}%")
            
            # Performance category
            sharpe = metrics.get('sharpe_ratio', 0)
            if sharpe >= 2.0:
                performance = "Excellent"
            elif sharpe >= 1.5:
                performance = "Good"
            elif sharpe >= 1.0:
                performance = "Acceptable"
            else:
                performance = "Poor"
                
            summary.append(f"\n### Overall Performance Assessment: **{performance}**")
            
            return '\n'.join(summary)
            
        else:  # HTML
            return self._generate_executive_summary_html(metrics)
            
    def _generate_executive_summary_html(self, metrics: Dict[str, float]) -> str:
        """Generate HTML executive summary."""
        html = """
        <div class="executive-summary">
            <h3>Key Performance Metrics</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Total Return</h4>
                    <p class="metric-value">{total_return:.2f}%</p>
                </div>
                <div class="metric-card">
                    <h4>Sharpe Ratio</h4>
                    <p class="metric-value">{sharpe_ratio:.3f}</p>
                </div>
                <div class="metric-card">
                    <h4>Max Drawdown</h4>
                    <p class="metric-value">{max_drawdown:.2f}%</p>
                </div>
                <div class="metric-card">
                    <h4>Win Rate</h4>
                    <p class="metric-value">{win_rate:.1f}%</p>
                </div>
            </div>
        </div>
        """.format(
            total_return=metrics.get('total_return', 0) * 100,
            sharpe_ratio=metrics.get('sharpe_ratio', 0),
            max_drawdown=abs(metrics.get('max_drawdown', 0)) * 100,
            win_rate=metrics.get('win_rate', 0) * 100
        )
        return html
        
    def _generate_performance_metrics(self, format: str) -> str:
        """Generate performance metrics section."""
        metrics = self.report_data.get('metrics', {})
        
        if format == 'markdown':
            content = []
            content.append("### Returns Analysis")
            content.append(f"- Total Return: {metrics.get('total_return', 0)*100:.2f}%")
            content.append(f"- Annual Return: {metrics.get('annual_return', 0)*100:.2f}%")
            content.append(f"- Monthly Return: {metrics.get('monthly_return', 0)*100:.2f}%")
            
            content.append("\n### Risk-Adjusted Returns")
            content.append(f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            content.append(f"- Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
            content.append(f"- Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
            
            return '\n'.join(content)
        else:
            return self.visualizations.create_performance_table(
                metrics, self.config.thresholds
            )
            
    def _generate_risk_analysis(self, format: str) -> str:
        """Generate risk analysis section."""
        metrics = self.report_data.get('metrics', {})
        
        if format == 'markdown':
            content = []
            content.append("### Risk Metrics")
            content.append(f"- Maximum Drawdown: {abs(metrics.get('max_drawdown', 0))*100:.2f}%")
            content.append(f"- Annual Volatility: {metrics.get('volatility', 0)*100:.2f}%")
            content.append(f"- Downside Deviation: {metrics.get('downside_deviation', 0)*100:.2f}%")
            
            return '\n'.join(content)
        else:
            return "<p>See risk analysis charts below.</p>"
            
    def _generate_trade_analysis(self, format: str) -> str:
        """Generate trade analysis section."""
        metrics = self.report_data.get('metrics', {})
        
        if format == 'markdown':
            content = []
            content.append("### Trade Statistics")
            content.append(f"- Total Trades: {metrics.get('total_trades', 0):,}")
            content.append(f"- Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            content.append(f"- Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            content.append(f"- Average Win: ${metrics.get('avg_win', 0):,.2f}")
            content.append(f"- Average Loss: ${metrics.get('avg_loss', 0):,.2f}")
            content.append(f"- Win/Loss Ratio: {metrics.get('win_loss_ratio', 0):.2f}")
            
            return '\n'.join(content)
        else:
            return "<p>See trade analysis charts below.</p>"
            
    def _generate_methodology(self, format: str) -> str:
        """Generate methodology section."""
        if format == 'markdown':
            content = []
            content.append("### Backtest Configuration")
            
            if 'config' in self.report_data:
                config = self.report_data['config']
                content.append(f"- Start Date: {config.get('start_date', 'N/A')}")
                content.append(f"- End Date: {config.get('end_date', 'N/A')}")
                content.append(f"- Initial Capital: ${config.get('initial_capital', 10000):,}")
                content.append(f"- Commission: {config.get('commission', 0)*100:.2f}%")
                
            return '\n'.join(content)
        else:
            return "<p>See technical details for backtest configuration.</p>"
            
    def _generate_recommendations(self, format: str) -> str:
        """Generate recommendations section."""
        metrics = self.report_data.get('metrics', {})
        sharpe = metrics.get('sharpe_ratio', 0)
        
        if format == 'markdown':
            content = []
            content.append("### Trading Recommendations")
            
            if sharpe >= 1.5:
                content.append("- ✅ Strategy shows strong performance and is recommended for live trading")
            elif sharpe >= 1.0:
                content.append("- ⚠️ Strategy shows acceptable performance but may benefit from optimization")
            else:
                content.append("- ❌ Strategy performance is below acceptable thresholds")
                
            content.append("\n### Risk Management Guidelines")
            content.append(f"- Maximum position size: Based on {abs(metrics.get('max_drawdown', 0.2))*100:.1f}% max drawdown")
            content.append("- Consider implementing stop-loss orders")
            content.append("- Monitor strategy performance regularly")
            
            return '\n'.join(content)
        else:
            return "<p>Recommendations based on performance metrics.</p>"
            
    def _get_html_template(self) -> str:
        """Get HTML template for reports."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: {font_family};
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }}
        h1, h2, h3 {{
            color: {primary_color};
        }}
        .header {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            text-align: center;
        }}
        .section {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }}
        .metric-card h4 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: {primary_color};
            margin: 0;
        }}
        .chart {{
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            margin: 20px 0;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{strategy_name}</h1>
        <p>Standardized Backtest Report</p>
        <p>Generated: {generated_date}</p>
    </div>
    
    {sections}
    
    <div class="section">
        <h2>Visualizations</h2>
        {charts}
    </div>
    
    <div class="footer">
        <p>Generated by Backtest Suite - Standard Report Generator</p>
    </div>
</body>
</html>
"""