"""
HTML Template Generator for Standard Reports

This module generates professional HTML reports with embedded visualizations
and interactive elements.
"""

from typing import Dict, Any, List
import base64
from io import BytesIO
import json
from datetime import datetime


def generate_html_report(report_data: Dict[str, Any], config: Any) -> str:
    """Generate complete HTML report from data"""
    
    # Extract data
    metadata = report_data.get("metadata", {})
    sections = report_data.get("sections", {})
    visualizations = report_data.get("visualizations", {})
    summary = report_data.get("backtest_summary", {})
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config.title} - {metadata.get('report_name', 'Backtest Report')}</title>
    {generate_styles(config)}
    {generate_scripts()}
</head>
<body>
    <div class="container">
        {generate_header(config, metadata)}
        {generate_navigation(sections)}
        
        <main>
            {generate_executive_summary(sections.get('executivesummary', {}))}
            {generate_performance_section(sections.get('performanceanalysis', {}), visualizations)}
            {generate_risk_section(sections.get('riskanalysis', {}), visualizations)}
            {generate_trade_section(sections.get('tradeanalysis', {}), visualizations)}
            {generate_regime_section(sections.get('marketregimeanalysis', {}))}
            {generate_technical_section(sections.get('technicaldetails', {}))}
        </main>
        
        {generate_footer(metadata)}
    </div>
    
    {generate_interactive_scripts()}
</body>
</html>
"""
    
    return html


def generate_styles(config: Any) -> str:
    """Generate CSS styles"""
    colors = config.color_scheme
    
    return f"""
    <style>
        :root {{
            --primary-color: {colors['primary']};
            --secondary-color: {colors['secondary']};
            --success-color: {colors['success']};
            --warning-color: {colors['warning']};
            --danger-color: {colors['danger']};
            --info-color: {colors['info']};
            --background-color: {colors['background']};
            --text-color: {colors['text']};
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: #f5f5f5;
            color: var(--text-color);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: var(--background-color);
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
        }}
        
        header {{
            background-color: var(--primary-color);
            color: white;
            padding: 2rem;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        
        header h2 {{
            font-size: 1.5rem;
            font-weight: normal;
            opacity: 0.9;
        }}
        
        nav {{
            background-color: #f8f9fa;
            padding: 1rem;
            border-bottom: 1px solid #dee2e6;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        
        nav ul {{
            list-style: none;
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
        }}
        
        nav li {{
            margin: 0 1rem;
        }}
        
        nav a {{
            color: var(--text-color);
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            transition: all 0.3s ease;
        }}
        
        nav a:hover, nav a.active {{
            background-color: var(--primary-color);
            color: white;
        }}
        
        main {{
            padding: 2rem;
        }}
        
        section {{
            margin-bottom: 3rem;
            padding: 2rem;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        section h2 {{
            color: var(--primary-color);
            margin-bottom: 1.5rem;
            font-size: 2rem;
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 0.5rem;
        }}
        
        section h3 {{
            color: var(--secondary-color);
            margin: 1.5rem 0 1rem;
            font-size: 1.5rem;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }}
        
        .metric-card {{
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            transition: transform 0.2s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }}
        
        .metric-card h4 {{
            color: var(--text-color);
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .metric-card .value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary-color);
        }}
        
        .metric-card.success .value {{
            color: var(--success-color);
        }}
        
        .metric-card.warning .value {{
            color: var(--warning-color);
        }}
        
        .metric-card.danger .value {{
            color: var(--danger-color);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: var(--text-color);
        }}
        
        tr:hover {{
            background-color: #f8f9fa;
        }}
        
        .chart-container {{
            margin: 2rem 0;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 8px;
        }}
        
        .alert {{
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
            border-left: 4px solid;
        }}
        
        .alert-success {{
            background-color: #d4edda;
            border-color: var(--success-color);
            color: #155724;
        }}
        
        .alert-warning {{
            background-color: #fff3cd;
            border-color: var(--warning-color);
            color: #856404;
        }}
        
        .alert-danger {{
            background-color: #f8d7da;
            border-color: var(--danger-color);
            color: #721c24;
        }}
        
        .alert-info {{
            background-color: #d1ecf1;
            border-color: var(--info-color);
            color: #0c5460;
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            font-size: 0.875rem;
            font-weight: 600;
            border-radius: 4px;
            margin: 0 0.25rem;
        }}
        
        .badge-success {{
            background-color: var(--success-color);
            color: white;
        }}
        
        .badge-warning {{
            background-color: var(--warning-color);
            color: white;
        }}
        
        .badge-danger {{
            background-color: var(--danger-color);
            color: white;
        }}
        
        .progress {{
            height: 20px;
            background-color: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
        }}
        
        .progress-bar {{
            height: 100%;
            background-color: var(--primary-color);
            transition: width 0.6s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.75rem;
        }}
        
        footer {{
            background-color: #f8f9fa;
            padding: 2rem;
            text-align: center;
            border-top: 1px solid #dee2e6;
        }}
        
        .tab-container {{
            margin: 2rem 0;
        }}
        
        .tab-buttons {{
            display: flex;
            border-bottom: 2px solid #dee2e6;
        }}
        
        .tab-button {{
            padding: 0.75rem 1.5rem;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1rem;
            color: var(--text-color);
            transition: all 0.3s ease;
        }}
        
        .tab-button:hover {{
            background-color: #f8f9fa;
        }}
        
        .tab-button.active {{
            color: var(--primary-color);
            border-bottom: 2px solid var(--primary-color);
            margin-bottom: -2px;
        }}
        
        .tab-content {{
            padding: 1.5rem;
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .trade-details-table {{
            font-size: 0.9rem;
        }}
        
        .trade-details-table th {{
            background-color: var(--primary-color);
            color: white;
            font-weight: 600;
            padding: 0.5rem;
        }}
        
        .trade-details-table td {{
            padding: 0.4rem;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .trade-details-table .profit {{
            color: var(--success-color);
            font-weight: 600;
        }}
        
        .trade-details-table .loss {{
            color: var(--danger-color);
            font-weight: 600;
        }}
        
        .price-level {{
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }}
        
        @media (max-width: 768px) {{
            .metric-grid {{
                grid-template-columns: 1fr;
            }}
            
            nav ul {{
                flex-direction: column;
                align-items: center;
            }}
            
            nav li {{
                margin: 0.25rem 0;
            }}
        }}
        
        @media print {{
            nav, footer {{
                display: none;
            }}
            
            section {{
                page-break-inside: avoid;
            }}
        }}
    </style>
    """


def generate_scripts() -> str:
    """Generate JavaScript libraries"""
    return """
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    """


def generate_header(config: Any, metadata: Dict[str, Any]) -> str:
    """Generate report header"""
    return f"""
    <header>
        <h1>{config.title}</h1>
        {f'<h2>{config.subtitle}</h2>' if config.subtitle else ''}
        <p>Generated on {metadata.get('generated_at', datetime.now().isoformat())}</p>
    </header>
    """


def generate_navigation(sections: Dict[str, Any]) -> str:
    """Generate navigation menu"""
    nav_items = []
    
    section_names = {
        'executivesummary': 'Executive Summary',
        'performanceanalysis': 'Performance Analysis',
        'riskanalysis': 'Risk Analysis',
        'tradeanalysis': 'Trade Analysis',
        'marketregimeanalysis': 'Market Regimes',
        'technicaldetails': 'Technical Details'
    }
    
    for section_id, section_name in section_names.items():
        if section_id in sections:
            nav_items.append(f'<li><a href="#{section_id}">{section_name}</a></li>')
    
    return f"""
    <nav>
        <ul>
            {' '.join(nav_items)}
        </ul>
    </nav>
    """


def generate_executive_summary(section_data: Dict[str, Any]) -> str:
    """Generate executive summary section"""
    if not section_data:
        return ""
    
    key_metrics = section_data.get('key_metrics', {})
    assessment = section_data.get('strategy_assessment', '')
    findings = section_data.get('key_findings', [])
    recommendations = section_data.get('recommendations', [])
    
    # Determine assessment class
    assessment_class = 'success'
    if 'Needs Improvement' in assessment:
        assessment_class = 'danger'
    elif 'Acceptable' in assessment:
        assessment_class = 'warning'
    
    metrics_html = []
    for metric, value in key_metrics.items():
        # Determine metric class based on value
        metric_class = ''
        if metric == 'Sharpe Ratio':
            val = float(value.replace(',', '')) if value != 'N/A' else 0
            metric_class = 'success' if val >= 1.5 else 'warning' if val >= 1.0 else 'danger'
        elif metric == 'Maximum Drawdown':
            val = float(value.replace('%', '')) if value != 'N/A' else 0
            metric_class = 'success' if val <= 15 else 'warning' if val <= 25 else 'danger'
        elif metric == 'Win Rate':
            val = float(value.replace('%', '')) if value != 'N/A' else 0
            metric_class = 'success' if val >= 60 else 'warning' if val >= 50 else 'danger'
            
        metrics_html.append(f"""
        <div class="metric-card {metric_class}">
            <h4>{metric}</h4>
            <div class="value">{value}</div>
        </div>
        """)
    
    findings_html = ''.join([f'<li>{finding}</li>' for finding in findings])
    recommendations_html = ''.join([f'<li>{rec}</li>' for rec in recommendations])
    
    return f"""
    <section id="executivesummary">
        <h2>Executive Summary</h2>
        
        <div class="alert alert-{assessment_class}">
            <strong>Overall Assessment:</strong> {assessment}
        </div>
        
        <h3>Key Performance Metrics</h3>
        <div class="metric-grid">
            {''.join(metrics_html)}
        </div>
        
        <h3>Performance Summary</h3>
        <p>{section_data.get('performance_summary', '')}</p>
        
        <h3>Risk Summary</h3>
        <p>{section_data.get('risk_summary', '')}</p>
        
        {f'''
        <h3>Key Findings</h3>
        <ul>
            {findings_html}
        </ul>
        ''' if findings else ''}
        
        {f'''
        <h3>Recommendations</h3>
        <ul>
            {recommendations_html}
        </ul>
        ''' if recommendations else ''}
    </section>
    """


def generate_performance_section(section_data: Dict[str, Any], visualizations: Dict[str, Any]) -> str:
    """Generate performance analysis section"""
    if not section_data:
        return ""
    
    return_analysis = section_data.get('return_analysis', {})
    risk_adjusted = section_data.get('risk_adjusted_metrics', {})
    benchmark = section_data.get('benchmark_comparison', {})
    attribution = section_data.get('performance_attribution', {})
    rolling = section_data.get('rolling_performance', {})
    significance = section_data.get('statistical_significance', {})
    
    # Create return statistics table
    return_stats = return_analysis.get('summary_statistics', {})
    return_table = create_table(
        ['Metric', 'Value'],
        [[k, v] for k, v in return_stats.items()]
    )
    
    # Create risk-adjusted metrics with interpretations
    risk_metrics_html = []
    for metric, data in risk_adjusted.items():
        if isinstance(data, dict):
            risk_metrics_html.append(f"""
            <div class="metric-card">
                <h4>{metric.replace('_', ' ').title()}</h4>
                <div class="value">{data.get('value', 'N/A')}</div>
                <p style="font-size: 0.9rem; margin-top: 0.5rem;">{data.get('interpretation', '')}</p>
            </div>
            """)
    
    # Benchmark comparison table
    benchmark_table = create_table(
        ['Metric', 'Value'],
        [[k.replace('_', ' ').title(), v] for k, v in benchmark.items()]
    )
    
    # Rolling performance tabs
    rolling_tabs = create_tabs(
        [f"{window} Day" for window in rolling.keys()],
        [create_table(
            ['Metric', 'Value'],
            [[k.replace('_', ' ').title(), v] for k, v in window_data.items()]
        ) for window_data in rolling.values()]
    )
    
    # Add visualizations
    viz_html = ""
    if 'equity_curve' in visualizations and 'figure' in visualizations['equity_curve']:
        viz_html += f"""
        <div class="chart-container">
            <div id="equity-curve-chart"></div>
        </div>
        <script>
            Plotly.newPlot('equity-curve-chart', 
                {json.dumps(visualizations['equity_curve']['figure'].to_plotly_json()['data'])},
                {json.dumps(visualizations['equity_curve']['figure'].to_plotly_json()['layout'])}
            );
        </script>
        """
    
    return f"""
    <section id="performanceanalysis">
        <h2>Performance Analysis</h2>
        
        <h3>Return Analysis</h3>
        {return_table}
        
        <h3>Risk-Adjusted Performance</h3>
        <div class="metric-grid">
            {''.join(risk_metrics_html)}
        </div>
        
        {viz_html}
        
        <h3>Benchmark Comparison</h3>
        {benchmark_table}
        
        <h3>Rolling Performance</h3>
        {rolling_tabs}
        
        <h3>Statistical Significance</h3>
        {create_significance_display(significance)}
    </section>
    """


def generate_risk_section(section_data: Dict[str, Any], visualizations: Dict[str, Any]) -> str:
    """Generate risk analysis section"""
    if not section_data:
        return ""
    
    drawdown = section_data.get('drawdown_analysis', {})
    volatility = section_data.get('volatility_analysis', {})
    var = section_data.get('var_analysis', {})
    stress = section_data.get('stress_testing', {})
    risk_metrics = section_data.get('risk_metrics', {})
    
    # Drawdown summary
    dd_summary = drawdown.get('maximum_drawdown', {})
    dd_duration = drawdown.get('drawdown_duration', {})
    top_drawdowns = drawdown.get('top_drawdowns', [])
    
    # Create drawdown table
    dd_table = create_table(
        ['Rank', 'Depth', 'Duration', 'Period'],
        [[dd['rank'], dd['depth'], dd['duration'], dd['period']] for dd in top_drawdowns]
    )
    
    # Volatility analysis
    vol_current = volatility.get('current_volatility', {})
    vol_regimes = volatility.get('volatility_regimes', {})
    
    # VaR display
    var_cards = []
    for confidence, metrics in var.items():
        if isinstance(metrics, dict):
            var_cards.append(f"""
            <div class="metric-card">
                <h4>VaR {confidence}</h4>
                <div class="value">{metrics.get('historical_var', 'N/A')}</div>
                <p style="font-size: 0.9rem;">{metrics.get('interpretation', '')}</p>
            </div>
            """)
    
    # Stress test scenarios
    stress_scenarios = []
    for scenario, data in stress.items():
        if isinstance(data, dict):
            stress_scenarios.append(f"""
            <div class="alert alert-warning">
                <h4>{scenario.replace('_', ' ').title()}</h4>
                <p><strong>Description:</strong> {data.get('description', '')}</p>
                <p><strong>Impact:</strong> {data.get('impact', '')}</p>
            </div>
            """)
    
    # Add drawdown visualization
    viz_html = ""
    if 'drawdown' in visualizations and 'figure' in visualizations['drawdown']:
        viz_html += f"""
        <div class="chart-container">
            <div id="drawdown-chart"></div>
        </div>
        <script>
            Plotly.newPlot('drawdown-chart', 
                {json.dumps(visualizations['drawdown']['figure'].to_plotly_json()['data'])},
                {json.dumps(visualizations['drawdown']['figure'].to_plotly_json()['layout'])}
            );
        </script>
        """
    
    return f"""
    <section id="riskanalysis">
        <h2>Risk Analysis</h2>
        
        <h3>Drawdown Analysis</h3>
        <div class="metric-grid">
            <div class="metric-card danger">
                <h4>Maximum Drawdown</h4>
                <div class="value">{dd_summary.get('value', 'N/A')}</div>
                <p>Date: {dd_summary.get('date', 'N/A')}</p>
            </div>
            <div class="metric-card">
                <h4>Average Drawdown</h4>
                <div class="value">{drawdown.get('average_drawdown', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Current Duration</h4>
                <div class="value">{dd_duration.get('current', 'N/A')}</div>
            </div>
        </div>
        
        {viz_html}
        
        <h4>Top 5 Drawdowns</h4>
        {dd_table}
        
        <h3>Volatility Analysis</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <h4>Current Volatility</h4>
                <div class="value">{vol_current.get('annualized', 'N/A')}</div>
                <p>Regime: {vol_regimes.get('current_regime', 'N/A')}</p>
            </div>
        </div>
        
        <h3>Value at Risk</h3>
        <div class="metric-grid">
            {''.join(var_cards)}
        </div>
        
        <h3>Stress Testing</h3>
        {''.join(stress_scenarios)}
    </section>
    """


def generate_trade_section(section_data: Dict[str, Any], visualizations: Dict[str, Any]) -> str:
    """Generate trade analysis section"""
    if not section_data:
        return ""
    
    if 'message' in section_data:
        return f"""
        <section id="tradeanalysis">
            <h2>Trade Analysis</h2>
            <div class="alert alert-info">
                {section_data['message']}
            </div>
        </section>
        """
    
    statistics = section_data.get('trade_statistics', {})
    win_loss = section_data.get('win_loss_analysis', {})
    duration = section_data.get('trade_duration_analysis', {})
    distribution = section_data.get('trade_distribution', {})
    
    # Trade summary metrics
    summary_metrics = []
    if 'summary' in statistics:
        for metric, value in statistics['summary'].items():
            summary_metrics.append(f"""
            <div class="metric-card">
                <h4>{metric.replace('_', ' ').title()}</h4>
                <div class="value">{value}</div>
            </div>
            """)
    
    # Win/Loss comparison
    win_data = win_loss.get('winning_trades', {})
    loss_data = win_loss.get('losing_trades', {})
    
    win_loss_comparison = create_table(
        ['Metric', 'Winners', 'Losers'],
        [
            ['Count', win_data.get('count', 0), loss_data.get('count', 0)],
            ['Average P&L', win_data.get('avg_win', 'N/A'), loss_data.get('avg_loss', 'N/A')],
            ['Median P&L', win_data.get('median_win', 'N/A'), loss_data.get('median_loss', 'N/A')],
            ['Largest', win_data.get('largest_win', 'N/A'), loss_data.get('largest_loss', 'N/A')],
            ['Avg Duration', win_data.get('avg_duration', 'N/A'), loss_data.get('avg_duration', 'N/A')]
        ]
    )
    
    # Add trade visualization
    viz_html = ""
    if 'trade_analysis' in visualizations and 'figure' in visualizations['trade_analysis']:
        viz_html += f"""
        <div class="chart-container">
            <div id="trade-analysis-chart"></div>
        </div>
        <script>
            Plotly.newPlot('trade-analysis-chart', 
                {json.dumps(visualizations['trade_analysis']['figure'].to_plotly_json()['data'])},
                {json.dumps(visualizations['trade_analysis']['figure'].to_plotly_json()['layout'])}
            );
        </script>
        """
    
    return f"""
    <section id="tradeanalysis">
        <h2>Trade Analysis</h2>
        
        <h3>Trade Statistics</h3>
        <div class="metric-grid">
            {''.join(summary_metrics)}
        </div>
        
        <h3>Win/Loss Analysis</h3>
        <div class="metric-grid">
            <div class="metric-card success">
                <h4>Win Rate</h4>
                <div class="value">{win_loss.get('win_rate', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Profit Factor</h4>
                <div class="value">{statistics.get('profitability', {}).get('profit_factor', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Win/Loss Ratio</h4>
                <div class="value">{win_loss.get('ratios', {}).get('win_loss_ratio', 'N/A')}</div>
            </div>
        </div>
        
        {win_loss_comparison}
        
        {viz_html}
        
        <h3>Trade Price Analysis</h3>
        {create_price_analysis_display(section_data.get('price_analysis', {}))}
        
        <h3>Trade Risk Analysis</h3>
        {create_risk_analysis_display(section_data.get('risk_analysis', {}))}
        
        <h3>Enhanced Trade Table</h3>
        {create_enhanced_trade_table_display(section_data.get('trade_table', {}))}
        
        <h3>Trade Duration Analysis</h3>
        {create_duration_display(duration)}
        
        <h3>Trade Distribution</h3>
        {create_distribution_display(distribution)}
    </section>
    """


def generate_regime_section(section_data: Dict[str, Any]) -> str:
    """Generate market regime analysis section"""
    if not section_data:
        return ""
    
    identification = section_data.get('regime_identification', {})
    performance = section_data.get('regime_performance', {})
    transitions = section_data.get('regime_transitions', {})
    adaptive = section_data.get('adaptive_behavior', {})
    correlations = section_data.get('correlation_analysis', {})
    
    # Current regime display
    current_regime = identification.get('current_regime', 'Unknown')
    regime_class = 'info'
    if 'Bull' in current_regime:
        regime_class = 'success'
    elif 'Bear' in current_regime:
        regime_class = 'danger'
    elif 'High Volatility' in current_regime:
        regime_class = 'warning'
    
    # Regime distribution
    regime_dist = identification.get('regime_distribution', {})
    regime_cards = []
    for regime, data in regime_dist.items():
        if isinstance(data, dict):
            regime_cards.append(f"""
            <div class="metric-card">
                <h4>{regime}</h4>
                <div class="value">{data.get('percentage', 'N/A')}</div>
                <p>{data.get('periods', 0)} periods</p>
            </div>
            """)
    
    # Regime performance comparison
    perf_comparison = []
    for regime, metrics in performance.items():
        if isinstance(metrics, dict):
            perf_comparison.append([
                regime.replace('_', ' ').title(),
                metrics.get('returns', 'N/A'),
                metrics.get('sharpe', 'N/A'),
                metrics.get('win_rate', 'N/A')
            ])
    
    perf_table = create_table(
        ['Regime', 'Annual Return', 'Sharpe Ratio', 'Win Rate'],
        perf_comparison
    )
    
    return f"""
    <section id="marketregimeanalysis">
        <h2>Market Regime Analysis</h2>
        
        <div class="alert alert-{regime_class}">
            <strong>Current Market Regime:</strong> {current_regime}
        </div>
        
        <h3>Regime Distribution</h3>
        <div class="metric-grid">
            {''.join(regime_cards)}
        </div>
        
        <h3>Performance by Regime</h3>
        {perf_table}
        
        <h3>Correlation Analysis</h3>
        {create_correlation_display(correlations)}
        
        <h3>Adaptive Behavior</h3>
        <p>{adaptive.get('performance_consistency', 'Analysis pending')}</p>
    </section>
    """


def generate_technical_section(section_data: Dict[str, Any]) -> str:
    """Generate technical details section"""
    if not section_data:
        return ""
    
    config = section_data.get('strategy_configuration', {})
    execution = section_data.get('execution_statistics', {})
    performance = section_data.get('computational_performance', {})
    data_quality = section_data.get('data_quality', {})
    assumptions = section_data.get('backtest_assumptions', {})
    notes = section_data.get('implementation_notes', [])
    
    # Strategy parameters table
    params = config.get('parameters', {})
    param_rows = []
    for param, info in params.items():
        if isinstance(info, dict):
            param_rows.append([
                param,
                info.get('value', ''),
                info.get('type', ''),
                info.get('description', '')
            ])
    
    param_table = create_table(
        ['Parameter', 'Value', 'Type', 'Description'],
        param_rows
    )
    
    # Execution statistics
    exec_metrics = []
    if 'order_execution' in execution:
        for metric, value in execution['order_execution'].items():
            exec_metrics.append(f"""
            <div class="metric-card">
                <h4>{metric.replace('_', ' ').title()}</h4>
                <div class="value">{value}</div>
            </div>
            """)
    
    # Implementation notes
    notes_html = ''.join([f'<li>{note}</li>' for note in notes])
    
    return f"""
    <section id="technicaldetails">
        <h2>Technical Details</h2>
        
        <h3>Strategy Configuration</h3>
        {param_table}
        
        <h3>Execution Statistics</h3>
        <div class="metric-grid">
            {''.join(exec_metrics)}
        </div>
        
        <h3>Data Quality</h3>
        {create_data_quality_display(data_quality)}
        
        <h3>Backtest Assumptions</h3>
        {create_assumptions_display(assumptions)}
        
        <h3>Implementation Notes</h3>
        <ul>
            {notes_html}
        </ul>
    </section>
    """


def generate_footer(metadata: Dict[str, Any]) -> str:
    """Generate report footer"""
    return f"""
    <footer>
        <p>Report generated by {metadata.get('generator_version', 'Backtest Suite')} 
           on {metadata.get('generated_at', datetime.now().isoformat())}</p>
        <p>&copy; 2024 Backtest Suite. All rights reserved.</p>
    </footer>
    """


def generate_interactive_scripts() -> str:
    """Generate interactive JavaScript"""
    return """
    <script>
        // Smooth scrolling for navigation
        document.querySelectorAll('nav a').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            });
        });
        
        // Active navigation highlighting
        const sections = document.querySelectorAll('section');
        const navLinks = document.querySelectorAll('nav a');
        
        window.addEventListener('scroll', () => {
            let current = '';
            sections.forEach(section => {
                const sectionTop = section.offsetTop;
                const sectionHeight = section.clientHeight;
                if (pageYOffset >= sectionTop - 200) {
                    current = section.getAttribute('id');
                }
            });
            
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href').slice(1) === current) {
                    link.classList.add('active');
                }
            });
        });
        
        // Tab functionality
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', function() {
                const tabContainer = this.closest('.tab-container');
                const tabId = this.dataset.tab;
                
                // Update active button
                tabContainer.querySelectorAll('.tab-button').forEach(btn => {
                    btn.classList.remove('active');
                });
                this.classList.add('active');
                
                // Update active content
                tabContainer.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                tabContainer.querySelector(`#${tabId}`).classList.add('active');
            });
        });
        
        // Make Plotly charts responsive
        window.addEventListener('resize', () => {
            const plots = document.querySelectorAll('[id$="-chart"]');
            plots.forEach(plot => {
                Plotly.Plots.resize(plot);
            });
        });
    </script>
    """


# Helper functions

def create_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Create an HTML table"""
    header_html = ''.join([f'<th>{h}</th>' for h in headers])
    rows_html = ''.join([
        '<tr>' + ''.join([f'<td>{cell}</td>' for cell in row]) + '</tr>'
        for row in rows
    ])
    
    return f"""
    <table>
        <thead>
            <tr>{header_html}</tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """


def create_tabs(tab_names: List[str], tab_contents: List[str]) -> str:
    """Create tabbed content"""
    tab_buttons = []
    content_divs = []
    
    for i, (name, content) in enumerate(zip(tab_names, tab_contents)):
        active = 'active' if i == 0 else ''
        tab_id = f"tab-{name.replace(' ', '-').lower()}"
        
        tab_buttons.append(
            f'<button class="tab-button {active}" data-tab="{tab_id}">{name}</button>'
        )
        content_divs.append(
            f'<div id="{tab_id}" class="tab-content {active}">{content}</div>'
        )
    
    return f"""
    <div class="tab-container">
        <div class="tab-buttons">
            {''.join(tab_buttons)}
        </div>
        {''.join(content_divs)}
    </div>
    """


def create_significance_display(significance: Dict[str, Any]) -> str:
    """Create statistical significance display"""
    if not significance:
        return "<p>No statistical analysis available</p>"
    
    mean_test = significance.get('mean_return_test', {})
    norm_test = significance.get('normality_test', {})
    
    sig_class = 'success' if mean_test.get('significant', False) else 'warning'
    
    return f"""
    <div class="alert alert-{sig_class}">
        <h4>Statistical Tests</h4>
        <p><strong>Mean Return Test:</strong> {mean_test.get('interpretation', 'N/A')}</p>
        <p>T-statistic: {mean_test.get('t_statistic', 'N/A')}, 
           P-value: {mean_test.get('p_value', 'N/A')}</p>
        <p><strong>Normality Test:</strong> {norm_test.get('interpretation', 'N/A')}</p>
    </div>
    """


def create_duration_display(duration: Dict[str, Any]) -> str:
    """Create trade duration display"""
    if not duration or 'message' in duration:
        return f"<p>{duration.get('message', 'No duration data available')}</p>"
    
    stats = duration.get('overall_statistics', {})
    buckets = duration.get('duration_buckets', {})
    optimal = duration.get('optimal_holding_period', {})
    
    # Duration statistics
    stats_html = create_table(
        ['Metric', 'Value'],
        [[k.replace('_', ' ').title(), v] for k, v in stats.items()]
    )
    
    # Duration buckets
    bucket_rows = []
    for bucket, data in buckets.items():
        if isinstance(data, dict):
            bucket_rows.append([
                bucket,
                data.get('count', 0),
                data.get('win_rate', 'N/A'),
                data.get('avg_pnl', 'N/A')
            ])
    
    buckets_table = create_table(
        ['Duration Range', 'Count', 'Win Rate', 'Avg P&L'],
        bucket_rows
    )
    
    # Optimal holding period
    optimal_html = ""
    if isinstance(optimal, dict) and 'optimal_duration' in optimal:
        optimal_html = f"""
        <div class="alert alert-info">
            <strong>Optimal Holding Period:</strong> {optimal.get('optimal_duration', 'N/A')}<br>
            Expected Win Rate: {optimal.get('expected_win_rate', 'N/A')}<br>
            Expected Return: {optimal.get('expected_return', 'N/A')}
        </div>
        """
    
    return f"""
    {stats_html}
    <h4>Performance by Duration</h4>
    {buckets_table}
    {optimal_html}
    """


def create_distribution_display(distribution: Dict[str, Any]) -> str:
    """Create trade distribution display"""
    if not distribution:
        return "<p>No distribution data available</p>"
    
    pnl_dist = distribution.get('pnl_distribution', {})
    
    # PnL distribution metrics
    dist_metrics = []
    for metric, value in pnl_dist.items():
        if metric != 'percentiles':
            dist_metrics.append(f"""
            <div class="metric-card">
                <h4>{metric.title()}</h4>
                <div class="value">{value}</div>
            </div>
            """)
    
    # Percentiles table
    percentiles = pnl_dist.get('percentiles', {})
    percentile_rows = [[k, v] for k, v in percentiles.items()]
    percentile_table = create_table(['Percentile', 'Value'], percentile_rows)
    
    return f"""
    <div class="metric-grid">
        {''.join(dist_metrics)}
    </div>
    <h4>P&L Percentiles</h4>
    {percentile_table}
    """


def create_correlation_display(correlations: Dict[str, Any]) -> str:
    """Create correlation analysis display"""
    if not correlations or 'message' in correlations:
        return f"<p>{correlations.get('message', 'No correlation data available')}</p>"
    
    overall = correlations.get('overall_correlation', 'N/A')
    rolling = correlations.get('rolling_correlation', {})
    
    return f"""
    <div class="metric-grid">
        <div class="metric-card">
            <h4>Overall Correlation</h4>
            <div class="value">{overall}</div>
        </div>
        <div class="metric-card">
            <h4>Current Correlation</h4>
            <div class="value">{rolling.get('current', 'N/A')}</div>
        </div>
        <div class="metric-card">
            <h4>Correlation Range</h4>
            <div class="value">{rolling.get('min', 'N/A')} to {rolling.get('max', 'N/A')}</div>
        </div>
    </div>
    """


def create_data_quality_display(data_quality: Dict[str, Any]) -> str:
    """Create data quality display"""
    if not data_quality:
        return "<p>No data quality information available</p>"
    
    coverage = data_quality.get('data_coverage', {})
    integrity = data_quality.get('data_integrity', {})
    
    quality_class = 'success'
    if float(str(coverage.get('missing_data', '0%')).replace('%', '')) > 1:
        quality_class = 'warning'
    if integrity.get('suspicious_values', 0) > 0:
        quality_class = 'danger'
    
    return f"""
    <div class="alert alert-{quality_class}">
        <h4>Data Coverage</h4>
        <p>Period: {coverage.get('start_date', 'N/A')} to {coverage.get('end_date', 'N/A')}</p>
        <p>Total Bars: {coverage.get('total_bars', 'N/A')}</p>
        <p>Missing Data: {coverage.get('missing_data', 'N/A')}</p>
        
        <h4>Data Integrity</h4>
        <p>Outliers: {integrity.get('outliers_detected', 0)}</p>
        <p>Data Gaps: {integrity.get('data_gaps', 0)}</p>
        <p>Suspicious Values: {integrity.get('suspicious_values', 0)}</p>
    </div>
    """


def create_assumptions_display(assumptions: Dict[str, Any]) -> str:
    """Create assumptions display"""
    if not assumptions:
        return "<p>No assumptions documented</p>"
    
    assumption_sections = []
    for category, items in assumptions.items():
        if isinstance(items, dict):
            items_html = ''.join([
                f'<li><strong>{k.replace("_", " ").title()}:</strong> {v}</li>'
                for k, v in items.items()
            ])
            assumption_sections.append(f"""
            <h4>{category.replace('_', ' ').title()}</h4>
            <ul>{items_html}</ul>
            """)
    
    return ''.join(assumption_sections)


def create_price_analysis_display(price_analysis: Dict[str, Any]) -> str:
    """Create price analysis display"""
    if not price_analysis:
        return "<p>No price analysis data available</p>"
    
    display_html = ""
    
    # Entry Price Analysis
    if 'entry_analysis' in price_analysis:
        entry_data = price_analysis['entry_analysis']
        display_html += f"""
        <h4>Entry Price Analysis</h4>
        <div class="metric-grid">
            <div class="metric-card">
                <h4>Average Entry Price</h4>
                <div class="value">{entry_data.get('avg_entry_price', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Median Entry Price</h4>
                <div class="value">{entry_data.get('median_entry_price', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Entry Price Range</h4>
                <div class="value">{entry_data.get('min_entry_price', 'N/A')} - {entry_data.get('max_entry_price', 'N/A')}</div>
            </div>
        </div>
        """
    
    # Exit Price Analysis
    if 'exit_analysis' in price_analysis:
        exit_data = price_analysis['exit_analysis']
        display_html += f"""
        <h4>Exit Price Analysis</h4>
        <div class="metric-grid">
            <div class="metric-card">
                <h4>Average Exit Price</h4>
                <div class="value">{exit_data.get('avg_exit_price', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Median Exit Price</h4>
                <div class="value">{exit_data.get('median_exit_price', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Exit Price Range</h4>
                <div class="value">{exit_data.get('min_exit_price', 'N/A')} - {exit_data.get('max_exit_price', 'N/A')}</div>
            </div>
        </div>
        """
    
    # Price Movement Analysis
    if 'price_movement' in price_analysis:
        movement_data = price_analysis['price_movement']
        display_html += f"""
        <h4>Price Movement Analysis</h4>
        <div class="metric-grid">
            <div class="metric-card">
                <h4>Average Price Change</h4>
                <div class="value">{movement_data.get('avg_price_change', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Best Price Movement</h4>
                <div class="value">{movement_data.get('max_favorable_move', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Worst Price Movement</h4>
                <div class="value">{movement_data.get('max_adverse_move', 'N/A')}</div>
            </div>
        </div>
        """
    
    # Stop Loss Analysis
    if 'stop_loss_analysis' in price_analysis:
        sl_data = price_analysis['stop_loss_analysis']
        display_html += f"""
        <h4>Stop Loss Analysis</h4>
        <div class="metric-grid">
            <div class="metric-card">
                <h4>Average Stop Loss Price</h4>
                <div class="value">{sl_data.get('avg_stop_loss_price', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Stop Loss Usage</h4>
                <div class="value">{sl_data.get('stop_loss_usage', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Average SL Distance</h4>
                <div class="value">{sl_data.get('avg_sl_distance', 'N/A')}</div>
            </div>
        </div>
        """
    
    # Take Profit Analysis
    if 'take_profit_analysis' in price_analysis:
        tp_data = price_analysis['take_profit_analysis']
        display_html += f"""
        <h4>Take Profit Analysis</h4>
        <div class="metric-grid">
            <div class="metric-card">
                <h4>Average Take Profit Price</h4>
                <div class="value">{tp_data.get('avg_take_profit_price', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Take Profit Usage</h4>
                <div class="value">{tp_data.get('take_profit_usage', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Average TP Distance</h4>
                <div class="value">{tp_data.get('avg_tp_distance', 'N/A')}</div>
            </div>
        </div>
        """
    
    return display_html


def create_risk_analysis_display(risk_analysis: Dict[str, Any]) -> str:
    """Create risk analysis display"""
    if not risk_analysis:
        return "<p>No risk analysis data available</p>"
    
    display_html = ""
    
    # Risk per Trade Analysis
    if 'risk_per_trade' in risk_analysis:
        risk_data = risk_analysis['risk_per_trade']
        display_html += f"""
        <h4>Risk per Trade Analysis</h4>
        <div class="metric-grid">
            <div class="metric-card">
                <h4>Average Risk per Trade</h4>
                <div class="value">{risk_data.get('avg_risk_per_trade', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Maximum Risk per Trade</h4>
                <div class="value">{risk_data.get('max_risk_per_trade', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Risk Consistency</h4>
                <div class="value">{risk_data.get('risk_consistency', 'N/A')}</div>
            </div>
        </div>
        """
    
    # Risk-Reward Ratio Analysis
    if 'risk_reward_ratio' in risk_analysis:
        rr_data = risk_analysis['risk_reward_ratio']
        display_html += f"""
        <h4>Risk-Reward Ratio Analysis</h4>
        <div class="metric-grid">
            <div class="metric-card">
                <h4>Average Risk-Reward Ratio</h4>
                <div class="value">{rr_data.get('avg_risk_reward_ratio', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Best Risk-Reward Ratio</h4>
                <div class="value">{rr_data.get('best_risk_reward_ratio', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Worst Risk-Reward Ratio</h4>
                <div class="value">{rr_data.get('worst_risk_reward_ratio', 'N/A')}</div>
            </div>
        </div>
        """
    
    # MAE Analysis
    if 'mae_analysis' in risk_analysis:
        mae_data = risk_analysis['mae_analysis']
        display_html += f"""
        <h4>Maximum Adverse Excursion (MAE) Analysis</h4>
        <div class="metric-grid">
            <div class="metric-card">
                <h4>Average MAE</h4>
                <div class="value">{mae_data.get('avg_mae', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Maximum MAE</h4>
                <div class="value">{mae_data.get('max_mae', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>MAE Volatility</h4>
                <div class="value">{mae_data.get('mae_std', 'N/A')}</div>
            </div>
        </div>
        """
    
    # MFE Analysis
    if 'mfe_analysis' in risk_analysis:
        mfe_data = risk_analysis['mfe_analysis']
        display_html += f"""
        <h4>Maximum Favorable Excursion (MFE) Analysis</h4>
        <div class="metric-grid">
            <div class="metric-card">
                <h4>Average MFE</h4>
                <div class="value">{mfe_data.get('avg_mfe', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>Maximum MFE</h4>
                <div class="value">{mfe_data.get('max_mfe', 'N/A')}</div>
            </div>
            <div class="metric-card">
                <h4>MFE Volatility</h4>
                <div class="value">{mfe_data.get('mfe_std', 'N/A')}</div>
            </div>
        </div>
        """
    
    return display_html


def create_enhanced_trade_table_display(trade_table: Dict[str, Any]) -> str:
    """Create enhanced trade table display"""
    if not trade_table or 'message' in trade_table:
        return f"<p>{trade_table.get('message', 'No trade table data available')}</p>"
    
    columns = trade_table.get('columns', [])
    data = trade_table.get('data', [])
    summary = trade_table.get('summary', {})
    
    if not columns or not data:
        return "<p>No trade data to display</p>"
    
    # Create table header
    header_html = ''.join([f'<th>{col}</th>' for col in columns])
    
    # Create table rows
    rows_html = []
    for row in data:
        row_cells = ''.join([f'<td>{row.get(col, "N/A")}</td>' for col in columns])
        rows_html.append(f'<tr>{row_cells}</tr>')
    
    # Create summary section
    summary_html = ""
    if summary:
        summary_html = f"""
        <div class="alert alert-info">
            <h4>Trade Summary</h4>
            <p><strong>Total Trades:</strong> {trade_table.get('total_trades', 'N/A')}</p>
            <p><strong>Total P&L:</strong> {summary.get('total_pnl', 'N/A')}</p>
            <p><strong>Average P&L:</strong> {summary.get('avg_pnl', 'N/A')}</p>
            <p><strong>Win Rate:</strong> {summary.get('win_rate', 'N/A')}</p>
            <p><strong>Best Trade:</strong> {summary.get('best_trade', 'N/A')}</p>
            <p><strong>Worst Trade:</strong> {summary.get('worst_trade', 'N/A')}</p>
        </div>
        """
    
    return f"""
    {summary_html}
    <div style="overflow-x: auto; margin-top: 1rem;">
        <table style="min-width: 100%;">
            <thead>
                <tr>{header_html}</tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """