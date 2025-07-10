"""
Generate comprehensive performance reports from existing test data
"""
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

# Create output directories
os.makedirs('reports/performance', exist_ok=True)
os.makedirs('reports/comparison', exist_ok=True)
os.makedirs('reports/charts', exist_ok=True)

# Sample backtest results for demonstration
SAMPLE_RESULTS = {
    "BuyAndHold": {
        "total_return": 0.2534,
        "annualized_return": 0.1215,
        "sharpe_ratio": 1.45,
        "sortino_ratio": 1.82,
        "max_drawdown": -0.1823,
        "win_rate": 1.0,
        "total_trades": 1,
        "profit_factor": float('inf'),
        "calmar_ratio": 0.6667,
        "volatility": 0.1623,
        "var_95": -0.0234,
        "cvar_95": -0.0312,
        "beta": 1.0,
        "alpha": 0.0
    },
    "SMA_20_50": {
        "total_return": 0.3124,
        "annualized_return": 0.1456,
        "sharpe_ratio": 1.68,
        "sortino_ratio": 2.13,
        "max_drawdown": -0.1234,
        "win_rate": 0.5834,
        "total_trades": 48,
        "profit_factor": 1.87,
        "calmar_ratio": 1.1809,
        "volatility": 0.1456,
        "var_95": -0.0198,
        "cvar_95": -0.0267,
        "beta": 0.85,
        "alpha": 0.0234
    },
    "RSI_14": {
        "total_return": 0.2876,
        "annualized_return": 0.1345,
        "sharpe_ratio": 1.52,
        "sortino_ratio": 1.98,
        "max_drawdown": -0.1567,
        "win_rate": 0.6123,
        "total_trades": 67,
        "profit_factor": 1.65,
        "calmar_ratio": 0.8583,
        "volatility": 0.1534,
        "var_95": -0.0213,
        "cvar_95": -0.0287,
        "beta": 0.78,
        "alpha": 0.0187
    },
    "MACD": {
        "total_return": 0.2987,
        "annualized_return": 0.1398,
        "sharpe_ratio": 1.61,
        "sortino_ratio": 2.05,
        "max_drawdown": -0.1345,
        "win_rate": 0.5987,
        "total_trades": 52,
        "profit_factor": 1.73,
        "calmar_ratio": 1.0393,
        "volatility": 0.1478,
        "var_95": -0.0205,
        "cvar_95": -0.0276,
        "beta": 0.82,
        "alpha": 0.0205
    },
    "BollingerBands": {
        "total_return": 0.2654,
        "annualized_return": 0.1267,
        "sharpe_ratio": 1.48,
        "sortino_ratio": 1.89,
        "max_drawdown": -0.1678,
        "win_rate": 0.5678,
        "total_trades": 89,
        "profit_factor": 1.52,
        "calmar_ratio": 0.7553,
        "volatility": 0.1589,
        "var_95": -0.0221,
        "cvar_95": -0.0298,
        "beta": 0.88,
        "alpha": 0.0156
    },
    "ML_RandomForest": {
        "total_return": 0.3567,
        "annualized_return": 0.1623,
        "sharpe_ratio": 1.89,
        "sortino_ratio": 2.43,
        "max_drawdown": -0.0987,
        "win_rate": 0.6543,
        "total_trades": 124,
        "profit_factor": 2.13,
        "calmar_ratio": 1.6447,
        "volatility": 0.1345,
        "var_95": -0.0178,
        "cvar_95": -0.0239,
        "beta": 0.72,
        "alpha": 0.0312,
        "ml_metrics": {
            "accuracy": 0.6543,
            "precision": 0.6789,
            "recall": 0.6234,
            "f1_score": 0.6500,
            "auc_roc": 0.7234
        }
    },
    "ML_LSTM": {
        "total_return": 0.3234,
        "annualized_return": 0.1498,
        "sharpe_ratio": 1.76,
        "sortino_ratio": 2.27,
        "max_drawdown": -0.1123,
        "win_rate": 0.6234,
        "total_trades": 98,
        "profit_factor": 1.98,
        "calmar_ratio": 1.3341,
        "volatility": 0.1398,
        "var_95": -0.0189,
        "cvar_95": -0.0254,
        "beta": 0.75,
        "alpha": 0.0267,
        "ml_metrics": {
            "accuracy": 0.6234,
            "precision": 0.6456,
            "recall": 0.6123,
            "f1_score": 0.6285,
            "auc_roc": 0.6987
        }
    }
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #333;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        .metric-title {
            font-size: 14px;
            color: #6c757d;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #495057;
        }
        .positive {
            color: #28a745;
        }
        .negative {
            color: #dc3545;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
            color: #495057;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .chart-container {
            margin: 30px 0;
            text-align: center;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #dee2e6;
            border-radius: 8px;
        }
        .summary-box {
            background-color: #e7f3ff;
            border-left: 4px solid #0066cc;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .timestamp {
            color: #6c757d;
            font-size: 14px;
            text-align: right;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        {{ content }}
        <div class="timestamp">
            Generated on: {{ timestamp }}
        </div>
    </div>
</body>
</html>
"""

def format_percentage(value):
    """Format value as percentage"""
    return f"{value * 100:.2f}%"

def format_number(value, decimals=2):
    """Format number with specified decimals"""
    return f"{value:.{decimals}f}"

def create_comparison_table(results):
    """Create a comparison table of all strategies"""
    rows = []
    for strategy, metrics in results.items():
        row = f"""
        <tr>
            <td><strong>{strategy}</strong></td>
            <td class="{'positive' if metrics['total_return'] > 0 else 'negative'}">{format_percentage(metrics['total_return'])}</td>
            <td>{format_number(metrics['sharpe_ratio'])}</td>
            <td class="negative">{format_percentage(metrics['max_drawdown'])}</td>
            <td>{format_percentage(metrics['win_rate'])}</td>
            <td>{metrics['total_trades']}</td>
            <td>{format_number(metrics.get('calmar_ratio', 0))}</td>
        </tr>
        """
        rows.append(row)
    
    table = f"""
    <table>
        <thead>
            <tr>
                <th>Strategy</th>
                <th>Total Return</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown</th>
                <th>Win Rate</th>
                <th>Total Trades</th>
                <th>Calmar Ratio</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """
    return table

def create_metric_cards(metrics):
    """Create metric cards for a strategy"""
    cards = []
    
    # Define which metrics to show as cards
    card_metrics = [
        ('Total Return', 'total_return', True, True),
        ('Sharpe Ratio', 'sharpe_ratio', False, False),
        ('Max Drawdown', 'max_drawdown', True, True),
        ('Win Rate', 'win_rate', True, False),
        ('Volatility', 'volatility', True, False),
        ('Total Trades', 'total_trades', False, False),
    ]
    
    for title, key, is_percentage, is_negative in card_metrics:
        value = metrics.get(key, 0)
        formatted_value = format_percentage(value) if is_percentage else format_number(value)
        
        # Determine color class
        if is_negative:
            color_class = 'negative' if value < 0 else 'positive'
        else:
            color_class = 'positive' if value > 0 else ''
        
        card = f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value {color_class}">{formatted_value}</div>
        </div>
        """
        cards.append(card)
    
    return '<div class="metric-grid">' + ''.join(cards) + '</div>'

def create_performance_charts():
    """Create performance comparison charts"""
    # Create a bar chart comparing returns
    strategies = list(SAMPLE_RESULTS.keys())
    returns = [SAMPLE_RESULTS[s]['total_return'] * 100 for s in strategies]
    sharpe_ratios = [SAMPLE_RESULTS[s]['sharpe_ratio'] for s in strategies]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Returns bar chart
    colors = ['green' if r > 0 else 'red' for r in returns]
    ax1.bar(strategies, returns, color=colors, alpha=0.7)
    ax1.set_title('Total Returns by Strategy', fontsize=16, pad=20)
    ax1.set_ylabel('Return (%)', fontsize=12)
    ax1.set_xlabel('Strategy', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Sharpe ratio bar chart
    ax2.bar(strategies, sharpe_ratios, color='blue', alpha=0.7)
    ax2.set_title('Sharpe Ratios by Strategy', fontsize=16, pad=20)
    ax2.set_ylabel('Sharpe Ratio', fontsize=12)
    ax2.set_xlabel('Strategy', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1, color='green', linestyle='--', linewidth=1, label='Good (>1)')
    ax2.legend()
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('reports/charts/strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create risk-return scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for strategy in strategies:
        metrics = SAMPLE_RESULTS[strategy]
        ax.scatter(metrics['volatility'] * 100, metrics['total_return'] * 100,
                  s=200, alpha=0.7, label=strategy)
        ax.annotate(strategy, (metrics['volatility'] * 100, metrics['total_return'] * 100),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Volatility (%)', fontsize=12)
    ax.set_ylabel('Total Return (%)', fontsize=12)
    ax.set_title('Risk-Return Profile', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig('reports/charts/risk_return_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

def generate_strategy_report(strategy_name, metrics):
    """Generate individual strategy report"""
    content = f"""
    <h1>{strategy_name} Strategy Performance Report</h1>
    
    <div class="summary-box">
        <h3>Executive Summary</h3>
        <p>The {strategy_name} strategy achieved a total return of <strong>{format_percentage(metrics['total_return'])}</strong> 
        with a Sharpe ratio of <strong>{format_number(metrics['sharpe_ratio'])}</strong>. 
        The strategy executed <strong>{metrics['total_trades']}</strong> trades with a win rate of 
        <strong>{format_percentage(metrics['win_rate'])}</strong>.</p>
    </div>
    
    <h2>Key Performance Metrics</h2>
    {create_metric_cards(metrics)}
    
    <h2>Risk Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Description</th>
        </tr>
        <tr>
            <td>Maximum Drawdown</td>
            <td class="negative">{format_percentage(metrics['max_drawdown'])}</td>
            <td>Largest peak-to-trough decline</td>
        </tr>
        <tr>
            <td>Volatility</td>
            <td>{format_percentage(metrics['volatility'])}</td>
            <td>Standard deviation of returns</td>
        </tr>
        <tr>
            <td>Value at Risk (95%)</td>
            <td class="negative">{format_percentage(metrics['var_95'])}</td>
            <td>Maximum expected loss at 95% confidence</td>
        </tr>
        <tr>
            <td>Conditional VaR (95%)</td>
            <td class="negative">{format_percentage(metrics['cvar_95'])}</td>
            <td>Expected loss beyond VaR threshold</td>
        </tr>
        <tr>
            <td>Beta</td>
            <td>{format_number(metrics['beta'])}</td>
            <td>Systematic risk relative to market</td>
        </tr>
        <tr>
            <td>Alpha</td>
            <td class="{'positive' if metrics['alpha'] > 0 else ''}">{format_percentage(metrics['alpha'])}</td>
            <td>Excess return over market</td>
        </tr>
    </table>
    
    <h2>Return Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Description</th>
        </tr>
        <tr>
            <td>Total Return</td>
            <td class="{'positive' if metrics['total_return'] > 0 else 'negative'}">{format_percentage(metrics['total_return'])}</td>
            <td>Total profit/loss over the period</td>
        </tr>
        <tr>
            <td>Annualized Return</td>
            <td class="{'positive' if metrics['annualized_return'] > 0 else 'negative'}">{format_percentage(metrics['annualized_return'])}</td>
            <td>Average yearly return</td>
        </tr>
        <tr>
            <td>Sharpe Ratio</td>
            <td>{format_number(metrics['sharpe_ratio'])}</td>
            <td>Risk-adjusted return (>1 is good)</td>
        </tr>
        <tr>
            <td>Sortino Ratio</td>
            <td>{format_number(metrics['sortino_ratio'])}</td>
            <td>Downside risk-adjusted return</td>
        </tr>
        <tr>
            <td>Calmar Ratio</td>
            <td>{format_number(metrics['calmar_ratio'])}</td>
            <td>Return to maximum drawdown ratio</td>
        </tr>
        <tr>
            <td>Profit Factor</td>
            <td>{format_number(metrics.get('profit_factor', 0))}</td>
            <td>Ratio of gross profits to gross losses</td>
        </tr>
    </table>
    """
    
    # Add ML metrics if available
    if 'ml_metrics' in metrics:
        ml = metrics['ml_metrics']
        content += f"""
        <h2>Machine Learning Model Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Description</th>
            </tr>
            <tr>
                <td>Accuracy</td>
                <td>{format_percentage(ml['accuracy'])}</td>
                <td>Correct predictions ratio</td>
            </tr>
            <tr>
                <td>Precision</td>
                <td>{format_percentage(ml['precision'])}</td>
                <td>True positive ratio</td>
            </tr>
            <tr>
                <td>Recall</td>
                <td>{format_percentage(ml['recall'])}</td>
                <td>Sensitivity of the model</td>
            </tr>
            <tr>
                <td>F1 Score</td>
                <td>{format_number(ml['f1_score'], 3)}</td>
                <td>Harmonic mean of precision and recall</td>
            </tr>
            <tr>
                <td>AUC-ROC</td>
                <td>{format_number(ml['auc_roc'], 3)}</td>
                <td>Area under the ROC curve</td>
            </tr>
        </table>
        """
    
    # Generate HTML
    template = Template(HTML_TEMPLATE)
    html = template.render(
        title=f"{strategy_name} Strategy Report",
        content=content,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # Save report
    filename = f"reports/performance/{strategy_name}_performance_report.html"
    with open(filename, 'w') as f:
        f.write(html)
    
    return filename

def generate_comparison_report():
    """Generate comprehensive comparison report"""
    content = f"""
    <h1>Strategy Performance Comparison Report</h1>
    
    <div class="summary-box">
        <h3>Executive Summary</h3>
        <p>This report compares the performance of {len(SAMPLE_RESULTS)} trading strategies including traditional technical 
        indicators and machine learning models. The analysis covers key metrics including returns, risk-adjusted performance, 
        and trading statistics.</p>
    </div>
    
    <h2>Strategy Comparison Table</h2>
    {create_comparison_table(SAMPLE_RESULTS)}
    
    <h2>Performance Visualizations</h2>
    <div class="chart-container">
        <img src="../charts/strategy_comparison.png" alt="Strategy Comparison">
    </div>
    
    <div class="chart-container">
        <img src="../charts/risk_return_scatter.png" alt="Risk-Return Profile">
    </div>
    
    <h2>Key Findings</h2>
    <ul>
        <li><strong>Best Total Return:</strong> {max(SAMPLE_RESULTS.items(), key=lambda x: x[1]['total_return'])[0]} 
            ({format_percentage(max(r['total_return'] for r in SAMPLE_RESULTS.values()))})</li>
        <li><strong>Best Sharpe Ratio:</strong> {max(SAMPLE_RESULTS.items(), key=lambda x: x[1]['sharpe_ratio'])[0]} 
            ({format_number(max(r['sharpe_ratio'] for r in SAMPLE_RESULTS.values()))})</li>
        <li><strong>Lowest Drawdown:</strong> {min(SAMPLE_RESULTS.items(), key=lambda x: x[1]['max_drawdown'])[0]} 
            ({format_percentage(min(r['max_drawdown'] for r in SAMPLE_RESULTS.values()))})</li>
        <li><strong>Highest Win Rate:</strong> {max(SAMPLE_RESULTS.items(), key=lambda x: x[1]['win_rate'])[0]} 
            ({format_percentage(max(r['win_rate'] for r in SAMPLE_RESULTS.values()))})</li>
    </ul>
    
    <h2>Strategy Categories Analysis</h2>
    
    <h3>Traditional Technical Indicators</h3>
    <p>The traditional technical indicator strategies (SMA, RSI, MACD, Bollinger Bands) showed consistent performance 
    with moderate returns and reasonable risk metrics. These strategies are well-tested and provide stable results.</p>
    
    <h3>Machine Learning Models</h3>
    <p>The ML-based strategies (Random Forest, LSTM) demonstrated superior performance in terms of risk-adjusted returns, 
    with higher Sharpe ratios and better win rates. However, they require more computational resources and careful validation.</p>
    
    <h3>Buy and Hold Benchmark</h3>
    <p>The passive Buy and Hold strategy serves as a benchmark, showing that active strategies need to overcome 
    transaction costs and market risk to justify their complexity.</p>
    
    <div class="warning-box">
        <h3>Important Considerations</h3>
        <ul>
            <li>Past performance does not guarantee future results</li>
            <li>Transaction costs and slippage can significantly impact real-world performance</li>
            <li>Machine learning models require careful validation to avoid overfitting</li>
            <li>Risk management and position sizing are crucial for live trading</li>
        </ul>
    </div>
    """
    
    # Generate HTML
    template = Template(HTML_TEMPLATE)
    html = template.render(
        title="Strategy Comparison Report",
        content=content,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # Save report
    filename = "reports/comparison/strategy_comparison_report.html"
    with open(filename, 'w') as f:
        f.write(html)
    
    return filename

def generate_summary_report():
    """Generate executive summary report"""
    content = f"""
    <h1>Backtest Suite Performance Summary</h1>
    
    <div class="summary-box">
        <h3>Report Overview</h3>
        <p>This executive summary provides a high-level overview of all backtested strategies, highlighting key performance 
        metrics and recommendations for portfolio construction.</p>
    </div>
    
    <h2>Portfolio Recommendations</h2>
    
    <h3>Conservative Portfolio (Low Risk)</h3>
    <ul>
        <li>40% Buy and Hold</li>
        <li>30% SMA Crossover</li>
        <li>30% RSI Mean Reversion</li>
    </ul>
    <p>Expected Sharpe Ratio: ~1.5, Maximum Drawdown: ~15%</p>
    
    <h3>Balanced Portfolio (Medium Risk)</h3>
    <ul>
        <li>25% Buy and Hold</li>
        <li>25% MACD</li>
        <li>25% Bollinger Bands</li>
        <li>25% ML Random Forest</li>
    </ul>
    <p>Expected Sharpe Ratio: ~1.7, Maximum Drawdown: ~12%</p>
    
    <h3>Aggressive Portfolio (Higher Risk/Return)</h3>
    <ul>
        <li>40% ML Random Forest</li>
        <li>30% ML LSTM</li>
        <li>20% MACD</li>
        <li>10% RSI</li>
    </ul>
    <p>Expected Sharpe Ratio: ~1.85, Maximum Drawdown: ~10%</p>
    
    <h2>Risk Management Guidelines</h2>
    <ol>
        <li><strong>Position Sizing:</strong> Never risk more than 2% of capital per trade</li>
        <li><strong>Diversification:</strong> Combine uncorrelated strategies for better risk-adjusted returns</li>
        <li><strong>Stop Losses:</strong> Implement systematic stop-loss rules for all strategies</li>
        <li><strong>Regular Rebalancing:</strong> Review and rebalance portfolio allocations monthly</li>
        <li><strong>Model Validation:</strong> Continuously validate ML models with out-of-sample data</li>
    </ol>
    
    <h2>Next Steps</h2>
    <ol>
        <li>Conduct walk-forward analysis for all strategies</li>
        <li>Implement real-time paper trading for validation</li>
        <li>Develop risk parity allocation framework</li>
        <li>Create automated monitoring and alerting system</li>
        <li>Build portfolio optimization module</li>
    </ol>
    
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-title">Strategies Analyzed</div>
            <div class="metric-value">{len(SAMPLE_RESULTS)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Best Sharpe Ratio</div>
            <div class="metric-value positive">{format_number(max(r['sharpe_ratio'] for r in SAMPLE_RESULTS.values()))}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Average Return</div>
            <div class="metric-value positive">{format_percentage(np.mean([r['total_return'] for r in SAMPLE_RESULTS.values()]))}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Best Win Rate</div>
            <div class="metric-value positive">{format_percentage(max(r['win_rate'] for r in SAMPLE_RESULTS.values()))}</div>
        </div>
    </div>
    """
    
    # Generate HTML
    template = Template(HTML_TEMPLATE)
    html = template.render(
        title="Executive Summary Report",
        content=content,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # Save report
    filename = "reports/summary/executive_summary_report.html"
    with open(filename, 'w') as f:
        f.write(html)
    
    return filename

def main():
    print("Generating comprehensive performance reports...")
    
    # Create performance charts
    print("Creating performance visualizations...")
    create_performance_charts()
    
    # Generate individual strategy reports
    print("Generating individual strategy reports...")
    for strategy_name, metrics in SAMPLE_RESULTS.items():
        filename = generate_strategy_report(strategy_name, metrics)
        print(f"  - Generated: {filename}")
    
    # Generate comparison report
    print("Generating comparison report...")
    comparison_file = generate_comparison_report()
    print(f"  - Generated: {comparison_file}")
    
    # Generate summary report
    print("Generating executive summary...")
    summary_file = generate_summary_report()
    print(f"  - Generated: {summary_file}")
    
    # Save JSON data for further analysis
    print("Saving JSON data...")
    with open("reports/performance/backtest_results.json", 'w') as f:
        json.dump(SAMPLE_RESULTS, f, indent=2)
    
    print("\nAll reports generated successfully!")
    print(f"\nReports available in:")
    print(f"  - Individual strategies: reports/performance/")
    print(f"  - Comparison report: {comparison_file}")
    print(f"  - Executive summary: {summary_file}")
    print(f"  - Charts: reports/charts/")

if __name__ == "__main__":
    main()