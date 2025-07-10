# ML Reports Guide

This guide explains how to use the ML Report Generator to create comprehensive HTML reports for machine learning results.

## Overview

The ML Report Generator creates interactive HTML reports with visualizations using Plotly for various ML analyses:

1. **Feature Analysis Report** - Feature importance, correlations, and distributions
2. **Performance Dashboard** - Model metrics, confusion matrices, ROC curves, and profit curves
3. **Optimization Results** - Parameter evolution and optimization progress
4. **Regime Analysis** - Market regime detection and transitions
5. **Strategy Comparison** - ML vs baseline strategy performance

## Installation

The report generator requires the following packages:
```bash
pip install plotly jinja2 seaborn matplotlib
```

## Basic Usage

```python
from src.ml.reports.report_generator import MLReportGenerator

# Initialize the report generator
report_gen = MLReportGenerator()

# Generate a feature analysis report
report_path = report_gen.generate_feature_analysis_report(
    feature_importance=feature_importance_df,
    correlation_matrix=correlation_df,
    feature_distributions=distributions_dict,
    metadata={'dataset': 'My Dataset', 'version': '1.0'}
)
```

## Report Types

### 1. Feature Analysis Report

Analyzes feature importance and relationships:

```python
report_path = report_gen.generate_feature_analysis_report(
    feature_importance=pd.DataFrame({
        'importance': [0.8, 0.6, 0.4, 0.2]
    }, index=['RSI', 'MACD', 'Volume', 'Price']),
    correlation_matrix=correlation_df,
    feature_distributions={
        'RSI': pd.Series(rsi_values),
        'MACD': pd.Series(macd_values)
    },
    metadata={'analysis_date': '2024-01-01'}
)
```

**Includes:**
- Feature importance bar chart
- Correlation heatmap
- Feature distribution histograms with KDE
- Statistical summaries

### 2. Performance Dashboard

Comprehensive model performance analysis:

```python
report_path = report_gen.generate_performance_dashboard(
    model_metrics={
        'RandomForest': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.78},
        'XGBoost': {'accuracy': 0.87, 'precision': 0.85, 'recall': 0.80}
    },
    confusion_matrices={
        'RandomForest': np.array([[850, 150], [220, 780]]),
        'XGBoost': np.array([[870, 130], [200, 800]])
    },
    roc_data={
        'RandomForest': {'fpr': fpr, 'tpr': tpr, 'auc': 0.88},
        'XGBoost': {'fpr': fpr, 'tpr': tpr, 'auc': 0.91}
    },
    profit_curves={
        'RandomForest': profit_df,
        'XGBoost': profit_df
    }
)
```

**Includes:**
- Model metrics comparison
- Confusion matrices
- ROC curves with AUC
- Cumulative profit curves
- Model summary cards

### 3. Optimization Results

Tracks optimization progress and parameter evolution:

```python
report_path = report_gen.generate_optimization_results_report(
    optimization_history=[
        {'iteration': 0, 'current_score': 0.5, 'best_score': 0.5},
        {'iteration': 1, 'current_score': 0.6, 'best_score': 0.6},
        # ... more iterations
    ],
    parameter_evolution=pd.DataFrame({
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [3, 5, 7]
    }),
    best_configurations=[
        {'rank': 1, 'score': 0.92, 'learning_rate': 0.01, 'max_depth': 7},
        {'rank': 2, 'score': 0.91, 'learning_rate': 0.015, 'max_depth': 6}
    ]
)
```

**Includes:**
- Optimization progress chart
- Parameter evolution plots
- Best configurations table
- Summary statistics

### 4. Regime Analysis

Market regime detection and analysis:

```python
report_path = report_gen.generate_regime_analysis_report(
    regime_data=pd.DataFrame({
        'regime': [0, 0, 1, 1, 2, 2],  # 0=Bull, 1=Bear, 2=Sideways
        'actual_regime': [0, 0, 1, 2, 2, 2]
    }, index=dates),
    transition_matrix=pd.DataFrame(
        [[0.7, 0.2, 0.1], [0.15, 0.6, 0.25], [0.2, 0.3, 0.5]],
        index=['Bull', 'Bear', 'Sideways'],
        columns=['Bull', 'Bear', 'Sideways']
    ),
    performance_by_regime={
        'Bull': {'return': 0.15, 'sharpe': 1.2},
        'Bear': {'return': -0.05, 'sharpe': -0.3},
        'Sideways': {'return': 0.03, 'sharpe': 0.5}
    },
    detection_accuracy={'accuracy': 0.78, 'precision': 0.75}
)
```

**Includes:**
- Regime timeline visualization
- Transition probability matrix
- Performance by regime
- Detection accuracy metrics
- Duration statistics

### 5. Strategy Comparison

Compare ML strategy against baseline:

```python
report_path = report_gen.generate_strategy_comparison_report(
    ml_performance=ml_performance_df,
    baseline_performance=baseline_performance_df,
    comparison_metrics={
        'ML Strategy': {'total_return': 0.45, 'sharpe_ratio': 1.2},
        'Baseline': {'total_return': 0.25, 'sharpe_ratio': 0.8}
    },
    trade_analysis={
        'ML Strategy': ml_trades_df,
        'Baseline': baseline_trades_df
    }
)
```

**Includes:**
- Cumulative returns comparison
- Performance metrics comparison
- Drawdown analysis
- Trade win rate analysis
- Improvement summary

## Customization

### Custom Templates

You can provide custom HTML templates:

```python
report_gen = MLReportGenerator(
    template_dir='/path/to/custom/templates',
    output_dir='/path/to/output/reports'
)
```

### Styling

Reports use Bootstrap 5 for styling. You can modify the CSS in the templates or add custom styles:

```python
# In your custom template
<style>
    .custom-card {
        background-color: #f0f0f0;
        border-radius: 10px;
    }
</style>
```

## Interactive Features

All reports include:
- **Interactive Plotly charts** - Zoom, pan, hover for details
- **Responsive design** - Works on desktop and mobile
- **Data tables** - Sortable and searchable
- **Export capabilities** - Save charts as PNG/SVG

## Best Practices

1. **Data Preparation**
   - Ensure data is clean and properly formatted
   - Use meaningful feature names
   - Include relevant metadata

2. **Performance**
   - For large datasets, consider sampling for distributions
   - Limit the number of features shown in distributions (top 10)
   - Pre-calculate expensive metrics

3. **Report Organization**
   - Generate reports in a structured directory
   - Use timestamps in filenames
   - Keep metadata for traceability

## Example Workflow

```python
# Complete workflow example
import pandas as pd
import numpy as np
from src.ml.reports.report_generator import MLReportGenerator

# 1. Initialize generator
report_gen = MLReportGenerator()

# 2. Prepare your ML results
# ... run your ML pipeline ...

# 3. Generate feature analysis
feature_report = report_gen.generate_feature_analysis_report(
    feature_importance=feature_importance_results,
    correlation_matrix=feature_correlations,
    feature_distributions=feature_data,
    metadata={'model': 'XGBoost', 'date': '2024-01-01'}
)

# 4. Generate performance dashboard after training
performance_report = report_gen.generate_performance_dashboard(
    model_metrics=evaluation_metrics,
    confusion_matrices=confusion_results,
    roc_data=roc_results,
    profit_curves=backtest_results
)

# 5. Generate optimization report after hyperparameter tuning
optimization_report = report_gen.generate_optimization_results_report(
    optimization_history=tuning_history,
    parameter_evolution=param_history,
    best_configurations=best_params
)

print(f"Reports generated in: {report_gen.output_dir}")
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install plotly jinja2 pandas numpy scipy
   ```

2. **Template Not Found**
   - Ensure template directory exists
   - Check file permissions

3. **Large Data Performance**
   - Sample data for visualizations
   - Use data aggregation for time series

4. **Browser Compatibility**
   - Use modern browsers for best experience
   - Chrome, Firefox, Safari, Edge supported

## Advanced Features

### Batch Report Generation

```python
# Generate multiple reports in batch
for model_name, results in all_model_results.items():
    report_path = report_gen.generate_performance_dashboard(
        model_metrics={model_name: results['metrics']},
        confusion_matrices={model_name: results['confusion']},
        roc_data={model_name: results['roc']},
        profit_curves={model_name: results['profits']},
        metadata={'model': model_name, 'timestamp': datetime.now()}
    )
```

### Custom Visualizations

You can add custom Plotly figures to reports:

```python
# Create custom figure
import plotly.graph_objects as go

custom_fig = go.Figure()
custom_fig.add_trace(go.Scatter3d(
    x=data['x'], y=data['y'], z=data['z'],
    mode='markers',
    marker=dict(size=5, color=data['color'])
))

# Convert to HTML and include in template
custom_html = custom_fig.to_html(include_plotlyjs=False)
```

### Report Automation

Set up automated report generation:

```python
# scheduled_reports.py
import schedule
import time

def generate_daily_reports():
    # Load latest results
    results = load_latest_ml_results()
    
    # Generate reports
    report_gen = MLReportGenerator()
    report_gen.generate_performance_dashboard(**results)

# Schedule daily at 6 PM
schedule.every().day.at("18:00").do(generate_daily_reports)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Conclusion

The ML Report Generator provides a comprehensive solution for visualizing and analyzing machine learning results. The interactive HTML reports make it easy to share insights with stakeholders and track model performance over time.