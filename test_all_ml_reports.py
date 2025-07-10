"""
Comprehensive test for all ML Report types
"""

import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

# Import report generator directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("report_generator", "src/ml/reports/report_generator.py")
report_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_module)
MLReportGenerator = report_module.MLReportGenerator


def generate_all_reports():
    """Generate all types of ML reports."""
    print("Generating All ML Report Types...")
    print("=" * 50)
    
    # Initialize report generator
    report_gen = MLReportGenerator()
    
    # Common metadata
    metadata = {
        'project': 'ML Backtest Suite',
        'version': '1.0',
        'author': 'ML Team',
        'date': datetime.now().strftime('%Y-%m-%d')
    }
    
    # 1. Feature Analysis Report
    print("\n1. Feature Analysis Report")
    print("-" * 30)
    
    features = ['RSI', 'MACD', 'Volume_Ratio', 'Price_Change', 'MA_Cross', 
                'Volatility', 'Market_Cap', 'PE_Ratio', 'Sentiment', 'Momentum']
    
    feature_importance = pd.DataFrame({
        'importance': np.array([0.35, 0.28, 0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04])
    }, index=features)
    
    # Create realistic correlation matrix
    n_features = len(features)
    correlation_matrix = np.eye(n_features)
    for i in range(n_features):
        for j in range(i+1, n_features):
            corr = np.random.uniform(-0.6, 0.6)
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr
    
    correlation_matrix = pd.DataFrame(correlation_matrix, index=features, columns=features)
    
    # Generate feature distributions
    feature_distributions = {}
    for i, feature in enumerate(features[:5]):
        if feature == 'RSI':
            # RSI bounded between 0 and 100
            values = np.random.beta(2, 2, 1000) * 100
        elif feature == 'Volume_Ratio':
            # Log-normal distribution for volume
            values = np.random.lognormal(0, 0.5, 1000)
        else:
            # Normal distributions with different parameters
            values = np.random.normal(i * 0.5, 1 + i * 0.1, 1000)
        feature_distributions[feature] = pd.Series(values)
    
    report1 = report_gen.generate_feature_analysis_report(
        feature_importance=feature_importance,
        correlation_matrix=correlation_matrix,
        feature_distributions=feature_distributions,
        metadata=metadata
    )
    print(f"✓ Generated: {os.path.basename(report1)}")
    
    # 2. Performance Dashboard
    print("\n2. Performance Dashboard")
    print("-" * 30)
    
    models = ['Random Forest', 'XGBoost', 'Neural Network', 'SVM']
    
    model_metrics = {}
    confusion_matrices = {}
    roc_data = {}
    
    for i, model in enumerate(models):
        # Generate realistic metrics
        base_accuracy = 0.75 + i * 0.03 + np.random.uniform(-0.02, 0.02)
        model_metrics[model] = {
            'accuracy': base_accuracy,
            'precision': base_accuracy - 0.03 + np.random.uniform(-0.02, 0.02),
            'recall': base_accuracy - 0.05 + np.random.uniform(-0.02, 0.02),
            'f1_score': base_accuracy - 0.02 + np.random.uniform(-0.02, 0.02),
            'auc': base_accuracy + 0.05 + np.random.uniform(-0.02, 0.02)
        }
        
        # Confusion matrix
        tp = int(850 + i * 20)
        tn = int(750 + i * 30)
        fp = int(250 - i * 20)
        fn = int(150 - i * 10)
        confusion_matrices[model] = np.array([[tn, fp], [fn, tp]])
        
        # ROC curve
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-(2 + i * 0.3) * fpr)
        roc_data[model] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': np.trapezoid(tpr, fpr)
        }
    
    # Profit curves
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    profit_curves = {}
    
    for i, model in enumerate(models):
        daily_returns = np.random.normal(0.0005 + i * 0.0001, 0.02, len(dates))
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        profit_curves[model] = pd.DataFrame({
            'cumulative_profit': cumulative_returns
        }, index=dates)
    
    report2 = report_gen.generate_performance_dashboard(
        model_metrics=model_metrics,
        confusion_matrices=confusion_matrices,
        roc_data=roc_data,
        profit_curves=profit_curves,
        metadata=metadata
    )
    print(f"✓ Generated: {os.path.basename(report2)}")
    
    # 3. Optimization Results
    print("\n3. Optimization Results")
    print("-" * 30)
    
    # Generate optimization history
    n_iterations = 100
    optimization_history = []
    best_score = 0.5
    
    for i in range(n_iterations):
        # Simulate improvement with noise
        improvement = 0.4 * (1 - np.exp(-i/20))
        current_score = 0.5 + improvement + np.random.normal(0, 0.02)
        
        if current_score > best_score:
            best_score = current_score
        
        optimization_history.append({
            'iteration': i,
            'current_score': current_score,
            'best_score': best_score,
            'improvement': improvement
        })
    
    # Parameter evolution
    params = ['learning_rate', 'max_depth', 'n_estimators', 'min_samples_split', 'subsample']
    parameter_evolution = pd.DataFrame()
    
    for param in params:
        if param == 'learning_rate':
            values = 0.1 * np.exp(-np.linspace(0, 2, n_iterations)) + 0.01
        elif param == 'max_depth':
            values = 3 + 5 * (1 - np.exp(-np.linspace(0, 3, n_iterations)))
        elif param == 'n_estimators':
            values = 100 + 400 * (1 - np.exp(-np.linspace(0, 2, n_iterations)))
        else:
            values = 0.5 + 0.3 * np.sin(np.linspace(0, 4*np.pi, n_iterations)) + np.random.normal(0, 0.05, n_iterations)
        
        parameter_evolution[param] = values
    
    # Best configurations
    best_configurations = [
        {
            'rank': 1,
            'score': 0.925,
            'learning_rate': 0.012,
            'max_depth': 7,
            'n_estimators': 450,
            'min_samples_split': 5,
            'subsample': 0.8
        },
        {
            'rank': 2,
            'score': 0.920,
            'learning_rate': 0.015,
            'max_depth': 6,
            'n_estimators': 400,
            'min_samples_split': 4,
            'subsample': 0.85
        },
        {
            'rank': 3,
            'score': 0.915,
            'learning_rate': 0.018,
            'max_depth': 8,
            'n_estimators': 350,
            'min_samples_split': 6,
            'subsample': 0.75
        }
    ]
    
    report3 = report_gen.generate_optimization_results_report(
        optimization_history=optimization_history,
        parameter_evolution=parameter_evolution,
        best_configurations=best_configurations,
        metadata=metadata
    )
    print(f"✓ Generated: {os.path.basename(report3)}")
    
    # 4. Regime Analysis
    print("\n4. Regime Analysis")
    print("-" * 30)
    
    # Generate regime data
    n_days = len(dates)
    regimes = ['Bull', 'Bear', 'Sideways']
    
    # Create realistic regime sequences
    regime_sequence = []
    current_regime = 0
    regime_duration = 0
    
    for i in range(n_days):
        if regime_duration > np.random.randint(20, 100):
            # Change regime
            current_regime = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])
            regime_duration = 0
        regime_sequence.append(current_regime)
        regime_duration += 1
    
    # Add some noise to actual regime
    actual_regime = regime_sequence.copy()
    for i in range(len(actual_regime)):
        if np.random.random() < 0.15:  # 15% error rate
            actual_regime[i] = np.random.choice([0, 1, 2])
    
    regime_data = pd.DataFrame({
        'regime': regime_sequence,
        'actual_regime': actual_regime
    }, index=dates)
    
    # Transition matrix
    transition_matrix = pd.DataFrame(
        [[0.70, 0.20, 0.10],
         [0.15, 0.65, 0.20],
         [0.25, 0.25, 0.50]],
        index=regimes,
        columns=regimes
    )
    
    # Performance by regime
    performance_by_regime = {
        'Bull': {
            'return': 0.18,
            'sharpe': 1.35,
            'max_dd': -0.08,
            'win_rate': 0.65
        },
        'Bear': {
            'return': -0.12,
            'sharpe': -0.45,
            'max_dd': -0.28,
            'win_rate': 0.35
        },
        'Sideways': {
            'return': 0.05,
            'sharpe': 0.62,
            'max_dd': -0.15,
            'win_rate': 0.52
        }
    }
    
    # Detection accuracy
    detection_accuracy = {
        'accuracy': 0.82,
        'precision': 0.79,
        'recall': 0.76,
        'f1_score': 0.77
    }
    
    report4 = report_gen.generate_regime_analysis_report(
        regime_data=regime_data,
        transition_matrix=transition_matrix,
        performance_by_regime=performance_by_regime,
        detection_accuracy=detection_accuracy,
        metadata=metadata
    )
    print(f"✓ Generated: {os.path.basename(report4)}")
    
    # 5. Strategy Comparison
    print("\n5. Strategy Comparison")
    print("-" * 30)
    
    # Generate performance data
    ml_returns = np.random.normal(0.0008, 0.015, n_days)
    baseline_returns = np.random.normal(0.0003, 0.018, n_days)
    
    # Add some correlation
    ml_returns = 0.7 * ml_returns + 0.3 * baseline_returns
    
    # Calculate cumulative returns and drawdown
    ml_cumulative = (1 + ml_returns).cumprod() - 1
    baseline_cumulative = (1 + baseline_returns).cumprod() - 1
    
    def calculate_drawdown(returns):
        cumulative = pd.Series((1 + returns).cumprod())
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.values
    
    ml_performance = pd.DataFrame({
        'returns': ml_returns,
        'cumulative_returns': ml_cumulative,
        'drawdown': calculate_drawdown(ml_returns)
    }, index=dates)
    
    baseline_performance = pd.DataFrame({
        'returns': baseline_returns,
        'cumulative_returns': baseline_cumulative,
        'drawdown': calculate_drawdown(baseline_returns)
    }, index=dates)
    
    # Comparison metrics
    def calculate_sharpe(returns, rf_rate=0.02):
        excess_returns = returns - rf_rate/252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    comparison_metrics = {
        'ML Strategy': {
            'total_return': ml_cumulative[-1],
            'sharpe_ratio': calculate_sharpe(ml_returns),
            'max_drawdown': ml_performance['drawdown'].min(),
            'win_rate': (ml_returns > 0).mean(),
            'avg_win': ml_returns[ml_returns > 0].mean(),
            'avg_loss': ml_returns[ml_returns <= 0].mean()
        },
        'Baseline Strategy': {
            'total_return': baseline_cumulative[-1],
            'sharpe_ratio': calculate_sharpe(baseline_returns),
            'max_drawdown': baseline_performance['drawdown'].min(),
            'win_rate': (baseline_returns > 0).mean(),
            'avg_win': baseline_returns[baseline_returns > 0].mean(),
            'avg_loss': baseline_returns[baseline_returns <= 0].mean()
        }
    }
    
    # Trade analysis
    trade_analysis = {}
    for strategy in ['ML Strategy', 'Baseline Strategy']:
        n_trades = 200
        if strategy == 'ML Strategy':
            profits = np.random.normal(80, 150, n_trades)
        else:
            profits = np.random.normal(50, 180, n_trades)
        
        trade_dates = pd.to_datetime('2022-01-01') + pd.to_timedelta(
            np.sort(np.random.randint(0, n_days, n_trades)), 'D'
        )
        
        trades_df = pd.DataFrame({
            'profit': profits
        }, index=trade_dates)
        
        trade_analysis[strategy] = trades_df
    
    report5 = report_gen.generate_strategy_comparison_report(
        ml_performance=ml_performance,
        baseline_performance=baseline_performance,
        comparison_metrics=comparison_metrics,
        trade_analysis=trade_analysis,
        metadata=metadata
    )
    print(f"✓ Generated: {os.path.basename(report5)}")
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ All reports generated successfully!")
    print(f"\nReports location: {report_gen.output_dir}")
    
    # List all reports
    import glob
    all_reports = sorted(glob.glob(os.path.join(report_gen.output_dir, '*.html')))
    print(f"\nTotal reports generated: {len(all_reports)}")
    
    print("\nReport types:")
    report_types = ['feature_analysis', 'performance_dashboard', 'optimization_results', 
                    'regime_analysis', 'strategy_comparison']
    
    for report_type in report_types:
        type_reports = [r for r in all_reports if report_type in r]
        if type_reports:
            print(f"  - {report_type}: {len(type_reports)} report(s)")


if __name__ == '__main__':
    generate_all_reports()