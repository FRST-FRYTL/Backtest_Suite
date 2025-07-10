"""
Example usage of ML Report Generator

This script demonstrates how to generate various ML reports.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Direct import without going through src.__init__
from src.ml.reports.report_generator import MLReportGenerator


def generate_sample_data():
    """Generate sample data for demonstration."""
    # Date range
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n_samples = len(dates)
    
    # Feature importance data
    features = ['RSI', 'MACD', 'Volume', 'Price_Change', 'MA_20', 'MA_50', 
                'Volatility', 'Market_Cap', 'PE_Ratio', 'Sentiment_Score']
    
    feature_importance = pd.DataFrame({
        'importance': np.random.uniform(0, 1, len(features))
    }, index=features)
    
    # Correlation matrix
    correlation_matrix = pd.DataFrame(
        np.random.uniform(-1, 1, (len(features), len(features))),
        index=features,
        columns=features
    )
    np.fill_diagonal(correlation_matrix.values, 1.0)
    
    # Feature distributions
    feature_distributions = {
        feature: pd.Series(np.random.normal(0, 1, 1000) + np.random.uniform(-2, 2))
        for feature in features[:5]
    }
    
    # Model metrics
    model_metrics = {
        'Random Forest': {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.78,
            'f1_score': 0.80,
            'auc': 0.88
        },
        'XGBoost': {
            'accuracy': 0.87,
            'precision': 0.85,
            'recall': 0.80,
            'f1_score': 0.82,
            'auc': 0.91
        },
        'Neural Network': {
            'accuracy': 0.83,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77,
            'auc': 0.86
        }
    }
    
    # Confusion matrices
    confusion_matrices = {
        'Random Forest': np.array([[850, 150], [220, 780]]),
        'XGBoost': np.array([[870, 130], [200, 800]]),
        'Neural Network': np.array([[830, 170], [250, 750]])
    }
    
    # ROC data
    roc_data = {}
    for model in model_metrics.keys():
        n_points = 100
        fpr = np.linspace(0, 1, n_points)
        # Generate realistic TPR curves
        if model == 'XGBoost':
            tpr = 1 - np.exp(-3 * fpr)
        elif model == 'Random Forest':
            tpr = 1 - np.exp(-2.5 * fpr)
        else:
            tpr = 1 - np.exp(-2 * fpr)
        
        roc_data[model] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': np.trapz(tpr, fpr)
        }
    
    # Profit curves
    profit_curves = {}
    for model in model_metrics.keys():
        base_return = np.random.normal(0.0005, 0.02, n_samples)
        if model == 'XGBoost':
            returns = base_return + np.random.normal(0.0002, 0.001, n_samples)
        elif model == 'Random Forest':
            returns = base_return + np.random.normal(0.0001, 0.001, n_samples)
        else:
            returns = base_return + np.random.normal(0, 0.001, n_samples)
        
        profit_curves[model] = pd.DataFrame({
            'cumulative_profit': (1 + returns).cumprod() - 1
        }, index=dates)
    
    # Optimization history
    optimization_history = []
    best_score = 0.5
    for i in range(100):
        current_score = 0.5 + (0.4 * (1 - np.exp(-i/20))) + np.random.normal(0, 0.02)
        if current_score > best_score:
            best_score = current_score
        optimization_history.append({
            'iteration': i,
            'current_score': current_score,
            'best_score': best_score
        })
    
    # Parameter evolution
    params = ['learning_rate', 'max_depth', 'n_estimators', 'min_samples_split', 
              'subsample', 'reg_alpha']
    parameter_evolution = pd.DataFrame({
        param: 0.5 + 0.3 * np.sin(np.linspace(0, 4*np.pi, 100)) + 
               np.random.normal(0, 0.05, 100)
        for param in params
    })
    
    # Best configurations
    best_configurations = [
        {
            'rank': 1,
            'score': 0.92,
            'learning_rate': 0.01,
            'max_depth': 7,
            'n_estimators': 500,
            'min_samples_split': 5
        },
        {
            'rank': 2,
            'score': 0.91,
            'learning_rate': 0.015,
            'max_depth': 6,
            'n_estimators': 400,
            'min_samples_split': 4
        },
        {
            'rank': 3,
            'score': 0.90,
            'learning_rate': 0.02,
            'max_depth': 8,
            'n_estimators': 300,
            'min_samples_split': 6
        }
    ]
    
    # Regime data
    regimes = ['Bull', 'Bear', 'Sideways']
    regime_data = pd.DataFrame({
        'regime': np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.3, 0.3]),
        'actual_regime': np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.3, 0.3])
    }, index=dates)
    
    # Smooth regime transitions
    regime_data['regime'] = regime_data['regime'].rolling(20).mean().fillna(0).round()
    regime_data['actual_regime'] = regime_data['actual_regime'].rolling(20).mean().fillna(0).round()
    
    # Transition matrix
    transition_matrix = pd.DataFrame(
        [[0.7, 0.2, 0.1],
         [0.15, 0.6, 0.25],
         [0.2, 0.3, 0.5]],
        index=regimes,
        columns=regimes
    )
    
    # Performance by regime
    performance_by_regime = {
        'Bull': {'return': 0.15, 'sharpe': 1.2, 'max_dd': -0.08},
        'Bear': {'return': -0.05, 'sharpe': -0.3, 'max_dd': -0.25},
        'Sideways': {'return': 0.03, 'sharpe': 0.5, 'max_dd': -0.12}
    }
    
    # Detection accuracy
    detection_accuracy = {
        'accuracy': 0.78,
        'precision': 0.75,
        'recall': 0.72,
        'f1_score': 0.73
    }
    
    # Strategy performance data
    ml_returns = np.random.normal(0.0008, 0.015, n_samples)
    baseline_returns = np.random.normal(0.0003, 0.018, n_samples)
    
    ml_performance = pd.DataFrame({
        'returns': ml_returns,
        'cumulative_returns': (1 + ml_returns).cumprod() - 1,
        'drawdown': calculate_drawdown(ml_returns)
    }, index=dates)
    
    baseline_performance = pd.DataFrame({
        'returns': baseline_returns,
        'cumulative_returns': (1 + baseline_returns).cumprod() - 1,
        'drawdown': calculate_drawdown(baseline_returns)
    }, index=dates)
    
    # Comparison metrics
    comparison_metrics = {
        'ML Strategy': {
            'total_return': ml_performance['cumulative_returns'].iloc[-1],
            'sharpe_ratio': calculate_sharpe(ml_returns),
            'max_drawdown': ml_performance['drawdown'].min(),
            'win_rate': (ml_returns > 0).mean()
        },
        'Baseline Strategy': {
            'total_return': baseline_performance['cumulative_returns'].iloc[-1],
            'sharpe_ratio': calculate_sharpe(baseline_returns),
            'max_drawdown': baseline_performance['drawdown'].min(),
            'win_rate': (baseline_returns > 0).mean()
        }
    }
    
    # Trade analysis
    trade_analysis = {}
    for strategy in ['ML Strategy', 'Baseline Strategy']:
        n_trades = 500
        trades = pd.DataFrame({
            'entry_date': pd.to_datetime('2020-01-01') + pd.to_timedelta(np.random.randint(0, 1000, n_trades), 'D'),
            'profit': np.random.normal(50, 200, n_trades)
        })
        trades = trades.set_index('entry_date').sort_index()
        trade_analysis[strategy] = trades
    
    return {
        'feature_importance': feature_importance,
        'correlation_matrix': correlation_matrix,
        'feature_distributions': feature_distributions,
        'model_metrics': model_metrics,
        'confusion_matrices': confusion_matrices,
        'roc_data': roc_data,
        'profit_curves': profit_curves,
        'optimization_history': optimization_history,
        'parameter_evolution': parameter_evolution,
        'best_configurations': best_configurations,
        'regime_data': regime_data,
        'transition_matrix': transition_matrix,
        'performance_by_regime': performance_by_regime,
        'detection_accuracy': detection_accuracy,
        'ml_performance': ml_performance,
        'baseline_performance': baseline_performance,
        'comparison_metrics': comparison_metrics,
        'trade_analysis': trade_analysis
    }


def calculate_drawdown(returns):
    """Calculate drawdown from returns."""
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown


def calculate_sharpe(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio."""
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def main():
    """Generate all example reports."""
    print("Generating ML Report Examples...")
    
    # Initialize report generator
    report_gen = MLReportGenerator()
    
    # Generate sample data
    data = generate_sample_data()
    
    # Metadata for all reports
    metadata = {
        'dataset': 'Sample Financial Data',
        'period': '2020-01-01 to 2023-12-31',
        'author': 'ML Team',
        'version': '1.0'
    }
    
    # 1. Feature Analysis Report
    print("\n1. Generating Feature Analysis Report...")
    feature_report = report_gen.generate_feature_analysis_report(
        feature_importance=data['feature_importance'],
        correlation_matrix=data['correlation_matrix'],
        feature_distributions=data['feature_distributions'],
        metadata=metadata
    )
    print(f"   Report saved to: {feature_report}")
    
    # 2. Performance Dashboard
    print("\n2. Generating Performance Dashboard...")
    performance_report = report_gen.generate_performance_dashboard(
        model_metrics=data['model_metrics'],
        confusion_matrices=data['confusion_matrices'],
        roc_data=data['roc_data'],
        profit_curves=data['profit_curves'],
        metadata=metadata
    )
    print(f"   Report saved to: {performance_report}")
    
    # 3. Optimization Results
    print("\n3. Generating Optimization Results Report...")
    optimization_report = report_gen.generate_optimization_results_report(
        optimization_history=data['optimization_history'],
        parameter_evolution=data['parameter_evolution'],
        best_configurations=data['best_configurations'],
        metadata=metadata
    )
    print(f"   Report saved to: {optimization_report}")
    
    # 4. Regime Analysis
    print("\n4. Generating Regime Analysis Report...")
    regime_report = report_gen.generate_regime_analysis_report(
        regime_data=data['regime_data'],
        transition_matrix=data['transition_matrix'],
        performance_by_regime=data['performance_by_regime'],
        detection_accuracy=data['detection_accuracy'],
        metadata=metadata
    )
    print(f"   Report saved to: {regime_report}")
    
    # 5. Strategy Comparison
    print("\n5. Generating Strategy Comparison Report...")
    comparison_report = report_gen.generate_strategy_comparison_report(
        ml_performance=data['ml_performance'],
        baseline_performance=data['baseline_performance'],
        comparison_metrics=data['comparison_metrics'],
        trade_analysis=data['trade_analysis'],
        metadata=metadata
    )
    print(f"   Report saved to: {comparison_report}")
    
    print("\n✅ All reports generated successfully!")
    print(f"\nReports are available in: {report_gen.output_dir}")


if __name__ == '__main__':
    main()