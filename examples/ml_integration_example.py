"""
Example: Integrating ML Report Generator with ML Strategies

This example shows how to generate reports from actual ML strategy results.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import report generator directly
import importlib.util
spec = importlib.util.spec_from_file_location("report_generator", "src/ml/reports/report_generator.py")
report_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_module)
MLReportGenerator = report_module.MLReportGenerator


def generate_ml_strategy_report():
    """
    Example of generating reports from ML strategy backtest results.
    """
    print("ML Strategy Report Generation Example")
    print("=" * 50)
    
    # Initialize report generator
    report_gen = MLReportGenerator()
    
    # Simulate ML strategy backtest results
    # In real usage, these would come from your ML pipeline
    
    # 1. Feature importance from model training
    print("\n1. Collecting feature importance data...")
    
    feature_names = [
        'RSI_14', 'RSI_30', 'MACD_Signal', 'MACD_Histogram',
        'BB_Width', 'BB_Position', 'Volume_Ratio', 'Price_ROC',
        'ATR_14', 'ADX_14', 'OBV_Change', 'Market_Regime'
    ]
    
    # Simulate feature importance scores from XGBoost
    feature_importance = pd.DataFrame({
        'importance': [0.28, 0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02]
    }, index=feature_names)
    
    # Generate correlation matrix from feature data
    n_features = len(feature_names)
    correlation_data = np.eye(n_features)
    
    # Add some realistic correlations
    # RSI indicators correlate
    correlation_data[0, 1] = correlation_data[1, 0] = 0.75
    # MACD components correlate
    correlation_data[2, 3] = correlation_data[3, 2] = 0.85
    # Bollinger Band metrics correlate
    correlation_data[4, 5] = correlation_data[5, 4] = 0.65
    # Volume and OBV correlate
    correlation_data[6, 10] = correlation_data[10, 6] = 0.55
    
    correlation_matrix = pd.DataFrame(
        correlation_data,
        index=feature_names,
        columns=feature_names
    )
    
    # Generate sample feature distributions
    n_samples = 1000
    feature_distributions = {
        'RSI_14': pd.Series(np.random.beta(2, 2, n_samples) * 100),  # RSI bounded 0-100
        'MACD_Signal': pd.Series(np.random.normal(0, 0.5, n_samples)),
        'Volume_Ratio': pd.Series(np.random.lognormal(0, 0.3, n_samples)),
        'ATR_14': pd.Series(np.random.gamma(2, 0.5, n_samples)),
        'Market_Regime': pd.Series(np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.35, 0.25]))
    }
    
    # Generate feature analysis report
    feature_report = report_gen.generate_feature_analysis_report(
        feature_importance=feature_importance,
        correlation_matrix=correlation_matrix,
        feature_distributions=feature_distributions,
        metadata={
            'model': 'XGBoost',
            'training_period': '2020-01-01 to 2022-12-31',
            'n_features': len(feature_names),
            'n_samples': n_samples
        }
    )
    print(f"✓ Feature analysis report: {os.path.basename(feature_report)}")
    
    # 2. Model performance metrics
    print("\n2. Collecting model performance data...")
    
    # Simulate cross-validation results for multiple models
    models = {
        'XGBoost': {
            'metrics': {
                'accuracy': 0.876,
                'precision': 0.852,
                'recall': 0.823,
                'f1_score': 0.837,
                'auc': 0.912
            },
            'confusion_matrix': np.array([[8543, 1457], [1823, 8177]])
        },
        'Random Forest': {
            'metrics': {
                'accuracy': 0.862,
                'precision': 0.841,
                'recall': 0.812,
                'f1_score': 0.826,
                'auc': 0.895
            },
            'confusion_matrix': np.array([[8321, 1679], [1933, 8067]])
        },
        'LightGBM': {
            'metrics': {
                'accuracy': 0.871,
                'precision': 0.848,
                'recall': 0.819,
                'f1_score': 0.833,
                'auc': 0.908
            },
            'confusion_matrix': np.array([[8456, 1544], [1862, 8138]])
        }
    }
    
    # Extract data for report
    model_metrics = {name: data['metrics'] for name, data in models.items()}
    confusion_matrices = {name: data['confusion_matrix'] for name, data in models.items()}
    
    # Generate ROC curves
    roc_data = {}
    for model_name in models.keys():
        n_points = 100
        fpr = np.linspace(0, 1, n_points)
        
        # Generate realistic ROC curves based on AUC
        auc = models[model_name]['metrics']['auc']
        # Use power function to create realistic curve
        power = -np.log(1 - auc) / np.log(2)
        tpr = 1 - (1 - fpr) ** power
        
        roc_data[model_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc
        }
    
    # Simulate profit curves from backtest
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    profit_curves = {}
    
    for model_name in models.keys():
        # Base returns with model-specific alpha
        base_returns = np.random.normal(0.0003, 0.018, len(dates))
        
        if model_name == 'XGBoost':
            alpha = 0.0004
        elif model_name == 'LightGBM':
            alpha = 0.0003
        else:
            alpha = 0.0002
        
        returns = base_returns + alpha + np.random.normal(0, 0.002, len(dates))
        cumulative_returns = (1 + returns).cumprod() - 1
        
        profit_curves[model_name] = pd.DataFrame({
            'cumulative_profit': cumulative_returns
        }, index=dates)
    
    # Generate performance dashboard
    performance_report = report_gen.generate_performance_dashboard(
        model_metrics=model_metrics,
        confusion_matrices=confusion_matrices,
        roc_data=roc_data,
        profit_curves=profit_curves,
        metadata={
            'evaluation_period': '2023-01-01 to 2023-12-31',
            'cross_validation': '5-fold TimeSeriesSplit',
            'models': list(models.keys())
        }
    )
    print(f"✓ Performance dashboard: {os.path.basename(performance_report)}")
    
    # 3. Hyperparameter optimization results
    print("\n3. Collecting optimization results...")
    
    # Simulate Bayesian optimization history
    n_iterations = 50
    optimization_history = []
    
    # Parameters being optimized
    param_ranges = {
        'learning_rate': (0.001, 0.1),
        'max_depth': (3, 10),
        'n_estimators': (100, 1000),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0)
    }
    
    # Simulate optimization progress
    best_score = 0.75
    for i in range(n_iterations):
        # Simulate improvement over iterations
        improvement_rate = 0.15 * (1 - np.exp(-i/10))
        current_score = 0.75 + improvement_rate + np.random.normal(0, 0.01)
        
        if current_score > best_score:
            best_score = current_score
        
        optimization_history.append({
            'iteration': i,
            'current_score': current_score,
            'best_score': best_score,
            'improvement': (best_score - 0.75) / 0.75 * 100
        })
    
    # Parameter evolution
    parameter_evolution = pd.DataFrame()
    
    for param, (min_val, max_val) in param_ranges.items():
        if param == 'learning_rate':
            # Learning rate typically decreases
            values = max_val * np.exp(-np.linspace(0, 2, n_iterations)) + min_val
        elif param == 'n_estimators':
            # Number of estimators typically increases
            values = min_val + (max_val - min_val) * (1 - np.exp(-np.linspace(0, 3, n_iterations)))
        else:
            # Other parameters explore the space
            values = min_val + (max_val - min_val) * (0.5 + 0.4 * np.sin(np.linspace(0, 4*np.pi, n_iterations)))
            values += np.random.normal(0, 0.05 * (max_val - min_val), n_iterations)
            values = np.clip(values, min_val, max_val)
        
        parameter_evolution[param] = values
    
    # Best configurations found
    best_configurations = [
        {
            'rank': 1,
            'score': 0.912,
            'learning_rate': 0.015,
            'max_depth': 7,
            'n_estimators': 800,
            'subsample': 0.85,
            'colsample_bytree': 0.8
        },
        {
            'rank': 2,
            'score': 0.908,
            'learning_rate': 0.018,
            'max_depth': 6,
            'n_estimators': 750,
            'subsample': 0.9,
            'colsample_bytree': 0.75
        },
        {
            'rank': 3,
            'score': 0.905,
            'learning_rate': 0.012,
            'max_depth': 8,
            'n_estimators': 850,
            'subsample': 0.8,
            'colsample_bytree': 0.85
        }
    ]
    
    # Generate optimization report
    optimization_report = report_gen.generate_optimization_results_report(
        optimization_history=optimization_history,
        parameter_evolution=parameter_evolution,
        best_configurations=best_configurations,
        metadata={
            'optimization_method': 'Bayesian Optimization',
            'objective': 'Maximize Sharpe Ratio',
            'n_iterations': n_iterations,
            'parameter_space': param_ranges
        }
    )
    print(f"✓ Optimization results: {os.path.basename(optimization_report)}")
    
    # 4. ML vs Baseline comparison
    print("\n4. Generating strategy comparison...")
    
    # Simulate ML strategy vs baseline performance
    n_days = len(dates)
    
    # ML strategy with edge
    ml_returns = np.random.normal(0.0008, 0.015, n_days)
    # Add some market regime awareness
    regime_boost = np.where(np.random.random(n_days) > 0.7, 0.0005, 0)
    ml_returns += regime_boost
    
    # Baseline buy-and-hold
    baseline_returns = np.random.normal(0.0003, 0.018, n_days)
    
    # Calculate performance metrics
    ml_cumulative = (1 + ml_returns).cumprod() - 1
    baseline_cumulative = (1 + baseline_returns).cumprod() - 1
    
    def calculate_drawdown(returns):
        cumulative = pd.Series((1 + returns).cumprod())
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.values
    
    def calculate_sharpe(returns, rf_rate=0.02):
        excess_returns = returns - rf_rate/252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    # Create performance DataFrames
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
    comparison_metrics = {
        'ML Strategy (XGBoost)': {
            'total_return': ml_cumulative[-1],
            'annual_return': ml_cumulative[-1],  # Assuming 1 year
            'sharpe_ratio': calculate_sharpe(ml_returns),
            'max_drawdown': ml_performance['drawdown'].min(),
            'win_rate': (ml_returns > 0).mean(),
            'profit_factor': ml_returns[ml_returns > 0].sum() / -ml_returns[ml_returns <= 0].sum()
        },
        'Buy & Hold': {
            'total_return': baseline_cumulative[-1],
            'annual_return': baseline_cumulative[-1],
            'sharpe_ratio': calculate_sharpe(baseline_returns),
            'max_drawdown': baseline_performance['drawdown'].min(),
            'win_rate': (baseline_returns > 0).mean(),
            'profit_factor': baseline_returns[baseline_returns > 0].sum() / -baseline_returns[baseline_returns <= 0].sum()
        }
    }
    
    # Simulate trade analysis
    trade_analysis = {}
    
    # ML strategy makes selective trades
    n_ml_trades = 150
    ml_trade_dates = pd.to_datetime('2023-01-01') + pd.to_timedelta(
        np.sort(np.random.choice(n_days, n_ml_trades, replace=False)), 'D'
    )
    ml_trades = pd.DataFrame({
        'profit': np.random.normal(100, 150, n_ml_trades)
    }, index=ml_trade_dates)
    trade_analysis['ML Strategy (XGBoost)'] = ml_trades
    
    # Baseline is buy and hold (daily "trades")
    baseline_trades = pd.DataFrame({
        'profit': baseline_returns * 10000  # Assuming $10k position
    }, index=dates)
    trade_analysis['Buy & Hold'] = baseline_trades
    
    # Generate comparison report
    comparison_report = report_gen.generate_strategy_comparison_report(
        ml_performance=ml_performance,
        baseline_performance=baseline_performance,
        comparison_metrics=comparison_metrics,
        trade_analysis=trade_analysis,
        metadata={
            'ml_model': 'XGBoost with regime detection',
            'baseline': 'Buy and Hold SPY',
            'period': '2023-01-01 to 2023-12-31',
            'initial_capital': 10000
        }
    )
    print(f"✓ Strategy comparison: {os.path.basename(comparison_report)}")
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ All ML strategy reports generated successfully!")
    print(f"\nReports location: {report_gen.output_dir}")
    print("\nThese reports provide comprehensive analysis of:")
    print("  - Feature importance and relationships")
    print("  - Model performance and validation")
    print("  - Hyperparameter optimization results")
    print("  - Strategy comparison with baseline")
    
    return {
        'feature_report': feature_report,
        'performance_report': performance_report,
        'optimization_report': optimization_report,
        'comparison_report': comparison_report
    }


if __name__ == '__main__':
    reports = generate_ml_strategy_report()