"""
Example: Running the 5-Loop ML Optimization System

This example demonstrates how to use the optimization orchestrator to find
optimal parameters for a machine learning trading strategy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yfinance as yf

from src.ml.optimization import OptimizationOrchestrator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_market_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch market data for optimization"""
    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    
    # Download data using yfinance
    data = yf.download(symbol, start=start_date, end=end_date)
    
    # Ensure column names are lowercase
    data.columns = [col.lower() for col in data.columns]
    
    # Add any missing columns with synthetic data
    if 'volume' not in data.columns:
        # Generate synthetic volume data
        avg_price = data['close'].mean()
        data['volume'] = np.random.poisson(1000000, len(data)) * (data['close'] / avg_price)
    
    return data


def run_basic_optimization():
    """Run basic optimization with default settings"""
    logger.info("Starting basic optimization example")
    
    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)  # 3 years of data
    
    data = fetch_market_data(
        'SPY',
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Initialize orchestrator
    orchestrator = OptimizationOrchestrator(
        config_path='config/optimization_config.yaml'
    )
    
    # Run optimization (1 round for demo)
    logger.info("Running 1-round optimization...")
    results = orchestrator.run_optimization(
        data=data,
        n_loops=1  # Just 1 complete cycle for demo
    )
    
    # Display results
    logger.info("\n" + "="*50)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("="*50)
    
    logger.info(f"Baseline Performance: {results['baseline_performance']:.4f}")
    logger.info(f"Final Performance: {results['final_performance']:.4f}")
    logger.info(f"Total Improvement: {results['total_improvement']*100:.2f}%")
    
    # Show best parameters for each loop
    logger.info("\nBest Parameters by Loop:")
    for loop_name, summary in results['loop_summaries'].items():
        logger.info(f"\n{loop_name.upper()}:")
        logger.info(f"  Best Value: {summary['best_value']:.4f}")
        logger.info(f"  Improvement: {summary['improvement']*100:.2f}%")
        logger.info(f"  Trials: {summary['n_trials']}")
    
    # Visualize results
    orchestrator.visualize_optimization_progress()
    logger.info("\nOptimization plots saved to results directory")
    
    return results


def run_advanced_optimization():
    """Run advanced optimization with custom parameters"""
    logger.info("Starting advanced optimization example")
    
    # Fetch data for multiple assets
    symbols = ['SPY', 'QQQ', 'IWM']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)  # 5 years of data
    
    all_data = {}
    for symbol in symbols:
        all_data[symbol] = fetch_market_data(
            symbol,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    
    # Use SPY as primary data
    data = all_data['SPY']
    
    # Custom initial parameters
    initial_params = {
        'feature_engineering': {
            'use_sma': True,
            'sma_periods': 20,
            'use_rsi': True,
            'rsi_period': 14,
            'use_macd': True,
            'n_lags': 5,
            'rolling_window': 20
        },
        'model_architecture': {
            'type': 'xgboost',
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1
            }
        },
        'regime_detection': {
            'method': 'ensemble',
            'n_regimes': 4,
            'ensemble': {
                'methods': ['volatility', 'trend']
            }
        },
        'risk_management': {
            'position_sizing': {
                'method': 'kelly',
                'kelly_fraction': 0.25
            },
            'stop_loss': {
                'enabled': True,
                'method': 'trailing'
            }
        }
    }
    
    # Initialize orchestrator with custom config
    orchestrator = OptimizationOrchestrator(
        config_path='config/optimization_config.yaml'
    )
    
    # Run multi-round optimization
    logger.info("Running 3-round optimization...")
    results = orchestrator.run_optimization(
        data=data,
        initial_params=initial_params,
        n_loops=3  # 3 complete cycles
    )
    
    # Save best parameters
    import yaml
    best_params_file = 'optimization_results/best_params_advanced.yaml'
    with open(best_params_file, 'w') as f:
        yaml.dump(results['best_params'], f)
    logger.info(f"\nBest parameters saved to {best_params_file}")
    
    return results


def demonstrate_individual_loops():
    """Demonstrate individual optimization loops"""
    logger.info("Demonstrating individual optimization loops")
    
    # Fetch sample data
    data = fetch_market_data(
        'AAPL',
        '2022-01-01',
        '2024-01-01'
    )
    
    # Load config
    import yaml
    with open('config/optimization_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Feature Optimization
    logger.info("\n1. FEATURE OPTIMIZATION")
    from src.ml.optimization import FeatureOptimization
    
    feature_opt = FeatureOptimization(config['feature_optimization'])
    
    # Create sample trial
    import optuna
    study = optuna.create_study(direction='maximize')
    
    def feature_objective(trial):
        params = feature_opt.get_trial_params(trial)
        score = feature_opt.evaluate(data, params)
        return score
    
    study.optimize(feature_objective, n_trials=10)
    logger.info(f"Best feature score: {study.best_value:.4f}")
    logger.info(f"Best feature params: {study.best_params}")
    
    # 2. Architecture Optimization
    logger.info("\n2. ARCHITECTURE OPTIMIZATION")
    from src.ml.optimization import ArchitectureOptimization
    
    arch_opt = ArchitectureOptimization(config['architecture_optimization'])
    
    def architecture_objective(trial):
        params = arch_opt.get_trial_params(trial)
        # Add features from previous optimization
        params['features'] = feature_opt.get_optimized_features(data, {'feature_engineering': study.best_params})
        score = arch_opt.evaluate(data, params)
        return score
    
    arch_study = optuna.create_study(direction='maximize')
    arch_study.optimize(architecture_objective, n_trials=10)
    logger.info(f"Best architecture score: {arch_study.best_value:.4f}")
    logger.info(f"Best model type: {arch_study.best_params.get('model_type')}")
    
    # 3. Regime Optimization
    logger.info("\n3. REGIME OPTIMIZATION")
    from src.ml.optimization import RegimeOptimization
    
    regime_opt = RegimeOptimization(config['regime_optimization'])
    
    def regime_objective(trial):
        params = regime_opt.get_trial_params(trial)
        score = regime_opt.evaluate(data, params)
        return score
    
    regime_study = optuna.create_study(direction='maximize')
    regime_study.optimize(regime_objective, n_trials=10)
    logger.info(f"Best regime score: {regime_study.best_value:.4f}")
    logger.info(f"Best regime method: {regime_study.best_params.get('regime_detection_method')}")
    
    # Show detected regimes
    regimes = regime_opt.get_regime_labels(data, {'regime_detection': regime_study.best_params})
    logger.info(f"Detected regimes: {regimes.value_counts().to_dict()}")


def backtest_optimized_strategy():
    """Backtest a strategy using optimized parameters"""
    logger.info("Backtesting optimized strategy")
    
    # Load previously saved optimized parameters
    import yaml
    try:
        with open('optimization_results/best_params_advanced.yaml', 'r') as f:
            best_params = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("No saved parameters found. Running optimization first...")
        results = run_basic_optimization()
        best_params = results['best_params']
    
    # Fetch fresh data for out-of-sample testing
    test_data = fetch_market_data(
        'SPY',
        '2024-01-01',
        datetime.now().strftime('%Y-%m-%d')
    )
    
    # Create a simple strategy using optimized parameters
    from src.ml.optimization import IntegrationOptimization
    integration = IntegrationOptimization({})
    
    # Build final strategy
    strategy = integration.build_final_strategy(test_data, best_params)
    
    logger.info("\nOptimized Strategy Performance:")
    for metric, value in strategy['performance_metrics'].items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Optimization Examples')
    parser.add_argument(
        '--example',
        choices=['basic', 'advanced', 'loops', 'backtest', 'all'],
        default='basic',
        help='Which example to run'
    )
    
    args = parser.parse_args()
    
    if args.example == 'basic' or args.example == 'all':
        logger.info("\n" + "="*70)
        logger.info("RUNNING BASIC OPTIMIZATION EXAMPLE")
        logger.info("="*70)
        run_basic_optimization()
    
    if args.example == 'advanced' or args.example == 'all':
        logger.info("\n" + "="*70)
        logger.info("RUNNING ADVANCED OPTIMIZATION EXAMPLE")
        logger.info("="*70)
        run_advanced_optimization()
    
    if args.example == 'loops' or args.example == 'all':
        logger.info("\n" + "="*70)
        logger.info("DEMONSTRATING INDIVIDUAL LOOPS")
        logger.info("="*70)
        demonstrate_individual_loops()
    
    if args.example == 'backtest' or args.example == 'all':
        logger.info("\n" + "="*70)
        logger.info("BACKTESTING OPTIMIZED STRATEGY")
        logger.info("="*70)
        backtest_optimized_strategy()
    
    logger.info("\n" + "="*70)
    logger.info("OPTIMIZATION EXAMPLES COMPLETED")
    logger.info("="*70)