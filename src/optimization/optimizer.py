"""Strategy parameter optimization framework."""

import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Any, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import differential_evolution
from sklearn.model_selection import TimeSeriesSplit

from ..backtesting import BacktestEngine
from ..strategies.builder import StrategyBuilder
from ..utils.metrics import PerformanceMetrics


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict]
    optimization_metric: str
    total_iterations: int
    
    def get_top_results(self, n: int = 10) -> List[Dict]:
        """Get top N results sorted by score."""
        sorted_results = sorted(
            self.all_results,
            key=lambda x: x['score'],
            reverse=True
        )
        return sorted_results[:n]
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame(self.all_results)


class StrategyOptimizer:
    """Optimize strategy parameters using various methods."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategy_builder: StrategyBuilder,
        optimization_metric: str = "sharpe_ratio",
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        n_jobs: int = -1
    ):
        """
        Initialize optimizer.
        
        Args:
            data: Market data for backtesting
            strategy_builder: Base strategy builder
            optimization_metric: Metric to optimize
            initial_capital: Starting capital
            commission_rate: Commission rate
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        """
        self.data = data
        self.strategy_builder = strategy_builder
        self.optimization_metric = optimization_metric
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        
        if n_jobs == -1:
            import multiprocessing
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = n_jobs
            
    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        progress_bar: bool = True
    ) -> OptimizationResult:
        """
        Perform grid search optimization.
        
        Args:
            param_grid: Dictionary of parameter names and values to test
            progress_bar: Show progress bar
            
        Returns:
            OptimizationResult object
        """
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        # Run backtests
        results = []
        
        if self.n_jobs == 1:
            # Sequential execution
            iterator = tqdm(param_combinations) if progress_bar else param_combinations
            for params in iterator:
                param_dict = dict(zip(param_names, params))
                score, metrics = self._evaluate_parameters(param_dict)
                results.append({
                    'params': param_dict,
                    'score': score,
                    'metrics': metrics
                })
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(
                        self._evaluate_parameters,
                        dict(zip(param_names, params))
                    ): dict(zip(param_names, params))
                    for params in param_combinations
                }
                
                # Process results
                iterator = tqdm(as_completed(futures), total=len(futures)) if progress_bar else as_completed(futures)
                for future in iterator:
                    param_dict = futures[future]
                    try:
                        score, metrics = future.result()
                        results.append({
                            'params': param_dict,
                            'score': score,
                            'metrics': metrics
                        })
                    except Exception as e:
                        print(f"Error with params {param_dict}: {e}")
                        results.append({
                            'params': param_dict,
                            'score': -np.inf,
                            'metrics': {}
                        })
                        
        # Find best result
        best_result = max(results, key=lambda x: x['score'])
        
        return OptimizationResult(
            best_params=best_result['params'],
            best_score=best_result['score'],
            all_results=results,
            optimization_metric=self.optimization_metric,
            total_iterations=len(param_combinations)
        )
        
    def random_search(
        self,
        param_distributions: Dict[str, Tuple[float, float]],
        n_iter: int = 100,
        progress_bar: bool = True
    ) -> OptimizationResult:
        """
        Perform random search optimization.
        
        Args:
            param_distributions: Parameter ranges (min, max)
            n_iter: Number of iterations
            progress_bar: Show progress bar
            
        Returns:
            OptimizationResult object
        """
        results = []
        
        for i in tqdm(range(n_iter)) if progress_bar else range(n_iter):
            # Sample random parameters
            param_dict = {}
            for param, (min_val, max_val) in param_distributions.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    param_dict[param] = np.random.randint(min_val, max_val + 1)
                else:
                    param_dict[param] = np.random.uniform(min_val, max_val)
                    
            # Evaluate
            score, metrics = self._evaluate_parameters(param_dict)
            results.append({
                'params': param_dict,
                'score': score,
                'metrics': metrics
            })
            
        # Find best result
        best_result = max(results, key=lambda x: x['score'])
        
        return OptimizationResult(
            best_params=best_result['params'],
            best_score=best_result['score'],
            all_results=results,
            optimization_metric=self.optimization_metric,
            total_iterations=n_iter
        )
        
    def differential_evolution(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        population_size: int = 15,
        max_iter: int = 100,
        progress_bar: bool = True
    ) -> OptimizationResult:
        """
        Optimize using differential evolution algorithm.
        
        Args:
            param_bounds: Parameter bounds
            population_size: DE population size
            max_iter: Maximum iterations
            progress_bar: Show progress bar
            
        Returns:
            OptimizationResult object
        """
        param_names = list(param_bounds.keys())
        bounds = [param_bounds[name] for name in param_names]
        
        # Track all evaluations
        self.de_results = []
        
        def objective(params):
            """Objective function for DE."""
            param_dict = dict(zip(param_names, params))
            score, metrics = self._evaluate_parameters(param_dict)
            
            # Store result
            self.de_results.append({
                'params': param_dict,
                'score': score,
                'metrics': metrics
            })
            
            # DE minimizes, so negate score
            return -score
            
        # Run optimization
        result = differential_evolution(
            objective,
            bounds,
            popsize=population_size,
            maxiter=max_iter,
            disp=progress_bar,
            workers=self.n_jobs if self.n_jobs > 1 else 1
        )
        
        # Get best parameters
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=self.de_results,
            optimization_metric=self.optimization_metric,
            total_iterations=len(self.de_results)
        )
        
    def _evaluate_parameters(
        self,
        params: Dict[str, Any]
    ) -> Tuple[float, Dict]:
        """
        Evaluate strategy with given parameters.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Tuple of (score, metrics)
        """
        try:
            # Create strategy with parameters
            strategy = self._build_strategy_with_params(params)
            
            # Run backtest
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate
            )
            
            results = engine.run(
                self.data,
                strategy,
                progress_bar=False
            )
            
            # Calculate metrics
            if 'equity_curve' in results and not results['equity_curve'].empty:
                metrics_calc = PerformanceMetrics.calculate(
                    results['equity_curve'],
                    results.get('trades', pd.DataFrame())
                )
                
                # Get optimization metric
                score = getattr(metrics_calc, self.optimization_metric, 0)
                
                # Convert metrics to dict
                metrics = {
                    'total_return': metrics_calc.total_return,
                    'sharpe_ratio': metrics_calc.sharpe_ratio,
                    'max_drawdown': metrics_calc.max_drawdown,
                    'win_rate': metrics_calc.win_rate,
                    'total_trades': metrics_calc.total_trades
                }
            else:
                score = -np.inf
                metrics = {}
                
            return score, metrics
            
        except Exception as e:
            print(f"Error evaluating parameters {params}: {e}")
            return -np.inf, {}
            
    def _build_strategy_with_params(
        self,
        params: Dict[str, Any]
    ) -> Any:
        """
        Build strategy with given parameters.
        
        This is a placeholder - should be customized based on strategy type.
        """
        # Clone base strategy
        builder = StrategyBuilder(name=self.strategy_builder.strategy.name)
        
        # Apply parameters
        # Example: RSI period, stop loss percentage, etc.
        # This needs to be customized based on your strategy
        
        # For demonstration, assuming some common parameters
        if 'rsi_period' in params:
            builder.add_entry_rule(f"rsi({int(params['rsi_period'])}) < 30")
            
        if 'stop_loss' in params:
            builder.set_risk_management(
                stop_loss=params['stop_loss'],
                stop_loss_type='percent'
            )
            
        return builder.build()
        
    def parameter_sensitivity(
        self,
        base_params: Dict[str, Any],
        param_ranges: Dict[str, Tuple[float, float]],
        n_steps: int = 10
    ) -> pd.DataFrame:
        """
        Analyze parameter sensitivity.
        
        Args:
            base_params: Base parameter values
            param_ranges: Ranges to test for each parameter
            n_steps: Number of steps in range
            
        Returns:
            DataFrame with sensitivity analysis
        """
        results = []
        
        for param_name, (min_val, max_val) in param_ranges.items():
            # Test range for this parameter
            test_values = np.linspace(min_val, max_val, n_steps)
            
            for test_val in test_values:
                # Create params with test value
                test_params = base_params.copy()
                test_params[param_name] = test_val
                
                # Evaluate
                score, metrics = self._evaluate_parameters(test_params)
                
                results.append({
                    'parameter': param_name,
                    'value': test_val,
                    'score': score,
                    **metrics
                })
                
        return pd.DataFrame(results)
        
    def cross_validation(
        self,
        params: Dict[str, Any],
        n_splits: int = 5,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Perform time series cross-validation.
        
        Args:
            params: Strategy parameters
            n_splits: Number of CV splits
            test_size: Test set size
            
        Returns:
            Dictionary with CV scores
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(self.data) * test_size))
        
        scores = []
        
        for train_idx, test_idx in tscv.split(self.data):
            # Split data
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]
            
            # Evaluate on test set
            self.data = test_data  # Temporarily set data
            score, _ = self._evaluate_parameters(params)
            scores.append(score)
            
        # Restore original data
        self.data = self.data
        
        return {
            'cv_mean_score': np.mean(scores),
            'cv_std_score': np.std(scores),
            'cv_scores': scores
        }