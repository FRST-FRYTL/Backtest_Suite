"""
Walk-Forward Parameter Optimization Framework

This module implements robust walk-forward optimization with proper
out-of-sample testing for the enhanced confluence strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from itertools import product
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationWindow:
    """Represents a single optimization window"""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    window_id: int

@dataclass
class ParameterSet:
    """Represents a set of strategy parameters"""
    confluence_threshold: float
    position_size: float
    stop_loss_multiplier: float
    take_profit_multiplier: float
    timeframe_weights: Dict[str, float]
    max_hold_days: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'confluence_threshold': self.confluence_threshold,
            'position_size': self.position_size,
            'stop_loss_multiplier': self.stop_loss_multiplier,
            'take_profit_multiplier': self.take_profit_multiplier,
            'timeframe_weights': self.timeframe_weights,
            'max_hold_days': self.max_hold_days
        }

@dataclass
class OptimizationResult:
    """Results from parameter optimization"""
    window: OptimizationWindow
    best_params: ParameterSet
    in_sample_performance: Dict[str, float]
    out_of_sample_performance: Dict[str, float]
    parameter_stability: float
    overfitting_score: float

class WalkForwardOptimizer:
    """
    Implements walk-forward optimization for robust parameter selection.
    """
    
    def __init__(
        self,
        objective_function: str = 'sharpe_ratio',
        train_window_days: int = 252,  # 1 year
        test_window_days: int = 63,     # 3 months
        step_days: int = 21,            # 1 month
        min_trades_required: int = 20
    ):
        """
        Initialize the walk-forward optimizer.
        
        Args:
            objective_function: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            train_window_days: Training window size in days
            test_window_days: Testing window size in days
            step_days: Step size for rolling windows
            min_trades_required: Minimum trades for valid optimization
        """
        self.objective_function = objective_function
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.min_trades_required = min_trades_required
        
        # Results storage
        self.optimization_results: List[OptimizationResult] = []
        self.parameter_history: List[Dict[str, Any]] = []
        
    def create_optimization_windows(
        self,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[OptimizationWindow]:
        """
        Create walk-forward optimization windows.
        
        Args:
            data: DataFrame with date index
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            List of optimization windows
        """
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        windows = []
        window_id = 0
        
        # Ensure we have enough data
        min_required_days = self.train_window_days + self.test_window_days
        if len(data) < min_required_days:
            raise ValueError(f"Insufficient data: need at least {min_required_days} days")
        
        # Create rolling windows
        start_idx = self.train_window_days
        while start_idx + self.test_window_days <= len(data):
            train_start_idx = start_idx - self.train_window_days
            train_end_idx = start_idx
            test_start_idx = start_idx
            test_end_idx = start_idx + self.test_window_days
            
            window = OptimizationWindow(
                train_start=data.index[train_start_idx],
                train_end=data.index[train_end_idx - 1],
                test_start=data.index[test_start_idx],
                test_end=data.index[test_end_idx - 1],
                window_id=window_id
            )
            
            windows.append(window)
            window_id += 1
            start_idx += self.step_days
            
        logger.info(f"Created {len(windows)} optimization windows")
        return windows
    
    def generate_parameter_grid(
        self,
        confluence_thresholds: List[float],
        position_sizes: List[float],
        stop_loss_multipliers: List[float],
        take_profit_multipliers: List[float],
        timeframe_combinations: List[Dict[str, float]],
        max_hold_days_options: List[int]
    ) -> List[ParameterSet]:
        """
        Generate parameter combinations for optimization.
        
        Args:
            confluence_thresholds: List of confluence threshold values
            position_sizes: List of position size values
            stop_loss_multipliers: List of stop loss multipliers
            take_profit_multipliers: List of take profit multipliers
            timeframe_combinations: List of timeframe weight dictionaries
            max_hold_days_options: List of max hold days options
            
        Returns:
            List of parameter sets
        """
        parameter_sets = []
        
        # Generate all combinations
        for params in product(
            confluence_thresholds,
            position_sizes,
            stop_loss_multipliers,
            take_profit_multipliers,
            timeframe_combinations,
            max_hold_days_options
        ):
            param_set = ParameterSet(
                confluence_threshold=params[0],
                position_size=params[1],
                stop_loss_multiplier=params[2],
                take_profit_multiplier=params[3],
                timeframe_weights=params[4],
                max_hold_days=params[5]
            )
            parameter_sets.append(param_set)
            
        logger.info(f"Generated {len(parameter_sets)} parameter combinations")
        return parameter_sets
    
    def optimize_window(
        self,
        window: OptimizationWindow,
        parameter_sets: List[ParameterSet],
        data: pd.DataFrame,
        backtest_function: Callable,
        parallel: bool = True,
        n_jobs: int = -1
    ) -> OptimizationResult:
        """
        Optimize parameters for a single window.
        
        Args:
            window: Optimization window
            parameter_sets: List of parameter sets to test
            data: Complete dataset
            backtest_function: Function to run backtest with parameters
            parallel: Whether to use parallel processing
            n_jobs: Number of parallel jobs (-1 for all cores)
            
        Returns:
            Optimization result for the window
        """
        # Split data for train/test
        train_data = data[(data.index >= window.train_start) & (data.index <= window.train_end)]
        test_data = data[(data.index >= window.test_start) & (data.index <= window.test_end)]
        
        logger.info(f"Optimizing window {window.window_id}: "
                   f"Train {window.train_start.date()} to {window.train_end.date()}, "
                   f"Test {window.test_start.date()} to {window.test_end.date()}")
        
        # Optimize on training data
        best_params = None
        best_score = -np.inf
        in_sample_results = {}
        
        if parallel and n_jobs != 1:
            # Parallel optimization
            n_workers = n_jobs if n_jobs > 0 else None
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_params = {
                    executor.submit(
                        self._evaluate_parameters,
                        params,
                        train_data,
                        backtest_function
                    ): params
                    for params in parameter_sets
                }
                
                for future in as_completed(future_to_params):
                    params = future_to_params[future]
                    try:
                        performance = future.result()
                        score = performance.get(self.objective_function, -np.inf)
                        
                        if score > best_score and performance.get('total_trades', 0) >= self.min_trades_required:
                            best_score = score
                            best_params = params
                            in_sample_results = performance
                            
                    except Exception as e:
                        logger.error(f"Error evaluating parameters: {e}")
        else:
            # Sequential optimization
            for params in parameter_sets:
                performance = self._evaluate_parameters(params, train_data, backtest_function)
                score = performance.get(self.objective_function, -np.inf)
                
                if score > best_score and performance.get('total_trades', 0) >= self.min_trades_required:
                    best_score = score
                    best_params = params
                    in_sample_results = performance
        
        if best_params is None:
            logger.warning(f"No valid parameters found for window {window.window_id}")
            # Use default parameters
            best_params = parameter_sets[0]
            in_sample_results = {'error': 'No valid parameters found'}
        
        # Test on out-of-sample data
        out_of_sample_results = self._evaluate_parameters(
            best_params, test_data, backtest_function
        )
        
        # Calculate stability and overfitting scores
        parameter_stability = self._calculate_parameter_stability(best_params)
        overfitting_score = self._calculate_overfitting_score(
            in_sample_results, out_of_sample_results
        )
        
        result = OptimizationResult(
            window=window,
            best_params=best_params,
            in_sample_performance=in_sample_results,
            out_of_sample_performance=out_of_sample_results,
            parameter_stability=parameter_stability,
            overfitting_score=overfitting_score
        )
        
        self.optimization_results.append(result)
        self.parameter_history.append({
            'window_id': window.window_id,
            'parameters': best_params.to_dict(),
            'in_sample_score': best_score,
            'out_of_sample_score': out_of_sample_results.get(self.objective_function, -np.inf)
        })
        
        return result
    
    def _evaluate_parameters(
        self,
        params: ParameterSet,
        data: pd.DataFrame,
        backtest_function: Callable
    ) -> Dict[str, float]:
        """
        Evaluate a parameter set on given data.
        
        Args:
            params: Parameter set to evaluate
            data: Data to backtest on
            backtest_function: Function to run backtest
            
        Returns:
            Performance metrics dictionary
        """
        try:
            # Run backtest with parameters
            results = backtest_function(data, params)
            
            # Extract key metrics
            metrics = {
                'total_return': results.get('total_return', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'max_drawdown': results.get('max_drawdown', 0),
                'total_trades': results.get('total_trades', 0),
                'win_rate': results.get('win_rate', 0),
                'profit_factor': results.get('profit_factor', 0),
                'calmar_ratio': abs(results.get('total_return', 0) / results.get('max_drawdown', -100)) if results.get('max_drawdown', 0) != 0 else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return {'error': str(e)}
    
    def _calculate_parameter_stability(self, params: ParameterSet) -> float:
        """
        Calculate stability score for parameters.
        
        Args:
            params: Parameter set
            
        Returns:
            Stability score (0-1)
        """
        if len(self.parameter_history) < 2:
            return 1.0
        
        # Compare with recent parameter selections
        recent_params = self.parameter_history[-5:]  # Last 5 windows
        
        stability_scores = []
        for hist in recent_params:
            hist_params = hist['parameters']
            
            # Calculate similarity
            confluence_diff = abs(params.confluence_threshold - hist_params.get('confluence_threshold', 0.65))
            position_diff = abs(params.position_size - hist_params.get('position_size', 0.2))
            
            similarity = 1 - (confluence_diff + position_diff) / 2
            stability_scores.append(similarity)
        
        return np.mean(stability_scores) if stability_scores else 0.5
    
    def _calculate_overfitting_score(
        self,
        in_sample: Dict[str, float],
        out_of_sample: Dict[str, float]
    ) -> float:
        """
        Calculate overfitting score.
        
        Args:
            in_sample: In-sample performance metrics
            out_of_sample: Out-of-sample performance metrics
            
        Returns:
            Overfitting score (0-1, higher means more overfitting)
        """
        if 'error' in in_sample or 'error' in out_of_sample:
            return 1.0
        
        # Compare key metrics
        metrics_to_compare = ['sharpe_ratio', 'total_return', 'win_rate']
        
        overfitting_scores = []
        for metric in metrics_to_compare:
            in_value = in_sample.get(metric, 0)
            out_value = out_of_sample.get(metric, 0)
            
            if in_value > 0:
                # Calculate degradation
                degradation = max(0, (in_value - out_value) / in_value)
                overfitting_scores.append(min(1.0, degradation))
        
        return np.mean(overfitting_scores) if overfitting_scores else 0.5
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization summary.
        
        Returns:
            Summary dictionary with key statistics
        """
        if not self.optimization_results:
            return {}
        
        # Extract performance metrics
        in_sample_sharpes = []
        out_sample_sharpes = []
        overfitting_scores = []
        
        for result in self.optimization_results:
            in_sample_sharpes.append(result.in_sample_performance.get('sharpe_ratio', 0))
            out_sample_sharpes.append(result.out_of_sample_performance.get('sharpe_ratio', 0))
            overfitting_scores.append(result.overfitting_score)
        
        # Analyze parameter consistency
        param_changes = self._analyze_parameter_changes()
        
        summary = {
            'total_windows': len(self.optimization_results),
            'avg_in_sample_sharpe': np.mean(in_sample_sharpes),
            'avg_out_sample_sharpe': np.mean(out_sample_sharpes),
            'sharpe_degradation': np.mean(in_sample_sharpes) - np.mean(out_sample_sharpes),
            'avg_overfitting_score': np.mean(overfitting_scores),
            'parameter_stability': np.mean([r.parameter_stability for r in self.optimization_results]),
            'parameter_changes': param_changes,
            'best_window': self._find_best_window(),
            'worst_window': self._find_worst_window(),
            'consistent_parameters': self._find_consistent_parameters()
        }
        
        return summary
    
    def _analyze_parameter_changes(self) -> Dict[str, Any]:
        """Analyze how parameters change over time."""
        if len(self.parameter_history) < 2:
            return {}
        
        confluence_values = [h['parameters']['confluence_threshold'] for h in self.parameter_history]
        position_values = [h['parameters']['position_size'] for h in self.parameter_history]
        
        return {
            'confluence_threshold': {
                'mean': np.mean(confluence_values),
                'std': np.std(confluence_values),
                'trend': 'increasing' if confluence_values[-1] > confluence_values[0] else 'decreasing'
            },
            'position_size': {
                'mean': np.mean(position_values),
                'std': np.std(position_values),
                'trend': 'increasing' if position_values[-1] > position_values[0] else 'decreasing'
            }
        }
    
    def _find_best_window(self) -> Dict[str, Any]:
        """Find the best performing window."""
        if not self.optimization_results:
            return {}
        
        best_result = max(
            self.optimization_results,
            key=lambda r: r.out_of_sample_performance.get('sharpe_ratio', -np.inf)
        )
        
        return {
            'window_id': best_result.window.window_id,
            'period': f"{best_result.window.test_start.date()} to {best_result.window.test_end.date()}",
            'sharpe_ratio': best_result.out_of_sample_performance.get('sharpe_ratio', 0),
            'parameters': best_result.best_params.to_dict()
        }
    
    def _find_worst_window(self) -> Dict[str, Any]:
        """Find the worst performing window."""
        if not self.optimization_results:
            return {}
        
        worst_result = min(
            self.optimization_results,
            key=lambda r: r.out_of_sample_performance.get('sharpe_ratio', np.inf)
        )
        
        return {
            'window_id': worst_result.window.window_id,
            'period': f"{worst_result.window.test_start.date()} to {worst_result.window.test_end.date()}",
            'sharpe_ratio': worst_result.out_of_sample_performance.get('sharpe_ratio', 0),
            'parameters': worst_result.best_params.to_dict()
        }
    
    def _find_consistent_parameters(self) -> Dict[str, Any]:
        """Find the most consistent parameter values."""
        if len(self.parameter_history) < 3:
            return {}
        
        # Count parameter value frequencies
        confluence_counts = {}
        position_counts = {}
        
        for hist in self.parameter_history:
            conf = round(hist['parameters']['confluence_threshold'], 2)
            pos = round(hist['parameters']['position_size'], 2)
            
            confluence_counts[conf] = confluence_counts.get(conf, 0) + 1
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        # Find most common values
        most_common_confluence = max(confluence_counts, key=confluence_counts.get)
        most_common_position = max(position_counts, key=position_counts.get)
        
        return {
            'confluence_threshold': most_common_confluence,
            'position_size': most_common_position,
            'confluence_frequency': confluence_counts[most_common_confluence] / len(self.parameter_history),
            'position_frequency': position_counts[most_common_position] / len(self.parameter_history)
        }
    
    def save_results(self, filepath: str):
        """
        Save optimization results to file.
        
        Args:
            filepath: Path to save results
        """
        results_data = {
            'summary': self.get_optimization_summary(),
            'parameter_history': self.parameter_history,
            'window_results': [
                {
                    'window_id': r.window.window_id,
                    'train_period': f"{r.window.train_start.date()} to {r.window.train_end.date()}",
                    'test_period': f"{r.window.test_start.date()} to {r.window.test_end.date()}",
                    'best_params': r.best_params.to_dict(),
                    'in_sample': r.in_sample_performance,
                    'out_of_sample': r.out_of_sample_performance,
                    'stability': r.parameter_stability,
                    'overfitting': r.overfitting_score
                }
                for r in self.optimization_results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Saved optimization results to {filepath}")