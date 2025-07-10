"""
Statistical Validation and Bootstrap Analysis Framework

This module provides comprehensive statistical validation including
bootstrap analysis, significance testing, and Monte Carlo simulations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from scipy import stats
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BootstrapResult:
    """Results from bootstrap analysis"""
    metric: str
    original_value: float
    bootstrap_mean: float
    bootstrap_std: float
    confidence_intervals: Dict[float, Tuple[float, float]]
    p_value: float
    is_significant: bool

@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation"""
    metric: str
    simulated_mean: float
    simulated_std: float
    percentile_5: float
    percentile_95: float
    probability_positive: float
    var_95: float
    cvar_95: float

class StatisticalValidator:
    """
    Comprehensive statistical validation for trading strategies.
    """
    
    def __init__(
        self,
        confidence_levels: List[float] = [0.95, 0.99],
        n_bootstrap: int = 1000,
        n_monte_carlo: int = 10000,
        min_samples_required: int = 30
    ):
        """
        Initialize the statistical validator.
        
        Args:
            confidence_levels: Confidence levels for intervals
            n_bootstrap: Number of bootstrap samples
            n_monte_carlo: Number of Monte Carlo simulations
            min_samples_required: Minimum samples for valid statistics
        """
        self.confidence_levels = confidence_levels
        self.n_bootstrap = n_bootstrap
        self.n_monte_carlo = n_monte_carlo
        self.min_samples_required = min_samples_required
        
        # Results storage
        self.bootstrap_results: Dict[str, BootstrapResult] = {}
        self.monte_carlo_results: Dict[str, MonteCarloResult] = {}
        
    def bootstrap_analysis(
        self,
        returns: Union[pd.Series, np.ndarray],
        metrics_to_test: Optional[List[str]] = None,
        parallel: bool = True,
        n_jobs: int = -1
    ) -> Dict[str, BootstrapResult]:
        """
        Perform bootstrap analysis on returns.
        
        Args:
            returns: Return series to analyze
            metrics_to_test: List of metrics to calculate
            parallel: Whether to use parallel processing
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary of bootstrap results by metric
        """
        if metrics_to_test is None:
            metrics_to_test = ['mean_return', 'sharpe_ratio', 'max_drawdown', 'var_95']
        
        # Convert to numpy array
        returns_array = np.array(returns)
        
        if len(returns_array) < self.min_samples_required:
            logger.warning(f"Insufficient samples ({len(returns_array)}) for bootstrap analysis")
            return {}
        
        results = {}
        
        for metric in metrics_to_test:
            logger.info(f"Bootstrapping {metric}...")
            
            # Calculate original metric value
            original_value = self._calculate_metric(returns_array, metric)
            
            # Perform bootstrap sampling
            if parallel and n_jobs != 1:
                bootstrap_values = self._parallel_bootstrap(
                    returns_array, metric, n_jobs
                )
            else:
                bootstrap_values = self._sequential_bootstrap(
                    returns_array, metric
                )
            
            # Calculate statistics
            bootstrap_mean = np.mean(bootstrap_values)
            bootstrap_std = np.std(bootstrap_values)
            
            # Calculate confidence intervals
            confidence_intervals = {}
            for conf_level in self.confidence_levels:
                lower_percentile = (1 - conf_level) / 2 * 100
                upper_percentile = (1 + conf_level) / 2 * 100
                
                ci_lower = np.percentile(bootstrap_values, lower_percentile)
                ci_upper = np.percentile(bootstrap_values, upper_percentile)
                
                confidence_intervals[conf_level] = (ci_lower, ci_upper)
            
            # Calculate p-value (test if significantly different from zero)
            if metric in ['mean_return', 'sharpe_ratio']:
                # One-sided test (positive is good)
                p_value = (bootstrap_values <= 0).sum() / len(bootstrap_values)
            else:
                # Two-sided test
                p_value = 2 * min(
                    (bootstrap_values <= 0).sum() / len(bootstrap_values),
                    (bootstrap_values >= 0).sum() / len(bootstrap_values)
                )
            
            # Determine significance
            is_significant = p_value < (1 - self.confidence_levels[0])
            
            result = BootstrapResult(
                metric=metric,
                original_value=original_value,
                bootstrap_mean=bootstrap_mean,
                bootstrap_std=bootstrap_std,
                confidence_intervals=confidence_intervals,
                p_value=p_value,
                is_significant=is_significant
            )
            
            results[metric] = result
            self.bootstrap_results[metric] = result
        
        return results
    
    def _calculate_metric(self, returns: np.ndarray, metric: str) -> float:
        """Calculate a specific metric from returns."""
        if metric == 'mean_return':
            return np.mean(returns)
        elif metric == 'sharpe_ratio':
            if np.std(returns) > 0:
                return np.mean(returns) / np.std(returns) * np.sqrt(252)
            return 0.0
        elif metric == 'max_drawdown':
            return self._calculate_max_drawdown(returns)
        elif metric == 'var_95':
            return np.percentile(returns, 5)
        elif metric == 'win_rate':
            return (returns > 0).mean()
        elif metric == 'skewness':
            return stats.skew(returns)
        elif metric == 'kurtosis':
            return stats.kurtosis(returns)
        else:
            logger.warning(f"Unknown metric: {metric}")
            return 0.0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative / rolling_max - 1)
        return drawdown.min()
    
    def _sequential_bootstrap(
        self,
        returns: np.ndarray,
        metric: str
    ) -> np.ndarray:
        """Perform sequential bootstrap sampling."""
        bootstrap_values = []
        
        for _ in range(self.n_bootstrap):
            # Sample with replacement
            bootstrap_sample = np.random.choice(
                returns, size=len(returns), replace=True
            )
            
            # Calculate metric
            metric_value = self._calculate_metric(bootstrap_sample, metric)
            bootstrap_values.append(metric_value)
        
        return np.array(bootstrap_values)
    
    def _parallel_bootstrap(
        self,
        returns: np.ndarray,
        metric: str,
        n_jobs: int
    ) -> np.ndarray:
        """Perform parallel bootstrap sampling."""
        n_workers = n_jobs if n_jobs > 0 else None
        
        # Split bootstrap iterations across workers
        iterations_per_worker = self.n_bootstrap // (n_workers or 4)
        
        bootstrap_values = []
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            
            for _ in range(n_workers or 4):
                future = executor.submit(
                    self._worker_bootstrap,
                    returns,
                    metric,
                    iterations_per_worker
                )
                futures.append(future)
            
            for future in as_completed(futures):
                worker_values = future.result()
                bootstrap_values.extend(worker_values)
        
        return np.array(bootstrap_values[:self.n_bootstrap])
    
    def _worker_bootstrap(
        self,
        returns: np.ndarray,
        metric: str,
        n_iterations: int
    ) -> List[float]:
        """Worker function for parallel bootstrap."""
        values = []
        
        for _ in range(n_iterations):
            bootstrap_sample = np.random.choice(
                returns, size=len(returns), replace=True
            )
            metric_value = self._calculate_metric(bootstrap_sample, metric)
            values.append(metric_value)
        
        return values
    
    def monte_carlo_simulation(
        self,
        returns: Union[pd.Series, np.ndarray],
        initial_capital: float = 10000,
        time_horizon_days: int = 252,
        metrics_to_simulate: Optional[List[str]] = None
    ) -> Dict[str, MonteCarloResult]:
        """
        Perform Monte Carlo simulation for strategy returns.
        
        Args:
            returns: Historical returns to base simulation on
            initial_capital: Starting capital
            time_horizon_days: Simulation time horizon
            metrics_to_simulate: Metrics to calculate
            
        Returns:
            Dictionary of Monte Carlo results
        """
        if metrics_to_simulate is None:
            metrics_to_simulate = ['terminal_wealth', 'total_return', 'max_drawdown']
        
        returns_array = np.array(returns)
        
        # Calculate return distribution parameters
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        # Account for fat tails using t-distribution if kurtosis is high
        kurtosis = stats.kurtosis(returns_array)
        use_t_dist = kurtosis > 3  # Leptokurtic distribution
        
        results = {}
        
        for metric in metrics_to_simulate:
            logger.info(f"Monte Carlo simulation for {metric}...")
            
            simulated_values = []
            
            for _ in range(self.n_monte_carlo):
                # Generate random returns
                if use_t_dist:
                    # Use t-distribution for fat tails
                    df = max(4, 30 - kurtosis)  # Degrees of freedom
                    random_returns = stats.t.rvs(
                        df, loc=mean_return, scale=std_return,
                        size=time_horizon_days
                    )
                else:
                    # Use normal distribution
                    random_returns = np.random.normal(
                        mean_return, std_return, time_horizon_days
                    )
                
                # Calculate metric for this simulation
                if metric == 'terminal_wealth':
                    terminal_value = initial_capital * (1 + random_returns).prod()
                    simulated_values.append(terminal_value)
                elif metric == 'total_return':
                    total_return = (1 + random_returns).prod() - 1
                    simulated_values.append(total_return)
                elif metric == 'max_drawdown':
                    max_dd = self._calculate_max_drawdown(random_returns)
                    simulated_values.append(max_dd)
            
            simulated_values = np.array(simulated_values)
            
            # Calculate statistics
            result = MonteCarloResult(
                metric=metric,
                simulated_mean=np.mean(simulated_values),
                simulated_std=np.std(simulated_values),
                percentile_5=np.percentile(simulated_values, 5),
                percentile_95=np.percentile(simulated_values, 95),
                probability_positive=(simulated_values > 0).mean(),
                var_95=np.percentile(simulated_values, 5),
                cvar_95=np.mean(simulated_values[simulated_values <= np.percentile(simulated_values, 5)])
            )
            
            results[metric] = result
            self.monte_carlo_results[metric] = result
        
        return results
    
    def statistical_significance_test(
        self,
        strategy_returns: Union[pd.Series, np.ndarray],
        benchmark_returns: Union[pd.Series, np.ndarray],
        test_type: str = 'paired_t'
    ) -> Dict[str, Any]:
        """
        Test statistical significance of strategy vs benchmark.
        
        Args:
            strategy_returns: Strategy return series
            benchmark_returns: Benchmark return series
            test_type: Type of test ('paired_t', 'wilcoxon', 'mann_whitney')
            
        Returns:
            Test results dictionary
        """
        strategy_array = np.array(strategy_returns)
        benchmark_array = np.array(benchmark_returns)
        
        # Ensure same length
        min_length = min(len(strategy_array), len(benchmark_array))
        strategy_array = strategy_array[:min_length]
        benchmark_array = benchmark_array[:min_length]
        
        excess_returns = strategy_array - benchmark_array
        
        results = {
            'test_type': test_type,
            'n_observations': len(excess_returns),
            'mean_excess_return': np.mean(excess_returns),
            'std_excess_return': np.std(excess_returns)
        }
        
        # Perform appropriate test
        if test_type == 'paired_t':
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(strategy_array, benchmark_array)
            results['t_statistic'] = t_stat
            results['p_value'] = p_value
            
        elif test_type == 'wilcoxon':
            # Wilcoxon signed-rank test (non-parametric)
            statistic, p_value = stats.wilcoxon(excess_returns)
            results['wilcoxon_statistic'] = statistic
            results['p_value'] = p_value
            
        elif test_type == 'mann_whitney':
            # Mann-Whitney U test (independent samples)
            statistic, p_value = stats.mannwhitneyu(
                strategy_array, benchmark_array, alternative='two-sided'
            )
            results['mann_whitney_statistic'] = statistic
            results['p_value'] = p_value
        
        # Determine significance at different levels
        results['significant_at_95'] = results['p_value'] < 0.05
        results['significant_at_99'] = results['p_value'] < 0.01
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(strategy_array) + np.var(benchmark_array)) / 2)
        if pooled_std > 0:
            results['cohens_d'] = (np.mean(strategy_array) - np.mean(benchmark_array)) / pooled_std
        else:
            results['cohens_d'] = 0.0
        
        return results
    
    def rolling_statistics(
        self,
        returns: Union[pd.Series, np.ndarray],
        window: int = 252,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling statistical metrics.
        
        Args:
            returns: Return series
            window: Rolling window size
            metrics: Metrics to calculate
            
        Returns:
            DataFrame with rolling statistics
        """
        if metrics is None:
            metrics = ['mean', 'std', 'sharpe', 'skew', 'kurtosis']
        
        returns_series = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        
        rolling_stats = pd.DataFrame(index=returns_series.index)
        
        for metric in metrics:
            if metric == 'mean':
                rolling_stats['rolling_mean'] = returns_series.rolling(window).mean()
            elif metric == 'std':
                rolling_stats['rolling_std'] = returns_series.rolling(window).std()
            elif metric == 'sharpe':
                rolling_mean = returns_series.rolling(window).mean()
                rolling_std = returns_series.rolling(window).std()
                rolling_stats['rolling_sharpe'] = (rolling_mean / rolling_std) * np.sqrt(252)
            elif metric == 'skew':
                rolling_stats['rolling_skew'] = returns_series.rolling(window).skew()
            elif metric == 'kurtosis':
                rolling_stats['rolling_kurtosis'] = returns_series.rolling(window).kurt()
        
        return rolling_stats
    
    def calculate_information_coefficient(
        self,
        predictions: Union[pd.Series, np.ndarray],
        actual_returns: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate Information Coefficient (IC) for predictions.
        
        Args:
            predictions: Predicted values/signals
            actual_returns: Actual returns
            
        Returns:
            Dictionary with IC metrics
        """
        predictions_array = np.array(predictions)
        returns_array = np.array(actual_returns)
        
        # Ensure same length
        min_length = min(len(predictions_array), len(returns_array))
        predictions_array = predictions_array[:min_length]
        returns_array = returns_array[:min_length]
        
        # Calculate Pearson correlation (IC)
        ic_pearson, p_value_pearson = stats.pearsonr(predictions_array, returns_array)
        
        # Calculate Spearman rank correlation (Rank IC)
        ic_spearman, p_value_spearman = stats.spearmanr(predictions_array, returns_array)
        
        # Calculate hit rate (directional accuracy)
        predicted_direction = np.sign(predictions_array)
        actual_direction = np.sign(returns_array)
        hit_rate = (predicted_direction == actual_direction).mean()
        
        return {
            'ic_pearson': ic_pearson,
            'ic_pearson_pvalue': p_value_pearson,
            'ic_spearman': ic_spearman,
            'ic_spearman_pvalue': p_value_spearman,
            'hit_rate': hit_rate,
            'ic_significant': p_value_pearson < 0.05,
            'predictive_power': abs(ic_pearson) > 0.03  # Common threshold
        }
    
    def robustness_test(
        self,
        returns: Union[pd.Series, np.ndarray],
        perturbation_std: float = 0.001,
        n_perturbations: int = 100
    ) -> Dict[str, Any]:
        """
        Test strategy robustness to small perturbations.
        
        Args:
            returns: Original return series
            perturbation_std: Standard deviation of perturbations
            n_perturbations: Number of perturbation tests
            
        Returns:
            Robustness test results
        """
        returns_array = np.array(returns)
        original_sharpe = self._calculate_metric(returns_array, 'sharpe_ratio')
        
        perturbed_sharpes = []
        
        for _ in range(n_perturbations):
            # Add small random perturbations
            noise = np.random.normal(0, perturbation_std, len(returns_array))
            perturbed_returns = returns_array + noise
            
            # Calculate metric on perturbed data
            perturbed_sharpe = self._calculate_metric(perturbed_returns, 'sharpe_ratio')
            perturbed_sharpes.append(perturbed_sharpe)
        
        perturbed_sharpes = np.array(perturbed_sharpes)
        
        # Calculate robustness metrics
        return {
            'original_sharpe': original_sharpe,
            'mean_perturbed_sharpe': np.mean(perturbed_sharpes),
            'std_perturbed_sharpe': np.std(perturbed_sharpes),
            'min_perturbed_sharpe': np.min(perturbed_sharpes),
            'max_perturbed_sharpe': np.max(perturbed_sharpes),
            'robustness_score': 1 - (np.std(perturbed_sharpes) / abs(original_sharpe)) if original_sharpe != 0 else 0,
            'stable_performance': np.std(perturbed_sharpes) < 0.1 * abs(original_sharpe)
        }
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Returns:
            Validation report dictionary
        """
        report = {
            'bootstrap_analysis': {},
            'monte_carlo_analysis': {},
            'summary': {}
        }
        
        # Bootstrap results
        for metric, result in self.bootstrap_results.items():
            report['bootstrap_analysis'][metric] = {
                'original_value': result.original_value,
                'bootstrap_mean': result.bootstrap_mean,
                'bootstrap_std': result.bootstrap_std,
                'confidence_intervals': result.confidence_intervals,
                'p_value': result.p_value,
                'is_significant': result.is_significant
            }
        
        # Monte Carlo results
        for metric, result in self.monte_carlo_results.items():
            report['monte_carlo_analysis'][metric] = {
                'simulated_mean': result.simulated_mean,
                'simulated_std': result.simulated_std,
                'percentile_5': result.percentile_5,
                'percentile_95': result.percentile_95,
                'probability_positive': result.probability_positive,
                'var_95': result.var_95,
                'cvar_95': result.cvar_95
            }
        
        # Summary statistics
        if self.bootstrap_results:
            significant_metrics = sum(
                1 for r in self.bootstrap_results.values() if r.is_significant
            )
            report['summary']['significant_metrics'] = significant_metrics
            report['summary']['total_metrics_tested'] = len(self.bootstrap_results)
            report['summary']['significance_rate'] = significant_metrics / len(self.bootstrap_results)
        
        if self.monte_carlo_results:
            positive_outcomes = sum(
                1 for r in self.monte_carlo_results.values() 
                if r.probability_positive > 0.5
            )
            report['summary']['positive_outcome_metrics'] = positive_outcomes
            report['summary']['monte_carlo_confidence'] = positive_outcomes / len(self.monte_carlo_results)
        
        return report
    
    def save_results(self, filepath: str):
        """Save validation results to file."""
        report = self.generate_validation_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved validation results to {filepath}")