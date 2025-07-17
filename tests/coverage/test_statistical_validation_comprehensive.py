"""
Comprehensive test suite for Statistical Validation module.

This test suite aims for 100% coverage of the statistical_validation.py module
by testing all methods, edge cases, parallel processing, and error conditions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
import json
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scipy import stats
import warnings

# Import modules to test
from src.analysis.statistical_validation import (
    StatisticalValidator, BootstrapResult, MonteCarloResult
)


class TestStatisticalValidator:
    """Comprehensive tests for StatisticalValidator class"""
    
    @pytest.fixture
    def validator(self):
        """Create StatisticalValidator instance"""
        return StatisticalValidator(
            confidence_levels=[0.95, 0.99],
            n_bootstrap=100,  # Reduced for faster tests
            n_monte_carlo=1000,  # Reduced for faster tests
            min_samples_required=30
        )
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return series"""
        np.random.seed(42)
        # Generate returns with some structure
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        returns = np.random.normal(0.0005, 0.01, len(dates))
        return pd.Series(returns, index=dates)
    
    @pytest.fixture
    def benchmark_returns(self):
        """Create benchmark return series"""
        np.random.seed(43)
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        returns = np.random.normal(0.0003, 0.008, len(dates))
        return pd.Series(returns, index=dates)
    
    def test_initialization(self, validator):
        """Test StatisticalValidator initialization"""
        assert validator.confidence_levels == [0.95, 0.99]
        assert validator.n_bootstrap == 100
        assert validator.n_monte_carlo == 1000
        assert validator.min_samples_required == 30
        assert isinstance(validator.bootstrap_results, dict)
        assert isinstance(validator.monte_carlo_results, dict)
    
    def test_bootstrap_analysis_sequential(self, validator, sample_returns):
        """Test sequential bootstrap analysis"""
        results = validator.bootstrap_analysis(
            returns=sample_returns,
            metrics_to_test=['mean_return', 'sharpe_ratio', 'max_drawdown', 'var_95'],
            parallel=False
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        assert len(results) == 4
        
        for metric in ['mean_return', 'sharpe_ratio', 'max_drawdown', 'var_95']:
            assert metric in results
            result = results[metric]
            assert isinstance(result, BootstrapResult)
            assert result.metric == metric
            assert isinstance(result.original_value, float)
            assert isinstance(result.bootstrap_mean, float)
            assert isinstance(result.bootstrap_std, float)
            assert isinstance(result.confidence_intervals, dict)
            assert 0.95 in result.confidence_intervals
            assert 0.99 in result.confidence_intervals
            assert isinstance(result.p_value, float)
            assert isinstance(result.is_significant, (bool, np.bool_))
    
    def test_bootstrap_analysis_parallel(self, validator, sample_returns):
        """Test parallel bootstrap analysis"""
        results = validator.bootstrap_analysis(
            returns=sample_returns,
            metrics_to_test=['mean_return', 'sharpe_ratio'],
            parallel=True,
            n_jobs=2
        )
        
        assert isinstance(results, dict)
        assert len(results) == 2
        assert 'mean_return' in results
        assert 'sharpe_ratio' in results
    
    def test_bootstrap_analysis_insufficient_samples(self, validator):
        """Test bootstrap with insufficient samples"""
        short_returns = pd.Series([0.01, 0.02, -0.01])
        results = validator.bootstrap_analysis(returns=short_returns)
        assert len(results) == 0
    
    def test_bootstrap_analysis_custom_metrics(self, validator, sample_returns):
        """Test bootstrap with custom metrics"""
        custom_metrics = ['win_rate', 'skewness', 'kurtosis']
        results = validator.bootstrap_analysis(
            returns=sample_returns,
            metrics_to_test=custom_metrics
        )
        
        assert len(results) == 3
        for metric in custom_metrics:
            assert metric in results
    
    def test_calculate_metric_all_types(self, validator):
        """Test metric calculation for all supported metrics"""
        returns = np.random.normal(0.001, 0.01, 252)
        
        # Test all metrics
        metrics = {
            'mean_return': validator._calculate_metric(returns, 'mean_return'),
            'sharpe_ratio': validator._calculate_metric(returns, 'sharpe_ratio'),
            'max_drawdown': validator._calculate_metric(returns, 'max_drawdown'),
            'var_95': validator._calculate_metric(returns, 'var_95'),
            'win_rate': validator._calculate_metric(returns, 'win_rate'),
            'skewness': validator._calculate_metric(returns, 'skewness'),
            'kurtosis': validator._calculate_metric(returns, 'kurtosis'),
            'unknown': validator._calculate_metric(returns, 'unknown_metric')
        }
        
        # Verify all metrics return floats
        for metric_name, value in metrics.items():
            assert isinstance(value, float)
        
        # Test edge cases
        zero_returns = np.zeros(100)
        sharpe_zero = validator._calculate_metric(zero_returns, 'sharpe_ratio')
        assert sharpe_zero == 0.0
        
        # Test unknown metric
        assert metrics['unknown'] == 0.0
    
    def test_calculate_max_drawdown(self, validator):
        """Test maximum drawdown calculation"""
        # Test with positive trend
        returns = np.array([0.01, 0.02, -0.03, 0.01, -0.01])
        max_dd = validator._calculate_max_drawdown(returns)
        assert isinstance(max_dd, float)
        assert max_dd < 0
        
        # Test with all positive returns
        positive_returns = np.array([0.01, 0.02, 0.01, 0.02])
        max_dd_pos = validator._calculate_max_drawdown(positive_returns)
        assert max_dd_pos == 0.0
        
        # Test with all negative returns
        negative_returns = np.array([-0.01, -0.02, -0.01, -0.02])
        max_dd_neg = validator._calculate_max_drawdown(negative_returns)
        assert max_dd_neg < 0  # Should be negative
        assert max_dd_neg >= -0.06  # But not worse than cumulative loss
    
    def test_worker_bootstrap(self, validator):
        """Test worker bootstrap function"""
        returns = np.random.normal(0.001, 0.01, 252)
        values = validator._worker_bootstrap(returns, 'sharpe_ratio', 10)
        
        assert isinstance(values, list)
        assert len(values) == 10
        assert all(isinstance(v, float) for v in values)
    
    def test_monte_carlo_simulation_normal(self, validator, sample_returns):
        """Test Monte Carlo simulation with normal distribution"""
        results = validator.monte_carlo_simulation(
            returns=sample_returns,
            initial_capital=10000,
            time_horizon_days=252,
            metrics_to_simulate=['terminal_wealth', 'total_return', 'max_drawdown']
        )
        
        assert isinstance(results, dict)
        assert len(results) == 3
        
        for metric in ['terminal_wealth', 'total_return', 'max_drawdown']:
            assert metric in results
            result = results[metric]
            assert isinstance(result, MonteCarloResult)
            assert result.metric == metric
            assert isinstance(result.simulated_mean, float)
            assert isinstance(result.simulated_std, float)
            assert isinstance(result.percentile_5, float)
            assert isinstance(result.percentile_95, float)
            assert isinstance(result.probability_positive, float)
            assert isinstance(result.var_95, float)
            assert isinstance(result.cvar_95, float)
            
            # Verify percentiles make sense
            assert result.percentile_5 <= result.simulated_mean <= result.percentile_95
            assert 0 <= result.probability_positive <= 1
    
    def test_monte_carlo_simulation_fat_tails(self, validator):
        """Test Monte Carlo simulation with fat-tailed distribution"""
        # Create returns with high kurtosis
        np.random.seed(42)
        returns = np.concatenate([
            np.random.normal(0.001, 0.005, 200),
            np.random.normal(0.001, 0.03, 50)  # Add some extreme values
        ])
        
        results = validator.monte_carlo_simulation(
            returns=returns,
            initial_capital=10000,
            time_horizon_days=100,
            metrics_to_simulate=['terminal_wealth']
        )
        
        assert 'terminal_wealth' in results
        assert results['terminal_wealth'].simulated_std > 0
    
    def test_statistical_significance_test_paired_t(self, validator, sample_returns, 
                                                   benchmark_returns):
        """Test paired t-test for statistical significance"""
        results = validator.statistical_significance_test(
            strategy_returns=sample_returns,
            benchmark_returns=benchmark_returns,
            test_type='paired_t'
        )
        
        assert results['test_type'] == 'paired_t'
        assert 'n_observations' in results
        assert 'mean_excess_return' in results
        assert 'std_excess_return' in results
        assert 't_statistic' in results
        assert 'p_value' in results
        assert 'significant_at_95' in results
        assert 'significant_at_99' in results
        assert 'cohens_d' in results
        
        # Verify Cohen's d calculation
        assert isinstance(results['cohens_d'], float)
    
    def test_statistical_significance_test_wilcoxon(self, validator, sample_returns,
                                                   benchmark_returns):
        """Test Wilcoxon signed-rank test"""
        results = validator.statistical_significance_test(
            strategy_returns=sample_returns,
            benchmark_returns=benchmark_returns,
            test_type='wilcoxon'
        )
        
        assert results['test_type'] == 'wilcoxon'
        assert 'wilcoxon_statistic' in results
        assert 'p_value' in results
    
    def test_statistical_significance_test_mann_whitney(self, validator, sample_returns,
                                                       benchmark_returns):
        """Test Mann-Whitney U test"""
        results = validator.statistical_significance_test(
            strategy_returns=sample_returns,
            benchmark_returns=benchmark_returns,
            test_type='mann_whitney'
        )
        
        assert results['test_type'] == 'mann_whitney'
        assert 'mann_whitney_statistic' in results
        assert 'p_value' in results
    
    def test_statistical_significance_test_edge_cases(self, validator):
        """Test statistical significance with edge cases"""
        # Test with different length series
        strategy = pd.Series([0.01, 0.02, 0.01, 0.02, 0.01])
        benchmark = pd.Series([0.005, 0.01, 0.005])
        
        results = validator.statistical_significance_test(
            strategy_returns=strategy,
            benchmark_returns=benchmark
        )
        assert results['n_observations'] == 3
        
        # Test with zero variance
        zero_strategy = pd.Series(np.zeros(100))
        zero_benchmark = pd.Series(np.zeros(100))
        
        results_zero = validator.statistical_significance_test(
            strategy_returns=zero_strategy,
            benchmark_returns=zero_benchmark
        )
        assert results_zero['cohens_d'] == 0.0
    
    def test_rolling_statistics(self, validator, sample_returns):
        """Test rolling statistics calculation"""
        rolling_stats = validator.rolling_statistics(
            returns=sample_returns,
            window=60,
            metrics=['mean', 'std', 'sharpe', 'skew', 'kurtosis']
        )
        
        assert isinstance(rolling_stats, pd.DataFrame)
        assert 'rolling_mean' in rolling_stats.columns
        assert 'rolling_std' in rolling_stats.columns
        assert 'rolling_sharpe' in rolling_stats.columns
        assert 'rolling_skew' in rolling_stats.columns
        assert 'rolling_kurtosis' in rolling_stats.columns
        
        # Check that rolling values start as NaN
        assert rolling_stats.iloc[:59].isna().all().all()
        assert not rolling_stats.iloc[60:].isna().all().all()
        
        # Test with numpy array input
        returns_array = sample_returns.values
        rolling_stats_array = validator.rolling_statistics(
            returns=returns_array,
            window=30,
            metrics=['mean', 'std']
        )
        assert isinstance(rolling_stats_array, pd.DataFrame)
    
    def test_calculate_information_coefficient(self, validator):
        """Test Information Coefficient calculation"""
        # Create predictions and returns with some correlation
        np.random.seed(42)
        predictions = np.random.normal(0, 1, 100)
        actual_returns = predictions * 0.3 + np.random.normal(0, 0.5, 100)
        
        ic_results = validator.calculate_information_coefficient(
            predictions=predictions,
            actual_returns=actual_returns
        )
        
        assert 'ic_pearson' in ic_results
        assert 'ic_pearson_pvalue' in ic_results
        assert 'ic_spearman' in ic_results
        assert 'ic_spearman_pvalue' in ic_results
        assert 'hit_rate' in ic_results
        assert 'ic_significant' in ic_results
        assert 'predictive_power' in ic_results
        
        # IC should be positive given the correlation
        assert ic_results['ic_pearson'] > 0
        assert 0 <= ic_results['hit_rate'] <= 1
        
        # Test with different length inputs
        short_predictions = predictions[:50]
        ic_results_short = validator.calculate_information_coefficient(
            predictions=short_predictions,
            actual_returns=actual_returns
        )
        assert ic_results_short['ic_pearson'] != ic_results['ic_pearson']
    
    def test_robustness_test(self, validator, sample_returns):
        """Test strategy robustness to perturbations"""
        robustness = validator.robustness_test(
            returns=sample_returns,
            perturbation_std=0.001,
            n_perturbations=50
        )
        
        assert 'original_sharpe' in robustness
        assert 'mean_perturbed_sharpe' in robustness
        assert 'std_perturbed_sharpe' in robustness
        assert 'min_perturbed_sharpe' in robustness
        assert 'max_perturbed_sharpe' in robustness
        assert 'robustness_score' in robustness
        assert 'stable_performance' in robustness
        
        # Robustness score should be between 0 and 1
        assert 0 <= robustness['robustness_score'] <= 1
        
        # Test with zero returns
        zero_returns = pd.Series(np.zeros(100))
        robustness_zero = validator.robustness_test(returns=zero_returns)
        assert robustness_zero['robustness_score'] == 0
    
    def test_generate_validation_report(self, validator, sample_returns, benchmark_returns):
        """Test validation report generation"""
        # First run some analyses to populate results
        validator.bootstrap_analysis(
            returns=sample_returns,
            metrics_to_test=['mean_return', 'sharpe_ratio']
        )
        
        validator.monte_carlo_simulation(
            returns=sample_returns,
            metrics_to_simulate=['terminal_wealth', 'total_return']
        )
        
        report = validator.generate_validation_report()
        
        assert isinstance(report, dict)
        assert 'bootstrap_analysis' in report
        assert 'monte_carlo_analysis' in report
        assert 'summary' in report
        
        # Check bootstrap analysis
        bootstrap = report['bootstrap_analysis']
        assert 'mean_return' in bootstrap
        assert 'sharpe_ratio' in bootstrap
        
        # Check monte carlo analysis
        monte_carlo = report['monte_carlo_analysis']
        assert 'terminal_wealth' in monte_carlo
        assert 'total_return' in monte_carlo
        
        # Check summary
        summary = report['summary']
        assert 'significant_metrics' in summary
        assert 'total_metrics_tested' in summary
        assert 'significance_rate' in summary
        assert 'positive_outcome_metrics' in summary
        assert 'monte_carlo_confidence' in summary
    
    def test_generate_validation_report_empty(self, validator):
        """Test validation report with no results"""
        report = validator.generate_validation_report()
        
        assert report['bootstrap_analysis'] == {}
        assert report['monte_carlo_analysis'] == {}
        assert 'significant_metrics' not in report['summary']
    
    def test_save_results(self, validator, sample_returns):
        """Test saving results to file"""
        # Run some analysis
        validator.bootstrap_analysis(
            returns=sample_returns,
            metrics_to_test=['mean_return']
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            validator.save_results(temp_path)
            
            # Verify file exists and can be loaded
            assert Path(temp_path).exists()
            
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert 'bootstrap_analysis' in loaded_data
            assert 'mean_return' in loaded_data['bootstrap_analysis']
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
    
    def test_parallel_bootstrap_edge_cases(self, validator, sample_returns):
        """Test parallel bootstrap with edge cases"""
        # Test with n_jobs = -1 (use all cores)
        results = validator.bootstrap_analysis(
            returns=sample_returns,
            metrics_to_test=['mean_return'],
            parallel=True,
            n_jobs=-1
        )
        assert 'mean_return' in results
        
        # Test with n_jobs = 0 (should use default)
        results = validator.bootstrap_analysis(
            returns=sample_returns,
            metrics_to_test=['sharpe_ratio'],
            parallel=True,
            n_jobs=0
        )
        assert 'sharpe_ratio' in results
    
    def test_confidence_intervals(self, validator, sample_returns):
        """Test confidence interval calculations"""
        results = validator.bootstrap_analysis(
            returns=sample_returns,
            metrics_to_test=['mean_return'],
            parallel=False
        )
        
        result = results['mean_return']
        ci_95 = result.confidence_intervals[0.95]
        ci_99 = result.confidence_intervals[0.99]
        
        # 99% CI should be wider than 95% CI
        assert (ci_99[1] - ci_99[0]) >= (ci_95[1] - ci_95[0])
        
        # Original value should typically be within confidence intervals
        # (not guaranteed but very likely with enough samples)
        assert ci_99[0] <= result.bootstrap_mean <= ci_99[1]
    
    def test_p_value_calculations(self, validator):
        """Test p-value calculations for different metrics"""
        # Create returns with positive mean for one-sided test
        positive_returns = np.random.normal(0.002, 0.01, 252)
        
        results = validator.bootstrap_analysis(
            returns=positive_returns,
            metrics_to_test=['mean_return', 'sharpe_ratio', 'max_drawdown']
        )
        
        # For positive metrics, p-value should be low if significantly positive
        assert results['mean_return'].p_value < 0.5
        assert results['sharpe_ratio'].p_value < 0.5
        
        # Create returns with negative mean
        negative_returns = np.random.normal(-0.002, 0.01, 252)
        
        results_neg = validator.bootstrap_analysis(
            returns=negative_returns,
            metrics_to_test=['mean_return']
        )
        
        # P-value should be high for negative returns
        assert results_neg['mean_return'].p_value > 0.5


def test_dataclass_properties():
    """Test dataclass properties and methods"""
    # Test BootstrapResult
    bootstrap_result = BootstrapResult(
        metric='sharpe_ratio',
        original_value=1.2,
        bootstrap_mean=1.18,
        bootstrap_std=0.15,
        confidence_intervals={0.95: (0.9, 1.5), 0.99: (0.8, 1.6)},
        p_value=0.03,
        is_significant=True
    )
    
    assert bootstrap_result.metric == 'sharpe_ratio'
    assert bootstrap_result.original_value == 1.2
    assert bootstrap_result.is_significant is True
    assert len(bootstrap_result.confidence_intervals) == 2
    
    # Test MonteCarloResult
    mc_result = MonteCarloResult(
        metric='terminal_wealth',
        simulated_mean=15000,
        simulated_std=2000,
        percentile_5=11000,
        percentile_95=19000,
        probability_positive=0.95,
        var_95=11000,
        cvar_95=10500
    )
    
    assert mc_result.metric == 'terminal_wealth'
    assert mc_result.simulated_mean == 15000
    assert mc_result.probability_positive == 0.95
    assert mc_result.cvar_95 < mc_result.var_95  # CVaR should be worse than VaR