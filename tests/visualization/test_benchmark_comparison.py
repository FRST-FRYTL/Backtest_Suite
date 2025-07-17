"""
Comprehensive tests for the BenchmarkComparison class.

This module provides complete test coverage for benchmark comparison functionality
including relative performance, risk metrics, and statistical analysis.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.visualization.benchmark_comparison import BenchmarkComparison


class TestBenchmarkComparison:
    """Comprehensive tests for BenchmarkComparison class."""
    
    @pytest.fixture
    def benchmark_comparison(self):
        """Create BenchmarkComparison instance."""
        return BenchmarkComparison()
    
    @pytest.fixture
    def sample_portfolio_returns(self):
        """Create sample portfolio returns."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Generate returns with some autocorrelation and positive drift
        returns = np.random.normal(0.0008, 0.02, len(dates))
        for i in range(1, len(returns)):
            returns[i] = 0.2 * returns[i-1] + 0.8 * returns[i]
        
        return pd.Series(returns, index=dates, name='portfolio')
    
    @pytest.fixture
    def sample_benchmark_returns(self):
        """Create sample benchmark returns."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(43)
        
        # Generate benchmark returns with lower volatility
        returns = np.random.normal(0.0005, 0.015, len(dates))
        
        return pd.Series(returns, index=dates, name='benchmark')
    
    @pytest.fixture
    def multiple_benchmarks(self):
        """Create multiple benchmark returns."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        benchmarks = pd.DataFrame(index=dates)
        
        # S&P 500
        np.random.seed(44)
        benchmarks['SP500'] = np.random.normal(0.0004, 0.012, len(dates))
        
        # NASDAQ
        np.random.seed(45)
        benchmarks['NASDAQ'] = np.random.normal(0.0006, 0.018, len(dates))
        
        # Bond Index
        np.random.seed(46)
        benchmarks['BONDS'] = np.random.normal(0.0002, 0.006, len(dates))
        
        return benchmarks
    
    def test_comparison_initialization(self, benchmark_comparison):
        """Test BenchmarkComparison initialization."""
        assert isinstance(benchmark_comparison, BenchmarkComparison)
        assert hasattr(benchmark_comparison, 'compare')
    
    def test_basic_comparison(self, benchmark_comparison, sample_portfolio_returns, sample_benchmark_returns):
        """Test basic portfolio vs benchmark comparison."""
        result = benchmark_comparison.compare(
            portfolio_returns=sample_portfolio_returns,
            benchmark_returns=sample_benchmark_returns
        )
        
        assert isinstance(result, dict)
        assert 'metrics' in result
        assert 'charts' in result
        assert 'analysis' in result
    
    def test_performance_metrics_calculation(self, benchmark_comparison, 
                                           sample_portfolio_returns, sample_benchmark_returns):
        """Test performance metrics calculation."""
        metrics = benchmark_comparison._calculate_comparison_metrics(
            sample_portfolio_returns,
            sample_benchmark_returns
        )
        
        assert isinstance(metrics, dict)
        
        # Check for key metrics
        expected_metrics = [
            'portfolio_total_return',
            'benchmark_total_return',
            'excess_return',
            'tracking_error',
            'information_ratio',
            'alpha',
            'beta',
            'correlation',
            'r_squared'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_create_performance_chart(self, benchmark_comparison,
                                     sample_portfolio_returns, sample_benchmark_returns):
        """Test performance comparison chart creation."""
        fig = benchmark_comparison.create_performance_chart(
            portfolio_returns=sample_portfolio_returns,
            benchmark_returns=sample_benchmark_returns
        )
        
        assert isinstance(fig, go.Figure)
        
        # Should have at least portfolio and benchmark traces
        assert len(fig.data) >= 2
        
        # Check for cumulative returns
        trace_names = [trace.name for trace in fig.data if hasattr(trace, 'name')]
        assert any('portfolio' in name.lower() for name in trace_names)
        assert any('benchmark' in name.lower() for name in trace_names)
    
    def test_create_relative_performance_chart(self, benchmark_comparison,
                                              sample_portfolio_returns, sample_benchmark_returns):
        """Test relative performance chart creation."""
        fig = benchmark_comparison.create_relative_performance_chart(
            portfolio_returns=sample_portfolio_returns,
            benchmark_returns=sample_benchmark_returns
        )
        
        assert isinstance(fig, go.Figure)
        
        # Should show excess returns or relative performance
        assert len(fig.data) >= 1
    
    def test_create_risk_comparison_chart(self, benchmark_comparison,
                                         sample_portfolio_returns, sample_benchmark_returns):
        """Test risk comparison visualization."""
        fig = benchmark_comparison.create_risk_comparison_chart(
            portfolio_returns=sample_portfolio_returns,
            benchmark_returns=sample_benchmark_returns
        )
        
        assert isinstance(fig, go.Figure)
        
        # Should compare risk metrics
        assert len(fig.data) > 0
    
    def test_rolling_beta_calculation(self, benchmark_comparison,
                                     sample_portfolio_returns, sample_benchmark_returns):
        """Test rolling beta calculation."""
        rolling_beta = benchmark_comparison._calculate_rolling_beta(
            portfolio_returns=sample_portfolio_returns,
            benchmark_returns=sample_benchmark_returns,
            window=60
        )
        
        assert isinstance(rolling_beta, pd.Series)
        assert len(rolling_beta) == len(sample_portfolio_returns)
        
        # Check that early values are NaN (due to window)
        assert rolling_beta[:59].isna().all()
        assert not rolling_beta[60:].isna().all()
    
    def test_create_rolling_metrics_chart(self, benchmark_comparison,
                                         sample_portfolio_returns, sample_benchmark_returns):
        """Test rolling metrics visualization."""
        fig = benchmark_comparison.create_rolling_metrics_chart(
            portfolio_returns=sample_portfolio_returns,
            benchmark_returns=sample_benchmark_returns,
            metrics=['beta', 'correlation', 'tracking_error'],
            window=30
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # One trace per metric
    
    def test_drawdown_comparison(self, benchmark_comparison,
                                sample_portfolio_returns, sample_benchmark_returns):
        """Test drawdown comparison."""
        fig = benchmark_comparison.create_drawdown_comparison_chart(
            portfolio_returns=sample_portfolio_returns,
            benchmark_returns=sample_benchmark_returns
        )
        
        assert isinstance(fig, go.Figure)
        
        # Should have portfolio and benchmark drawdown traces
        assert len(fig.data) >= 2
        
        # Verify drawdowns are negative
        for trace in fig.data:
            if hasattr(trace, 'y'):
                assert all(val <= 0 for val in trace.y if not pd.isna(val))
    
    def test_scatter_plot_comparison(self, benchmark_comparison,
                                    sample_portfolio_returns, sample_benchmark_returns):
        """Test scatter plot comparison."""
        fig = benchmark_comparison.create_scatter_comparison(
            portfolio_returns=sample_portfolio_returns,
            benchmark_returns=sample_benchmark_returns
        )
        
        assert isinstance(fig, go.Figure)
        
        # Should have scatter trace and regression line
        assert len(fig.data) >= 2
        
        # Check for regression line
        assert any(trace.mode == 'lines' for trace in fig.data if hasattr(trace, 'mode'))
    
    def test_multiple_benchmarks_comparison(self, benchmark_comparison,
                                          sample_portfolio_returns, multiple_benchmarks):
        """Test comparison against multiple benchmarks."""
        result = benchmark_comparison.compare_multiple_benchmarks(
            portfolio_returns=sample_portfolio_returns,
            benchmarks=multiple_benchmarks
        )
        
        assert isinstance(result, dict)
        assert len(result) == len(multiple_benchmarks.columns)
        
        # Check each benchmark comparison
        for benchmark_name in multiple_benchmarks.columns:
            assert benchmark_name in result
            assert 'metrics' in result[benchmark_name]
            assert 'alpha' in result[benchmark_name]['metrics']
            assert 'beta' in result[benchmark_name]['metrics']
    
    def test_create_benchmark_table(self, benchmark_comparison,
                                   sample_portfolio_returns, multiple_benchmarks):
        """Test benchmark comparison table creation."""
        fig = benchmark_comparison.create_benchmark_comparison_table(
            portfolio_returns=sample_portfolio_returns,
            benchmarks=multiple_benchmarks
        )
        
        assert isinstance(fig, go.Figure)
        
        # Should have table trace
        assert any(isinstance(trace, go.Table) for trace in fig.data)
    
    def test_period_analysis(self, benchmark_comparison,
                            sample_portfolio_returns, sample_benchmark_returns):
        """Test period-based analysis."""
        analysis = benchmark_comparison.analyze_by_period(
            portfolio_returns=sample_portfolio_returns,
            benchmark_returns=sample_benchmark_returns,
            periods=['monthly', 'quarterly', 'yearly']
        )
        
        assert isinstance(analysis, dict)
        assert 'monthly' in analysis
        assert 'quarterly' in analysis
        assert 'yearly' in analysis
        
        # Check monthly analysis
        monthly = analysis['monthly']
        assert isinstance(monthly, pd.DataFrame)
        assert 'portfolio' in monthly.columns
        assert 'benchmark' in monthly.columns
        assert 'excess' in monthly.columns
    
    def test_up_down_capture_ratios(self, benchmark_comparison,
                                   sample_portfolio_returns, sample_benchmark_returns):
        """Test up/down capture ratio calculation."""
        capture_ratios = benchmark_comparison._calculate_capture_ratios(
            portfolio_returns=sample_portfolio_returns,
            benchmark_returns=sample_benchmark_returns
        )
        
        assert isinstance(capture_ratios, dict)
        assert 'up_capture' in capture_ratios
        assert 'down_capture' in capture_ratios
        assert 'capture_ratio' in capture_ratios
        
        # Ratios should be positive
        assert capture_ratios['up_capture'] >= 0
        assert capture_ratios['down_capture'] >= 0
    
    def test_create_capture_ratio_chart(self, benchmark_comparison,
                                       sample_portfolio_returns, sample_benchmark_returns):
        """Test capture ratio visualization."""
        fig = benchmark_comparison.create_capture_ratio_chart(
            portfolio_returns=sample_portfolio_returns,
            benchmark_returns=sample_benchmark_returns
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_attribution_analysis(self, benchmark_comparison,
                                 sample_portfolio_returns, sample_benchmark_returns):
        """Test performance attribution analysis."""
        attribution = benchmark_comparison.perform_attribution_analysis(
            portfolio_returns=sample_portfolio_returns,
            benchmark_returns=sample_benchmark_returns,
            risk_free_rate=0.02
        )
        
        assert isinstance(attribution, dict)
        assert 'selection' in attribution
        assert 'timing' in attribution
        assert 'total_excess' in attribution
    
    def test_style_analysis(self, benchmark_comparison,
                           sample_portfolio_returns, multiple_benchmarks):
        """Test style analysis (factor-based)."""
        style_weights = benchmark_comparison.perform_style_analysis(
            portfolio_returns=sample_portfolio_returns,
            factor_returns=multiple_benchmarks
        )
        
        assert isinstance(style_weights, dict)
        
        # Weights should sum to approximately 1
        total_weight = sum(style_weights.values())
        assert 0.9 <= total_weight <= 1.1
    
    def test_conditional_performance(self, benchmark_comparison,
                                    sample_portfolio_returns, sample_benchmark_returns):
        """Test conditional performance analysis."""
        # Analyze performance in different market conditions
        conditional = benchmark_comparison.analyze_conditional_performance(
            portfolio_returns=sample_portfolio_returns,
            benchmark_returns=sample_benchmark_returns,
            conditions={
                'bull_market': lambda x: x > 0,
                'bear_market': lambda x: x < 0,
                'high_volatility': lambda x: abs(x) > x.std()
            }
        )
        
        assert isinstance(conditional, dict)
        assert 'bull_market' in conditional
        assert 'bear_market' in conditional
        assert 'high_volatility' in conditional
    
    def test_save_comparison_report(self, benchmark_comparison,
                                   sample_portfolio_returns, sample_benchmark_returns):
        """Test saving comparison report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'comparison_report.html')
            
            benchmark_comparison.save_comparison_report(
                portfolio_returns=sample_portfolio_returns,
                benchmark_returns=sample_benchmark_returns,
                output_path=output_path
            )
            
            assert os.path.exists(output_path)
            
            # Verify content
            with open(output_path, 'r') as f:
                content = f.read()
                assert 'Benchmark Comparison' in content
    
    def test_empty_data_handling(self, benchmark_comparison):
        """Test handling of empty data."""
        empty_returns = pd.Series([])
        
        with pytest.raises((ValueError, IndexError)):
            benchmark_comparison.compare(empty_returns, empty_returns)
    
    def test_mismatched_dates_handling(self, benchmark_comparison,
                                      sample_portfolio_returns):
        """Test handling of mismatched date ranges."""
        # Create benchmark with different dates
        different_dates = pd.date_range('2023-02-01', periods=200, freq='D')
        benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, len(different_dates)),
            index=different_dates
        )
        
        # Should handle date alignment
        result = benchmark_comparison.compare(
            portfolio_returns=sample_portfolio_returns,
            benchmark_returns=benchmark_returns
        )
        
        assert isinstance(result, dict)
        assert 'metrics' in result