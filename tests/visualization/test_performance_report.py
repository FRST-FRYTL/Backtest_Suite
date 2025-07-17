"""
Comprehensive tests for the PerformanceAnalysisReport class.

This module provides complete test coverage for performance analysis reporting
including metrics calculation, visualization, and statistical analysis.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

from src.visualization.performance_report import PerformanceAnalysisReport


class TestPerformanceAnalysisReport:
    """Comprehensive tests for PerformanceAnalysisReport class."""
    
    @pytest.fixture
    def report_generator(self):
        """Create PerformanceAnalysisReport instance."""
        return PerformanceAnalysisReport()
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns series."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Generate realistic returns with some autocorrelation
        returns = np.random.normal(0.0005, 0.02, len(dates))
        # Add some momentum
        for i in range(1, len(returns)):
            returns[i] = 0.3 * returns[i-1] + 0.7 * returns[i]
        
        return pd.Series(returns, index=dates, name='returns')
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades with detailed metrics."""
        np.random.seed(42)
        
        trades_data = []
        base_date = datetime(2023, 1, 1)
        
        for i in range(50):
            entry_date = base_date + timedelta(days=i*5)
            exit_date = entry_date + timedelta(days=np.random.randint(1, 10))
            
            entry_price = 100 + np.random.uniform(-20, 20)
            exit_price = entry_price * (1 + np.random.uniform(-0.1, 0.1))
            quantity = np.random.randint(10, 100)
            
            pnl = (exit_price - entry_price) * quantity
            
            trades_data.append({
                'trade_id': i + 1,
                'entry_time': entry_date,
                'exit_time': exit_date,
                'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN']),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'profit_loss': pnl,
                'return_pct': ((exit_price - entry_price) / entry_price) * 100,
                'holding_period': (exit_date - entry_date).days,
                'commission': 2.0,
                'mae': np.random.uniform(0, abs(pnl) * 0.5),  # Maximum adverse excursion
                'mfe': np.random.uniform(abs(pnl) * 0.5, abs(pnl) * 1.5)  # Maximum favorable excursion
            })
        
        return pd.DataFrame(trades_data)
    
    @pytest.fixture
    def sample_portfolio_data(self, sample_returns):
        """Create sample portfolio data."""
        initial_capital = 100000
        
        # Generate portfolio values
        cumulative_returns = (1 + sample_returns).cumprod()
        portfolio_value = initial_capital * cumulative_returns
        
        # Generate positions
        positions = pd.DataFrame(index=sample_returns.index)
        positions['AAPL'] = np.random.randint(-50, 100, len(sample_returns.index))
        positions['GOOGL'] = np.random.randint(-30, 80, len(sample_returns.index))
        positions['MSFT'] = np.random.randint(-40, 90, len(sample_returns.index))
        
        # Calculate exposures
        prices = {
            'AAPL': 150 + np.random.randn(len(sample_returns.index)).cumsum(),
            'GOOGL': 2800 + 10 * np.random.randn(len(sample_returns.index)).cumsum(),
            'MSFT': 350 + 2 * np.random.randn(len(sample_returns.index)).cumsum()
        }
        
        exposure = pd.DataFrame(index=sample_returns.index)
        for symbol in positions.columns:
            exposure[symbol] = positions[symbol] * prices[symbol]
        
        return {
            'portfolio_value': portfolio_value,
            'returns': sample_returns,
            'positions': positions,
            'exposure': exposure,
            'cash': initial_capital * 0.2 * np.ones(len(sample_returns.index))
        }
    
    def test_report_initialization(self, report_generator):
        """Test PerformanceAnalysisReport initialization."""
        assert isinstance(report_generator, PerformanceAnalysisReport)
        assert hasattr(report_generator, 'generate_report')
    
    def test_generate_basic_report(self, report_generator, sample_returns, sample_trades):
        """Test basic report generation."""
        report = report_generator.generate_report(
            returns=sample_returns,
            trades=sample_trades
        )
        
        assert isinstance(report, dict)
        assert 'metrics' in report
        assert 'charts' in report
        assert 'analysis' in report
    
    def test_calculate_performance_metrics(self, report_generator, sample_returns):
        """Test performance metrics calculation."""
        metrics = report_generator._calculate_performance_metrics(sample_returns)
        
        assert isinstance(metrics, dict)
        
        # Check key metrics
        expected_metrics = [
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'max_drawdown', 'var_95', 'cvar_95'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_calculate_trade_statistics(self, report_generator, sample_trades):
        """Test trade statistics calculation."""
        stats = report_generator._calculate_trade_statistics(sample_trades)
        
        assert isinstance(stats, dict)
        
        # Check trade statistics
        expected_stats = [
            'total_trades', 'winning_trades', 'losing_trades',
            'win_rate', 'profit_factor', 'avg_win', 'avg_loss',
            'largest_win', 'largest_loss', 'avg_holding_period'
        ]
        
        for stat in expected_stats:
            assert stat in stats
    
    def test_create_returns_analysis_chart(self, report_generator, sample_returns):
        """Test returns analysis chart creation."""
        fig = report_generator._create_returns_analysis_chart(sample_returns)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        
        # Should have multiple subplots (returns, distribution, etc.)
        assert hasattr(fig, '_grid_ref')
    
    def test_create_drawdown_analysis_chart(self, report_generator, sample_returns):
        """Test drawdown analysis chart creation."""
        cumulative_returns = (1 + sample_returns).cumprod()
        
        fig = report_generator._create_drawdown_analysis_chart(cumulative_returns)
        
        assert isinstance(fig, go.Figure)
        
        # Verify drawdown data
        drawdown_trace = fig.data[0]
        assert all(val <= 0 for val in drawdown_trace.y if not pd.isna(val))
    
    def test_create_trade_analysis_charts(self, report_generator, sample_trades):
        """Test trade analysis charts creation."""
        charts = report_generator._create_trade_analysis_charts(sample_trades)
        
        assert isinstance(charts, list)
        assert len(charts) > 0
        
        for chart in charts:
            assert isinstance(chart, go.Figure)
    
    def test_create_risk_metrics_chart(self, report_generator, sample_returns):
        """Test risk metrics visualization."""
        fig = report_generator._create_risk_metrics_chart(sample_returns)
        
        assert isinstance(fig, go.Figure)
        
        # Should include VaR, CVaR visualization
        fig_str = str(fig)
        assert 'var' in fig_str.lower() or 'risk' in fig_str.lower()
    
    def test_rolling_statistics(self, report_generator, sample_returns):
        """Test rolling statistics calculation and visualization."""
        rolling_stats = report_generator._calculate_rolling_statistics(
            sample_returns,
            window=30
        )
        
        assert isinstance(rolling_stats, pd.DataFrame)
        assert 'rolling_return' in rolling_stats.columns
        assert 'rolling_volatility' in rolling_stats.columns
        assert 'rolling_sharpe' in rolling_stats.columns
        
        # Test visualization
        fig = report_generator._create_rolling_statistics_chart(rolling_stats)
        assert isinstance(fig, go.Figure)
    
    def test_monthly_returns_heatmap(self, report_generator, sample_returns):
        """Test monthly returns heatmap creation."""
        fig = report_generator._create_monthly_returns_heatmap(sample_returns)
        
        assert isinstance(fig, go.Figure)
        
        # Should have heatmap trace
        assert any(
            hasattr(trace, 'type') and trace.type == 'heatmap'
            for trace in fig.data
        )
    
    def test_trade_efficiency_analysis(self, report_generator, sample_trades):
        """Test trade efficiency analysis (MAE/MFE)."""
        if 'mae' in sample_trades.columns and 'mfe' in sample_trades.columns:
            analysis = report_generator._analyze_trade_efficiency(sample_trades)
            
            assert isinstance(analysis, dict)
            assert 'avg_efficiency' in analysis
            assert 'edge_ratio' in analysis
            
            # Test visualization
            fig = report_generator._create_mae_mfe_chart(sample_trades)
            assert isinstance(fig, go.Figure)
    
    def test_win_loss_streaks_analysis(self, report_generator, sample_trades):
        """Test win/loss streaks analysis."""
        streaks = report_generator._analyze_win_loss_streaks(sample_trades)
        
        assert isinstance(streaks, dict)
        assert 'max_winning_streak' in streaks
        assert 'max_losing_streak' in streaks
        assert 'current_streak' in streaks
    
    def test_position_sizing_analysis(self, report_generator, sample_trades):
        """Test position sizing analysis."""
        analysis = report_generator._analyze_position_sizing(sample_trades)
        
        assert isinstance(analysis, dict)
        assert 'avg_position_size' in analysis
        assert 'position_size_variance' in analysis
        
        # Test visualization
        fig = report_generator._create_position_sizing_chart(sample_trades)
        assert isinstance(fig, go.Figure)
    
    def test_time_based_analysis(self, report_generator, sample_trades):
        """Test time-based performance analysis."""
        # Hour of day analysis
        hourly_analysis = report_generator._analyze_performance_by_hour(sample_trades)
        assert isinstance(hourly_analysis, pd.DataFrame)
        
        # Day of week analysis
        daily_analysis = report_generator._analyze_performance_by_day(sample_trades)
        assert isinstance(daily_analysis, pd.DataFrame)
        
        # Monthly analysis
        monthly_analysis = report_generator._analyze_performance_by_month(sample_trades)
        assert isinstance(monthly_analysis, pd.DataFrame)
    
    def test_correlation_analysis(self, report_generator, sample_portfolio_data):
        """Test correlation analysis between positions."""
        correlations = report_generator._calculate_position_correlations(
            sample_portfolio_data['positions']
        )
        
        assert isinstance(correlations, pd.DataFrame)
        assert correlations.shape[0] == correlations.shape[1]
        
        # Test visualization
        fig = report_generator._create_correlation_heatmap(correlations)
        assert isinstance(fig, go.Figure)
    
    def test_exposure_analysis(self, report_generator, sample_portfolio_data):
        """Test exposure analysis."""
        exposure_stats = report_generator._analyze_exposure(
            sample_portfolio_data['exposure'],
            sample_portfolio_data['portfolio_value']
        )
        
        assert isinstance(exposure_stats, dict)
        assert 'gross_exposure' in exposure_stats
        assert 'net_exposure' in exposure_stats
        assert 'concentration' in exposure_stats
    
    def test_risk_adjusted_metrics(self, report_generator, sample_returns):
        """Test calculation of risk-adjusted metrics."""
        risk_free_rate = 0.02
        
        metrics = report_generator._calculate_risk_adjusted_metrics(
            sample_returns,
            risk_free_rate=risk_free_rate
        )
        
        assert isinstance(metrics, dict)
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'calmar_ratio' in metrics
        assert 'information_ratio' in metrics
    
    def test_performance_attribution(self, report_generator, sample_portfolio_data):
        """Test performance attribution analysis."""
        attribution = report_generator._calculate_performance_attribution(
            sample_portfolio_data['returns'],
            sample_portfolio_data['positions'],
            sample_portfolio_data['exposure']
        )
        
        assert isinstance(attribution, dict)
        assert 'selection_effect' in attribution
        assert 'allocation_effect' in attribution
    
    def test_save_report(self, report_generator, sample_returns, sample_trades):
        """Test report saving functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_report.html')
            
            report_generator.save_report(
                returns=sample_returns,
                trades=sample_trades,
                output_path=output_path
            )
            
            assert os.path.exists(output_path)
            
            # Verify content
            with open(output_path, 'r') as f:
                content = f.read()
                assert 'Performance Analysis Report' in content
    
    def test_export_metrics_to_csv(self, report_generator, sample_returns, sample_trades):
        """Test metrics export to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'metrics.csv')
            
            report_generator.export_metrics_to_csv(
                returns=sample_returns,
                trades=sample_trades,
                output_path=csv_path
            )
            
            assert os.path.exists(csv_path)
            
            # Verify content
            df = pd.read_csv(csv_path)
            assert len(df) > 0
    
    def test_benchmark_comparison(self, report_generator, sample_returns):
        """Test benchmark comparison functionality."""
        # Create benchmark returns
        benchmark_returns = sample_returns * 0.7 + np.random.normal(0, 0.005, len(sample_returns))
        
        comparison = report_generator._compare_to_benchmark(
            sample_returns,
            benchmark_returns
        )
        
        assert isinstance(comparison, dict)
        assert 'alpha' in comparison
        assert 'beta' in comparison
        assert 'tracking_error' in comparison
        assert 'information_ratio' in comparison
    
    def test_empty_data_handling(self, report_generator):
        """Test handling of empty data."""
        empty_returns = pd.Series([])
        empty_trades = pd.DataFrame()
        
        # Should handle gracefully
        with pytest.raises((ValueError, IndexError)):
            report_generator.generate_report(empty_returns, empty_trades)
    
    def test_custom_metrics(self, report_generator, sample_returns):
        """Test custom metrics calculation."""
        def custom_metric(returns):
            return returns.mean() / returns.std() * np.sqrt(252)
        
        custom_metrics = {
            'custom_sharpe': custom_metric
        }
        
        metrics = report_generator._calculate_performance_metrics(
            sample_returns,
            custom_metrics=custom_metrics
        )
        
        assert 'custom_sharpe' in metrics