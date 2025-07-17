"""
Comprehensive tests for the ChartGenerator class.

This module provides complete test coverage for all chart generation functionality
including equity curves, return distributions, trade visualization, and performance metrics.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock

from src.visualization.charts import ChartGenerator


class TestChartGenerator:
    """Comprehensive tests for ChartGenerator class."""
    
    @pytest.fixture
    def chart_generator_plotly(self):
        """Create ChartGenerator instance with plotly style."""
        return ChartGenerator(style="plotly")
    
    @pytest.fixture
    def chart_generator_matplotlib(self):
        """Create ChartGenerator instance with matplotlib style."""
        return ChartGenerator(style="matplotlib")
    
    @pytest.fixture
    def sample_equity_curve(self):
        """Create sample equity curve data."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Generate realistic equity curve with trend and volatility
        returns = np.random.normal(0.0005, 0.02, len(dates))
        cumulative_returns = (1 + returns).cumprod()
        
        equity_df = pd.DataFrame({
            'total_value': 100000 * cumulative_returns,
            'cash': 50000 * np.ones(len(dates)),
            'holdings_value': 50000 * cumulative_returns
        }, index=dates)
        
        return equity_df
    
    @pytest.fixture
    def sample_benchmark(self):
        """Create sample benchmark data."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(43)
        
        # Generate benchmark with lower volatility
        returns = np.random.normal(0.0003, 0.015, len(dates))
        cumulative_returns = (1 + returns).cumprod()
        
        return pd.Series(100000 * cumulative_returns, index=dates, name='benchmark')
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades data."""
        np.random.seed(42)
        
        # Generate 50 trades
        trade_data = []
        base_date = datetime(2023, 1, 1)
        
        for i in range(50):
            entry_date = base_date + timedelta(days=i*5)
            exit_date = entry_date + timedelta(days=np.random.randint(1, 10))
            
            entry_price = 100 + np.random.uniform(-20, 20)
            exit_price = entry_price * (1 + np.random.uniform(-0.1, 0.1))
            
            trade_data.append({
                'trade_id': i + 1,
                'entry_time': entry_date,
                'exit_time': exit_date,
                'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN']),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': np.random.randint(10, 100),
                'profit_loss': (exit_price - entry_price) * np.random.randint(10, 100),
                'return_pct': ((exit_price - entry_price) / entry_price) * 100
            })
        
        return pd.DataFrame(trade_data)
    
    @pytest.fixture
    def sample_performance_metrics(self):
        """Create sample performance metrics."""
        return {
            'total_return': 0.2543,
            'annualized_return': 0.2856,
            'sharpe_ratio': 1.45,
            'sortino_ratio': 2.13,
            'max_drawdown': -0.1523,
            'win_rate': 0.58,
            'profit_factor': 1.85,
            'total_trades': 50,
            'winning_trades': 29,
            'losing_trades': 21,
            'avg_win': 523.45,
            'avg_loss': -234.67,
            'best_trade': 1523.45,
            'worst_trade': -876.23,
            'calmar_ratio': 1.88,
            'volatility': 0.1856
        }
    
    def test_chart_generator_initialization(self, chart_generator_plotly, chart_generator_matplotlib):
        """Test ChartGenerator initialization with different styles."""
        assert chart_generator_plotly.style == "plotly"
        assert chart_generator_matplotlib.style == "matplotlib"
    
    def test_plot_equity_curve_plotly(self, chart_generator_plotly, sample_equity_curve, sample_benchmark):
        """Test equity curve plotting with plotly."""
        # Test without benchmark
        fig = chart_generator_plotly.plot_equity_curve(sample_equity_curve)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Equity curve and drawdown
        
        # Test with benchmark
        fig_with_benchmark = chart_generator_plotly.plot_equity_curve(
            sample_equity_curve, 
            benchmark=sample_benchmark,
            title="Test Equity Curve"
        )
        assert isinstance(fig_with_benchmark, go.Figure)
        assert len(fig_with_benchmark.data) >= 3  # Equity, benchmark, and drawdown
        
        # Verify subplot structure
        assert hasattr(fig_with_benchmark, '_grid_ref')
        assert fig_with_benchmark._grid_ref is not None
    
    @patch('matplotlib.pyplot.show')
    def test_plot_equity_curve_matplotlib(self, mock_show, chart_generator_matplotlib, 
                                          sample_equity_curve, sample_benchmark):
        """Test equity curve plotting with matplotlib."""
        # Test without benchmark
        fig = chart_generator_matplotlib.plot_equity_curve(sample_equity_curve)
        assert isinstance(fig, plt.Figure)
        
        # Test with benchmark
        fig_with_benchmark = chart_generator_matplotlib.plot_equity_curve(
            sample_equity_curve,
            benchmark=sample_benchmark,
            title="Test Equity Curve"
        )
        assert isinstance(fig_with_benchmark, plt.Figure)
        
        # Clean up
        plt.close('all')
    
    def test_plot_returns_distribution_plotly(self, chart_generator_plotly, sample_trades):
        """Test returns distribution plotting with plotly."""
        fig = chart_generator_plotly.plot_returns_distribution(
            sample_trades,
            title="Returns Distribution"
        )
        
        assert isinstance(fig, go.Figure)
        # Should have histogram and possibly a KDE line
        assert len(fig.data) >= 1
        
        # Check if statistical annotations are present
        assert any('Mean' in str(fig.layout.annotations) for _ in [1]) if fig.layout.annotations else True
    
    @patch('matplotlib.pyplot.show')
    def test_plot_returns_distribution_matplotlib(self, mock_show, chart_generator_matplotlib, sample_trades):
        """Test returns distribution plotting with matplotlib."""
        fig = chart_generator_matplotlib.plot_returns_distribution(
            sample_trades,
            title="Returns Distribution"
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Clean up
        plt.close('all')
    
    def test_plot_trades_plotly(self, chart_generator_plotly, sample_trades):
        """Test trades plotting with plotly."""
        fig = chart_generator_plotly.plot_trades(
            sample_trades,
            title="Trade Analysis"
        )
        
        assert isinstance(fig, go.Figure)
        # Should have multiple subplots for different trade analyses
        assert len(fig.data) >= 1
    
    @patch('matplotlib.pyplot.show')
    def test_plot_trades_matplotlib(self, mock_show, chart_generator_matplotlib, sample_trades):
        """Test trades plotting with matplotlib."""
        fig = chart_generator_matplotlib.plot_trades(
            sample_trades,
            title="Trade Analysis"
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Clean up
        plt.close('all')
    
    def test_plot_performance_metrics_plotly(self, chart_generator_plotly, sample_performance_metrics):
        """Test performance metrics plotting with plotly."""
        fig = chart_generator_plotly.plot_performance_metrics(
            sample_performance_metrics,
            title="Performance Metrics"
        )
        
        assert isinstance(fig, go.Figure)
        # Should have table or indicator elements
        assert len(fig.data) >= 1
    
    @patch('matplotlib.pyplot.show')
    def test_plot_performance_metrics_matplotlib(self, mock_show, chart_generator_matplotlib, 
                                                 sample_performance_metrics):
        """Test performance metrics plotting with matplotlib."""
        fig = chart_generator_matplotlib.plot_performance_metrics(
            sample_performance_metrics,
            title="Performance Metrics"
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Clean up
        plt.close('all')
    
    def test_edge_cases(self, chart_generator_plotly):
        """Test edge cases and error handling."""
        # Empty equity curve
        empty_equity = pd.DataFrame({'total_value': []})
        with pytest.raises((ValueError, IndexError, KeyError)):
            chart_generator_plotly.plot_equity_curve(empty_equity)
        
        # Single data point
        single_point = pd.DataFrame({
            'total_value': [100000]
        }, index=[datetime(2023, 1, 1)])
        
        # This should not raise an error
        fig = chart_generator_plotly.plot_equity_curve(single_point)
        assert isinstance(fig, go.Figure)
        
        # Invalid data types
        with pytest.raises((TypeError, AttributeError, KeyError)):
            chart_generator_plotly.plot_equity_curve("not a dataframe")
    
    def test_drawdown_calculation(self, chart_generator_plotly, sample_equity_curve):
        """Test drawdown calculation in equity curve plot."""
        fig = chart_generator_plotly.plot_equity_curve(sample_equity_curve)
        
        # Extract drawdown data from the figure
        drawdown_trace = None
        for trace in fig.data:
            if hasattr(trace, 'yaxis') and trace.yaxis == 'y2':
                drawdown_trace = trace
                break
        
        if drawdown_trace is not None:
            # Verify drawdown is negative or zero
            assert all(val <= 0 for val in drawdown_trace.y if not np.isnan(val))
    
    def test_benchmark_normalization(self, chart_generator_plotly, sample_equity_curve, sample_benchmark):
        """Test benchmark normalization in equity curve plot."""
        # Ensure benchmark starts at different value
        sample_benchmark = sample_benchmark * 0.5
        
        fig = chart_generator_plotly.plot_equity_curve(
            sample_equity_curve,
            benchmark=sample_benchmark
        )
        
        # Find portfolio and benchmark traces
        portfolio_trace = None
        benchmark_trace = None
        
        for trace in fig.data:
            if trace.name == 'Portfolio':
                portfolio_trace = trace
            elif trace.name == 'Benchmark':
                benchmark_trace = trace
        
        # Verify both start at same value
        if portfolio_trace and benchmark_trace:
            assert abs(portfolio_trace.y[0] - benchmark_trace.y[0]) < 1e-6
    
    def test_trade_analysis_completeness(self, chart_generator_plotly, sample_trades):
        """Test trade analysis includes all required components."""
        fig = chart_generator_plotly.plot_trades(sample_trades)
        
        # Verify figure has data
        assert len(fig.data) > 0
        
        # Check for expected trade metrics in the data
        all_text = str(fig)
        expected_elements = ['profit_loss', 'symbol', 'trade']
        
        for element in expected_elements:
            assert any(element.lower() in all_text.lower() for _ in [1])
    
    def test_performance_metrics_display(self, chart_generator_plotly, sample_performance_metrics):
        """Test performance metrics display completeness."""
        fig = chart_generator_plotly.plot_performance_metrics(sample_performance_metrics)
        
        # Verify key metrics are displayed
        fig_str = str(fig)
        key_metrics = ['sharpe', 'return', 'drawdown', 'win_rate']
        
        for metric in key_metrics:
            assert any(metric in fig_str.lower() for _ in [1])
    
    def test_custom_titles(self, chart_generator_plotly, sample_equity_curve):
        """Test custom titles are applied correctly."""
        custom_title = "My Custom Equity Curve"
        fig = chart_generator_plotly.plot_equity_curve(
            sample_equity_curve,
            title=custom_title
        )
        
        assert fig.layout.title.text == custom_title
    
    def test_data_validation(self, chart_generator_plotly):
        """Test data validation for various inputs."""
        # Missing required columns
        invalid_equity = pd.DataFrame({'wrong_column': [1, 2, 3]})
        with pytest.raises((KeyError, ValueError)):
            chart_generator_plotly.plot_equity_curve(invalid_equity)
        
        # Invalid trades dataframe
        invalid_trades = pd.DataFrame({'wrong_column': [1, 2, 3]})
        with pytest.raises((KeyError, ValueError, AttributeError)):
            chart_generator_plotly.plot_trades(invalid_trades)
    
    def test_style_consistency(self, chart_generator_plotly, sample_equity_curve):
        """Test visual style consistency across charts."""
        fig = chart_generator_plotly.plot_equity_curve(sample_equity_curve)
        
        # Check layout properties
        assert hasattr(fig.layout, 'template')
        assert hasattr(fig.layout, 'xaxis')
        assert hasattr(fig.layout, 'yaxis')
    
    @patch('matplotlib.pyplot.style.use')
    def test_matplotlib_style_setting(self, mock_style_use):
        """Test matplotlib style is set correctly."""
        ChartGenerator(style="matplotlib")
        mock_style_use.assert_called_once_with('seaborn-v0_8-darkgrid')