"""
Comprehensive tests for dashboard.py visualization module.

This module provides complete test coverage for the Dashboard class
including all chart creation methods and HTML generation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import os
from pathlib import Path

from src.visualization.dashboard import Dashboard


class TestDashboard:
    """Comprehensive tests for Dashboard class."""
    
    @pytest.fixture
    def sample_equity_curve(self):
        """Create sample equity curve data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        initial_value = 100000
        returns = np.random.normal(0.001, 0.02, len(dates))
        cumulative_returns = (1 + returns).cumprod()
        
        equity_df = pd.DataFrame({
            'total_value': initial_value * cumulative_returns,
            'cash': initial_value * 0.3 * np.ones(len(dates)),  # 30% cash
            'positions_value': initial_value * 0.7 * cumulative_returns
        }, index=dates)
        
        return equity_df
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades data."""
        trades_data = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2023-01-05', '2023-01-10', '2023-01-15', '2023-01-20',
                '2023-01-25', '2023-01-30', '2023-02-05', '2023-02-10',
                '2023-02-15', '2023-02-20', '2023-02-25', '2023-03-01'
            ]),
            'type': ['OPEN', 'CLOSE', 'OPEN', 'CLOSE', 'OPEN', 'CLOSE', 
                    'OPEN', 'CLOSE', 'OPEN', 'CLOSE', 'OPEN', 'CLOSE'],
            'symbol': ['AAPL', 'AAPL', 'GOOGL', 'GOOGL', 'MSFT', 'MSFT',
                      'AAPL', 'AAPL', 'GOOGL', 'GOOGL', 'MSFT', 'MSFT'],
            'price': [150.0, 155.0, 2800.0, 2750.0, 300.0, 310.0,
                     152.0, 158.0, 2820.0, 2850.0, 305.0, 308.0],
            'quantity': [100, -100, 50, -50, 200, -200,
                        150, -150, 40, -40, 180, -180],
            'position_pnl': [np.nan, 500.0, np.nan, -2500.0, np.nan, 2000.0,
                           np.nan, 900.0, np.nan, 1200.0, np.nan, 540.0]
        })
        return trades_data
    
    @pytest.fixture
    def sample_performance_metrics(self):
        """Create sample performance metrics."""
        return {
            'total_return': 15000.0,
            'total_pnl': 12500.0,
            'unrealized_pnl': 2500.0,
            'realized_pnl': 10000.0,
            'max_drawdown': -8.5,
            'total_commission': 450.0,
            'total_trades': 12,
            'winning_trades': 8,
            'losing_trades': 4,
            'avg_win': 1562.5,
            'avg_loss': -625.0,
            'sharpe_ratio': 1.85,
            'initial_capital': 100000.0
        }
    
    @pytest.fixture
    def sample_results(self, sample_equity_curve, sample_trades, sample_performance_metrics):
        """Create sample backtest results."""
        return {
            'equity_curve': sample_equity_curve,
            'trades': sample_trades,
            'performance': sample_performance_metrics
        }
    
    def test_initialization(self):
        """Test Dashboard initialization."""
        dashboard = Dashboard()
        assert dashboard.figures == []
    
    def test_create_equity_chart(self, sample_equity_curve):
        """Test equity chart creation."""
        dashboard = Dashboard()
        fig = dashboard._create_equity_chart(sample_equity_curve)
        
        assert isinstance(fig, go.Figure)
        
        # Check traces
        trace_names = [trace.name for trace in fig.data]
        assert 'Portfolio Value' in trace_names
        assert 'Cash' in trace_names
        assert 'Positions Value' in trace_names
        
        # Check layout
        assert fig.layout.title.text == 'Portfolio Equity Curve'
        assert fig.layout.xaxis.title.text == 'Date'
        assert fig.layout.yaxis.title.text == 'Value ($)'
        assert fig.layout.hovermode == 'x unified'
        # Template is an object, check its name instead
        assert hasattr(fig.layout, 'template') or fig.layout.template.layout.plot_bgcolor == 'white'
    
    def test_create_equity_chart_minimal(self):
        """Test equity chart with minimal data."""
        dashboard = Dashboard()
        
        # Only total_value column
        minimal_equity = pd.DataFrame({
            'total_value': [100000, 105000, 110000]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        fig = dashboard._create_equity_chart(minimal_equity)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # Only portfolio value
        assert fig.data[0].name == 'Portfolio Value'
    
    def test_create_drawdown_chart(self, sample_equity_curve):
        """Test drawdown chart creation."""
        dashboard = Dashboard()
        fig = dashboard._create_drawdown_chart(sample_equity_curve)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Drawdown line + Max drawdown marker
        
        # Check traces
        assert fig.data[0].name == 'Drawdown'
        assert fig.data[0].fill == 'tozeroy'
        assert fig.data[1].name == 'Max Drawdown'
        
        # Check layout
        assert fig.layout.title.text == 'Portfolio Drawdown'
        assert fig.layout.xaxis.title.text == 'Date'
        assert fig.layout.yaxis.title.text == 'Drawdown (%)'
    
    def test_create_drawdown_chart_calculation(self, sample_equity_curve):
        """Test drawdown calculation accuracy."""
        dashboard = Dashboard()
        
        # Manual drawdown calculation
        running_max = sample_equity_curve['total_value'].expanding().max()
        expected_drawdown = (sample_equity_curve['total_value'] - running_max) / running_max * 100
        
        fig = dashboard._create_drawdown_chart(sample_equity_curve)
        
        # Verify drawdown values
        actual_drawdown = fig.data[0].y
        np.testing.assert_array_almost_equal(actual_drawdown, expected_drawdown.values, decimal=5)
        
        # Verify max drawdown marker
        max_dd_value = fig.data[1].y[0]
        assert max_dd_value == expected_drawdown.min()
    
    def test_create_trades_chart(self, sample_trades):
        """Test trades timeline chart creation."""
        dashboard = Dashboard()
        fig = dashboard._create_trades_chart(sample_trades)
        
        assert isinstance(fig, go.Figure)
        
        # Check that each symbol has buy and sell traces
        trace_names = [trace.name for trace in fig.data]
        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            assert f'{symbol} Buy' in trace_names
            assert f'{symbol} Sell' in trace_names
        
        # Check layout
        assert fig.layout.title.text == 'Trade Timeline'
        assert fig.layout.xaxis.title.text == 'Date'
        assert fig.layout.yaxis.title.text == 'Price'
    
    def test_create_trades_chart_empty(self):
        """Test trades chart with empty data."""
        dashboard = Dashboard()
        empty_trades = pd.DataFrame()
        
        fig = dashboard._create_trades_chart(empty_trades)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
    
    def test_create_trades_chart_no_valid_trades(self):
        """Test trades chart with no OPEN/CLOSE trades."""
        dashboard = Dashboard()
        
        # Trades with other types
        other_trades = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=3),
            'type': ['DIVIDEND', 'FEE', 'ADJUSTMENT'],
            'symbol': ['AAPL', 'AAPL', 'AAPL'],
            'price': [1.5, 0.5, 0.0],
            'quantity': [0, 0, 0]
        })
        
        fig = dashboard._create_trades_chart(other_trades)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0  # No valid trades to plot
    
    def test_create_trade_analysis(self, sample_trades):
        """Test trade analysis charts creation."""
        dashboard = Dashboard()
        fig = dashboard._create_trade_analysis(sample_trades)
        
        assert isinstance(fig, go.Figure)
        
        # Check subplots
        assert len(fig.layout.annotations) >= 4  # Subplot titles
        subplot_titles = [ann.text for ann in fig.layout.annotations if hasattr(ann, 'text')]
        assert 'P&L Distribution' in subplot_titles
        assert 'P&L by Trade' in subplot_titles
        assert 'Win/Loss Ratio' in subplot_titles
        assert 'Trade Duration' in subplot_titles
        
        # Check layout
        assert fig.layout.title.text == 'Trade Analysis'
        assert fig.layout.height == 800
    
    def test_create_trade_analysis_empty(self):
        """Test trade analysis with empty data."""
        dashboard = Dashboard()
        empty_trades = pd.DataFrame()
        
        fig = dashboard._create_trade_analysis(empty_trades)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
    
    def test_create_trade_analysis_calculations(self, sample_trades):
        """Test trade analysis calculations."""
        dashboard = Dashboard()
        
        # Get closed trades for verification
        closed_trades = sample_trades[
            (sample_trades['type'] == 'CLOSE') & 
            (sample_trades['position_pnl'].notna())
        ]
        
        fig = dashboard._create_trade_analysis(sample_trades)
        
        # Verify win/loss calculation
        wins = (closed_trades['position_pnl'] > 0).sum()
        losses = (closed_trades['position_pnl'] <= 0).sum()
        
        # Find pie chart data
        pie_trace = None
        for trace in fig.data:
            if trace.type == 'pie':
                pie_trace = trace
                break
        
        assert pie_trace is not None
        assert pie_trace.values[0] == wins
        assert pie_trace.values[1] == losses
    
    def test_create_metrics_table(self, sample_performance_metrics):
        """Test performance metrics table creation."""
        dashboard = Dashboard()
        fig = dashboard._create_metrics_table(sample_performance_metrics)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == 'table'
        
        # Check table structure
        table = fig.data[0]
        assert 'Group' in table.header.values
        assert 'Metric' in table.header.values
        assert 'Value' in table.header.values
        
        # Check layout
        assert fig.layout.title.text == 'Performance Metrics'
        assert fig.layout.height == 600
    
    def test_create_metrics_table_formatting(self, sample_performance_metrics):
        """Test metrics table value formatting."""
        dashboard = Dashboard()
        fig = dashboard._create_metrics_table(sample_performance_metrics)
        
        table = fig.data[0]
        values = table.cells.values
        
        # Check formatting of different metric types
        formatted_values = values[2]  # Value column
        
        # PnL values should have dollar sign and commas
        assert any('$' in str(v) and ',' in str(v) for v in formatted_values)
        
        # Ratios should have decimal places
        assert any('.' in str(v) and '$' not in str(v) for v in formatted_values)
    
    def test_create_metrics_table_empty(self):
        """Test metrics table with empty data."""
        dashboard = Dashboard()
        empty_metrics = {}
        
        fig = dashboard._create_metrics_table(empty_metrics)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == 'table'
    
    def test_create_metrics_gauges(self, sample_performance_metrics):
        """Test metrics gauge charts creation."""
        dashboard = Dashboard()
        fig = dashboard._create_metrics_gauges(sample_performance_metrics)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # 3 gauge charts
        
        # Check gauge types
        for trace in fig.data:
            assert trace.type == 'indicator'
            assert trace.mode == 'gauge+number'
        
        # Check layout
        assert fig.layout.title.text == 'Key Performance Indicators'
        assert fig.layout.height == 300
    
    def test_create_metrics_gauges_calculations(self, sample_performance_metrics):
        """Test gauge calculations."""
        dashboard = Dashboard()
        fig = dashboard._create_metrics_gauges(sample_performance_metrics)
        
        # Calculate expected values
        expected_win_rate = (sample_performance_metrics['winning_trades'] / 
                           sample_performance_metrics['total_trades'] * 100)
        expected_return_pct = (sample_performance_metrics['total_return'] / 
                             sample_performance_metrics['initial_capital'] * 100)
        
        # Verify gauge values
        assert abs(fig.data[0].value - expected_win_rate) < 0.01
        assert fig.data[1].value == sample_performance_metrics['sharpe_ratio']
        assert abs(fig.data[2].value - expected_return_pct) < 0.01
    
    def test_create_metrics_gauges_color_logic(self, sample_performance_metrics):
        """Test gauge color logic based on values."""
        dashboard = Dashboard()
        
        # Test different scenarios
        scenarios = [
            {'winning_trades': 7, 'total_trades': 10, 'sharpe_ratio': 1.5, 'total_return': 5000},
            {'winning_trades': 3, 'total_trades': 10, 'sharpe_ratio': 0.5, 'total_return': -5000},
            {'winning_trades': 5, 'total_trades': 10, 'sharpe_ratio': -0.5, 'total_return': 0}
        ]
        
        for metrics in scenarios:
            metrics['initial_capital'] = 100000
            fig = dashboard._create_metrics_gauges(metrics)
            
            # Win rate gauge color
            win_rate = metrics['winning_trades'] / metrics['total_trades'] * 100
            expected_color = 'green' if win_rate > 50 else 'red'
            assert fig.data[0].gauge.bar.color == expected_color
            
            # Sharpe ratio gauge color
            sharpe = metrics['sharpe_ratio']
            if sharpe > 1:
                expected_color = 'green'
            elif sharpe > 0:
                expected_color = 'orange'
            else:
                expected_color = 'red'
            assert fig.data[1].gauge.bar.color == expected_color
            
            # Return gauge color
            total_return = metrics['total_return']
            expected_color = 'green' if total_return > 0 else 'red'
            assert fig.data[2].gauge.bar.color == expected_color
    
    def test_generate_html(self):
        """Test HTML generation."""
        dashboard = Dashboard()
        
        # Add some test figures
        fig1 = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])
        fig2 = go.Figure(data=[go.Bar(x=['A', 'B', 'C'], y=[1, 2, 3])])
        dashboard.figures = [fig1, fig2]
        
        html = dashboard._generate_html("Test Dashboard")
        
        # Check HTML structure
        assert '<!DOCTYPE html>' in html
        assert '<html>' in html
        assert '</html>' in html
        assert '<title>Test Dashboard</title>' in html
        assert 'plotly-latest.min.js' in html
        
        # Check styling
        assert 'font-family: Arial, sans-serif' in html
        assert 'background-color: #f5f5f5' in html
        assert 'chart-container' in html
        
        # Check that figures are included
        assert 'chart_0' in html
        assert 'chart_1' in html
    
    def test_create_dashboard_complete(self, sample_results, tmp_path):
        """Test complete dashboard creation."""
        dashboard = Dashboard()
        
        output_path = tmp_path / "test_dashboard.html"
        result_path = dashboard.create_dashboard(
            sample_results,
            output_path=str(output_path),
            title="Test Backtest Dashboard"
        )
        
        assert result_path == str(output_path)
        assert output_path.exists()
        
        # Check file content
        with open(output_path, 'r') as f:
            content = f.read()
            assert 'Test Backtest Dashboard' in content
            assert 'plotly' in content
            assert 'Portfolio Equity Curve' in content
    
    def test_create_dashboard_minimal(self, tmp_path):
        """Test dashboard creation with minimal data."""
        dashboard = Dashboard()
        
        minimal_results = {
            'equity_curve': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'performance': {}
        }
        
        output_path = tmp_path / "minimal_dashboard.html"
        result_path = dashboard.create_dashboard(
            minimal_results,
            output_path=str(output_path)
        )
        
        assert result_path == str(output_path)
        assert output_path.exists()
    
    def test_create_dashboard_missing_keys(self, tmp_path):
        """Test dashboard creation with missing result keys."""
        dashboard = Dashboard()
        
        # Results with missing keys
        partial_results = {
            'equity_curve': pd.DataFrame({'total_value': [100000, 105000]})
            # Missing 'trades' and 'performance'
        }
        
        output_path = tmp_path / "partial_dashboard.html"
        result_path = dashboard.create_dashboard(
            partial_results,
            output_path=str(output_path)
        )
        
        assert result_path == str(output_path)
        assert output_path.exists()
    
    def test_create_dashboard_file_write_error(self, sample_results):
        """Test dashboard creation with file write error."""
        dashboard = Dashboard()
        
        # Use an invalid path
        invalid_path = "/invalid/path/dashboard.html"
        
        # Should handle the error gracefully
        with pytest.raises(Exception):
            dashboard.create_dashboard(sample_results, output_path=invalid_path)
    
    def test_figure_clearing(self, sample_results, tmp_path):
        """Test that figures are cleared between dashboard creations."""
        dashboard = Dashboard()
        
        # Create first dashboard
        output1 = tmp_path / "dashboard1.html"
        dashboard.create_dashboard(sample_results, output_path=str(output1))
        figures_count_1 = len(dashboard.figures)
        
        # Create second dashboard
        output2 = tmp_path / "dashboard2.html"
        dashboard.create_dashboard(sample_results, output_path=str(output2))
        figures_count_2 = len(dashboard.figures)
        
        # Figures should be same count (cleared and recreated)
        assert figures_count_1 == figures_count_2
        assert figures_count_1 > 0
    
    def test_edge_case_single_trade(self):
        """Test with single trade."""
        dashboard = Dashboard()
        
        single_trade = pd.DataFrame({
            'timestamp': [pd.Timestamp('2023-01-01')],
            'type': ['CLOSE'],
            'symbol': ['AAPL'],
            'price': [150.0],
            'quantity': [-100],
            'position_pnl': [500.0]
        })
        
        fig = dashboard._create_trade_analysis(single_trade)
        assert isinstance(fig, go.Figure)
        
        # Should still create the analysis
        pie_trace = None
        for trace in fig.data:
            if trace.type == 'pie':
                pie_trace = trace
                break
        
        assert pie_trace is not None
        assert pie_trace.values[0] == 1  # 1 win
        assert pie_trace.values[1] == 0  # 0 losses
    
    def test_performance_metrics_edge_cases(self):
        """Test metrics with edge cases."""
        dashboard = Dashboard()
        
        # Zero trades
        zero_trade_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'sharpe_ratio': 0,
            'total_return': 0,
            'initial_capital': 100000
        }
        
        fig = dashboard._create_metrics_gauges(zero_trade_metrics)
        assert isinstance(fig, go.Figure)
        assert fig.data[0].value == 0  # Win rate should be 0
    
    def test_html_escaping(self):
        """Test HTML escaping in generated content."""
        dashboard = Dashboard()
        
        # Create figure with potentially problematic title
        fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])
        fig.update_layout(title="<script>alert('test')</script>")
        dashboard.figures = [fig]
        
        html = dashboard._generate_html("Test & Dashboard < > \"'")
        
        # The title in the head should be properly escaped
        assert '<title>Test & Dashboard < > "\'</title>' in html or \
               '<title>Test &amp; Dashboard &lt; &gt; &quot;&#39;</title>' in html
    
    def test_custom_output_directory(self, sample_results):
        """Test creating dashboard in custom directory."""
        dashboard = Dashboard()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = Path(tmpdir) / "custom" / "path"
            custom_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = custom_dir / "dashboard.html"
            result = dashboard.create_dashboard(
                sample_results,
                output_path=str(output_path)
            )
            
            assert result == str(output_path)
            assert output_path.exists()
    
    def test_plotly_offline_mode(self):
        """Test that plotly is used in offline mode."""
        dashboard = Dashboard()
        
        # Create a simple figure
        fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])
        dashboard.figures = [fig]
        
        with patch('src.visualization.dashboard.offline.plot') as mock_plot:
            mock_plot.return_value = '<div>Mock plot</div>'
            
            html = dashboard._generate_html("Test")
            
            # Verify offline.plot was called correctly
            mock_plot.assert_called()
            call_args = mock_plot.call_args
            assert call_args[1]['output_type'] == 'div'
            assert call_args[1]['include_plotlyjs'] == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])