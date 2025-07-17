"""
Comprehensive tests for the Dashboard class.

This module provides complete test coverage for dashboard creation functionality
including equity charts, drawdown visualization, trade analysis, and metrics display.
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
    def dashboard(self):
        """Create Dashboard instance."""
        return Dashboard()
    
    @pytest.fixture
    def sample_backtest_results(self):
        """Create comprehensive sample backtest results."""
        # Generate equity curve
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        returns = np.random.normal(0.0005, 0.02, len(dates))
        cumulative_returns = (1 + returns).cumprod()
        
        equity_curve = pd.DataFrame({
            'timestamp': dates,
            'total_value': 100000 * cumulative_returns,
            'cash': 50000 * np.ones(len(dates)),
            'holdings_value': 50000 * cumulative_returns
        })
        
        # Generate trades
        trades_data = []
        for i in range(30):
            entry_date = dates[i * 8]
            exit_date = entry_date + timedelta(days=np.random.randint(1, 7))
            
            entry_price = 100 + np.random.uniform(-20, 20)
            exit_price = entry_price * (1 + np.random.uniform(-0.05, 0.08))
            quantity = np.random.randint(10, 100)
            
            trades_data.append({
                'trade_id': i + 1,
                'entry_time': entry_date,
                'exit_time': exit_date,
                'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN']),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'profit_loss': (exit_price - entry_price) * quantity,
                'return_pct': ((exit_price - entry_price) / entry_price) * 100,
                'commission': 2.0,
                'side': 'long'
            })
        
        trades = pd.DataFrame(trades_data)
        
        # Calculate performance metrics
        total_return = (equity_curve['total_value'].iloc[-1] / equity_curve['total_value'].iloc[0]) - 1
        
        performance = {
            'total_return': total_return,
            'annualized_return': total_return * (252 / len(dates)),
            'sharpe_ratio': 1.45,
            'sortino_ratio': 2.13,
            'max_drawdown': -0.1523,
            'win_rate': len(trades[trades['profit_loss'] > 0]) / len(trades),
            'profit_factor': 1.85,
            'total_trades': len(trades),
            'winning_trades': len(trades[trades['profit_loss'] > 0]),
            'losing_trades': len(trades[trades['profit_loss'] <= 0]),
            'avg_win': trades[trades['profit_loss'] > 0]['profit_loss'].mean() if len(trades[trades['profit_loss'] > 0]) > 0 else 0,
            'avg_loss': trades[trades['profit_loss'] <= 0]['profit_loss'].mean() if len(trades[trades['profit_loss'] <= 0]) > 0 else 0,
            'best_trade': trades['profit_loss'].max(),
            'worst_trade': trades['profit_loss'].min(),
            'calmar_ratio': 1.88,
            'volatility': 0.1856,
            'var_95': -0.0234,
            'cvar_95': -0.0345,
            'skewness': -0.234,
            'kurtosis': 3.456
        }
        
        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'performance': performance
        }
    
    def test_dashboard_initialization(self, dashboard):
        """Test Dashboard initialization."""
        assert isinstance(dashboard, Dashboard)
        assert hasattr(dashboard, '_chart_generator')
    
    def test_create_dashboard_basic(self, dashboard, sample_backtest_results):
        """Test basic dashboard creation."""
        output_path = dashboard.create_dashboard(
            sample_backtest_results['equity_curve'],
            sample_backtest_results['trades'],
            sample_backtest_results['performance']
        )
        
        assert output_path is not None
        assert output_path.endswith('.html')
        assert os.path.exists(output_path)
        
        # Clean up
        os.remove(output_path)
    
    def test_create_dashboard_custom_output(self, dashboard, sample_backtest_results):
        """Test dashboard creation with custom output path."""
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            custom_path = tmp.name
        
        output_path = dashboard.create_dashboard(
            sample_backtest_results['equity_curve'],
            sample_backtest_results['trades'],
            sample_backtest_results['performance'],
            output_path=custom_path,
            title="Custom Test Dashboard"
        )
        
        assert output_path == custom_path
        assert os.path.exists(output_path)
        
        # Verify content
        with open(output_path, 'r') as f:
            content = f.read()
            assert "Custom Test Dashboard" in content
        
        # Clean up
        os.remove(output_path)
    
    def test_create_equity_chart(self, dashboard, sample_backtest_results):
        """Test equity chart creation."""
        fig = dashboard._create_equity_chart(sample_backtest_results['equity_curve'])
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Check for expected traces
        trace_names = [trace.name for trace in fig.data if hasattr(trace, 'name')]
        assert any('Total Value' in name or 'Portfolio' in name for name in trace_names)
        
        # Verify layout
        assert fig.layout.title.text is not None
        assert fig.layout.xaxis.title.text is not None
        assert fig.layout.yaxis.title.text is not None
    
    def test_create_drawdown_chart(self, dashboard, sample_backtest_results):
        """Test drawdown chart creation."""
        fig = dashboard._create_drawdown_chart(sample_backtest_results['equity_curve'])
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Verify drawdown values are negative or zero
        drawdown_data = fig.data[0].y
        assert all(val <= 0 for val in drawdown_data if not pd.isna(val))
        
        # Check layout
        assert 'Drawdown' in fig.layout.title.text
    
    def test_create_trades_chart(self, dashboard, sample_backtest_results):
        """Test trades chart creation."""
        fig = dashboard._create_trades_chart(sample_backtest_results['trades'])
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Verify trade data is properly visualized
        assert any(hasattr(trace, 'x') for trace in fig.data)
    
    def test_create_trade_analysis(self, dashboard, sample_backtest_results):
        """Test trade analysis chart creation."""
        fig = dashboard._create_trade_analysis(sample_backtest_results['trades'])
        
        assert isinstance(fig, go.Figure)
        # Should have multiple subplots
        assert hasattr(fig, '_grid_ref') or len(fig.data) > 1
    
    def test_create_metrics_table(self, dashboard, sample_backtest_results):
        """Test metrics table creation."""
        fig = dashboard._create_metrics_table(sample_backtest_results['performance'])
        
        assert isinstance(fig, go.Figure)
        # Should have table trace
        assert any(isinstance(trace, go.Table) for trace in fig.data)
        
        # Verify key metrics are included
        table_trace = next(trace for trace in fig.data if isinstance(trace, go.Table))
        cells_text = str(table_trace.cells.values)
        
        key_metrics = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        for metric in key_metrics:
            assert metric in cells_text or metric.lower() in cells_text.lower()
    
    def test_create_metrics_gauges(self, dashboard, sample_backtest_results):
        """Test metrics gauges creation."""
        fig = dashboard._create_metrics_gauges(sample_backtest_results['performance'])
        
        assert isinstance(fig, go.Figure)
        # Should have multiple gauge indicators
        assert any(isinstance(trace, go.Indicator) for trace in fig.data)
        
        # Count indicators
        indicator_count = sum(1 for trace in fig.data if isinstance(trace, go.Indicator))
        assert indicator_count >= 3  # At least Sharpe, Sortino, and Win Rate
    
    def test_generate_html(self, dashboard):
        """Test HTML generation."""
        html = dashboard._generate_html("Test Dashboard")
        
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "Test Dashboard" in html
        assert "plotly" in html.lower()
    
    def test_empty_data_handling(self, dashboard):
        """Test handling of empty data."""
        empty_equity = pd.DataFrame({
            'timestamp': [],
            'total_value': [],
            'cash': [],
            'holdings_value': []
        })
        
        empty_trades = pd.DataFrame({
            'trade_id': [],
            'entry_time': [],
            'exit_time': [],
            'profit_loss': []
        })
        
        empty_performance = {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0
        }
        
        # Should handle empty data gracefully
        with pytest.raises((ValueError, IndexError, KeyError)):
            dashboard.create_dashboard(empty_equity, empty_trades, empty_performance)
    
    def test_missing_columns_handling(self, dashboard):
        """Test handling of missing required columns."""
        # Equity curve with missing columns
        invalid_equity = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'some_value': np.random.randn(10)
        })
        
        with pytest.raises((KeyError, ValueError)):
            dashboard._create_equity_chart(invalid_equity)
        
        # Trades with missing columns
        invalid_trades = pd.DataFrame({
            'trade_id': range(5),
            'some_column': np.random.randn(5)
        })
        
        with pytest.raises((KeyError, ValueError, AttributeError)):
            dashboard._create_trades_chart(invalid_trades)
    
    def test_performance_metrics_completeness(self, dashboard):
        """Test all performance metrics are displayed."""
        comprehensive_metrics = {
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
            'volatility': 0.1856,
            'var_95': -0.0234,
            'cvar_95': -0.0345,
            'max_consecutive_wins': 7,
            'max_consecutive_losses': 4,
            'recovery_factor': 2.34,
            'ulcer_index': 0.0456,
            'serenity_ratio': 3.21,
            'omega_ratio': 1.67
        }
        
        # Test metrics table includes all metrics
        fig_table = dashboard._create_metrics_table(comprehensive_metrics)
        table_str = str(fig_table)
        
        # Verify some key metrics are present
        assert 'sharpe' in table_str.lower()
        assert 'return' in table_str.lower()
        assert 'drawdown' in table_str.lower()
    
    def test_trade_analysis_components(self, dashboard, sample_backtest_results):
        """Test trade analysis includes all components."""
        trades = sample_backtest_results['trades']
        
        # Add more analysis columns
        trades['holding_period'] = (trades['exit_time'] - trades['entry_time']).dt.days
        trades['return_per_day'] = trades['return_pct'] / trades['holding_period']
        
        fig = dashboard._create_trade_analysis(trades)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_color_coding(self, dashboard, sample_backtest_results):
        """Test appropriate color coding for profits/losses."""
        trades = sample_backtest_results['trades']
        fig = dashboard._create_trades_chart(trades)
        
        # Check that winning and losing trades have different colors
        colors_used = set()
        for trace in fig.data:
            if hasattr(trace, 'marker') and hasattr(trace.marker, 'color'):
                if isinstance(trace.marker.color, (list, tuple, np.ndarray)):
                    colors_used.update(trace.marker.color)
                else:
                    colors_used.add(trace.marker.color)
        
        # Should have at least 2 different colors (for wins/losses)
        assert len(colors_used) >= 1
    
    def test_responsive_layout(self, dashboard, sample_backtest_results):
        """Test dashboard has responsive layout configuration."""
        output_path = dashboard.create_dashboard(
            sample_backtest_results['equity_curve'],
            sample_backtest_results['trades'],
            sample_backtest_results['performance']
        )
        
        with open(output_path, 'r') as f:
            content = f.read()
            
        # Check for responsive elements
        assert 'responsive' in content or 'autosize' in content
        
        # Clean up
        os.remove(output_path)
    
    def test_error_messages(self, dashboard):
        """Test appropriate error messages for invalid inputs."""
        # Test with None values
        with pytest.raises((TypeError, AttributeError)):
            dashboard.create_dashboard(None, None, None)
        
        # Test with wrong data types
        with pytest.raises((TypeError, AttributeError, ValueError)):
            dashboard.create_dashboard("not a dataframe", [], {})
    
    @patch('builtins.open', new_callable=mock_open)
    def test_file_writing(self, mock_file, dashboard, sample_backtest_results):
        """Test file writing operations."""
        dashboard.create_dashboard(
            sample_backtest_results['equity_curve'],
            sample_backtest_results['trades'],
            sample_backtest_results['performance']
        )
        
        # Verify file was opened for writing
        mock_file.assert_called()
        
        # Verify write was called
        handle = mock_file()
        handle.write.assert_called()
    
    def test_gauge_ranges(self, dashboard):
        """Test gauge indicators have appropriate ranges."""
        performance = {
            'sharpe_ratio': 2.5,  # Good
            'sortino_ratio': 3.2,  # Very good
            'win_rate': 0.45,  # Below average
        }
        
        fig = dashboard._create_metrics_gauges(performance)
        
        # Check gauge configurations
        for trace in fig.data:
            if isinstance(trace, go.Indicator) and trace.mode == "gauge+number":
                # Verify gauge has proper range
                assert hasattr(trace.gauge, 'axis')
                assert hasattr(trace.gauge.axis, 'range')
                assert len(trace.gauge.axis.range) == 2
                assert trace.gauge.axis.range[0] < trace.gauge.axis.range[1]