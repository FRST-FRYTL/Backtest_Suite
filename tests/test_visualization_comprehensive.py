"""
Comprehensive tests for visualization components.

This module provides complete test coverage for all visualization functions
including chart generation, dashboard creation, and interactive plotting.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path

from src.reporting.visualizations import ReportVisualizations
from src.reporting.visualization_types import ChartType, VisualizationConfig
from src.visualization.comprehensive_trading_dashboard import ComprehensiveTradingDashboard
from src.visualization.supertrend_dashboard import SuperTrendDashboard


class TestReportVisualizations:
    """Comprehensive tests for ReportVisualizations class."""
    
    @pytest.fixture
    def sample_equity_data(self):
        """Create sample equity curve data."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        returns = np.random.normal(0.001, 0.02, len(dates))
        equity = pd.Series(100000 * (1 + returns).cumprod(), index=dates, name='equity')
        
        return equity
    
    @pytest.fixture
    def sample_trades_data(self):
        """Create sample trades data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        trades = pd.DataFrame({
            'trade_id': range(1, 51),
            'entry_time': dates,
            'exit_time': dates + timedelta(days=np.random.randint(1, 10, 50)),
            'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT'], 50),
            'side': np.random.choice(['long', 'short'], 50),
            'entry_price': np.random.uniform(100, 200, 50),
            'exit_price': np.random.uniform(95, 210, 50),
            'stop_loss': np.random.uniform(90, 110, 50),
            'take_profit': np.random.uniform(190, 220, 50),
            'size': np.random.uniform(100, 1000, 50),
            'pnl': np.random.normal(100, 500, 50),
            'duration': np.random.uniform(1, 240, 50),  # hours
            'exit_reason': np.random.choice(['target', 'stop', 'time'], 50)
        })
        
        return trades
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
        
        data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.5, len(dates)),
            'high': prices + np.abs(np.random.normal(2, 1, len(dates))),
            'low': prices - np.abs(np.random.normal(2, 1, len(dates))),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Ensure OHLC relationships
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
    
    def test_initialization(self):
        """Test ReportVisualizations initialization."""
        viz = ReportVisualizations()
        
        assert isinstance(viz.config, VisualizationConfig)
        assert viz.config.theme == 'plotly'
        assert viz.config.width == 1200
        assert viz.config.height == 600
        assert viz.config.show_grid == True
        assert viz.config.show_legend == True
    
    def test_initialization_custom_config(self):
        """Test initialization with custom configuration."""
        config = VisualizationConfig(
            theme='plotly_dark',
            width=1600,
            height=800,
            show_grid=False,
            show_legend=False
        )
        
        viz = ReportVisualizations(config)
        
        assert viz.config.theme == 'plotly_dark'
        assert viz.config.width == 1600
        assert viz.config.height == 800
        assert viz.config.show_grid == False
        assert viz.config.show_legend == False
    
    def test_create_equity_curve(self, sample_equity_data):
        """Test equity curve visualization."""
        viz = ReportVisualizations()
        fig = viz.create_equity_curve(sample_equity_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Check that equity curve trace exists
        trace_names = [trace.name for trace in fig.data]
        assert 'Equity Curve' in trace_names
        
        # Check layout
        assert fig.layout.title.text == 'Portfolio Equity Curve'
        assert fig.layout.xaxis.title.text == 'Date'
        assert fig.layout.yaxis.title.text == 'Portfolio Value ($)'
        
        # Check data integrity
        equity_trace = fig.data[0]
        assert len(equity_trace.x) == len(sample_equity_data)
        assert len(equity_trace.y) == len(sample_equity_data)
    
    def test_create_equity_curve_with_drawdown(self, sample_equity_data):
        """Test equity curve with drawdown visualization."""
        viz = ReportVisualizations()
        fig = viz.create_equity_curve(sample_equity_data, show_drawdown=True)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Equity + drawdown
        
        # Check for drawdown trace
        trace_names = [trace.name for trace in fig.data]
        assert 'Drawdown' in trace_names
        
        # Check for secondary y-axis
        assert fig.layout.yaxis2 is not None
        assert fig.layout.yaxis2.title.text == 'Drawdown (%)'
    
    def test_create_returns_distribution(self, sample_equity_data):
        """Test returns distribution visualization."""
        viz = ReportVisualizations()
        returns = sample_equity_data.pct_change().dropna()
        fig = viz.create_returns_distribution(returns)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Check histogram trace
        assert fig.data[0].type == 'histogram'
        assert fig.layout.title.text == 'Returns Distribution'
        assert fig.layout.xaxis.title.text == 'Daily Returns'
        assert fig.layout.yaxis.title.text == 'Frequency'
    
    def test_create_returns_distribution_with_stats(self, sample_equity_data):
        """Test returns distribution with statistics."""
        viz = ReportVisualizations()
        returns = sample_equity_data.pct_change().dropna()
        fig = viz.create_returns_distribution(returns, show_stats=True)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Histogram + normal distribution
        
        # Check that statistics are shown in annotations
        annotations = fig.layout.annotations
        assert annotations is not None
        assert len(annotations) > 0
    
    def test_create_rolling_metrics(self, sample_equity_data):
        """Test rolling metrics visualization."""
        viz = ReportVisualizations()
        returns = sample_equity_data.pct_change().dropna()
        fig = viz.create_rolling_metrics(returns, window=30)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Rolling return + rolling volatility
        
        # Check for subplots
        assert len(fig.layout.annotations) >= 2  # Subplot titles
        
        # Check trace names
        trace_names = [trace.name for trace in fig.data]
        assert 'Rolling Return' in trace_names
        assert 'Rolling Volatility' in trace_names
    
    def test_create_trade_distribution(self, sample_trades_data):
        """Test trade distribution visualization."""
        viz = ReportVisualizations()
        fig = viz.create_trade_distribution(sample_trades_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Check histogram
        assert fig.data[0].type == 'histogram'
        assert fig.layout.title.text == 'Trade P&L Distribution'
        assert fig.layout.xaxis.title.text == 'P&L ($)'
        assert fig.layout.yaxis.title.text == 'Number of Trades'
    
    def test_create_trade_distribution_with_stats(self, sample_trades_data):
        """Test trade distribution with statistics."""
        viz = ReportVisualizations()
        fig = viz.create_trade_distribution(sample_trades_data, show_stats=True)
        
        assert isinstance(fig, go.Figure)
        
        # Check that statistics are displayed
        annotations = fig.layout.annotations
        assert annotations is not None
        assert len(annotations) > 0
    
    def test_create_monthly_returns_heatmap(self, sample_equity_data):
        """Test monthly returns heatmap."""
        viz = ReportVisualizations()
        fig = viz.create_monthly_returns_heatmap(sample_equity_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Check heatmap
        assert fig.data[0].type == 'heatmap'
        assert fig.layout.title.text == 'Monthly Returns Heatmap'
        assert fig.layout.xaxis.title.text == 'Month'
        assert fig.layout.yaxis.title.text == 'Year'
    
    def test_create_trade_timeline(self, sample_trades_data):
        """Test trade timeline visualization."""
        viz = ReportVisualizations()
        fig = viz.create_trade_timeline(sample_trades_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Check scatter plot
        assert fig.data[0].type == 'scatter'
        assert fig.layout.title.text == 'Trade Timeline'
        assert fig.layout.xaxis.title.text == 'Date'
        assert fig.layout.yaxis.title.text == 'P&L ($)'
    
    def test_create_trade_timeline_with_symbols(self, sample_trades_data):
        """Test trade timeline with symbol coloring."""
        viz = ReportVisualizations()
        fig = viz.create_trade_timeline(sample_trades_data, color_by='symbol')
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Check that different symbols have different colors
        trace_names = [trace.name for trace in fig.data]
        unique_symbols = sample_trades_data['symbol'].unique()
        
        # Each symbol should have its own trace
        for symbol in unique_symbols:
            assert symbol in trace_names
    
    def test_create_win_loss_chart(self, sample_trades_data):
        """Test win/loss chart visualization."""
        viz = ReportVisualizations()
        fig = viz.create_win_loss_chart(sample_trades_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Check bar chart
        assert fig.data[0].type == 'bar'
        assert fig.layout.title.text == 'Win/Loss Analysis'
        assert fig.layout.xaxis.title.text == 'Trade Type'
        assert fig.layout.yaxis.title.text == 'Count'
    
    def test_create_trade_duration_histogram(self, sample_trades_data):
        """Test trade duration histogram."""
        viz = ReportVisualizations()
        fig = viz.create_trade_duration_histogram(sample_trades_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Check histogram
        assert fig.data[0].type == 'histogram'
        assert fig.layout.title.text == 'Trade Duration Distribution'
        assert fig.layout.xaxis.title.text == 'Duration (hours)'
        assert fig.layout.yaxis.title.text == 'Number of Trades'
    
    def test_create_trade_price_chart(self, sample_trades_data, sample_price_data):
        """Test trade price chart visualization."""
        viz = ReportVisualizations()
        fig = viz.create_trade_price_chart(sample_trades_data, sample_price_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Check that price data is plotted
        trace_names = [trace.name for trace in fig.data]
        assert 'Price' in trace_names
        
        # Check for entry/exit markers
        assert 'Entry' in trace_names
        assert 'Exit' in trace_names
    
    def test_create_trade_price_chart_with_levels(self, sample_trades_data, sample_price_data):
        """Test trade price chart with stop loss and take profit levels."""
        viz = ReportVisualizations()
        fig = viz.create_trade_price_chart(
            sample_trades_data, 
            sample_price_data, 
            show_stop_loss=True,
            show_take_profit=True
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Check for stop loss and take profit traces
        trace_names = [trace.name for trace in fig.data]
        assert 'Stop Loss' in trace_names
        assert 'Take Profit' in trace_names
    
    def test_create_stop_loss_analysis(self, sample_trades_data):
        """Test stop loss analysis visualization."""
        viz = ReportVisualizations()
        fig = viz.create_stop_loss_analysis(sample_trades_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Check for subplots
        assert len(fig.layout.annotations) >= 2  # Subplot titles
        
        # Check layout
        assert fig.layout.title.text == 'Stop Loss Analysis'
    
    def test_create_stop_loss_analysis_empty_data(self):
        """Test stop loss analysis with empty data."""
        viz = ReportVisualizations()
        empty_trades = pd.DataFrame()
        fig = viz.create_stop_loss_analysis(empty_trades)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
    
    def test_create_trade_risk_chart(self, sample_trades_data):
        """Test trade risk chart visualization."""
        viz = ReportVisualizations()
        fig = viz.create_trade_risk_chart(sample_trades_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Check for subplots
        assert len(fig.layout.annotations) >= 2  # Subplot titles
        
        # Check layout
        assert fig.layout.title.text == 'Trade Risk Analysis'
    
    def test_create_trade_risk_chart_empty_data(self):
        """Test trade risk chart with empty data."""
        viz = ReportVisualizations()
        empty_trades = pd.DataFrame()
        fig = viz.create_trade_risk_chart(empty_trades)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
    
    def test_create_performance_metrics_table(self):
        """Test performance metrics table creation."""
        viz = ReportVisualizations()
        
        metrics = {
            'Total Return': '15.2%',
            'Sharpe Ratio': '1.85',
            'Max Drawdown': '-8.3%',
            'Win Rate': '62.5%',
            'Profit Factor': '2.15'
        }
        
        fig = viz.create_performance_metrics_table(metrics)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Check table
        assert fig.data[0].type == 'table'
        assert fig.layout.title.text == 'Performance Metrics'
    
    def test_create_correlation_matrix(self):
        """Test correlation matrix visualization."""
        viz = ReportVisualizations()
        
        # Create sample correlation data
        np.random.seed(42)
        data = pd.DataFrame({
            'Strategy A': np.random.randn(100),
            'Strategy B': np.random.randn(100),
            'Strategy C': np.random.randn(100),
            'Benchmark': np.random.randn(100)
        })
        
        fig = viz.create_correlation_matrix(data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Check heatmap
        assert fig.data[0].type == 'heatmap'
        assert fig.layout.title.text == 'Correlation Matrix'
    
    def test_create_scatter_plot(self):
        """Test scatter plot creation."""
        viz = ReportVisualizations()
        
        # Create sample data
        np.random.seed(42)
        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.5
        
        fig = viz.create_scatter_plot(x, y, 'Risk', 'Return')
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        
        # Check scatter plot
        assert fig.data[0].type == 'scatter'
        assert fig.layout.xaxis.title.text == 'Risk'
        assert fig.layout.yaxis.title.text == 'Return'
    
    def test_create_scatter_plot_with_trend(self):
        """Test scatter plot with trend line."""
        viz = ReportVisualizations()
        
        # Create sample data with clear trend
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1 + np.random.randn(100) * 0.5
        
        fig = viz.create_scatter_plot(x, y, 'X', 'Y', show_trend=True)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Scatter + trend line
        
        # Check for trend line
        trace_names = [trace.name for trace in fig.data]
        assert 'Trend' in trace_names
    
    def test_save_figure(self, tmp_path):
        """Test figure saving functionality."""
        viz = ReportVisualizations()
        
        # Create simple figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name='test'))
        
        # Save as HTML
        html_path = tmp_path / "test_figure.html"
        viz.save_figure(fig, str(html_path), format='html')
        
        assert html_path.exists()
        
        # Check HTML content
        with open(html_path, 'r') as f:
            content = f.read()
            assert 'plotly' in content.lower()
    
    def test_save_figure_png(self, tmp_path):
        """Test figure saving as PNG."""
        viz = ReportVisualizations()
        
        # Create simple figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name='test'))
        
        # Save as PNG
        png_path = tmp_path / "test_figure.png"
        
        # This might fail if kaleido is not installed
        try:
            viz.save_figure(fig, str(png_path), format='png')
            assert png_path.exists()
        except Exception:
            pytest.skip("PNG export requires kaleido package")
    
    def test_apply_theme(self):
        """Test theme application."""
        viz = ReportVisualizations()
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name='test'))
        
        # Apply theme
        themed_fig = viz.apply_theme(fig, 'plotly_dark')
        
        assert isinstance(themed_fig, go.Figure)
        assert themed_fig.layout.template.layout.plot_bgcolor == 'rgb(17,17,17)'
    
    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data."""
        viz = ReportVisualizations()
        
        # Test with None data
        fig = viz.create_equity_curve(None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        fig = viz.create_trade_distribution(empty_df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
    
    def test_performance_large_dataset(self):
        """Test performance with large dataset."""
        viz = ReportVisualizations()
        
        # Create large dataset
        dates = pd.date_range('2020-01-01', periods=10000, freq='D')
        large_equity = pd.Series(
            100000 * (1 + np.random.normal(0.001, 0.02, len(dates))).cumprod(),
            index=dates
        )
        
        import time
        start_time = time.time()
        fig = viz.create_equity_curve(large_equity)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 5  # Less than 5 seconds
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestVisualizationTypes:
    """Test visualization types and configuration."""
    
    def test_chart_type_enum(self):
        """Test ChartType enum."""
        assert hasattr(ChartType, 'LINE')
        assert hasattr(ChartType, 'BAR')
        assert hasattr(ChartType, 'SCATTER')
        assert hasattr(ChartType, 'HEATMAP')
        assert hasattr(ChartType, 'HISTOGRAM')
        assert hasattr(ChartType, 'PIE')
        assert hasattr(ChartType, 'CANDLESTICK')
        assert hasattr(ChartType, 'OHLC')
        assert hasattr(ChartType, 'VIOLIN')
        assert hasattr(ChartType, 'BOX')
        assert hasattr(ChartType, 'SUNBURST')
        assert hasattr(ChartType, 'TREEMAP')
        assert hasattr(ChartType, 'WATERFALL')
        assert hasattr(ChartType, 'FUNNEL')
        assert hasattr(ChartType, 'GAUGE')
        assert hasattr(ChartType, 'INDICATOR')
        assert hasattr(ChartType, 'TABLE')
        assert hasattr(ChartType, 'SANKEY')
        assert hasattr(ChartType, 'POLAR')
        assert hasattr(ChartType, 'RADAR')
        assert hasattr(ChartType, 'SURFACE')
        assert hasattr(ChartType, 'MESH3D')
        assert hasattr(ChartType, 'CONE')
        assert hasattr(ChartType, 'STREAMTUBE')
        assert hasattr(ChartType, 'VOLUME')
        assert hasattr(ChartType, 'ISOSURFACE')
    
    def test_visualization_config_defaults(self):
        """Test VisualizationConfig default values."""
        config = VisualizationConfig()
        
        assert config.theme == 'plotly'
        assert config.width == 1200
        assert config.height == 600
        assert config.show_grid == True
        assert config.show_legend == True
        assert config.color_palette == 'plotly'
        assert config.font_size == 12
        assert config.font_family == 'Arial'
        assert config.background_color == 'white'
        assert config.grid_color == 'lightgray'
        assert config.text_color == 'black'
        assert config.margin == {'l': 40, 'r': 40, 't': 40, 'b': 40}
        assert config.export_format == 'html'
        assert config.export_scale == 1
        assert config.export_engine == 'kaleido'
    
    def test_visualization_config_custom(self):
        """Test VisualizationConfig with custom values."""
        config = VisualizationConfig(
            theme='plotly_dark',
            width=1600,
            height=800,
            show_grid=False,
            show_legend=False,
            color_palette='viridis',
            font_size=14,
            font_family='Helvetica',
            background_color='black',
            grid_color='gray',
            text_color='white',
            margin={'l': 60, 'r': 60, 't': 60, 'b': 60},
            export_format='png',
            export_scale=2,
            export_engine='orca'
        )
        
        assert config.theme == 'plotly_dark'
        assert config.width == 1600
        assert config.height == 800
        assert config.show_grid == False
        assert config.show_legend == False
        assert config.color_palette == 'viridis'
        assert config.font_size == 14
        assert config.font_family == 'Helvetica'
        assert config.background_color == 'black'
        assert config.grid_color == 'gray'
        assert config.text_color == 'white'
        assert config.margin == {'l': 60, 'r': 60, 't': 60, 'b': 60}
        assert config.export_format == 'png'
        assert config.export_scale == 2
        assert config.export_engine == 'orca'
    
    def test_visualization_config_validation(self):
        """Test VisualizationConfig validation."""
        # Test invalid width
        with pytest.raises(ValueError):
            VisualizationConfig(width=0)
        
        # Test invalid height
        with pytest.raises(ValueError):
            VisualizationConfig(height=0)
        
        # Test invalid font_size
        with pytest.raises(ValueError):
            VisualizationConfig(font_size=0)
        
        # Test invalid export_scale
        with pytest.raises(ValueError):
            VisualizationConfig(export_scale=0)


class TestComprehensiveTradingDashboard:
    """Test comprehensive trading dashboard functionality."""
    
    @pytest.fixture
    def sample_dashboard_data(self):
        """Create sample data for dashboard testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Price data
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
        price_data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.5, len(dates)),
            'high': prices + np.abs(np.random.normal(2, 1, len(dates))),
            'low': prices - np.abs(np.random.normal(2, 1, len(dates))),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Ensure OHLC relationships
        price_data['high'] = price_data[['open', 'high', 'close']].max(axis=1)
        price_data['low'] = price_data[['open', 'low', 'close']].min(axis=1)
        
        # Equity curve
        equity_curve = pd.Series(
            100000 * (1 + np.random.normal(0.001, 0.02, len(dates))).cumprod(),
            index=dates
        )
        
        # Trades
        trades = pd.DataFrame({
            'entry_time': pd.to_datetime(['2023-01-05', '2023-01-15', '2023-01-25']),
            'exit_time': pd.to_datetime(['2023-01-10', '2023-01-20', '2023-01-30']),
            'side': ['long', 'short', 'long'],
            'entry_price': [105.0, 108.0, 102.0],
            'exit_price': [110.0, 106.0, 107.0],
            'pnl': [500.0, 200.0, 500.0],
            'size': [100, 100, 100]
        })
        
        return {
            'price_data': price_data,
            'equity_curve': equity_curve,
            'trades': trades,
            'symbol': 'TEST'
        }
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization."""
        dashboard = ComprehensiveTradingDashboard()
        
        assert dashboard.config is not None
        assert isinstance(dashboard.config, VisualizationConfig)
        assert dashboard.visualizations is not None
        assert isinstance(dashboard.visualizations, ReportVisualizations)
    
    def test_create_dashboard(self, sample_dashboard_data):
        """Test dashboard creation."""
        dashboard = ComprehensiveTradingDashboard()
        
        html_content = dashboard.create_dashboard(
            price_data=sample_dashboard_data['price_data'],
            equity_curve=sample_dashboard_data['equity_curve'],
            trades=sample_dashboard_data['trades'],
            symbol=sample_dashboard_data['symbol']
        )
        
        assert isinstance(html_content, str)
        assert len(html_content) > 0
        assert 'plotly' in html_content.lower()
        assert 'TEST' in html_content  # Symbol should be in the content
    
    def test_create_dashboard_minimal_data(self):
        """Test dashboard creation with minimal data."""
        dashboard = ComprehensiveTradingDashboard()
        
        # Minimal price data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        minimal_price_data = pd.DataFrame({
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [102] * 10,
            'volume': [1000000] * 10
        }, index=dates)
        
        html_content = dashboard.create_dashboard(
            price_data=minimal_price_data,
            symbol='MINIMAL'
        )
        
        assert isinstance(html_content, str)
        assert len(html_content) > 0
        assert 'MINIMAL' in html_content
    
    def test_save_dashboard(self, sample_dashboard_data, tmp_path):
        """Test dashboard saving."""
        dashboard = ComprehensiveTradingDashboard()
        
        output_path = tmp_path / "test_dashboard.html"
        
        dashboard.save_dashboard(
            price_data=sample_dashboard_data['price_data'],
            equity_curve=sample_dashboard_data['equity_curve'],
            trades=sample_dashboard_data['trades'],
            symbol=sample_dashboard_data['symbol'],
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        
        # Check file content
        with open(output_path, 'r') as f:
            content = f.read()
            assert 'plotly' in content.lower()
            assert 'TEST' in content


class TestSuperTrendDashboard:
    """Test SuperTrend dashboard functionality."""
    
    @pytest.fixture
    def sample_supertrend_data(self):
        """Create sample SuperTrend data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Price data
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
        price_data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.5, len(dates)),
            'high': prices + np.abs(np.random.normal(2, 1, len(dates))),
            'low': prices - np.abs(np.random.normal(2, 1, len(dates))),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Ensure OHLC relationships
        price_data['high'] = price_data[['open', 'high', 'close']].max(axis=1)
        price_data['low'] = price_data[['open', 'low', 'close']].min(axis=1)
        
        # SuperTrend indicator data
        supertrend_data = pd.DataFrame({
            'trend': np.random.choice([1, -1], len(dates)),
            'upper_band': price_data['close'] + np.random.uniform(2, 5, len(dates)),
            'lower_band': price_data['close'] - np.random.uniform(2, 5, len(dates)),
            'signal_strength': np.random.uniform(1, 10, len(dates))
        }, index=dates)
        
        return {
            'price_data': price_data,
            'supertrend_data': supertrend_data,
            'symbol': 'TEST'
        }
    
    def test_supertrend_dashboard_initialization(self):
        """Test SuperTrend dashboard initialization."""
        dashboard = SuperTrendDashboard()
        
        assert dashboard.config is not None
        assert isinstance(dashboard.config, VisualizationConfig)
        assert dashboard.visualizations is not None
        assert isinstance(dashboard.visualizations, ReportVisualizations)
    
    def test_create_supertrend_dashboard(self, sample_supertrend_data):
        """Test SuperTrend dashboard creation."""
        dashboard = SuperTrendDashboard()
        
        html_content = dashboard.create_dashboard(
            price_data=sample_supertrend_data['price_data'],
            supertrend_data=sample_supertrend_data['supertrend_data'],
            symbol=sample_supertrend_data['symbol']
        )
        
        assert isinstance(html_content, str)
        assert len(html_content) > 0
        assert 'plotly' in html_content.lower()
        assert 'SuperTrend' in html_content
        assert 'TEST' in html_content
    
    def test_save_supertrend_dashboard(self, sample_supertrend_data, tmp_path):
        """Test SuperTrend dashboard saving."""
        dashboard = SuperTrendDashboard()
        
        output_path = tmp_path / "supertrend_dashboard.html"
        
        dashboard.save_dashboard(
            price_data=sample_supertrend_data['price_data'],
            supertrend_data=sample_supertrend_data['supertrend_data'],
            symbol=sample_supertrend_data['symbol'],
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        
        # Check file content
        with open(output_path, 'r') as f:
            content = f.read()
            assert 'plotly' in content.lower()
            assert 'SuperTrend' in content
            assert 'TEST' in content


class TestVisualizationIntegration:
    """Integration tests for visualization components."""
    
    def test_full_visualization_pipeline(self):
        """Test complete visualization pipeline."""
        # Create comprehensive test data
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Price data
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
        price_data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.5, len(dates)),
            'high': prices + np.abs(np.random.normal(2, 1, len(dates))),
            'low': prices - np.abs(np.random.normal(2, 1, len(dates))),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Ensure OHLC relationships
        price_data['high'] = price_data[['open', 'high', 'close']].max(axis=1)
        price_data['low'] = price_data[['open', 'low', 'close']].min(axis=1)
        
        # Equity curve
        equity_curve = pd.Series(
            100000 * (1 + price_data['close'].pct_change().fillna(0)).cumprod(),
            index=dates
        )
        
        # Trades
        n_trades = 50
        trades = pd.DataFrame({
            'entry_time': pd.to_datetime(np.random.choice(dates[:-5], n_trades)),
            'exit_time': pd.to_datetime(np.random.choice(dates[5:], n_trades)),
            'side': np.random.choice(['long', 'short'], n_trades),
            'entry_price': np.random.uniform(95, 105, n_trades),
            'exit_price': np.random.uniform(95, 105, n_trades),
            'pnl': np.random.normal(100, 500, n_trades),
            'size': np.random.uniform(100, 1000, n_trades)
        })
        
        # Initialize visualization components
        viz = ReportVisualizations()
        
        # Create all major visualizations
        figures = {}
        
        # Equity curve
        figures['equity'] = viz.create_equity_curve(equity_curve, show_drawdown=True)
        
        # Returns distribution
        returns = equity_curve.pct_change().dropna()
        figures['returns_dist'] = viz.create_returns_distribution(returns, show_stats=True)
        
        # Rolling metrics
        figures['rolling'] = viz.create_rolling_metrics(returns, window=30)
        
        # Trade distribution
        figures['trade_dist'] = viz.create_trade_distribution(trades, show_stats=True)
        
        # Trade timeline
        figures['trade_timeline'] = viz.create_trade_timeline(trades, color_by='side')
        
        # Monthly returns heatmap
        figures['monthly'] = viz.create_monthly_returns_heatmap(equity_curve)
        
        # Win/loss chart
        figures['win_loss'] = viz.create_win_loss_chart(trades)
        
        # Duration histogram
        figures['duration'] = viz.create_trade_duration_histogram(trades)
        
        # Verify all figures were created successfully
        for name, fig in figures.items():
            assert isinstance(fig, go.Figure), f"Figure {name} is not a valid Plotly figure"
            assert len(fig.data) > 0, f"Figure {name} has no data"
        
        # Test dashboard creation
        dashboard = ComprehensiveTradingDashboard()
        html_content = dashboard.create_dashboard(
            price_data=price_data,
            equity_curve=equity_curve,
            trades=trades,
            symbol='INTEGRATION_TEST'
        )
        
        assert isinstance(html_content, str)
        assert len(html_content) > 0
        assert 'INTEGRATION_TEST' in html_content
    
    def test_visualization_error_handling(self):
        """Test visualization error handling."""
        viz = ReportVisualizations()
        
        # Test with None data
        fig = viz.create_equity_curve(None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        fig = viz.create_trade_distribution(empty_df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        
        # Test with invalid data types
        invalid_data = "invalid_data"
        fig = viz.create_equity_curve(invalid_data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
    
    def test_visualization_performance(self):
        """Test visualization performance with large datasets."""
        # Create large dataset
        dates = pd.date_range('2020-01-01', periods=5000, freq='D')
        large_equity = pd.Series(
            100000 * (1 + np.random.normal(0.001, 0.02, len(dates))).cumprod(),
            index=dates
        )
        
        viz = ReportVisualizations()
        
        import time
        start_time = time.time()
        
        # Create multiple visualizations
        fig1 = viz.create_equity_curve(large_equity)
        fig2 = viz.create_returns_distribution(large_equity.pct_change().dropna())
        fig3 = viz.create_monthly_returns_heatmap(large_equity)
        
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 10  # Less than 10 seconds
        
        # All figures should be valid
        for fig in [fig1, fig2, fig3]:
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0
    
    def test_theme_consistency(self):
        """Test theme consistency across visualizations."""
        # Test with different themes
        themes = ['plotly', 'plotly_dark', 'plotly_white', 'ggplot2', 'seaborn']
        
        for theme in themes:
            config = VisualizationConfig(theme=theme)
            viz = ReportVisualizations(config)
            
            # Create sample data
            dates = pd.date_range('2023-01-01', periods=100, freq='D')
            equity = pd.Series(
                100000 * (1 + np.random.normal(0.001, 0.02, len(dates))).cumprod(),
                index=dates
            )
            
            # Create figure
            fig = viz.create_equity_curve(equity)
            
            # Apply theme
            themed_fig = viz.apply_theme(fig, theme)
            
            assert isinstance(themed_fig, go.Figure)
            assert len(themed_fig.data) > 0
            
            # Check that theme was applied
            if theme == 'plotly_dark':
                assert themed_fig.layout.template.layout.plot_bgcolor == 'rgb(17,17,17)'
            elif theme == 'plotly_white':
                assert themed_fig.layout.template.layout.plot_bgcolor == 'white'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])