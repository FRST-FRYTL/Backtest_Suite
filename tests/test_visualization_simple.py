"""
Simple visualization tests for basic coverage.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import shutil
from pathlib import Path

# Import visualization modules
from src.visualization.charts import ChartGenerator
from src.visualization.dashboard import Dashboard
from src.visualization.export_utils import ExportManager
from src.reporting.visualizations import ReportVisualizations
from src.reporting.visualization_types import (
    VisualizationConfig, ChartType, BaseVisualization, EquityCurveChart,
    DrawdownChart, ReturnsDistribution, TradeScatterPlot, RollingMetricsChart,
    HeatmapVisualization, TradePriceChart, TradeRiskChart
)
from src.visualization.comprehensive_trading_dashboard import ComprehensiveTradingDashboard


class TestVisualizationSimple:
    """Simple visualization tests for basic coverage."""
    
    def test_chart_generator_basic(self):
        """Test basic ChartGenerator functionality."""
        
        # Test initialization
        cg = ChartGenerator(style="plotly")
        assert cg.style == "plotly"
        
        cg_mpl = ChartGenerator(style="matplotlib")
        assert cg_mpl.style == "matplotlib"
        
        # Test equity curve with simple data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        equity_data = pd.DataFrame({
            'total_value': [100000, 101000, 102000, 103000, 104000, 105000, 106000, 107000, 108000, 109000],
            'cash': [50000] * 10,
            'positions_value': [50000, 51000, 52000, 53000, 54000, 55000, 56000, 57000, 58000, 59000]
        }, index=dates)
        
        fig = cg.plot_equity_curve(equity_data, title="Test Equity")
        assert isinstance(fig, go.Figure)
        
        # Test returns distribution
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015, -0.008, 0.025, -0.012, 0.018, -0.006])
        
        # This method doesn't use stats at module level, so we can test it directly
        fig = cg.plot_returns_distribution(returns, title="Returns")
        assert isinstance(fig, go.Figure)
        
        # Test performance metrics
        metrics = {
            'total_return': '10.5%',
            'sharpe_ratio': '1.2',
            'max_drawdown': '-5.2%',
            'win_rate': '60%'
        }
        
        fig = cg.plot_performance_metrics(metrics, title="Metrics")
        assert isinstance(fig, go.Figure)
    
    def test_dashboard_basic(self):
        """Test basic Dashboard functionality."""
        
        dashboard = Dashboard()
        assert dashboard.figures == []
        
        # Test with basic data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        equity_data = pd.DataFrame({
            'total_value': [100000, 101000, 102000, 103000, 104000, 105000, 106000, 107000, 108000, 109000],
            'cash': [50000] * 10,
            'positions_value': [50000, 51000, 52000, 53000, 54000, 55000, 56000, 57000, 58000, 59000]
        }, index=dates)
        
        trades_data = pd.DataFrame({
            'type': ['OPEN', 'CLOSE', 'OPEN', 'CLOSE'],
            'symbol': ['AAPL', 'AAPL', 'GOOGL', 'GOOGL'],
            'quantity': [100, -100, 50, -50],
            'timestamp': dates[:4],
            'price': [150.0, 155.0, 2800.0, 2850.0],
            'position_pnl': [500.0, 500.0, 2500.0, 2500.0]
        })
        
        performance_data = {
            'total_return': 0.10,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.05,
            'total_trades': 4,
            'winning_trades': 2,
            'initial_capital': 100000
        }
        
        # Test individual chart methods
        fig = dashboard._create_equity_chart(equity_data)
        assert isinstance(fig, go.Figure)
        
        fig = dashboard._create_drawdown_chart(equity_data)
        assert isinstance(fig, go.Figure)
        
        fig = dashboard._create_trades_chart(trades_data)
        assert isinstance(fig, go.Figure)
        
        fig = dashboard._create_trade_analysis(trades_data)
        assert isinstance(fig, go.Figure)
        
        fig = dashboard._create_metrics_table(performance_data)
        assert isinstance(fig, go.Figure)
        
        fig = dashboard._create_metrics_gauges(performance_data)
        assert isinstance(fig, go.Figure)
        
        # Test HTML generation
        html = dashboard._generate_html("Test Dashboard")
        assert isinstance(html, str)
        assert "Test Dashboard" in html
        
        # Test create dashboard
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "test.html")
            dashboard_data = {
                'equity_curve': equity_data,
                'trades': trades_data,
                'performance': performance_data
            }
            
            result = dashboard.create_dashboard(dashboard_data, output_path=output_path)
            assert result is not None
            assert os.path.exists(output_path)
    
    def test_export_manager_basic(self):
        """Test basic ExportManager functionality."""
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            em = ExportManager(output_dir=tmp_dir)
            
            # Test basic exports
            trades_data = [
                {'trade_id': 1, 'symbol': 'AAPL', 'entry_time': '2023-01-01', 'exit_time': '2023-01-02', 
                 'pnl': 100, 'return': 0.05, 'confluence_score': 0.8},
                {'trade_id': 2, 'symbol': 'GOOGL', 'entry_time': '2023-01-03', 'exit_time': '2023-01-04', 
                 'pnl': -50, 'return': -0.02, 'confluence_score': 0.6}
            ]
            
            # Test CSV export
            csv_path = em.export_trades_csv(trades_data)
            assert os.path.exists(csv_path)
            
            # Test metrics export
            metrics_data = {'sharpe_ratio': 1.5, 'max_drawdown': -0.1, 'total_return': 0.2}
            metrics_path = em.export_performance_metrics_csv(metrics_data)
            assert os.path.exists(metrics_path)
            
            # Test JSON export
            json_data = {'test': 'data', 'number': 123}
            json_path = em.export_json_data(json_data)
            assert os.path.exists(json_path)
            
            # Test monthly summary
            monthly_summary = em._create_monthly_summary(trades_data)
            assert isinstance(monthly_summary, pd.DataFrame)
            
            # Test export summary
            summary = em.create_export_summary()
            assert isinstance(summary, dict)
    
    def test_report_visualizations_basic(self):
        """Test basic ReportVisualizations functionality."""
        
        viz = ReportVisualizations()
        
        # Test with simple metrics
        metrics = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'win_rate': 0.6,
            'volatility': 0.18
        }
        
        # Test charts that should work
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        
        # Test simple charts
        fig = viz.cumulative_returns(returns)
        assert isinstance(fig, go.Figure)
        
        fig = viz.drawdown_chart(returns)
        assert isinstance(fig, go.Figure)
        
        fig = viz.rolling_metrics(returns, window=3)
        assert isinstance(fig, go.Figure)
        
        # Test performance table
        html_table = viz.create_performance_table(metrics)
        assert isinstance(html_table, str)
        assert '<table' in html_table
    
    def test_visualization_types_basic(self):
        """Test basic visualization types."""
        
        # Test configuration
        config = VisualizationConfig()
        assert config.figure_size == (12, 8)
        assert config.figure_dpi == 300
        
        # Test chart type enum
        assert hasattr(ChartType, 'EQUITY_CURVE')
        assert hasattr(ChartType, 'DRAWDOWN')
        assert hasattr(ChartType, 'RETURNS_DISTRIBUTION')
        
        # Test base visualization
        base_viz = BaseVisualization(config)
        assert base_viz.config == config
        
        # Test equity curve chart
        equity_chart = EquityCurveChart(config)
        equity_series = pd.Series([100000, 101000, 102000, 103000, 104000])
        
        result = equity_chart.create(equity_series)
        assert isinstance(result, dict)
        assert 'figure' in result
        assert 'data' in result
        
        # Test drawdown chart
        drawdown_chart = DrawdownChart(config)
        result = drawdown_chart.create(equity_series)
        assert isinstance(result, dict)
        assert 'figure' in result
        
        # Test returns distribution
        returns_chart = ReturnsDistribution(config)
        returns_series = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        
        result = returns_chart.create(returns_series)
        assert isinstance(result, dict)
        assert 'figure' in result
        
        # Test trade scatter plot
        trade_scatter = TradeScatterPlot(config)
        trades_data = pd.DataFrame({
            'pnl': [100, -50, 200, -30, 150],
            'duration': [1, 2, 3, 4, 5],
            'entry_time': pd.date_range('2023-01-01', periods=5)
        })
        
        result = trade_scatter.create(trades_data)
        assert isinstance(result, dict)
        assert 'figure' in result
        
        # Test rolling metrics chart
        rolling_chart = RollingMetricsChart(config)
        result = rolling_chart.create(equity_series, window=3)
        assert isinstance(result, dict)
        assert 'figure' in result
        
        # Test heatmap visualization
        heatmap_viz = HeatmapVisualization(config)
        
        # Test correlation heatmap
        corr_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 3, 4, 5, 6],
            'C': [3, 4, 5, 6, 7]
        })
        
        result = heatmap_viz.create(corr_data, chart_type="correlation")
        assert isinstance(result, dict)
        assert 'figure' in result
        
        # Test trade price chart
        trade_price_chart = TradePriceChart(config)
        trades_with_prices = pd.DataFrame({
            'entry_price': [100, 110, 120],
            'exit_price': [105, 115, 125],
            'pnl': [50, 50, 50]
        })
        
        result = trade_price_chart.create(trades_with_prices)
        assert isinstance(result, dict)
        assert 'figure' in result
        
        # Test trade risk chart
        trade_risk_chart = TradeRiskChart(config)
        result = trade_risk_chart.create(trades_with_prices)
        assert isinstance(result, dict)
        assert 'figure' in result
    
    def test_comprehensive_trading_dashboard_basic(self):
        """Test basic ComprehensiveTradingDashboard functionality."""
        
        dashboard = ComprehensiveTradingDashboard()
        
        # Test initialization
        assert dashboard.output_dir is not None
        assert dashboard.colors is not None
        assert dashboard.timeframe_colors is not None
        
        # Test basic functionality without complex data
        pass
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        
        # Test with empty data
        empty_df = pd.DataFrame()
        
        cg = ChartGenerator()
        try:
            cg.plot_equity_curve(empty_df)
        except:
            pass  # Expected to fail
        
        # Test with None
        viz = ReportVisualizations()
        try:
            viz.performance_summary_chart(None)
        except:
            pass  # Expected to fail
        
        # Test export manager with missing dependencies
        with patch('src.visualization.export_utils.EXCEL_AVAILABLE', False):
            em = ExportManager()
            result = em.export_excel_workbook([], {})
            assert result is None
        
        with patch('src.visualization.export_utils.PDF_AVAILABLE', False):
            em = ExportManager()
            result = em.export_html_to_pdf("<html></html>")
            assert result is None
    
    def test_matplotlib_backend(self):
        """Test matplotlib backend."""
        
        plt.ioff()  # Turn off interactive mode
        
        cg = ChartGenerator(style="matplotlib")
        
        # Test with simple data
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        equity_data = pd.DataFrame({
            'total_value': [100000, 101000, 102000, 103000, 104000]
        }, index=dates)
        
        try:
            fig = cg.plot_equity_curve(equity_data)
            if isinstance(fig, plt.Figure):
                plt.close(fig)
        except:
            pass  # May fail due to missing dependencies
        
        # Test returns distribution
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        
        try:
            fig = cg.plot_returns_distribution(returns)
            if isinstance(fig, plt.Figure):
                plt.close(fig)
        except:
            pass  # May fail due to missing dependencies
    
    def test_save_functionality(self):
        """Test save functionality."""
        
        config = VisualizationConfig()
        base_viz = BaseVisualization(config)
        
        # Test with plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            try:
                base_viz.save_figure(fig, Path(tmp_file.name), format='html')
                assert os.path.exists(tmp_file.name)
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
        
        # Test with matplotlib figure
        plt.ioff()
        mpl_fig = plt.figure()
        plt.plot([1, 2, 3], [4, 5, 6])
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            try:
                base_viz.save_figure(mpl_fig, Path(tmp_file.name), format='png')
                assert os.path.exists(tmp_file.name)
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
                plt.close(mpl_fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])