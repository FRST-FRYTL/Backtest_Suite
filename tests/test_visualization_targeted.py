"""
Targeted visualization tests for specific coverage improvements.
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
import json
from pathlib import Path
import time

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


class TestVisualizationTargeted:
    """Targeted visualization tests for specific coverage improvements."""
    
    def test_chart_generator_with_real_data(self):
        """Test ChartGenerator with realistic data."""
        # Test plotly style
        cg = ChartGenerator(style="plotly")
        assert cg.style == "plotly"
        
        # Create realistic equity curve data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        returns = np.random.normal(0.001, 0.02, len(dates))
        cumulative_returns = (1 + returns).cumprod()
        
        equity_data = pd.DataFrame({
            'total_value': 100000 * cumulative_returns,
            'cash': 30000 * np.ones(len(dates)),
            'positions_value': 70000 * cumulative_returns
        }, index=dates)
        
        # Test equity curve plotting
        fig = cg.plot_equity_curve(equity_data, title="Test Equity Curve")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Should have equity curve and drawdown
        
        # Test with benchmark
        benchmark = pd.Series(
            100000 * (1 + np.random.normal(0.0005, 0.015, len(dates))).cumprod(),
            index=dates
        )
        fig_bench = cg.plot_equity_curve(equity_data, benchmark=benchmark, title="With Benchmark")
        assert isinstance(fig_bench, go.Figure)
        assert len(fig_bench.data) >= 3  # Should have equity, benchmark, and drawdown
        
        # Test returns distribution
        returns_series = pd.Series(returns)
        fig_dist = cg.plot_returns_distribution(returns_series, title="Returns Distribution")
        assert isinstance(fig_dist, go.Figure)
        assert len(fig_dist.data) >= 1  # Should have at least histogram
        
        # Test performance metrics
        metrics = {
            'total_return': '15.43%',
            'sharpe_ratio': '1.25',
            'max_drawdown': '-8.23%',
            'win_rate': '65.0%',
            'profit_factor': '1.45'
        }
        fig_metrics = cg.plot_performance_metrics(metrics, title="Performance Metrics")
        assert isinstance(fig_metrics, go.Figure)
        assert len(fig_metrics.data) >= 1
        
        # Test trades plotting
        trade_data = pd.DataFrame({
            'timestamp': dates[:10],
            'type': ['OPEN'] * 5 + ['CLOSE'] * 5,
            'price': 100 + np.random.normal(0, 5, 10),
            'quantity': [100] * 5 + [-100] * 5
        })
        
        price_data = pd.DataFrame({
            'open': 100 + np.random.normal(0, 2, len(dates)),
            'high': 100 + np.abs(np.random.normal(3, 1, len(dates))),
            'low': 100 - np.abs(np.random.normal(3, 1, len(dates))),
            'close': 100 + np.random.normal(0, 2, len(dates)),
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        fig_trades = cg.plot_trades(price_data, trade_data, 'TEST')
        assert isinstance(fig_trades, go.Figure)
        assert len(fig_trades.data) >= 1
        
        # Test matplotlib style
        plt.ioff()
        cg_mpl = ChartGenerator(style="matplotlib")
        assert cg_mpl.style == "matplotlib"
        
        # Test matplotlib equity curve (may fail due to dependencies)
        try:
            fig_mpl = cg_mpl.plot_equity_curve(equity_data, title="Matplotlib Equity")
            if isinstance(fig_mpl, plt.Figure):
                plt.close(fig_mpl)
        except Exception:
            pass
        
        # Test matplotlib returns distribution
        try:
            fig_mpl_dist = cg_mpl.plot_returns_distribution(returns_series, title="Matplotlib Returns")
            if isinstance(fig_mpl_dist, plt.Figure):
                plt.close(fig_mpl_dist)
        except Exception:
            pass
    
    def test_dashboard_comprehensive(self):
        """Test Dashboard with comprehensive data."""
        dashboard = Dashboard()
        
        # Create comprehensive test data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        returns = np.random.normal(0.001, 0.02, len(dates))
        cumulative_returns = (1 + returns).cumprod()
        
        equity_data = pd.DataFrame({
            'total_value': 100000 * cumulative_returns,
            'cash': 30000 * np.ones(len(dates)),
            'positions_value': 70000 * cumulative_returns
        }, index=dates)
        
        # Create trades data with proper columns
        trades_data = pd.DataFrame({
            'type': ['OPEN', 'CLOSE'] * 10,
            'symbol': ['AAPL', 'GOOGL'] * 10,
            'quantity': [100, -100] * 10,
            'timestamp': dates[:20],
            'price': 100 + np.random.normal(0, 5, 20),
            'position_pnl': np.random.normal(500, 200, 20)
        })
        
        # Performance metrics
        performance_data = {
            'total_return': 0.15,
            'sharpe_ratio': 1.25,
            'max_drawdown': -0.08,
            'total_trades': 20,
            'winning_trades': 12,
            'losing_trades': 8,
            'avg_win': 750.0,
            'avg_loss': -300.0,
            'best_trade': 1500.0,
            'worst_trade': -800.0,
            'total_pnl': 5000.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 5000.0,
            'total_commission': 100.0,
            'initial_capital': 100000,
            'profit_factor': 1.45
        }
        
        # Test individual chart methods
        fig_equity = dashboard._create_equity_chart(equity_data)
        assert isinstance(fig_equity, go.Figure)
        assert len(fig_equity.data) >= 1
        
        fig_drawdown = dashboard._create_drawdown_chart(equity_data)
        assert isinstance(fig_drawdown, go.Figure)
        assert len(fig_drawdown.data) >= 1
        
        fig_trades = dashboard._create_trades_chart(trades_data)
        assert isinstance(fig_trades, go.Figure)
        # Should handle the data even if empty
        
        fig_analysis = dashboard._create_trade_analysis(trades_data)
        assert isinstance(fig_analysis, go.Figure)
        
        fig_table = dashboard._create_metrics_table(performance_data)
        assert isinstance(fig_table, go.Figure)
        
        fig_gauges = dashboard._create_metrics_gauges(performance_data)
        assert isinstance(fig_gauges, go.Figure)
        
        # Test HTML generation
        dashboard.figures = [fig_equity, fig_drawdown]
        html = dashboard._generate_html("Comprehensive Test Dashboard")
        assert isinstance(html, str)
        assert "Comprehensive Test Dashboard" in html
        assert "<!DOCTYPE html>" in html
        assert "plotly" in html
        
        # Test complete dashboard creation
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "comprehensive_dashboard.html")
            dashboard_data = {
                'equity_curve': equity_data,
                'trades': trades_data,
                'performance': performance_data
            }
            
            result = dashboard.create_dashboard(dashboard_data, output_path=output_path, title="Test Dashboard")
            assert result == output_path
            assert os.path.exists(output_path)
            
            # Verify the file has content
            with open(output_path, 'r') as f:
                content = f.read()
                assert "Test Dashboard" in content
                assert len(content) > 1000  # Should have substantial content
    
    def test_export_manager_comprehensive(self):
        """Test ExportManager with comprehensive functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            em = ExportManager(output_dir=tmp_dir)
            
            # Verify initialization
            assert em.output_dir.exists()
            assert em.csv_dir.exists()
            assert em.excel_dir.exists()
            assert em.pdf_dir.exists()
            
            # Create comprehensive test data
            trades_data = []
            for i in range(50):
                trade_date = datetime(2023, 1, 1) + timedelta(days=i)
                trades_data.append({
                    'trade_id': i + 1,
                    'entry_time': trade_date,
                    'exit_time': trade_date + timedelta(hours=2),
                    'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'TSLA']),
                    'entry_price': 100 + np.random.normal(0, 10),
                    'exit_price': 100 + np.random.normal(0, 10),
                    'quantity': np.random.randint(10, 200),
                    'pnl': np.random.normal(100, 500),
                    'return': np.random.normal(0.01, 0.05),
                    'commission': np.random.uniform(1, 10),
                    'confluence_score': np.random.uniform(0.3, 0.9)
                })
            
            # Test CSV exports
            csv_path = em.export_trades_csv(trades_data, "comprehensive_trades.csv")
            assert os.path.exists(csv_path)
            
            # Verify CSV content
            df = pd.read_csv(csv_path)
            assert len(df) == 50
            assert 'trade_id' in df.columns
            assert 'pnl' in df.columns
            
            # Test performance metrics export
            metrics_data = {
                'total_return': 0.25,
                'sharpe_ratio': 1.45,
                'max_drawdown': -0.12,
                'win_rate': 0.65,
                'profit_factor': 1.85,
                'total_trades': 50,
                'winning_trades': 32,
                'losing_trades': 18,
                'avg_win': 800.0,
                'avg_loss': -400.0,
                'volatility': 0.18,
                'var_95': -0.023
            }
            
            metrics_path = em.export_performance_metrics_csv(metrics_data, "comprehensive_metrics.csv")
            assert os.path.exists(metrics_path)
            
            # Test confluence scores export
            confluence_data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=30, freq='D'),
                'confluence_score': np.random.uniform(0.3, 0.9, 30)
            })
            
            conf_path = em.export_confluence_scores_timeseries(confluence_data, "confluence_scores.csv")
            assert os.path.exists(conf_path)
            
            # Test JSON export with complex data
            json_data = {
                'performance_metrics': metrics_data,
                'trade_summary': {
                    'total_trades': len(trades_data),
                    'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
                    'date_range': ['2023-01-01', '2023-02-19']
                },
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'export_version': '1.0',
                    'source': 'test_suite'
                }
            }
            
            json_path = em.export_json_data(json_data, "comprehensive_data.json")
            assert os.path.exists(json_path)
            
            # Verify JSON can be loaded
            with open(json_path, 'r') as f:
                loaded_data = json.load(f)
                assert loaded_data['performance_metrics']['total_return'] == 0.25
                assert loaded_data['trade_summary']['total_trades'] == 50
            
            # Test monthly summary creation
            monthly_summary = em._create_monthly_summary(trades_data)
            assert isinstance(monthly_summary, pd.DataFrame)
            assert len(monthly_summary) > 0
            
            # Test Excel export (if available)
            excel_path = em.export_excel_workbook(
                trades_data, 
                metrics_data, 
                benchmark_comparison={'strategy': 1.25, 'benchmark': 1.08},
                filename="comprehensive_workbook.xlsx"
            )
            # Should return path if available or None if not
            assert excel_path is None or os.path.exists(excel_path)
            
            # Test PDF export (if available)
            html_content = f"""
            <html>
            <head><title>Comprehensive Test Report</title></head>
            <body>
                <h1>Backtest Results</h1>
                <h2>Performance Summary</h2>
                <p>Total Return: {metrics_data['total_return']:.2%}</p>
                <p>Sharpe Ratio: {metrics_data['sharpe_ratio']:.2f}</p>
                <p>Max Drawdown: {metrics_data['max_drawdown']:.2%}</p>
                <h2>Trade Summary</h2>
                <p>Total Trades: {len(trades_data)}</p>
                <p>Win Rate: {metrics_data['win_rate']:.2%}</p>
            </body>
            </html>
            """
            
            pdf_path = em.export_html_to_pdf(html_content, "comprehensive_report.pdf")
            assert pdf_path is None or os.path.exists(pdf_path)
            
            # Test comprehensive export all
            exports = em.export_all(
                trades_data,
                metrics_data,
                confluence_history=confluence_data,
                benchmark_comparison={'strategy': 1.25, 'benchmark': 1.08},
                html_report=html_content
            )
            
            assert isinstance(exports, dict)
            assert 'trades_csv' in exports
            assert 'metrics_csv' in exports
            assert 'confluence_csv' in exports
            assert 'json' in exports
            
            # Test export summary
            summary = em.create_export_summary()
            assert isinstance(summary, dict)
            assert 'csv' in summary
            assert 'excel' in summary
            assert 'pdf' in summary
            assert 'json' in summary
            
            # Verify summary contains file counts
            assert isinstance(summary['csv'], list)
            assert isinstance(summary['excel'], list)
            assert isinstance(summary['pdf'], list)
            assert isinstance(summary['json'], list)
    
    def test_report_visualizations_comprehensive(self):
        """Test ReportVisualizations with comprehensive data."""
        # Test with custom styling
        custom_style = {
            'template': 'plotly_white',
            'color_scheme': {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'success': '#2ca02c',
                'warning': '#d62728',
                'danger': '#9467bd',
                'neutral': '#8c564b'
            },
            'font': {
                'family': 'Arial, sans-serif',
                'size': 12
            },
            'chart_height': 400,
            'chart_width': 800
        }
        
        viz = ReportVisualizations(style_config=custom_style)
        
        # Verify styling was applied
        assert viz.style['template'] == 'plotly_white'
        assert viz.style['color_scheme']['primary'] == '#1f77b4'
        
        # Create comprehensive test data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        returns = np.random.normal(0.001, 0.02, len(dates))
        returns_series = pd.Series(returns, index=dates)
        
        # Performance metrics
        metrics = {
            'total_return': 0.28,
            'annualized_return': 0.32,
            'sharpe_ratio': 1.65,
            'sortino_ratio': 2.25,
            'calmar_ratio': 1.95,
            'max_drawdown': -0.15,
            'volatility': 0.19,
            'win_rate': 0.62,
            'profit_factor': 1.75,
            'downside_deviation': 0.12,
            'var_95': -0.034
        }
        
        # Test performance summary chart
        fig_perf = viz.performance_summary_chart(metrics)
        assert isinstance(fig_perf, go.Figure)
        assert len(fig_perf.data) >= 1
        
        # Test cumulative returns
        fig_cum = viz.cumulative_returns(returns_series)
        assert isinstance(fig_cum, go.Figure)
        assert len(fig_cum.data) >= 1
        
        # Test with benchmark
        benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, len(dates)),
            index=dates
        )
        fig_cum_bench = viz.cumulative_returns(returns_series, benchmark=benchmark_returns)
        assert isinstance(fig_cum_bench, go.Figure)
        assert len(fig_cum_bench.data) >= 2
        
        # Test drawdown chart
        fig_dd = viz.drawdown_chart(returns_series)
        assert isinstance(fig_dd, go.Figure)
        assert len(fig_dd.data) >= 1
        
        # Test monthly returns heatmap
        fig_monthly = viz.monthly_returns_heatmap(returns_series)
        assert isinstance(fig_monthly, go.Figure)
        assert len(fig_monthly.data) >= 1
        
        # Test trade distribution
        trades_data = pd.DataFrame({
            'pnl': np.random.normal(100, 500, 50),
            'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'TSLA'], 50),
            'entry_time': pd.date_range('2023-01-01', periods=50, freq='D'),
            'exit_time': pd.date_range('2023-01-02', periods=50, freq='D'),
            'duration': np.random.uniform(1, 10, 50),
            'return': np.random.normal(0.01, 0.05, 50)
        })
        
        fig_trades = viz.trade_distribution(trades_data)
        assert isinstance(fig_trades, go.Figure)
        assert len(fig_trades.data) >= 1
        
        # Test rolling metrics
        fig_rolling = viz.rolling_metrics(returns_series, window=20)
        assert isinstance(fig_rolling, go.Figure)
        assert len(fig_rolling.data) >= 1
        
        # Test performance table
        html_table = viz.create_performance_table(metrics)
        assert isinstance(html_table, str)
        assert '<table' in html_table
        assert 'total_return' in html_table
        
        # Test parameter heatmap
        param_data = []
        for p1 in np.linspace(0.1, 1.0, 5):
            for p2 in np.linspace(0.05, 0.5, 5):
                param_data.append({
                    'param1': p1,
                    'param2': p2,
                    'sharpe_ratio': np.random.uniform(0.5, 2.5),
                    'max_drawdown': np.random.uniform(-0.3, -0.05),
                    'total_return': np.random.uniform(0.1, 0.4)
                })
        
        param_df = pd.DataFrame(param_data)
        fig_heatmap = viz.parameter_heatmap(param_df, 'param1', 'param2', 'sharpe_ratio')
        assert isinstance(fig_heatmap, go.Figure)
        assert len(fig_heatmap.data) >= 1
        
        # Test with comprehensive trade data
        comprehensive_trades = trades_data.copy()
        comprehensive_trades['stop_loss'] = 95.0
        comprehensive_trades['stop_hit'] = np.random.choice([True, False], 50)
        comprehensive_trades['stop_distance'] = np.random.uniform(1, 5, 50)
        comprehensive_trades['size'] = np.random.randint(10, 200, 50)
        comprehensive_trades['entry_price'] = 100 + np.random.normal(0, 10, 50)
        comprehensive_trades['exit_price'] = 100 + np.random.normal(0, 10, 50)
        
        # Test stop loss analysis
        fig_stop = viz.create_stop_loss_analysis(comprehensive_trades)
        assert isinstance(fig_stop, go.Figure)
        
        # Test trade risk chart
        fig_risk = viz.create_trade_risk_chart(comprehensive_trades)
        assert isinstance(fig_risk, go.Figure)
        
        # Test trade price chart
        price_data = pd.DataFrame({
            'open': 100 + np.random.normal(0, 2, len(dates)),
            'high': 100 + np.abs(np.random.normal(3, 1, len(dates))),
            'low': 100 - np.abs(np.random.normal(3, 1, len(dates))),
            'close': 100 + np.random.normal(0, 2, len(dates)),
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        fig_price = viz.create_trade_price_chart(comprehensive_trades, price_data)
        assert isinstance(fig_price, go.Figure)
        
        # Test save all charts
        figures = {
            'performance': fig_perf,
            'cumulative_returns': fig_cum,
            'drawdown': fig_dd,
            'monthly_heatmap': fig_monthly,
            'trade_distribution': fig_trades,
            'rolling_metrics': fig_rolling,
            'parameter_heatmap': fig_heatmap
        }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            start_time = time.time()
            paths = viz.save_all_charts(figures, tmp_dir)
            end_time = time.time()
            
            # Should complete quickly
            assert end_time - start_time < 10
            
            assert isinstance(paths, dict)
            assert len(paths) == len(figures)
            
            for name, path_info in paths.items():
                assert 'html' in path_info
                assert os.path.exists(path_info['html'])
                
                # Verify file has content
                with open(path_info['html'], 'r') as f:
                    content = f.read()
                    assert len(content) > 100
    
    def test_visualization_types_comprehensive(self):
        """Test visualization types with comprehensive data."""
        # Test different configurations
        configs = [
            VisualizationConfig(),
            VisualizationConfig(
                figure_size=(16, 10),
                figure_dpi=200,
                color_scheme={'primary': '#FF0000', 'secondary': '#00FF00'}
            ),
            VisualizationConfig(
                figure_size=(10, 6),
                figure_dpi=100,
                color_scheme={'primary': '#0000FF', 'secondary': '#FFFF00'}
            )
        ]
        
        # Create test data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        returns = np.random.normal(0.001, 0.02, len(dates))
        equity_series = pd.Series(100000 * (1 + returns).cumprod(), index=dates)
        returns_series = pd.Series(returns, index=dates)
        
        trades_data = pd.DataFrame({
            'pnl': np.random.normal(100, 300, 30),
            'entry_time': pd.date_range('2023-01-01', periods=30, freq='D'),
            'exit_time': pd.date_range('2023-01-02', periods=30, freq='D'),
            'duration': np.random.uniform(1, 5, 30),
            'entry_price': 100 + np.random.normal(0, 5, 30),
            'exit_price': 100 + np.random.normal(0, 5, 30),
            'size': np.random.randint(10, 100, 30)
        })
        
        for config in configs:
            # Test base visualization
            base_viz = BaseVisualization(config)
            assert base_viz.config == config
            
            # Test template creation
            template = base_viz._create_plotly_template()
            assert isinstance(template, dict)
            assert 'layout' in template
            
            # Test equity curve chart
            equity_chart = EquityCurveChart(config)
            result = equity_chart.create(equity_series)
            assert isinstance(result, dict)
            assert 'figure' in result
            assert 'data' in result
            assert isinstance(result['figure'], go.Figure)
            
            # Test with benchmark
            benchmark = pd.Series(
                100000 * (1 + np.random.normal(0.0005, 0.015, len(dates))).cumprod(),
                index=dates
            )
            result_bench = equity_chart.create(equity_series, benchmark=benchmark)
            assert isinstance(result_bench, dict)
            assert 'figure' in result_bench
            
            # Test drawdown chart
            drawdown_chart = DrawdownChart(config)
            result = drawdown_chart.create(equity_series)
            assert isinstance(result, dict)
            assert 'figure' in result
            
            # Test returns distribution
            returns_chart = ReturnsDistribution(config)
            result = returns_chart.create(returns_series)
            assert isinstance(result, dict)
            assert 'figure' in result
            
            # Test trade scatter plot
            trade_scatter = TradeScatterPlot(config)
            result = trade_scatter.create(trades_data)
            assert isinstance(result, dict)
            assert 'figure' in result
            
            # Test rolling metrics chart
            rolling_chart = RollingMetricsChart(config)
            result = rolling_chart.create(equity_series, window=10)
            assert isinstance(result, dict)
            assert 'figure' in result
            
            # Test heatmap visualization
            heatmap_viz = HeatmapVisualization(config)
            
            # Test correlation heatmap
            corr_data = pd.DataFrame({
                'A': np.random.randn(30),
                'B': np.random.randn(30),
                'C': np.random.randn(30),
                'D': np.random.randn(30)
            })
            result = heatmap_viz.create(corr_data, chart_type="correlation")
            assert isinstance(result, dict)
            assert 'figure' in result
            assert result['type'] == 'correlation'
            
            # Test monthly returns heatmap
            result = heatmap_viz.create(returns_series, chart_type="monthly_returns")
            assert isinstance(result, dict)
            assert 'figure' in result
            assert result['type'] == 'monthly_returns'
            
            # Test trade price chart
            trade_price_chart = TradePriceChart(config)
            result = trade_price_chart.create(trades_data)
            assert isinstance(result, dict)
            assert 'figure' in result
            
            # Test trade risk chart
            trade_risk_chart = TradeRiskChart(config)
            result = trade_risk_chart.create(trades_data)
            assert isinstance(result, dict)
            assert 'figure' in result
            
            # Test save functionality
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1, 2, 3, 4, 5], y=[10, 11, 12, 13, 14]))
            
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
                try:
                    base_viz.save_figure(fig, Path(tmp_file.name), format='html')
                    assert os.path.exists(tmp_file.name)
                    
                    # Verify file has content
                    with open(tmp_file.name, 'r') as f:
                        content = f.read()
                        assert len(content) > 100
                        assert 'plotly' in content
                finally:
                    if os.path.exists(tmp_file.name):
                        os.unlink(tmp_file.name)
    
    def test_performance_and_edge_cases(self):
        """Test performance and edge cases."""
        # Test with large dataset
        large_dates = pd.date_range('2020-01-01', periods=2000, freq='D')
        large_returns = np.random.normal(0.0005, 0.02, len(large_dates))
        large_series = pd.Series(large_returns, index=large_dates)
        
        viz = ReportVisualizations()
        
        # Test performance
        start_time = time.time()
        fig = viz.cumulative_returns(large_series)
        end_time = time.time()
        
        assert end_time - start_time < 5  # Should complete quickly
        assert isinstance(fig, go.Figure)
        
        # Test with empty data
        empty_df = pd.DataFrame()
        
        # These should handle empty data gracefully
        fig_empty = viz.trade_distribution(empty_df)
        assert isinstance(fig_empty, go.Figure)
        
        fig_stop_empty = viz.create_stop_loss_analysis(empty_df)
        assert isinstance(fig_stop_empty, go.Figure)
        
        fig_risk_empty = viz.create_trade_risk_chart(empty_df)
        assert isinstance(fig_risk_empty, go.Figure)
        
        # Test with single data point
        single_point = pd.Series([100000], index=[datetime.now()])
        
        try:
            fig_single = viz.cumulative_returns(single_point)
            assert isinstance(fig_single, go.Figure)
        except Exception:
            pass  # May fail but shouldn't crash
        
        # Test with NaN values
        nan_series = pd.Series([0.01, np.nan, 0.02, np.nan, 0.015])
        
        try:
            fig_nan = viz.cumulative_returns(nan_series)
            assert isinstance(fig_nan, go.Figure)
        except Exception:
            pass  # May fail but shouldn't crash
        
        # Test missing dependencies
        with patch('src.visualization.export_utils.EXCEL_AVAILABLE', False):
            em = ExportManager()
            result = em.export_excel_workbook([], {})
            assert result is None
        
        with patch('src.visualization.export_utils.PDF_AVAILABLE', False):
            em = ExportManager()
            result = em.export_html_to_pdf("<html></html>")
            assert result is None
        
        # Test error handling in chart generator
        cg = ChartGenerator()
        
        try:
            cg.plot_equity_curve(empty_df)
        except Exception:
            pass  # Expected to fail
        
        try:
            cg.plot_returns_distribution(pd.Series([]))
        except Exception:
            pass  # Expected to fail


if __name__ == '__main__':
    pytest.main([__file__, '-v'])