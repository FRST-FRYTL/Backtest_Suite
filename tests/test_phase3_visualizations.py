"""
Tests for Phase 3 Visualization Components

This module tests all visualization and reporting components
created in Phase 3 of the enhanced confluence strategy.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import visualization modules
from visualization.multi_timeframe_chart import MultiTimeframeMasterChart
from visualization.confluence_charts import ConfluenceCharts
from visualization.trade_explorer import InteractiveTradeExplorer
from visualization.timeframe_charts import TimeframeCharts
from visualization.executive_summary import ExecutiveSummaryDashboard
from visualization.performance_report import PerformanceAnalysisReport
from visualization.benchmark_comparison import BenchmarkComparison
from visualization.export_utils import ExportManager

# Import data structures
from data.multi_timeframe_data_manager import Timeframe

class TestPhase3Visualizations:
    """Test suite for Phase 3 visualization components."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        # Date range
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        
        # Multi-timeframe data
        data_by_timeframe = {}
        indicators_by_timeframe = {}
        
        for timeframe in [Timeframe.DAY_1, Timeframe.WEEK_1]:
            # Create OHLCV data
            prices = 100 + np.random.randn(len(dates)).cumsum()
            ohlcv = pd.DataFrame({
                'open': prices + np.random.randn(len(dates)) * 0.5,
                'high': prices + abs(np.random.randn(len(dates))),
                'low': prices - abs(np.random.randn(len(dates))),
                'close': prices,
                'volume': np.random.randint(1000000, 5000000, len(dates))
            }, index=dates)
            
            data_by_timeframe[timeframe] = ohlcv
            
            # Create indicators
            indicators_by_timeframe[timeframe] = {
                'sma_20': ohlcv['close'].rolling(20).mean(),
                'sma_50': ohlcv['close'].rolling(50).mean(),
                'bb_upper': ohlcv['close'].rolling(20).mean() + 2 * ohlcv['close'].rolling(20).std(),
                'bb_lower': ohlcv['close'].rolling(20).mean() - 2 * ohlcv['close'].rolling(20).std()
            }
        
        # Sample trades
        trades = []
        for i in range(20):
            entry_date = dates[i * 10]
            exit_date = entry_date + timedelta(days=np.random.randint(5, 20))
            
            trade = {
                'trade_id': f'T{i+1:03d}',
                'symbol': 'TEST',
                'entry_time': entry_date,
                'exit_time': exit_date,
                'entry_price': 100 + i,
                'exit_price': 100 + i + np.random.randn() * 5,
                'position_size': 0.1,
                'return': np.random.randn() * 0.05,
                'pnl': np.random.randn() * 100,
                'confluence_score': 0.5 + np.random.rand() * 0.4,
                'hold_days': (exit_date - entry_date).days,
                'exit_reason': np.random.choice(['take_profit', 'stop_loss', 'signal']),
                'timeframe_scores': {
                    '1H': np.random.rand(),
                    '4H': np.random.rand(),
                    '1D': np.random.rand(),
                    '1W': np.random.rand(),
                    '1M': np.random.rand()
                }
            }
            trade['return'] = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
            trades.append(trade)
        
        # Equity curve
        equity_curve = pd.Series(
            10000 * (1 + np.random.randn(len(dates)) * 0.01).cumprod(),
            index=dates
        )
        
        # Strategy metrics
        metrics = {
            'total_return': 0.25,
            'annual_return': 0.25,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.10,
            'volatility': 0.16,
            'win_rate': 0.60,
            'total_trades': len(trades),
            'profit_factor': 1.8
        }
        
        return {
            'data_by_timeframe': data_by_timeframe,
            'indicators_by_timeframe': indicators_by_timeframe,
            'trades': trades,
            'equity_curve': equity_curve,
            'metrics': metrics
        }
    
    def test_multi_timeframe_chart(self, sample_data):
        """Test multi-timeframe master chart creation."""
        chart_creator = MultiTimeframeMasterChart()
        
        # Create master chart
        fig = chart_creator.create_master_chart(
            sample_data['data_by_timeframe'],
            sample_data['indicators_by_timeframe'],
            sample_data['trades'],
            'TEST'
        )
        
        # Verify figure created
        assert fig is not None
        assert len(fig.data) > 0
        assert fig.layout.title.text == 'TEST - Multi-Timeframe Analysis'
    
    def test_confluence_charts(self, sample_data):
        """Test confluence analysis charts."""
        confluence_charts = ConfluenceCharts()
        
        # Create confluence history
        dates = sample_data['equity_curve'].index[::10]
        confluence_history = pd.DataFrame({
            '1D': np.random.rand(len(dates)),
            '1W': np.random.rand(len(dates)),
            '1M': np.random.rand(len(dates))
        }, index=dates)
        
        # Test heatmap
        heatmap = confluence_charts.create_confluence_heatmap(
            confluence_history,
            {'1D': 0.5, '1W': 0.3, '1M': 0.2}
        )
        assert heatmap is not None
        
        # Test distribution
        scores = pd.Series([t['confluence_score'] for t in sample_data['trades']])
        returns = pd.Series([t['return'] for t in sample_data['trades']])
        
        dist_fig = confluence_charts.create_confluence_distribution(scores, returns)
        assert dist_fig is not None
    
    def test_trade_explorer(self, sample_data):
        """Test interactive trade explorer."""
        explorer = InteractiveTradeExplorer()
        
        # Test trade table
        table_fig = explorer.create_trade_table(sample_data['trades'])
        assert table_fig is not None
        assert len(table_fig.data) == 1  # One table
        
        # Test scatter matrix
        scatter_fig = explorer.create_trade_scatter_matrix(sample_data['trades'])
        assert scatter_fig is not None
        
        # Test performance summary
        summary_fig = explorer.create_trade_performance_summary(sample_data['trades'])
        assert summary_fig is not None
    
    def test_timeframe_charts(self, sample_data):
        """Test timeframe participation visualizations."""
        tf_charts = TimeframeCharts()
        
        # Test participation chart
        participation_fig = tf_charts.create_timeframe_participation_chart(
            sample_data['trades']
        )
        assert participation_fig is not None
        
        # Test importance pie
        pie_fig = tf_charts.create_timeframe_importance_pie(sample_data['trades'])
        assert pie_fig is not None
        
        # Test timeline
        timeline_fig = tf_charts.create_participation_timeline(sample_data['trades'])
        assert timeline_fig is not None
    
    def test_executive_summary(self, sample_data):
        """Test executive summary dashboard."""
        dashboard = ExecutiveSummaryDashboard()
        
        # Create benchmark results
        benchmark_results = {
            'total_return': 0.20,
            'annual_return': 0.20,
            'sharpe_ratio': 1.0,
            'max_drawdown': -0.15,
            'volatility': 0.20,
            'equity_curve': sample_data['equity_curve'] * 0.9
        }
        
        # Create dashboard
        html_content = dashboard.create_dashboard(
            sample_data['metrics'],
            benchmark_results,
            sample_data['trades'],
            sample_data['equity_curve']
        )
        
        # Verify HTML content
        assert html_content is not None
        assert '<html>' in html_content
        assert 'Executive Summary' in html_content
    
    def test_performance_report(self, sample_data):
        """Test performance analysis reports."""
        report = PerformanceAnalysisReport()
        
        # Create reports
        figs = report.create_performance_report(
            sample_data['trades'],
            sample_data['equity_curve'],
            sample_data['metrics']
        )
        
        # Verify all expected figures created
        expected_figs = [
            'trade_stats', 'returns_heatmap', 'drawdown_analysis',
            'win_loss_dist', 'rolling_metrics', 'duration_analysis',
            'market_conditions'
        ]
        
        for fig_name in expected_figs:
            assert fig_name in figs
            assert figs[fig_name] is not None
    
    def test_benchmark_comparison(self, sample_data):
        """Test benchmark comparison system."""
        comparison = BenchmarkComparison()
        
        # Create benchmark results
        benchmark_results = {
            'buy_hold': {
                'total_return': 0.20,
                'annual_return': 0.20,
                'sharpe_ratio': 1.0,
                'max_drawdown': -0.15,
                'volatility': 0.20,
                'equity_curve': sample_data['equity_curve'] * 0.9
            }
        }
        
        # Add equity curve to strategy results
        strategy_results = {
            **sample_data['metrics'],
            'equity_curve': sample_data['equity_curve']
        }
        
        # Create comparisons
        figs = comparison.create_benchmark_comparison(
            strategy_results,
            benchmark_results
        )
        
        # Verify figures created
        assert 'metrics_comparison' in figs
        assert 'relative_performance' in figs
        assert 'alpha_chart' in figs
    
    def test_export_manager(self, sample_data, tmp_path):
        """Test export functionality."""
        # Use temporary directory for testing
        export_mgr = ExportManager(str(tmp_path))
        
        # Test CSV export
        csv_path = export_mgr.export_trades_csv(sample_data['trades'])
        assert Path(csv_path).exists()
        
        # Test metrics export
        metrics_path = export_mgr.export_performance_metrics_csv(
            sample_data['metrics']
        )
        assert Path(metrics_path).exists()
        
        # Test JSON export
        json_path = export_mgr.export_json_data({
            'metrics': sample_data['metrics'],
            'trades': sample_data['trades']
        })
        assert Path(json_path).exists()
    
    def test_chart_saving(self, sample_data, tmp_path):
        """Test chart saving functionality."""
        chart_creator = MultiTimeframeMasterChart()
        
        # Create and save chart
        fig = chart_creator.create_master_chart(
            sample_data['data_by_timeframe'],
            sample_data['indicators_by_timeframe'],
            sample_data['trades'],
            'TEST'
        )
        
        # Save to temp directory
        save_path = chart_creator.save_chart(
            fig, 
            'test_chart.html',
            str(tmp_path)
        )
        
        assert Path(save_path).exists()
        assert save_path.endswith('.html')
    
    def test_error_handling(self):
        """Test error handling in visualization components."""
        # Test with empty data
        chart_creator = MultiTimeframeMasterChart()
        explorer = InteractiveTradeExplorer()
        
        # Empty trades should not crash
        empty_trades = []
        fig = explorer.create_trade_table(empty_trades)
        assert fig is not None
        
        # Missing data should be handled gracefully
        incomplete_trade = [{
            'trade_id': 'T001',
            'return': 0.05
            # Missing other fields
        }]
        
        fig = explorer.create_trade_performance_summary(incomplete_trade)
        assert fig is not None

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])