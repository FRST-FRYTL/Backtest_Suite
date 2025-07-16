"""
Tests for Visualization Types

This module tests all visualization components in the reporting system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import plotly.graph_objects as go

from src.reporting import ReportConfig
from src.reporting.visualization_types import (
    EquityCurveChart,
    DrawdownChart,
    ReturnsDistribution,
    TradeScatterPlot,
    RollingMetricsChart,
    HeatmapVisualization
)


class TestBaseVisualization:
    """Test base visualization functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return ReportConfig()
    
    @pytest.fixture
    def sample_equity_curve(self):
        """Create sample equity curve"""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.0003, 0.01, len(dates))
        equity = pd.Series((1 + returns).cumprod() * 100000, index=dates)
        return equity
    
    def test_equity_curve_chart(self, config, sample_equity_curve):
        """Test equity curve visualization"""
        chart = EquityCurveChart(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "equity_curve.png"
            
            result = chart.create(
                equity_curve=sample_equity_curve,
                save_path=save_path
            )
            
            # Check result structure
            assert 'figure' in result
            assert 'data' in result
            assert isinstance(result['figure'], go.Figure)
            
            # Check data calculations
            assert 'final_value' in result['data']
            assert 'peak_value' in result['data']
            assert 'total_return' in result['data']
            
            # Verify calculations
            assert result['data']['final_value'] == sample_equity_curve.iloc[-1]
            assert result['data']['peak_value'] == sample_equity_curve.max()
    
    def test_equity_curve_with_benchmark(self, config, sample_equity_curve):
        """Test equity curve with benchmark comparison"""
        chart = EquityCurveChart(config)
        
        # Create benchmark
        benchmark = sample_equity_curve * 0.9  # Underperform by 10%
        
        result = chart.create(
            equity_curve=sample_equity_curve,
            benchmark=benchmark
        )
        
        # Check that benchmark trace was added
        assert len(result['figure'].data) >= 2
        assert any('Benchmark' in trace.name for trace in result['figure'].data 
                  if hasattr(trace, 'name') and trace.name)
    
    def test_drawdown_chart(self, config, sample_equity_curve):
        """Test drawdown visualization"""
        chart = DrawdownChart(config)
        
        result = chart.create(equity_curve=sample_equity_curve)
        
        # Check result
        assert 'figure' in result
        assert 'data' in result
        
        # Check drawdown calculations
        assert 'max_drawdown' in result['data']
        assert 'max_drawdown_date' in result['data']
        assert 'avg_drawdown' in result['data']
        assert 'current_drawdown' in result['data']
        
        # Verify max drawdown is negative
        assert result['data']['max_drawdown'] <= 0
    
    def test_returns_distribution(self, config, sample_equity_curve):
        """Test returns distribution visualization"""
        chart = ReturnsDistribution(config)
        
        returns = sample_equity_curve.pct_change().dropna()
        result = chart.create(returns=returns)
        
        # Check result
        assert 'figure' in result
        assert 'data' in result
        
        # Check statistical calculations
        stats = result['data']
        assert 'mean_return' in stats
        assert 'std_return' in stats
        assert 'skewness' in stats
        assert 'kurtosis' in stats
        assert 'var_95' in stats
        assert 'cvar_95' in stats
        
        # Verify calculations
        assert abs(stats['mean_return'] - returns.mean()) < 1e-10
        assert abs(stats['std_return'] - returns.std()) < 1e-10


class TestTradeVisualization:
    """Test trade-related visualizations"""
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades data"""
        n_trades = 100
        
        trades = pd.DataFrame({
            'entry_time': pd.date_range('2023-01-01', periods=n_trades, freq='D'),
            'exit_time': pd.date_range('2023-01-02', periods=n_trades, freq='D'),
            'pnl': np.concatenate([
                np.random.normal(150, 50, 60),   # Winners
                np.random.normal(-100, 40, 40)   # Losers
            ]),
            'duration': np.random.uniform(1, 48, n_trades),
            'side': np.random.choice(['long', 'short'], n_trades)
        })
        
        return trades
    
    def test_trade_scatter_plot(self, sample_trades):
        """Test trade scatter plot visualization"""
        config = ReportConfig()
        chart = TradeScatterPlot(config)
        
        result = chart.create(trades=sample_trades)
        
        # Check result
        assert 'figure' in result
        assert 'data' in result
        
        # Check trade statistics
        stats = result['data']
        assert stats['total_trades'] == len(sample_trades)
        assert stats['winning_trades'] == (sample_trades['pnl'] > 0).sum()
        assert stats['losing_trades'] == (sample_trades['pnl'] <= 0).sum()
        assert 0 <= stats['win_rate'] <= 1
    
    def test_trade_scatter_empty_trades(self):
        """Test trade scatter with empty trades"""
        config = ReportConfig()
        chart = TradeScatterPlot(config)
        
        result = chart.create(trades=pd.DataFrame())
        
        assert 'message' in result
        assert 'No trades' in result['message']


class TestRollingMetrics:
    """Test rolling metrics visualizations"""
    
    @pytest.fixture
    def long_equity_curve(self):
        """Create longer equity curve for rolling metrics"""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.0004, 0.012, len(dates))
        equity = pd.Series((1 + returns).cumprod() * 100000, index=dates)
        return equity
    
    def test_rolling_metrics_chart(self, long_equity_curve):
        """Test rolling metrics visualization"""
        config = ReportConfig()
        chart = RollingMetricsChart(config)
        
        result = chart.create(
            equity_curve=long_equity_curve,
            window=252  # 1-year rolling
        )
        
        # Check result
        assert 'figure' in result
        assert 'data' in result
        
        # Check rolling calculations
        data = result['data']
        assert 'current_rolling_return' in data
        assert 'current_rolling_vol' in data
        assert 'current_rolling_sharpe' in data
        assert 'avg_rolling_sharpe' in data
        assert 'sharpe_consistency' in data
        
        # Verify consistency is between 0 and 1
        assert 0 <= data['sharpe_consistency'] <= 1
    
    def test_rolling_metrics_short_series(self):
        """Test rolling metrics with short series"""
        config = ReportConfig()
        chart = RollingMetricsChart(config)
        
        # Create short series
        short_equity = pd.Series([100000, 101000, 102000, 101500])
        
        result = chart.create(
            equity_curve=short_equity,
            window=252  # Window larger than series
        )
        
        # Should handle gracefully
        assert 'figure' in result
        assert result['data']['current_rolling_return'] == 0  # No data


class TestHeatmapVisualization:
    """Test heatmap visualizations"""
    
    def test_correlation_heatmap(self):
        """Test correlation matrix heatmap"""
        config = ReportConfig()
        chart = HeatmapVisualization(config)
        
        # Create sample correlation data
        data = pd.DataFrame({
            'Strategy': [1.0, 0.3, -0.2, 0.5],
            'Market': [0.3, 1.0, 0.7, 0.2],
            'Volume': [-0.2, 0.7, 1.0, -0.1],
            'Volatility': [0.5, 0.2, -0.1, 1.0]
        })
        
        result = chart.create(
            data=data,
            chart_type='correlation'
        )
        
        assert 'figure' in result
        assert result['type'] == 'correlation'
        assert isinstance(result['figure'], go.Figure)
    
    def test_monthly_returns_heatmap(self):
        """Test monthly returns heatmap"""
        config = ReportConfig()
        chart = HeatmapVisualization(config)
        
        # Create sample monthly returns
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        daily_returns = pd.Series(
            np.random.normal(0.001, 0.02, len(dates)),
            index=dates
        )
        
        result = chart.create(
            data=daily_returns,
            chart_type='monthly_returns'
        )
        
        assert 'figure' in result
        assert result['type'] == 'monthly_returns'
    
    def test_parameter_sensitivity_heatmap(self):
        """Test parameter sensitivity heatmap"""
        config = ReportConfig()
        chart = HeatmapVisualization(config)
        
        # Create sensitivity analysis data
        param1_values = [10, 20, 30, 40, 50]
        param2_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        # Create results matrix (Sharpe ratios)
        results = np.random.uniform(0.5, 2.5, (len(param2_values), len(param1_values)))
        
        sensitivity_data = {
            'param1_name': 'Lookback Period',
            'param1_values': param1_values,
            'param2_name': 'Threshold',
            'param2_values': param2_values,
            'results': results,
            'metric': 'Sharpe Ratio'
        }
        
        result = chart.create(
            data=sensitivity_data,
            chart_type='parameter_sensitivity'
        )
        
        assert 'figure' in result
        assert result['type'] == 'parameter_sensitivity'
        
        # Check that optimal point is marked
        assert len(result['figure'].data) >= 2  # Heatmap + marker


class TestVisualizationSaving:
    """Test saving functionality for visualizations"""
    
    def test_save_plotly_figure(self):
        """Test saving Plotly figures"""
        config = ReportConfig()
        chart = EquityCurveChart(config)
        
        # Create simple equity curve
        equity = pd.Series([100000, 105000, 110000])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_plot.html"
            
            result = chart.create(
                equity_curve=equity,
                save_path=save_path
            )
            
            # HTML format should be created
            # Note: PNG format requires kaleido or orca
            assert save_path.exists()
    
    def test_custom_color_scheme(self):
        """Test custom color scheme application"""
        custom_colors = {
            'primary': '#FF0000',
            'secondary': '#00FF00',
            'success': '#0000FF',
            'warning': '#FFFF00',
            'danger': '#FF00FF',
            'info': '#00FFFF',
            'background': '#FFFFFF',
            'text': '#000000'
        }
        
        config = ReportConfig(color_scheme=custom_colors)
        chart = EquityCurveChart(config)
        
        # Verify colors are applied
        assert chart.colors['primary'] == '#FF0000'
        assert chart.colors['success'] == '#0000FF'


class TestVisualizationEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        config = ReportConfig()
        
        # Empty equity curve
        chart = EquityCurveChart(config)
        result = chart.create(equity_curve=pd.Series())
        assert 'figure' in result
        
        # Empty returns
        chart = ReturnsDistribution(config)
        result = chart.create(returns=pd.Series())
        assert 'figure' in result
    
    def test_single_point_data(self):
        """Test handling of single data point"""
        config = ReportConfig()
        
        # Single point equity curve
        chart = DrawdownChart(config)
        result = chart.create(equity_curve=pd.Series([100000]))
        
        assert 'figure' in result
        assert result['data']['max_drawdown'] == 0
    
    def test_nan_handling(self):
        """Test handling of NaN values"""
        config = ReportConfig()
        
        # Equity curve with NaN
        equity = pd.Series([100000, np.nan, 105000, 110000])
        
        chart = EquityCurveChart(config)
        result = chart.create(equity_curve=equity)
        
        # Should handle NaN gracefully
        assert 'figure' in result
        assert not np.isnan(result['data']['total_return'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])