"""
Comprehensive tests for the ComprehensiveTradingDashboard class.

This module provides complete test coverage for the comprehensive trading dashboard
including all visualization components, interactive features, and export functionality.
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
import json
from pathlib import Path

from src.visualization.comprehensive_trading_dashboard import ComprehensiveTradingDashboard


class TestComprehensiveTradingDashboard:
    """Comprehensive tests for ComprehensiveTradingDashboard class."""
    
    @pytest.fixture
    def dashboard(self):
        """Create ComprehensiveTradingDashboard instance."""
        return ComprehensiveTradingDashboard()
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        returns = np.random.normal(0.0002, 0.02, len(dates))
        close_prices = 100 * (1 + returns).cumprod()
        
        # Add OHLC data
        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates))))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        volume = np.random.randint(1000000, 5000000, len(dates))
        
        price_data = pd.DataFrame({
            'date': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })
        
        price_data.set_index('date', inplace=True)
        return price_data
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample trading signals."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Generate signals
        signals = pd.DataFrame(index=dates)
        
        # Buy/sell signals (sparse)
        buy_indices = np.random.choice(len(dates), 15, replace=False)
        sell_indices = np.random.choice(len(dates), 15, replace=False)
        
        signals['buy'] = 0
        signals['sell'] = 0
        signals.iloc[buy_indices, signals.columns.get_loc('buy')] = 1
        signals.iloc[sell_indices, signals.columns.get_loc('sell')] = 1
        
        # Position sizing
        signals['position'] = 0
        current_position = 0
        
        for i in range(len(signals)):
            if signals.iloc[i]['buy'] == 1:
                current_position = 1
            elif signals.iloc[i]['sell'] == 1:
                current_position = 0
            signals.iloc[i, signals.columns.get_loc('position')] = current_position
        
        return signals
    
    @pytest.fixture
    def sample_backtest_results(self, sample_price_data):
        """Create comprehensive backtest results."""
        dates = sample_price_data.index
        np.random.seed(42)
        
        # Equity curve
        returns = np.random.normal(0.0005, 0.02, len(dates))
        cumulative_returns = (1 + returns).cumprod()
        
        equity_curve = pd.DataFrame({
            'timestamp': dates,
            'total_value': 100000 * cumulative_returns,
            'cash': 50000 * np.ones(len(dates)),
            'holdings_value': 50000 * cumulative_returns,
            'returns': returns,
            'cumulative_returns': cumulative_returns
        })
        
        # Trades
        trades_data = []
        for i in range(30):
            entry_idx = i * 8
            if entry_idx >= len(dates) - 10:
                break
                
            entry_date = dates[entry_idx]
            exit_date = dates[min(entry_idx + np.random.randint(2, 8), len(dates)-1)]
            
            entry_price = sample_price_data.loc[entry_date, 'close']
            exit_price = sample_price_data.loc[exit_date, 'close']
            quantity = np.random.randint(10, 100)
            
            trades_data.append({
                'trade_id': i + 1,
                'entry_time': entry_date,
                'exit_time': exit_date,
                'symbol': 'TEST',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'profit_loss': (exit_price - entry_price) * quantity,
                'return_pct': ((exit_price - entry_price) / entry_price) * 100,
                'commission': 2.0,
                'side': 'long',
                'strategy': 'TestStrategy'
            })
        
        trades = pd.DataFrame(trades_data)
        
        # Performance metrics
        total_return = (equity_curve['total_value'].iloc[-1] / equity_curve['total_value'].iloc[0]) - 1
        
        performance = {
            'total_return': total_return,
            'annualized_return': total_return * (252 / len(dates)),
            'sharpe_ratio': 1.45,
            'sortino_ratio': 2.13,
            'max_drawdown': -0.1523,
            'win_rate': len(trades[trades['profit_loss'] > 0]) / len(trades) if len(trades) > 0 else 0,
            'profit_factor': 1.85,
            'total_trades': len(trades),
            'winning_trades': len(trades[trades['profit_loss'] > 0]) if len(trades) > 0 else 0,
            'losing_trades': len(trades[trades['profit_loss'] <= 0]) if len(trades) > 0 else 0
        }
        
        # Indicators
        indicators = pd.DataFrame(index=dates)
        indicators['sma_20'] = sample_price_data['close'].rolling(20).mean()
        indicators['sma_50'] = sample_price_data['close'].rolling(50).mean()
        indicators['rsi'] = 50 + np.random.normal(0, 20, len(dates))
        indicators['rsi'] = indicators['rsi'].clip(0, 100)
        
        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'performance': performance,
            'indicators': indicators
        }
    
    def test_dashboard_initialization(self, dashboard):
        """Test dashboard initialization."""
        assert isinstance(dashboard, ComprehensiveTradingDashboard)
        assert hasattr(dashboard, 'create_dashboard')
    
    def test_create_full_dashboard(self, dashboard, sample_price_data, sample_signals, sample_backtest_results):
        """Test full dashboard creation."""
        output_path = dashboard.create_dashboard(
            price_data=sample_price_data,
            signals=sample_signals,
            equity_curve=sample_backtest_results['equity_curve'],
            trades=sample_backtest_results['trades'],
            performance_metrics=sample_backtest_results['performance'],
            indicators=sample_backtest_results['indicators']
        )
        
        assert output_path is not None
        assert os.path.exists(output_path)
        assert output_path.endswith('.html')
        
        # Verify content
        with open(output_path, 'r') as f:
            content = f.read()
            assert 'Comprehensive Trading Dashboard' in content
            assert 'plotly' in content
        
        # Clean up
        os.remove(output_path)
    
    def test_create_price_chart(self, dashboard, sample_price_data, sample_signals):
        """Test price chart creation with signals."""
        fig = dashboard._create_price_chart(sample_price_data, sample_signals)
        
        assert isinstance(fig, go.Figure)
        
        # Check for candlestick trace
        has_candlestick = any(
            isinstance(trace, go.Candlestick) or 
            (hasattr(trace, 'type') and trace.type == 'candlestick')
            for trace in fig.data
        )
        assert has_candlestick
        
        # Check for signal markers
        signal_traces = [trace for trace in fig.data if hasattr(trace, 'mode') and 'markers' in trace.mode]
        assert len(signal_traces) >= 2  # Buy and sell signals
    
    def test_create_equity_performance_chart(self, dashboard, sample_backtest_results):
        """Test equity performance chart creation."""
        fig = dashboard._create_equity_performance_chart(
            sample_backtest_results['equity_curve'],
            sample_backtest_results['performance']
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        
        # Should have multiple subplots
        assert hasattr(fig, '_grid_ref')
    
    def test_create_trade_analysis_chart(self, dashboard, sample_backtest_results):
        """Test trade analysis chart creation."""
        fig = dashboard._create_trade_analysis_chart(sample_backtest_results['trades'])
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_create_metrics_summary(self, dashboard, sample_backtest_results):
        """Test metrics summary creation."""
        fig = dashboard._create_metrics_summary(sample_backtest_results['performance'])
        
        assert isinstance(fig, go.Figure)
        
        # Should have table or indicator traces
        has_table_or_indicator = any(
            isinstance(trace, (go.Table, go.Indicator))
            for trace in fig.data
        )
        assert has_table_or_indicator
    
    def test_create_indicator_chart(self, dashboard, sample_price_data, sample_backtest_results):
        """Test indicator chart creation."""
        fig = dashboard._create_indicator_chart(
            sample_price_data,
            sample_backtest_results['indicators']
        )
        
        assert isinstance(fig, go.Figure)
        
        # Check for price and indicator traces
        assert len(fig.data) >= 3  # Price + at least 2 indicators
    
    def test_dashboard_with_minimal_data(self, dashboard, sample_price_data):
        """Test dashboard creation with minimal data."""
        minimal_equity = pd.DataFrame({
            'timestamp': sample_price_data.index[:10],
            'total_value': 100000 * np.ones(10)
        })
        
        output_path = dashboard.create_dashboard(
            price_data=sample_price_data[:10],
            equity_curve=minimal_equity
        )
        
        assert os.path.exists(output_path)
        
        # Clean up
        os.remove(output_path)
    
    def test_dashboard_custom_title(self, dashboard, sample_price_data, sample_backtest_results):
        """Test dashboard with custom title."""
        custom_title = "My Strategy Dashboard"
        
        output_path = dashboard.create_dashboard(
            price_data=sample_price_data,
            equity_curve=sample_backtest_results['equity_curve'],
            title=custom_title
        )
        
        with open(output_path, 'r') as f:
            content = f.read()
            assert custom_title in content
        
        # Clean up
        os.remove(output_path)
    
    def test_dashboard_layout_configuration(self, dashboard):
        """Test dashboard layout configuration."""
        layout_config = dashboard._get_layout_config()
        
        assert isinstance(layout_config, dict)
        assert 'height' in layout_config
        assert 'showlegend' in layout_config
        assert 'template' in layout_config
    
    def test_interactive_features(self, dashboard, sample_price_data, sample_signals):
        """Test interactive features configuration."""
        fig = dashboard._create_price_chart(sample_price_data, sample_signals)
        
        # Check for rangeslider
        assert hasattr(fig.layout.xaxis, 'rangeslider')
        
        # Check for hover configuration
        assert hasattr(fig.layout, 'hovermode')
    
    def test_multi_timeframe_support(self, dashboard):
        """Test multi-timeframe data support."""
        # Create data for multiple timeframes
        dates = pd.date_range('2023-01-01', periods=1000, freq='h')
        
        hourly_data = pd.DataFrame({
            'open': 100 + np.random.randn(1000).cumsum(),
            'high': 101 + np.random.randn(1000).cumsum(),
            'low': 99 + np.random.randn(1000).cumsum(),
            'close': 100 + np.random.randn(1000).cumsum(),
            'volume': np.random.randint(1000, 5000, 1000)
        }, index=dates)
        
        # Dashboard should handle different timeframes
        fig = dashboard._create_price_chart(hourly_data)
        assert isinstance(fig, go.Figure)
    
    def test_error_handling_missing_columns(self, dashboard):
        """Test error handling for missing required columns."""
        invalid_price_data = pd.DataFrame({
            'price': [100, 101, 102],
            'time': pd.date_range('2023-01-01', periods=3)
        })
        
        with pytest.raises((KeyError, ValueError)):
            dashboard._create_price_chart(invalid_price_data)
    
    def test_volume_subplot(self, dashboard, sample_price_data):
        """Test volume subplot creation."""
        fig = dashboard._create_price_chart(sample_price_data, include_volume=True)
        
        # Should have volume bars
        volume_traces = [
            trace for trace in fig.data 
            if hasattr(trace, 'yaxis') and trace.yaxis == 'y2'
        ]
        assert len(volume_traces) > 0
    
    def test_drawdown_visualization(self, dashboard, sample_backtest_results):
        """Test drawdown visualization."""
        fig = dashboard._create_equity_performance_chart(
            sample_backtest_results['equity_curve'],
            sample_backtest_results['performance']
        )
        
        # Should include drawdown subplot
        fig_str = str(fig)
        assert 'drawdown' in fig_str.lower()
    
    def test_trade_markers_on_price_chart(self, dashboard, sample_price_data, sample_backtest_results):
        """Test trade entry/exit markers on price chart."""
        fig = dashboard._create_price_chart(
            sample_price_data,
            trades=sample_backtest_results['trades']
        )
        
        # Should have trade markers
        marker_traces = [
            trace for trace in fig.data
            if hasattr(trace, 'mode') and 'markers' in str(trace.mode)
        ]
        assert len(marker_traces) >= 2  # Entry and exit markers
    
    def test_performance_comparison(self, dashboard, sample_backtest_results):
        """Test performance comparison features."""
        # Add benchmark data
        benchmark_returns = np.random.normal(0.0003, 0.015, len(sample_backtest_results['equity_curve']))
        benchmark_equity = 100000 * (1 + benchmark_returns).cumprod()
        
        fig = dashboard._create_equity_performance_chart(
            sample_backtest_results['equity_curve'],
            sample_backtest_results['performance'],
            benchmark=pd.Series(benchmark_equity, index=sample_backtest_results['equity_curve']['timestamp'])
        )
        
        # Should have benchmark trace
        trace_names = [trace.name for trace in fig.data if hasattr(trace, 'name')]
        assert any('benchmark' in name.lower() for name in trace_names)
    
    def test_export_functionality(self, dashboard, sample_price_data, sample_backtest_results):
        """Test dashboard export functionality."""
        # Test multiple export formats
        output_paths = dashboard.create_dashboard(
            price_data=sample_price_data,
            equity_curve=sample_backtest_results['equity_curve'],
            export_formats=['html', 'png']
        )
        
        if isinstance(output_paths, list):
            for path in output_paths:
                assert os.path.exists(path)
                # Clean up
                os.remove(path)
        else:
            assert os.path.exists(output_paths)
            os.remove(output_paths)
    
    def test_theme_support(self, dashboard, sample_price_data):
        """Test theme support."""
        # Test with dark theme
        fig = dashboard._create_price_chart(sample_price_data, theme='dark')
        
        assert fig.layout.template is not None
        
    def test_responsive_design(self, dashboard, sample_price_data, sample_backtest_results):
        """Test responsive design configuration."""
        output_path = dashboard.create_dashboard(
            price_data=sample_price_data,
            equity_curve=sample_backtest_results['equity_curve'],
            responsive=True
        )
        
        with open(output_path, 'r') as f:
            content = f.read()
            assert 'responsive' in content or 'autosize' in content
        
        # Clean up
        os.remove(output_path)