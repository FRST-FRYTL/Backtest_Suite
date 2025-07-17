"""
Comprehensive tests for the EnhancedInteractiveCharts class.

This module provides complete test coverage for enhanced interactive charting
including real-time updates, advanced interactivity, and custom visualizations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os

from src.visualization.enhanced_interactive_charts import EnhancedInteractiveCharts
class TestEnhancedInteractiveCharts:
    """Comprehensive tests for EnhancedInteractiveCharts class."""
    
    @pytest.fixture
    def charts(self):
        """Create EnhancedInteractiveCharts instance."""
        return EnhancedInteractiveCharts()
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data with multiple timeframes."""
        # Daily data
        daily_dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        daily_returns = np.random.normal(0.0002, 0.02, len(daily_dates))
        daily_close = 100 * (1 + daily_returns).cumprod()
        
        daily_data = pd.DataFrame({
            'date': daily_dates,
            'open': daily_close * (1 - np.abs(np.random.normal(0, 0.003, len(daily_dates)))),
            'high': daily_close * (1 + np.abs(np.random.normal(0, 0.005, len(daily_dates)))),
            'low': daily_close * (1 - np.abs(np.random.normal(0, 0.005, len(daily_dates)))),
            'close': daily_close,
            'volume': np.random.randint(1000000, 5000000, len(daily_dates))
        }).set_index('date')
        
        # Intraday data (1-hour)
        hourly_dates = pd.date_range('2023-12-01', periods=24*30, freq='h')
        hourly_returns = np.random.normal(0.00002, 0.005, len(hourly_dates))
        hourly_close = 100 * (1 + hourly_returns).cumprod()
        
        hourly_data = pd.DataFrame({
            'date': hourly_dates,
            'open': hourly_close * (1 - np.abs(np.random.normal(0, 0.001, len(hourly_dates)))),
            'high': hourly_close * (1 + np.abs(np.random.normal(0, 0.002, len(hourly_dates)))),
            'low': hourly_close * (1 - np.abs(np.random.normal(0, 0.002, len(hourly_dates)))),
            'close': hourly_close,
            'volume': np.random.randint(100000, 500000, len(hourly_dates))
        }).set_index('date')
        
        return {
            'daily': daily_data,
            'hourly': hourly_data
        }
    
    @pytest.fixture
    def sample_indicators(self, sample_market_data):
        """Create sample technical indicators."""
        daily_data = sample_market_data['daily']
        
        indicators = pd.DataFrame(index=daily_data.index)
        
        # Moving averages
        indicators['sma_20'] = daily_data['close'].rolling(20).mean()
        indicators['sma_50'] = daily_data['close'].rolling(50).mean()
        indicators['ema_12'] = daily_data['close'].ewm(span=12).mean()
        indicators['ema_26'] = daily_data['close'].ewm(span=26).mean()
        
        # MACD
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # RSI
        delta = daily_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        indicators['bb_middle'] = daily_data['close'].rolling(20).mean()
        bb_std = daily_data['close'].rolling(20).std()
        indicators['bb_upper'] = indicators['bb_middle'] + 2 * bb_std
        indicators['bb_lower'] = indicators['bb_middle'] - 2 * bb_std
        
        # Volume indicators
        indicators['volume_sma'] = daily_data['volume'].rolling(20).mean()
        indicators['obv'] = (np.sign(daily_data['close'].diff()) * daily_data['volume']).cumsum()
        
        return indicators
    
    @pytest.fixture
    def sample_signals(self, sample_market_data):
        """Create sample trading signals."""
        daily_data = sample_market_data['daily']
        
        signals = pd.DataFrame(index=daily_data.index)
        signals['buy'] = 0
        signals['sell'] = 0
        signals['position'] = 0
        
        # Generate some signals
        for i in range(20, len(signals)):
            if i % 15 == 0:
                signals.iloc[i, signals.columns.get_loc('buy')] = 1
            elif i % 20 == 0:
                signals.iloc[i, signals.columns.get_loc('sell')] = 1
        
        # Calculate positions
        position = 0
        for i in range(len(signals)):
            if signals.iloc[i]['buy'] == 1:
                position = 1
            elif signals.iloc[i]['sell'] == 1:
                position = 0
            signals.iloc[i, signals.columns.get_loc('position')] = position
        
        return signals
    
    def test_charts_initialization(self, charts):
        """Test EnhancedInteractiveCharts initialization."""
        assert isinstance(charts, EnhancedInteractiveCharts)
        assert hasattr(charts, 'create_advanced_chart')
    
    def test_create_advanced_candlestick_chart(self, charts, sample_market_data, sample_indicators):
        """Test advanced candlestick chart creation."""
        fig = charts.create_advanced_chart(
            price_data=sample_market_data['daily'],
            indicators=sample_indicators,
            chart_type='candlestick'
        )
        
        assert isinstance(fig, go.Figure)
        
        # Check for candlestick trace
        has_candlestick = any(
            isinstance(trace, go.Candlestick) or 
            (hasattr(trace, 'type') and trace.type == 'candlestick')
            for trace in fig.data
        )
        assert has_candlestick
        
        # Check for indicator traces
        trace_names = [trace.name for trace in fig.data if hasattr(trace, 'name')]
        assert any('sma' in name.lower() for name in trace_names)
    
    def test_create_multi_panel_chart(self, charts, sample_market_data, sample_indicators):
        """Test multi-panel chart creation."""
        fig = charts.create_multi_panel_chart(
            price_data=sample_market_data['daily'],
            indicators=sample_indicators,
            panels=['price', 'volume', 'rsi', 'macd']
        )
        
        assert isinstance(fig, go.Figure)
        # Should have multiple subplots
        assert hasattr(fig, '_grid_ref')
    
    def test_create_heatmap_visualization(self, charts, sample_market_data):
        """Test heatmap visualization for correlations."""
        # Calculate returns for multiple assets
        returns_data = pd.DataFrame()
        
        for i, symbol in enumerate(['AAPL', 'GOOGL', 'MSFT', 'AMZN']):
            returns_data[symbol] = sample_market_data['daily']['close'].pct_change() + np.random.normal(0, 0.01, len(sample_market_data['daily']))
        
        fig = charts.create_correlation_heatmap(returns_data)
        
        assert isinstance(fig, go.Figure)
        # Should have heatmap trace
        assert any(
            hasattr(trace, 'type') and trace.type == 'heatmap'
            for trace in fig.data
        )
    
    def test_create_3d_surface_chart(self, charts, sample_market_data):
        """Test 3D surface chart for volatility surface."""
        # Create sample volatility surface data
        strikes = np.linspace(80, 120, 20)
        maturities = np.linspace(0.1, 2, 20)
        
        X, Y = np.meshgrid(strikes, maturities)
        Z = 0.2 + 0.1 * np.sin(X/10) + 0.05 * Y
        
        fig = charts.create_volatility_surface(
            strikes=strikes,
            maturities=maturities,
            volatilities=Z
        )
        
        assert isinstance(fig, go.Figure)
        # Should have surface trace
        assert any(
            hasattr(trace, 'type') and trace.type == 'surface'
            for trace in fig.data
        )
    
    def test_real_time_update_capability(self, charts, sample_market_data):
        """Test real-time update functionality."""
        initial_fig = charts.create_advanced_chart(
            price_data=sample_market_data['daily'][-50:],
            enable_streaming=True
        )
        
        assert isinstance(initial_fig, go.Figure)
        
        # Simulate new data point
        new_data = pd.DataFrame({
            'open': [sample_market_data['daily']['close'].iloc[-1]],
            'high': [sample_market_data['daily']['close'].iloc[-1] * 1.01],
            'low': [sample_market_data['daily']['close'].iloc[-1] * 0.99],
            'close': [sample_market_data['daily']['close'].iloc[-1] * 1.005],
            'volume': [2000000]
        }, index=[sample_market_data['daily'].index[-1] + timedelta(days=1)])
        
        # Update chart
        updated_fig = charts.update_streaming_data(initial_fig, new_data)
        assert isinstance(updated_fig, (go.Figure, dict))
    
    def test_interactive_drawing_tools(self, charts, sample_market_data):
        """Test interactive drawing tools configuration."""
        fig = charts.create_advanced_chart(
            price_data=sample_market_data['daily'],
            enable_drawing_tools=True,
            drawing_tools=['line', 'rect', 'circle', 'fib_retracement']
        )
        
        assert isinstance(fig, go.Figure)
        
        # Check for drawing tools in config
        config = charts._get_drawing_tools_config()
        assert isinstance(config, dict)
        assert 'modeBarButtonsToAdd' in config
    
    def test_create_market_profile_chart(self, charts, sample_market_data):
        """Test market profile (volume profile) chart."""
        fig = charts.create_market_profile(
            price_data=sample_market_data['daily'],
            profile_type='volume'
        )
        
        assert isinstance(fig, go.Figure)
        # Should have horizontal bar chart for volume profile
        assert len(fig.data) > 0
    
    def test_create_order_flow_visualization(self, charts):
        """Test order flow visualization."""
        # Create sample order flow data
        timestamps = pd.date_range('2023-12-01 09:30', periods=100, freq='1min')
        
        order_flow_data = pd.DataFrame({
            'timestamp': timestamps,
            'bid_volume': np.random.randint(100, 1000, 100),
            'ask_volume': np.random.randint(100, 1000, 100),
            'delta': np.random.randint(-500, 500, 100),
            'cumulative_delta': np.random.randint(-500, 500, 100).cumsum()
        })
        
        fig = charts.create_order_flow_chart(order_flow_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_create_range_bars_chart(self, charts, sample_market_data):
        """Test range bars chart creation."""
        fig = charts.create_range_bars(
            price_data=sample_market_data['hourly'],
            range_size=1.0
        )
        
        assert isinstance(fig, go.Figure)
    
    def test_custom_indicator_overlay(self, charts, sample_market_data):
        """Test custom indicator overlay functionality."""
        # Create custom indicator
        custom_indicator = sample_market_data['daily']['close'].rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        
        fig = charts.create_advanced_chart(
            price_data=sample_market_data['daily'],
            custom_indicators={
                'trend_strength': custom_indicator
            }
        )
        
        assert isinstance(fig, go.Figure)
        
        # Check for custom indicator
        trace_names = [trace.name for trace in fig.data if hasattr(trace, 'name')]
        assert any('trend_strength' in name.lower() for name in trace_names)
    
    def test_multi_timeframe_synchronization(self, charts, sample_market_data):
        """Test multi-timeframe chart synchronization."""
        fig = charts.create_multi_timeframe_chart(
            timeframes={
                'Daily': sample_market_data['daily'],
                'Hourly': sample_market_data['hourly']
            },
            sync_zoom=True,
            sync_pan=True
        )
        
        assert isinstance(fig, go.Figure)
    
    def test_annotation_management(self, charts, sample_market_data):
        """Test annotation management system."""
        # Create annotations
        annotations = [
            {
                'date': sample_market_data['daily'].index[50],
                'price': sample_market_data['daily']['close'].iloc[50],
                'text': 'Important level',
                'type': 'support'
            },
            {
                'date': sample_market_data['daily'].index[100],
                'price': sample_market_data['daily']['close'].iloc[100],
                'text': 'Breakout point',
                'type': 'breakout'
            }
        ]
        
        fig = charts.create_advanced_chart(
            price_data=sample_market_data['daily'],
            annotations=annotations
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) >= len(annotations)
    
    def test_pattern_recognition_overlay(self, charts, sample_market_data):
        """Test pattern recognition overlay."""
        # Simulate detected patterns
        patterns = [
            {
                'type': 'head_and_shoulders',
                'start_date': sample_market_data['daily'].index[20],
                'end_date': sample_market_data['daily'].index[40],
                'confidence': 0.85
            },
            {
                'type': 'triangle',
                'start_date': sample_market_data['daily'].index[60],
                'end_date': sample_market_data['daily'].index[80],
                'confidence': 0.75
            }
        ]
        
        fig = charts.add_pattern_overlays(
            base_chart=charts.create_advanced_chart(sample_market_data['daily']),
            patterns=patterns
        )
        
        assert isinstance(fig, go.Figure)
    
    def test_performance_overlay(self, charts, sample_market_data, sample_signals):
        """Test performance metrics overlay on chart."""
        # Calculate simple performance
        returns = sample_market_data['daily']['close'].pct_change()
        signal_returns = returns * sample_signals['position'].shift(1)
        cumulative_returns = (1 + signal_returns).cumprod()
        
        fig = charts.add_performance_overlay(
            price_chart=charts.create_advanced_chart(sample_market_data['daily']),
            performance_data=cumulative_returns,
            metrics={
                'total_return': cumulative_returns.iloc[-1] - 1,
                'sharpe_ratio': 1.45,
                'max_drawdown': -0.15
            }
        )
        
        assert isinstance(fig, go.Figure)
    
    def test_save_chart_configuration(self, charts, sample_market_data):
        """Test saving and loading chart configurations."""
        # Create chart with specific configuration
        config = {
            'indicators': ['sma_20', 'rsi'],
            'chart_type': 'candlestick',
            'theme': 'dark',
            'height': 800
        }
        
        fig = charts.create_advanced_chart(
            price_data=sample_market_data['daily'],
            **config
        )
        
        # Save configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            charts.save_configuration(config, f.name)
            temp_file = f.name
        
        # Load configuration
        loaded_config = charts.load_configuration(temp_file)
        
        assert loaded_config == config
        
        # Clean up
        os.remove(temp_file)
    
    def test_export_interactive_html(self, charts, sample_market_data):
        """Test export to interactive HTML."""
        fig = charts.create_advanced_chart(sample_market_data['daily'])
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            charts.export_to_html(fig, f.name, include_plotlyjs='cdn')
            temp_file = f.name
        
        assert os.path.exists(temp_file)
        
        # Verify content
        with open(temp_file, 'r') as f:
            content = f.read()
            assert 'plotly' in content
        
        # Clean up
        os.remove(temp_file)
    
    def test_responsive_design_configuration(self, charts, sample_market_data):
        """Test responsive design configuration."""
        fig = charts.create_advanced_chart(
            price_data=sample_market_data['daily'],
            responsive=True,
            autosize=True
        )
        
        assert fig.layout.autosize is True
    
    def test_custom_color_schemes(self, charts, sample_market_data):
        """Test custom color scheme support."""
        custom_colors = {
            'up_candle': '#00ff00',
            'down_candle': '#ff0000',
            'volume_bars': '#3366cc',
            'background': '#1e1e1e',
            'grid': '#333333'
        }
        
        fig = charts.create_advanced_chart(
            price_data=sample_market_data['daily'],
            color_scheme=custom_colors
        )
        
        assert isinstance(fig, go.Figure)
    
    def test_mobile_optimization(self, charts, sample_market_data):
        """Test mobile optimization features."""
        fig = charts.create_advanced_chart(
            price_data=sample_market_data['daily'],
            mobile_optimized=True
        )
        
        # Check for mobile-friendly configuration
        assert isinstance(fig, go.Figure)
        # Should have simplified layout for mobile
        assert fig.layout.showlegend is False or fig.layout.legend.orientation == 'h'