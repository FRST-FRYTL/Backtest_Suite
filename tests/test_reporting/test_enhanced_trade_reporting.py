"""
Unit tests for enhanced trade reporting features.

This module tests the new trade price reporting functionality including
trade price charts, stop loss analysis, and risk per trade analysis.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from unittest.mock import patch, MagicMock

from src.reporting.report_config import ReportConfig, TradeReportingConfig
from src.reporting.visualizations import ReportVisualizations
from src.reporting.markdown_template import (
    generate_trade_analysis_md,
    create_enhanced_trade_summary,
    format_trade_table_row
)


class TestTradeReportingConfig:
    """Test TradeReportingConfig class"""
    
    def test_default_config(self):
        """Test default trade reporting configuration"""
        config = TradeReportingConfig()
        
        assert config.enable_detailed_trade_prices == True
        assert config.price_display_format == "absolute"
        assert config.show_entry_exit_prices == True
        assert config.show_stop_loss_prices == True
        assert config.show_take_profit_prices == True
        assert config.enable_stop_loss_analysis == True
        assert config.enable_risk_per_trade_analysis == True
        assert config.max_trades_in_detailed_table == 100
        assert config.include_trade_timing_analysis == True
        assert config.show_trade_price_charts == True
    
    def test_custom_config(self):
        """Test custom trade reporting configuration"""
        config = TradeReportingConfig(
            enable_detailed_trade_prices=False,
            price_display_format="percentage",
            max_trades_in_detailed_table=50
        )
        
        assert config.enable_detailed_trade_prices == False
        assert config.price_display_format == "percentage"
        assert config.max_trades_in_detailed_table == 50


class TestEnhancedReportConfig:
    """Test enhanced ReportConfig with trade reporting"""
    
    def test_report_config_with_trade_reporting(self):
        """Test ReportConfig includes TradeReportingConfig"""
        config = ReportConfig()
        
        assert hasattr(config, 'trade_reporting')
        assert isinstance(config.trade_reporting, TradeReportingConfig)
    
    def test_trade_analysis_section_visualization(self):
        """Test trade analysis section includes new visualizations"""
        config = ReportConfig()
        trade_section = config.sections.get('trade_analysis')
        
        assert trade_section is not None
        expected_visualizations = [
            "trade_distribution", 
            "win_loss_chart", 
            "trade_duration_histogram",
            "trade_price_chart",
            "stop_loss_analysis",
            "trade_risk_chart"
        ]
        
        for viz in expected_visualizations:
            assert viz in trade_section.visualizations
    
    def test_new_trade_metrics(self):
        """Test new trade-related metrics are included"""
        config = ReportConfig()
        
        expected_metrics = [
            'avg_entry_price',
            'avg_exit_price',
            'stop_loss_hit_rate',
            'avg_risk_per_trade',
            'risk_reward_ratio'
        ]
        
        for metric in expected_metrics:
            assert metric in config.metrics
            assert 'name' in config.metrics[metric]
            assert 'format' in config.metrics[metric]
            assert 'description' in config.metrics[metric]


class TestEnhancedVisualizations:
    """Test enhanced visualization functions"""
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades with price data"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        trades = pd.DataFrame({
            'trade_id': range(1, 51),
            'entry_time': dates,
            'exit_time': dates + timedelta(days=2),
            'side': np.random.choice(['long', 'short'], 50),
            'entry_price': np.random.uniform(100, 200, 50),
            'exit_price': np.random.uniform(95, 210, 50),
            'stop_loss': np.random.uniform(90, 110, 50),
            'take_profit': np.random.uniform(190, 220, 50),
            'pnl': np.random.normal(50, 100, 50),
            'size': np.random.uniform(100, 1000, 50),
            'duration': np.random.uniform(1, 48, 50),
            'exit_reason': np.random.choice(['target', 'stop', 'time'], 50)
        })
        
        return trades
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = pd.DataFrame({
            'close': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(105, 205, 100),
            'low': np.random.uniform(95, 195, 100)
        }, index=dates)
        
        return prices
    
    def test_create_trade_price_chart(self, sample_trades, sample_price_data):
        """Test trade price chart creation"""
        viz = ReportVisualizations()
        fig = viz.create_trade_price_chart(sample_trades, sample_price_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        
        # Check that different trace types are included
        trace_names = [trace.name for trace in fig.data]
        expected_traces = ['Price', 'Entry', 'Exit', 'Stop Loss', 'Take Profit']
        
        for expected in expected_traces:
            assert expected in trace_names
    
    def test_create_trade_price_chart_empty_data(self):
        """Test trade price chart with empty data"""
        viz = ReportVisualizations()
        empty_trades = pd.DataFrame()
        fig = viz.create_trade_price_chart(empty_trades)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
    
    def test_create_stop_loss_analysis(self, sample_trades):
        """Test stop loss analysis chart creation"""
        viz = ReportVisualizations()
        fig = viz.create_stop_loss_analysis(sample_trades)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        
        # Check that the figure has multiple subplots
        assert fig.layout.annotations is not None
        assert len(fig.layout.annotations) == 4  # 4 subplot titles
    
    def test_create_stop_loss_analysis_no_stop_data(self):
        """Test stop loss analysis with no stop loss data"""
        viz = ReportVisualizations()
        trades_no_stop = pd.DataFrame({
            'entry_price': [100, 110, 120],
            'exit_price': [105, 115, 125],
            'pnl': [5, 5, 5]
        })
        
        fig = viz.create_stop_loss_analysis(trades_no_stop)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
    
    def test_create_trade_risk_chart(self, sample_trades):
        """Test trade risk chart creation"""
        viz = ReportVisualizations()
        fig = viz.create_trade_risk_chart(sample_trades)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        
        # Check that the figure has multiple subplots
        assert fig.layout.annotations is not None
        assert len(fig.layout.annotations) == 4  # 4 subplot titles
    
    def test_create_trade_risk_chart_empty_data(self):
        """Test trade risk chart with empty data"""
        viz = ReportVisualizations()
        empty_trades = pd.DataFrame()
        fig = viz.create_trade_risk_chart(empty_trades)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
    
    def test_create_trade_risk_chart_minimal_data(self):
        """Test trade risk chart with minimal data"""
        viz = ReportVisualizations()
        minimal_trades = pd.DataFrame({
            'pnl': [10, -5, 15, -8, 20]
        })
        
        fig = viz.create_trade_risk_chart(minimal_trades)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestEnhancedMarkdownTemplate:
    """Test enhanced markdown template functions"""
    
    @pytest.fixture
    def sample_trade_section_data(self):
        """Create sample trade section data"""
        return {
            'trade_statistics': {
                'summary': {
                    'total_trades': 100,
                    'winning_trades': 60,
                    'losing_trades': 40,
                    'avg_trade_pnl': 50.25
                }
            },
            'win_loss_analysis': {
                'win_rate': '60.0%',
                'winning_trades': {'count': 60, 'avg_win': '$75.50'},
                'losing_trades': {'count': 40, 'avg_loss': '$-45.25'},
                'ratios': {'win_loss_ratio': '1.67'}
            },
            'price_analysis': {
                'avg_entry_price': '$150.25',
                'avg_exit_price': '$155.75',
                'price_improvement': '3.7%'
            },
            'stop_loss_analysis': {
                'stop_loss_hit_rate': '25.0%',
                'avg_stop_distance': '2.5%',
                'stop_loss_effectiveness': 'Good'
            },
            'risk_analysis': {
                'avg_risk_per_trade': '2.2%',
                'max_risk_per_trade': '5.0%',
                'risk_reward_ratio': '1:2.3'
            },
            'detailed_trades': [
                {
                    'trade_id': 1,
                    'entry_time': '2023-01-01 09:30:00',
                    'exit_time': '2023-01-01 15:30:00',
                    'side': 'long',
                    'entry_price': '$150.00',
                    'exit_price': '$155.00',
                    'stop_loss': '$145.00',
                    'take_profit': '$160.00',
                    'pnl': '$500.00',
                    'duration': '6h'
                },
                {
                    'trade_id': 2,
                    'entry_time': '2023-01-02 10:00:00',
                    'exit_time': '2023-01-02 14:00:00',
                    'side': 'short',
                    'entry_price': '$148.00',
                    'exit_price': '$145.00',
                    'stop_loss': '$152.00',
                    'take_profit': '$140.00',
                    'pnl': '$300.00',
                    'duration': '4h'
                }
            ]
        }
    
    def test_generate_trade_analysis_md(self, sample_trade_section_data):
        """Test trade analysis markdown generation"""
        markdown = generate_trade_analysis_md(sample_trade_section_data)
        
        assert "## Trade Analysis" in markdown
        assert "### Trade Statistics" in markdown
        assert "### Win/Loss Analysis" in markdown
        assert "### Price Analysis" in markdown
        assert "### Stop Loss Analysis" in markdown
        assert "### Risk per Trade Analysis" in markdown
        assert "### Detailed Trades" in markdown
        
        # Check that data is properly formatted
        assert "100" in markdown  # total trades
        assert "60.0%" in markdown  # win rate
        assert "$150.25" in markdown  # avg entry price
        assert "25.0%" in markdown  # stop loss hit rate
        assert "2.2%" in markdown  # avg risk per trade
    
    def test_generate_trade_analysis_md_empty_data(self):
        """Test trade analysis markdown with empty data"""
        markdown = generate_trade_analysis_md({})
        
        assert markdown == ""
    
    def test_generate_trade_analysis_md_with_message(self):
        """Test trade analysis markdown with message"""
        data = {'message': 'No trades available for analysis'}
        markdown = generate_trade_analysis_md(data)
        
        assert "## Trade Analysis" in markdown
        assert "No trades available for analysis" in markdown
    
    def test_format_trade_table_row_absolute(self):
        """Test formatting trade table row with absolute prices"""
        trade = {
            'trade_id': 1,
            'entry_time': '2023-01-01 09:30:00',
            'exit_time': '2023-01-01 15:30:00',
            'side': 'long',
            'entry_price': 150.00,
            'exit_price': 155.00,
            'stop_loss': 145.00,
            'take_profit': 160.00,
            'pnl': 500.00,
            'duration': '6h'
        }
        
        row = format_trade_table_row(trade, "absolute")
        
        assert row[0] == "1"  # trade_id
        assert row[4] == "150.0"  # entry_price
        assert row[5] == "155.0"  # exit_price
        assert row[6] == "145.0"  # stop_loss
        assert row[8] == "500.0"  # pnl
    
    def test_format_trade_table_row_percentage(self):
        """Test formatting trade table row with percentage prices"""
        trade = {
            'trade_id': 1,
            'entry_price': 150.00,
            'exit_price': 155.00,
            'pnl': 500.00
        }
        
        row = format_trade_table_row(trade, "percentage")
        
        assert row[0] == "1"  # trade_id
        assert row[4] == "150.0"  # entry_price (unchanged)
        assert row[5] == "3.33%"  # exit_price as percentage change
    
    def test_create_enhanced_trade_summary(self):
        """Test enhanced trade summary creation"""
        trades = pd.DataFrame({
            'pnl': [100, -50, 75, -25, 150],
            'entry_price': [100, 110, 105, 115, 120],
            'exit_price': [110, 105, 115, 110, 135],
            'stop_loss': [95, 105, 100, 110, 115],
            'size': [100, 100, 100, 100, 100],
            'exit_reason': ['target', 'stop', 'target', 'stop', 'target']
        })
        
        summary = create_enhanced_trade_summary(trades)
        
        assert summary['total_trades'] == 5
        assert summary['winning_trades'] == 3
        assert summary['losing_trades'] == 2
        assert summary['win_rate'] == 60.0
        assert 'avg_entry_price' in summary
        assert 'avg_exit_price' in summary
        assert 'stop_loss_hit_rate' in summary
        assert 'avg_risk_per_trade' in summary
    
    def test_create_enhanced_trade_summary_empty(self):
        """Test enhanced trade summary with empty data"""
        empty_trades = pd.DataFrame()
        summary = create_enhanced_trade_summary(empty_trades)
        
        assert 'message' in summary
        assert 'No trades available' in summary['message']


class TestIntegrationEnhancedReporting:
    """Integration tests for enhanced trade reporting"""
    
    @pytest.fixture
    def comprehensive_backtest_data(self):
        """Create comprehensive backtest data with trade details"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Equity curve
        returns = np.random.normal(0.001, 0.02, len(dates))
        equity_curve = pd.Series((1 + returns).cumprod() * 100000, index=dates)
        
        # Detailed trades with all price information
        n_trades = 50
        trades = pd.DataFrame({
            'trade_id': range(1, n_trades + 1),
            'entry_time': pd.to_datetime(np.random.choice(dates[:-5], n_trades)),
            'exit_time': pd.to_datetime(np.random.choice(dates[5:], n_trades)),
            'side': np.random.choice(['long', 'short'], n_trades),
            'entry_price': np.random.uniform(100, 200, n_trades),
            'exit_price': np.random.uniform(95, 210, n_trades),
            'stop_loss': np.random.uniform(90, 110, n_trades),
            'take_profit': np.random.uniform(190, 220, n_trades),
            'size': np.random.uniform(100, 1000, n_trades),
            'duration': np.random.uniform(1, 48, n_trades),
            'exit_reason': np.random.choice(['target', 'stop', 'time'], n_trades)
        })
        
        # Calculate P&L
        trades['pnl'] = (trades['exit_price'] - trades['entry_price']) * trades['size']
        trades.loc[trades['side'] == 'short', 'pnl'] *= -1
        
        # Metrics
        metrics = {
            'total_return': 0.15,
            'annual_return': 0.15,
            'volatility': 0.18,
            'sharpe_ratio': 0.83,
            'max_drawdown': -0.12,
            'win_rate': 0.58,
            'profit_factor': 1.35,
            'avg_entry_price': trades['entry_price'].mean(),
            'avg_exit_price': trades['exit_price'].mean(),
            'stop_loss_hit_rate': 0.25,
            'avg_risk_per_trade': 0.025,
            'risk_reward_ratio': 1.8
        }
        
        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'metrics': metrics,
            'returns': equity_curve.pct_change().dropna(),
            'strategy_params': {
                'name': 'Enhanced Trade Test Strategy',
                'lookback': 20,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04
            }
        }
    
    def test_enhanced_reporting_config_integration(self, comprehensive_backtest_data):
        """Test that enhanced reporting config works with full data"""
        config = ReportConfig()
        
        # Test that trade reporting config is properly integrated
        assert config.trade_reporting.enable_detailed_trade_prices == True
        assert config.trade_reporting.enable_stop_loss_analysis == True
        assert config.trade_reporting.enable_risk_per_trade_analysis == True
        
        # Test that new metrics are available
        trades = comprehensive_backtest_data['trades']
        summary = create_enhanced_trade_summary(trades)
        
        expected_fields = [
            'total_trades', 'winning_trades', 'losing_trades', 'win_rate',
            'avg_entry_price', 'avg_exit_price', 'stop_loss_hit_rate',
            'avg_risk_per_trade'
        ]
        
        for field in expected_fields:
            assert field in summary
    
    def test_enhanced_visualizations_integration(self, comprehensive_backtest_data):
        """Test that enhanced visualizations work with comprehensive data"""
        viz = ReportVisualizations()
        trades = comprehensive_backtest_data['trades']
        
        # Test all new visualization functions
        trade_price_chart = viz.create_trade_price_chart(trades)
        stop_loss_chart = viz.create_stop_loss_analysis(trades)
        risk_chart = viz.create_trade_risk_chart(trades)
        
        # Verify charts are created successfully
        assert isinstance(trade_price_chart, go.Figure)
        assert isinstance(stop_loss_chart, go.Figure)
        assert isinstance(risk_chart, go.Figure)
        
        # Verify charts have data
        assert len(trade_price_chart.data) > 0
        assert len(stop_loss_chart.data) > 0
        assert len(risk_chart.data) > 0
    
    def test_enhanced_markdown_generation_integration(self, comprehensive_backtest_data):
        """Test that enhanced markdown generation works with comprehensive data"""
        trades = comprehensive_backtest_data['trades']
        
        # Create section data similar to what would be generated
        section_data = {
            'trade_statistics': {
                'summary': {
                    'total_trades': len(trades),
                    'avg_pnl': trades['pnl'].mean(),
                    'total_pnl': trades['pnl'].sum()
                }
            },
            'price_analysis': {
                'avg_entry_price': f"${trades['entry_price'].mean():.2f}",
                'avg_exit_price': f"${trades['exit_price'].mean():.2f}",
                'price_range': f"${trades['entry_price'].min():.2f} - ${trades['entry_price'].max():.2f}"
            },
            'stop_loss_analysis': {
                'stop_loss_hit_rate': f"{(trades['exit_reason'] == 'stop').mean() * 100:.1f}%",
                'avg_stop_distance': "2.5%"
            },
            'detailed_trades': trades.to_dict('records')[:10]  # First 10 trades
        }
        
        markdown = generate_trade_analysis_md(section_data)
        
        # Verify comprehensive markdown is generated
        assert "## Trade Analysis" in markdown
        assert "### Trade Statistics" in markdown
        assert "### Price Analysis" in markdown
        assert "### Stop Loss Analysis" in markdown
        assert "### Detailed Trades" in markdown
        
        # Verify data is properly included
        assert str(len(trades)) in markdown
        assert "avg_entry_price" in markdown.lower()
        assert "stop_loss_hit_rate" in markdown.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])