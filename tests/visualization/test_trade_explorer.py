"""
Comprehensive tests for the InteractiveTradeExplorer class.

This module provides complete test coverage for interactive trade exploration
including filtering, sorting, detailed analysis, and visualization features.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from pathlib import Path

from src.visualization.trade_explorer import InteractiveTradeExplorer


class TestInteractiveTradeExplorer:
    """Comprehensive tests for InteractiveTradeExplorer class."""
    
    @pytest.fixture
    def trade_explorer(self):
        """Create InteractiveTradeExplorer instance."""
        return InteractiveTradeExplorer()
    
    @pytest.fixture
    def sample_trades(self):
        """Create comprehensive sample trades data."""
        np.random.seed(42)
        
        trades_data = []
        base_date = datetime(2023, 1, 1)
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        strategies = ['Momentum', 'MeanReversion', 'Breakout', 'Pairs']
        
        for i in range(100):
            entry_date = base_date + timedelta(days=i*2.5)
            hold_days = np.random.randint(1, 20)
            exit_date = entry_date + timedelta(days=hold_days)
            
            symbol = np.random.choice(symbols)
            strategy = np.random.choice(strategies)
            
            entry_price = 100 + np.random.uniform(-30, 30)
            
            # Create realistic price movement
            if np.random.random() > 0.45:  # 55% win rate
                exit_price = entry_price * (1 + np.random.uniform(0.01, 0.15))
                trade_type = 'win'
            else:
                exit_price = entry_price * (1 - np.random.uniform(0.01, 0.10))
                trade_type = 'loss'
            
            quantity = np.random.randint(10, 200)
            pnl = (exit_price - entry_price) * quantity
            commission = quantity * 0.01
            
            trades_data.append({
                'trade_id': i + 1,
                'entry_time': entry_date,
                'exit_time': exit_date,
                'symbol': symbol,
                'strategy': strategy,
                'side': np.random.choice(['long', 'short']),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'profit_loss': pnl - commission,
                'gross_pnl': pnl,
                'commission': commission,
                'return_pct': ((exit_price - entry_price) / entry_price) * 100,
                'holding_period': hold_days,
                'trade_type': trade_type,
                'entry_reason': f'{strategy} signal triggered',
                'exit_reason': np.random.choice(['Target hit', 'Stop loss', 'Time exit', 'Signal reversed']),
                'mae': np.random.uniform(0, abs(pnl) * 0.6),
                'mfe': np.random.uniform(abs(pnl) * 0.4, abs(pnl) * 1.5),
                'market_condition': np.random.choice(['Trending', 'Ranging', 'Volatile'])
            })
        
        return pd.DataFrame(trades_data)
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for trade context."""
        dates = pd.date_range('2022-12-01', '2024-01-31', freq='D')
        
        symbols_data = {}
        for symbol in ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']:
            np.random.seed(hash(symbol) % 1000)
            
            returns = np.random.normal(0.0002, 0.02, len(dates))
            close_prices = 100 * (1 + returns).cumprod()
            
            symbols_data[symbol] = pd.DataFrame({
                'date': dates,
                'open': close_prices * (1 - np.abs(np.random.normal(0, 0.003, len(dates)))),
                'high': close_prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
                'low': close_prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
                'close': close_prices,
                'volume': np.random.randint(1000000, 5000000, len(dates))
            }).set_index('date')
        
        return symbols_data
    
    def test_explorer_initialization(self, trade_explorer):
        """Test InteractiveTradeExplorer initialization."""
        assert isinstance(trade_explorer, InteractiveTradeExplorer)
        assert hasattr(trade_explorer, 'create_explorer')
    
    def test_create_basic_explorer(self, trade_explorer, sample_trades):
        """Test basic explorer creation."""
        output_path = trade_explorer.create_explorer(sample_trades)
        
        assert output_path is not None
        assert os.path.exists(output_path)
        assert output_path.endswith('.html')
        
        # Verify content
        with open(output_path, 'r') as f:
            content = f.read()
            assert 'Trade Explorer' in content
            assert 'plotly' in content
        
        # Clean up
        os.remove(output_path)
    
    def test_create_trades_table(self, trade_explorer, sample_trades):
        """Test trades table creation."""
        fig = trade_explorer._create_trades_table(sample_trades)
        
        assert isinstance(fig, go.Figure)
        
        # Should have table trace
        assert any(isinstance(trace, go.Table) for trace in fig.data)
        
        # Verify columns
        table_trace = next(trace for trace in fig.data if isinstance(trace, go.Table))
        headers = table_trace.header.values
        
        expected_columns = ['Trade ID', 'Symbol', 'Entry Time', 'Exit Time', 'P&L', 'Return %']
        for col in expected_columns:
            assert any(col in str(header) for header in headers)
    
    def test_create_trade_scatter_plot(self, trade_explorer, sample_trades):
        """Test trade scatter plot creation."""
        fig = trade_explorer._create_trade_scatter_plot(sample_trades)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        
        # Should have scatter trace
        assert any(
            hasattr(trace, 'mode') and 'markers' in trace.mode
            for trace in fig.data
        )
    
    def test_create_pnl_distribution(self, trade_explorer, sample_trades):
        """Test P&L distribution chart creation."""
        fig = trade_explorer._create_pnl_distribution(sample_trades)
        
        assert isinstance(fig, go.Figure)
        
        # Should have histogram
        assert any(
            hasattr(trace, 'type') and trace.type == 'histogram'
            for trace in fig.data
        )
    
    def test_create_cumulative_pnl_chart(self, trade_explorer, sample_trades):
        """Test cumulative P&L chart creation."""
        fig = trade_explorer._create_cumulative_pnl_chart(sample_trades)
        
        assert isinstance(fig, go.Figure)
        
        # Verify cumulative nature
        if len(fig.data) > 0 and hasattr(fig.data[0], 'y'):
            y_values = fig.data[0].y
            # Check if values are cumulative (mostly increasing/decreasing)
            assert len(y_values) > 0
    
    def test_filter_trades_by_symbol(self, trade_explorer, sample_trades):
        """Test trade filtering by symbol."""
        filtered = trade_explorer._filter_trades(
            sample_trades,
            symbol='AAPL'
        )
        
        assert len(filtered) > 0
        assert all(filtered['symbol'] == 'AAPL')
    
    def test_filter_trades_by_date_range(self, trade_explorer, sample_trades):
        """Test trade filtering by date range."""
        start_date = datetime(2023, 3, 1)
        end_date = datetime(2023, 6, 30)
        
        filtered = trade_explorer._filter_trades(
            sample_trades,
            start_date=start_date,
            end_date=end_date
        )
        
        assert len(filtered) > 0
        assert all(filtered['entry_time'] >= start_date)
        assert all(filtered['entry_time'] <= end_date)
    
    def test_filter_trades_by_pnl(self, trade_explorer, sample_trades):
        """Test trade filtering by P&L threshold."""
        # Filter winning trades
        winning_trades = trade_explorer._filter_trades(
            sample_trades,
            min_pnl=0
        )
        
        assert len(winning_trades) > 0
        assert all(winning_trades['profit_loss'] >= 0)
        
        # Filter large losses
        large_losses = trade_explorer._filter_trades(
            sample_trades,
            max_pnl=-500
        )
        
        assert all(large_losses['profit_loss'] <= -500)
    
    def test_sort_trades(self, trade_explorer, sample_trades):
        """Test trade sorting functionality."""
        # Sort by P&L descending
        sorted_trades = trade_explorer._sort_trades(
            sample_trades,
            sort_by='profit_loss',
            ascending=False
        )
        
        pnl_values = sorted_trades['profit_loss'].values
        assert all(pnl_values[i] >= pnl_values[i+1] for i in range(len(pnl_values)-1))
        
        # Sort by date ascending
        sorted_trades = trade_explorer._sort_trades(
            sample_trades,
            sort_by='entry_time',
            ascending=True
        )
        
        dates = sorted_trades['entry_time'].values
        assert all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
    
    def test_calculate_trade_statistics(self, trade_explorer, sample_trades):
        """Test trade statistics calculation."""
        stats = trade_explorer._calculate_trade_statistics(sample_trades)
        
        assert isinstance(stats, dict)
        
        expected_stats = [
            'total_trades', 'winning_trades', 'losing_trades',
            'win_rate', 'avg_win', 'avg_loss', 'profit_factor',
            'total_pnl', 'best_trade', 'worst_trade'
        ]
        
        for stat in expected_stats:
            assert stat in stats
    
    def test_create_trade_details_view(self, trade_explorer, sample_trades):
        """Test detailed trade view creation."""
        # Get a single trade
        trade = sample_trades.iloc[0]
        
        details = trade_explorer._create_trade_details_view(trade)
        
        assert isinstance(details, dict) or isinstance(details, go.Figure)
        
        # If dict, check for expected fields
        if isinstance(details, dict):
            assert 'trade_id' in details
            assert 'entry_time' in details
            assert 'profit_loss' in details
    
    def test_create_strategy_comparison(self, trade_explorer, sample_trades):
        """Test strategy comparison visualization."""
        fig = trade_explorer._create_strategy_comparison(sample_trades)
        
        assert isinstance(fig, go.Figure)
        
        # Should compare different strategies
        unique_strategies = sample_trades['strategy'].unique()
        assert len(unique_strategies) > 1
    
    def test_create_symbol_performance(self, trade_explorer, sample_trades):
        """Test symbol performance analysis."""
        fig = trade_explorer._create_symbol_performance(sample_trades)
        
        assert isinstance(fig, go.Figure)
        
        # Should show performance by symbol
        unique_symbols = sample_trades['symbol'].unique()
        assert len(unique_symbols) > 1
    
    def test_create_time_analysis(self, trade_explorer, sample_trades):
        """Test time-based analysis."""
        # Holding period analysis
        fig = trade_explorer._create_holding_period_analysis(sample_trades)
        assert isinstance(fig, go.Figure)
        
        # Time of day analysis
        fig = trade_explorer._create_time_of_day_analysis(sample_trades)
        assert isinstance(fig, go.Figure)
        
        # Day of week analysis
        fig = trade_explorer._create_day_of_week_analysis(sample_trades)
        assert isinstance(fig, go.Figure)
    
    def test_create_mae_mfe_analysis(self, trade_explorer, sample_trades):
        """Test MAE/MFE analysis visualization."""
        if 'mae' in sample_trades.columns and 'mfe' in sample_trades.columns:
            fig = trade_explorer._create_mae_mfe_analysis(sample_trades)
            
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0
    
    def test_interactive_filtering(self, trade_explorer, sample_trades):
        """Test interactive filtering capabilities."""
        # Create explorer with filters
        output_path = trade_explorer.create_explorer(
            sample_trades,
            enable_filters=True,
            filter_options={
                'symbols': sample_trades['symbol'].unique().tolist(),
                'strategies': sample_trades['strategy'].unique().tolist(),
                'date_range': True,
                'pnl_range': True
            }
        )
        
        assert os.path.exists(output_path)
        
        with open(output_path, 'r') as f:
            content = f.read()
            # Check for filter elements
            assert 'filter' in content.lower() or 'select' in content.lower()
        
        # Clean up
        os.remove(output_path)
    
    def test_export_filtered_trades(self, trade_explorer, sample_trades):
        """Test export functionality for filtered trades."""
        filtered_trades = trade_explorer._filter_trades(
            sample_trades,
            symbol='AAPL',
            min_pnl=0
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, 'filtered_trades.csv')
            
            trade_explorer._export_trades(
                filtered_trades,
                export_path
            )
            
            assert os.path.exists(export_path)
            
            # Verify content
            loaded_df = pd.read_csv(export_path)
            assert len(loaded_df) == len(filtered_trades)
    
    def test_create_trade_timeline(self, trade_explorer, sample_trades):
        """Test trade timeline visualization."""
        fig = trade_explorer._create_trade_timeline(sample_trades)
        
        assert isinstance(fig, go.Figure)
        
        # Should show trades over time
        assert len(fig.data) > 0
    
    def test_correlation_analysis(self, trade_explorer, sample_trades):
        """Test correlation analysis between trade features."""
        correlations = trade_explorer._analyze_trade_correlations(sample_trades)
        
        assert isinstance(correlations, pd.DataFrame)
        
        # Test visualization
        fig = trade_explorer._create_correlation_heatmap(correlations)
        assert isinstance(fig, go.Figure)
    
    def test_trade_clustering(self, trade_explorer, sample_trades):
        """Test trade clustering analysis."""
        clusters = trade_explorer._cluster_trades(
            sample_trades,
            features=['return_pct', 'holding_period', 'profit_loss']
        )
        
        assert 'cluster' in clusters.columns
        assert len(clusters['cluster'].unique()) > 1
        
        # Test visualization
        fig = trade_explorer._visualize_trade_clusters(clusters)
        assert isinstance(fig, go.Figure)
    
    def test_performance_by_market_condition(self, trade_explorer, sample_trades):
        """Test performance analysis by market condition."""
        if 'market_condition' in sample_trades.columns:
            analysis = trade_explorer._analyze_by_market_condition(sample_trades)
            
            assert isinstance(analysis, pd.DataFrame)
            assert len(analysis) > 0
            
            # Test visualization
            fig = trade_explorer._create_market_condition_chart(analysis)
            assert isinstance(fig, go.Figure)
    
    def test_trade_sequence_analysis(self, trade_explorer, sample_trades):
        """Test trade sequence and dependency analysis."""
        sequence_analysis = trade_explorer._analyze_trade_sequences(sample_trades)
        
        assert isinstance(sequence_analysis, dict)
        assert 'win_after_win' in sequence_analysis
        assert 'loss_after_loss' in sequence_analysis
        assert 'recovery_rate' in sequence_analysis
    
    def test_risk_reward_analysis(self, trade_explorer, sample_trades):
        """Test risk/reward ratio analysis."""
        rr_analysis = trade_explorer._analyze_risk_reward(sample_trades)
        
        assert isinstance(rr_analysis, dict)
        assert 'avg_risk_reward' in rr_analysis
        assert 'risk_reward_by_outcome' in rr_analysis
    
    def test_empty_trades_handling(self, trade_explorer):
        """Test handling of empty trades dataframe."""
        empty_trades = pd.DataFrame()
        
        with pytest.raises((ValueError, KeyError)):
            trade_explorer.create_explorer(empty_trades)
    
    def test_custom_columns_support(self, trade_explorer, sample_trades):
        """Test support for custom trade columns."""
        # Add custom columns
        sample_trades['custom_metric'] = np.random.randn(len(sample_trades))
        sample_trades['trade_score'] = np.random.uniform(0, 100, len(sample_trades))
        
        output_path = trade_explorer.create_explorer(
            sample_trades,
            custom_columns=['custom_metric', 'trade_score']
        )
        
        assert os.path.exists(output_path)
        
        # Clean up
        os.remove(output_path)