"""
Example Usage of Test Fixtures

This module demonstrates how to use the comprehensive test fixtures
to create test data that avoids pandas compatibility issues.
"""

import pytest
import pandas as pd
import numpy as np

from tests.fixtures import (
    # Data fixtures
    create_sample_price_data,
    create_sample_ohlcv_data,
    create_empty_dataframe,
    create_multi_asset_data,
    
    # Analysis fixtures
    create_strategy_performance_data,
    create_benchmark_data,
    create_portfolio_data,
    create_market_data,
    
    # Visualization fixtures
    create_equity_curve_data,
    create_drawdown_data,
    create_trade_scatter_data,
    create_heatmap_data,
    create_rolling_metrics_data,
    
    # Trade fixtures
    create_trade_data,
    create_position_data,
    create_order_data,
    
    # Performance fixtures
    create_performance_metrics,
    create_sharpe_data,
    create_returns_distribution
)

from tests.fixtures.fixture_config import FixtureConfig, get_fixture_defaults


class TestFixtureExamples:
    """Examples of using test fixtures in tests."""
    
    def test_basic_price_data(self):
        """Example: Create basic OHLCV price data."""
        # Create price data with default parameters
        price_data = create_sample_price_data()
        
        # Verify structure
        assert isinstance(price_data, pd.DataFrame)
        assert len(price_data) == 252  # One year of daily data
        assert all(col in price_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        
        # Verify OHLC relationships
        assert (price_data['high'] >= price_data['low']).all()
        assert (price_data['high'] >= price_data['open']).all()
        assert (price_data['high'] >= price_data['close']).all()
        
        # Verify datetime index
        assert isinstance(price_data.index, pd.DatetimeIndex)
    
    def test_empty_data_handling(self):
        """Example: Handle empty DataFrames safely."""
        # Create empty DataFrame
        empty_df = create_empty_dataframe()
        
        # Verify it's safe to use
        assert isinstance(empty_df, pd.DataFrame)
        assert len(empty_df) == 0
        assert len(empty_df.columns) == 5  # Default OHLCV columns
        
        # Operations that might fail with improper empty DataFrames
        mean_price = empty_df['close'].mean()  # Should not raise error
        assert pd.isna(mean_price)
    
    def test_comprehensive_performance_data(self):
        """Example: Create comprehensive performance analysis data."""
        # Create performance data
        perf_data = create_strategy_performance_data(
            annual_return=0.20,
            annual_volatility=0.15,
            sharpe_ratio=1.33,
            max_drawdown=-0.10
        )
        
        # Access different components
        returns_df = perf_data['returns']
        metrics_df = perf_data['metrics']
        monthly_returns = perf_data['monthly_returns']
        
        # Verify metrics are within expected ranges
        assert metrics_df['sharpe_ratio'].iloc[0] > 1.0
        assert metrics_df['max_drawdown'].iloc[0] > -0.15
        assert metrics_df['annual_return'].iloc[0] > 0.15
    
    def test_visualization_ready_data(self):
        """Example: Create data ready for visualization."""
        # Create equity curve for plotting
        equity_curve = create_equity_curve_data(
            periods=252,
            annual_return=0.15,
            volatility=0.20,
            smooth=True  # Smoother curve for better visualization
        )
        
        # Create drawdown data
        drawdown_data = create_drawdown_data(equity_curve=equity_curve)
        
        # Verify structure for visualization
        assert 'equity' in drawdown_data.columns
        assert 'drawdown' in drawdown_data.columns
        assert 'drawdown_pct' in drawdown_data.columns
        assert 'duration_days' in drawdown_data.columns
        
        # Data should be ready for plotting without further processing
        assert drawdown_data['drawdown'].min() < 0  # Has drawdowns
        assert drawdown_data['equity'].iloc[-1] > drawdown_data['equity'].iloc[0]  # Profitable
    
    def test_realistic_trade_data(self):
        """Example: Create realistic trade data with all metadata."""
        # Create trades with full metadata
        trades = create_trade_data(
            n_trades=100,
            win_rate=0.60,
            avg_win_pnl=800,
            avg_loss_pnl=-400,
            include_metadata=True
        )
        
        # Verify comprehensive trade data
        required_columns = [
            'trade_id', 'symbol', 'side', 'entry_date', 'exit_date',
            'entry_price', 'exit_price', 'position_size', 'pnl',
            'stop_loss', 'take_profit', 'exit_reason', 'commission',
            'slippage', 'net_pnl', 'strategy', 'mae', 'mfe'
        ]
        
        assert all(col in trades.columns for col in required_columns)
        
        # Verify realistic relationships
        assert (trades['commission'] > 0).all()  # All trades have commission
        assert (trades['mfe'] >= 0).all()  # MFE is always positive
        assert (trades['mae'] <= 0).all()  # MAE is always negative
        
        # Win rate should be close to target
        actual_win_rate = trades['is_winner'].mean()
        assert 0.55 <= actual_win_rate <= 0.65
    
    def test_multi_timeframe_analysis(self):
        """Example: Create data for multi-timeframe analysis."""
        # Daily data
        daily_data = create_sample_price_data(periods=252, freq='D')
        
        # Intraday data for the same period
        intraday_data = create_sample_price_data(
            periods=252 * 8 * 60,  # Minutes in trading day
            freq='min',
            volatility=0.002  # Lower volatility for minute data
        )
        
        # Rolling metrics for multiple windows
        returns = daily_data['close'].pct_change()
        rolling_metrics = create_rolling_metrics_data(
            returns=returns,
            windows=[5, 20, 60]  # Weekly, monthly, quarterly
        )
        
        # Verify multi-timeframe structure
        assert 'window_5' in rolling_metrics
        assert 'window_20' in rolling_metrics
        assert 'window_60' in rolling_metrics
        
        # Each window should have comprehensive metrics
        for window_key in rolling_metrics:
            window_data = rolling_metrics[window_key]
            assert 'rolling_sharpe' in window_data.columns
            assert 'rolling_volatility' in window_data.columns
            assert 'rolling_max_drawdown' in window_data.columns
    
    def test_edge_case_handling(self):
        """Example: Test edge cases with fixtures."""
        # Single row data
        single_row = create_sample_price_data(periods=1)
        assert len(single_row) == 1
        assert not single_row.empty
        
        # Very volatile data
        volatile_data = create_sample_price_data(
            volatility=0.50,  # 50% annual volatility
            periods=100
        )
        
        daily_returns = volatile_data['close'].pct_change()
        assert daily_returns.std() > 0.02  # High daily volatility
        
        # No winning trades
        losing_trades = create_trade_data(
            n_trades=50,
            win_rate=0.0,  # All losing trades
            avg_loss_pnl=-500
        )
        
        assert (losing_trades['pnl'] < 0).all()
        assert losing_trades['is_winner'].sum() == 0
    
    def test_pandas_compatibility(self):
        """Example: Ensure fixtures work with pandas operations."""
        # Create various data types
        price_data = create_sample_price_data()
        trade_data = create_trade_data()
        position_data = create_position_data()
        
        # Test common pandas operations that might fail with improper data
        
        # Groupby operations
        trade_summary = trade_data.groupby('symbol').agg({
            'pnl': ['sum', 'mean', 'count'],
            'duration_days': 'mean'
        })
        assert not trade_summary.empty
        
        # Merge operations
        merged = pd.merge(
            trade_data[['symbol', 'pnl']],
            position_data[['symbol', 'market_value']],
            on='symbol',
            how='outer'
        )
        assert len(merged) > 0
        
        # Resample operations
        monthly_ohlc = price_data.resample('M').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        assert len(monthly_ohlc) > 0
        
        # Rolling calculations
        rolling_mean = price_data['close'].rolling(20).mean()
        assert rolling_mean.notna().sum() > 0
    
    @pytest.fixture
    def complete_test_dataset(self):
        """Fixture providing a complete dataset for integration tests."""
        defaults = get_fixture_defaults()
        
        # Create comprehensive test data
        dataset = {
            'price_data': create_sample_ohlcv_data(
                symbols=defaults['symbols'],
                periods=defaults['periods']
            ),
            'performance': create_performance_metrics(
                periods=defaults['periods'],
                target_sharpe=defaults['sharpe_ratio']
            ),
            'trades': create_trade_data(
                n_trades=100,
                symbols=defaults['symbols'],
                win_rate=defaults['win_rate']
            ),
            'positions': create_position_data(
                symbols=defaults['symbols']
            ),
            'market_data': create_market_data(
                periods=defaults['periods'],
                include_factors=True
            )
        }
        
        return dataset
    
    def test_integration_with_complete_dataset(self, complete_test_dataset):
        """Example: Use complete dataset for integration testing."""
        dataset = complete_test_dataset
        
        # All components should be present
        assert 'price_data' in dataset
        assert 'performance' in dataset
        assert 'trades' in dataset
        assert 'positions' in dataset
        assert 'market_data' in dataset
        
        # Components should be compatible
        symbols = dataset['positions']['symbol'].unique()
        for symbol in symbols:
            if symbol != 'PORTFOLIO_TOTAL':
                assert symbol in dataset['price_data']
                assert symbol in dataset['trades']['symbol'].values
        
        # Performance metrics should be complete
        metrics = dataset['performance']
        assert metrics['sharpe_ratio'] > 0
        assert metrics['total_return'] != 0
        assert 'equity_curve' in metrics
        assert 'drawdown' in metrics


if __name__ == "__main__":
    # Example of running tests
    test = TestFixtureExamples()
    
    print("Testing basic price data...")
    test.test_basic_price_data()
    print("✓ Basic price data test passed")
    
    print("\nTesting empty data handling...")
    test.test_empty_data_handling()
    print("✓ Empty data handling test passed")
    
    print("\nTesting visualization ready data...")
    test.test_visualization_ready_data()
    print("✓ Visualization data test passed")
    
    print("\nTesting realistic trade data...")
    test.test_realistic_trade_data()
    print("✓ Trade data test passed")
    
    print("\nTesting pandas compatibility...")
    test.test_pandas_compatibility()
    print("✓ Pandas compatibility test passed")
    
    print("\nAll fixture examples passed! ✅")