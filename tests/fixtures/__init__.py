"""
Test Fixtures Package

This package provides comprehensive test fixtures for all modules,
designed to prevent pandas compatibility issues and provide
realistic financial data structures.
"""

from .data_fixtures import *
from .analysis_fixtures import *
from .visualization_fixtures import *
from .trade_fixtures import *
from .performance_fixtures import *

__all__ = [
    # Data fixtures
    'create_sample_price_data',
    'create_sample_ohlcv_data',
    'create_sample_returns_data',
    'create_empty_dataframe',
    'create_single_row_dataframe',
    
    # Analysis fixtures
    'create_strategy_performance_data',
    'create_benchmark_data',
    'create_portfolio_data',
    'create_market_data',
    
    # Visualization fixtures
    'create_equity_curve_data',
    'create_drawdown_data',
    'create_trade_scatter_data',
    'create_heatmap_data',
    
    # Trade fixtures
    'create_trade_data',
    'create_trade_history',
    'create_position_data',
    'create_order_data',
    
    # Performance fixtures
    'create_performance_metrics',
    'create_rolling_metrics',
    'create_sharpe_data',
    'create_returns_distribution'
]