# Test Fixtures Documentation

This package provides comprehensive test fixtures for the Backtest Suite, designed to create realistic financial data while avoiding pandas compatibility issues.

## Overview

The fixtures are organized into five main categories:

1. **Data Fixtures** - Basic price data, OHLCV, returns, multi-asset data
2. **Analysis Fixtures** - Strategy performance, benchmarks, portfolios, market data
3. **Visualization Fixtures** - Equity curves, drawdowns, heatmaps, charts
4. **Trade Fixtures** - Trades, positions, orders, execution data
5. **Performance Fixtures** - Metrics, rolling analysis, distributions

## Quick Start

```python
from tests.fixtures import (
    create_sample_price_data,
    create_trade_data,
    create_equity_curve_data,
    create_performance_metrics
)

# Create basic price data
price_data = create_sample_price_data(periods=252)

# Create trade data with metadata
trades = create_trade_data(
    n_trades=100,
    win_rate=0.60,
    include_metadata=True
)

# Create equity curve for visualization
equity_curve = create_equity_curve_data(
    annual_return=0.15,
    volatility=0.20
)

# Create comprehensive performance metrics
metrics = create_performance_metrics(
    target_sharpe=1.5,
    target_annual_return=0.15
)
```

## Key Features

### 1. Pandas Compatibility
All fixtures are designed to avoid common pandas issues:
- Proper datetime indices
- Consistent dtypes
- No `_NoValueType` errors
- Safe empty DataFrame handling

### 2. Realistic Financial Data
- Proper OHLC relationships
- Realistic volatility patterns
- Correlated multi-asset returns
- Market microstructure features

### 3. Comprehensive Coverage
- Edge cases (empty, single row)
- Missing data patterns
- Intraday and daily frequencies
- Multiple asset classes

## Fixture Categories

### Data Fixtures (`data_fixtures.py`)

```python
# Basic price data
price_df = create_sample_price_data(
    periods=252,
    initial_price=100,
    volatility=0.02,
    trend=0.0001
)

# Multi-asset correlated data
multi_asset_df = create_multi_asset_data(
    assets=['AAPL', 'GOOGL', 'MSFT'],
    correlation=0.3
)

# Intraday minute bars
intraday_df = create_intraday_data(
    date='2023-01-01',
    hours=8,
    minute_bars=True
)

# Handle edge cases
empty_df = create_empty_dataframe()
single_row_df = create_single_row_dataframe()
```

### Analysis Fixtures (`analysis_fixtures.py`)

```python
# Strategy performance with target metrics
perf_data = create_strategy_performance_data(
    annual_return=0.15,
    annual_volatility=0.20,
    sharpe_ratio=0.75,
    max_drawdown=-0.15
)

# Portfolio allocation data
portfolio_data = create_portfolio_data(
    n_assets=5,
    rebalance_frequency='monthly'
)

# Market data with sectors and factors
market_data = create_market_data(
    sectors=['Tech', 'Finance', 'Healthcare'],
    include_factors=True
)
```

### Visualization Fixtures (`visualization_fixtures.py`)

```python
# Equity curve for charts
equity = create_equity_curve_data(smooth=True)

# Drawdown analysis
drawdown_df = create_drawdown_data(equity_curve=equity)

# Trade scatter plot data
scatter_data = create_trade_scatter_data(
    n_trades=100,
    win_rate=0.55
)

# Monthly returns heatmap
heatmap = create_heatmap_data(years=3)

# Candlestick with signals
candles = create_candlestick_data(
    include_signals=True
)
```

### Trade Fixtures (`trade_fixtures.py`)

```python
# Comprehensive trade data
trades = create_trade_data(
    n_trades=50,
    win_rate=0.60,
    include_metadata=True  # Adds stops, commissions, etc.
)

# Current positions
positions = create_position_data(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    include_closed=False
)

# Order execution data
orders = create_order_data(
    n_orders=20,
    include_cancelled=True
)

# Detailed trade history
history = create_trade_history(
    trade_id=1,
    n_updates=10
)
```

### Performance Fixtures (`performance_fixtures.py`)

```python
# Comprehensive metrics
metrics = create_performance_metrics(
    target_sharpe=1.5,
    target_annual_return=0.15
)

# Rolling performance analysis
rolling = create_rolling_metrics_data(
    windows=[20, 60, 252]
)

# Returns with specific distributions
returns = create_returns_distribution(
    distribution_type='fat_tails',
    skew=-0.5,
    excess_kurtosis=3.0
)

# Factor performance analysis
factors = create_factor_performance_data(
    factors=['momentum', 'value', 'quality'],
    include_regime_changes=True
)
```

## Configuration

The `fixture_config.py` module provides centralized configuration:

```python
from tests.fixtures.fixture_config import FixtureConfig

# Access default parameters
defaults = FixtureConfig.DEFAULT_SYMBOLS
date_range = FixtureConfig.get_default_date_range()

# Ensure proper dtypes
df = FixtureConfig.ensure_proper_dtypes(df)

# Validate OHLCV relationships
df = FixtureConfig.validate_ohlcv(df)

# Handle empty DataFrames safely
df = FixtureConfig.handle_empty_data(df)
```

## Best Practices

### 1. Always Set Random Seed
```python
data = create_sample_price_data(seed=42)  # Reproducible
```

### 2. Use Proper Date Handling
```python
# Good: Use string dates
data = create_trade_data(start_date='2023-01-01')

# Also good: Use pandas Timestamp
data = create_trade_data(start_date=pd.Timestamp('2023-01-01'))
```

### 3. Handle Edge Cases
```python
# Test with empty data
empty_trades = create_trade_data(n_trades=0)

# Test with single data point
single_price = create_sample_price_data(periods=1)
```

### 4. Verify Data Integrity
```python
# Always verify OHLC relationships
assert (df['high'] >= df['low']).all()
assert (df['high'] >= df[['open', 'close']].max(axis=1)).all()

# Verify datetime index
assert isinstance(df.index, pd.DatetimeIndex)
```

## Common Issues and Solutions

### Issue: Pandas `_NoValueType` Error
**Solution**: Use fixtures that properly initialize DataFrames with correct dtypes

### Issue: Empty DataFrame Operations
**Solution**: Use `create_empty_dataframe()` which handles empty data safely

### Issue: Missing Data in Visualizations
**Solution**: Use fixtures with `handle_missing_data()` patterns

### Issue: Unrealistic Test Data
**Solution**: Use parameter ranges based on real market data (see FixtureConfig)

## Examples

See `example_usage.py` for comprehensive examples including:
- Basic usage patterns
- Integration testing
- Edge case handling
- Pandas compatibility testing
- Multi-timeframe analysis

## Contributing

When adding new fixtures:
1. Follow the existing naming convention: `create_[type]_data()`
2. Include proper docstrings with parameter descriptions
3. Set default random seed for reproducibility
4. Ensure pandas compatibility (no `_NoValueType` issues)
5. Add validation for data integrity
6. Include edge case handling