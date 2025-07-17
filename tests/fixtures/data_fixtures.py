"""
Data Fixtures for Testing

Provides comprehensive data fixtures that avoid pandas compatibility issues
and represent realistic financial data structures.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union


def create_sample_price_data(
    periods: int = 252,
    start_date: str = '2023-01-01',
    freq: str = 'D',
    initial_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Create realistic price data with proper OHLCV structure.
    
    Args:
        periods: Number of periods
        start_date: Start date
        freq: Frequency ('D', 'H', 'min')
        initial_price: Initial price level
        volatility: Daily volatility
        trend: Daily trend component
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create date index
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Generate returns with trend and volatility
    returns = np.random.normal(trend, volatility, periods)
    
    # Create cumulative price series
    close_prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC data
    daily_range = np.abs(np.random.normal(0, volatility/2, periods))
    
    open_prices = close_prices * (1 + np.random.normal(0, volatility/4, periods))
    high_prices = np.maximum(open_prices, close_prices) + daily_range * close_prices
    low_prices = np.minimum(open_prices, close_prices) - daily_range * close_prices
    
    # Generate volume data
    base_volume = 1000000
    volume = base_volume + np.random.exponential(base_volume/10, periods).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    # Ensure proper OHLC relationships
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def create_sample_ohlcv_data(
    symbols: List[str] = ['AAPL', 'GOOGL', 'MSFT'],
    periods: int = 100,
    start_date: str = '2023-01-01',
    seed: Optional[int] = 42
) -> Dict[str, pd.DataFrame]:
    """
    Create OHLCV data for multiple symbols.
    
    Args:
        symbols: List of symbols
        periods: Number of periods per symbol
        start_date: Start date
        seed: Random seed
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    if seed is not None:
        np.random.seed(seed)
    
    data = {}
    
    for i, symbol in enumerate(symbols):
        # Vary parameters by symbol
        initial_price = 100 * (1 + i * 0.5)
        volatility = 0.02 * (1 + i * 0.1)
        trend = 0.0001 * (1 - i * 0.05)
        
        data[symbol] = create_sample_price_data(
            periods=periods,
            start_date=start_date,
            initial_price=initial_price,
            volatility=volatility,
            trend=trend,
            seed=seed + i if seed else None
        )
    
    return data


def create_sample_returns_data(
    periods: int = 252,
    start_date: str = '2023-01-01',
    freq: str = 'D',
    mean_return: float = 0.0001,
    volatility: float = 0.02,
    seed: Optional[int] = 42
) -> pd.Series:
    """
    Create returns data.
    
    Args:
        periods: Number of periods
        start_date: Start date
        freq: Frequency
        mean_return: Mean return
        volatility: Return volatility
        seed: Random seed
        
    Returns:
        Series of returns with datetime index
    """
    if seed is not None:
        np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    returns = np.random.normal(mean_return, volatility, periods)
    
    return pd.Series(returns, index=dates, name='returns')


def create_empty_dataframe(columns: List[str] = None) -> pd.DataFrame:
    """
    Create an empty DataFrame with specified columns.
    
    Args:
        columns: List of column names
        
    Returns:
        Empty DataFrame with proper structure
    """
    if columns is None:
        columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Create with datetime index to avoid issues
    index = pd.DatetimeIndex([], name='date')
    
    df = pd.DataFrame(columns=columns, index=index)
    
    # Set proper dtypes
    numeric_columns = [col for col in columns if col not in ['symbol', 'side', 'exit_reason']]
    for col in numeric_columns:
        df[col] = pd.Series(dtype='float64')
    
    return df


def create_single_row_dataframe(
    date: str = '2023-01-01',
    price: float = 100.0
) -> pd.DataFrame:
    """
    Create a DataFrame with a single row of data.
    
    Args:
        date: Date for the row
        price: Base price
        
    Returns:
        DataFrame with single row
    """
    index = pd.DatetimeIndex([date])
    
    df = pd.DataFrame({
        'open': [price],
        'high': [price * 1.01],
        'low': [price * 0.99],
        'close': [price * 1.005],
        'volume': [1000000]
    }, index=index)
    
    return df


def create_multi_asset_data(
    assets: List[str] = ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
    periods: int = 252,
    start_date: str = '2023-01-01',
    correlation: float = 0.3,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Create correlated multi-asset price data.
    
    Args:
        assets: List of asset names
        periods: Number of periods
        start_date: Start date
        correlation: Average correlation between assets
        seed: Random seed
        
    Returns:
        DataFrame with assets as columns, dates as index
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_assets = len(assets)
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Create correlation matrix
    corr_matrix = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Generate correlated returns
    mean_returns = np.random.uniform(-0.0001, 0.0003, n_assets)
    volatilities = np.random.uniform(0.015, 0.025, n_assets)
    
    # Cholesky decomposition for correlated random numbers
    L = np.linalg.cholesky(corr_matrix)
    
    # Generate independent random returns
    independent_returns = np.random.normal(0, 1, (periods, n_assets))
    
    # Create correlated returns
    correlated_returns = independent_returns @ L.T
    
    # Scale by volatility and add mean
    returns = correlated_returns * volatilities + mean_returns
    
    # Convert to prices
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    # Create DataFrame
    df = pd.DataFrame(prices, index=dates, columns=assets)
    
    return df


def create_intraday_data(
    date: str = '2023-01-01',
    hours: int = 8,
    minute_bars: bool = True,
    initial_price: float = 100.0,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Create intraday price data.
    
    Args:
        date: Date for intraday data
        hours: Number of trading hours
        minute_bars: If True, create minute bars; else hour bars
        initial_price: Starting price
        seed: Random seed
        
    Returns:
        DataFrame with intraday OHLCV data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create time index
    start_time = pd.Timestamp(f'{date} 09:30:00')
    
    if minute_bars:
        periods = hours * 60
        freq = 'min'
        volatility = 0.0002  # Minute volatility
    else:
        periods = hours
        freq = 'H'
        volatility = 0.002  # Hourly volatility
    
    times = pd.date_range(start=start_time, periods=periods, freq=freq)
    
    # Generate intraday price movement
    returns = np.random.normal(0, volatility, periods)
    
    # Add intraday patterns (U-shape volume, higher volatility at open/close)
    time_of_day = np.arange(periods) / periods
    volatility_multiplier = 1 + 0.5 * (np.abs(time_of_day - 0.5) * 2) ** 2
    returns = returns * volatility_multiplier
    
    # Create prices
    close_prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    tick_range = np.abs(np.random.normal(0, volatility/2, periods))
    
    open_prices = close_prices * (1 + np.random.normal(0, volatility/4, periods))
    high_prices = np.maximum(open_prices, close_prices) + tick_range * close_prices
    low_prices = np.minimum(open_prices, close_prices) - tick_range * close_prices
    
    # U-shaped volume pattern
    base_volume = 10000 if minute_bars else 100000
    volume_pattern = 1 + 2 * (np.abs(time_of_day - 0.5) * 2) ** 2
    volume = (base_volume * volume_pattern + 
              np.random.exponential(base_volume/10, periods)).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=times)
    
    # Ensure proper OHLC relationships
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def create_missing_data_sample(
    periods: int = 100,
    missing_pct: float = 0.1,
    pattern: str = 'random',
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Create price data with missing values for testing.
    
    Args:
        periods: Number of periods
        missing_pct: Percentage of missing data
        pattern: 'random', 'sequential', or 'end'
        seed: Random seed
        
    Returns:
        DataFrame with missing values
    """
    # Start with complete data
    df = create_sample_price_data(periods=periods, seed=seed)
    
    n_missing = int(periods * missing_pct)
    
    if pattern == 'random':
        # Random missing values
        missing_idx = np.random.choice(periods, n_missing, replace=False)
        df.iloc[missing_idx] = np.nan
    
    elif pattern == 'sequential':
        # Sequential missing values (e.g., weekend/holiday gaps)
        start_idx = np.random.randint(0, periods - n_missing)
        df.iloc[start_idx:start_idx + n_missing] = np.nan
    
    elif pattern == 'end':
        # Missing values at the end (incomplete data)
        df.iloc[-n_missing:] = np.nan
    
    return df