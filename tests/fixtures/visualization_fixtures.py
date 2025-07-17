"""
Visualization Fixtures for Testing

Provides comprehensive fixtures specifically designed for visualization modules,
ensuring proper data structures that work with pandas and plotly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any


def create_equity_curve_data(
    periods: int = 252,
    start_date: str = '2023-01-01',
    initial_value: float = 100000,
    annual_return: float = 0.15,
    volatility: float = 0.20,
    smooth: bool = False,
    seed: Optional[int] = 42
) -> pd.Series:
    """
    Create equity curve data for visualization.
    
    Args:
        periods: Number of periods
        start_date: Start date
        initial_value: Starting portfolio value
        annual_return: Target annual return
        volatility: Annual volatility
        smooth: Whether to smooth the curve
        seed: Random seed
        
    Returns:
        Series with equity values and datetime index
    """
    if seed is not None:
        np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Generate returns
    daily_return = annual_return / 252
    daily_vol = volatility / np.sqrt(252)
    
    if smooth:
        # Generate smoother equity curve
        trend = np.linspace(0, daily_return * periods, periods)
        noise = np.cumsum(np.random.normal(0, daily_vol, periods))
        equity_values = initial_value * np.exp(trend + noise * 0.3)
    else:
        # Generate realistic equity curve
        returns = np.random.normal(daily_return, daily_vol, periods)
        equity_values = initial_value * (1 + returns).cumprod()
    
    return pd.Series(equity_values, index=dates, name='equity')


def create_drawdown_data(
    equity_curve: Optional[pd.Series] = None,
    periods: int = 252,
    start_date: str = '2023-01-01',
    max_drawdown: float = -0.20,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Create drawdown data for visualization.
    
    Args:
        equity_curve: Existing equity curve (if None, creates one)
        periods: Number of periods
        start_date: Start date
        max_drawdown: Target maximum drawdown
        seed: Random seed
        
    Returns:
        DataFrame with drawdown data
    """
    if equity_curve is None:
        equity_curve = create_equity_curve_data(
            periods=periods,
            start_date=start_date,
            seed=seed
        )
    
    # Calculate drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    
    # Calculate drawdown duration
    is_drawdown = drawdown < 0
    drawdown_groups = (is_drawdown != is_drawdown.shift()).cumsum()
    
    duration = pd.Series(index=equity_curve.index, dtype='float64')
    for group in drawdown_groups[is_drawdown].unique():
        mask = (drawdown_groups == group) & is_drawdown
        group_duration = mask.cumsum()
        duration[mask] = group_duration[mask]
    
    # Create DataFrame
    drawdown_df = pd.DataFrame({
        'equity': equity_curve,
        'rolling_max': rolling_max,
        'drawdown': drawdown,
        'drawdown_pct': drawdown * 100,
        'duration_days': duration,
        'is_drawdown': is_drawdown
    })
    
    return drawdown_df


def create_trade_scatter_data(
    n_trades: int = 100,
    start_date: str = '2023-01-01',
    end_date: str = '2023-12-31',
    win_rate: float = 0.55,
    avg_win: float = 500,
    avg_loss: float = -300,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Create trade scatter plot data.
    
    Args:
        n_trades: Number of trades
        start_date: Start date
        end_date: End date
        win_rate: Winning trade percentage
        avg_win: Average winning trade profit
        avg_loss: Average losing trade loss
        seed: Random seed
        
    Returns:
        DataFrame with trade data for scatter plots
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate trade dates
    date_range = pd.date_range(start=start_date, end=end_date)
    trade_dates = np.sort(np.random.choice(date_range, n_trades, replace=False))
    
    # Generate trade outcomes
    n_wins = int(n_trades * win_rate)
    outcomes = np.array([True] * n_wins + [False] * (n_trades - n_wins))
    np.random.shuffle(outcomes)
    
    # Generate P&L
    pnl = np.where(
        outcomes,
        np.random.normal(avg_win, avg_win * 0.5, n_trades),
        np.random.normal(avg_loss, abs(avg_loss) * 0.5, n_trades)
    )
    
    # Generate hold times (hours)
    hold_times = np.random.exponential(24, n_trades)  # Average 24 hours
    
    # Generate trade sizes
    trade_sizes = np.random.lognormal(7, 1, n_trades)  # Log-normal distribution
    
    # Calculate return percentage
    return_pct = pnl / trade_sizes * 100
    
    # Create DataFrame
    trades_df = pd.DataFrame({
        'date': trade_dates,
        'pnl': pnl,
        'return_pct': return_pct,
        'hold_time_hours': hold_times,
        'trade_size': trade_sizes,
        'is_win': outcomes,
        'trade_type': np.random.choice(['long', 'short'], n_trades),
        'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN'], n_trades)
    })
    
    return trades_df


def create_heatmap_data(
    years: int = 3,
    start_year: int = 2021,
    mean_return: float = 0.0005,
    volatility: float = 0.02,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Create monthly/yearly returns heatmap data.
    
    Args:
        years: Number of years
        start_year: Starting year
        mean_return: Mean daily return
        volatility: Daily volatility
        seed: Random seed
        
    Returns:
        DataFrame with monthly returns suitable for heatmap
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate daily returns
    start_date = f'{start_year}-01-01'
    end_date = f'{start_year + years - 1}-12-31'
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    daily_returns = np.random.normal(mean_return, volatility, len(dates))
    returns_series = pd.Series(daily_returns, index=dates)
    
    # Calculate monthly returns
    monthly_returns = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Reshape for heatmap (months as columns, years as rows)
    heatmap_data = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values * 100  # Convert to percentage
    })
    
    # Pivot to create heatmap structure
    heatmap_pivot = heatmap_data.pivot(index='year', columns='month', values='return')
    
    # Add month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    heatmap_pivot.columns = month_names
    
    return heatmap_pivot


def create_rolling_metrics_data(
    periods: int = 504,  # 2 years of trading days
    start_date: str = '2022-01-01',
    window_sizes: List[int] = [20, 60, 252],
    seed: Optional[int] = 42
) -> Dict[str, pd.DataFrame]:
    """
    Create rolling metrics data for visualization.
    
    Args:
        periods: Number of periods
        start_date: Start date
        window_sizes: List of rolling window sizes
        seed: Random seed
        
    Returns:
        Dictionary with rolling metrics DataFrames
    """
    if seed is not None:
        np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Generate base returns
    returns = np.random.normal(0.0003, 0.02, periods)
    returns_series = pd.Series(returns, index=dates)
    
    rolling_data = {}
    
    for window in window_sizes:
        # Calculate rolling metrics
        rolling_mean = returns_series.rolling(window).mean() * 252
        rolling_std = returns_series.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_mean / rolling_std).fillna(0)
        
        # Rolling win rate
        rolling_win_rate = (returns_series > 0).rolling(window).mean()
        
        # Rolling max drawdown
        equity = (1 + returns_series).cumprod()
        rolling_dd = equity.rolling(window).apply(
            lambda x: (x[-1] - x.max()) / x.max() if x.max() > 0 else 0
        )
        
        rolling_data[f'window_{window}'] = pd.DataFrame({
            'returns': returns_series,
            'rolling_return': rolling_mean,
            'rolling_volatility': rolling_std,
            'rolling_sharpe': rolling_sharpe,
            'rolling_win_rate': rolling_win_rate,
            'rolling_drawdown': rolling_dd
        })
    
    return rolling_data


def create_multi_series_comparison_data(
    n_series: int = 3,
    periods: int = 252,
    start_date: str = '2023-01-01',
    correlation: float = 0.5,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Create multiple time series for comparison visualization.
    
    Args:
        n_series: Number of series to compare
        periods: Number of periods
        start_date: Start date
        correlation: Average correlation between series
        seed: Random seed
        
    Returns:
        DataFrame with multiple series for comparison
    """
    if seed is not None:
        np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Create correlation matrix
    corr_matrix = np.full((n_series, n_series), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Generate correlated returns
    L = np.linalg.cholesky(corr_matrix)
    independent_returns = np.random.normal(0, 0.02, (periods, n_series))
    correlated_returns = independent_returns @ L.T
    
    # Add different trends to each series
    trends = np.linspace(-0.0002, 0.0005, n_series)
    for i in range(n_series):
        correlated_returns[:, i] += trends[i]
    
    # Create equity curves
    initial_values = 100000 * (1 + np.arange(n_series) * 0.1)
    equity_curves = initial_values * (1 + correlated_returns).cumprod(axis=0)
    
    # Create DataFrame
    series_names = [f'Strategy_{i+1}' for i in range(n_series)]
    comparison_df = pd.DataFrame(equity_curves, index=dates, columns=series_names)
    
    return comparison_df


def create_candlestick_data(
    periods: int = 60,
    start_date: str = '2023-10-01',
    initial_price: float = 100,
    trend: float = 0.001,
    volatility: float = 0.02,
    include_signals: bool = True,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Create OHLCV data for candlestick charts with optional signals.
    
    Args:
        periods: Number of periods
        start_date: Start date
        initial_price: Starting price
        trend: Daily trend
        volatility: Daily volatility
        include_signals: Include buy/sell signals
        seed: Random seed
        
    Returns:
        Dictionary with OHLCV data and signals
    """
    if seed is not None:
        np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Generate price movements
    returns = np.random.normal(trend, volatility, periods)
    close_prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    daily_range = np.abs(np.random.normal(0, volatility, periods))
    
    open_prices = close_prices * (1 + np.random.normal(0, volatility/4, periods))
    high_prices = np.maximum(open_prices, close_prices) + daily_range * close_prices
    low_prices = np.minimum(open_prices, close_prices) - daily_range * close_prices
    
    # Volume with trend
    base_volume = 1000000
    volume_trend = np.linspace(1, 1.5, periods)
    volume = (base_volume * volume_trend * 
              (1 + np.random.exponential(0.5, periods))).astype(int)
    
    ohlcv_df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    # Ensure OHLC relationships
    ohlcv_df['high'] = ohlcv_df[['open', 'high', 'close']].max(axis=1)
    ohlcv_df['low'] = ohlcv_df[['open', 'low', 'close']].min(axis=1)
    
    result = {'ohlcv': ohlcv_df}
    
    # Add signals if requested
    if include_signals:
        # Generate some trading signals
        sma_20 = close_prices.rolling(20).mean()
        sma_50 = close_prices.rolling(50).mean()
        
        # Buy signals when SMA20 crosses above SMA50
        buy_signals = (sma_20 > sma_50) & (sma_20.shift(1) <= sma_50.shift(1))
        sell_signals = (sma_20 < sma_50) & (sma_20.shift(1) >= sma_50.shift(1))
        
        signals_df = pd.DataFrame({
            'buy': buy_signals,
            'sell': sell_signals,
            'buy_price': np.where(buy_signals, close_prices, np.nan),
            'sell_price': np.where(sell_signals, close_prices, np.nan)
        }, index=dates)
        
        result['signals'] = signals_df
        result['indicators'] = pd.DataFrame({
            'sma_20': sma_20,
            'sma_50': sma_50
        }, index=dates)
    
    return result