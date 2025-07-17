"""
Analysis Fixtures for Testing

Provides comprehensive fixtures for analysis modules including
strategy performance, benchmarks, and portfolio data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


def create_strategy_performance_data(
    periods: int = 252,
    start_date: str = '2023-01-01',
    initial_capital: float = 100000,
    annual_return: float = 0.15,
    annual_volatility: float = 0.20,
    max_drawdown: float = -0.15,
    sharpe_ratio: float = 0.75,
    seed: Optional[int] = 42
) -> Dict[str, pd.DataFrame]:
    """
    Create comprehensive strategy performance data.
    
    Args:
        periods: Number of trading days
        start_date: Start date
        initial_capital: Starting capital
        annual_return: Target annual return
        annual_volatility: Target annual volatility
        max_drawdown: Maximum drawdown (negative)
        sharpe_ratio: Target Sharpe ratio
        seed: Random seed
        
    Returns:
        Dictionary with performance DataFrames
    """
    if seed is not None:
        np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Generate daily returns
    daily_return = annual_return / 252
    daily_volatility = annual_volatility / np.sqrt(252)
    
    returns = np.random.normal(daily_return, daily_volatility, periods)
    
    # Adjust returns to achieve target metrics
    returns = returns * (sharpe_ratio * daily_volatility / returns.std())
    returns = returns + (daily_return - returns.mean())
    
    # Create equity curve
    equity_curve = initial_capital * (1 + returns).cumprod()
    
    # Calculate drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    
    # Adjust for max drawdown constraint
    if drawdown.min() < max_drawdown:
        scale_factor = max_drawdown / drawdown.min()
        returns = returns * scale_factor
        equity_curve = initial_capital * (1 + returns).cumprod()
        drawdown = (equity_curve - equity_curve.expanding().max()) / equity_curve.expanding().max()
    
    # Create DataFrames
    performance_data = {
        'returns': pd.DataFrame({
            'strategy_returns': returns,
            'cumulative_returns': (1 + returns).cumprod() - 1,
            'equity': equity_curve,
            'drawdown': drawdown
        }, index=dates),
        
        'metrics': pd.DataFrame({
            'annual_return': [returns.mean() * 252],
            'annual_volatility': [returns.std() * np.sqrt(252)],
            'sharpe_ratio': [(returns.mean() / returns.std()) * np.sqrt(252)],
            'max_drawdown': [drawdown.min()],
            'calmar_ratio': [(returns.mean() * 252) / abs(drawdown.min())],
            'win_rate': [(returns > 0).mean()],
            'profit_factor': [returns[returns > 0].sum() / abs(returns[returns < 0].sum())],
            'recovery_factor': [(equity_curve.iloc[-1] - initial_capital) / (initial_capital * abs(drawdown.min()))]
        }),
        
        'monthly_returns': returns.resample('M').apply(lambda x: (1 + x).prod() - 1),
        'yearly_returns': returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    }
    
    return performance_data


def create_benchmark_data(
    periods: int = 252,
    start_date: str = '2023-01-01',
    benchmark_type: str = 'market',
    annual_return: float = 0.10,
    annual_volatility: float = 0.15,
    correlation_to_strategy: float = 0.6,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Create benchmark comparison data.
    
    Args:
        periods: Number of periods
        start_date: Start date
        benchmark_type: 'market', 'sector', or 'risk_free'
        annual_return: Benchmark annual return
        annual_volatility: Benchmark volatility
        correlation_to_strategy: Correlation with strategy
        seed: Random seed
        
    Returns:
        DataFrame with benchmark data
    """
    if seed is not None:
        np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    if benchmark_type == 'risk_free':
        # Constant risk-free rate
        daily_rate = annual_return / 252
        returns = np.full(periods, daily_rate)
        prices = 100 * (1 + returns).cumprod()
    else:
        # Market or sector benchmark
        daily_return = annual_return / 252
        daily_volatility = annual_volatility / np.sqrt(252)
        
        # Generate correlated returns if needed
        if correlation_to_strategy > 0:
            # This would be correlated with strategy returns
            strategy_returns = np.random.normal(0.0005, 0.02, periods)
            independent_returns = np.random.normal(0, 1, periods)
            
            # Create correlated returns
            returns = (correlation_to_strategy * strategy_returns + 
                      np.sqrt(1 - correlation_to_strategy**2) * independent_returns * daily_volatility +
                      daily_return)
        else:
            returns = np.random.normal(daily_return, daily_volatility, periods)
        
        prices = 100 * (1 + returns).cumprod()
    
    benchmark_df = pd.DataFrame({
        'price': prices,
        'returns': returns,
        'cumulative_returns': (1 + returns).cumprod() - 1
    }, index=dates)
    
    return benchmark_df


def create_portfolio_data(
    n_assets: int = 5,
    periods: int = 252,
    start_date: str = '2023-01-01',
    rebalance_frequency: str = 'monthly',
    initial_weights: Optional[List[float]] = None,
    seed: Optional[int] = 42
) -> Dict[str, pd.DataFrame]:
    """
    Create portfolio allocation and performance data.
    
    Args:
        n_assets: Number of assets in portfolio
        periods: Number of periods
        start_date: Start date
        rebalance_frequency: 'daily', 'weekly', 'monthly'
        initial_weights: Initial portfolio weights
        seed: Random seed
        
    Returns:
        Dictionary with portfolio DataFrames
    """
    if seed is not None:
        np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Generate asset names
    assets = [f'Asset_{i+1}' for i in range(n_assets)]
    
    # Initial weights
    if initial_weights is None:
        initial_weights = np.random.dirichlet(np.ones(n_assets))
    
    # Generate asset returns
    mean_returns = np.random.uniform(-0.0002, 0.0008, n_assets)
    volatilities = np.random.uniform(0.01, 0.03, n_assets)
    
    returns_data = pd.DataFrame(
        np.random.normal(mean_returns, volatilities, (periods, n_assets)),
        index=dates,
        columns=assets
    )
    
    # Create weight evolution
    weights_data = pd.DataFrame(index=dates, columns=assets)
    weights_data.iloc[0] = initial_weights
    
    # Rebalancing logic
    if rebalance_frequency == 'daily':
        rebalance_mask = np.ones(periods, dtype=bool)
    elif rebalance_frequency == 'weekly':
        rebalance_mask = dates.weekday == 0  # Monday
    elif rebalance_frequency == 'monthly':
        rebalance_mask = dates.day == 1
    else:
        rebalance_mask = np.zeros(periods, dtype=bool)
        rebalance_mask[0] = True
    
    current_weights = initial_weights.copy()
    portfolio_value = 100000
    values = [portfolio_value]
    
    for i in range(1, periods):
        # Update weights based on returns
        asset_returns = returns_data.iloc[i].values
        current_weights = current_weights * (1 + asset_returns)
        current_weights = current_weights / current_weights.sum()
        
        # Rebalance if scheduled
        if rebalance_mask[i]:
            current_weights = initial_weights.copy()
        
        weights_data.iloc[i] = current_weights
        
        # Calculate portfolio return
        portfolio_return = np.sum(current_weights * asset_returns)
        portfolio_value *= (1 + portfolio_return)
        values.append(portfolio_value)
    
    # Create portfolio summary
    portfolio_data = {
        'weights': weights_data,
        'asset_returns': returns_data,
        'portfolio_value': pd.Series(values, index=dates),
        'positions': pd.DataFrame({
            asset: weights_data[asset] * pd.Series(values, index=dates)
            for asset in assets
        })
    }
    
    return portfolio_data


def create_market_data(
    sectors: List[str] = ['Technology', 'Finance', 'Healthcare', 'Energy', 'Consumer'],
    periods: int = 252,
    start_date: str = '2023-01-01',
    include_factors: bool = True,
    seed: Optional[int] = 42
) -> Dict[str, pd.DataFrame]:
    """
    Create comprehensive market data including sectors and factors.
    
    Args:
        sectors: List of sector names
        periods: Number of periods
        start_date: Start date
        include_factors: Include factor data (momentum, value, etc.)
        seed: Random seed
        
    Returns:
        Dictionary with market DataFrames
    """
    if seed is not None:
        np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Market index
    market_return = 0.10 / 252  # 10% annual
    market_vol = 0.15 / np.sqrt(252)
    market_returns = np.random.normal(market_return, market_vol, periods)
    market_index = 100 * (1 + market_returns).cumprod()
    
    # Sector indices
    sector_data = {}
    for i, sector in enumerate(sectors):
        # Each sector has different characteristics
        sector_beta = 0.7 + 0.1 * i
        sector_alpha = (-0.02 + 0.01 * i) / 252
        
        sector_returns = (sector_beta * market_returns + 
                         sector_alpha + 
                         np.random.normal(0, market_vol * 0.5, periods))
        
        sector_data[sector] = 100 * (1 + sector_returns).cumprod()
    
    sector_df = pd.DataFrame(sector_data, index=dates)
    
    market_data = {
        'market_index': pd.DataFrame({
            'index_value': market_index,
            'returns': market_returns
        }, index=dates),
        'sectors': sector_df
    }
    
    # Add factor data if requested
    if include_factors:
        factors = {
            'momentum': np.random.normal(0.0002, 0.01, periods),
            'value': np.random.normal(0.0001, 0.008, periods),
            'size': np.random.normal(-0.0001, 0.012, periods),
            'quality': np.random.normal(0.00015, 0.007, periods),
            'volatility': np.random.normal(-0.0002, 0.015, periods)
        }
        
        market_data['factors'] = pd.DataFrame(factors, index=dates)
    
    return market_data


def create_risk_metrics_data(
    periods: int = 252,
    start_date: str = '2023-01-01',
    include_var: bool = True,
    include_stress_tests: bool = True,
    seed: Optional[int] = 42
) -> Dict[str, pd.DataFrame]:
    """
    Create risk metrics and analysis data.
    
    Args:
        periods: Number of periods
        start_date: Start date
        include_var: Include Value at Risk calculations
        include_stress_tests: Include stress test scenarios
        seed: Random seed
        
    Returns:
        Dictionary with risk metrics DataFrames
    """
    if seed is not None:
        np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Generate returns for risk calculations
    returns = np.random.normal(0.0003, 0.02, periods)
    
    # Rolling volatility
    rolling_vol = pd.Series(returns, index=dates).rolling(20).std() * np.sqrt(252)
    
    # Rolling beta (to market)
    market_returns = np.random.normal(0.0004, 0.015, periods)
    rolling_beta = pd.Series(returns, index=dates).rolling(60).corr(
        pd.Series(market_returns, index=dates)
    )
    
    risk_data = {
        'volatility': pd.DataFrame({
            'realized_vol': rolling_vol,
            'returns': returns
        }),
        'beta': pd.DataFrame({
            'beta': rolling_beta,
            'market_returns': market_returns
        })
    }
    
    # Add VaR if requested
    if include_var:
        # Historical VaR
        var_95 = pd.Series(returns, index=dates).rolling(100).quantile(0.05)
        var_99 = pd.Series(returns, index=dates).rolling(100).quantile(0.01)
        
        # CVaR (Conditional VaR)
        cvar_95 = pd.Series(returns, index=dates).rolling(100).apply(
            lambda x: x[x <= x.quantile(0.05)].mean()
        )
        
        risk_data['var'] = pd.DataFrame({
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95
        })
    
    # Add stress tests if requested
    if include_stress_tests:
        scenarios = {
            'market_crash': returns * 3,  # 3x volatility
            'flash_crash': returns - 0.05,  # 5% sudden drop
            'volatility_spike': returns * np.random.uniform(1, 5, periods),
            'correlation_breakdown': returns * np.random.choice([-1, 1], periods)
        }
        
        risk_data['stress_tests'] = pd.DataFrame(scenarios, index=dates)
    
    return risk_data