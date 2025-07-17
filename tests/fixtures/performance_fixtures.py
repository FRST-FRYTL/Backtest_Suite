"""
Performance Fixtures for Testing

Provides comprehensive performance metrics and analysis fixtures.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats


def create_performance_metrics(
    periods: int = 252,
    start_date: str = '2023-01-01',
    strategy_name: str = 'test_strategy',
    benchmark_name: str = 'SPY',
    target_sharpe: float = 1.5,
    target_annual_return: float = 0.15,
    target_max_drawdown: float = -0.10,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Create comprehensive performance metrics.
    
    Args:
        periods: Number of periods
        start_date: Start date
        strategy_name: Name of strategy
        benchmark_name: Name of benchmark
        target_sharpe: Target Sharpe ratio
        target_annual_return: Target annual return
        target_max_drawdown: Target maximum drawdown
        seed: Random seed
        
    Returns:
        Dictionary with performance metrics
    """
    if seed is not None:
        np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Generate returns to match target metrics
    daily_return = target_annual_return / 252
    daily_vol = (target_annual_return / target_sharpe) / np.sqrt(252)
    
    # Strategy returns
    strategy_returns = np.random.normal(daily_return, daily_vol, periods)
    
    # Benchmark returns (slightly lower Sharpe)
    benchmark_returns = np.random.normal(
        daily_return * 0.7, 
        daily_vol * 1.2, 
        periods
    )
    
    # Calculate cumulative returns
    strategy_cum_returns = (1 + strategy_returns).cumprod()
    benchmark_cum_returns = (1 + benchmark_returns).cumprod()
    
    # Calculate drawdowns
    strategy_equity = 100000 * strategy_cum_returns
    strategy_rolling_max = strategy_equity.expanding().max()
    strategy_drawdown = (strategy_equity - strategy_rolling_max) / strategy_rolling_max
    
    benchmark_equity = 100000 * benchmark_cum_returns
    benchmark_rolling_max = benchmark_equity.expanding().max()
    benchmark_drawdown = (benchmark_equity - benchmark_rolling_max) / benchmark_rolling_max
    
    # Adjust if drawdown exceeds target
    if strategy_drawdown.min() < target_max_drawdown:
        scale = target_max_drawdown / strategy_drawdown.min()
        strategy_returns *= scale
        strategy_cum_returns = (1 + strategy_returns).cumprod()
        strategy_equity = 100000 * strategy_cum_returns
        strategy_rolling_max = strategy_equity.expanding().max()
        strategy_drawdown = (strategy_equity - strategy_rolling_max) / strategy_rolling_max
    
    # Calculate comprehensive metrics
    metrics = {
        'strategy_name': strategy_name,
        'benchmark_name': benchmark_name,
        'start_date': start_date,
        'end_date': dates[-1].strftime('%Y-%m-%d'),
        'trading_days': periods,
        
        # Returns
        'total_return': (strategy_cum_returns[-1] - 1) * 100,
        'benchmark_return': (benchmark_cum_returns[-1] - 1) * 100,
        'annual_return': strategy_returns.mean() * 252 * 100,
        'benchmark_annual_return': benchmark_returns.mean() * 252 * 100,
        'excess_return': (strategy_returns.mean() - benchmark_returns.mean()) * 252 * 100,
        
        # Risk metrics
        'volatility': strategy_returns.std() * np.sqrt(252) * 100,
        'benchmark_volatility': benchmark_returns.std() * np.sqrt(252) * 100,
        'downside_volatility': strategy_returns[strategy_returns < 0].std() * np.sqrt(252) * 100,
        'max_drawdown': strategy_drawdown.min() * 100,
        'benchmark_max_drawdown': benchmark_drawdown.min() * 100,
        
        # Risk-adjusted returns
        'sharpe_ratio': (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252),
        'sortino_ratio': (strategy_returns.mean() / 
                         strategy_returns[strategy_returns < 0].std()) * np.sqrt(252),
        'calmar_ratio': (strategy_returns.mean() * 252) / abs(strategy_drawdown.min()),
        'information_ratio': ((strategy_returns - benchmark_returns).mean() / 
                             (strategy_returns - benchmark_returns).std()) * np.sqrt(252),
        
        # Distribution metrics
        'skewness': stats.skew(strategy_returns),
        'kurtosis': stats.kurtosis(strategy_returns),
        'var_95': np.percentile(strategy_returns, 5) * 100,
        'cvar_95': strategy_returns[strategy_returns <= np.percentile(strategy_returns, 5)].mean() * 100,
        
        # Win/loss metrics
        'win_rate': (strategy_returns > 0).mean() * 100,
        'avg_win': strategy_returns[strategy_returns > 0].mean() * 100,
        'avg_loss': strategy_returns[strategy_returns < 0].mean() * 100,
        'profit_factor': (strategy_returns[strategy_returns > 0].sum() / 
                         abs(strategy_returns[strategy_returns < 0].sum())),
        
        # Correlation and beta
        'correlation': np.corrcoef(strategy_returns, benchmark_returns)[0, 1],
        'beta': np.cov(strategy_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns),
        'alpha': (strategy_returns.mean() - 
                 (np.cov(strategy_returns, benchmark_returns)[0, 1] / 
                  np.var(benchmark_returns)) * benchmark_returns.mean()) * 252 * 100,
        
        # Time series
        'returns': pd.Series(strategy_returns, index=dates),
        'benchmark_returns': pd.Series(benchmark_returns, index=dates),
        'equity_curve': pd.Series(strategy_equity, index=dates),
        'benchmark_equity': pd.Series(benchmark_equity, index=dates),
        'drawdown': pd.Series(strategy_drawdown, index=dates)
    }
    
    return metrics


def create_rolling_metrics(
    returns: Optional[pd.Series] = None,
    periods: int = 504,
    start_date: str = '2022-01-01',
    windows: List[int] = [20, 60, 126, 252],
    seed: Optional[int] = 42
) -> Dict[str, pd.DataFrame]:
    """
    Create rolling performance metrics.
    
    Args:
        returns: Return series (if None, generates one)
        periods: Number of periods
        start_date: Start date
        windows: Rolling window sizes
        seed: Random seed
        
    Returns:
        Dictionary with rolling metrics DataFrames
    """
    if seed is not None:
        np.random.seed(seed)
    
    if returns is None:
        dates = pd.date_range(start=start_date, periods=periods, freq='D')
        returns = pd.Series(
            np.random.normal(0.0004, 0.02, periods),
            index=dates,
            name='returns'
        )
    
    rolling_metrics = {}
    
    for window in windows:
        # Annualization factor
        ann_factor = 252 / window
        
        # Rolling returns
        rolling_return = returns.rolling(window).mean() * 252
        
        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling Sharpe
        rolling_sharpe = (returns.rolling(window).mean() / 
                         returns.rolling(window).std()) * np.sqrt(252)
        
        # Rolling Sortino
        def sortino(x):
            if len(x) < 2:
                return np.nan
            downside = x[x < 0]
            if len(downside) < 2:
                return np.nan
            return (x.mean() / downside.std()) * np.sqrt(252)
        
        rolling_sortino = returns.rolling(window).apply(sortino)
        
        # Rolling max drawdown
        def max_dd(x):
            cum_returns = (1 + x).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdown = (cum_returns - rolling_max) / rolling_max
            return drawdown.min()
        
        rolling_max_dd = returns.rolling(window).apply(max_dd)
        
        # Rolling win rate
        rolling_win_rate = (returns > 0).rolling(window).mean()
        
        # Rolling beta (to a synthetic market)
        market_returns = returns + np.random.normal(0, 0.005, len(returns))
        rolling_beta = returns.rolling(window).corr(market_returns)
        
        # Compile metrics
        metrics_df = pd.DataFrame({
            'returns': returns,
            'rolling_return': rolling_return,
            'rolling_volatility': rolling_vol,
            'rolling_sharpe': rolling_sharpe,
            'rolling_sortino': rolling_sortino,
            'rolling_max_drawdown': rolling_max_dd,
            'rolling_win_rate': rolling_win_rate,
            'rolling_beta': rolling_beta
        })
        
        rolling_metrics[f'window_{window}'] = metrics_df
    
    return rolling_metrics


def create_sharpe_data(
    n_strategies: int = 5,
    periods: int = 252,
    start_date: str = '2023-01-01',
    sharpe_range: Tuple[float, float] = (0.5, 2.5),
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Create data for Sharpe ratio analysis.
    
    Args:
        n_strategies: Number of strategies
        periods: Number of periods
        start_date: Start date
        sharpe_range: Range of Sharpe ratios
        seed: Random seed
        
    Returns:
        DataFrame with strategy returns and Sharpe ratios
    """
    if seed is not None:
        np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Generate strategies with different Sharpe ratios
    sharpe_ratios = np.linspace(sharpe_range[0], sharpe_range[1], n_strategies)
    
    strategies_data = {}
    metrics_list = []
    
    for i, target_sharpe in enumerate(sharpe_ratios):
        strategy_name = f'Strategy_{i+1}'
        
        # Calculate required return and volatility
        annual_vol = 0.15  # 15% volatility
        annual_return = target_sharpe * annual_vol
        
        daily_return = annual_return / 252
        daily_vol = annual_vol / np.sqrt(252)
        
        # Generate returns
        returns = np.random.normal(daily_return, daily_vol, periods)
        
        # Adjust to match target Sharpe
        current_sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        adjustment = target_sharpe / current_sharpe
        returns = returns * adjustment + daily_return * (1 - adjustment)
        
        strategies_data[strategy_name] = returns
        
        # Calculate actual metrics
        actual_sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        actual_return = returns.mean() * 252
        actual_vol = returns.std() * np.sqrt(252)
        
        metrics_list.append({
            'strategy': strategy_name,
            'target_sharpe': target_sharpe,
            'actual_sharpe': actual_sharpe,
            'annual_return': actual_return * 100,
            'annual_volatility': actual_vol * 100,
            'max_drawdown': calculate_max_drawdown(returns) * 100,
            'win_rate': (returns > 0).mean() * 100
        })
    
    # Create returns DataFrame
    returns_df = pd.DataFrame(strategies_data, index=dates)
    
    # Add metrics as attribute
    returns_df.attrs['metrics'] = pd.DataFrame(metrics_list)
    
    return returns_df


def create_returns_distribution(
    distribution_type: str = 'normal',
    periods: int = 1000,
    mean_return: float = 0.0004,
    volatility: float = 0.02,
    skew: float = -0.5,
    excess_kurtosis: float = 3.0,
    seed: Optional[int] = 42
) -> pd.Series:
    """
    Create returns with specific distribution characteristics.
    
    Args:
        distribution_type: 'normal', 'skewed', 'fat_tails', or 'mixed'
        periods: Number of periods
        mean_return: Mean return
        volatility: Return volatility
        skew: Target skewness
        excess_kurtosis: Target excess kurtosis
        seed: Random seed
        
    Returns:
        Series of returns with specified distribution
    """
    if seed is not None:
        np.random.seed(seed)
    
    dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')
    
    if distribution_type == 'normal':
        # Standard normal distribution
        returns = np.random.normal(mean_return, volatility, periods)
    
    elif distribution_type == 'skewed':
        # Skewed distribution using mixture
        # Negative skew: more extreme negative returns
        normal_component = np.random.normal(mean_return, volatility * 0.8, periods)
        extreme_component = np.random.normal(mean_return - volatility * 2, volatility * 1.5, periods)
        
        # Mix components
        mix_prob = 0.05  # 5% extreme events
        mask = np.random.random(periods) < mix_prob
        returns = np.where(mask, extreme_component, normal_component)
    
    elif distribution_type == 'fat_tails':
        # Fat-tailed distribution using Student's t
        df = 4  # degrees of freedom for fat tails
        returns = stats.t.rvs(df, loc=mean_return, scale=volatility, size=periods)
    
    elif distribution_type == 'mixed':
        # Realistic mixed distribution
        # Normal market (85%)
        # Volatile market (10%)
        # Crisis (5%)
        
        regimes = np.random.choice(['normal', 'volatile', 'crisis'], 
                                  size=periods, 
                                  p=[0.85, 0.10, 0.05])
        
        returns = np.zeros(periods)
        
        returns[regimes == 'normal'] = np.random.normal(
            mean_return, volatility * 0.7, 
            (regimes == 'normal').sum()
        )
        
        returns[regimes == 'volatile'] = np.random.normal(
            mean_return * 0.5, volatility * 2, 
            (regimes == 'volatile').sum()
        )
        
        returns[regimes == 'crisis'] = np.random.normal(
            mean_return * -5, volatility * 4, 
            (regimes == 'crisis').sum()
        )
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    return pd.Series(returns, index=dates, name='returns')


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Helper function to calculate maximum drawdown."""
    cum_returns = (1 + returns).cumprod()
    rolling_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - rolling_max) / rolling_max
    return drawdown.min()


def create_factor_performance_data(
    factors: List[str] = ['momentum', 'value', 'size', 'quality', 'low_vol'],
    periods: int = 1260,  # 5 years
    start_date: str = '2019-01-01',
    include_regime_changes: bool = True,
    seed: Optional[int] = 42
) -> Dict[str, pd.DataFrame]:
    """
    Create factor performance data for analysis.
    
    Args:
        factors: List of factor names
        periods: Number of periods
        start_date: Start date
        include_regime_changes: Include regime changes in performance
        seed: Random seed
        
    Returns:
        Dictionary with factor performance data
    """
    if seed is not None:
        np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Define factor characteristics
    factor_chars = {
        'momentum': {'mean': 0.0006, 'vol': 0.018, 'regime_sensitivity': 0.8},
        'value': {'mean': 0.0004, 'vol': 0.015, 'regime_sensitivity': -0.5},
        'size': {'mean': 0.0002, 'vol': 0.020, 'regime_sensitivity': -0.3},
        'quality': {'mean': 0.0005, 'vol': 0.012, 'regime_sensitivity': 0.2},
        'low_vol': {'mean': 0.0003, 'vol': 0.008, 'regime_sensitivity': -0.7}
    }
    
    # Generate market regimes if requested
    if include_regime_changes:
        # Define regimes: bull, bear, high_vol, low_vol
        regime_lengths = [100, 150, 200, 250]
        regimes = []
        
        current_pos = 0
        while current_pos < periods:
            regime_type = np.random.choice(['bull', 'bear', 'high_vol', 'low_vol'])
            regime_length = np.random.choice(regime_lengths)
            regimes.extend([regime_type] * min(regime_length, periods - current_pos))
            current_pos += regime_length
        
        regimes = regimes[:periods]
        
        # Regime effects on factors
        regime_effects = {
            'bull': {'momentum': 1.5, 'value': 0.5, 'size': 1.2, 'quality': 1.1, 'low_vol': 0.7},
            'bear': {'momentum': 0.3, 'value': 1.5, 'size': 0.5, 'quality': 1.3, 'low_vol': 1.5},
            'high_vol': {'momentum': 0.8, 'value': 1.1, 'size': 0.6, 'quality': 1.2, 'low_vol': 1.8},
            'low_vol': {'momentum': 1.2, 'value': 0.9, 'size': 1.1, 'quality': 1.0, 'low_vol': 0.5}
        }
    else:
        regimes = ['normal'] * periods
        regime_effects = {'normal': {f: 1.0 for f in factors}}
    
    # Generate factor returns
    factor_returns = {}
    
    for factor in factors:
        if factor in factor_chars:
            char = factor_chars[factor]
            base_returns = np.random.normal(char['mean'], char['vol'], periods)
            
            # Apply regime effects
            adjusted_returns = base_returns.copy()
            for i, regime in enumerate(regimes):
                if regime in regime_effects and factor in regime_effects[regime]:
                    adjusted_returns[i] *= regime_effects[regime][factor]
            
            factor_returns[factor] = adjusted_returns
    
    # Create correlation between factors
    correlation_matrix = np.array([
        [1.0, -0.3, -0.2, 0.4, -0.5],   # momentum
        [-0.3, 1.0, 0.3, 0.2, 0.4],     # value
        [-0.2, 0.3, 1.0, -0.1, 0.2],    # size
        [0.4, 0.2, -0.1, 1.0, 0.3],     # quality
        [-0.5, 0.4, 0.2, 0.3, 1.0]      # low_vol
    ])
    
    # Apply correlation structure
    L = np.linalg.cholesky(correlation_matrix[:len(factors), :len(factors)])
    independent_returns = np.array(list(factor_returns.values()))
    correlated_returns = L @ independent_returns
    
    # Create DataFrames
    factor_df = pd.DataFrame(
        correlated_returns.T,
        index=dates,
        columns=factors
    )
    
    # Add regime information
    regime_df = pd.DataFrame({
        'regime': regimes,
        'regime_code': pd.Categorical(regimes).codes
    }, index=dates)
    
    # Calculate factor statistics
    stats_data = []
    for factor in factors:
        stats_data.append({
            'factor': factor,
            'annual_return': factor_df[factor].mean() * 252 * 100,
            'annual_volatility': factor_df[factor].std() * np.sqrt(252) * 100,
            'sharpe_ratio': (factor_df[factor].mean() / factor_df[factor].std()) * np.sqrt(252),
            'max_drawdown': calculate_max_drawdown(factor_df[factor].values) * 100,
            'skewness': stats.skew(factor_df[factor]),
            'kurtosis': stats.kurtosis(factor_df[factor])
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    return {
        'returns': factor_df,
        'regimes': regime_df,
        'statistics': stats_df,
        'correlation': pd.DataFrame(
            np.corrcoef(factor_df.T),
            index=factors,
            columns=factors
        )
    }