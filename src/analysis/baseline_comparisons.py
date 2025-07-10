"""
Baseline and Benchmark Comparison System

This module implements comprehensive baseline comparisons including
buy-and-hold, benchmark portfolios, and statistical analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BaselineResults:
    """Results from baseline strategy analysis"""
    strategy_name: str
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    var_95: float
    cvar_95: float
    total_trades: int
    total_contributions: float
    dividend_income: float
    transaction_costs: float
    equity_curve: pd.Series
    monthly_returns: pd.Series
    drawdown_series: pd.Series

class BaselineComparison:
    """
    Comprehensive baseline comparison system for strategy evaluation.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize baseline comparison system.
        
        Args:
            risk_free_rate: Annual risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate
        self.benchmark_data = {}
        
    def create_buy_hold_baseline(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 10000,
        monthly_contribution: float = 500,
        transaction_cost: float = 0.001
    ) -> BaselineResults:
        """
        Create comprehensive buy-and-hold baseline.
        
        Args:
            symbol: Symbol to buy and hold
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Initial investment amount
            monthly_contribution: Monthly contribution amount
            transaction_cost: Transaction cost as percentage
            
        Returns:
            BaselineResults with buy-and-hold performance
        """
        logger.info(f"Creating buy-and-hold baseline for {symbol}")
        
        # Download data
        data = self._download_data(symbol, start_date, end_date)
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Calculate buy-and-hold performance
        equity_curve = pd.Series(index=data.index, dtype=float)
        shares_held = 0
        cash = initial_capital
        total_contributions = initial_capital
        total_costs = 0
        
        # Initial purchase
        first_price = data.iloc[0]['close']
        shares_to_buy = cash / first_price
        transaction_cost_amount = shares_to_buy * first_price * transaction_cost
        shares_held = shares_to_buy
        cash = 0
        total_costs += transaction_cost_amount
        
        # Monthly contributions
        monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        for i, date in enumerate(data.index):
            # Check if this is a monthly contribution date
            # Convert to datetime objects for comparison
            date_dt = pd.Timestamp(date).tz_localize(None) if pd.Timestamp(date).tz else pd.Timestamp(date)
            monthly_dates_dt = [pd.Timestamp(md).tz_localize(None) if pd.Timestamp(md).tz else pd.Timestamp(md) for md in monthly_dates]
            
            if any(abs((date_dt - md).days) <= 5 for md in monthly_dates_dt) and i > 0:
                # Monthly contribution
                price = data.loc[date, 'close']
                additional_shares = monthly_contribution / price
                transaction_cost_amount = monthly_contribution * transaction_cost
                shares_held += additional_shares
                total_contributions += monthly_contribution
                total_costs += transaction_cost_amount
            
            # Update equity curve
            current_price = data.loc[date, 'close']
            portfolio_value = shares_held * current_price
            equity_curve.loc[date] = portfolio_value
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        monthly_returns = equity_curve.resample('M').last().pct_change().dropna()
        
        # Calculate metrics
        total_return = (equity_curve.iloc[-1] / total_contributions - 1) * 100
        annual_return = self._calculate_annual_return(equity_curve)
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        sortino_ratio = self._calculate_sortino_ratio(returns)
        var_95 = returns.quantile(0.05) * 100
        cvar_95 = returns[returns <= returns.quantile(0.05)].mean() * 100
        
        # Calculate drawdown series
        drawdown_series = self._calculate_drawdown_series(equity_curve)
        
        return BaselineResults(
            strategy_name=f"Buy-and-Hold {symbol}",
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            total_trades=len(monthly_dates),  # Number of purchases
            total_contributions=total_contributions,
            dividend_income=0,  # Could be enhanced to include dividends
            transaction_costs=total_costs,
            equity_curve=equity_curve,
            monthly_returns=monthly_returns,
            drawdown_series=drawdown_series
        )
    
    def create_equal_weight_portfolio(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 10000,
        monthly_contribution: float = 500,
        rebalance_frequency: str = 'M'
    ) -> BaselineResults:
        """
        Create equal-weight portfolio baseline.
        
        Args:
            symbols: List of symbols for portfolio
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital
            monthly_contribution: Monthly contribution
            rebalance_frequency: Rebalancing frequency ('M', 'Q', 'A')
            
        Returns:
            BaselineResults for equal-weight portfolio
        """
        logger.info(f"Creating equal-weight portfolio with {len(symbols)} symbols")
        
        # Download data for all symbols
        portfolio_data = {}
        for symbol in symbols:
            data = self._download_data(symbol, start_date, end_date)
            if not data.empty:
                portfolio_data[symbol] = data['close']
        
        if not portfolio_data:
            raise ValueError("No data available for any symbols")
        
        # Align all data to common date range
        combined_data = pd.DataFrame(portfolio_data)
        combined_data = combined_data.dropna()
        
        # Calculate equal-weight portfolio returns
        equal_weights = 1.0 / len(combined_data.columns)
        daily_returns = combined_data.pct_change().dropna()
        portfolio_returns = (daily_returns * equal_weights).sum(axis=1)
        
        # Rebalancing logic
        if rebalance_frequency == 'M':
            rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        elif rebalance_frequency == 'Q':
            rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='QS')
        else:
            rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='AS')
        
        # Calculate equity curve with rebalancing
        equity_curve = pd.Series(index=combined_data.index, dtype=float)
        current_value = initial_capital
        total_contributions = initial_capital
        
        for i, date in enumerate(combined_data.index):
            if i == 0:
                equity_curve.loc[date] = current_value
            else:
                # Apply daily return
                daily_return = portfolio_returns.loc[date]
                current_value *= (1 + daily_return)
                
                # Check for monthly contribution
                if any(abs((date - rd).days) <= 5 for rd in rebalance_dates) and i > 0:
                    current_value += monthly_contribution
                    total_contributions += monthly_contribution
                
                equity_curve.loc[date] = current_value
        
        # Calculate metrics
        returns = equity_curve.pct_change().dropna()
        monthly_returns = equity_curve.resample('M').last().pct_change().dropna()
        
        total_return = (equity_curve.iloc[-1] / total_contributions - 1) * 100
        annual_return = self._calculate_annual_return(equity_curve)
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        sortino_ratio = self._calculate_sortino_ratio(returns)
        var_95 = returns.quantile(0.05) * 100
        cvar_95 = returns[returns <= returns.quantile(0.05)].mean() * 100
        
        drawdown_series = self._calculate_drawdown_series(equity_curve)
        
        return BaselineResults(
            strategy_name=f"Equal-Weight Portfolio ({len(symbols)} assets)",
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            total_trades=len(rebalance_dates) * len(symbols),
            total_contributions=total_contributions,
            dividend_income=0,
            transaction_costs=0,
            equity_curve=equity_curve,
            monthly_returns=monthly_returns,
            drawdown_series=drawdown_series
        )
    
    def create_60_40_portfolio(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 10000,
        monthly_contribution: float = 500,
        stock_etf: str = 'SPY',
        bond_etf: str = 'TLT',
        alternative_etf: str = 'GLD',
        alternative_weight: float = 0.1
    ) -> BaselineResults:
        """
        Create 60/40 portfolio with alternatives.
        
        Args:
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital
            monthly_contribution: Monthly contribution
            stock_etf: Stock ETF symbol
            bond_etf: Bond ETF symbol
            alternative_etf: Alternative ETF symbol
            alternative_weight: Weight for alternatives
            
        Returns:
            BaselineResults for 60/40 portfolio
        """
        logger.info("Creating 60/40 portfolio with alternatives")
        
        # Download data
        stock_data = self._download_data(stock_etf, start_date, end_date)['close']
        bond_data = self._download_data(bond_etf, start_date, end_date)['close']
        alt_data = self._download_data(alternative_etf, start_date, end_date)['close']
        
        # Combine data
        portfolio_data = pd.DataFrame({
            'stocks': stock_data,
            'bonds': bond_data,
            'alternatives': alt_data
        }).dropna()
        
        # Calculate returns
        returns = portfolio_data.pct_change().dropna()
        
        # Portfolio weights
        stock_weight = 0.6 - alternative_weight
        bond_weight = 0.4 - alternative_weight
        
        # Calculate portfolio returns
        portfolio_returns = (
            returns['stocks'] * stock_weight +
            returns['bonds'] * bond_weight +
            returns['alternatives'] * alternative_weight
        )
        
        # Calculate equity curve
        equity_curve = pd.Series(index=returns.index, dtype=float)
        current_value = initial_capital
        total_contributions = initial_capital
        
        monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        for i, date in enumerate(returns.index):
            if i == 0:
                equity_curve.loc[date] = current_value
            else:
                # Apply daily return
                daily_return = portfolio_returns.loc[date]
                current_value *= (1 + daily_return)
                
                # Check for monthly contribution
                if any(abs((date - md).days) <= 5 for md in monthly_dates):
                    current_value += monthly_contribution
                    total_contributions += monthly_contribution
                
                equity_curve.loc[date] = current_value
        
        # Calculate metrics
        monthly_returns = equity_curve.resample('M').last().pct_change().dropna()
        
        total_return = (equity_curve.iloc[-1] / total_contributions - 1) * 100
        annual_return = self._calculate_annual_return(equity_curve)
        volatility = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        var_95 = portfolio_returns.quantile(0.05) * 100
        cvar_95 = portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.05)].mean() * 100
        
        drawdown_series = self._calculate_drawdown_series(equity_curve)
        
        return BaselineResults(
            strategy_name=f"60/40 Portfolio ({stock_etf}/{bond_etf}) + {alternative_weight*100:.0f}% {alternative_etf}",
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            total_trades=len(monthly_dates) * 3,  # 3 assets
            total_contributions=total_contributions,
            dividend_income=0,
            transaction_costs=0,
            equity_curve=equity_curve,
            monthly_returns=monthly_returns,
            drawdown_series=drawdown_series
        )
    
    def compare_strategies(
        self,
        strategy_results: BaselineResults,
        baseline_results: List[BaselineResults]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare strategy performance against baselines.
        
        Args:
            strategy_results: Strategy results to compare
            baseline_results: List of baseline results
            
        Returns:
            Comparison metrics dictionary
        """
        comparisons = {}
        
        for baseline in baseline_results:
            comparison = {
                'alpha_total_return': strategy_results.total_return - baseline.total_return,
                'alpha_annual_return': strategy_results.annual_return - baseline.annual_return,
                'sharpe_ratio_diff': strategy_results.sharpe_ratio - baseline.sharpe_ratio,
                'max_drawdown_diff': strategy_results.max_drawdown - baseline.max_drawdown,
                'volatility_diff': strategy_results.volatility - baseline.volatility,
                'calmar_ratio_diff': strategy_results.calmar_ratio - baseline.calmar_ratio,
                'sortino_ratio_diff': strategy_results.sortino_ratio - baseline.sortino_ratio,
                'information_ratio': self._calculate_information_ratio(
                    strategy_results.equity_curve, baseline.equity_curve
                ),
                'tracking_error': self._calculate_tracking_error(
                    strategy_results.equity_curve, baseline.equity_curve
                ),
                'up_capture': self._calculate_up_capture(
                    strategy_results.equity_curve, baseline.equity_curve
                ),
                'down_capture': self._calculate_down_capture(
                    strategy_results.equity_curve, baseline.equity_curve
                ),
                'beta': self._calculate_beta(
                    strategy_results.equity_curve, baseline.equity_curve
                )
            }
            
            comparisons[baseline.strategy_name] = comparison
        
        return comparisons
    
    def _download_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download and cache data for a symbol."""
        cache_key = f"{symbol}_{start_date}_{end_date}"
        
        if cache_key in self.benchmark_data:
            return self.benchmark_data[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            data.columns = [col.lower() for col in data.columns]
            
            self.benchmark_data[cache_key] = data
            return data
        except Exception as e:
            logger.error(f"Failed to download data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_annual_return(self, equity_curve: pd.Series) -> float:
        """Calculate annualized return."""
        if len(equity_curve) < 2:
            return 0.0
        
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        
        if years <= 0:
            return 0.0
        
        annual_return = (1 + total_return) ** (1 / years) - 1
        return annual_return * 100
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - (self.risk_free_rate / 252)
        return (excess_returns / returns.std()) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve / rolling_max - 1) * 100
        return drawdown.min()
    
    def _calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        rolling_max = equity_curve.expanding().max()
        drawdown_series = (equity_curve / rolling_max - 1) * 100
        return drawdown_series
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - (self.risk_free_rate / 252)
        return (excess_returns / downside_returns.std()) * np.sqrt(252)
    
    def _calculate_information_ratio(self, strategy_curve: pd.Series, benchmark_curve: pd.Series) -> float:
        """Calculate information ratio."""
        strategy_returns = strategy_curve.pct_change().dropna()
        benchmark_returns = benchmark_curve.pct_change().dropna()
        
        # Align returns
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) < 2:
            return 0.0
        
        strategy_returns = strategy_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        excess_returns = strategy_returns - benchmark_returns
        
        if excess_returns.std() == 0:
            return 0.0
        
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    
    def _calculate_tracking_error(self, strategy_curve: pd.Series, benchmark_curve: pd.Series) -> float:
        """Calculate tracking error."""
        strategy_returns = strategy_curve.pct_change().dropna()
        benchmark_returns = benchmark_curve.pct_change().dropna()
        
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) < 2:
            return 0.0
        
        strategy_returns = strategy_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        excess_returns = strategy_returns - benchmark_returns
        return excess_returns.std() * np.sqrt(252) * 100
    
    def _calculate_up_capture(self, strategy_curve: pd.Series, benchmark_curve: pd.Series) -> float:
        """Calculate up capture ratio."""
        strategy_returns = strategy_curve.pct_change().dropna()
        benchmark_returns = benchmark_curve.pct_change().dropna()
        
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) < 2:
            return 0.0
        
        strategy_returns = strategy_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        up_periods = benchmark_returns > 0
        
        if not up_periods.any():
            return 0.0
        
        strategy_up_return = strategy_returns[up_periods].mean()
        benchmark_up_return = benchmark_returns[up_periods].mean()
        
        if benchmark_up_return == 0:
            return 0.0
        
        return (strategy_up_return / benchmark_up_return) * 100
    
    def _calculate_down_capture(self, strategy_curve: pd.Series, benchmark_curve: pd.Series) -> float:
        """Calculate down capture ratio."""
        strategy_returns = strategy_curve.pct_change().dropna()
        benchmark_returns = benchmark_curve.pct_change().dropna()
        
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) < 2:
            return 0.0
        
        strategy_returns = strategy_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        down_periods = benchmark_returns < 0
        
        if not down_periods.any():
            return 0.0
        
        strategy_down_return = strategy_returns[down_periods].mean()
        benchmark_down_return = benchmark_returns[down_periods].mean()
        
        if benchmark_down_return == 0:
            return 0.0
        
        return (strategy_down_return / benchmark_down_return) * 100
    
    def _calculate_beta(self, strategy_curve: pd.Series, benchmark_curve: pd.Series) -> float:
        """Calculate beta."""
        strategy_returns = strategy_curve.pct_change().dropna()
        benchmark_returns = benchmark_curve.pct_change().dropna()
        
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) < 2:
            return 0.0
        
        strategy_returns = strategy_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            return 0.0
        
        return covariance / benchmark_variance