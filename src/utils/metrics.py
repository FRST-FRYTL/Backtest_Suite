"""Performance metrics calculator for backtesting results."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class PerformanceMetrics:
    """Calculate and store performance metrics."""
    
    # Basic metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    # Trade metrics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Risk-adjusted metrics
    information_ratio: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    
    @classmethod
    def calculate(
        cls,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> 'PerformanceMetrics':
        """
        Calculate all performance metrics.
        
        Args:
            equity_curve: DataFrame with portfolio value over time
            trades: DataFrame with trade history
            benchmark: Optional benchmark returns series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            
        Returns:
            PerformanceMetrics instance
        """
        metrics = cls()
        
        if equity_curve.empty:
            return metrics
            
        # Calculate returns
        returns = equity_curve['total_value'].pct_change().dropna()
        
        # Basic return metrics
        metrics.total_return = (
            (equity_curve['total_value'].iloc[-1] - equity_curve['total_value'].iloc[0]) /
            equity_curve['total_value'].iloc[0] * 100
        )
        
        # Annualized return
        total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = total_days / 365.25
        metrics.annualized_return = (
            (1 + metrics.total_return / 100) ** (1 / years) - 1
        ) * 100 if years > 0 else 0
        
        # Trade statistics
        if not trades.empty:
            metrics.total_trades = len(trades[trades['type'] == 'CLOSE'])
            
            # Calculate trade returns
            trade_returns = []
            for i in range(len(trades)):
                if trades.iloc[i]['type'] == 'CLOSE' and trades.iloc[i]['position_pnl'] is not None:
                    trade_returns.append(trades.iloc[i]['position_pnl'])
                    
            if trade_returns:
                trade_returns = np.array(trade_returns)
                metrics.winning_trades = sum(trade_returns > 0)
                metrics.losing_trades = sum(trade_returns <= 0)
                metrics.win_rate = metrics.winning_trades / len(trade_returns) * 100
                
                # Win/loss statistics
                winning_returns = trade_returns[trade_returns > 0]
                losing_returns = trade_returns[trade_returns <= 0]
                
                if len(winning_returns) > 0:
                    metrics.avg_win = np.mean(winning_returns)
                    metrics.largest_win = np.max(winning_returns)
                    
                if len(losing_returns) > 0:
                    metrics.avg_loss = np.mean(losing_returns)
                    metrics.largest_loss = np.min(losing_returns)
                    
                # Profit factor
                gross_profit = np.sum(winning_returns) if len(winning_returns) > 0 else 0
                gross_loss = abs(np.sum(losing_returns)) if len(losing_returns) > 0 else 1
                metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
                
                # Expectancy
                metrics.expectancy = (
                    (metrics.win_rate / 100 * metrics.avg_win) +
                    ((1 - metrics.win_rate / 100) * metrics.avg_loss)
                )
                
        # Risk metrics
        if len(returns) > 1:
            # Volatility
            metrics.volatility = returns.std() * np.sqrt(periods_per_year) * 100
            
            # Sharpe ratio
            excess_returns = returns - risk_free_rate / periods_per_year
            if returns.std() > 0:
                metrics.sharpe_ratio = (
                    excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
                )
                
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                if downside_std > 0:
                    metrics.sortino_ratio = (
                        excess_returns.mean() / downside_std * np.sqrt(periods_per_year)
                    )
                    
        # Drawdown analysis
        drawdown_data = cls._calculate_drawdowns(equity_curve['total_value'])
        metrics.max_drawdown = drawdown_data['max_drawdown']
        metrics.max_drawdown_duration = drawdown_data['max_duration']
        
        # Calmar ratio
        if metrics.max_drawdown > 0 and years > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown
            
        # Benchmark comparison
        if benchmark is not None and len(benchmark) > 0:
            # Align dates
            aligned_data = pd.DataFrame({
                'portfolio': returns,
                'benchmark': benchmark
            }).dropna()
            
            if len(aligned_data) > 1:
                # Beta and Alpha
                covariance = np.cov(aligned_data['portfolio'], aligned_data['benchmark'])
                benchmark_variance = np.var(aligned_data['benchmark'])
                
                if benchmark_variance > 0:
                    metrics.beta = covariance[0, 1] / benchmark_variance
                    
                    # Alpha (Jensen's alpha)
                    benchmark_return = aligned_data['benchmark'].mean() * periods_per_year
                    expected_return = risk_free_rate + metrics.beta * (benchmark_return - risk_free_rate)
                    metrics.alpha = metrics.annualized_return / 100 - expected_return
                    
                # Information ratio
                tracking_error = (aligned_data['portfolio'] - aligned_data['benchmark']).std()
                if tracking_error > 0:
                    excess_return = aligned_data['portfolio'].mean() - aligned_data['benchmark'].mean()
                    metrics.information_ratio = (
                        excess_return / tracking_error * np.sqrt(periods_per_year)
                    )
                    
        return metrics
        
    @staticmethod
    def _calculate_drawdowns(equity_curve: pd.Series) -> Dict:
        """Calculate drawdown statistics."""
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown series
        drawdown = (equity_curve - running_max) / running_max * 100
        
        # Find maximum drawdown
        max_drawdown = abs(drawdown.min())
        
        # Calculate drawdown durations
        is_drawdown = drawdown < 0
        
        # Find drawdown periods
        drawdown_start = (~is_drawdown).shift(1) & is_drawdown
        drawdown_id = drawdown_start.cumsum()
        
        # Calculate durations
        durations = is_drawdown.groupby(drawdown_id).sum()
        max_duration = int(durations.max()) if len(durations) > 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_duration': max_duration,
            'drawdown_series': drawdown
        }
        
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            # Returns
            'total_return': f"{self.total_return:.2f}%",
            'annualized_return': f"{self.annualized_return:.2f}%",
            
            # Risk
            'volatility': f"{self.volatility:.2f}%",
            'max_drawdown': f"{self.max_drawdown:.2f}%",
            'max_drawdown_duration': f"{self.max_drawdown_duration} days",
            
            # Risk-adjusted returns
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'sortino_ratio': f"{self.sortino_ratio:.2f}",
            'calmar_ratio': f"{self.calmar_ratio:.2f}",
            
            # Trade statistics
            'total_trades': self.total_trades,
            'win_rate': f"{self.win_rate:.2f}%",
            'profit_factor': f"{self.profit_factor:.2f}",
            'expectancy': f"${self.expectancy:.2f}",
            
            # Win/Loss
            'avg_win': f"${self.avg_win:.2f}",
            'avg_loss': f"${self.avg_loss:.2f}",
            'largest_win': f"${self.largest_win:.2f}",
            'largest_loss': f"${self.largest_loss:.2f}",
            
            # Market correlation
            'alpha': f"{self.alpha:.4f}",
            'beta': f"{self.beta:.2f}",
            'information_ratio': f"{self.information_ratio:.2f}"
        }
        
    def generate_report(self) -> str:
        """Generate a formatted performance report."""
        report = """
=== PERFORMANCE REPORT ===

RETURNS
-------
Total Return: {total_return}
Annualized Return: {annualized_return}

RISK METRICS
------------
Volatility: {volatility}
Max Drawdown: {max_drawdown}
Max DD Duration: {max_drawdown_duration}

RISK-ADJUSTED RETURNS
--------------------
Sharpe Ratio: {sharpe_ratio}
Sortino Ratio: {sortino_ratio}
Calmar Ratio: {calmar_ratio}

TRADE STATISTICS
----------------
Total Trades: {total_trades}
Win Rate: {win_rate}
Profit Factor: {profit_factor}
Expectancy: {expectancy}

Average Win: {avg_win}
Average Loss: {avg_loss}
Largest Win: {largest_win}
Largest Loss: {largest_loss}

MARKET CORRELATION
------------------
Alpha: {alpha}
Beta: {beta}
Information Ratio: {information_ratio}
        """.format(**self.to_dict())
        
        return report
        
    @staticmethod
    def calculate_rolling_metrics(
        equity_curve: pd.DataFrame,
        window: int = 252,
        min_periods: int = 30
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            equity_curve: Portfolio value over time
            window: Rolling window size
            min_periods: Minimum periods required
            
        Returns:
            DataFrame with rolling metrics
        """
        returns = equity_curve['total_value'].pct_change().dropna()
        
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling returns
        rolling_metrics['return'] = returns.rolling(
            window=window, min_periods=min_periods
        ).apply(lambda x: (1 + x).prod() - 1) * 100
        
        # Rolling volatility
        rolling_metrics['volatility'] = returns.rolling(
            window=window, min_periods=min_periods
        ).std() * np.sqrt(252) * 100
        
        # Rolling Sharpe
        rolling_metrics['sharpe'] = returns.rolling(
            window=window, min_periods=min_periods
        ).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        
        # Rolling max drawdown
        def rolling_max_dd(values):
            cumulative = (1 + values).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return abs(drawdown.min()) * 100
            
        rolling_metrics['max_drawdown'] = returns.rolling(
            window=window, min_periods=min_periods
        ).apply(rolling_max_dd)
        
        return rolling_metrics