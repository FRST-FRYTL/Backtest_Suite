"""
Comprehensive Performance Attribution System

This module provides detailed performance attribution analysis to understand
the sources of returns and risk in the enhanced confluence strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AttributionResult:
    """Results from performance attribution analysis"""
    total_return: float
    attribution_components: Dict[str, float]
    factor_contributions: Dict[str, float]
    timing_contribution: float
    selection_contribution: float
    interaction_effect: float
    risk_adjusted_attribution: Dict[str, float]

@dataclass
class TimeSeriesAttribution:
    """Time series of attribution results"""
    dates: pd.DatetimeIndex
    cumulative_attribution: pd.DataFrame
    period_attribution: pd.DataFrame
    rolling_attribution: pd.DataFrame

class PerformanceAttributor:
    """
    Comprehensive performance attribution analysis system.
    """
    
    def __init__(self):
        """Initialize the performance attributor."""
        self.attribution_history: List[AttributionResult] = []
        self.factor_exposures: Dict[str, pd.Series] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        
    def calculate_return_attribution(
        self,
        trades: List[Dict[str, Any]],
        portfolio_returns: pd.Series,
        market_returns: pd.Series,
        factor_returns: Optional[Dict[str, pd.Series]] = None
    ) -> AttributionResult:
        """
        Calculate comprehensive return attribution.
        
        Args:
            trades: List of trade dictionaries with details
            portfolio_returns: Portfolio return series
            market_returns: Market return series
            factor_returns: Optional factor return series
            
        Returns:
            Attribution result with component breakdown
        """
        total_return = (1 + portfolio_returns).prod() - 1
        
        # Component attribution
        attribution_components = {}
        
        # 1. Market timing attribution
        timing_contribution = self._calculate_timing_attribution(
            trades, market_returns
        )
        attribution_components['timing'] = timing_contribution
        
        # 2. Security selection attribution
        selection_contribution = self._calculate_selection_attribution(
            trades, portfolio_returns, market_returns
        )
        attribution_components['selection'] = selection_contribution
        
        # 3. Confluence signal attribution
        confluence_contribution = self._calculate_confluence_attribution(trades)
        attribution_components['confluence'] = confluence_contribution
        
        # 4. Risk management attribution
        risk_contribution = self._calculate_risk_attribution(trades)
        attribution_components['risk_management'] = risk_contribution
        
        # 5. Timeframe attribution
        timeframe_contributions = self._calculate_timeframe_attribution(trades)
        attribution_components.update(timeframe_contributions)
        
        # Factor contributions (if factors provided)
        factor_contributions = {}
        if factor_returns:
            factor_contributions = self._calculate_factor_attribution(
                portfolio_returns, factor_returns
            )
        
        # Interaction effect
        sum_of_parts = sum(attribution_components.values())
        interaction_effect = total_return - sum_of_parts
        
        # Risk-adjusted attribution
        risk_adjusted_attribution = self._calculate_risk_adjusted_attribution(
            attribution_components, portfolio_returns
        )
        
        result = AttributionResult(
            total_return=total_return,
            attribution_components=attribution_components,
            factor_contributions=factor_contributions,
            timing_contribution=timing_contribution,
            selection_contribution=selection_contribution,
            interaction_effect=interaction_effect,
            risk_adjusted_attribution=risk_adjusted_attribution
        )
        
        self.attribution_history.append(result)
        return result
    
    def _calculate_timing_attribution(
        self,
        trades: List[Dict[str, Any]],
        market_returns: pd.Series
    ) -> float:
        """Calculate market timing contribution."""
        if not trades:
            return 0.0
        
        timing_returns = []
        
        for trade in trades:
            entry_date = pd.to_datetime(trade['entry_date'])
            exit_date = pd.to_datetime(trade['exit_date'])
            
            # Get market return during trade period
            mask = (market_returns.index >= entry_date) & (market_returns.index <= exit_date)
            if mask.any():
                period_market_return = (1 + market_returns[mask]).prod() - 1
                
                # Compare to average market return
                avg_market_return = market_returns.mean() * len(market_returns[mask])
                
                timing_value = period_market_return - avg_market_return
                timing_returns.append(timing_value * trade.get('position_weight', 0.1))
        
        return sum(timing_returns)
    
    def _calculate_selection_attribution(
        self,
        trades: List[Dict[str, Any]],
        portfolio_returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """Calculate security selection contribution."""
        if not trades:
            return 0.0
        
        selection_returns = []
        
        for trade in trades:
            trade_return = trade.get('return', 0)
            entry_date = pd.to_datetime(trade['entry_date'])
            exit_date = pd.to_datetime(trade['exit_date'])
            
            # Get market return for same period
            mask = (market_returns.index >= entry_date) & (market_returns.index <= exit_date)
            if mask.any():
                period_market_return = (1 + market_returns[mask]).prod() - 1
                
                # Selection effect = trade return - market return
                selection_effect = trade_return - period_market_return
                selection_returns.append(selection_effect * trade.get('position_weight', 0.1))
        
        return sum(selection_returns)
    
    def _calculate_confluence_attribution(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate attribution from confluence signals."""
        if not trades:
            return 0.0
        
        # Group trades by confluence score ranges
        confluence_buckets = {
            'high': [],      # > 0.8
            'medium': [],    # 0.65 - 0.8
            'low': []        # < 0.65
        }
        
        for trade in trades:
            confluence_score = trade.get('confluence_score', 0.5)
            trade_return = trade.get('return', 0)
            
            if confluence_score > 0.8:
                confluence_buckets['high'].append(trade_return)
            elif confluence_score > 0.65:
                confluence_buckets['medium'].append(trade_return)
            else:
                confluence_buckets['low'].append(trade_return)
        
        # Calculate contribution
        contributions = []
        
        if confluence_buckets['high']:
            high_avg = np.mean(confluence_buckets['high'])
            contributions.append(high_avg * len(confluence_buckets['high']))
        
        if confluence_buckets['medium']:
            medium_avg = np.mean(confluence_buckets['medium'])
            contributions.append(medium_avg * len(confluence_buckets['medium']))
        
        if confluence_buckets['low']:
            low_avg = np.mean(confluence_buckets['low'])
            # Negative contribution for low confluence trades
            contributions.append(low_avg * len(confluence_buckets['low']) * -0.5)
        
        total_trades = sum(len(b) for b in confluence_buckets.values())
        if total_trades > 0:
            return sum(contributions) / total_trades
        
        return 0.0
    
    def _calculate_risk_attribution(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate attribution from risk management."""
        if not trades:
            return 0.0
        
        risk_contributions = []
        
        for trade in trades:
            exit_reason = trade.get('exit_reason', '')
            trade_return = trade.get('return', 0)
            max_risk = trade.get('max_risk', 0.05)
            
            # Positive contribution if stop loss saved capital
            if exit_reason == 'stop_loss' and trade_return < 0:
                # Saved capital = max_risk - actual loss
                saved = max(0, max_risk + trade_return)
                risk_contributions.append(saved)
            
            # Positive contribution if take profit captured gains
            elif exit_reason == 'take_profit' and trade_return > 0:
                # Discipline factor
                risk_contributions.append(trade_return * 0.1)
        
        return sum(risk_contributions) / len(trades) if trades else 0.0
    
    def _calculate_timeframe_attribution(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate attribution by timeframe."""
        timeframe_returns = {
            'tf_1H': [],
            'tf_4H': [],
            'tf_1D': [],
            'tf_1W': [],
            'tf_1M': []
        }
        
        for trade in trades:
            trade_return = trade.get('return', 0)
            timeframe_scores = trade.get('timeframe_scores', {})
            
            # Weight return by timeframe score
            for tf, score in timeframe_scores.items():
                tf_key = f'tf_{tf}'
                if tf_key in timeframe_returns:
                    contribution = trade_return * score
                    timeframe_returns[tf_key].append(contribution)
        
        # Calculate average contribution by timeframe
        timeframe_contributions = {}
        for tf, returns in timeframe_returns.items():
            if returns:
                timeframe_contributions[tf] = np.mean(returns)
            else:
                timeframe_contributions[tf] = 0.0
        
        return timeframe_contributions
    
    def _calculate_factor_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """Calculate factor-based attribution."""
        factor_contributions = {}
        
        # Align all series to common dates
        common_index = portfolio_returns.index
        for factor_name, factor_series in factor_returns.items():
            common_index = common_index.intersection(factor_series.index)
        
        portfolio_aligned = portfolio_returns.loc[common_index]
        
        # Run regression to get factor loadings
        for factor_name, factor_series in factor_returns.items():
            factor_aligned = factor_series.loc[common_index]
            
            # Simple regression
            if len(factor_aligned) > 10:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    factor_aligned, portfolio_aligned
                )
                
                # Factor contribution = beta * factor return
                factor_return = (1 + factor_aligned).prod() - 1
                contribution = slope * factor_return
                
                factor_contributions[factor_name] = contribution
            else:
                factor_contributions[factor_name] = 0.0
        
        return factor_contributions
    
    def _calculate_risk_adjusted_attribution(
        self,
        attribution_components: Dict[str, float],
        portfolio_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate risk-adjusted attribution."""
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        
        risk_adjusted = {}
        for component, value in attribution_components.items():
            # Simple risk adjustment using portfolio volatility
            risk_adjusted[f'{component}_risk_adj'] = value / portfolio_vol if portfolio_vol > 0 else 0
        
        return risk_adjusted
    
    def calculate_time_series_attribution(
        self,
        trades: List[Dict[str, Any]],
        portfolio_values: pd.Series,
        window: int = 20
    ) -> TimeSeriesAttribution:
        """
        Calculate attribution over time.
        
        Args:
            trades: List of trades with timestamps
            portfolio_values: Portfolio value series
            window: Rolling window for attribution
            
        Returns:
            Time series attribution results
        """
        # Create daily attribution series
        dates = portfolio_values.index
        attribution_data = []
        
        # Group trades by date
        trades_by_date = {}
        for trade in trades:
            exit_date = pd.to_datetime(trade['exit_date']).date()
            if exit_date not in trades_by_date:
                trades_by_date[exit_date] = []
            trades_by_date[exit_date].append(trade)
        
        # Calculate daily attribution
        for date in dates:
            date_key = date.date()
            
            if date_key in trades_by_date:
                day_trades = trades_by_date[date_key]
                
                # Calculate component contributions
                daily_attribution = {
                    'date': date,
                    'selection': sum(t.get('return', 0) - t.get('benchmark_return', 0) 
                                   for t in day_trades) / len(day_trades),
                    'timing': sum(t.get('timing_value', 0) for t in day_trades) / len(day_trades),
                    'confluence': sum(t.get('confluence_score', 0.5) * t.get('return', 0) 
                                    for t in day_trades) / len(day_trades),
                    'trade_count': len(day_trades)
                }
            else:
                daily_attribution = {
                    'date': date,
                    'selection': 0,
                    'timing': 0,
                    'confluence': 0,
                    'trade_count': 0
                }
            
            attribution_data.append(daily_attribution)
        
        # Create DataFrames
        period_attribution = pd.DataFrame(attribution_data).set_index('date')
        
        # Calculate cumulative attribution
        cumulative_attribution = period_attribution.cumsum()
        
        # Calculate rolling attribution
        rolling_attribution = period_attribution.rolling(window=window).mean()
        
        return TimeSeriesAttribution(
            dates=dates,
            cumulative_attribution=cumulative_attribution,
            period_attribution=period_attribution,
            rolling_attribution=rolling_attribution
        )
    
    def decompose_alpha(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Decompose alpha into various components.
        
        Args:
            strategy_returns: Strategy return series
            benchmark_returns: Benchmark return series
            risk_free_rate: Risk-free rate
            
        Returns:
            Alpha decomposition dictionary
        """
        # Calculate basic metrics
        strategy_mean = strategy_returns.mean() * 252
        benchmark_mean = benchmark_returns.mean() * 252
        
        strategy_vol = strategy_returns.std() * np.sqrt(252)
        benchmark_vol = benchmark_returns.std() * np.sqrt(252)
        
        # Calculate beta
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        variance_benchmark = np.var(benchmark_returns)
        beta = covariance / variance_benchmark if variance_benchmark > 0 else 1.0
        
        # Jensen's alpha
        jensen_alpha = strategy_mean - (risk_free_rate + beta * (benchmark_mean - risk_free_rate))
        
        # Information ratio
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        # Treynor ratio
        treynor_ratio = (strategy_mean - risk_free_rate) / beta if beta > 0 else 0
        
        # M-squared (Modigliani-Modigliani measure)
        sharpe_strategy = (strategy_mean - risk_free_rate) / strategy_vol if strategy_vol > 0 else 0
        m_squared = risk_free_rate + sharpe_strategy * benchmark_vol
        
        # Decomposition
        decomposition = {
            'total_alpha': jensen_alpha,
            'selection_alpha': jensen_alpha * 0.6,  # Simplified allocation
            'timing_alpha': jensen_alpha * 0.3,
            'risk_alpha': jensen_alpha * 0.1,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'm_squared': m_squared,
            'beta': beta,
            'tracking_error': tracking_error,
            'active_return': strategy_mean - benchmark_mean
        }
        
        return decomposition
    
    def analyze_factor_exposures(
        self,
        returns: pd.Series,
        factors: Dict[str, pd.Series],
        lookback: int = 252
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze rolling factor exposures.
        
        Args:
            returns: Strategy returns
            factors: Factor return series
            lookback: Lookback period for rolling analysis
            
        Returns:
            Factor exposure analysis
        """
        exposures = {}
        
        for factor_name, factor_returns in factors.items():
            # Align series
            common_idx = returns.index.intersection(factor_returns.index)
            returns_aligned = returns.loc[common_idx]
            factor_aligned = factor_returns.loc[common_idx]
            
            if len(common_idx) < lookback:
                continue
            
            # Calculate rolling exposures
            rolling_betas = []
            rolling_correlations = []
            
            for i in range(lookback, len(common_idx)):
                window_returns = returns_aligned.iloc[i-lookback:i]
                window_factor = factor_aligned.iloc[i-lookback:i]
                
                # Beta
                cov = np.cov(window_returns, window_factor)[0, 1]
                var = np.var(window_factor)
                beta = cov / var if var > 0 else 0
                rolling_betas.append(beta)
                
                # Correlation
                corr = np.corrcoef(window_returns, window_factor)[0, 1]
                rolling_correlations.append(corr)
            
            exposures[factor_name] = {
                'current_beta': rolling_betas[-1] if rolling_betas else 0,
                'avg_beta': np.mean(rolling_betas) if rolling_betas else 0,
                'beta_stability': 1 - np.std(rolling_betas) if rolling_betas else 0,
                'avg_correlation': np.mean(rolling_correlations) if rolling_correlations else 0,
                'max_correlation': np.max(np.abs(rolling_correlations)) if rolling_correlations else 0
            }
        
        return exposures
    
    def calculate_performance_consistency(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        periods: List[str] = ['daily', 'weekly', 'monthly']
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance consistency across different time periods.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            periods: List of periods to analyze
            
        Returns:
            Consistency metrics by period
        """
        consistency_metrics = {}
        
        for period in periods:
            if period == 'daily':
                period_returns = returns
                period_benchmark = benchmark_returns
            elif period == 'weekly':
                period_returns = returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
                period_benchmark = benchmark_returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
            elif period == 'monthly':
                period_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                period_benchmark = benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            else:
                continue
            
            # Calculate metrics
            outperformance = period_returns - period_benchmark
            
            consistency_metrics[period] = {
                'win_rate': (period_returns > 0).mean(),
                'outperformance_rate': (outperformance > 0).mean(),
                'avg_return': period_returns.mean(),
                'return_volatility': period_returns.std(),
                'downside_deviation': period_returns[period_returns < 0].std() if len(period_returns[period_returns < 0]) > 0 else 0,
                'best_period': period_returns.max(),
                'worst_period': period_returns.min(),
                'positive_periods': (period_returns > 0).sum(),
                'negative_periods': (period_returns < 0).sum()
            }
        
        return consistency_metrics
    
    def generate_attribution_report(self) -> Dict[str, Any]:
        """Generate comprehensive attribution report."""
        if not self.attribution_history:
            return {}
        
        latest_attribution = self.attribution_history[-1]
        
        # Aggregate historical attribution
        historical_components = {}
        for attr in self.attribution_history:
            for component, value in attr.attribution_components.items():
                if component not in historical_components:
                    historical_components[component] = []
                historical_components[component].append(value)
        
        # Calculate averages and trends
        component_analysis = {}
        for component, values in historical_components.items():
            component_analysis[component] = {
                'current': values[-1] if values else 0,
                'average': np.mean(values) if values else 0,
                'volatility': np.std(values) if values else 0,
                'trend': 'increasing' if len(values) > 1 and values[-1] > values[0] else 'decreasing',
                'contribution_pct': (np.mean(values) / latest_attribution.total_return * 100) if latest_attribution.total_return != 0 else 0
            }
        
        report = {
            'total_return': latest_attribution.total_return,
            'attribution_components': latest_attribution.attribution_components,
            'component_analysis': component_analysis,
            'factor_contributions': latest_attribution.factor_contributions,
            'risk_adjusted_attribution': latest_attribution.risk_adjusted_attribution,
            'summary': {
                'primary_return_driver': max(latest_attribution.attribution_components, 
                                           key=latest_attribution.attribution_components.get),
                'selection_vs_timing': {
                    'selection': latest_attribution.selection_contribution,
                    'timing': latest_attribution.timing_contribution,
                    'ratio': latest_attribution.selection_contribution / latest_attribution.timing_contribution 
                            if latest_attribution.timing_contribution != 0 else 0
                }
            }
        }
        
        return report