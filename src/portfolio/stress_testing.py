"""
Stress Testing Framework

This module implements comprehensive stress testing including historical scenarios,
Monte Carlo simulations, and sensitivity analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StressScenario:
    """Represents a stress test scenario."""
    
    def __init__(
        self,
        name: str,
        description: str,
        market_shocks: Dict[str, float],
        duration_days: int = 1,
        correlation_shift: float = 0.0
    ):
        """
        Initialize stress scenario.
        
        Args:
            name: Scenario name
            description: Scenario description
            market_shocks: Shocks to apply {asset: return_shock}
            duration_days: Scenario duration
            correlation_shift: Increase in correlations
        """
        self.name = name
        self.description = description
        self.market_shocks = market_shocks
        self.duration_days = duration_days
        self.correlation_shift = correlation_shift

class StressTester:
    """
    Comprehensive stress testing framework for portfolios.
    """
    
    def __init__(
        self,
        confidence_levels: List[float] = [0.95, 0.99],
        n_simulations: int = 10000,
        time_horizon: int = 252  # 1 year
    ):
        """
        Initialize stress tester.
        
        Args:
            confidence_levels: VaR/CVaR confidence levels
            n_simulations: Number of Monte Carlo simulations
            time_horizon: Time horizon in days
        """
        self.confidence_levels = confidence_levels
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon
        
        # Predefined scenarios
        self.scenarios = self._initialize_scenarios()
        
    def _initialize_scenarios(self) -> List[StressScenario]:
        """Initialize predefined stress scenarios."""
        scenarios = [
            # 2008 Financial Crisis
            StressScenario(
                name="2008 Financial Crisis",
                description="Global financial meltdown",
                market_shocks={
                    'equity': -0.37,
                    'bonds': 0.05,
                    'commodities': -0.30,
                    'real_estate': -0.40,
                    'default': -0.35
                },
                duration_days=60,
                correlation_shift=0.3
            ),
            
            # COVID-19 Crash
            StressScenario(
                name="COVID-19 Crash",
                description="March 2020 pandemic crash",
                market_shocks={
                    'equity': -0.34,
                    'bonds': 0.08,
                    'commodities': -0.25,
                    'real_estate': -0.15,
                    'default': -0.30
                },
                duration_days=30,
                correlation_shift=0.4
            ),
            
            # Tech Bubble Burst
            StressScenario(
                name="Dot-Com Crash",
                description="2000-2002 tech bubble burst",
                market_shocks={
                    'equity': -0.49,
                    'bonds': 0.10,
                    'tech': -0.78,
                    'default': -0.40
                },
                duration_days=90,
                correlation_shift=0.2
            ),
            
            # Interest Rate Shock
            StressScenario(
                name="Interest Rate Shock",
                description="Rapid rate increase scenario",
                market_shocks={
                    'equity': -0.15,
                    'bonds': -0.20,
                    'real_estate': -0.25,
                    'default': -0.18
                },
                duration_days=30,
                correlation_shift=0.1
            ),
            
            # Inflation Surge
            StressScenario(
                name="Inflation Surge",
                description="High inflation scenario",
                market_shocks={
                    'equity': -0.10,
                    'bonds': -0.15,
                    'commodities': 0.25,
                    'real_estate': 0.05,
                    'default': -0.08
                },
                duration_days=180,
                correlation_shift=0.05
            ),
            
            # Geopolitical Crisis
            StressScenario(
                name="Geopolitical Crisis",
                description="Major geopolitical event",
                market_shocks={
                    'equity': -0.20,
                    'bonds': 0.10,
                    'commodities': 0.15,
                    'emerging_markets': -0.30,
                    'default': -0.15
                },
                duration_days=45,
                correlation_shift=0.25
            )
        ]
        
        return scenarios
    
    def run_historical_stress_test(
        self,
        portfolio_weights: Dict[str, float],
        returns_data: pd.DataFrame,
        scenarios: Optional[List[StressScenario]] = None
    ) -> pd.DataFrame:
        """
        Run historical stress test scenarios.
        
        Args:
            portfolio_weights: Portfolio weights
            returns_data: Historical returns data
            scenarios: Custom scenarios (uses defaults if None)
            
        Returns:
            DataFrame with stress test results
        """
        scenarios = scenarios or self.scenarios
        results = []
        
        for scenario in scenarios:
            # Apply shocks to portfolio
            portfolio_return = 0
            
            for asset, weight in portfolio_weights.items():
                # Get asset class or use default shock
                asset_class = self._get_asset_class(asset)
                shock = scenario.market_shocks.get(
                    asset_class, 
                    scenario.market_shocks.get('default', -0.10)
                )
                
                portfolio_return += weight * shock
            
            # Calculate risk metrics under stress
            stressed_vol = self._calculate_stressed_volatility(
                returns_data, 
                portfolio_weights,
                scenario.correlation_shift
            )
            
            results.append({
                'scenario': scenario.name,
                'description': scenario.description,
                'portfolio_return': portfolio_return,
                'stressed_volatility': stressed_vol,
                'duration_days': scenario.duration_days,
                'max_loss': portfolio_return,
                'recovery_time': self._estimate_recovery_time(portfolio_return)
            })
        
        return pd.DataFrame(results)
    
    def run_monte_carlo_stress_test(
        self,
        portfolio_weights: Dict[str, float],
        returns_data: pd.DataFrame,
        n_simulations: Optional[int] = None
    ) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
        """
        Run Monte Carlo stress test.
        
        Args:
            portfolio_weights: Portfolio weights
            returns_data: Historical returns data
            n_simulations: Number of simulations
            
        Returns:
            Dictionary with simulation results
        """
        n_simulations = n_simulations or self.n_simulations
        
        # Calculate statistics
        assets = list(portfolio_weights.keys())
        weights = np.array([portfolio_weights[asset] for asset in assets])
        
        # Filter returns data
        asset_returns = returns_data[assets]
        mean_returns = asset_returns.mean().values
        cov_matrix = asset_returns.cov().values
        
        # Generate scenarios
        random_returns = np.random.multivariate_normal(
            mean_returns,
            cov_matrix,
            size=(n_simulations, self.time_horizon)
        )
        
        # Calculate portfolio returns for each scenario
        portfolio_returns = np.zeros((n_simulations, self.time_horizon))
        
        for i in range(n_simulations):
            for t in range(self.time_horizon):
                portfolio_returns[i, t] = random_returns[i, t] @ weights
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1)
        
        # Calculate metrics
        final_returns = cumulative_returns[:, -1] - 1
        max_drawdowns = self._calculate_max_drawdowns(cumulative_returns)
        
        # VaR and CVaR
        var_results = {}
        cvar_results = {}
        
        for confidence in self.confidence_levels:
            var_threshold = np.percentile(final_returns, (1 - confidence) * 100)
            var_results[f'VaR_{int(confidence*100)}'] = var_threshold
            cvar_results[f'CVaR_{int(confidence*100)}'] = final_returns[final_returns <= var_threshold].mean()
        
        # Compile results
        results = {
            'summary': pd.DataFrame({
                'metric': ['mean_return', 'median_return', 'std_dev', 'skewness', 'kurtosis',
                          'max_return', 'min_return', 'max_drawdown'] + 
                         list(var_results.keys()) + list(cvar_results.keys()),
                'value': [
                    final_returns.mean(),
                    np.median(final_returns),
                    final_returns.std(),
                    stats.skew(final_returns),
                    stats.kurtosis(final_returns),
                    final_returns.max(),
                    final_returns.min(),
                    max_drawdowns.mean()
                ] + list(var_results.values()) + list(cvar_results.values())
            }),
            'simulated_returns': final_returns,
            'simulated_paths': cumulative_returns,
            'drawdowns': max_drawdowns
        }
        
        return results
    
    def run_sensitivity_analysis(
        self,
        portfolio_weights: Dict[str, float],
        returns_data: pd.DataFrame,
        factor_ranges: Dict[str, Tuple[float, float]],
        n_steps: int = 20
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis on portfolio.
        
        Args:
            portfolio_weights: Portfolio weights
            returns_data: Historical returns data
            factor_ranges: Ranges for factors {factor: (min, max)}
            n_steps: Number of steps in sensitivity
            
        Returns:
            DataFrame with sensitivity results
        """
        results = []
        
        # Default factors if not specified
        if not factor_ranges:
            factor_ranges = {
                'market_return': (-0.30, 0.30),
                'volatility_mult': (0.5, 2.0),
                'correlation_add': (-0.3, 0.3)
            }
        
        # Base case metrics
        base_metrics = self._calculate_portfolio_metrics(
            portfolio_weights,
            returns_data
        )
        
        # Test each factor
        for factor, (min_val, max_val) in factor_ranges.items():
            factor_values = np.linspace(min_val, max_val, n_steps)
            
            for value in factor_values:
                # Apply factor shock
                shocked_returns = self._apply_factor_shock(
                    returns_data,
                    factor,
                    value
                )
                
                # Calculate metrics
                metrics = self._calculate_portfolio_metrics(
                    portfolio_weights,
                    shocked_returns
                )
                
                results.append({
                    'factor': factor,
                    'factor_value': value,
                    'return': metrics['return'],
                    'volatility': metrics['volatility'],
                    'sharpe': metrics['sharpe'],
                    'max_drawdown': metrics['max_drawdown'],
                    'return_change': metrics['return'] - base_metrics['return'],
                    'vol_change': metrics['volatility'] - base_metrics['volatility']
                })
        
        return pd.DataFrame(results)
    
    def calculate_tail_risk_measures(
        self,
        portfolio_weights: Dict[str, float],
        returns_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate tail risk measures.
        
        Args:
            portfolio_weights: Portfolio weights
            returns_data: Historical returns data
            
        Returns:
            Dictionary of tail risk measures
        """
        # Calculate portfolio returns
        assets = list(portfolio_weights.keys())
        weights = np.array([portfolio_weights[asset] for asset in assets])
        portfolio_returns = returns_data[assets] @ weights
        
        # Tail risk measures
        measures = {}
        
        # Expected Shortfall (CVaR) at multiple levels
        for confidence in self.confidence_levels:
            var_threshold = np.percentile(portfolio_returns, (1 - confidence) * 100)
            measures[f'ES_{int(confidence*100)}'] = portfolio_returns[portfolio_returns <= var_threshold].mean()
        
        # Tail ratio
        right_tail = portfolio_returns[portfolio_returns > np.percentile(portfolio_returns, 95)].mean()
        left_tail = abs(portfolio_returns[portfolio_returns < np.percentile(portfolio_returns, 5)].mean())
        measures['tail_ratio'] = right_tail / left_tail if left_tail > 0 else np.inf
        
        # Maximum drawdown duration
        cum_returns = (1 + portfolio_returns).cumprod()
        drawdown_periods = self._calculate_drawdown_periods(cum_returns)
        measures['max_dd_duration'] = max(drawdown_periods) if drawdown_periods else 0
        measures['avg_dd_duration'] = np.mean(drawdown_periods) if drawdown_periods else 0
        
        # Downside deviation
        downside_returns = portfolio_returns[portfolio_returns < 0]
        measures['downside_deviation'] = downside_returns.std() * np.sqrt(252)
        
        # Sortino ratio
        excess_return = portfolio_returns.mean() * 252 - 0.02  # Assume 2% risk-free rate
        measures['sortino_ratio'] = excess_return / measures['downside_deviation'] if measures['downside_deviation'] > 0 else 0
        
        # Calmar ratio
        annual_return = portfolio_returns.mean() * 252
        max_dd = self._calculate_max_drawdown(cum_returns)
        measures['calmar_ratio'] = annual_return / abs(max_dd) if max_dd < 0 else 0
        
        return measures
    
    def generate_stress_report(
        self,
        portfolio_weights: Dict[str, float],
        returns_data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive stress test report.
        
        Args:
            portfolio_weights: Portfolio weights
            returns_data: Historical returns data
            output_path: Optional path to save report
            
        Returns:
            Dictionary of report components
        """
        logger.info("Generating stress test report...")
        
        # Run all stress tests
        report = {}
        
        # Historical scenarios
        report['historical_stress'] = self.run_historical_stress_test(
            portfolio_weights,
            returns_data
        )
        
        # Monte Carlo simulation
        mc_results = self.run_monte_carlo_stress_test(
            portfolio_weights,
            returns_data,
            n_simulations=5000  # Reduced for performance
        )
        report['monte_carlo_summary'] = mc_results['summary']
        
        # Sensitivity analysis
        report['sensitivity'] = self.run_sensitivity_analysis(
            portfolio_weights,
            returns_data,
            factor_ranges=None
        )
        
        # Tail risk measures
        tail_measures = self.calculate_tail_risk_measures(
            portfolio_weights,
            returns_data
        )
        report['tail_risk'] = pd.DataFrame([tail_measures]).T.reset_index()
        report['tail_risk'].columns = ['measure', 'value']
        
        # Save if path provided
        if output_path:
            with pd.ExcelWriter(output_path) as writer:
                for sheet_name, df in report.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"Stress report saved to {output_path}")
        
        return report
    
    def _get_asset_class(self, asset: str) -> str:
        """Map asset to asset class."""
        # Simple mapping - can be enhanced
        asset_lower = asset.lower()
        
        if any(term in asset_lower for term in ['stock', 'equity', 'spy', 'qqq']):
            return 'equity'
        elif any(term in asset_lower for term in ['bond', 'treasury', 'tlt', 'agg']):
            return 'bonds'
        elif any(term in asset_lower for term in ['commodity', 'gold', 'oil', 'gld']):
            return 'commodities'
        elif any(term in asset_lower for term in ['reit', 'real_estate', 'vnq']):
            return 'real_estate'
        elif any(term in asset_lower for term in ['tech', 'nasdaq', 'xlk']):
            return 'tech'
        elif any(term in asset_lower for term in ['emerging', 'eem', 'vwo']):
            return 'emerging_markets'
        else:
            return 'default'
    
    def _calculate_stressed_volatility(
        self,
        returns_data: pd.DataFrame,
        weights: Dict[str, float],
        correlation_shift: float
    ) -> float:
        """Calculate volatility under stressed correlations."""
        assets = list(weights.keys())
        w = np.array([weights[asset] for asset in assets])
        
        # Get covariance matrix
        cov_matrix = returns_data[assets].cov().values
        
        # Stress correlations
        corr_matrix = returns_data[assets].corr().values
        stressed_corr = corr_matrix + correlation_shift * (1 - corr_matrix)
        stressed_corr = np.clip(stressed_corr, -1, 1)
        np.fill_diagonal(stressed_corr, 1)
        
        # Convert back to covariance
        std_devs = np.sqrt(np.diag(cov_matrix))
        stressed_cov = np.outer(std_devs, std_devs) * stressed_corr
        
        # Portfolio volatility
        portfolio_var = w @ stressed_cov @ w
        return np.sqrt(portfolio_var * 252)
    
    def _estimate_recovery_time(self, drawdown: float) -> int:
        """Estimate recovery time based on drawdown magnitude."""
        # Empirical relationship
        if drawdown > -0.05:
            return 30
        elif drawdown > -0.10:
            return 60
        elif drawdown > -0.20:
            return 180
        elif drawdown > -0.30:
            return 365
        else:
            return 730
    
    def _calculate_max_drawdowns(self, cumulative_returns: np.ndarray) -> np.ndarray:
        """Calculate maximum drawdown for each path."""
        max_drawdowns = np.zeros(cumulative_returns.shape[0])
        
        for i in range(cumulative_returns.shape[0]):
            running_max = np.maximum.accumulate(cumulative_returns[i])
            drawdowns = (cumulative_returns[i] - running_max) / running_max
            max_drawdowns[i] = drawdowns.min()
        
        return max_drawdowns
    
    def _calculate_max_drawdown(self, cum_returns: pd.Series) -> float:
        """Calculate maximum drawdown for a series."""
        running_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - running_max) / running_max
        return drawdowns.min()
    
    def _calculate_drawdown_periods(self, cum_returns: pd.Series) -> List[int]:
        """Calculate drawdown period lengths."""
        running_max = cum_returns.expanding().max()
        drawdown = cum_returns < running_max
        
        periods = []
        current_period = 0
        
        for is_dd in drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            periods.append(current_period)
        
        return periods
    
    def _calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        returns_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate portfolio metrics."""
        assets = list(weights.keys())
        w = np.array([weights[asset] for asset in assets])
        portfolio_returns = returns_data[assets] @ w
        
        cum_returns = (1 + portfolio_returns).cumprod()
        
        return {
            'return': portfolio_returns.mean() * 252,
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe': (portfolio_returns.mean() * 252 - 0.02) / (portfolio_returns.std() * np.sqrt(252)),
            'max_drawdown': self._calculate_max_drawdown(cum_returns)
        }
    
    def _apply_factor_shock(
        self,
        returns_data: pd.DataFrame,
        factor: str,
        value: float
    ) -> pd.DataFrame:
        """Apply factor shock to returns."""
        shocked_returns = returns_data.copy()
        
        if factor == 'market_return':
            # Add constant return shock
            shocked_returns = shocked_returns + value / 252
        elif factor == 'volatility_mult':
            # Scale returns by volatility multiplier
            shocked_returns = shocked_returns * value
        elif factor == 'correlation_add':
            # This would require more complex manipulation
            # For now, just return original
            pass
        
        return shocked_returns