"""
Portfolio Optimization Module

This module implements various portfolio optimization algorithms including
Mean-Variance Optimization, Risk Parity, and Maximum Sharpe Ratio optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy.optimize import minimize
from scipy import stats
import cvxpy as cp
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Portfolio optimization objectives."""
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"
    MIN_CVaR = "min_cvar"
    TARGET_RETURN = "target_return"
    TARGET_RISK = "target_risk"

class PortfolioOptimizer:
    """
    Advanced portfolio optimization with multiple objectives and constraints.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02,
        confidence_level: float = 0.95,
        rebalance_frequency: str = 'monthly'
    ):
        """
        Initialize portfolio optimizer.
        
        Args:
            returns: DataFrame of asset returns
            risk_free_rate: Annual risk-free rate
            confidence_level: Confidence level for VaR/CVaR
            rebalance_frequency: Frequency of rebalancing
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.rebalance_frequency = rebalance_frequency
        
        # Calculate statistics
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.n_assets = len(returns.columns)
        
        # Annualization factor
        self.periods_per_year = self._get_annualization_factor()
        
    def _get_annualization_factor(self) -> int:
        """Get annualization factor based on data frequency."""
        if self.returns.empty:
            return 252
        
        # Infer frequency from index
        freq = pd.infer_freq(self.returns.index)
        if freq:
            if freq.startswith('D'):
                return 252
            elif freq.startswith('W'):
                return 52
            elif freq.startswith('M'):
                return 12
            elif freq.startswith('Q'):
                return 4
            elif freq.startswith('Y'):
                return 1
        
        # Default to daily
        return 252
    
    def optimize(
        self,
        objective: OptimizationObjective,
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Optimize portfolio based on specified objective.
        
        Args:
            objective: Optimization objective
            constraints: Portfolio constraints
            target_return: Target return for constrained optimization
            target_risk: Target risk for constrained optimization
            
        Returns:
            Dictionary with optimal weights and metrics
        """
        constraints = constraints or {}
        
        # Set default constraints
        constraints.setdefault('long_only', True)
        constraints.setdefault('max_weight', 0.4)
        constraints.setdefault('min_weight', 0.0)
        constraints.setdefault('leverage', 1.0)
        
        if objective == OptimizationObjective.MIN_VARIANCE:
            return self._minimize_variance(constraints)
        elif objective == OptimizationObjective.MAX_SHARPE:
            return self._maximize_sharpe(constraints)
        elif objective == OptimizationObjective.RISK_PARITY:
            return self._risk_parity(constraints)
        elif objective == OptimizationObjective.MAX_DIVERSIFICATION:
            return self._maximize_diversification(constraints)
        elif objective == OptimizationObjective.MIN_CVaR:
            return self._minimize_cvar(constraints)
        elif objective == OptimizationObjective.TARGET_RETURN:
            return self._target_return(target_return, constraints)
        elif objective == OptimizationObjective.TARGET_RISK:
            return self._target_risk(target_risk, constraints)
        else:
            raise ValueError(f"Unknown objective: {objective}")
    
    def _minimize_variance(self, constraints: Dict) -> Dict:
        """Minimize portfolio variance."""
        # Use CVXPY for convex optimization
        w = cp.Variable(self.n_assets)
        
        # Objective: minimize variance
        portfolio_variance = cp.quad_form(w, self.cov_matrix.values)
        objective = cp.Minimize(portfolio_variance)
        
        # Constraints
        constraints_list = [cp.sum(w) == constraints['leverage']]
        
        if constraints['long_only']:
            constraints_list.append(w >= constraints['min_weight'])
        
        if constraints['max_weight'] < 1:
            constraints_list.append(w <= constraints['max_weight'])
        
        # Solve
        prob = cp.Problem(objective, constraints_list)
        prob.solve()
        
        if prob.status != cp.OPTIMAL:
            logger.warning(f"Optimization failed: {prob.status}")
            # Return equal weights as fallback
            weights = np.ones(self.n_assets) / self.n_assets
        else:
            weights = w.value
        
        return self._calculate_portfolio_metrics(weights)
    
    def _maximize_sharpe(self, constraints: Dict) -> Dict:
        """Maximize Sharpe ratio."""
        def neg_sharpe(w):
            # Calculate portfolio metrics
            port_return = np.dot(w, self.mean_returns)
            port_vol = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
            
            # Annualize
            annual_return = port_return * self.periods_per_year
            annual_vol = port_vol * np.sqrt(self.periods_per_year)
            
            # Sharpe ratio (negative for minimization)
            sharpe = -(annual_return - self.risk_free_rate) / annual_vol
            return sharpe
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Bounds
        bounds = tuple((constraints['min_weight'], constraints['max_weight']) 
                      for _ in range(self.n_assets))
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - constraints['leverage']}]
        
        # Optimize
        result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if not result.success:
            logger.warning(f"Sharpe optimization failed: {result.message}")
            weights = x0
        else:
            weights = result.x
        
        return self._calculate_portfolio_metrics(weights)
    
    def _risk_parity(self, constraints: Dict) -> Dict:
        """Risk parity optimization."""
        def risk_parity_objective(w):
            # Calculate marginal risk contributions
            portfolio_vol = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
            marginal_contrib = np.dot(self.cov_matrix, w) / portfolio_vol
            contrib = w * marginal_contrib
            
            # Risk parity: minimize difference in risk contributions
            avg_contrib = np.mean(contrib)
            return np.sum((contrib - avg_contrib) ** 2)
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Bounds
        bounds = tuple((0.01, constraints['max_weight']) for _ in range(self.n_assets))
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - constraints['leverage']}]
        
        # Optimize
        result = minimize(risk_parity_objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=cons)
        
        if not result.success:
            logger.warning(f"Risk parity optimization failed: {result.message}")
            weights = x0
        else:
            weights = result.x
        
        return self._calculate_portfolio_metrics(weights)
    
    def _maximize_diversification(self, constraints: Dict) -> Dict:
        """Maximize diversification ratio."""
        def neg_diversification(w):
            # Weighted average of individual volatilities
            avg_vol = np.dot(w, np.sqrt(np.diag(self.cov_matrix)))
            
            # Portfolio volatility
            port_vol = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
            
            # Diversification ratio (negative for minimization)
            div_ratio = -avg_vol / port_vol
            return div_ratio
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Bounds
        bounds = tuple((constraints['min_weight'], constraints['max_weight']) 
                      for _ in range(self.n_assets))
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - constraints['leverage']}]
        
        # Optimize
        result = minimize(neg_diversification, x0, method='SLSQP', 
                         bounds=bounds, constraints=cons)
        
        if not result.success:
            logger.warning(f"Diversification optimization failed: {result.message}")
            weights = x0
        else:
            weights = result.x
        
        return self._calculate_portfolio_metrics(weights)
    
    def _minimize_cvar(self, constraints: Dict) -> Dict:
        """Minimize Conditional Value at Risk (CVaR)."""
        # Number of scenarios
        n_scenarios = len(self.returns)
        
        # Variables
        w = cp.Variable(self.n_assets)
        z = cp.Variable(n_scenarios)
        zeta = cp.Variable()
        
        # Portfolio returns for each scenario
        portfolio_returns = self.returns.values @ w
        
        # CVaR formulation
        cvar = zeta + 1 / ((1 - self.confidence_level) * n_scenarios) * cp.sum(z)
        
        # Constraints
        constraints_list = [
            z >= -portfolio_returns - zeta,
            z >= 0,
            cp.sum(w) == constraints['leverage']
        ]
        
        if constraints['long_only']:
            constraints_list.append(w >= constraints['min_weight'])
        
        if constraints['max_weight'] < 1:
            constraints_list.append(w <= constraints['max_weight'])
        
        # Objective
        objective = cp.Minimize(cvar)
        
        # Solve
        prob = cp.Problem(objective, constraints_list)
        prob.solve()
        
        if prob.status != cp.OPTIMAL:
            logger.warning(f"CVaR optimization failed: {prob.status}")
            weights = np.ones(self.n_assets) / self.n_assets
        else:
            weights = w.value
        
        return self._calculate_portfolio_metrics(weights)
    
    def _target_return(self, target_return: float, constraints: Dict) -> Dict:
        """Optimize for target return with minimum risk."""
        # Use CVXPY
        w = cp.Variable(self.n_assets)
        
        # Objective: minimize variance
        portfolio_variance = cp.quad_form(w, self.cov_matrix.values)
        objective = cp.Minimize(portfolio_variance)
        
        # Expected return
        expected_return = self.mean_returns.values @ w
        
        # Constraints
        constraints_list = [
            cp.sum(w) == constraints['leverage'],
            expected_return * self.periods_per_year >= target_return
        ]
        
        if constraints['long_only']:
            constraints_list.append(w >= constraints['min_weight'])
        
        if constraints['max_weight'] < 1:
            constraints_list.append(w <= constraints['max_weight'])
        
        # Solve
        prob = cp.Problem(objective, constraints_list)
        prob.solve()
        
        if prob.status != cp.OPTIMAL:
            logger.warning(f"Target return optimization failed: {prob.status}")
            weights = np.ones(self.n_assets) / self.n_assets
        else:
            weights = w.value
        
        return self._calculate_portfolio_metrics(weights)
    
    def _target_risk(self, target_risk: float, constraints: Dict) -> Dict:
        """Optimize for target risk with maximum return."""
        def neg_return(w):
            return -np.dot(w, self.mean_returns)
        
        def volatility_constraint(w):
            port_vol = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
            annual_vol = port_vol * np.sqrt(self.periods_per_year)
            return annual_vol - target_risk
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Bounds
        bounds = tuple((constraints['min_weight'], constraints['max_weight']) 
                      for _ in range(self.n_assets))
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - constraints['leverage']},
            {'type': 'eq', 'fun': volatility_constraint}
        ]
        
        # Optimize
        result = minimize(neg_return, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if not result.success:
            logger.warning(f"Target risk optimization failed: {result.message}")
            weights = x0
        else:
            weights = result.x
        
        return self._calculate_portfolio_metrics(weights)
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict:
        """Calculate comprehensive portfolio metrics."""
        # Portfolio return and volatility
        port_return = np.dot(weights, self.mean_returns)
        port_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        port_vol = np.sqrt(port_variance)
        
        # Annualized metrics
        annual_return = port_return * self.periods_per_year
        annual_vol = port_vol * np.sqrt(self.periods_per_year)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # Risk contributions
        marginal_contrib = np.dot(self.cov_matrix, weights) / port_vol if port_vol > 0 else np.zeros_like(weights)
        risk_contrib = weights * marginal_contrib
        
        # Diversification ratio
        avg_vol = np.dot(weights, np.sqrt(np.diag(self.cov_matrix)))
        div_ratio = avg_vol / port_vol if port_vol > 0 else 1
        
        # VaR and CVaR
        portfolio_returns = self.returns @ weights
        var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        
        # Maximum drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(self.periods_per_year)
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Create results dictionary
        results = {
            'weights': weights,
            'asset_names': list(self.returns.columns),
            'expected_return': annual_return,
            'volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var * np.sqrt(self.periods_per_year),
            'cvar_95': cvar * np.sqrt(self.periods_per_year),
            'diversification_ratio': div_ratio,
            'risk_contributions': risk_contrib,
            'effective_assets': 1 / np.sum(weights ** 2)  # Herfindahl index
        }
        
        return results
    
    def efficient_frontier(
        self,
        n_portfolios: int = 100,
        constraints: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier.
        
        Args:
            n_portfolios: Number of portfolios on frontier
            constraints: Portfolio constraints
            
        Returns:
            DataFrame with frontier portfolios
        """
        constraints = constraints or {}
        
        # Get min variance portfolio
        min_var = self.optimize(OptimizationObjective.MIN_VARIANCE, constraints)
        
        # Get max return portfolio (equal weight in highest returning assets)
        returns_rank = self.mean_returns.rank(ascending=False)
        n_top = min(3, self.n_assets)
        max_return_weights = np.zeros(self.n_assets)
        top_assets = returns_rank[returns_rank <= n_top].index
        max_return_weights[self.returns.columns.isin(top_assets)] = 1 / n_top
        max_return_metrics = self._calculate_portfolio_metrics(max_return_weights)
        
        # Target returns
        min_ret = min_var['expected_return']
        max_ret = max_return_metrics['expected_return']
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        # Calculate frontier
        frontier_data = []
        
        for target_ret in target_returns:
            try:
                result = self._target_return(target_ret, constraints)
                frontier_data.append({
                    'return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe': result['sharpe_ratio'],
                    'weights': result['weights']
                })
            except:
                continue
        
        return pd.DataFrame(frontier_data)
    
    def black_litterman(
        self,
        market_cap_weights: Dict[str, float],
        views: Dict[str, float],
        view_confidence: Dict[str, float],
        tau: float = 0.025
    ) -> Dict:
        """
        Black-Litterman portfolio optimization.
        
        Args:
            market_cap_weights: Market capitalization weights
            views: Dict of asset -> expected return views
            view_confidence: Dict of asset -> confidence in view
            tau: Scaling factor for prior covariance
            
        Returns:
            Optimized portfolio
        """
        # Ensure alignment
        assets = self.returns.columns.tolist()
        w_market = np.array([market_cap_weights.get(asset, 0) for asset in assets])
        
        # Implied equilibrium returns
        risk_aversion = (self.mean_returns @ w_market) / (w_market @ self.cov_matrix @ w_market)
        pi = risk_aversion * self.cov_matrix @ w_market
        
        # Views matrix
        P = np.zeros((len(views), self.n_assets))
        Q = np.zeros(len(views))
        omega_diag = []
        
        for i, (asset, view_return) in enumerate(views.items()):
            if asset in assets:
                asset_idx = assets.index(asset)
                P[i, asset_idx] = 1
                Q[i] = view_return / self.periods_per_year  # Convert to period return
                confidence = view_confidence.get(asset, 0.5)
                omega_diag.append((1 - confidence) * tau * self.cov_matrix.iloc[asset_idx, asset_idx])
        
        omega = np.diag(omega_diag)
        
        # Black-Litterman formula
        tau_sigma = tau * self.cov_matrix.values
        inv_tau_sigma = np.linalg.inv(tau_sigma)
        inv_omega = np.linalg.inv(omega)
        
        # Posterior mean
        mu_bl = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P) @ (inv_tau_sigma @ pi + P.T @ inv_omega @ Q)
        
        # Posterior covariance
        sigma_bl = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)
        
        # Update optimizer with BL estimates
        self.mean_returns = pd.Series(mu_bl, index=assets)
        self.cov_matrix = pd.DataFrame(sigma_bl, index=assets, columns=assets)
        
        # Optimize with new estimates
        return self.optimize(OptimizationObjective.MAX_SHARPE)