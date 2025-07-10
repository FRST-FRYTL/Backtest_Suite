"""
Risk Modeling Agent for ML Pipeline

Models and analyzes various risk factors for robust backtesting
and strategy development.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from sklearn.decomposition import PCA
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent


class RiskModelingAgent(BaseAgent):
    """
    Agent responsible for comprehensive risk modeling including:
    - Value at Risk (VaR) and Conditional VaR
    - Risk factor decomposition
    - Tail risk analysis
    - Correlation and covariance modeling
    - Stress testing and scenario analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RiskModelingAgent", config)
        self.risk_metrics = {}
        self.risk_factors = None
        self.covariance_matrix = None
        self.stress_scenarios = {}
        self.tail_risk_analysis = {}
        
    def initialize(self) -> bool:
        """Initialize risk modeling resources."""
        try:
            self.logger.info("Initializing Risk Modeling Agent")
            
            # Validate required configuration
            required_keys = ["risk_metrics", "confidence_levels", "lookback_period"]
            if not self.validate_config(required_keys):
                return False
            
            # Initialize risk settings
            self.risk_metrics_config = self.config.get("risk_metrics", [
                "var", "cvar", "sharpe", "sortino", "max_drawdown"
            ])
            self.confidence_levels = self.config.get("confidence_levels", [0.95, 0.99])
            self.lookback_period = self.config.get("lookback_period", 252)
            
            # Initialize risk model parameters
            self.risk_free_rate = self.config.get("risk_free_rate", 0.02)
            self.target_return = self.config.get("target_return", 0.0)
            
            # Initialize stress test scenarios
            self.stress_scenarios_config = self.config.get("stress_scenarios", {
                "market_crash": {"market": -0.20, "volatility": 2.0},
                "flash_crash": {"market": -0.10, "volatility": 3.0},
                "liquidity_crisis": {"market": -0.05, "volatility": 1.5}
            })
            
            self.logger.info("Risk Modeling Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False
    
    def execute(self, returns: pd.DataFrame, portfolio_weights: Optional[np.ndarray] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Execute risk modeling analysis.
        
        Args:
            returns: DataFrame of asset returns
            portfolio_weights: Optional portfolio weights
            
        Returns:
            Dict containing risk analysis results
        """
        try:
            # Prepare data
            if isinstance(returns, pd.Series):
                returns = returns.to_frame()
            
            # Calculate basic risk metrics
            basic_metrics = self._calculate_basic_risk_metrics(returns, portfolio_weights)
            
            # Calculate VaR and CVaR
            var_analysis = self._calculate_var_cvar(returns, portfolio_weights)
            
            # Perform risk factor analysis
            factor_analysis = self._analyze_risk_factors(returns)
            
            # Model covariance structure
            covariance_analysis = self._model_covariance(returns)
            
            # Analyze tail risks
            tail_analysis = self._analyze_tail_risks(returns, portfolio_weights)
            
            # Perform stress testing
            stress_results = self._perform_stress_tests(returns, portfolio_weights)
            
            # Calculate risk attribution
            risk_attribution = self._calculate_risk_attribution(
                returns, portfolio_weights
            )
            
            # Generate risk visualizations
            viz_results = self._generate_risk_visualizations(
                returns, portfolio_weights, var_analysis
            )
            
            # Compile results
            self.risk_metrics = {
                "basic_metrics": basic_metrics,
                "var_analysis": var_analysis,
                "factor_analysis": factor_analysis,
                "covariance_analysis": covariance_analysis,
                "tail_analysis": tail_analysis,
                "stress_testing": stress_results,
                "risk_attribution": risk_attribution
            }
            
            return {
                "risk_metrics": self.risk_metrics,
                "risk_summary": self._generate_risk_summary(),
                "recommendations": self._generate_risk_recommendations(),
                "visualizations": viz_results
            }
            
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            raise
    
    def _calculate_basic_risk_metrics(self, returns: pd.DataFrame,
                                    weights: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calculate basic risk metrics."""
        self.logger.info("Calculating basic risk metrics")
        
        # Portfolio returns if weights provided
        if weights is not None:
            portfolio_returns = (returns * weights).sum(axis=1)
        else:
            portfolio_returns = returns.mean(axis=1) if returns.shape[1] > 1 else returns.iloc[:, 0]
        
        metrics = {
            "volatility": {
                "daily": float(portfolio_returns.std()),
                "annual": float(portfolio_returns.std() * np.sqrt(252))
            },
            "returns": {
                "mean_daily": float(portfolio_returns.mean()),
                "mean_annual": float(portfolio_returns.mean() * 252)
            },
            "sharpe_ratio": float(
                (portfolio_returns.mean() - self.risk_free_rate/252) / 
                portfolio_returns.std() * np.sqrt(252)
            ) if portfolio_returns.std() > 0 else 0,
            "sortino_ratio": self._calculate_sortino_ratio(portfolio_returns),
            "max_drawdown": float(self._calculate_max_drawdown(portfolio_returns)),
            "calmar_ratio": self._calculate_calmar_ratio(portfolio_returns),
            "skewness": float(portfolio_returns.skew()),
            "kurtosis": float(portfolio_returns.kurtosis()),
            "jarque_bera": self._calculate_jarque_bera(portfolio_returns)
        }
        
        return metrics
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - self.target_return/252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_deviation = np.sqrt((downside_returns ** 2).mean())
        
        if downside_deviation == 0:
            return 0.0
        
        return float(excess_returns.mean() / downside_deviation * np.sqrt(252))
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        annual_return = returns.mean() * 252
        max_dd = abs(self._calculate_max_drawdown(returns))
        
        if max_dd == 0:
            return 0.0
        
        return float(annual_return / max_dd)
    
    def _calculate_jarque_bera(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate Jarque-Bera test for normality."""
        jb_stat, p_value = jarque_bera(returns.dropna())
        return {
            "statistic": float(jb_stat),
            "p_value": float(p_value),
            "is_normal": p_value > 0.05
        }
    
    def _calculate_var_cvar(self, returns: pd.DataFrame,
                          weights: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calculate Value at Risk and Conditional Value at Risk."""
        self.logger.info("Calculating VaR and CVaR")
        
        # Portfolio returns
        if weights is not None:
            portfolio_returns = (returns * weights).sum(axis=1)
        else:
            portfolio_returns = returns.mean(axis=1) if returns.shape[1] > 1 else returns.iloc[:, 0]
        
        var_results = {}
        
        for confidence in self.confidence_levels:
            # Historical VaR
            var_historical = np.percentile(portfolio_returns, (1 - confidence) * 100)
            
            # Conditional VaR (Expected Shortfall)
            cvar_historical = portfolio_returns[portfolio_returns <= var_historical].mean()
            
            # Parametric VaR (assuming normal distribution)
            var_parametric = (
                portfolio_returns.mean() - 
                stats.norm.ppf(confidence) * portfolio_returns.std()
            )
            
            # Parametric CVaR
            alpha = 1 - confidence
            pdf_alpha = stats.norm.pdf(stats.norm.ppf(alpha))
            cvar_parametric = (
                portfolio_returns.mean() - 
                portfolio_returns.std() * pdf_alpha / alpha
            )
            
            # Monte Carlo VaR
            var_mc, cvar_mc = self._monte_carlo_var(
                portfolio_returns.mean(),
                portfolio_returns.std(),
                confidence,
                n_simulations=10000
            )
            
            var_results[f"{int(confidence*100)}%"] = {
                "historical": {
                    "var_daily": float(var_historical),
                    "var_annual": float(var_historical * np.sqrt(252)),
                    "cvar_daily": float(cvar_historical),
                    "cvar_annual": float(cvar_historical * np.sqrt(252))
                },
                "parametric": {
                    "var_daily": float(var_parametric),
                    "var_annual": float(var_parametric * np.sqrt(252)),
                    "cvar_daily": float(cvar_parametric),
                    "cvar_annual": float(cvar_parametric * np.sqrt(252))
                },
                "monte_carlo": {
                    "var_daily": float(var_mc),
                    "var_annual": float(var_mc * np.sqrt(252)),
                    "cvar_daily": float(cvar_mc),
                    "cvar_annual": float(cvar_mc * np.sqrt(252))
                }
            }
        
        return var_results
    
    def _monte_carlo_var(self, mean: float, std: float, confidence: float,
                        n_simulations: int = 10000) -> Tuple[float, float]:
        """Calculate VaR using Monte Carlo simulation."""
        simulated_returns = np.random.normal(mean, std, n_simulations)
        var = np.percentile(simulated_returns, (1 - confidence) * 100)
        cvar = simulated_returns[simulated_returns <= var].mean()
        return var, cvar
    
    def _analyze_risk_factors(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Analyze risk factors using PCA."""
        self.logger.info("Analyzing risk factors")
        
        if returns.shape[1] < 2:
            return {"message": "Single asset, no factor analysis needed"}
        
        # Standardize returns
        returns_std = (returns - returns.mean()) / returns.std()
        
        # Perform PCA
        pca = PCA()
        pca.fit(returns_std.dropna())
        
        # Analyze components
        n_components_95 = np.argmax(pca.explained_variance_ratio_.cumsum() >= 0.95) + 1
        
        # Create factor loadings
        loadings = pd.DataFrame(
            pca.components_[:5],  # Top 5 factors
            columns=returns.columns,
            index=[f'Factor_{i+1}' for i in range(5)]
        )
        
        self.risk_factors = {
            "explained_variance": pca.explained_variance_ratio_[:5].tolist(),
            "cumulative_variance": pca.explained_variance_ratio_[:5].cumsum().tolist(),
            "n_factors_95_variance": int(n_components_95),
            "factor_loadings": loadings.to_dict(),
            "risk_concentration": float(pca.explained_variance_ratio_[0])
        }
        
        return self.risk_factors
    
    def _model_covariance(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Model covariance structure."""
        self.logger.info("Modeling covariance structure")
        
        # Empirical covariance
        emp_cov = EmpiricalCovariance()
        emp_cov.fit(returns.dropna())
        
        # Ledoit-Wolf shrunk covariance
        lw_cov = LedoitWolf()
        lw_cov.fit(returns.dropna())
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Analyze correlation structure
        corr_analysis = {
            "mean_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()),
            "max_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()),
            "min_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min())
        }
        
        self.covariance_matrix = lw_cov.covariance_
        
        return {
            "covariance_type": "Ledoit-Wolf Shrinkage",
            "shrinkage_constant": float(lw_cov.shrinkage_),
            "correlation_analysis": corr_analysis,
            "condition_number": float(np.linalg.cond(self.covariance_matrix))
        }
    
    def _analyze_tail_risks(self, returns: pd.DataFrame,
                          weights: Optional[np.ndarray]) -> Dict[str, Any]:
        """Analyze tail risk characteristics."""
        self.logger.info("Analyzing tail risks")
        
        # Portfolio returns
        if weights is not None:
            portfolio_returns = (returns * weights).sum(axis=1)
        else:
            portfolio_returns = returns.mean(axis=1) if returns.shape[1] > 1 else returns.iloc[:, 0]
        
        # Extreme value analysis
        threshold = np.percentile(np.abs(portfolio_returns), 95)
        extreme_returns = portfolio_returns[np.abs(portfolio_returns) > threshold]
        
        # Fit Generalized Pareto Distribution to tails
        left_tail = portfolio_returns[portfolio_returns < -threshold]
        right_tail = portfolio_returns[portfolio_returns > threshold]
        
        tail_analysis = {
            "extreme_events": {
                "count": len(extreme_returns),
                "percentage": float(len(extreme_returns) / len(portfolio_returns) * 100),
                "mean_extreme_loss": float(left_tail.mean()) if len(left_tail) > 0 else 0,
                "mean_extreme_gain": float(right_tail.mean()) if len(right_tail) > 0 else 0
            },
            "tail_index": self._estimate_tail_index(portfolio_returns),
            "expected_shortfall_ratio": self._calculate_expected_shortfall_ratio(portfolio_returns),
            "tail_dependence": self._calculate_tail_dependence(returns) if returns.shape[1] > 1 else None
        }
        
        self.tail_risk_analysis = tail_analysis
        return tail_analysis
    
    def _estimate_tail_index(self, returns: pd.Series) -> float:
        """Estimate tail index using Hill estimator."""
        sorted_returns = np.sort(np.abs(returns))[::-1]
        k = int(len(returns) * 0.1)  # Use top 10% for estimation
        
        if k < 10:
            return 0.0
        
        hill_estimator = k / np.sum(np.log(sorted_returns[:k] / sorted_returns[k]))
        return float(hill_estimator)
    
    def _calculate_expected_shortfall_ratio(self, returns: pd.Series) -> float:
        """Calculate ratio of expected shortfall to VaR."""
        var_95 = np.percentile(returns, 5)
        es_95 = returns[returns <= var_95].mean()
        
        if var_95 == 0:
            return 0.0
        
        return float(es_95 / var_95)
    
    def _calculate_tail_dependence(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate tail dependence coefficients."""
        if returns.shape[1] < 2:
            return {}
        
        # Simple tail dependence for first two assets
        asset1 = returns.iloc[:, 0]
        asset2 = returns.iloc[:, 1]
        
        # Lower tail dependence
        threshold = 0.05
        lower_tail = (asset1 <= asset1.quantile(threshold)) & (asset2 <= asset2.quantile(threshold))
        lower_tail_dep = lower_tail.sum() / (len(returns) * threshold)
        
        # Upper tail dependence
        upper_tail = (asset1 >= asset1.quantile(1-threshold)) & (asset2 >= asset2.quantile(1-threshold))
        upper_tail_dep = upper_tail.sum() / (len(returns) * threshold)
        
        return {
            "lower_tail_dependence": float(lower_tail_dep),
            "upper_tail_dependence": float(upper_tail_dep)
        }
    
    def _perform_stress_tests(self, returns: pd.DataFrame,
                            weights: Optional[np.ndarray]) -> Dict[str, Any]:
        """Perform stress testing scenarios."""
        self.logger.info("Performing stress tests")
        
        stress_results = {}
        
        # Portfolio returns
        if weights is not None:
            portfolio_returns = (returns * weights).sum(axis=1)
        else:
            portfolio_returns = returns.mean(axis=1) if returns.shape[1] > 1 else returns.iloc[:, 0]
        
        # Current portfolio metrics
        current_vol = portfolio_returns.std()
        current_return = portfolio_returns.mean()
        
        for scenario_name, scenario_params in self.stress_scenarios_config.items():
            # Apply stress scenario
            stressed_returns = portfolio_returns * (1 + scenario_params["market"])
            stressed_vol = current_vol * scenario_params["volatility"]
            
            # Calculate stressed metrics
            stressed_var_95 = np.percentile(stressed_returns, 5)
            stressed_cvar_95 = stressed_returns[stressed_returns <= stressed_var_95].mean()
            
            stress_results[scenario_name] = {
                "scenario": scenario_params,
                "impact": {
                    "return_impact": float(stressed_returns.mean() - current_return),
                    "volatility_impact": float(stressed_vol - current_vol),
                    "var_95_impact": float(stressed_var_95 - np.percentile(portfolio_returns, 5)),
                    "max_loss": float(stressed_returns.min())
                },
                "survival_probability": float((stressed_returns > -0.5).mean())
            }
        
        self.stress_scenarios = stress_results
        return stress_results
    
    def _calculate_risk_attribution(self, returns: pd.DataFrame,
                                  weights: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calculate risk attribution for portfolio."""
        self.logger.info("Calculating risk attribution")
        
        if returns.shape[1] == 1 or weights is None:
            return {"message": "Risk attribution requires multiple assets and weights"}
        
        # Calculate marginal VaR
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_vol = portfolio_returns.std()
        
        marginal_vars = []
        component_vars = []
        
        for i, asset in enumerate(returns.columns):
            # Marginal VaR
            asset_return = returns.iloc[:, i]
            correlation = asset_return.corr(portfolio_returns)
            marginal_var = correlation * asset_return.std() / portfolio_vol
            marginal_vars.append(marginal_var)
            
            # Component VaR
            component_var = weights[i] * marginal_var
            component_vars.append(component_var)
        
        # Normalize component VaR
        total_risk = sum(component_vars)
        risk_contributions = [cv / total_risk for cv in component_vars]
        
        return {
            "marginal_var": {
                asset: float(mv) for asset, mv in zip(returns.columns, marginal_vars)
            },
            "component_var": {
                asset: float(cv) for asset, cv in zip(returns.columns, component_vars)
            },
            "risk_contributions": {
                asset: float(rc) for asset, rc in zip(returns.columns, risk_contributions)
            },
            "diversification_ratio": float(
                sum(weights * returns.std()) / portfolio_vol
            ) if portfolio_vol > 0 else 1.0
        }
    
    def _generate_risk_visualizations(self, returns: pd.DataFrame,
                                    weights: Optional[np.ndarray],
                                    var_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate risk analysis visualizations."""
        self.logger.info("Generating risk visualizations")
        
        viz_paths = {}
        
        try:
            # Portfolio returns
            if weights is not None:
                portfolio_returns = (returns * weights).sum(axis=1)
            else:
                portfolio_returns = returns.mean(axis=1) if returns.shape[1] > 1 else returns.iloc[:, 0]
            
            # Risk metrics dashboard
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Returns distribution with VaR/CVaR
            ax = axes[0, 0]
            ax.hist(portfolio_returns, bins=50, alpha=0.7, density=True)
            
            # Add VaR and CVaR lines
            var_95 = var_analysis["95%"]["historical"]["var_daily"]
            cvar_95 = var_analysis["95%"]["historical"]["cvar_daily"]
            ax.axvline(var_95, color='red', linestyle='--', label=f'VaR 95%: {var_95:.3f}')
            ax.axvline(cvar_95, color='darkred', linestyle='--', label=f'CVaR 95%: {cvar_95:.3f}')
            
            # Fit normal distribution
            mu, sigma = portfolio_returns.mean(), portfolio_returns.std()
            x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'b-', label='Normal')
            
            ax.set_xlabel('Returns')
            ax.set_ylabel('Density')
            ax.set_title('Return Distribution with Risk Metrics')
            ax.legend()
            
            # Q-Q plot
            ax = axes[0, 1]
            stats.probplot(portfolio_returns, dist="norm", plot=ax)
            ax.set_title('Q-Q Plot (Normality Test)')
            
            # Rolling volatility
            ax = axes[1, 0]
            rolling_vol = portfolio_returns.rolling(window=21).std() * np.sqrt(252)
            rolling_vol.plot(ax=ax)
            ax.set_xlabel('Date')
            ax.set_ylabel('Annualized Volatility')
            ax.set_title('21-Day Rolling Volatility')
            ax.grid(True)
            
            # Drawdown chart
            ax = axes[1, 1]
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            drawdown.plot(ax=ax, color='red')
            ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown')
            ax.set_title('Portfolio Drawdown')
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig('/tmp/risk_dashboard.png')
            viz_paths["risk_dashboard"] = '/tmp/risk_dashboard.png'
            plt.close()
            
            # Correlation heatmap if multiple assets
            if returns.shape[1] > 1:
                plt.figure(figsize=(10, 8))
                corr_matrix = returns.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=1)
                plt.title('Asset Correlation Matrix')
                plt.tight_layout()
                plt.savefig('/tmp/correlation_matrix.png')
                viz_paths["correlation_matrix"] = '/tmp/correlation_matrix.png'
                plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to generate some visualizations: {str(e)}")
        
        return viz_paths
    
    def _generate_risk_summary(self) -> Dict[str, Any]:
        """Generate risk summary."""
        basic = self.risk_metrics.get("basic_metrics", {})
        var_95 = self.risk_metrics.get("var_analysis", {}).get("95%", {}).get("historical", {})
        
        return {
            "risk_level": self._assess_risk_level(),
            "key_metrics": {
                "annual_volatility": basic.get("volatility", {}).get("annual", 0),
                "sharpe_ratio": basic.get("sharpe_ratio", 0),
                "max_drawdown": basic.get("max_drawdown", 0),
                "var_95_daily": var_95.get("var_daily", 0)
            },
            "risk_characteristics": {
                "is_normal": basic.get("jarque_bera", {}).get("is_normal", False),
                "tail_risk": self.tail_risk_analysis.get("tail_index", 0) < 3,
                "high_correlation": self.risk_factors.get("risk_concentration", 0) > 0.5
            }
        }
    
    def _assess_risk_level(self) -> str:
        """Assess overall risk level."""
        vol = self.risk_metrics.get("basic_metrics", {}).get("volatility", {}).get("annual", 0)
        max_dd = abs(self.risk_metrics.get("basic_metrics", {}).get("max_drawdown", 0))
        
        if vol > 0.3 or max_dd > 0.2:
            return "High"
        elif vol > 0.15 or max_dd > 0.1:
            return "Medium"
        else:
            return "Low"
    
    def _generate_risk_recommendations(self) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        # Volatility recommendations
        vol = self.risk_metrics.get("basic_metrics", {}).get("volatility", {}).get("annual", 0)
        if vol > 0.25:
            recommendations.append("Consider reducing position sizes due to high volatility")
        
        # Drawdown recommendations
        max_dd = abs(self.risk_metrics.get("basic_metrics", {}).get("max_drawdown", 0))
        if max_dd > 0.15:
            recommendations.append("Implement stop-loss strategies to limit drawdowns")
        
        # Tail risk recommendations
        tail_index = self.tail_risk_analysis.get("tail_index", 3)
        if tail_index < 3:
            recommendations.append("Consider tail risk hedging strategies")
        
        # Correlation recommendations
        if self.risk_factors and self.risk_factors.get("risk_concentration", 0) > 0.6:
            recommendations.append("Diversify to reduce concentration risk")
        
        # Stress test recommendations
        for scenario, results in self.stress_scenarios.items():
            if results["survival_probability"] < 0.95:
                recommendations.append(f"Prepare contingency plans for {scenario} scenario")
        
        return recommendations