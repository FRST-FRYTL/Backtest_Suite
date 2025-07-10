"""
Performance Analysis Agent for ML Pipeline

Analyzes model and strategy performance with comprehensive metrics
and diagnostics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent


class PerformanceAnalysisAgent(BaseAgent):
    """
    Agent responsible for comprehensive performance analysis including:
    - Model performance metrics
    - Trading strategy performance
    - Attribution analysis
    - Performance stability analysis
    - Comparative benchmarking
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("PerformanceAnalysisAgent", config)
        self.performance_metrics = {}
        self.attribution_results = {}
        self.stability_analysis = {}
        self.benchmark_comparison = {}
        
    def initialize(self) -> bool:
        """Initialize performance analysis resources."""
        try:
            self.logger.info("Initializing Performance Analysis Agent")
            
            # Validate required configuration
            required_keys = ["metrics", "benchmark", "analysis_periods"]
            if not self.validate_config(required_keys):
                return False
            
            # Initialize analysis settings
            self.metrics_config = self.config.get("metrics", [
                "returns", "sharpe", "sortino", "calmar", "win_rate"
            ])
            self.benchmark = self.config.get("benchmark", "buy_and_hold")
            self.analysis_periods = self.config.get("analysis_periods", [
                "daily", "weekly", "monthly", "yearly"
            ])
            
            # Initialize performance thresholds
            self.performance_thresholds = self.config.get("thresholds", {
                "min_sharpe": 1.0,
                "min_win_rate": 0.5,
                "max_drawdown": 0.2
            })
            
            # Initialize transaction costs
            self.transaction_costs = self.config.get("transaction_costs", {
                "commission": 0.001,  # 0.1%
                "slippage": 0.0005   # 0.05%
            })
            
            self.logger.info("Performance Analysis Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False
    
    def execute(self, predictions: Union[pd.Series, np.ndarray],
                actual: Union[pd.Series, np.ndarray],
                prices: Optional[pd.DataFrame] = None,
                positions: Optional[pd.Series] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Execute performance analysis.
        
        Args:
            predictions: Model predictions
            actual: Actual values/returns
            prices: Price data for strategy performance
            positions: Trading positions/signals
            
        Returns:
            Dict containing performance analysis results
        """
        try:
            # Analyze model performance
            model_performance = self._analyze_model_performance(
                predictions, actual, kwargs.get("task_type", "regression")
            )
            
            # Analyze trading performance if positions provided
            if positions is not None and prices is not None:
                trading_performance = self._analyze_trading_performance(
                    prices, positions
                )
            else:
                trading_performance = None
            
            # Perform attribution analysis
            attribution = self._perform_attribution_analysis(
                predictions, actual, prices, positions
            )
            
            # Analyze performance stability
            stability = self._analyze_performance_stability(
                predictions, actual, prices, positions
            )
            
            # Benchmark comparison
            benchmark = self._compare_to_benchmark(
                predictions, actual, prices, positions
            )
            
            # Performance by period
            period_analysis = self._analyze_by_period(
                predictions, actual, prices, positions
            )
            
            # Generate visualizations
            viz_results = self._generate_performance_visualizations(
                predictions, actual, prices, positions
            )
            
            # Compile results
            self.performance_metrics = {
                "model_performance": model_performance,
                "trading_performance": trading_performance,
                "attribution": attribution,
                "stability": stability,
                "benchmark_comparison": benchmark,
                "period_analysis": period_analysis
            }
            
            return {
                "performance_metrics": self.performance_metrics,
                "summary": self._generate_performance_summary(),
                "recommendations": self._generate_performance_recommendations(),
                "visualizations": viz_results
            }
            
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            raise
    
    def _analyze_model_performance(self, predictions: Union[pd.Series, np.ndarray],
                                 actual: Union[pd.Series, np.ndarray],
                                 task_type: str = "regression") -> Dict[str, Any]:
        """Analyze model prediction performance."""
        self.logger.info("Analyzing model performance")
        
        # Convert to numpy arrays
        y_pred = np.array(predictions)
        y_true = np.array(actual)
        
        if task_type == "regression":
            metrics = {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
                "directional_accuracy": float(
                    np.mean(np.sign(y_pred) == np.sign(y_true))
                )
            }
            
            # Additional regression diagnostics
            residuals = y_true - y_pred
            metrics["residual_analysis"] = {
                "mean": float(residuals.mean()),
                "std": float(residuals.std()),
                "skewness": float(stats.skew(residuals)),
                "kurtosis": float(stats.kurtosis(residuals)),
                "autocorrelation": float(
                    pd.Series(residuals).autocorr() if len(residuals) > 1 else 0
                )
            }
            
        else:  # classification
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average='weighted')),
                "recall": float(recall_score(y_true, y_pred, average='weighted')),
                "f1": float(f1_score(y_true, y_pred, average='weighted'))
            }
            
            # ROC AUC for binary classification
            if len(np.unique(y_true)) == 2:
                if hasattr(predictions, 'iloc'):
                    # If predictions are probabilities
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred))
                
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            
            # Classification report
            metrics["classification_report"] = classification_report(
                y_true, y_pred, output_dict=True
            )
        
        return metrics
    
    def _analyze_trading_performance(self, prices: pd.DataFrame,
                                   positions: pd.Series) -> Dict[str, Any]:
        """Analyze trading strategy performance."""
        self.logger.info("Analyzing trading performance")
        
        # Calculate returns
        price_returns = prices['close'].pct_change() if 'close' in prices.columns else prices.pct_change()
        strategy_returns = positions.shift(1) * price_returns
        
        # Apply transaction costs
        trades = positions.diff().abs()
        transaction_costs = trades * (
            self.transaction_costs["commission"] + 
            self.transaction_costs["slippage"]
        )
        net_returns = strategy_returns - transaction_costs
        
        # Calculate cumulative returns
        cum_returns = (1 + net_returns).cumprod()
        
        # Performance metrics
        metrics = {
            "total_return": float((cum_returns.iloc[-1] - 1)),
            "annual_return": float(net_returns.mean() * 252),
            "annual_volatility": float(net_returns.std() * np.sqrt(252)),
            "sharpe_ratio": float(
                net_returns.mean() / net_returns.std() * np.sqrt(252)
            ) if net_returns.std() > 0 else 0,
            "sortino_ratio": self._calculate_sortino_ratio(net_returns),
            "calmar_ratio": self._calculate_calmar_ratio(net_returns),
            "max_drawdown": float(self._calculate_max_drawdown(cum_returns)),
            "win_rate": float((net_returns > 0).mean()),
            "profit_factor": self._calculate_profit_factor(net_returns),
            "number_of_trades": int(trades.sum() / 2),
            "average_trade_return": float(
                net_returns[trades != 0].mean()
            ) if (trades != 0).any() else 0,
            "best_trade": float(net_returns.max()),
            "worst_trade": float(net_returns.min()),
            "total_transaction_costs": float(transaction_costs.sum())
        }
        
        # Risk metrics
        metrics["risk_metrics"] = {
            "var_95": float(np.percentile(net_returns, 5)),
            "cvar_95": float(net_returns[net_returns <= np.percentile(net_returns, 5)].mean()),
            "downside_deviation": float(net_returns[net_returns < 0].std() * np.sqrt(252)),
            "upside_deviation": float(net_returns[net_returns > 0].std() * np.sqrt(252))
        }
        
        return metrics
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_deviation = np.sqrt((downside_returns ** 2).mean())
        
        if downside_deviation == 0:
            return 0.0
        
        return float(excess_returns.mean() / downside_deviation * np.sqrt(252))
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        annual_return = returns.mean() * 252
        cum_returns = (1 + returns).cumprod()
        max_dd = abs(self._calculate_max_drawdown(cum_returns))
        
        if max_dd == 0:
            return 0.0
        
        return float(annual_return / max_dd)
    
    def _calculate_max_drawdown(self, cum_returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return float(drawdown.min())
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor."""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return float('inf') if gains > 0 else 0.0
        
        return float(gains / losses)
    
    def _perform_attribution_analysis(self, predictions: Union[pd.Series, np.ndarray],
                                    actual: Union[pd.Series, np.ndarray],
                                    prices: Optional[pd.DataFrame],
                                    positions: Optional[pd.Series]) -> Dict[str, Any]:
        """Perform performance attribution analysis."""
        self.logger.info("Performing attribution analysis")
        
        if positions is None or prices is None:
            return {"message": "Attribution requires positions and prices"}
        
        price_returns = prices['close'].pct_change() if 'close' in prices.columns else prices.pct_change()
        strategy_returns = positions.shift(1) * price_returns
        
        # Timing vs selection attribution
        # Timing: how well we time entries/exits
        # Selection: how well we predict direction
        
        # Calculate components
        timing_component = (positions.shift(1) - positions.shift(1).mean()) * price_returns
        selection_component = positions.shift(1).mean() * price_returns
        
        attribution = {
            "timing": {
                "contribution": float(timing_component.sum()),
                "percentage": float(
                    timing_component.sum() / strategy_returns.sum() * 100
                ) if strategy_returns.sum() != 0 else 0
            },
            "selection": {
                "contribution": float(selection_component.sum()),
                "percentage": float(
                    selection_component.sum() / strategy_returns.sum() * 100
                ) if strategy_returns.sum() != 0 else 0
            }
        }
        
        # Skill metrics
        if isinstance(predictions, (pd.Series, np.ndarray)) and len(predictions) > 0:
            # Information coefficient (IC)
            ic = pd.Series(predictions).corr(pd.Series(actual))
            
            # Hit rate
            if len(predictions) == len(positions):
                correct_direction = (
                    (predictions > 0) & (actual > 0) |
                    (predictions < 0) & (actual < 0)
                )
                hit_rate = correct_direction.mean()
            else:
                hit_rate = 0.5
            
            attribution["skill_metrics"] = {
                "information_coefficient": float(ic) if not np.isnan(ic) else 0,
                "hit_rate": float(hit_rate),
                "information_ratio": float(
                    strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                ) if strategy_returns.std() > 0 else 0
            }
        
        self.attribution_results = attribution
        return attribution
    
    def _analyze_performance_stability(self, predictions: Union[pd.Series, np.ndarray],
                                     actual: Union[pd.Series, np.ndarray],
                                     prices: Optional[pd.DataFrame],
                                     positions: Optional[pd.Series]) -> Dict[str, Any]:
        """Analyze performance stability over time."""
        self.logger.info("Analyzing performance stability")
        
        stability_results = {}
        
        # Model stability
        if len(predictions) > 100:
            # Rolling accuracy/error
            window = 50
            rolling_errors = []
            
            for i in range(window, len(predictions)):
                window_pred = predictions[i-window:i]
                window_actual = actual[i-window:i]
                
                if isinstance(actual[0], (int, np.integer)):  # Classification
                    accuracy = accuracy_score(window_actual, window_pred)
                    rolling_errors.append(1 - accuracy)
                else:  # Regression
                    mse = mean_squared_error(window_actual, window_pred)
                    rolling_errors.append(mse)
            
            stability_results["model_stability"] = {
                "error_mean": float(np.mean(rolling_errors)),
                "error_std": float(np.std(rolling_errors)),
                "error_trend": float(np.polyfit(range(len(rolling_errors)), rolling_errors, 1)[0]),
                "coefficient_of_variation": float(
                    np.std(rolling_errors) / np.mean(rolling_errors)
                ) if np.mean(rolling_errors) > 0 else 0
            }
        
        # Trading stability
        if positions is not None and prices is not None:
            price_returns = prices['close'].pct_change() if 'close' in prices.columns else prices.pct_change()
            strategy_returns = positions.shift(1) * price_returns
            
            # Rolling Sharpe ratio
            rolling_sharpe = strategy_returns.rolling(window=252).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
            )
            
            # Drawdown analysis
            cum_returns = (1 + strategy_returns).cumprod()
            drawdowns = self._calculate_rolling_drawdowns(cum_returns)
            
            stability_results["trading_stability"] = {
                "sharpe_stability": {
                    "mean": float(rolling_sharpe.mean()),
                    "std": float(rolling_sharpe.std()),
                    "min": float(rolling_sharpe.min()),
                    "max": float(rolling_sharpe.max())
                },
                "drawdown_analysis": {
                    "avg_drawdown": float(drawdowns.mean()),
                    "drawdown_frequency": float((drawdowns < -0.05).mean()),
                    "recovery_time": self._calculate_avg_recovery_time(drawdowns)
                },
                "consistency": {
                    "positive_months": float(
                        strategy_returns.resample('M').sum().apply(lambda x: x > 0).mean()
                    ),
                    "longest_winning_streak": self._calculate_longest_streak(strategy_returns, True),
                    "longest_losing_streak": self._calculate_longest_streak(strategy_returns, False)
                }
            }
        
        self.stability_analysis = stability_results
        return stability_results
    
    def _calculate_rolling_drawdowns(self, cum_returns: pd.Series) -> pd.Series:
        """Calculate rolling drawdowns."""
        running_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - running_max) / running_max
        return drawdowns
    
    def _calculate_avg_recovery_time(self, drawdowns: pd.Series) -> float:
        """Calculate average recovery time from drawdowns."""
        in_drawdown = drawdowns < 0
        drawdown_periods = []
        
        start = None
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start is None:
                start = i
            elif not is_dd and start is not None:
                drawdown_periods.append(i - start)
                start = None
        
        if drawdown_periods:
            return float(np.mean(drawdown_periods))
        return 0.0
    
    def _calculate_longest_streak(self, returns: pd.Series, positive: bool) -> int:
        """Calculate longest winning or losing streak."""
        if positive:
            condition = returns > 0
        else:
            condition = returns < 0
        
        streaks = []
        current_streak = 0
        
        for is_condition in condition:
            if is_condition:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        return max(streaks) if streaks else 0
    
    def _compare_to_benchmark(self, predictions: Union[pd.Series, np.ndarray],
                            actual: Union[pd.Series, np.ndarray],
                            prices: Optional[pd.DataFrame],
                            positions: Optional[pd.Series]) -> Dict[str, Any]:
        """Compare performance to benchmark."""
        self.logger.info("Comparing to benchmark")
        
        if prices is None:
            return {"message": "Benchmark comparison requires price data"}
        
        price_returns = prices['close'].pct_change() if 'close' in prices.columns else prices.pct_change()
        
        # Calculate benchmark returns (buy and hold)
        benchmark_returns = price_returns
        benchmark_cum_returns = (1 + benchmark_returns).cumprod()
        
        # Calculate strategy returns if positions provided
        if positions is not None:
            strategy_returns = positions.shift(1) * price_returns
            strategy_cum_returns = (1 + strategy_returns).cumprod()
        else:
            # Use predictions as signals
            pred_signals = pd.Series(predictions) > 0
            strategy_returns = pred_signals.shift(1) * price_returns
            strategy_cum_returns = (1 + strategy_returns).cumprod()
        
        comparison = {
            "returns": {
                "strategy": float((strategy_cum_returns.iloc[-1] - 1)),
                "benchmark": float((benchmark_cum_returns.iloc[-1] - 1)),
                "excess": float((strategy_cum_returns.iloc[-1] - benchmark_cum_returns.iloc[-1]))
            },
            "risk_adjusted": {
                "strategy_sharpe": float(
                    strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                ) if strategy_returns.std() > 0 else 0,
                "benchmark_sharpe": float(
                    benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)
                ) if benchmark_returns.std() > 0 else 0,
                "information_ratio": float(
                    (strategy_returns - benchmark_returns).mean() /
                    (strategy_returns - benchmark_returns).std() * np.sqrt(252)
                ) if (strategy_returns - benchmark_returns).std() > 0 else 0
            },
            "risk": {
                "strategy_vol": float(strategy_returns.std() * np.sqrt(252)),
                "benchmark_vol": float(benchmark_returns.std() * np.sqrt(252)),
                "tracking_error": float((strategy_returns - benchmark_returns).std() * np.sqrt(252))
            },
            "relative_metrics": {
                "alpha": float((strategy_returns - benchmark_returns).mean() * 252),
                "beta": float(strategy_returns.cov(benchmark_returns) / benchmark_returns.var())
                    if benchmark_returns.var() > 0 else 1.0,
                "correlation": float(strategy_returns.corr(benchmark_returns))
            }
        }
        
        self.benchmark_comparison = comparison
        return comparison
    
    def _analyze_by_period(self, predictions: Union[pd.Series, np.ndarray],
                         actual: Union[pd.Series, np.ndarray],
                         prices: Optional[pd.DataFrame],
                         positions: Optional[pd.Series]) -> Dict[str, Any]:
        """Analyze performance by different time periods."""
        self.logger.info("Analyzing performance by period")
        
        if prices is None or positions is None:
            return {"message": "Period analysis requires prices and positions"}
        
        price_returns = prices['close'].pct_change() if 'close' in prices.columns else prices.pct_change()
        strategy_returns = positions.shift(1) * price_returns
        
        # Convert to series with datetime index if needed
        if not isinstance(strategy_returns.index, pd.DatetimeIndex):
            strategy_returns = pd.Series(
                strategy_returns.values,
                index=pd.date_range(start='2020-01-01', periods=len(strategy_returns), freq='D')
            )
        
        period_results = {}
        
        # Analyze different periods
        periods = {
            "daily": strategy_returns,
            "weekly": strategy_returns.resample('W').sum(),
            "monthly": strategy_returns.resample('M').sum(),
            "quarterly": strategy_returns.resample('Q').sum(),
            "yearly": strategy_returns.resample('Y').sum()
        }
        
        for period_name, period_returns in periods.items():
            if len(period_returns) > 0:
                period_results[period_name] = {
                    "mean_return": float(period_returns.mean()),
                    "volatility": float(period_returns.std()),
                    "sharpe_ratio": float(
                        period_returns.mean() / period_returns.std() * np.sqrt(len(period_returns))
                    ) if period_returns.std() > 0 else 0,
                    "win_rate": float((period_returns > 0).mean()),
                    "best_period": float(period_returns.max()),
                    "worst_period": float(period_returns.min()),
                    "period_count": len(period_returns)
                }
        
        return period_results
    
    def _generate_performance_visualizations(self, predictions: Union[pd.Series, np.ndarray],
                                           actual: Union[pd.Series, np.ndarray],
                                           prices: Optional[pd.DataFrame],
                                           positions: Optional[pd.Series]) -> Dict[str, str]:
        """Generate performance visualizations."""
        self.logger.info("Generating performance visualizations")
        
        viz_paths = {}
        
        try:
            # Model performance plots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Predictions vs Actual
            ax = axes[0, 0]
            ax.scatter(actual, predictions, alpha=0.5, s=10)
            ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 
                   'r--', label='Perfect Prediction')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Predictions vs Actual')
            ax.legend()
            
            # Residuals
            ax = axes[0, 1]
            residuals = actual - predictions
            ax.scatter(predictions, residuals, alpha=0.5, s=10)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Residuals')
            ax.set_title('Residual Plot')
            
            # Trading performance if available
            if prices is not None and positions is not None:
                price_returns = prices['close'].pct_change() if 'close' in prices.columns else prices.pct_change()
                strategy_returns = positions.shift(1) * price_returns
                
                # Cumulative returns
                ax = axes[1, 0]
                strategy_cum = (1 + strategy_returns).cumprod()
                benchmark_cum = (1 + price_returns).cumprod()
                
                strategy_cum.plot(ax=ax, label='Strategy', linewidth=2)
                benchmark_cum.plot(ax=ax, label='Buy & Hold', linewidth=2, alpha=0.7)
                ax.set_xlabel('Time')
                ax.set_ylabel('Cumulative Returns')
                ax.set_title('Strategy Performance')
                ax.legend()
                ax.grid(True)
                
                # Monthly returns heatmap
                ax = axes[1, 1]
                if isinstance(strategy_returns.index, pd.DatetimeIndex):
                    monthly_returns = strategy_returns.resample('M').sum()
                    monthly_returns_pivot = pd.DataFrame({
                        'Year': monthly_returns.index.year,
                        'Month': monthly_returns.index.month,
                        'Return': monthly_returns.values
                    }).pivot(index='Month', columns='Year', values='Return')
                    
                    sns.heatmap(monthly_returns_pivot, annot=True, fmt='.2%', 
                               cmap='RdYlGn', center=0, ax=ax)
                    ax.set_title('Monthly Returns Heatmap')
                else:
                    ax.text(0.5, 0.5, 'Datetime index required for monthly heatmap',
                           ha='center', va='center', transform=ax.transAxes)
            
            plt.tight_layout()
            plt.savefig('/tmp/performance_analysis.png')
            viz_paths["performance_analysis"] = '/tmp/performance_analysis.png'
            plt.close()
            
            # Additional plots for trading performance
            if prices is not None and positions is not None:
                plt.figure(figsize=(14, 8))
                
                # Drawdown chart
                plt.subplot(2, 1, 1)
                cum_returns = (1 + strategy_returns).cumprod()
                running_max = cum_returns.expanding().max()
                drawdown = (cum_returns - running_max) / running_max
                
                drawdown.plot(color='red', linewidth=1)
                plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
                plt.title('Strategy Drawdown')
                plt.ylabel('Drawdown')
                plt.grid(True)
                
                # Rolling metrics
                plt.subplot(2, 1, 2)
                rolling_sharpe = strategy_returns.rolling(window=252).apply(
                    lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
                )
                rolling_sharpe.plot(label='Rolling Sharpe Ratio', linewidth=2)
                plt.axhline(y=1, color='r', linestyle='--', label='Sharpe = 1')
                plt.title('Rolling Performance Metrics')
                plt.ylabel('Sharpe Ratio')
                plt.xlabel('Time')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig('/tmp/trading_performance.png')
                viz_paths["trading_performance"] = '/tmp/trading_performance.png'
                plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to generate some visualizations: {str(e)}")
        
        return viz_paths
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        model_perf = self.performance_metrics.get("model_performance", {})
        trading_perf = self.performance_metrics.get("trading_performance", {})
        
        summary = {
            "model_quality": self._assess_model_quality(model_perf),
            "trading_quality": self._assess_trading_quality(trading_perf),
            "key_strengths": self._identify_strengths(),
            "key_weaknesses": self._identify_weaknesses(),
            "overall_rating": self._calculate_overall_rating()
        }
        
        return summary
    
    def _assess_model_quality(self, metrics: Dict[str, Any]) -> str:
        """Assess model quality based on metrics."""
        if "r2" in metrics:
            r2 = metrics["r2"]
            if r2 > 0.8:
                return "Excellent"
            elif r2 > 0.6:
                return "Good"
            elif r2 > 0.4:
                return "Fair"
            else:
                return "Poor"
        elif "accuracy" in metrics:
            acc = metrics["accuracy"]
            if acc > 0.9:
                return "Excellent"
            elif acc > 0.8:
                return "Good"
            elif acc > 0.7:
                return "Fair"
            else:
                return "Poor"
        return "Unknown"
    
    def _assess_trading_quality(self, metrics: Dict[str, Any]) -> str:
        """Assess trading strategy quality."""
        if not metrics:
            return "Not Available"
        
        sharpe = metrics.get("sharpe_ratio", 0)
        win_rate = metrics.get("win_rate", 0)
        
        if sharpe > 2 and win_rate > 0.6:
            return "Excellent"
        elif sharpe > 1 and win_rate > 0.5:
            return "Good"
        elif sharpe > 0.5:
            return "Fair"
        else:
            return "Poor"
    
    def _identify_strengths(self) -> List[str]:
        """Identify performance strengths."""
        strengths = []
        
        model_perf = self.performance_metrics.get("model_performance", {})
        trading_perf = self.performance_metrics.get("trading_performance", {})
        
        # Model strengths
        if model_perf.get("r2", 0) > 0.7:
            strengths.append("High model explanatory power")
        if model_perf.get("directional_accuracy", 0) > 0.6:
            strengths.append("Good directional prediction accuracy")
        
        # Trading strengths
        if trading_perf:
            if trading_perf.get("sharpe_ratio", 0) > 1.5:
                strengths.append("Strong risk-adjusted returns")
            if trading_perf.get("win_rate", 0) > 0.55:
                strengths.append("Consistent win rate")
            if abs(trading_perf.get("max_drawdown", 0)) < 0.1:
                strengths.append("Limited drawdowns")
        
        return strengths
    
    def _identify_weaknesses(self) -> List[str]:
        """Identify performance weaknesses."""
        weaknesses = []
        
        model_perf = self.performance_metrics.get("model_performance", {})
        trading_perf = self.performance_metrics.get("trading_performance", {})
        stability = self.performance_metrics.get("stability", {})
        
        # Model weaknesses
        if model_perf.get("r2", 1) < 0.3:
            weaknesses.append("Low model predictive power")
        
        residuals = model_perf.get("residual_analysis", {})
        if abs(residuals.get("skewness", 0)) > 1:
            weaknesses.append("Skewed prediction errors")
        
        # Trading weaknesses
        if trading_perf:
            if trading_perf.get("sharpe_ratio", 0) < 0.5:
                weaknesses.append("Poor risk-adjusted returns")
            if abs(trading_perf.get("max_drawdown", 0)) > 0.2:
                weaknesses.append("Large drawdowns")
        
        # Stability weaknesses
        if stability.get("model_stability", {}).get("coefficient_of_variation", 0) > 0.5:
            weaknesses.append("Unstable model performance")
        
        return weaknesses
    
    def _calculate_overall_rating(self) -> float:
        """Calculate overall performance rating (0-100)."""
        rating = 50  # Base rating
        
        model_perf = self.performance_metrics.get("model_performance", {})
        trading_perf = self.performance_metrics.get("trading_performance", {})
        
        # Model performance contribution
        if "r2" in model_perf:
            rating += model_perf["r2"] * 20
        elif "accuracy" in model_perf:
            rating += (model_perf["accuracy"] - 0.5) * 40
        
        # Trading performance contribution
        if trading_perf:
            sharpe = trading_perf.get("sharpe_ratio", 0)
            rating += min(sharpe * 10, 20)
            
            win_rate = trading_perf.get("win_rate", 0.5)
            rating += (win_rate - 0.5) * 20
            
            max_dd = abs(trading_perf.get("max_drawdown", 0))
            rating -= max_dd * 50
        
        return float(np.clip(rating, 0, 100))
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        model_perf = self.performance_metrics.get("model_performance", {})
        trading_perf = self.performance_metrics.get("trading_performance", {})
        stability = self.performance_metrics.get("stability", {})
        
        # Model recommendations
        if model_perf.get("r2", 1) < 0.5:
            recommendations.append("Consider feature engineering to improve model predictive power")
        
        if model_perf.get("directional_accuracy", 0) < 0.55:
            recommendations.append("Focus on improving directional prediction accuracy")
        
        # Trading recommendations
        if trading_perf:
            if trading_perf.get("sharpe_ratio", 0) < 1:
                recommendations.append("Optimize position sizing to improve risk-adjusted returns")
            
            if trading_perf.get("number_of_trades", 0) > 1000:
                recommendations.append("Consider reducing trading frequency to minimize costs")
            
            if abs(trading_perf.get("max_drawdown", 0)) > 0.15:
                recommendations.append("Implement risk management rules to limit drawdowns")
        
        # Stability recommendations
        if stability.get("model_stability", {}).get("error_trend", 0) > 0:
            recommendations.append("Model performance degrading over time - consider retraining")
        
        return recommendations