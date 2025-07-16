"""
Report Sections for Standardized Reporting

This module defines all standard report sections with consistent formatting
and calculation methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportSection(ABC):
    """Base class for all report sections"""
    
    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__.lower()
        
    @abstractmethod
    def generate(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the section content"""
        pass
    
    def format_number(self, value: float, format_type: str = "general") -> str:
        """Format numbers consistently across the report"""
        if pd.isna(value):
            return "N/A"
            
        formats = {
            "general": "{:,.2f}",
            "percentage": "{:.2%}",
            "currency": "${:,.2f}",
            "ratio": "{:.2f}",
            "integer": "{:,.0f}"
        }
        
        return formats.get(format_type, "{:.2f}").format(value)


class ExecutiveSummary(ReportSection):
    """Executive summary section providing high-level overview"""
    
    def generate(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary"""
        metrics = backtest_results.get("metrics", {})
        trades = backtest_results.get("trades", pd.DataFrame())
        
        # Calculate key highlights
        total_return = metrics.get("total_return", 0)
        sharpe_ratio = metrics.get("sharpe_ratio", 0)
        max_drawdown = metrics.get("max_drawdown", 0)
        win_rate = metrics.get("win_rate", 0)
        
        # Determine strategy assessment
        assessment = self._assess_strategy(metrics)
        
        return {
            "key_metrics": {
                "Total Return": self.format_number(total_return, "percentage"),
                "Sharpe Ratio": self.format_number(sharpe_ratio, "ratio"),
                "Maximum Drawdown": self.format_number(abs(max_drawdown), "percentage"),
                "Win Rate": self.format_number(win_rate, "percentage"),
                "Total Trades": self.format_number(len(trades), "integer")
            },
            "performance_summary": self._generate_performance_summary(metrics),
            "risk_summary": self._generate_risk_summary(metrics),
            "strategy_assessment": assessment,
            "key_findings": self._extract_key_findings(backtest_results),
            "recommendations": self._generate_recommendations(metrics, assessment)
        }
    
    def _assess_strategy(self, metrics: Dict[str, Any]) -> str:
        """Assess overall strategy performance"""
        score = 0
        max_score = 5
        
        # Sharpe ratio assessment
        sharpe = metrics.get("sharpe_ratio", 0)
        if sharpe >= 2.0:
            score += 1
        elif sharpe >= 1.0:
            score += 0.5
            
        # Drawdown assessment
        max_dd = abs(metrics.get("max_drawdown", 0))
        if max_dd <= 0.10:
            score += 1
        elif max_dd <= 0.20:
            score += 0.5
            
        # Win rate assessment
        win_rate = metrics.get("win_rate", 0)
        if win_rate >= 0.60:
            score += 1
        elif win_rate >= 0.50:
            score += 0.5
            
        # Profit factor assessment
        profit_factor = metrics.get("profit_factor", 0)
        if profit_factor >= 2.0:
            score += 1
        elif profit_factor >= 1.5:
            score += 0.5
            
        # Consistency assessment (based on rolling metrics)
        consistency_score = metrics.get("consistency_score", 0.5)
        score += consistency_score
        
        # Convert score to assessment
        percentage = (score / max_score) * 100
        
        if percentage >= 80:
            return "Excellent - Strategy shows strong performance across all metrics"
        elif percentage >= 60:
            return "Good - Strategy performs well with minor areas for improvement"
        elif percentage >= 40:
            return "Acceptable - Strategy is viable but requires optimization"
        else:
            return "Needs Improvement - Strategy shows significant weaknesses"
    
    def _generate_performance_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate narrative performance summary"""
        total_return = metrics.get("total_return", 0)
        annual_return = metrics.get("annual_return", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        
        summary = f"The strategy generated a total return of {total_return:.1%} "
        summary += f"with an annualized return of {annual_return:.1%}. "
        
        if sharpe >= 1.5:
            summary += f"The Sharpe ratio of {sharpe:.2f} indicates excellent risk-adjusted returns. "
        elif sharpe >= 1.0:
            summary += f"The Sharpe ratio of {sharpe:.2f} indicates good risk-adjusted returns. "
        else:
            summary += f"The Sharpe ratio of {sharpe:.2f} suggests room for improvement in risk-adjusted returns. "
            
        return summary
    
    def _generate_risk_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate narrative risk summary"""
        max_dd = abs(metrics.get("max_drawdown", 0))
        vol = metrics.get("volatility", 0)
        var_95 = metrics.get("var_95", 0)
        
        summary = f"Risk analysis shows a maximum drawdown of {max_dd:.1%} "
        summary += f"with annualized volatility of {vol:.1%}. "
        summary += f"The 95% Value at Risk is {abs(var_95):.1%}, "
        
        if max_dd <= 0.15:
            summary += "indicating well-controlled downside risk."
        elif max_dd <= 0.25:
            summary += "indicating moderate downside risk."
        else:
            summary += "indicating significant downside risk that requires attention."
            
        return summary
    
    def _extract_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """Extract key findings from results"""
        findings = []
        metrics = results.get("metrics", {})
        
        # Performance findings
        if metrics.get("sharpe_ratio", 0) >= 1.5:
            findings.append("Strategy demonstrates strong risk-adjusted performance")
            
        if metrics.get("win_rate", 0) >= 0.60:
            findings.append("High win rate indicates effective trade selection")
            
        # Risk findings
        if abs(metrics.get("max_drawdown", 0)) <= 0.15:
            findings.append("Drawdowns remain within acceptable limits")
            
        # Consistency findings
        if metrics.get("consistency_score", 0) >= 0.7:
            findings.append("Performance shows good consistency over time")
            
        return findings
    
    def _generate_recommendations(self, metrics: Dict[str, Any], assessment: str) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on Sharpe ratio
        if metrics.get("sharpe_ratio", 0) < 1.0:
            recommendations.append("Consider parameter optimization to improve risk-adjusted returns")
            
        # Based on drawdown
        if abs(metrics.get("max_drawdown", 0)) > 0.20:
            recommendations.append("Implement additional risk controls to reduce maximum drawdown")
            
        # Based on win rate
        if metrics.get("win_rate", 0) < 0.50:
            recommendations.append("Review entry criteria to improve trade selection accuracy")
            
        # Based on profit factor
        if metrics.get("profit_factor", 0) < 1.5:
            recommendations.append("Optimize exit strategies to improve profit factor")
            
        return recommendations


class PerformanceAnalysis(ReportSection):
    """Detailed performance analysis section"""
    
    def generate(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance analysis"""
        equity_curve = backtest_results.get("equity_curve", pd.Series())
        metrics = backtest_results.get("metrics", {})
        trades = backtest_results.get("trades", pd.DataFrame())
        
        return {
            "return_analysis": self._analyze_returns(equity_curve, metrics),
            "risk_adjusted_metrics": self._calculate_risk_adjusted_metrics(metrics),
            "benchmark_comparison": self._compare_to_benchmark(backtest_results),
            "performance_attribution": self._attribute_performance(trades, equity_curve),
            "rolling_performance": self._analyze_rolling_performance(equity_curve),
            "statistical_significance": self._test_statistical_significance(equity_curve)
        }
    
    def _analyze_returns(self, equity_curve: pd.Series, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze return characteristics"""
        returns = equity_curve.pct_change().dropna()
        
        return {
            "summary_statistics": {
                "Total Return": self.format_number(metrics.get("total_return", 0), "percentage"),
                "Annualized Return": self.format_number(metrics.get("annual_return", 0), "percentage"),
                "Daily Mean Return": self.format_number(returns.mean(), "percentage"),
                "Daily Volatility": self.format_number(returns.std(), "percentage"),
                "Annualized Volatility": self.format_number(returns.std() * np.sqrt(252), "percentage")
            },
            "distribution_analysis": {
                "Skewness": self.format_number(returns.skew(), "ratio"),
                "Kurtosis": self.format_number(returns.kurtosis(), "ratio"),
                "Best Day": self.format_number(returns.max(), "percentage"),
                "Worst Day": self.format_number(returns.min(), "percentage"),
                "Positive Days": self.format_number((returns > 0).mean(), "percentage")
            },
            "return_periods": self._analyze_return_periods(returns)
        }
    
    def _calculate_risk_adjusted_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk-adjusted metrics"""
        return {
            "sharpe_ratio": {
                "value": self.format_number(metrics.get("sharpe_ratio", 0), "ratio"),
                "interpretation": self._interpret_sharpe(metrics.get("sharpe_ratio", 0))
            },
            "sortino_ratio": {
                "value": self.format_number(metrics.get("sortino_ratio", 0), "ratio"),
                "interpretation": self._interpret_sortino(metrics.get("sortino_ratio", 0))
            },
            "calmar_ratio": {
                "value": self.format_number(metrics.get("calmar_ratio", 0), "ratio"),
                "interpretation": self._interpret_calmar(metrics.get("calmar_ratio", 0))
            },
            "information_ratio": {
                "value": self.format_number(metrics.get("information_ratio", 0), "ratio"),
                "interpretation": self._interpret_information_ratio(metrics.get("information_ratio", 0))
            }
        }
    
    def _compare_to_benchmark(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance to benchmark"""
        benchmark_data = results.get("benchmark", {})
        strategy_return = results["metrics"].get("total_return", 0)
        benchmark_return = benchmark_data.get("total_return", 0)
        
        return {
            "excess_return": self.format_number(strategy_return - benchmark_return, "percentage"),
            "tracking_error": self.format_number(benchmark_data.get("tracking_error", 0), "percentage"),
            "information_ratio": self.format_number(benchmark_data.get("information_ratio", 0), "ratio"),
            "beta": self.format_number(benchmark_data.get("beta", 1.0), "ratio"),
            "alpha": self.format_number(benchmark_data.get("alpha", 0), "percentage"),
            "correlation": self.format_number(benchmark_data.get("correlation", 0), "ratio")
        }
    
    def _attribute_performance(self, trades: pd.DataFrame, equity_curve: pd.Series) -> Dict[str, Any]:
        """Attribute performance to various factors"""
        if trades.empty:
            return {"message": "No trades available for attribution analysis"}
            
        # Long vs Short attribution
        long_trades = trades[trades["side"] == "long"]
        short_trades = trades[trades["side"] == "short"]
        
        attribution = {
            "by_direction": {
                "long_contribution": self.format_number(
                    long_trades["pnl"].sum() / equity_curve.iloc[-1] if not long_trades.empty else 0,
                    "percentage"
                ),
                "short_contribution": self.format_number(
                    short_trades["pnl"].sum() / equity_curve.iloc[-1] if not short_trades.empty else 0,
                    "percentage"
                )
            }
        }
        
        # Time-based attribution
        if "entry_time" in trades.columns:
            trades["hour"] = pd.to_datetime(trades["entry_time"]).dt.hour
            hourly_pnl = trades.groupby("hour")["pnl"].sum()
            attribution["by_hour"] = {
                f"{hour:02d}:00": self.format_number(pnl / equity_curve.iloc[-1], "percentage")
                for hour, pnl in hourly_pnl.items()
            }
            
        return attribution
    
    def _analyze_rolling_performance(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Analyze rolling performance metrics"""
        returns = equity_curve.pct_change().dropna()
        
        # Calculate rolling metrics
        rolling_windows = [30, 60, 90, 180, 252]
        rolling_metrics = {}
        
        for window in rolling_windows:
            if len(returns) >= window:
                rolling_returns = returns.rolling(window).mean() * 252
                rolling_vol = returns.rolling(window).std() * np.sqrt(252)
                rolling_sharpe = rolling_returns / rolling_vol
                
                rolling_metrics[f"{window}_day"] = {
                    "avg_return": self.format_number(rolling_returns.mean(), "percentage"),
                    "avg_volatility": self.format_number(rolling_vol.mean(), "percentage"),
                    "avg_sharpe": self.format_number(rolling_sharpe.mean(), "ratio"),
                    "consistency": self.format_number((rolling_sharpe > 0).mean(), "percentage")
                }
                
        return rolling_metrics
    
    def _test_statistical_significance(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Test statistical significance of returns"""
        returns = equity_curve.pct_change().dropna()
        
        # T-test for mean return different from zero
        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        
        # Test for normality
        _, normality_p = stats.normaltest(returns)
        
        return {
            "mean_return_test": {
                "t_statistic": self.format_number(t_stat, "ratio"),
                "p_value": self.format_number(p_value, "ratio"),
                "significant": p_value < 0.05,
                "interpretation": "Returns are statistically significant" if p_value < 0.05 
                                else "Returns are not statistically significant"
            },
            "normality_test": {
                "p_value": self.format_number(normality_p, "ratio"),
                "is_normal": normality_p > 0.05,
                "interpretation": "Returns follow normal distribution" if normality_p > 0.05
                                else "Returns do not follow normal distribution"
            }
        }
    
    def _analyze_return_periods(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze returns by different time periods"""
        periods = {}
        
        # Monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        periods["monthly"] = {
            "average": self.format_number(monthly_returns.mean(), "percentage"),
            "best": self.format_number(monthly_returns.max(), "percentage"),
            "worst": self.format_number(monthly_returns.min(), "percentage"),
            "positive_months": self.format_number((monthly_returns > 0).mean(), "percentage")
        }
        
        # Quarterly returns
        quarterly_returns = returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)
        if len(quarterly_returns) > 0:
            periods["quarterly"] = {
                "average": self.format_number(quarterly_returns.mean(), "percentage"),
                "best": self.format_number(quarterly_returns.max(), "percentage"),
                "worst": self.format_number(quarterly_returns.min(), "percentage"),
                "positive_quarters": self.format_number((quarterly_returns > 0).mean(), "percentage")
            }
            
        # Annual returns
        annual_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        if len(annual_returns) > 0:
            periods["annual"] = {
                "average": self.format_number(annual_returns.mean(), "percentage"),
                "best": self.format_number(annual_returns.max(), "percentage"),
                "worst": self.format_number(annual_returns.min(), "percentage"),
                "positive_years": self.format_number((annual_returns > 0).mean(), "percentage")
            }
            
        return periods
    
    def _interpret_sharpe(self, sharpe: float) -> str:
        """Interpret Sharpe ratio value"""
        if sharpe >= 2.0:
            return "Excellent - Very strong risk-adjusted returns"
        elif sharpe >= 1.5:
            return "Very Good - Strong risk-adjusted returns"
        elif sharpe >= 1.0:
            return "Good - Acceptable risk-adjusted returns"
        elif sharpe >= 0.5:
            return "Adequate - Marginal risk-adjusted returns"
        else:
            return "Poor - Insufficient risk-adjusted returns"
    
    def _interpret_sortino(self, sortino: float) -> str:
        """Interpret Sortino ratio value"""
        if sortino >= 2.0:
            return "Excellent - Very strong downside risk management"
        elif sortino >= 1.5:
            return "Very Good - Strong downside risk management"
        elif sortino >= 1.0:
            return "Good - Acceptable downside risk management"
        else:
            return "Needs Improvement - High downside risk"
    
    def _interpret_calmar(self, calmar: float) -> str:
        """Interpret Calmar ratio value"""
        if calmar >= 3.0:
            return "Excellent - Outstanding return/drawdown ratio"
        elif calmar >= 2.0:
            return "Very Good - Strong return/drawdown ratio"
        elif calmar >= 1.0:
            return "Good - Acceptable return/drawdown ratio"
        else:
            return "Poor - Low return relative to drawdown"
    
    def _interpret_information_ratio(self, ir: float) -> str:
        """Interpret Information ratio value"""
        if ir >= 1.0:
            return "Excellent - Strong active management"
        elif ir >= 0.5:
            return "Good - Positive active management"
        elif ir >= 0:
            return "Neutral - Marginal value add"
        else:
            return "Poor - Negative active management"


class RiskAnalysis(ReportSection):
    """Comprehensive risk analysis section"""
    
    def generate(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk analysis"""
        equity_curve = backtest_results.get("equity_curve", pd.Series())
        metrics = backtest_results.get("metrics", {})
        trades = backtest_results.get("trades", pd.DataFrame())
        
        return {
            "drawdown_analysis": self._analyze_drawdowns(equity_curve),
            "volatility_analysis": self._analyze_volatility(equity_curve),
            "var_analysis": self._analyze_value_at_risk(equity_curve),
            "stress_testing": self._perform_stress_tests(equity_curve, trades),
            "risk_metrics": self._calculate_risk_metrics(metrics),
            "risk_decomposition": self._decompose_risk(equity_curve, trades)
        }
    
    def _analyze_drawdowns(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Analyze drawdown characteristics"""
        # Calculate drawdown series
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        
        # Find drawdown periods
        is_drawdown = drawdown < 0
        drawdown_starts = (~is_drawdown).shift(1) & is_drawdown
        drawdown_ends = is_drawdown.shift(1) & (~is_drawdown)
        
        # Calculate drawdown statistics
        max_drawdown = drawdown.min()
        
        # Duration analysis
        drawdown_periods = []
        current_start = None
        
        for idx in drawdown.index:
            if drawdown_starts.loc[idx] and current_start is None:
                current_start = idx
            elif drawdown_ends.loc[idx] and current_start is not None:
                duration = (idx - current_start).days if hasattr(idx - current_start, 'days') else len(drawdown.loc[current_start:idx])
                depth = drawdown.loc[current_start:idx].min()
                drawdown_periods.append({
                    "start": current_start,
                    "end": idx,
                    "duration": duration,
                    "depth": depth
                })
                current_start = None
        
        # Sort by depth to get top drawdowns
        drawdown_periods.sort(key=lambda x: x["depth"])
        top_5_drawdowns = drawdown_periods[:5]
        
        return {
            "maximum_drawdown": {
                "value": self.format_number(abs(max_drawdown), "percentage"),
                "date": str(drawdown.idxmin()) if not drawdown.empty else "N/A"
            },
            "average_drawdown": self.format_number(abs(drawdown[drawdown < 0].mean()) if any(drawdown < 0) else 0, "percentage"),
            "drawdown_duration": {
                "current": self._calculate_current_drawdown_duration(drawdown),
                "average": self._calculate_average_drawdown_duration(drawdown_periods),
                "maximum": max([d["duration"] for d in drawdown_periods]) if drawdown_periods else 0
            },
            "recovery_analysis": self._analyze_recovery_times(drawdown_periods),
            "top_drawdowns": [
                {
                    "rank": i + 1,
                    "depth": self.format_number(abs(dd["depth"]), "percentage"),
                    "duration": f"{dd['duration']} days",
                    "period": f"{dd['start']} to {dd['end']}"
                }
                for i, dd in enumerate(top_5_drawdowns)
            ]
        }
    
    def _analyze_volatility(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Analyze volatility patterns"""
        returns = equity_curve.pct_change().dropna()
        
        # Calculate various volatility measures
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Rolling volatility
        rolling_vols = {
            "30_day": returns.rolling(30).std() * np.sqrt(252),
            "60_day": returns.rolling(60).std() * np.sqrt(252),
            "90_day": returns.rolling(90).std() * np.sqrt(252)
        }
        
        # Volatility regime analysis
        vol_regimes = self._identify_volatility_regimes(returns)
        
        return {
            "current_volatility": {
                "daily": self.format_number(daily_vol, "percentage"),
                "annualized": self.format_number(annual_vol, "percentage")
            },
            "volatility_percentiles": {
                "10th": self.format_number(returns.std() * np.sqrt(252) * 0.1, "percentage"),
                "25th": self.format_number(returns.std() * np.sqrt(252) * 0.25, "percentage"),
                "50th": self.format_number(returns.std() * np.sqrt(252) * 0.5, "percentage"),
                "75th": self.format_number(returns.std() * np.sqrt(252) * 0.75, "percentage"),
                "90th": self.format_number(returns.std() * np.sqrt(252) * 0.9, "percentage")
            },
            "rolling_volatility_stats": {
                period: {
                    "current": self.format_number(vol.iloc[-1] if not vol.empty else 0, "percentage"),
                    "average": self.format_number(vol.mean(), "percentage"),
                    "max": self.format_number(vol.max(), "percentage"),
                    "min": self.format_number(vol.min(), "percentage")
                }
                for period, vol in rolling_vols.items() if len(vol) > 0
            },
            "volatility_regimes": vol_regimes
        }
    
    def _analyze_value_at_risk(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculate Value at Risk metrics"""
        returns = equity_curve.pct_change().dropna()
        
        # Parametric VaR (assuming normal distribution)
        confidence_levels = [0.95, 0.99]
        var_results = {}
        
        for confidence in confidence_levels:
            z_score = stats.norm.ppf(1 - confidence)
            var_parametric = returns.mean() + z_score * returns.std()
            
            # Historical VaR
            var_historical = returns.quantile(1 - confidence)
            
            # CVaR (Conditional VaR)
            cvar = returns[returns <= var_historical].mean()
            
            var_results[f"{int(confidence*100)}%"] = {
                "parametric_var": self.format_number(abs(var_parametric), "percentage"),
                "historical_var": self.format_number(abs(var_historical), "percentage"),
                "cvar": self.format_number(abs(cvar), "percentage"),
                "interpretation": self._interpret_var(var_historical, confidence)
            }
            
        return var_results
    
    def _perform_stress_tests(self, equity_curve: pd.Series, trades: pd.DataFrame) -> Dict[str, Any]:
        """Perform stress testing scenarios"""
        returns = equity_curve.pct_change().dropna()
        
        stress_scenarios = {
            "market_crash": {
                "description": "20% market decline",
                "impact": self._simulate_scenario(returns, -0.20),
                "recovery_time": self._estimate_recovery_time(0.20)
            },
            "volatility_spike": {
                "description": "3x volatility increase",
                "impact": self._simulate_volatility_scenario(returns, 3.0),
                "effect_on_sharpe": self._calculate_sharpe_impact(returns, 3.0)
            },
            "extended_drawdown": {
                "description": "6-month losing streak",
                "impact": self._simulate_losing_streak(returns, 126),
                "survival_probability": self._calculate_survival_probability(returns, 126)
            },
            "correlation_breakdown": {
                "description": "Strategy correlation reversal",
                "impact": "Analysis requires market data",
                "recommendation": "Monitor correlation stability"
            }
        }
        
        return stress_scenarios
    
    def _calculate_risk_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        return {
            "downside_risk": {
                "downside_deviation": self.format_number(metrics.get("downside_deviation", 0), "percentage"),
                "sortino_ratio": self.format_number(metrics.get("sortino_ratio", 0), "ratio"),
                "omega_ratio": self.format_number(metrics.get("omega_ratio", 1), "ratio"),
                "gain_to_pain_ratio": self.format_number(metrics.get("gain_to_pain_ratio", 0), "ratio")
            },
            "tail_risk": {
                "maximum_loss": self.format_number(metrics.get("worst_trade", 0), "percentage"),
                "tail_ratio": self.format_number(metrics.get("tail_ratio", 1), "ratio"),
                "common_sense_ratio": self.format_number(metrics.get("common_sense_ratio", 1), "ratio")
            },
            "risk_efficiency": {
                "return_over_max_dd": self.format_number(
                    metrics.get("annual_return", 0) / abs(metrics.get("max_drawdown", 1)),
                    "ratio"
                ),
                "profit_to_max_dd": self.format_number(
                    metrics.get("total_return", 0) / abs(metrics.get("max_drawdown", 1)),
                    "ratio"
                ),
                "risk_adjusted_return": self.format_number(
                    metrics.get("annual_return", 0) / metrics.get("volatility", 1),
                    "ratio"
                )
            }
        }
    
    def _decompose_risk(self, equity_curve: pd.Series, trades: pd.DataFrame) -> Dict[str, Any]:
        """Decompose risk by various factors"""
        returns = equity_curve.pct_change().dropna()
        
        decomposition = {
            "by_time": self._decompose_risk_by_time(returns),
            "by_trade_size": self._decompose_risk_by_trade_size(trades) if not trades.empty else {},
            "by_market_condition": self._decompose_risk_by_market_condition(returns),
            "concentration_risk": self._analyze_concentration_risk(trades) if not trades.empty else {}
        }
        
        return decomposition
    
    # Helper methods
    def _calculate_current_drawdown_duration(self, drawdown: pd.Series) -> str:
        """Calculate duration of current drawdown"""
        if drawdown.iloc[-1] >= 0:
            return "Not in drawdown"
            
        # Find start of current drawdown
        for i in range(len(drawdown) - 1, -1, -1):
            if drawdown.iloc[i] >= 0:
                start_idx = i + 1
                duration = len(drawdown) - start_idx
                return f"{duration} periods"
                
        return f"{len(drawdown)} periods"
    
    def _calculate_average_drawdown_duration(self, drawdown_periods: List[Dict]) -> str:
        """Calculate average drawdown duration"""
        if not drawdown_periods:
            return "0 days"
            
        avg_duration = np.mean([d["duration"] for d in drawdown_periods])
        return f"{avg_duration:.0f} days"
    
    def _analyze_recovery_times(self, drawdown_periods: List[Dict]) -> Dict[str, Any]:
        """Analyze recovery times from drawdowns"""
        if not drawdown_periods:
            return {"message": "No completed drawdown periods"}
            
        recovery_times = [d["duration"] for d in drawdown_periods]
        
        return {
            "average_recovery": f"{np.mean(recovery_times):.0f} days",
            "median_recovery": f"{np.median(recovery_times):.0f} days",
            "longest_recovery": f"{np.max(recovery_times):.0f} days",
            "recovery_rate": self.format_number(
                len([r for r in recovery_times if r <= 30]) / len(recovery_times),
                "percentage"
            ) + " within 30 days"
        }
    
    def _identify_volatility_regimes(self, returns: pd.Series) -> Dict[str, Any]:
        """Identify volatility regimes"""
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()
        
        # Define regimes
        low_vol_threshold = vol_mean - vol_std
        high_vol_threshold = vol_mean + vol_std
        
        current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else 0
        
        if current_vol < low_vol_threshold:
            current_regime = "Low Volatility"
        elif current_vol > high_vol_threshold:
            current_regime = "High Volatility"
        else:
            current_regime = "Normal Volatility"
            
        return {
            "current_regime": current_regime,
            "regime_thresholds": {
                "low": self.format_number(low_vol_threshold, "percentage"),
                "high": self.format_number(high_vol_threshold, "percentage")
            },
            "time_in_regimes": {
                "low": self.format_number((rolling_vol < low_vol_threshold).mean(), "percentage"),
                "normal": self.format_number(
                    ((rolling_vol >= low_vol_threshold) & (rolling_vol <= high_vol_threshold)).mean(),
                    "percentage"
                ),
                "high": self.format_number((rolling_vol > high_vol_threshold).mean(), "percentage")
            }
        }
    
    def _interpret_var(self, var: float, confidence: float) -> str:
        """Interpret VaR value"""
        var_abs = abs(var)
        confidence_pct = int(confidence * 100)
        
        if var_abs < 0.02:
            return f"Low risk - {confidence_pct}% probability of losing less than {var_abs:.1%}"
        elif var_abs < 0.05:
            return f"Moderate risk - {confidence_pct}% probability of losing less than {var_abs:.1%}"
        else:
            return f"High risk - {confidence_pct}% probability of losing less than {var_abs:.1%}"
    
    def _simulate_scenario(self, returns: pd.Series, shock: float) -> str:
        """Simulate market shock scenario"""
        shocked_equity = (1 + returns).cumprod() * (1 + shock)
        new_drawdown = (shocked_equity.iloc[-1] - shocked_equity.max()) / shocked_equity.max()
        return f"{new_drawdown:.1%} drawdown from peak"
    
    def _simulate_volatility_scenario(self, returns: pd.Series, vol_multiplier: float) -> str:
        """Simulate volatility change scenario"""
        new_vol = returns.std() * vol_multiplier * np.sqrt(252)
        return f"Annualized volatility increases to {new_vol:.1%}"
    
    def _calculate_sharpe_impact(self, returns: pd.Series, vol_multiplier: float) -> str:
        """Calculate impact on Sharpe ratio from volatility change"""
        current_sharpe = returns.mean() / returns.std() * np.sqrt(252)
        new_sharpe = current_sharpe / vol_multiplier
        return f"Sharpe ratio decreases from {current_sharpe:.2f} to {new_sharpe:.2f}"
    
    def _simulate_losing_streak(self, returns: pd.Series, periods: int) -> str:
        """Simulate extended losing streak"""
        avg_loss = returns[returns < 0].mean() if any(returns < 0) else -0.01
        total_loss = (1 + avg_loss) ** periods - 1
        return f"Estimated {total_loss:.1%} loss over {periods} trading days"
    
    def _calculate_survival_probability(self, returns: pd.Series, periods: int) -> str:
        """Calculate probability of surviving losing streak"""
        # Simplified calculation based on historical win rate
        win_rate = (returns > 0).mean()
        survival_prob = 1 - (1 - win_rate) ** periods
        return f"{survival_prob:.1%} probability of at least one winning day"
    
    def _decompose_risk_by_time(self, returns: pd.Series) -> Dict[str, Any]:
        """Decompose risk by time periods"""
        # by hour of day (if intraday data)
        if hasattr(returns.index, 'hour'):
            hourly_vol = returns.groupby(returns.index.hour).std() * np.sqrt(252)
            return {
                "by_hour": {
                    f"{hour:02d}:00": self.format_number(vol, "percentage")
                    for hour, vol in hourly_vol.items()
                }
            }
        
        # by day of week
        if hasattr(returns.index, 'dayofweek'):
            daily_vol = returns.groupby(returns.index.dayofweek).std() * np.sqrt(252)
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            return {
                "by_day": {
                    days[day]: self.format_number(vol, "percentage")
                    for day, vol in daily_vol.items() if day < 5
                }
            }
            
        return {"message": "Insufficient data for time-based decomposition"}
    
    def _decompose_risk_by_trade_size(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Decompose risk by trade size"""
        if "size" not in trades.columns or "pnl" not in trades.columns:
            return {"message": "Trade size data not available"}
            
        # Create size buckets
        trades["size_bucket"] = pd.qcut(trades["size"], q=4, labels=["Small", "Medium", "Large", "XLarge"])
        
        risk_by_size = {}
        for bucket in ["Small", "Medium", "Large", "XLarge"]:
            bucket_trades = trades[trades["size_bucket"] == bucket]
            if not bucket_trades.empty:
                risk_by_size[bucket] = {
                    "avg_risk": self.format_number(bucket_trades["pnl"].std(), "currency"),
                    "max_loss": self.format_number(bucket_trades["pnl"].min(), "currency"),
                    "loss_frequency": self.format_number((bucket_trades["pnl"] < 0).mean(), "percentage")
                }
                
        return risk_by_size
    
    def _decompose_risk_by_market_condition(self, returns: pd.Series) -> Dict[str, Any]:
        """Decompose risk by market conditions"""
        # Use rolling correlation with general market direction
        market_direction = returns.rolling(20).mean()
        
        trending_up = market_direction > market_direction.std()
        trending_down = market_direction < -market_direction.std()
        ranging = ~trending_up & ~trending_down
        
        return {
            "trending_up": {
                "volatility": self.format_number(returns[trending_up].std() * np.sqrt(252), "percentage"),
                "frequency": self.format_number(trending_up.mean(), "percentage")
            },
            "trending_down": {
                "volatility": self.format_number(returns[trending_down].std() * np.sqrt(252), "percentage"),
                "frequency": self.format_number(trending_down.mean(), "percentage")
            },
            "ranging": {
                "volatility": self.format_number(returns[ranging].std() * np.sqrt(252), "percentage"),
                "frequency": self.format_number(ranging.mean(), "percentage")
            }
        }
    
    def _analyze_concentration_risk(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze concentration risk in trading"""
        if trades.empty or "pnl" not in trades.columns:
            return {"message": "No trade data available"}
            
        # Calculate concentration metrics
        total_pnl = trades["pnl"].sum()
        sorted_trades = trades.sort_values("pnl", ascending=False)
        
        # Top 10% concentration
        top_10_pct_count = max(1, int(len(trades) * 0.1))
        top_10_pct_pnl = sorted_trades.head(top_10_pct_count)["pnl"].sum()
        
        # Largest trade impact
        largest_win = trades["pnl"].max()
        largest_loss = trades["pnl"].min()
        
        return {
            "top_10_percent_impact": self.format_number(top_10_pct_pnl / total_pnl, "percentage"),
            "largest_win_impact": self.format_number(largest_win / total_pnl, "percentage"),
            "largest_loss_impact": self.format_number(abs(largest_loss) / total_pnl, "percentage"),
            "concentration_warning": "High concentration risk" if abs(top_10_pct_pnl / total_pnl) > 0.5 else "Acceptable concentration"
        }


class TradeAnalysis(ReportSection):
    """Detailed trade analysis section"""
    
    def generate(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trade analysis"""
        trades = backtest_results.get("trades", pd.DataFrame())
        
        if trades.empty:
            return {"message": "No trades executed during backtest period"}
            
        return {
            "trade_statistics": self._calculate_trade_statistics(trades),
            "win_loss_analysis": self._analyze_win_loss(trades),
            "trade_duration_analysis": self._analyze_trade_duration(trades),
            "trade_distribution": self._analyze_trade_distribution(trades),
            "entry_exit_analysis": self._analyze_entry_exit(trades),
            "trade_clustering": self._analyze_trade_clustering(trades),
            "price_analysis": self._analyze_trade_prices(trades),
            "risk_analysis": self._analyze_trade_risk(trades),
            "trade_table": self._create_enhanced_trade_table(trades)
        }
    
    def _calculate_trade_statistics(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive trade statistics"""
        return {
            "summary": {
                "total_trades": len(trades),
                "avg_trades_per_day": self.format_number(self._calculate_trades_per_day(trades), "ratio"),
                "avg_position_size": self.format_number(trades["size"].mean() if "size" in trades else 0, "currency"),
                "total_volume": self.format_number(trades["size"].sum() if "size" in trades else 0, "currency")
            },
            "profitability": {
                "total_pnl": self.format_number(trades["pnl"].sum(), "currency"),
                "avg_pnl_per_trade": self.format_number(trades["pnl"].mean(), "currency"),
                "median_pnl": self.format_number(trades["pnl"].median(), "currency"),
                "pnl_std_dev": self.format_number(trades["pnl"].std(), "currency")
            },
            "efficiency": {
                "profit_factor": self.format_number(self._calculate_profit_factor(trades), "ratio"),
                "expectancy": self.format_number(self._calculate_expectancy(trades), "currency"),
                "edge_ratio": self.format_number(self._calculate_edge_ratio(trades), "ratio")
            }
        }
    
    def _analyze_win_loss(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze winning and losing trades"""
        winners = trades[trades["pnl"] > 0]
        losers = trades[trades["pnl"] < 0]
        
        return {
            "win_rate": self.format_number(len(winners) / len(trades), "percentage"),
            "winning_trades": {
                "count": len(winners),
                "avg_win": self.format_number(winners["pnl"].mean() if not winners.empty else 0, "currency"),
                "median_win": self.format_number(winners["pnl"].median() if not winners.empty else 0, "currency"),
                "largest_win": self.format_number(winners["pnl"].max() if not winners.empty else 0, "currency"),
                "avg_duration": self._calculate_avg_duration(winners)
            },
            "losing_trades": {
                "count": len(losers),
                "avg_loss": self.format_number(losers["pnl"].mean() if not losers.empty else 0, "currency"),
                "median_loss": self.format_number(losers["pnl"].median() if not losers.empty else 0, "currency"),
                "largest_loss": self.format_number(losers["pnl"].min() if not losers.empty else 0, "currency"),
                "avg_duration": self._calculate_avg_duration(losers)
            },
            "ratios": {
                "win_loss_ratio": self.format_number(
                    abs(winners["pnl"].mean() / losers["pnl"].mean()) if not losers.empty and losers["pnl"].mean() != 0 else 0,
                    "ratio"
                ),
                "avg_win_to_avg_loss": self.format_number(
                    winners["pnl"].mean() / abs(losers["pnl"].mean()) if not losers.empty and losers["pnl"].mean() != 0 else 0,
                    "ratio"
                )
            },
            "consecutive_analysis": self._analyze_consecutive_trades(trades)
        }
    
    def _analyze_trade_duration(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trade holding periods"""
        if "duration" not in trades.columns and "entry_time" in trades.columns and "exit_time" in trades.columns:
            trades["duration"] = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 3600  # hours
            
        if "duration" not in trades.columns:
            return {"message": "Trade duration data not available"}
            
        return {
            "overall_statistics": {
                "avg_duration": f"{trades['duration'].mean():.1f} hours",
                "median_duration": f"{trades['duration'].median():.1f} hours",
                "min_duration": f"{trades['duration'].min():.1f} hours",
                "max_duration": f"{trades['duration'].max():.1f} hours"
            },
            "duration_by_outcome": {
                "winning_trades": f"{trades[trades['pnl'] > 0]['duration'].mean():.1f} hours",
                "losing_trades": f"{trades[trades['pnl'] < 0]['duration'].mean():.1f} hours"
            },
            "duration_buckets": self._create_duration_buckets(trades),
            "optimal_holding_period": self._find_optimal_holding_period(trades)
        }
    
    def _analyze_trade_distribution(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distribution of trades"""
        return {
            "pnl_distribution": {
                "skewness": self.format_number(trades["pnl"].skew(), "ratio"),
                "kurtosis": self.format_number(trades["pnl"].kurtosis(), "ratio"),
                "percentiles": {
                    "5th": self.format_number(trades["pnl"].quantile(0.05), "currency"),
                    "25th": self.format_number(trades["pnl"].quantile(0.25), "currency"),
                    "50th": self.format_number(trades["pnl"].quantile(0.50), "currency"),
                    "75th": self.format_number(trades["pnl"].quantile(0.75), "currency"),
                    "95th": self.format_number(trades["pnl"].quantile(0.95), "currency")
                }
            },
            "monthly_distribution": self._analyze_monthly_distribution(trades),
            "day_of_week_distribution": self._analyze_day_of_week_distribution(trades)
        }
    
    def _analyze_entry_exit(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze entry and exit characteristics"""
        analysis = {}
        
        # Entry analysis
        if "entry_reason" in trades.columns:
            entry_performance = trades.groupby("entry_reason").agg({
                "pnl": ["count", "sum", "mean"],
                "duration": "mean" if "duration" in trades.columns else lambda x: 0
            })
            
            analysis["entry_reasons"] = {
                reason: {
                    "count": int(stats["pnl"]["count"]),
                    "total_pnl": self.format_number(stats["pnl"]["sum"], "currency"),
                    "avg_pnl": self.format_number(stats["pnl"]["mean"], "currency"),
                    "win_rate": self.format_number(
                        (trades[trades["entry_reason"] == reason]["pnl"] > 0).mean(),
                        "percentage"
                    )
                }
                for reason, stats in entry_performance.iterrows()
            }
            
        # Exit analysis
        if "exit_reason" in trades.columns:
            exit_performance = trades.groupby("exit_reason").agg({
                "pnl": ["count", "sum", "mean"]
            })
            
            analysis["exit_reasons"] = {
                reason: {
                    "count": int(stats["pnl"]["count"]),
                    "total_pnl": self.format_number(stats["pnl"]["sum"], "currency"),
                    "avg_pnl": self.format_number(stats["pnl"]["mean"], "currency")
                }
                for reason, stats in exit_performance.iterrows()
            }
            
        return analysis if analysis else {"message": "Entry/exit reason data not available"}
    
    def _analyze_trade_clustering(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trade clustering patterns"""
        if "entry_time" not in trades.columns:
            return {"message": "Trade timing data not available"}
            
        # Sort trades by entry time
        trades_sorted = trades.sort_values("entry_time")
        
        # Calculate time between trades
        time_between_trades = trades_sorted["entry_time"].diff().dt.total_seconds() / 3600  # hours
        
        # Identify clusters (trades within 1 hour of each other)
        cluster_threshold = 1  # hour
        clusters = []
        current_cluster = [0]
        
        for i in range(1, len(time_between_trades)):
            if time_between_trades.iloc[i] <= cluster_threshold:
                current_cluster.append(i)
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [i]
                
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
            
        # Analyze cluster performance
        cluster_analysis = {
            "total_clusters": len(clusters),
            "avg_cluster_size": np.mean([len(c) for c in clusters]) if clusters else 0,
            "largest_cluster": max([len(c) for c in clusters]) if clusters else 0,
            "cluster_performance": self._analyze_cluster_performance(trades_sorted, clusters)
        }
        
        return cluster_analysis
    
    def _analyze_trade_prices(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trade price information"""
        price_analysis = {}
        
        # Entry price analysis
        if "entry_price" in trades.columns:
            entry_prices = trades["entry_price"]
            price_analysis["entry_analysis"] = {
                "avg_entry_price": self.format_number(entry_prices.mean(), "currency"),
                "median_entry_price": self.format_number(entry_prices.median(), "currency"),
                "min_entry_price": self.format_number(entry_prices.min(), "currency"),
                "max_entry_price": self.format_number(entry_prices.max(), "currency"),
                "entry_price_std": self.format_number(entry_prices.std(), "currency")
            }
        
        # Exit price analysis
        if "exit_price" in trades.columns:
            exit_prices = trades["exit_price"]
            price_analysis["exit_analysis"] = {
                "avg_exit_price": self.format_number(exit_prices.mean(), "currency"),
                "median_exit_price": self.format_number(exit_prices.median(), "currency"),
                "min_exit_price": self.format_number(exit_prices.min(), "currency"),
                "max_exit_price": self.format_number(exit_prices.max(), "currency"),
                "exit_price_std": self.format_number(exit_prices.std(), "currency")
            }
        
        # Price movement analysis
        if "entry_price" in trades.columns and "exit_price" in trades.columns:
            price_movement = (trades["exit_price"] - trades["entry_price"]) / trades["entry_price"]
            price_analysis["price_movement"] = {
                "avg_price_change": self.format_number(price_movement.mean(), "percentage"),
                "median_price_change": self.format_number(price_movement.median(), "percentage"),
                "max_favorable_move": self.format_number(price_movement.max(), "percentage"),
                "max_adverse_move": self.format_number(price_movement.min(), "percentage"),
                "price_change_volatility": self.format_number(price_movement.std(), "percentage")
            }
        
        # Stop loss analysis
        if "stop_loss_price" in trades.columns:
            stop_losses = trades["stop_loss_price"].dropna()
            if not stop_losses.empty:
                price_analysis["stop_loss_analysis"] = {
                    "avg_stop_loss_price": self.format_number(stop_losses.mean(), "currency"),
                    "median_stop_loss_price": self.format_number(stop_losses.median(), "currency"),
                    "stop_loss_usage": self.format_number(len(stop_losses) / len(trades), "percentage")
                }
                
                # Stop loss distance analysis
                if "entry_price" in trades.columns:
                    sl_distance = (trades["entry_price"] - trades["stop_loss_price"]).abs() / trades["entry_price"]
                    price_analysis["stop_loss_analysis"]["avg_sl_distance"] = self.format_number(sl_distance.mean(), "percentage")
                    price_analysis["stop_loss_analysis"]["median_sl_distance"] = self.format_number(sl_distance.median(), "percentage")
        
        # Take profit analysis
        if "take_profit_price" in trades.columns:
            take_profits = trades["take_profit_price"].dropna()
            if not take_profits.empty:
                price_analysis["take_profit_analysis"] = {
                    "avg_take_profit_price": self.format_number(take_profits.mean(), "currency"),
                    "median_take_profit_price": self.format_number(take_profits.median(), "currency"),
                    "take_profit_usage": self.format_number(len(take_profits) / len(trades), "percentage")
                }
                
                # Take profit distance analysis
                if "entry_price" in trades.columns:
                    tp_distance = (trades["take_profit_price"] - trades["entry_price"]).abs() / trades["entry_price"]
                    price_analysis["take_profit_analysis"]["avg_tp_distance"] = self.format_number(tp_distance.mean(), "percentage")
                    price_analysis["take_profit_analysis"]["median_tp_distance"] = self.format_number(tp_distance.median(), "percentage")
        
        return price_analysis
    
    def _analyze_trade_risk(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trade risk metrics"""
        risk_analysis = {}
        
        # Risk per trade analysis
        if "entry_price" in trades.columns and "stop_loss_price" in trades.columns:
            risk_per_trade = (trades["entry_price"] - trades["stop_loss_price"]).abs() / trades["entry_price"]
            risk_per_trade = risk_per_trade.dropna()
            
            if not risk_per_trade.empty:
                risk_analysis["risk_per_trade"] = {
                    "avg_risk_per_trade": self.format_number(risk_per_trade.mean(), "percentage"),
                    "median_risk_per_trade": self.format_number(risk_per_trade.median(), "percentage"),
                    "max_risk_per_trade": self.format_number(risk_per_trade.max(), "percentage"),
                    "min_risk_per_trade": self.format_number(risk_per_trade.min(), "percentage"),
                    "risk_consistency": self.format_number(risk_per_trade.std(), "percentage")
                }
        
        # Risk-reward ratio analysis
        if all(col in trades.columns for col in ["entry_price", "exit_price", "stop_loss_price"]):
            # Calculate potential reward and risk
            potential_reward = (trades["exit_price"] - trades["entry_price"]).abs() / trades["entry_price"]
            potential_risk = (trades["entry_price"] - trades["stop_loss_price"]).abs() / trades["entry_price"]
            
            risk_reward_ratio = potential_reward / potential_risk
            risk_reward_ratio = risk_reward_ratio.replace([np.inf, -np.inf], np.nan).dropna()
            
            if not risk_reward_ratio.empty:
                risk_analysis["risk_reward_ratio"] = {
                    "avg_risk_reward_ratio": self.format_number(risk_reward_ratio.mean(), "ratio"),
                    "median_risk_reward_ratio": self.format_number(risk_reward_ratio.median(), "ratio"),
                    "best_risk_reward_ratio": self.format_number(risk_reward_ratio.max(), "ratio"),
                    "worst_risk_reward_ratio": self.format_number(risk_reward_ratio.min(), "ratio")
                }
        
        # MAE/MFE analysis (Maximum Adverse/Favorable Excursion)
        if "mae" in trades.columns and "mfe" in trades.columns:
            mae_values = trades["mae"].dropna()
            mfe_values = trades["mfe"].dropna()
            
            if not mae_values.empty:
                risk_analysis["mae_analysis"] = {
                    "avg_mae": self.format_number(mae_values.mean(), "percentage"),
                    "median_mae": self.format_number(mae_values.median(), "percentage"),
                    "max_mae": self.format_number(mae_values.max(), "percentage"),
                    "mae_std": self.format_number(mae_values.std(), "percentage")
                }
            
            if not mfe_values.empty:
                risk_analysis["mfe_analysis"] = {
                    "avg_mfe": self.format_number(mfe_values.mean(), "percentage"),
                    "median_mfe": self.format_number(mfe_values.median(), "percentage"),
                    "max_mfe": self.format_number(mfe_values.max(), "percentage"),
                    "mfe_std": self.format_number(mfe_values.std(), "percentage")
                }
        
        return risk_analysis
    
    def _create_enhanced_trade_table(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Create enhanced trade table with all price information"""
        if trades.empty:
            return {"message": "No trades to display"}
        
        # Define columns to include in the table
        base_columns = ['symbol', 'side', 'quantity', 'entry_time', 'exit_time', 'pnl']
        price_columns = ['entry_price', 'exit_price', 'stop_loss_price', 'take_profit_price']
        analysis_columns = ['mae', 'mfe', 'duration', 'exit_reason']
        
        # Build table data
        table_data = []
        available_columns = [col for col in base_columns + price_columns + analysis_columns if col in trades.columns]
        
        for idx, row in trades.iterrows():
            trade_row = {}
            
            # Basic trade information
            trade_row['Trade ID'] = f"#{idx + 1}"
            
            for col in available_columns:
                formatted_col = col.replace('_', ' ').title()
                value = row[col]
                
                if pd.isna(value):
                    trade_row[formatted_col] = 'N/A'
                elif 'price' in col:
                    trade_row[formatted_col] = self.format_number(value, "currency")
                elif col == 'pnl':
                    trade_row[formatted_col] = self.format_number(value, "currency")
                elif col in ['mae', 'mfe']:
                    trade_row[formatted_col] = self.format_number(value, "percentage")
                elif col == 'duration':
                    trade_row[formatted_col] = f"{value:.1f}h" if isinstance(value, (int, float)) else str(value)
                elif col in ['entry_time', 'exit_time']:
                    trade_row[formatted_col] = value.strftime('%Y-%m-%d %H:%M') if pd.notna(value) else 'N/A'
                else:
                    trade_row[formatted_col] = str(value)
            
            # Calculate additional metrics if possible
            if 'entry_price' in trades.columns and 'exit_price' in trades.columns:
                price_change = (row['exit_price'] - row['entry_price']) / row['entry_price']
                trade_row['Price Change'] = self.format_number(price_change, "percentage")
            
            if 'entry_price' in trades.columns and 'stop_loss_price' in trades.columns and pd.notna(row['stop_loss_price']):
                risk_pct = abs(row['entry_price'] - row['stop_loss_price']) / row['entry_price']
                trade_row['Risk %'] = self.format_number(risk_pct, "percentage")
            
            table_data.append(trade_row)
        
        return {
            "columns": list(table_data[0].keys()) if table_data else [],
            "data": table_data,
            "total_trades": len(trades),
            "summary": {
                "total_pnl": self.format_number(trades['pnl'].sum(), "currency"),
                "avg_pnl": self.format_number(trades['pnl'].mean(), "currency"),
                "win_rate": self.format_number((trades['pnl'] > 0).mean(), "percentage"),
                "best_trade": self.format_number(trades['pnl'].max(), "currency"),
                "worst_trade": self.format_number(trades['pnl'].min(), "currency")
            }
        }
    
    # Helper methods
    def _calculate_trades_per_day(self, trades: pd.DataFrame) -> float:
        """Calculate average trades per day"""
        if "entry_time" not in trades.columns:
            return 0
            
        trading_days = (trades["entry_time"].max() - trades["entry_time"].min()).days
        return len(trades) / max(trading_days, 1)
    
    def _calculate_profit_factor(self, trades: pd.DataFrame) -> float:
        """Calculate profit factor"""
        gross_profit = trades[trades["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trades[trades["pnl"] < 0]["pnl"].sum())
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _calculate_expectancy(self, trades: pd.DataFrame) -> float:
        """Calculate trade expectancy"""
        win_rate = (trades["pnl"] > 0).mean()
        avg_win = trades[trades["pnl"] > 0]["pnl"].mean() if any(trades["pnl"] > 0) else 0
        avg_loss = abs(trades[trades["pnl"] < 0]["pnl"].mean()) if any(trades["pnl"] < 0) else 0
        
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    def _calculate_edge_ratio(self, trades: pd.DataFrame) -> float:
        """Calculate edge ratio"""
        total_pnl = trades["pnl"].sum()
        total_mae = trades["mae"].sum() if "mae" in trades.columns else trades["pnl"].abs().sum()
        
        return total_pnl / total_mae if total_mae > 0 else 0
    
    def _calculate_avg_duration(self, trades: pd.DataFrame) -> str:
        """Calculate average trade duration"""
        if "duration" in trades.columns and not trades.empty:
            return f"{trades['duration'].mean():.1f} hours"
        elif "entry_time" in trades.columns and "exit_time" in trades.columns and not trades.empty:
            duration = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 3600
            return f"{duration.mean():.1f} hours"
        else:
            return "N/A"
    
    def _analyze_consecutive_trades(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze consecutive wins/losses"""
        if trades.empty:
            return {}
            
        # Calculate consecutive runs
        trade_results = (trades["pnl"] > 0).astype(int)
        runs = []
        current_run = 1
        current_type = trade_results.iloc[0]
        
        for i in range(1, len(trade_results)):
            if trade_results.iloc[i] == current_type:
                current_run += 1
            else:
                runs.append((current_type, current_run))
                current_type = trade_results.iloc[i]
                current_run = 1
                
        runs.append((current_type, current_run))
        
        # Analyze runs
        win_runs = [run[1] for run in runs if run[0] == 1]
        loss_runs = [run[1] for run in runs if run[0] == 0]
        
        return {
            "max_consecutive_wins": max(win_runs) if win_runs else 0,
            "max_consecutive_losses": max(loss_runs) if loss_runs else 0,
            "avg_win_streak": np.mean(win_runs) if win_runs else 0,
            "avg_loss_streak": np.mean(loss_runs) if loss_runs else 0
        }
    
    def _create_duration_buckets(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Create duration buckets for analysis"""
        if "duration" not in trades.columns:
            return {}
            
        buckets = {
            "< 1 hour": trades[trades["duration"] < 1],
            "1-4 hours": trades[(trades["duration"] >= 1) & (trades["duration"] < 4)],
            "4-24 hours": trades[(trades["duration"] >= 4) & (trades["duration"] < 24)],
            "1-7 days": trades[(trades["duration"] >= 24) & (trades["duration"] < 168)],
            "> 7 days": trades[trades["duration"] >= 168]
        }
        
        return {
            bucket: {
                "count": len(data),
                "win_rate": self.format_number((data["pnl"] > 0).mean(), "percentage"),
                "avg_pnl": self.format_number(data["pnl"].mean(), "currency")
            }
            for bucket, data in buckets.items() if not data.empty
        }
    
    def _find_optimal_holding_period(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Find optimal holding period based on performance"""
        if "duration" not in trades.columns or len(trades) < 10:
            return {"message": "Insufficient data for analysis"}
            
        # Group by duration ranges and calculate metrics
        duration_performance = []
        
        for hours in [1, 4, 8, 24, 48, 168]:  # 1h, 4h, 8h, 1d, 2d, 1w
            subset = trades[trades["duration"] <= hours]
            if not subset.empty:
                duration_performance.append({
                    "max_duration": hours,
                    "win_rate": (subset["pnl"] > 0).mean(),
                    "avg_pnl": subset["pnl"].mean(),
                    "sharpe": subset["pnl"].mean() / subset["pnl"].std() if subset["pnl"].std() > 0 else 0
                })
                
        # Find optimal based on Sharpe ratio
        if duration_performance:
            optimal = max(duration_performance, key=lambda x: x["sharpe"])
            return {
                "optimal_duration": f"<= {optimal['max_duration']} hours",
                "expected_win_rate": self.format_number(optimal["win_rate"], "percentage"),
                "expected_return": self.format_number(optimal["avg_pnl"], "currency")
            }
            
        return {"message": "Unable to determine optimal holding period"}
    
    def _analyze_monthly_distribution(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trade distribution by month"""
        if "entry_time" not in trades.columns:
            return {}
            
        trades["month"] = pd.to_datetime(trades["entry_time"]).dt.to_period('M')
        monthly_stats = trades.groupby("month").agg({
            "pnl": ["count", "sum", "mean"],
            
        })
        
        return {
            str(month): {
                "trades": int(stats["pnl"]["count"]),
                "total_pnl": self.format_number(stats["pnl"]["sum"], "currency"),
                "avg_pnl": self.format_number(stats["pnl"]["mean"], "currency"),
                "win_rate": self.format_number(
                    (trades[trades["month"] == month]["pnl"] > 0).mean(),
                    "percentage"
                )
            }
            for month, stats in monthly_stats.iterrows()
        }
    
    def _analyze_day_of_week_distribution(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trade distribution by day of week"""
        if "entry_time" not in trades.columns:
            return {}
            
        trades["dow"] = pd.to_datetime(trades["entry_time"]).dt.day_name()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        dow_stats = {}
        for day in days_order:
            day_trades = trades[trades["dow"] == day]
            if not day_trades.empty:
                dow_stats[day] = {
                    "trades": len(day_trades),
                    "total_pnl": self.format_number(day_trades["pnl"].sum(), "currency"),
                    "avg_pnl": self.format_number(day_trades["pnl"].mean(), "currency"),
                    "win_rate": self.format_number((day_trades["pnl"] > 0).mean(), "percentage")
                }
                
        return dow_stats
    
    def _analyze_cluster_performance(self, trades: pd.DataFrame, clusters: List[List[int]]) -> Dict[str, Any]:
        """Analyze performance of trade clusters"""
        if not clusters:
            return {"message": "No trade clusters found"}
            
        cluster_stats = []
        for cluster_indices in clusters:
            cluster_trades = trades.iloc[cluster_indices]
            cluster_stats.append({
                "size": len(cluster_indices),
                "total_pnl": cluster_trades["pnl"].sum(),
                "win_rate": (cluster_trades["pnl"] > 0).mean()
            })
            
        # Aggregate statistics
        avg_cluster_pnl = np.mean([c["total_pnl"] for c in cluster_stats])
        avg_cluster_win_rate = np.mean([c["win_rate"] for c in cluster_stats])
        
        return {
            "avg_cluster_pnl": self.format_number(avg_cluster_pnl, "currency"),
            "avg_cluster_win_rate": self.format_number(avg_cluster_win_rate, "percentage"),
            "best_cluster_pnl": self.format_number(max(c["total_pnl"] for c in cluster_stats), "currency"),
            "worst_cluster_pnl": self.format_number(min(c["total_pnl"] for c in cluster_stats), "currency")
        }


class MarketRegimeAnalysis(ReportSection):
    """Market regime analysis section"""
    
    def generate(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market regime analysis"""
        equity_curve = backtest_results.get("equity_curve", pd.Series())
        market_data = backtest_results.get("market_data", pd.DataFrame())
        trades = backtest_results.get("trades", pd.DataFrame())
        
        return {
            "regime_identification": self._identify_market_regimes(market_data, equity_curve),
            "regime_performance": self._analyze_regime_performance(market_data, equity_curve, trades),
            "regime_transitions": self._analyze_regime_transitions(market_data),
            "adaptive_behavior": self._analyze_adaptive_behavior(market_data, trades),
            "correlation_analysis": self._analyze_regime_correlations(market_data, equity_curve)
        }
    
    def _identify_market_regimes(self, market_data: pd.DataFrame, equity_curve: pd.Series) -> Dict[str, Any]:
        """Identify different market regimes"""
        if market_data.empty:
            # Use equity curve returns as proxy
            returns = equity_curve.pct_change().dropna()
            volatility = returns.rolling(20).std() * np.sqrt(252)
            trend = returns.rolling(20).mean() * 252
        else:
            # Use actual market data
            returns = market_data["close"].pct_change().dropna() if "close" in market_data else pd.Series()
            volatility = returns.rolling(20).std() * np.sqrt(252)
            trend = returns.rolling(20).mean() * 252
            
        # Define regime thresholds
        vol_median = volatility.median()
        
        # Classify regimes
        regimes = pd.Series(index=returns.index, dtype=str)
        
        # Bull market: positive trend, normal volatility
        bull_mask = (trend > 0.05) & (volatility <= vol_median * 1.5)
        regimes[bull_mask] = "Bull Market"
        
        # Bear market: negative trend, any volatility
        bear_mask = trend < -0.05
        regimes[bear_mask] = "Bear Market"
        
        # High volatility: any trend, high volatility
        high_vol_mask = volatility > vol_median * 1.5
        regimes[high_vol_mask] = "High Volatility"
        
        # Range-bound: low trend, normal volatility
        range_mask = (trend.abs() <= 0.05) & (volatility <= vol_median * 1.5)
        regimes[range_mask] = "Range-Bound"
        
        # Fill any remaining
        regimes[regimes == ""] = "Transitional"
        
        # Current regime
        current_regime = regimes.iloc[-1] if not regimes.empty else "Unknown"
        
        # Regime statistics
        regime_counts = regimes.value_counts()
        regime_percentages = regime_counts / len(regimes)
        
        return {
            "current_regime": current_regime,
            "regime_distribution": {
                regime: {
                    "periods": int(count),
                    "percentage": self.format_number(regime_percentages[regime], "percentage")
                }
                for regime, count in regime_counts.items()
            },
            "regime_characteristics": self._describe_regime_characteristics(),
            "regime_duration": self._calculate_regime_durations(regimes)
        }
    
    def _analyze_regime_performance(
        self,
        market_data: pd.DataFrame,
        equity_curve: pd.Series,
        trades: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze strategy performance in different regimes"""
        # This is simplified - in practice would use actual regime classification
        returns = equity_curve.pct_change().dropna()
        
        # Mock regime classification for demonstration
        volatility = returns.rolling(20).std() * np.sqrt(252)
        vol_median = volatility.median()
        
        # Simple regime classification
        high_vol_periods = volatility > vol_median * 1.5
        low_vol_periods = ~high_vol_periods
        
        regime_performance = {
            "high_volatility": {
                "returns": self.format_number(
                    returns[high_vol_periods].mean() * 252 if any(high_vol_periods) else 0,
                    "percentage"
                ),
                "sharpe": self.format_number(
                    returns[high_vol_periods].mean() / returns[high_vol_periods].std() * np.sqrt(252)
                    if any(high_vol_periods) and returns[high_vol_periods].std() > 0 else 0,
                    "ratio"
                ),
                "win_rate": self._calculate_regime_win_rate(trades, high_vol_periods),
                "avg_trade_pnl": self._calculate_regime_avg_pnl(trades, high_vol_periods)
            },
            "low_volatility": {
                "returns": self.format_number(
                    returns[low_vol_periods].mean() * 252 if any(low_vol_periods) else 0,
                    "percentage"
                ),
                "sharpe": self.format_number(
                    returns[low_vol_periods].mean() / returns[low_vol_periods].std() * np.sqrt(252)
                    if any(low_vol_periods) and returns[low_vol_periods].std() > 0 else 0,
                    "ratio"
                ),
                "win_rate": self._calculate_regime_win_rate(trades, low_vol_periods),
                "avg_trade_pnl": self._calculate_regime_avg_pnl(trades, low_vol_periods)
            }
        }
        
        return regime_performance
    
    def _analyze_regime_transitions(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze transitions between market regimes"""
        # Simplified transition analysis
        return {
            "transition_frequency": "Analysis requires regime classification",
            "avg_transition_duration": "Analysis requires regime classification",
            "transition_impact": "Analysis requires regime classification",
            "most_common_transitions": {
                "Bull to Range": "Historical data required",
                "Range to Bull": "Historical data required",
                "Bull to Bear": "Historical data required"
            }
        }
    
    def _analyze_adaptive_behavior(self, market_data: pd.DataFrame, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how strategy adapts to different regimes"""
        if trades.empty:
            return {"message": "No trades available for adaptive analysis"}
            
        # Analyze trade frequency changes
        # This is simplified - would need actual regime data
        
        return {
            "trade_frequency_adaptation": "Analysis requires regime classification",
            "position_sizing_adaptation": "Analysis requires regime classification",
            "risk_parameter_adaptation": "Analysis requires regime classification",
            "performance_consistency": self._assess_performance_consistency(trades)
        }
    
    def _analyze_regime_correlations(self, market_data: pd.DataFrame, equity_curve: pd.Series) -> Dict[str, Any]:
        """Analyze correlations in different regimes"""
        if market_data.empty or "close" not in market_data:
            return {"message": "Market data required for correlation analysis"}
            
        # Calculate rolling correlations
        strategy_returns = equity_curve.pct_change().dropna()
        market_returns = market_data["close"].pct_change().dropna()
        
        # Align indices
        common_idx = strategy_returns.index.intersection(market_returns.index)
        strategy_returns = strategy_returns[common_idx]
        market_returns = market_returns[common_idx]
        
        # Rolling correlation
        rolling_corr = strategy_returns.rolling(60).corr(market_returns)
        
        return {
            "overall_correlation": self.format_number(strategy_returns.corr(market_returns), "ratio"),
            "rolling_correlation": {
                "current": self.format_number(rolling_corr.iloc[-1] if not rolling_corr.empty else 0, "ratio"),
                "average": self.format_number(rolling_corr.mean(), "ratio"),
                "min": self.format_number(rolling_corr.min(), "ratio"),
                "max": self.format_number(rolling_corr.max(), "ratio")
            },
            "correlation_stability": self.format_number(rolling_corr.std(), "ratio"),
            "regime_specific_correlations": "Requires regime classification"
        }
    
    # Helper methods
    def _describe_regime_characteristics(self) -> Dict[str, Any]:
        """Describe characteristics of each regime"""
        return {
            "Bull Market": {
                "description": "Sustained upward trend with normal volatility",
                "typical_duration": "6-18 months",
                "key_indicators": "Rising prices, positive sentiment, increasing volume"
            },
            "Bear Market": {
                "description": "Sustained downward trend with elevated volatility",
                "typical_duration": "3-12 months",
                "key_indicators": "Falling prices, negative sentiment, high volatility"
            },
            "High Volatility": {
                "description": "Elevated market uncertainty and price swings",
                "typical_duration": "1-3 months",
                "key_indicators": "Large daily moves, increased option premiums, news-driven"
            },
            "Range-Bound": {
                "description": "Sideways market with defined support/resistance",
                "typical_duration": "2-6 months",
                "key_indicators": "Low trend, normal volatility, technical levels hold"
            }
        }
    
    def _calculate_regime_durations(self, regimes: pd.Series) -> Dict[str, Any]:
        """Calculate average duration of each regime"""
        regime_durations = {}
        
        # Group consecutive regime periods
        regime_changes = regimes != regimes.shift()
        regime_groups = regime_changes.cumsum()
        
        for regime in regimes.unique():
            regime_periods = regimes[regimes == regime].groupby(regime_groups).size()
            if not regime_periods.empty:
                regime_durations[regime] = {
                    "avg_duration": f"{regime_periods.mean():.0f} periods",
                    "max_duration": f"{regime_periods.max():.0f} periods",
                    "current_duration": self._get_current_regime_duration(regimes, regime)
                }
                
        return regime_durations
    
    def _get_current_regime_duration(self, regimes: pd.Series, regime: str) -> str:
        """Get duration of current regime if active"""
        if regimes.iloc[-1] != regime:
            return "Not active"
            
        # Count backwards until regime changes
        count = 0
        for i in range(len(regimes) - 1, -1, -1):
            if regimes.iloc[i] == regime:
                count += 1
            else:
                break
                
        return f"{count} periods"
    
    def _calculate_regime_win_rate(self, trades: pd.DataFrame, regime_mask: pd.Series) -> str:
        """Calculate win rate during specific regime"""
        if trades.empty or "entry_time" not in trades.columns:
            return "N/A"
            
        # Filter trades that occurred during the regime
        regime_trades = trades[trades["entry_time"].isin(regime_mask[regime_mask].index)]
        
        if regime_trades.empty:
            return "No trades"
            
        win_rate = (regime_trades["pnl"] > 0).mean()
        return self.format_number(win_rate, "percentage")
    
    def _calculate_regime_avg_pnl(self, trades: pd.DataFrame, regime_mask: pd.Series) -> str:
        """Calculate average PnL during specific regime"""
        if trades.empty or "entry_time" not in trades.columns:
            return "N/A"
            
        # Filter trades that occurred during the regime
        regime_trades = trades[trades["entry_time"].isin(regime_mask[regime_mask].index)]
        
        if regime_trades.empty:
            return "No trades"
            
        avg_pnl = regime_trades["pnl"].mean()
        return self.format_number(avg_pnl, "currency")
    
    def _assess_performance_consistency(self, trades: pd.DataFrame) -> str:
        """Assess consistency of performance"""
        if trades.empty or len(trades) < 20:
            return "Insufficient data for consistency analysis"
            
        # Calculate rolling metrics
        trades["cumulative_pnl"] = trades["pnl"].cumsum()
        
        # Simple consistency check - coefficient of variation
        returns_cv = trades["pnl"].std() / abs(trades["pnl"].mean()) if trades["pnl"].mean() != 0 else float('inf')
        
        if returns_cv < 1:
            return "High consistency - stable performance across regimes"
        elif returns_cv < 2:
            return "Moderate consistency - some variation across regimes"
        else:
            return "Low consistency - significant variation across regimes"


class TechnicalDetails(ReportSection):
    """Technical implementation details section"""
    
    def generate(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical details"""
        strategy_params = backtest_results.get("strategy_params", {})
        execution_stats = backtest_results.get("execution_stats", {})
        
        return {
            "strategy_configuration": self._document_strategy_config(strategy_params),
            "execution_statistics": self._analyze_execution_stats(execution_stats),
            "computational_performance": self._analyze_computational_performance(backtest_results),
            "data_quality": self._assess_data_quality(backtest_results),
            "backtest_assumptions": self._document_assumptions(),
            "implementation_notes": self._generate_implementation_notes(strategy_params)
        }
    
    def _document_strategy_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Document strategy configuration"""
        return {
            "parameters": {
                param: {
                    "value": value,
                    "type": type(value).__name__,
                    "description": self._get_parameter_description(param)
                }
                for param, value in params.items()
            },
            "optimization_settings": params.get("optimization", {
                "method": "Not optimized",
                "objective": "N/A",
                "constraints": []
            }),
            "risk_settings": {
                "position_sizing": params.get("position_sizing", "Fixed"),
                "max_position_size": params.get("max_position_size", 1.0),
                "stop_loss": params.get("stop_loss", "None"),
                "take_profit": params.get("take_profit", "None")
            }
        }
    
    def _analyze_execution_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze execution statistics"""
        if not stats:
            return {"message": "No execution statistics available"}
            
        return {
            "order_execution": {
                "total_orders": stats.get("total_orders", 0),
                "filled_orders": stats.get("filled_orders", 0),
                "rejected_orders": stats.get("rejected_orders", 0),
                "avg_fill_time": f"{stats.get('avg_fill_time', 0):.2f} ms"
            },
            "slippage_analysis": {
                "avg_slippage": self.format_number(stats.get("avg_slippage", 0), "percentage"),
                "max_slippage": self.format_number(stats.get("max_slippage", 0), "percentage"),
                "slippage_cost": self.format_number(stats.get("slippage_cost", 0), "currency")
            },
            "transaction_costs": {
                "total_commission": self.format_number(stats.get("total_commission", 0), "currency"),
                "total_spread_cost": self.format_number(stats.get("total_spread_cost", 0), "currency"),
                "avg_cost_per_trade": self.format_number(stats.get("avg_cost_per_trade", 0), "currency")
            }
        }
    
    def _analyze_computational_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze computational performance"""
        perf_stats = results.get("performance_stats", {})
        
        return {
            "execution_time": {
                "total_time": f"{perf_stats.get('total_time', 0):.2f} seconds",
                "avg_time_per_bar": f"{perf_stats.get('avg_time_per_bar', 0):.4f} seconds",
                "bars_processed": perf_stats.get("bars_processed", 0)
            },
            "memory_usage": {
                "peak_memory": f"{perf_stats.get('peak_memory', 0):.2f} MB",
                "avg_memory": f"{perf_stats.get('avg_memory', 0):.2f} MB"
            },
            "optimization_performance": {
                "iterations": perf_stats.get("optimization_iterations", 0),
                "convergence_time": f"{perf_stats.get('convergence_time', 0):.2f} seconds"
            }
        }
    
    def _assess_data_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of input data"""
        data_stats = results.get("data_statistics", {})
        
        return {
            "data_coverage": {
                "start_date": str(data_stats.get("start_date", "N/A")),
                "end_date": str(data_stats.get("end_date", "N/A")),
                "total_bars": data_stats.get("total_bars", 0),
                "missing_data": self.format_number(data_stats.get("missing_data_pct", 0), "percentage")
            },
            "data_integrity": {
                "outliers_detected": data_stats.get("outliers", 0),
                "data_gaps": data_stats.get("gaps", 0),
                "suspicious_values": data_stats.get("suspicious_values", 0)
            },
            "data_adjustments": {
                "splits_adjusted": data_stats.get("splits_adjusted", False),
                "dividends_adjusted": data_stats.get("dividends_adjusted", False),
                "currency_normalized": data_stats.get("currency_normalized", False)
            }
        }
    
    def _document_assumptions(self) -> Dict[str, Any]:
        """Document backtest assumptions"""
        return {
            "market_assumptions": {
                "liquidity": "Infinite liquidity assumed",
                "market_impact": "No market impact modeled",
                "order_execution": "All orders filled at specified price",
                "trading_hours": "Regular market hours only"
            },
            "cost_assumptions": {
                "commission": "Fixed per-trade or percentage-based",
                "slippage": "Fixed percentage or dynamic model",
                "financing": "Overnight financing costs included",
                "taxes": "Tax implications not considered"
            },
            "risk_assumptions": {
                "position_limits": "As configured in strategy",
                "margin_requirements": "Not enforced unless specified",
                "portfolio_constraints": "Single strategy backtest"
            },
            "data_assumptions": {
                "price_data": "Adjusted for splits and dividends",
                "frequency": "As provided in input data",
                "accuracy": "Data assumed to be accurate"
            }
        }
    
    def _generate_implementation_notes(self, params: Dict[str, Any]) -> List[str]:
        """Generate implementation notes and warnings"""
        notes = []
        
        # Check for common issues
        if params.get("stop_loss") is None:
            notes.append("⚠️ No stop-loss configured - strategy has unlimited downside risk")
            
        if params.get("position_sizing") == "Fixed" and params.get("max_position_size", 1) == 1:
            notes.append("ℹ️ Using 100% position sizing - consider risk management")
            
        if params.get("optimization_method") == "Grid Search":
            notes.append("ℹ️ Grid search optimization may lead to overfitting")
            
        # Add general notes
        notes.extend([
            "✓ Backtest includes transaction costs and slippage",
            "✓ Results are based on historical data and may not predict future performance",
            "ℹ️ Consider out-of-sample testing for validation",
            "ℹ️ Monitor live performance for strategy degradation"
        ])
        
        return notes
    
    def _get_parameter_description(self, param: str) -> str:
        """Get description for common parameters"""
        descriptions = {
            "lookback_period": "Number of periods for indicator calculation",
            "entry_threshold": "Threshold value for trade entry signals",
            "exit_threshold": "Threshold value for trade exit signals",
            "stop_loss": "Maximum acceptable loss per trade",
            "take_profit": "Target profit level per trade",
            "position_sizing": "Method for determining trade size",
            "max_positions": "Maximum number of concurrent positions",
            "risk_per_trade": "Maximum risk per trade as percentage of capital"
        }
        
        return descriptions.get(param, "Strategy-specific parameter")


# Import scipy for statistical functions
from scipy import stats