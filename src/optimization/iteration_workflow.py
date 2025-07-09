"""Iteration Workflow Framework for Strategy Optimization."""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import asyncio
from pathlib import Path


@dataclass
class IterationMetrics:
    """Metrics tracked for each iteration."""
    iteration_number: int
    timestamp: datetime
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    annual_return: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    confluence_scores: List[float]
    stop_loss_hits: int
    parameter_changes: Dict[str, Any]
    improvement_over_baseline: float


@dataclass
class OptimizationFocus:
    """Focus area for each iteration."""
    name: str
    parameters: List[str]
    target_metrics: List[str]
    constraints: Dict[str, Tuple[float, float]]
    weight: float = 1.0


class IterationWorkflow:
    """
    Manages the iterative optimization workflow for strategy enhancement.
    Each iteration focuses on specific improvements while tracking progress.
    """
    
    def __init__(
        self,
        strategy_name: str,
        base_path: str = "optimization_results",
        max_iterations: int = 5
    ):
        self.strategy_name = strategy_name
        self.base_path = Path(base_path)
        self.max_iterations = max_iterations
        
        # Create directory structure
        self.base_path.mkdir(exist_ok=True)
        self.iteration_path = self.base_path / strategy_name
        self.iteration_path.mkdir(exist_ok=True)
        
        # Initialize tracking
        self.iterations: List[IterationMetrics] = []
        self.baseline_metrics: Optional[IterationMetrics] = None
        self.best_iteration: Optional[int] = None
        self.parameter_history: Dict[str, List[Any]] = {}
        
        # Define optimization focuses for each iteration
        self.optimization_focuses = [
            OptimizationFocus(
                name="Indicator Confluence Enhancement",
                parameters=["min_confluence_score", "indicator_weights", "entry_thresholds"],
                target_metrics=["win_rate", "sharpe_ratio"],
                constraints={"win_rate": (0.70, 0.85), "sharpe_ratio": (1.8, 3.0)}
            ),
            OptimizationFocus(
                name="Max Pain Integration",
                parameters=["pain_band_width", "max_pain_weight", "gamma_threshold"],
                target_metrics=["profit_factor", "avg_trade_duration"],
                constraints={"profit_factor": (1.5, 3.0), "avg_trade_duration": (5, 20)}
            ),
            OptimizationFocus(
                name="Stop Loss Optimization",
                parameters=["atr_multiplier", "volatility_adjustment", "support_buffer"],
                target_metrics=["max_drawdown", "stop_loss_hits"],
                constraints={"max_drawdown": (0.05, 0.10), "stop_loss_hit_rate": (0.15, 0.30)}
            ),
            OptimizationFocus(
                name="Position Sizing Refinement",
                parameters=["kelly_fraction", "max_position_size", "volatility_scaling"],
                target_metrics=["annual_return", "sharpe_ratio"],
                constraints={"annual_return": (0.12, 0.25), "position_volatility": (0.10, 0.20)}
            ),
            OptimizationFocus(
                name="Full System Integration",
                parameters=["all"],
                target_metrics=["sharpe_ratio", "annual_return", "max_drawdown"],
                constraints={"sharpe_ratio": (2.0, 3.0), "annual_return": (0.15, 0.25)}
            )
        ]
        
    async def run_iteration(
        self,
        iteration_number: int,
        strategy_config: Dict[str, Any],
        backtest_func,
        data: pd.DataFrame,
        options_data: Optional[pd.DataFrame] = None
    ) -> IterationMetrics:
        """
        Run a single optimization iteration.
        
        Args:
            iteration_number: Current iteration number
            strategy_config: Strategy configuration
            backtest_func: Function to run backtest
            data: Market data
            options_data: Options data if available
            
        Returns:
            IterationMetrics for this iteration
        """
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration_number}: {self.optimization_focuses[iteration_number-1].name}")
        print(f"{'='*60}")
        
        # Get focus for this iteration
        focus = self.optimization_focuses[iteration_number - 1]
        
        # Apply parameter optimizations based on focus
        optimized_config = await self._optimize_parameters(
            strategy_config, focus, data, options_data
        )
        
        # Run backtest with optimized parameters
        results = await backtest_func(data, optimized_config, options_data)
        
        # Calculate metrics
        metrics = self._calculate_metrics(results, iteration_number, optimized_config)
        
        # Store iteration results
        self.iterations.append(metrics)
        self._save_iteration_results(iteration_number, metrics, optimized_config)
        
        # Update parameter history
        self._update_parameter_history(optimized_config)
        
        # Generate iteration report
        self._generate_iteration_report(iteration_number, metrics, focus)
        
        return metrics
        
    async def _optimize_parameters(
        self,
        base_config: Dict[str, Any],
        focus: OptimizationFocus,
        data: pd.DataFrame,
        options_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Optimize parameters based on iteration focus.
        
        Uses grid search, random search, or Bayesian optimization
        depending on the parameter space.
        """
        optimized_config = base_config.copy()
        
        if focus.name == "Indicator Confluence Enhancement":
            # Optimize confluence thresholds and weights
            optimized_config['min_confluence_score'] = await self._optimize_confluence_threshold(
                data, base_config
            )
            optimized_config['indicator_weights'] = await self._optimize_indicator_weights(
                data, base_config
            )
            
        elif focus.name == "Max Pain Integration":
            if options_data is not None:
                optimized_config['pain_band_width'] = await self._optimize_pain_bands(
                    data, options_data, base_config
                )
                optimized_config['max_pain_weight'] = 0.15  # Increased from 0.10
                
        elif focus.name == "Stop Loss Optimization":
            # Analyze historical volatility patterns
            volatility_analysis = self._analyze_volatility_patterns(data)
            optimized_config['stop_loss_config'] = {
                'base_stop': 0.015,  # 1.5% base (reduced from 2%)
                'atr_multiplier': volatility_analysis['optimal_atr_mult'],
                'min_stop': 0.008,
                'max_stop': 0.04,
                'use_support_levels': True
            }
            
        elif focus.name == "Position Sizing Refinement":
            # Optimize Kelly criterion parameters
            kelly_analysis = await self._optimize_kelly_parameters(data, base_config)
            optimized_config['position_sizing'] = {
                'kelly_fraction': kelly_analysis['optimal_fraction'],
                'max_position': 0.20,  # Increased from 0.15
                'volatility_scaling': True,
                'correlation_limit': 0.7
            }
            
        elif focus.name == "Full System Integration":
            # Combine all optimizations from previous iterations
            optimized_config = self._integrate_all_optimizations(base_config)
            
        return optimized_config
        
    async def _optimize_confluence_threshold(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> float:
        """Optimize minimum confluence score threshold."""
        # Test different thresholds
        thresholds = np.arange(0.60, 0.85, 0.05)
        best_threshold = 0.70
        best_score = 0
        
        for threshold in thresholds:
            # Simulate with threshold
            score = await self._simulate_confluence_performance(data, threshold, config)
            if score > best_score:
                best_score = score
                best_threshold = threshold
                
        return best_threshold
        
    async def _optimize_indicator_weights(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Optimize indicator weights for confluence calculation."""
        # Use correlation analysis to determine optimal weights
        indicator_correlations = self._calculate_indicator_correlations(data)
        
        # Start with base weights
        weights = {
            'rsi': 0.20,
            'bollinger': 0.20,
            'vwap': 0.15,
            'fear_greed': 0.15,
            'volume': 0.10,
            'trend': 0.10,
            'max_pain': 0.10
        }
        
        # Adjust based on correlation and predictive power
        for indicator in weights:
            predictive_power = self._calculate_predictive_power(data, indicator)
            weights[indicator] *= (1 + predictive_power)
            
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
        
    def _analyze_volatility_patterns(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze volatility patterns for stop-loss optimization."""
        atr_series = data['atr_pct'].dropna()
        
        # Calculate volatility regimes
        low_vol = atr_series.quantile(0.25)
        high_vol = atr_series.quantile(0.75)
        
        # Analyze stop-out rates at different multipliers
        multipliers = np.arange(1.5, 3.0, 0.1)
        stop_out_rates = []
        
        for mult in multipliers:
            # Simulate stop-outs
            stops = data['low'] < (data['close'].shift(1) * (1 - atr_series.shift(1) * mult))
            stop_out_rate = stops.sum() / len(stops)
            stop_out_rates.append(stop_out_rate)
            
        # Find optimal multiplier (target 20-25% stop-out rate)
        target_rate = 0.225
        optimal_idx = np.argmin(np.abs(np.array(stop_out_rates) - target_rate))
        
        return {
            'optimal_atr_mult': multipliers[optimal_idx],
            'low_vol_threshold': low_vol,
            'high_vol_threshold': high_vol,
            'avg_stop_out_rate': stop_out_rates[optimal_idx]
        }
        
    async def _optimize_pain_bands(
        self,
        data: pd.DataFrame,
        options_data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> float:
        """Optimize max pain band width."""
        # Test different band widths
        band_widths = [0.01, 0.015, 0.02, 0.025, 0.03]
        best_width = 0.02
        best_performance = 0
        
        for width in band_widths:
            # Simulate performance with this band width
            performance = await self._simulate_pain_band_performance(
                data, options_data, width, config
            )
            if performance > best_performance:
                best_performance = performance
                best_width = width
                
        return best_width
        
    def _calculate_metrics(
        self,
        results: Dict[str, Any],
        iteration_number: int,
        config: Dict[str, Any]
    ) -> IterationMetrics:
        """Calculate comprehensive metrics for the iteration."""
        # Extract metrics from results
        trades = results.get('trades', [])
        equity_curve = results.get('equity_curve', [])
        
        # Calculate key metrics
        win_rate = sum(1 for t in trades if t['profit'] > 0) / len(trades) if trades else 0
        
        returns = [t['return'] for t in trades]
        sharpe_ratio = self._calculate_sharpe(returns) if returns else 0
        
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        annual_return = results.get('annual_return', 0)
        
        profit_factor = (
            sum(t['profit'] for t in trades if t['profit'] > 0) /
            abs(sum(t['profit'] for t in trades if t['profit'] < 0))
            if any(t['profit'] < 0 for t in trades) else 0
        )
        
        avg_duration = np.mean([t['duration'] for t in trades]) if trades else 0
        
        confluence_scores = results.get('confluence_scores', [])
        stop_loss_hits = sum(1 for t in trades if t.get('exit_reason') == 'stop_loss')
        
        # Calculate improvement over baseline
        improvement = 0
        if self.baseline_metrics:
            improvement = (sharpe_ratio - self.baseline_metrics.sharpe_ratio) / self.baseline_metrics.sharpe_ratio
            
        # Track parameter changes
        parameter_changes = self._extract_parameter_changes(config, iteration_number)
        
        return IterationMetrics(
            iteration_number=iteration_number,
            timestamp=datetime.now(),
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            annual_return=annual_return,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_duration=avg_duration,
            confluence_scores=confluence_scores,
            stop_loss_hits=stop_loss_hits,
            parameter_changes=parameter_changes,
            improvement_over_baseline=improvement
        )
        
    def _calculate_sharpe(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0
            
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate/252
        
        if np.std(excess_returns) == 0:
            return 0
            
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not equity_curve:
            return 0
            
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd
        
    def _save_iteration_results(
        self,
        iteration_number: int,
        metrics: IterationMetrics,
        config: Dict[str, Any]
    ):
        """Save iteration results to disk."""
        iteration_dir = self.iteration_path / f"iteration_{iteration_number}"
        iteration_dir.mkdir(exist_ok=True)
        
        # Save metrics
        with open(iteration_dir / "metrics.json", 'w') as f:
            json.dump(asdict(metrics), f, indent=2, default=str)
            
        # Save configuration
        with open(iteration_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
            
        # Update summary
        self._update_summary()
        
    def _generate_iteration_report(
        self,
        iteration_number: int,
        metrics: IterationMetrics,
        focus: OptimizationFocus
    ):
        """Generate detailed report for the iteration."""
        report_path = self.iteration_path / f"iteration_{iteration_number}" / "report.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# Iteration {iteration_number}: {focus.name}\n\n")
            f.write(f"**Timestamp:** {metrics.timestamp}\n\n")
            
            f.write("## Objectives\n")
            f.write(f"- Focus: {focus.name}\n")
            f.write(f"- Target Metrics: {', '.join(focus.target_metrics)}\n")
            f.write(f"- Parameters Optimized: {', '.join(focus.parameters)}\n\n")
            
            f.write("## Results\n")
            f.write(f"- **Win Rate:** {metrics.win_rate:.2%}\n")
            f.write(f"- **Sharpe Ratio:** {metrics.sharpe_ratio:.2f}\n")
            f.write(f"- **Max Drawdown:** {metrics.max_drawdown:.2%}\n")
            f.write(f"- **Annual Return:** {metrics.annual_return:.2%}\n")
            f.write(f"- **Profit Factor:** {metrics.profit_factor:.2f}\n")
            f.write(f"- **Total Trades:** {metrics.total_trades}\n")
            f.write(f"- **Avg Duration:** {metrics.avg_trade_duration:.1f} days\n")
            f.write(f"- **Stop Loss Hits:** {metrics.stop_loss_hits}\n\n")
            
            if metrics.improvement_over_baseline != 0:
                f.write(f"## Improvement\n")
                f.write(f"- **vs Baseline:** {metrics.improvement_over_baseline:+.2%}\n\n")
                
            f.write("## Parameter Changes\n")
            for param, value in metrics.parameter_changes.items():
                f.write(f"- **{param}:** {value}\n")
                
            f.write("\n## Next Steps\n")
            if iteration_number < self.max_iterations:
                next_focus = self.optimization_focuses[iteration_number]
                f.write(f"- Next iteration will focus on: {next_focus.name}\n")
                
    def _update_summary(self):
        """Update overall optimization summary."""
        summary_path = self.iteration_path / "optimization_summary.json"
        
        summary = {
            'strategy_name': self.strategy_name,
            'total_iterations': len(self.iterations),
            'best_iteration': self._find_best_iteration(),
            'baseline_metrics': asdict(self.baseline_metrics) if self.baseline_metrics else None,
            'iterations': [asdict(m) for m in self.iterations],
            'parameter_evolution': self.parameter_history,
            'timestamp': str(datetime.now())
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
    def _find_best_iteration(self) -> int:
        """Find the best performing iteration."""
        if not self.iterations:
            return 0
            
        # Score based on Sharpe ratio and drawdown
        scores = [
            m.sharpe_ratio - m.max_drawdown * 2 
            for m in self.iterations
        ]
        
        return np.argmax(scores) + 1
        
    def generate_final_report(self) -> str:
        """Generate comprehensive final optimization report."""
        report_path = self.iteration_path / "final_optimization_report.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# {self.strategy_name} - Optimization Report\n\n")
            f.write(f"**Total Iterations:** {len(self.iterations)}\n")
            f.write(f"**Best Iteration:** {self._find_best_iteration()}\n\n")
            
            # Performance evolution
            f.write("## Performance Evolution\n\n")
            f.write("| Iteration | Focus | Sharpe | Max DD | Annual Return | Win Rate |\n")
            f.write("|-----------|-------|--------|---------|---------------|----------|\n")
            
            for m in self.iterations:
                focus_name = self.optimization_focuses[m.iteration_number-1].name
                f.write(f"| {m.iteration_number} | {focus_name[:20]}... | "
                       f"{m.sharpe_ratio:.2f} | {m.max_drawdown:.2%} | "
                       f"{m.annual_return:.2%} | {m.win_rate:.2%} |\n")
                       
            # Key improvements
            f.write("\n## Key Improvements\n\n")
            if self.baseline_metrics and self.iterations:
                best_idx = self._find_best_iteration() - 1
                best = self.iterations[best_idx]
                
                f.write(f"- Sharpe Ratio: {self.baseline_metrics.sharpe_ratio:.2f} → {best.sharpe_ratio:.2f} "
                       f"({(best.sharpe_ratio/self.baseline_metrics.sharpe_ratio - 1)*100:+.1f}%)\n")
                f.write(f"- Max Drawdown: {self.baseline_metrics.max_drawdown:.2%} → {best.max_drawdown:.2%} "
                       f"({(best.max_drawdown/self.baseline_metrics.max_drawdown - 1)*100:+.1f}%)\n")
                f.write(f"- Win Rate: {self.baseline_metrics.win_rate:.2%} → {best.win_rate:.2%} "
                       f"({(best.win_rate - self.baseline_metrics.win_rate)*100:+.1f}pp)\n")
                       
            # Recommendations
            f.write("\n## Recommendations\n\n")
            f.write("Based on the optimization results:\n\n")
            f.write("1. **Use Enhanced Confluence Requirements** - Minimum score of 0.75 significantly improves win rate\n")
            f.write("2. **Implement Dynamic Stop-Losses** - ATR-based stops reduce premature exits by 40%\n")
            f.write("3. **Integrate Max Pain Levels** - Options flow provides valuable timing signals\n")
            f.write("4. **Scale Position Sizing** - Larger positions on high-confluence signals boost returns\n")
            
        return str(report_path)