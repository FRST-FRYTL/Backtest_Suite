"""Walk-forward analysis for strategy optimization."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

from .optimizer import StrategyOptimizer, OptimizationResult
from ..backtesting import BacktestEngine
from ..utils.metrics import PerformanceMetrics


@dataclass
class WalkForwardWindow:
    """Single window in walk-forward analysis."""
    
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    in_sample_results: OptimizationResult
    out_sample_performance: Dict
    
    @property
    def train_days(self) -> int:
        """Number of training days."""
        return (self.train_end - self.train_start).days
        
    @property
    def test_days(self) -> int:
        """Number of test days."""
        return (self.test_end - self.test_start).days


class WalkForwardAnalysis:
    """Perform walk-forward optimization analysis."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        optimizer: StrategyOptimizer,
        train_period_days: int = 252,
        test_period_days: int = 63,
        step_days: Optional[int] = None
    ):
        """
        Initialize walk-forward analysis.
        
        Args:
            data: Market data
            optimizer: Strategy optimizer instance
            train_period_days: Training period length
            test_period_days: Test period length
            step_days: Step size (default: test_period_days)
        """
        self.data = data
        self.optimizer = optimizer
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.step_days = step_days or test_period_days
        
        self.windows: List[WalkForwardWindow] = []
        self.combined_results: Optional[pd.DataFrame] = None
        
    def run(
        self,
        param_grid: Dict[str, List],
        optimization_method: str = "grid_search",
        progress_bar: bool = True
    ) -> Dict:
        """
        Run walk-forward analysis.
        
        Args:
            param_grid: Parameter search space
            optimization_method: Optimization method to use
            progress_bar: Show progress bar
            
        Returns:
            Dictionary with analysis results
        """
        # Generate windows
        windows = self._generate_windows()
        
        # Run optimization for each window
        all_equity_curves = []
        
        iterator = tqdm(windows, desc="Walk-forward windows") if progress_bar else windows
        
        for train_start, train_end, test_start, test_end in iterator:
            # Get training data
            train_data = self.data[
                (self.data.index >= train_start) & 
                (self.data.index <= train_end)
            ]
            
            # Get test data
            test_data = self.data[
                (self.data.index >= test_start) & 
                (self.data.index <= test_end)
            ]
            
            # Optimize on training data
            self.optimizer.data = train_data
            
            if optimization_method == "grid_search":
                in_sample_results = self.optimizer.grid_search(
                    param_grid,
                    progress_bar=False
                )
            elif optimization_method == "random_search":
                in_sample_results = self.optimizer.random_search(
                    param_grid,
                    n_iter=100,
                    progress_bar=False
                )
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")
                
            # Test on out-of-sample data
            best_params = in_sample_results.best_params
            out_sample_performance = self._test_parameters(
                test_data,
                best_params
            )
            
            # Store window results
            window = WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                in_sample_results=in_sample_results,
                out_sample_performance=out_sample_performance
            )
            self.windows.append(window)
            
            # Collect equity curve
            if 'equity_curve' in out_sample_performance:
                all_equity_curves.append(out_sample_performance['equity_curve'])
                
        # Combine results
        self._combine_results(all_equity_curves)
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_statistics()
        
        return {
            'windows': self.windows,
            'combined_equity_curve': self.combined_results,
            'overall_statistics': overall_stats,
            'parameter_stability': self._analyze_parameter_stability()
        }
        
    def _generate_windows(self) -> List[Tuple[pd.Timestamp, ...]]:
        """Generate train/test windows."""
        windows = []
        
        data_start = self.data.index[0]
        data_end = self.data.index[-1]
        
        # Start with first possible window
        train_start = data_start
        
        while True:
            train_end = train_start + pd.Timedelta(days=self.train_period_days)
            test_start = train_end
            test_end = test_start + pd.Timedelta(days=self.test_period_days)
            
            # Check if we have enough data
            if test_end > data_end:
                break
                
            windows.append((train_start, train_end, test_start, test_end))
            
            # Move to next window
            train_start += pd.Timedelta(days=self.step_days)
            
        return windows
        
    def _test_parameters(
        self,
        test_data: pd.DataFrame,
        params: Dict
    ) -> Dict:
        """Test parameters on out-of-sample data."""
        # Build strategy with parameters
        strategy = self.optimizer._build_strategy_with_params(params)
        
        # Run backtest
        engine = BacktestEngine(
            initial_capital=self.optimizer.initial_capital,
            commission_rate=self.optimizer.commission_rate
        )
        
        results = engine.run(
            test_data,
            strategy,
            progress_bar=False
        )
        
        # Calculate metrics
        metrics = {}
        if 'equity_curve' in results and not results['equity_curve'].empty:
            metrics_calc = PerformanceMetrics.calculate(
                results['equity_curve'],
                results.get('trades', pd.DataFrame())
            )
            
            metrics = {
                'total_return': metrics_calc.total_return,
                'sharpe_ratio': metrics_calc.sharpe_ratio,
                'max_drawdown': metrics_calc.max_drawdown,
                'win_rate': metrics_calc.win_rate,
                'total_trades': metrics_calc.total_trades,
                'equity_curve': results['equity_curve']
            }
            
        return metrics
        
    def _combine_results(self, equity_curves: List[pd.DataFrame]) -> None:
        """Combine equity curves from all windows."""
        if not equity_curves:
            return
            
        # Normalize and combine
        combined_data = []
        
        for i, curve in enumerate(equity_curves):
            if i == 0:
                # First curve starts at initial capital
                combined_data.append(curve)
            else:
                # Subsequent curves start from last value
                last_value = combined_data[-1]['total_value'].iloc[-1]
                scale_factor = last_value / curve['total_value'].iloc[0]
                scaled_curve = curve.copy()
                scaled_curve['total_value'] *= scale_factor
                combined_data.append(scaled_curve)
                
        self.combined_results = pd.concat(combined_data)
        
    def _calculate_overall_statistics(self) -> Dict:
        """Calculate statistics across all windows."""
        if not self.windows:
            return {}
            
        # Collect metrics from all windows
        in_sample_scores = []
        out_sample_returns = []
        out_sample_sharpes = []
        out_sample_drawdowns = []
        
        for window in self.windows:
            in_sample_scores.append(window.in_sample_results.best_score)
            
            out_perf = window.out_sample_performance
            if out_perf:
                out_sample_returns.append(out_perf.get('total_return', 0))
                out_sample_sharpes.append(out_perf.get('sharpe_ratio', 0))
                out_sample_drawdowns.append(out_perf.get('max_drawdown', 0))
                
        # Calculate overall performance
        if self.combined_results is not None and not self.combined_results.empty:
            overall_metrics = PerformanceMetrics.calculate(
                self.combined_results,
                pd.DataFrame()  # No trades for combined
            )
            
            overall_performance = {
                'total_return': overall_metrics.total_return,
                'annualized_return': overall_metrics.annualized_return,
                'sharpe_ratio': overall_metrics.sharpe_ratio,
                'max_drawdown': overall_metrics.max_drawdown,
                'volatility': overall_metrics.volatility
            }
        else:
            overall_performance = {}
            
        return {
            'n_windows': len(self.windows),
            'avg_in_sample_score': np.mean(in_sample_scores),
            'avg_out_sample_return': np.mean(out_sample_returns),
            'avg_out_sample_sharpe': np.mean(out_sample_sharpes),
            'avg_out_sample_drawdown': np.mean(out_sample_drawdowns),
            'out_sample_consistency': np.std(out_sample_returns),
            'overall_performance': overall_performance
        }
        
    def _analyze_parameter_stability(self) -> pd.DataFrame:
        """Analyze how stable parameters are across windows."""
        if not self.windows:
            return pd.DataFrame()
            
        # Collect best parameters from each window
        param_history = []
        
        for i, window in enumerate(self.windows):
            params = window.in_sample_results.best_params.copy()
            params['window'] = i
            params['test_start'] = window.test_start
            param_history.append(params)
            
        df = pd.DataFrame(param_history)
        
        # Calculate statistics for each parameter
        param_stats = {}
        
        for col in df.columns:
            if col not in ['window', 'test_start']:
                if df[col].dtype in [np.float64, np.int64]:
                    param_stats[col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'cv': df[col].std() / df[col].mean() if df[col].mean() != 0 else 0,
                        'min': df[col].min(),
                        'max': df[col].max()
                    }
                    
        return pd.DataFrame(param_stats).T
        
    def plot_results(self) -> Dict:
        """Generate plots for walk-forward analysis."""
        # This would be implemented in the visualization module
        pass