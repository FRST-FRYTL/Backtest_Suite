"""
Optimization Orchestrator for Managing 5-Loop ML Optimization System

This module coordinates the execution of five optimization loops:
1. Feature Engineering Optimization
2. Model Architecture Optimization
3. Market Regime Optimization
4. Risk Management Optimization
5. Integration & Ensemble Optimization
"""

import os
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from dataclasses import dataclass, asdict
import yaml

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Store results from each optimization loop"""
    loop_name: str
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    improvement: float
    timestamp: datetime
    study_name: str
    convergence_history: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class OptimizationOrchestrator:
    """
    Orchestrates the 5-loop optimization process for ML trading strategies
    """
    
    def __init__(self, config_path: str = "config/optimization_config.yaml"):
        """
        Initialize the optimization orchestrator
        
        Args:
            config_path: Path to optimization configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.results_dir = Path(self.config['general']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimization loops
        self.loops = self._initialize_loops()
        
        # Track optimization progress
        self.loop_results: Dict[str, OptimizationResult] = {}
        self.baseline_performance: Optional[float] = None
        self.current_performance: Optional[float] = None
        
        # Set up logging
        self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load optimization configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Configure logging for optimization process"""
        log_file = self.results_dir / f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    def _initialize_loops(self) -> Dict[str, Any]:
        """Initialize all optimization loops"""
        from .feature_optimization import FeatureOptimization
        from .architecture_optimization import ArchitectureOptimization
        from .regime_optimization import RegimeOptimization
        from .risk_optimization import RiskOptimization
        from .integration_optimization import IntegrationOptimization
        
        return {
            'feature': FeatureOptimization(self.config['feature_optimization']),
            'architecture': ArchitectureOptimization(self.config['architecture_optimization']),
            'regime': RegimeOptimization(self.config['regime_optimization']),
            'risk': RiskOptimization(self.config['risk_optimization']),
            'integration': IntegrationOptimization(self.config['integration_optimization'])
        }
    
    def run_optimization(self, 
                        data: pd.DataFrame,
                        initial_params: Optional[Dict[str, Any]] = None,
                        n_loops: int = 5) -> Dict[str, Any]:
        """
        Run the complete 5-loop optimization process
        
        Args:
            data: Market data for optimization
            initial_params: Initial parameters (optional)
            n_loops: Number of optimization loops to run
            
        Returns:
            Dictionary containing best parameters and results
        """
        logger.info(f"Starting {n_loops}-loop optimization process")
        
        # Initialize parameters
        current_params = initial_params or self._get_default_params()
        
        # Evaluate baseline performance
        self.baseline_performance = self._evaluate_baseline(data, current_params)
        logger.info(f"Baseline performance: {self.baseline_performance:.4f}")
        
        # Run optimization loops
        loop_sequence = ['feature', 'architecture', 'regime', 'risk', 'integration']
        
        for loop_idx in range(n_loops):
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting optimization round {loop_idx + 1}/{n_loops}")
            logger.info(f"{'='*50}")
            
            for loop_name in loop_sequence:
                logger.info(f"\nRunning {loop_name} optimization...")
                
                # Run optimization loop
                result = self._run_single_loop(
                    loop_name, 
                    data, 
                    current_params
                )
                
                # Update parameters with optimized values
                current_params = self._merge_params(current_params, result.best_params)
                
                # Store results
                self.loop_results[f"{loop_name}_{loop_idx}"] = result
                
                # Log improvement
                improvement_pct = result.improvement * 100
                logger.info(
                    f"{loop_name} optimization completed. "
                    f"Improvement: {improvement_pct:.2f}%"
                )
            
            # Evaluate overall performance after each round
            self.current_performance = self._evaluate_performance(data, current_params)
            overall_improvement = (
                (self.current_performance - self.baseline_performance) / 
                abs(self.baseline_performance) * 100
            )
            
            logger.info(f"\nRound {loop_idx + 1} completed.")
            logger.info(f"Current performance: {self.current_performance:.4f}")
            logger.info(f"Total improvement: {overall_improvement:.2f}%")
            
            # Check early stopping
            if self._should_stop_early(loop_idx):
                logger.info("Early stopping criteria met. Ending optimization.")
                break
        
        # Save final results
        final_results = self._compile_final_results(current_params)
        self._save_results(final_results)
        
        return final_results
    
    def _run_single_loop(self, 
                        loop_name: str, 
                        data: pd.DataFrame,
                        current_params: Dict[str, Any]) -> OptimizationResult:
        """Run a single optimization loop"""
        loop = self.loops[loop_name]
        
        # Create Optuna study
        study_name = f"{loop_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sampler = TPESampler(seed=self.config['general']['random_seed'])
        
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=sampler
        )
        
        # Define objective function
        def objective(trial):
            # Get hyperparameters from trial
            trial_params = loop.get_trial_params(trial)
            
            # Merge with current parameters
            merged_params = self._merge_params(current_params, trial_params)
            
            # Evaluate performance
            performance = loop.evaluate(data, merged_params)
            
            return performance
        
        # Run optimization
        n_trials = self.config[f'{loop_name}_optimization']['n_trials']
        study.optimize(
            objective, 
            n_trials=n_trials,
            callbacks=[self._optuna_callback]
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        # Calculate improvement
        baseline = loop.evaluate(data, current_params)
        improvement = (best_value - baseline) / abs(baseline) if baseline != 0 else 0
        
        # Get convergence history
        convergence_history = [trial.value for trial in study.trials if trial.value is not None]
        
        return OptimizationResult(
            loop_name=loop_name,
            best_params=best_params,
            best_value=best_value,
            n_trials=len(study.trials),
            improvement=improvement,
            timestamp=datetime.now(),
            study_name=study_name,
            convergence_history=convergence_history
        )
    
    def _optuna_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback for Optuna optimization"""
        if trial.number % 10 == 0:
            logger.info(
                f"Trial {trial.number}: {trial.value:.4f} "
                f"(best: {study.best_value:.4f})"
            )
    
    def _merge_params(self, 
                     base_params: Dict[str, Any], 
                     new_params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge parameter dictionaries"""
        merged = base_params.copy()
        
        for key, value in new_params.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = self._merge_params(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _evaluate_baseline(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """Evaluate baseline performance"""
        # Use integration loop for overall performance evaluation
        return self.loops['integration'].evaluate(data, params)
    
    def _evaluate_performance(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """Evaluate current performance"""
        return self.loops['integration'].evaluate(data, params)
    
    def _should_stop_early(self, loop_idx: int) -> bool:
        """Check if optimization should stop early"""
        if loop_idx < 1:
            return False
        
        # Get early stopping config
        min_improvement = self.config['general']['early_stopping']['min_improvement']
        patience = self.config['general']['early_stopping']['patience']
        
        # Check recent improvements
        recent_improvements = []
        for i in range(max(0, loop_idx - patience + 1), loop_idx + 1):
            round_results = [
                r for k, r in self.loop_results.items() 
                if k.endswith(f"_{i}")
            ]
            if round_results:
                avg_improvement = np.mean([r.improvement for r in round_results])
                recent_improvements.append(avg_improvement)
        
        # Stop if recent improvements are below threshold
        if recent_improvements:
            return all(imp < min_improvement for imp in recent_improvements)
        
        return False
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters from config"""
        return self.config['default_parameters']
    
    def _compile_final_results(self, best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Compile final optimization results"""
        # Calculate total improvement
        total_improvement = 0
        if self.baseline_performance and self.current_performance:
            total_improvement = (
                (self.current_performance - self.baseline_performance) / 
                abs(self.baseline_performance)
            )
        
        # Get best results from each loop type
        loop_summaries = {}
        for loop_name in ['feature', 'architecture', 'regime', 'risk', 'integration']:
            loop_results = [
                r for k, r in self.loop_results.items() 
                if k.startswith(loop_name)
            ]
            if loop_results:
                best_loop_result = max(loop_results, key=lambda x: x.best_value)
                loop_summaries[loop_name] = best_loop_result.to_dict()
        
        return {
            'best_params': best_params,
            'baseline_performance': self.baseline_performance,
            'final_performance': self.current_performance,
            'total_improvement': total_improvement,
            'loop_summaries': loop_summaries,
            'all_results': {k: v.to_dict() for k, v in self.loop_results.items()},
            'optimization_completed': datetime.now().isoformat()
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save optimization results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON
        json_path = self.results_dir / f"optimization_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save best parameters separately
        params_path = self.results_dir / f"best_params_{timestamp}.yaml"
        with open(params_path, 'w') as f:
            yaml.dump(results['best_params'], f)
        
        # Save as pickle for full object preservation
        pickle_path = self.results_dir / f"optimization_results_{timestamp}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved to {self.results_dir}")
    
    def load_best_params(self, results_file: str) -> Dict[str, Any]:
        """Load best parameters from a previous optimization run"""
        results_path = self.results_dir / results_file
        
        if results_path.suffix == '.json':
            with open(results_path, 'r') as f:
                results = json.load(f)
        elif results_path.suffix == '.pkl':
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {results_path.suffix}")
        
        return results['best_params']
    
    def visualize_optimization_progress(self):
        """Generate visualization of optimization progress"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot convergence for each loop type
        loop_names = ['feature', 'architecture', 'regime', 'risk', 'integration']
        
        for idx, loop_name in enumerate(loop_names):
            ax = axes[idx]
            
            # Get all results for this loop type
            loop_results = [
                (k, v) for k, v in self.loop_results.items() 
                if k.startswith(loop_name)
            ]
            
            for name, result in loop_results:
                if result.convergence_history:
                    ax.plot(result.convergence_history, label=name, alpha=0.7)
            
            ax.set_title(f'{loop_name.capitalize()} Optimization')
            ax.set_xlabel('Trial')
            ax.set_ylabel('Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot overall improvement
        ax = axes[5]
        rounds = []
        improvements = []
        
        for i in range(len(self.loop_results) // 5):
            round_results = [
                r for k, r in self.loop_results.items() 
                if k.endswith(f"_{i}")
            ]
            if round_results:
                avg_improvement = np.mean([r.improvement for r in round_results])
                rounds.append(i + 1)
                improvements.append(avg_improvement * 100)
        
        ax.bar(rounds, improvements)
        ax.set_title('Average Improvement per Round')
        ax.set_xlabel('Optimization Round')
        ax.set_ylabel('Improvement (%)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f"optimization_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {plot_path}")