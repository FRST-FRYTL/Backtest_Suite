"""
Optimization Agent for ML Pipeline

Optimizes model hyperparameters, portfolio weights, and trading strategies
using various optimization techniques.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from scipy.optimize import minimize, differential_evolution, dual_annealing
from sklearn.model_selection import ParameterGrid
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent


class OptimizationAgent(BaseAgent):
    """
    Agent responsible for optimization tasks including:
    - Hyperparameter optimization
    - Portfolio weight optimization
    - Strategy parameter tuning
    - Multi-objective optimization
    - Constraint satisfaction
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("OptimizationAgent", config)
        self.optimization_history = []
        self.best_solution = None
        self.pareto_front = []
        self.convergence_data = []
        
    def initialize(self) -> bool:
        """Initialize optimization resources."""
        try:
            self.logger.info("Initializing Optimization Agent")
            
            # Validate required configuration
            required_keys = ["optimization_method", "objective", "constraints"]
            if not self.validate_config(required_keys):
                return False
            
            # Initialize optimization settings
            self.optimization_method = self.config.get("optimization_method", "optuna")
            self.objective = self.config.get("objective", "maximize_sharpe")
            self.constraints = self.config.get("constraints", {})
            
            # Initialize optimization parameters
            self.n_trials = self.config.get("n_trials", 100)
            self.timeout = self.config.get("timeout", 3600)  # 1 hour
            self.n_jobs = self.config.get("n_jobs", -1)
            
            # Initialize bounds and constraints
            self.bounds = self.config.get("bounds", {})
            self.constraint_functions = []
            
            # Multi-objective settings
            self.multi_objective = self.config.get("multi_objective", False)
            self.objectives = self.config.get("objectives", ["return", "risk"])
            
            self.logger.info("Optimization Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False
    
    def execute(self, objective_function: Callable, search_space: Dict[str, Any],
                initial_guess: Optional[Dict[str, float]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Execute optimization process.
        
        Args:
            objective_function: Function to optimize
            search_space: Parameter search space
            initial_guess: Initial parameter values
            
        Returns:
            Dict containing optimization results
        """
        try:
            # Select optimization method
            if self.optimization_method == "optuna":
                results = self._optimize_with_optuna(
                    objective_function, search_space, **kwargs
                )
            elif self.optimization_method == "scipy":
                results = self._optimize_with_scipy(
                    objective_function, search_space, initial_guess, **kwargs
                )
            elif self.optimization_method == "genetic":
                results = self._optimize_with_genetic(
                    objective_function, search_space, **kwargs
                )
            elif self.optimization_method == "grid_search":
                results = self._optimize_with_grid_search(
                    objective_function, search_space, **kwargs
                )
            else:
                results = self._optimize_with_bayesian(
                    objective_function, search_space, **kwargs
                )
            
            # Analyze optimization results
            analysis = self._analyze_optimization_results(results)
            
            # Perform sensitivity analysis
            sensitivity = self._perform_sensitivity_analysis(
                objective_function, results["best_params"], search_space
            )
            
            # Validate optimal solution
            validation = self._validate_solution(
                objective_function, results["best_params"], **kwargs
            )
            
            # Store results
            self.best_solution = results["best_params"]
            self.optimization_history.append(results)
            
            return {
                "best_params": results["best_params"],
                "best_value": results["best_value"],
                "optimization_history": results.get("history", []),
                "convergence_analysis": analysis,
                "sensitivity_analysis": sensitivity,
                "validation_results": validation,
                "optimization_metadata": {
                    "method": self.optimization_method,
                    "n_evaluations": results.get("n_evaluations", 0),
                    "time_elapsed": results.get("time_elapsed", 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            raise
    
    def _optimize_with_optuna(self, objective_function: Callable,
                            search_space: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Optimize using Optuna framework."""
        self.logger.info("Running Optuna optimization")
        
        import time
        start_time = time.time()
        
        # Create objective wrapper
        def optuna_objective(trial):
            # Sample parameters from search space
            params = {}
            for param_name, param_config in search_space.items():
                if param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, 
                        param_config["low"], 
                        param_config["high"],
                        log=param_config.get("log", False)
                    )
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False)
                    )
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config["choices"]
                    )
            
            # Evaluate objective
            try:
                value = objective_function(params, **kwargs)
                
                # Store intermediate values for convergence analysis
                self.convergence_data.append({
                    "trial": trial.number,
                    "value": value,
                    "params": params.copy()
                })
                
                return value
            except Exception as e:
                self.logger.warning(f"Trial {trial.number} failed: {str(e)}")
                return float('-inf') if self._is_maximization() else float('inf')
        
        # Select sampler
        if self.config.get("sampler", "tpe") == "cmaes":
            sampler = CmaEsSampler(seed=42)
        else:
            sampler = TPESampler(seed=42)
        
        # Create study
        study = optuna.create_study(
            direction="maximize" if self._is_maximization() else "minimize",
            sampler=sampler,
            pruner=MedianPruner()
        )
        
        # Add constraints if any
        if self.constraints:
            for constraint_name, constraint_config in self.constraints.items():
                study.add_user_attr(f"constraint_{constraint_name}", constraint_config)
        
        # Optimize
        study.optimize(
            optuna_objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=1  # Set to 1 to avoid multiprocessing issues
        )
        
        # Extract results
        history = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    "params": trial.params,
                    "value": trial.value,
                    "number": trial.number
                })
        
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "history": history,
            "n_evaluations": len(study.trials),
            "time_elapsed": time.time() - start_time,
            "study": study
        }
    
    def _optimize_with_scipy(self, objective_function: Callable,
                           search_space: Dict[str, Any],
                           initial_guess: Optional[Dict[str, float]],
                           **kwargs) -> Dict[str, Any]:
        """Optimize using SciPy optimization methods."""
        self.logger.info("Running SciPy optimization")
        
        import time
        start_time = time.time()
        
        # Convert search space to bounds
        param_names = list(search_space.keys())
        bounds = []
        for param in param_names:
            if search_space[param]["type"] in ["float", "int"]:
                bounds.append((search_space[param]["low"], search_space[param]["high"]))
            else:
                # Skip categorical parameters for scipy
                continue
        
        # Create initial guess if not provided
        if initial_guess is None:
            x0 = [(b[0] + b[1]) / 2 for b in bounds]
        else:
            x0 = [initial_guess.get(param, (bounds[i][0] + bounds[i][1]) / 2) 
                  for i, param in enumerate(param_names)]
        
        # Create objective wrapper
        def scipy_objective(x):
            params = {param_names[i]: x[i] for i in range(len(x))}
            value = objective_function(params, **kwargs)
            
            # Store for convergence analysis
            self.convergence_data.append({
                "iteration": len(self.convergence_data),
                "value": value,
                "params": params.copy()
            })
            
            # Minimize by default in scipy
            return -value if self._is_maximization() else value
        
        # Add constraints
        constraints = []
        if self.constraints:
            for constraint_name, constraint_config in self.constraints.items():
                if constraint_config["type"] == "ineq":
                    constraints.append({
                        "type": "ineq",
                        "fun": lambda x: constraint_config["limit"] - scipy_objective(x)
                    })
        
        # Optimize
        result = minimize(
            scipy_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.n_trials,
                'disp': True
            }
        )
        
        # Extract results
        best_params = {param_names[i]: result.x[i] for i in range(len(result.x))}
        best_value = -result.fun if self._is_maximization() else result.fun
        
        return {
            "best_params": best_params,
            "best_value": best_value,
            "history": self.convergence_data,
            "n_evaluations": result.nfev,
            "time_elapsed": time.time() - start_time,
            "success": result.success,
            "message": result.message
        }
    
    def _optimize_with_genetic(self, objective_function: Callable,
                             search_space: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Optimize using genetic algorithm (differential evolution)."""
        self.logger.info("Running genetic algorithm optimization")
        
        import time
        start_time = time.time()
        
        # Convert search space to bounds
        param_names = []
        bounds = []
        for param, config in search_space.items():
            if config["type"] in ["float", "int"]:
                param_names.append(param)
                bounds.append((config["low"], config["high"]))
        
        # Create objective wrapper
        def genetic_objective(x):
            params = {param_names[i]: x[i] for i in range(len(x))}
            value = objective_function(params, **kwargs)
            
            # Store for convergence analysis
            self.convergence_data.append({
                "generation": len(self.convergence_data) // 15,  # Approximate generation
                "value": value,
                "params": params.copy()
            })
            
            return -value if self._is_maximization() else value
        
        # Run differential evolution
        result = differential_evolution(
            genetic_objective,
            bounds,
            strategy='best1bin',
            maxiter=self.n_trials // 15,  # Generations
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=True
        )
        
        # Extract results
        best_params = {param_names[i]: result.x[i] for i in range(len(result.x))}
        best_value = -result.fun if self._is_maximization() else result.fun
        
        return {
            "best_params": best_params,
            "best_value": best_value,
            "history": self.convergence_data,
            "n_evaluations": result.nfev,
            "time_elapsed": time.time() - start_time,
            "success": result.success,
            "message": result.message
        }
    
    def _optimize_with_grid_search(self, objective_function: Callable,
                                 search_space: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Optimize using grid search."""
        self.logger.info("Running grid search optimization")
        
        import time
        start_time = time.time()
        
        # Create parameter grid
        param_grid = {}
        for param, config in search_space.items():
            if config["type"] == "float":
                param_grid[param] = np.linspace(
                    config["low"], config["high"], 
                    config.get("n_steps", 10)
                )
            elif config["type"] == "int":
                param_grid[param] = list(range(
                    config["low"], config["high"] + 1,
                    config.get("step", 1)
                ))
            elif config["type"] == "categorical":
                param_grid[param] = config["choices"]
        
        # Generate all combinations
        grid = ParameterGrid(param_grid)
        
        # Evaluate all combinations
        results = []
        best_value = float('-inf') if self._is_maximization() else float('inf')
        best_params = None
        
        for i, params in enumerate(grid):
            try:
                value = objective_function(params, **kwargs)
                results.append({
                    "params": params,
                    "value": value,
                    "iteration": i
                })
                
                # Update best
                if self._is_maximization() and value > best_value:
                    best_value = value
                    best_params = params.copy()
                elif not self._is_maximization() and value < best_value:
                    best_value = value
                    best_params = params.copy()
                
            except Exception as e:
                self.logger.warning(f"Grid point {i} failed: {str(e)}")
        
        return {
            "best_params": best_params,
            "best_value": best_value,
            "history": results,
            "n_evaluations": len(results),
            "time_elapsed": time.time() - start_time,
            "grid_size": len(grid)
        }
    
    def _optimize_with_bayesian(self, objective_function: Callable,
                              search_space: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Optimize using Bayesian optimization."""
        self.logger.info("Running Bayesian optimization")
        
        # For this implementation, we'll use Optuna with Gaussian Process
        # In practice, you might want to use specialized libraries like scikit-optimize
        
        return self._optimize_with_optuna(objective_function, search_space, **kwargs)
    
    def _is_maximization(self) -> bool:
        """Check if objective is maximization."""
        return self.objective.startswith("maximize") or self.objective in ["sharpe", "return"]
    
    def _analyze_optimization_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization convergence and efficiency."""
        self.logger.info("Analyzing optimization results")
        
        history = results.get("history", [])
        if not history:
            return {"message": "No optimization history available"}
        
        # Extract values
        if isinstance(history[0], dict):
            values = [h["value"] for h in history]
        else:
            values = history
        
        # Calculate convergence metrics
        best_values = []
        current_best = float('-inf') if self._is_maximization() else float('inf')
        
        for value in values:
            if self._is_maximization() and value > current_best:
                current_best = value
            elif not self._is_maximization() and value < current_best:
                current_best = value
            best_values.append(current_best)
        
        # Convergence analysis
        convergence_rate = self._calculate_convergence_rate(best_values)
        efficiency = self._calculate_optimization_efficiency(values, best_values)
        
        return {
            "convergence_rate": convergence_rate,
            "efficiency_metrics": efficiency,
            "final_improvement": float(
                (best_values[-1] - best_values[0]) / abs(best_values[0])
            ) if best_values[0] != 0 else 0,
            "iterations_to_convergence": self._find_convergence_point(best_values),
            "value_statistics": {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "best": float(best_values[-1]),
                "worst": float(min(values) if self._is_maximization() else max(values))
            }
        }
    
    def _calculate_convergence_rate(self, best_values: List[float]) -> float:
        """Calculate convergence rate."""
        if len(best_values) < 2:
            return 0.0
        
        # Fit exponential decay/growth
        x = np.arange(len(best_values))
        y = np.array(best_values)
        
        # Normalize to [0, 1]
        y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y
        
        # Fit linear regression to log values
        if np.all(y_norm > 0):
            log_y = np.log(y_norm + 1e-10)
            slope, _ = np.polyfit(x, log_y, 1)
            return float(abs(slope))
        
        return 0.0
    
    def _calculate_optimization_efficiency(self, values: List[float], 
                                         best_values: List[float]) -> Dict[str, float]:
        """Calculate optimization efficiency metrics."""
        # Function evaluations to reach 90% of final improvement
        final_improvement = best_values[-1] - best_values[0]
        target = best_values[0] + 0.9 * final_improvement
        
        evals_to_90 = len(best_values)
        for i, value in enumerate(best_values):
            if (self._is_maximization() and value >= target) or \
               (not self._is_maximization() and value <= target):
                evals_to_90 = i + 1
                break
        
        # Random search efficiency
        random_efficiency = len([v for v in values if 
                               (self._is_maximization() and v > np.median(values)) or
                               (not self._is_maximization() and v < np.median(values))]) / len(values)
        
        return {
            "evaluations_to_90_percent": evals_to_90,
            "efficiency_vs_random": float(random_efficiency),
            "improvement_per_evaluation": float(
                (best_values[-1] - best_values[0]) / len(best_values)
            ) if len(best_values) > 0 else 0
        }
    
    def _find_convergence_point(self, best_values: List[float], 
                               tolerance: float = 0.001) -> int:
        """Find iteration where optimization converged."""
        if len(best_values) < 10:
            return len(best_values)
        
        # Look for point where improvement becomes negligible
        window = 10
        for i in range(window, len(best_values)):
            recent_values = best_values[i-window:i]
            if np.std(recent_values) / abs(np.mean(recent_values)) < tolerance:
                return i
        
        return len(best_values)
    
    def _perform_sensitivity_analysis(self, objective_function: Callable,
                                    best_params: Dict[str, float],
                                    search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Perform sensitivity analysis around optimal solution."""
        self.logger.info("Performing sensitivity analysis")
        
        sensitivity_results = {}
        base_value = objective_function(best_params)
        
        # Analyze each parameter
        for param_name, param_value in best_params.items():
            if param_name not in search_space:
                continue
                
            param_config = search_space[param_name]
            if param_config["type"] == "categorical":
                continue
                
            # Create perturbations
            perturbations = np.linspace(-0.1, 0.1, 11)  # ±10% range
            param_sensitivity = []
            
            for perturb in perturbations:
                # Create perturbed parameters
                perturbed_params = best_params.copy()
                
                if param_config["type"] == "float":
                    new_value = param_value * (1 + perturb)
                    new_value = np.clip(new_value, param_config["low"], param_config["high"])
                else:  # int
                    new_value = int(param_value * (1 + perturb))
                    new_value = np.clip(new_value, param_config["low"], param_config["high"])
                
                perturbed_params[param_name] = new_value
                
                # Evaluate
                try:
                    perturbed_value = objective_function(perturbed_params)
                    param_sensitivity.append({
                        "perturbation": float(perturb),
                        "value": float(perturbed_value),
                        "change": float((perturbed_value - base_value) / base_value) 
                                 if base_value != 0 else 0
                    })
                except:
                    param_sensitivity.append({
                        "perturbation": float(perturb),
                        "value": None,
                        "change": None
                    })
            
            # Calculate sensitivity metrics
            valid_changes = [p["change"] for p in param_sensitivity if p["change"] is not None]
            if valid_changes:
                sensitivity_results[param_name] = {
                    "sensitivity_data": param_sensitivity,
                    "average_sensitivity": float(np.mean(np.abs(valid_changes))),
                    "max_change": float(max(np.abs(valid_changes))),
                    "is_sensitive": max(np.abs(valid_changes)) > 0.05  # 5% threshold
                }
        
        return sensitivity_results
    
    def _validate_solution(self, objective_function: Callable,
                         best_params: Dict[str, float], **kwargs) -> Dict[str, Any]:
        """Validate optimal solution with additional checks."""
        self.logger.info("Validating optimal solution")
        
        validation_results = {
            "is_valid": True,
            "checks": {}
        }
        
        # Re-evaluate best solution
        try:
            validation_value = objective_function(best_params, **kwargs)
            validation_results["checks"]["re_evaluation"] = {
                "passed": True,
                "value": float(validation_value)
            }
        except Exception as e:
            validation_results["checks"]["re_evaluation"] = {
                "passed": False,
                "error": str(e)
            }
            validation_results["is_valid"] = False
        
        # Check constraints
        if self.constraints:
            for constraint_name, constraint_config in self.constraints.items():
                if constraint_config["type"] == "bounds":
                    param = constraint_config["parameter"]
                    if param in best_params:
                        value = best_params[param]
                        passed = (constraint_config["min"] <= value <= constraint_config["max"])
                        validation_results["checks"][f"constraint_{constraint_name}"] = {
                            "passed": passed,
                            "value": value,
                            "bounds": [constraint_config["min"], constraint_config["max"]]
                        }
                        if not passed:
                            validation_results["is_valid"] = False
        
        # Stability check - evaluate nearby points
        stability_check = self._check_solution_stability(
            objective_function, best_params, kwargs
        )
        validation_results["checks"]["stability"] = stability_check
        
        if not stability_check["is_stable"]:
            validation_results["is_valid"] = False
        
        return validation_results
    
    def _check_solution_stability(self, objective_function: Callable,
                                params: Dict[str, float],
                                kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Check if solution is stable in neighborhood."""
        base_value = objective_function(params, **kwargs)
        
        # Test small perturbations
        n_tests = 10
        perturbation_size = 0.01  # 1%
        
        values = []
        for _ in range(n_tests):
            perturbed_params = params.copy()
            for param_name, param_value in params.items():
                if isinstance(param_value, (int, float)):
                    noise = np.random.normal(0, perturbation_size * abs(param_value))
                    perturbed_params[param_name] = param_value + noise
            
            try:
                value = objective_function(perturbed_params, **kwargs)
                values.append(value)
            except:
                pass
        
        if values:
            variation = np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else 0
            is_stable = variation < 0.05  # 5% coefficient of variation
        else:
            is_stable = False
            variation = float('inf')
        
        return {
            "is_stable": is_stable,
            "coefficient_of_variation": float(variation),
            "n_successful_evaluations": len(values)
        }
    
    def optimize_portfolio_weights(self, returns: pd.DataFrame,
                                 constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize portfolio weights for given returns."""
        self.logger.info("Optimizing portfolio weights")
        
        n_assets = returns.shape[1]
        
        # Define objective function (Sharpe ratio)
        def portfolio_objective(weights):
            weights = np.array(weights)
            portfolio_returns = returns @ weights
            
            if portfolio_returns.std() == 0:
                return 0
            
            sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
            return float(sharpe)
        
        # Set up constraints
        scipy_constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        if constraints:
            if "min_weight" in constraints:
                scipy_constraints.append({
                    "type": "ineq",
                    "fun": lambda w: w - constraints["min_weight"]
                })
            if "max_weight" in constraints:
                scipy_constraints.append({
                    "type": "ineq", 
                    "fun": lambda w: constraints["max_weight"] - w
                })
        
        # Initial guess (equal weights)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Bounds (0 to 1 for each weight)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Optimize
        result = minimize(
            lambda w: -portfolio_objective(w),  # Minimize negative Sharpe
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=scipy_constraints
        )
        
        optimal_weights = result.x
        optimal_sharpe = -result.fun
        
        # Calculate portfolio metrics
        portfolio_returns = returns @ optimal_weights
        
        return {
            "optimal_weights": {
                returns.columns[i]: float(optimal_weights[i]) 
                for i in range(n_assets)
            },
            "portfolio_metrics": {
                "sharpe_ratio": float(optimal_sharpe),
                "annual_return": float(portfolio_returns.mean() * 252),
                "annual_volatility": float(portfolio_returns.std() * np.sqrt(252)),
                "max_drawdown": float(self._calculate_max_drawdown(portfolio_returns))
            },
            "optimization_success": result.success,
            "optimization_message": result.message
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return float(drawdown.min())
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        if not self.optimization_history:
            return {"message": "No optimization history available"}
        
        latest = self.optimization_history[-1]
        
        return {
            "total_optimizations": len(self.optimization_history),
            "best_solution": self.best_solution,
            "latest_optimization": {
                "method": self.optimization_method,
                "best_value": latest.get("best_value"),
                "n_evaluations": latest.get("n_evaluations"),
                "time_elapsed": latest.get("time_elapsed")
            },
            "convergence_data": {
                "total_evaluations": len(self.convergence_data),
                "improvement_achieved": bool(self.best_solution)
            }
        }