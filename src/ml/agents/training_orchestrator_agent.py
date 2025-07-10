"""
Training Orchestrator Agent for ML Pipeline

Orchestrates the model training process including hyperparameter tuning,
cross-validation, and training optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, TimeSeriesSplit,
    StratifiedKFold, KFold, cross_validate
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
import joblib
import time
import gc
import psutil
import os
from datetime import datetime
import json
import optuna
from optuna.samplers import TPESampler

from .base_agent import BaseAgent


class TrainingOrchestratorAgent(BaseAgent):
    """
    Agent responsible for orchestrating model training including:
    - Hyperparameter optimization
    - Cross-validation strategies
    - Training monitoring and early stopping
    - Resource management
    - Model checkpointing and versioning
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("TrainingOrchestratorAgent", config)
        self.trained_models = {}
        self.training_history = []
        self.best_model = None
        self.best_params = None
        self.training_metrics = {}
        self.resource_usage = {}
        
    def initialize(self) -> bool:
        """Initialize training orchestrator resources."""
        try:
            self.logger.info("Initializing Training Orchestrator Agent")
            
            # Validate required configuration
            required_keys = ["optimization_method", "cv_strategy", "metric"]
            if not self.validate_config(required_keys):
                return False
            
            # Initialize optimization settings
            self.optimization_method = self.config.get("optimization_method", "random_search")
            self.cv_strategy = self.config.get("cv_strategy", "kfold")
            self.metric = self.config.get("metric", "mse")
            self.n_trials = self.config.get("n_trials", 50)
            
            # Initialize training settings
            self.early_stopping = self.config.get("early_stopping", {
                "enabled": True,
                "patience": 10,
                "min_delta": 0.0001
            })
            
            # Initialize resource limits
            self.resource_limits = self.config.get("resource_limits", {
                "max_time_minutes": 60,
                "max_memory_gb": 8,
                "n_jobs": -1
            })
            
            # Create model checkpoint directory
            self.checkpoint_dir = self.config.get("checkpoint_dir", "models/checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            self.logger.info("Training Orchestrator Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False
    
    def execute(self, model_config: Dict[str, Any], X_train: pd.DataFrame, 
                y_train: pd.Series, X_val: Optional[pd.DataFrame] = None,
                y_val: Optional[pd.Series] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute model training orchestration.
        
        Args:
            model_config: Model configuration from architecture agent
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Dict containing trained models and training results
        """
        try:
            # Monitor initial resources
            initial_resources = self._get_resource_usage()
            
            # Set up cross-validation strategy
            cv_splitter = self._create_cv_splitter(X_train, y_train)
            
            # Prepare model and search space
            model_class = model_config["model_class"]
            base_params = model_config.get("base_params", {})
            search_space = model_config.get("search_space", {})
            
            # Execute hyperparameter optimization
            if search_space:
                optimization_results = self._optimize_hyperparameters(
                    model_class, base_params, search_space,
                    X_train, y_train, cv_splitter
                )
            else:
                # Train with default parameters
                optimization_results = self._train_default_model(
                    model_class, base_params, X_train, y_train, cv_splitter
                )
            
            # Train final model with best parameters
            final_model = self._train_final_model(
                model_class, optimization_results["best_params"],
                X_train, y_train, X_val, y_val
            )
            
            # Evaluate model performance
            evaluation_results = self._evaluate_model(
                final_model, X_train, y_train, X_val, y_val
            )
            
            # Save model checkpoint
            checkpoint_path = self._save_checkpoint(
                final_model, optimization_results, evaluation_results
            )
            
            # Monitor final resources
            final_resources = self._get_resource_usage()
            resource_usage = self._calculate_resource_usage(
                initial_resources, final_resources
            )
            
            # Store results
            self.best_model = final_model
            self.best_params = optimization_results["best_params"]
            self.training_metrics = evaluation_results
            
            return {
                "model": final_model,
                "best_params": optimization_results["best_params"],
                "optimization_history": optimization_results.get("history", []),
                "cv_scores": optimization_results.get("cv_scores", {}),
                "evaluation_metrics": evaluation_results,
                "resource_usage": resource_usage,
                "checkpoint_path": checkpoint_path,
                "training_time": optimization_results.get("training_time", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            raise
    
    def _create_cv_splitter(self, X: pd.DataFrame, y: pd.Series):
        """Create appropriate cross-validation splitter."""
        self.logger.info(f"Creating CV splitter: {self.cv_strategy}")
        
        n_splits = self.config.get("cv_folds", 5)
        
        if self.cv_strategy == "timeseries":
            return TimeSeriesSplit(n_splits=n_splits)
        elif self.cv_strategy == "stratified":
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:  # kfold
            return KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    def _optimize_hyperparameters(self, model_class, base_params: Dict[str, Any],
                                search_space: Dict[str, Any], X: pd.DataFrame,
                                y: pd.Series, cv_splitter) -> Dict[str, Any]:
        """Optimize hyperparameters using configured method."""
        self.logger.info(f"Starting hyperparameter optimization: {self.optimization_method}")
        
        start_time = time.time()
        
        if self.optimization_method == "optuna":
            results = self._optuna_optimization(
                model_class, base_params, search_space, X, y, cv_splitter
            )
        elif self.optimization_method == "grid_search":
            results = self._grid_search_optimization(
                model_class, base_params, search_space, X, y, cv_splitter
            )
        else:  # random_search
            results = self._random_search_optimization(
                model_class, base_params, search_space, X, y, cv_splitter
            )
        
        results["training_time"] = time.time() - start_time
        self.logger.info(f"Optimization completed in {results['training_time']:.2f} seconds")
        
        return results
    
    def _optuna_optimization(self, model_class, base_params: Dict[str, Any],
                           search_space: Dict[str, Any], X: pd.DataFrame,
                           y: pd.Series, cv_splitter) -> Dict[str, Any]:
        """Optimize using Optuna framework."""
        self.logger.info("Running Optuna optimization")
        
        def objective(trial):
            # Sample hyperparameters
            params = base_params.copy()
            for param_name, param_range in search_space.items():
                if isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
                elif isinstance(param_range, tuple) and len(param_range) == 2:
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(
                            param_name, param_range[0], param_range[1]
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, param_range[0], param_range[1]
                        )
            
            # Create model
            model = model_class(**params)
            
            # Cross-validation
            scores = cross_validate(
                model, X, y, cv=cv_splitter,
                scoring=self._get_scoring_function(),
                n_jobs=self.resource_limits.get("n_jobs", -1)
            )
            
            return scores['test_score'].mean()
        
        # Create study
        study = optuna.create_study(
            direction="maximize" if self.metric in ["accuracy", "f1", "roc_auc", "r2"] else "minimize",
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(
            objective, 
            n_trials=self.n_trials,
            timeout=self.resource_limits.get("max_time_minutes", 60) * 60
        )
        
        # Get results
        history = []
        for trial in study.trials:
            history.append({
                "params": trial.params,
                "value": trial.value,
                "state": str(trial.state)
            })
        
        return {
            "best_params": {**base_params, **study.best_params},
            "best_score": study.best_value,
            "history": history,
            "n_trials": len(study.trials)
        }
    
    def _grid_search_optimization(self, model_class, base_params: Dict[str, Any],
                                search_space: Dict[str, Any], X: pd.DataFrame,
                                y: pd.Series, cv_splitter) -> Dict[str, Any]:
        """Optimize using GridSearchCV."""
        self.logger.info("Running Grid Search optimization")
        
        # Create base model
        model = model_class(**base_params)
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            model, search_space, cv=cv_splitter,
            scoring=self._get_scoring_function(),
            n_jobs=self.resource_limits.get("n_jobs", -1),
            verbose=1
        )
        
        # Fit
        grid_search.fit(X, y)
        
        # Extract results
        results_df = pd.DataFrame(grid_search.cv_results_)
        history = []
        for idx, params in enumerate(results_df["params"]):
            history.append({
                "params": params,
                "mean_score": results_df.loc[idx, "mean_test_score"],
                "std_score": results_df.loc[idx, "std_test_score"]
            })
        
        return {
            "best_params": {**base_params, **grid_search.best_params_},
            "best_score": grid_search.best_score_,
            "history": history,
            "cv_scores": {
                "mean": results_df.loc[grid_search.best_index_, "mean_test_score"],
                "std": results_df.loc[grid_search.best_index_, "std_test_score"]
            }
        }
    
    def _random_search_optimization(self, model_class, base_params: Dict[str, Any],
                                  search_space: Dict[str, Any], X: pd.DataFrame,
                                  y: pd.Series, cv_splitter) -> Dict[str, Any]:
        """Optimize using RandomizedSearchCV."""
        self.logger.info("Running Random Search optimization")
        
        # Create base model
        model = model_class(**base_params)
        
        # Create RandomizedSearchCV
        random_search = RandomizedSearchCV(
            model, search_space, n_iter=self.n_trials,
            cv=cv_splitter, scoring=self._get_scoring_function(),
            n_jobs=self.resource_limits.get("n_jobs", -1),
            random_state=42, verbose=1
        )
        
        # Fit
        random_search.fit(X, y)
        
        # Extract results
        results_df = pd.DataFrame(random_search.cv_results_)
        history = []
        for idx, params in enumerate(results_df["params"]):
            history.append({
                "params": params,
                "mean_score": results_df.loc[idx, "mean_test_score"],
                "std_score": results_df.loc[idx, "std_test_score"]
            })
        
        return {
            "best_params": {**base_params, **random_search.best_params_},
            "best_score": random_search.best_score_,
            "history": history,
            "cv_scores": {
                "mean": results_df.loc[random_search.best_index_, "mean_test_score"],
                "std": results_df.loc[random_search.best_index_, "std_test_score"]
            }
        }
    
    def _train_default_model(self, model_class, params: Dict[str, Any],
                           X: pd.DataFrame, y: pd.Series, cv_splitter) -> Dict[str, Any]:
        """Train model with default parameters."""
        self.logger.info("Training with default parameters")
        
        model = model_class(**params)
        
        # Cross-validation
        cv_results = cross_validate(
            model, X, y, cv=cv_splitter,
            scoring=self._get_scoring_function(),
            n_jobs=self.resource_limits.get("n_jobs", -1),
            return_train_score=True
        )
        
        return {
            "best_params": params,
            "best_score": cv_results['test_score'].mean(),
            "cv_scores": {
                "train_mean": cv_results['train_score'].mean(),
                "train_std": cv_results['train_score'].std(),
                "test_mean": cv_results['test_score'].mean(),
                "test_std": cv_results['test_score'].std()
            }
        }
    
    def _train_final_model(self, model_class, best_params: Dict[str, Any],
                         X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: Optional[pd.DataFrame] = None,
                         y_val: Optional[pd.Series] = None):
        """Train final model with best parameters."""
        self.logger.info("Training final model with best parameters")
        
        # Create model with best parameters
        model = model_class(**best_params)
        
        # Check if model supports early stopping
        if hasattr(model, 'fit') and 'eval_set' in model.fit.__code__.co_varnames:
            if X_val is not None and y_val is not None:
                self.logger.info("Using validation set for early stopping")
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=self.early_stopping.get("patience", 10),
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
        else:
            # Standard fit
            model.fit(X_train, y_train)
        
        return model
    
    def _evaluate_model(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: Optional[pd.DataFrame] = None,
                       y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Evaluate trained model."""
        self.logger.info("Evaluating model performance")
        
        results = {}
        
        # Training set evaluation
        train_pred = model.predict(X_train)
        results["train"] = self._calculate_metrics(y_train, train_pred)
        
        # Validation set evaluation
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val)
            results["validation"] = self._calculate_metrics(y_val, val_pred)
            
            # Calculate overfitting metrics
            results["overfitting"] = {
                "train_val_gap": abs(
                    results["train"][self.metric] - results["validation"][self.metric]
                )
            }
        
        return results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        if self.metric in ["mse", "rmse", "mae", "r2"]:  # Regression metrics
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["r2"] = r2_score(y_true, y_pred)
        else:  # Classification metrics
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            
            # Handle multi-class vs binary
            average = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
            metrics["precision"] = precision_score(y_true, y_pred, average=average)
            metrics["recall"] = recall_score(y_true, y_pred, average=average)
            metrics["f1"] = f1_score(y_true, y_pred, average=average)
            
            # ROC AUC for binary classification
            if len(np.unique(y_true)) == 2 and hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_val)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        
        return metrics
    
    def _get_scoring_function(self) -> str:
        """Get sklearn scoring function name."""
        scoring_map = {
            "mse": "neg_mean_squared_error",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
            "accuracy": "accuracy",
            "f1": "f1_weighted",
            "roc_auc": "roc_auc"
        }
        return scoring_map.get(self.metric, "neg_mean_squared_error")
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        process = psutil.Process(os.getpid())
        return {
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(interval=0.1),
            "num_threads": process.num_threads()
        }
    
    def _calculate_resource_usage(self, initial: Dict[str, float],
                                final: Dict[str, float]) -> Dict[str, Any]:
        """Calculate resource usage during training."""
        return {
            "peak_memory_mb": final["memory_mb"],
            "memory_increase_mb": final["memory_mb"] - initial["memory_mb"],
            "avg_cpu_percent": (initial["cpu_percent"] + final["cpu_percent"]) / 2,
            "max_threads": max(initial["num_threads"], final["num_threads"])
        }
    
    def _save_checkpoint(self, model, optimization_results: Dict[str, Any],
                        evaluation_results: Dict[str, Any]) -> str:
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model.__class__.__name__
        
        checkpoint = {
            "timestamp": timestamp,
            "model_name": model_name,
            "best_params": optimization_results["best_params"],
            "cv_score": optimization_results["best_score"],
            "evaluation_metrics": evaluation_results,
            "config": self.config
        }
        
        # Save model
        model_path = os.path.join(self.checkpoint_dir, f"{model_name}_{timestamp}.pkl")
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata_path = os.path.join(self.checkpoint_dir, f"{model_name}_{timestamp}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        self.logger.info(f"Model checkpoint saved: {model_path}")
        
        return model_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        model = joblib.load(checkpoint_path)
        
        # Load metadata
        metadata_path = checkpoint_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return model, metadata
        
        return model, None
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            "best_model": self.best_model.__class__.__name__ if self.best_model else None,
            "best_params": self.best_params,
            "training_metrics": self.training_metrics,
            "resource_usage": self.resource_usage,
            "training_history": self.training_history
        }