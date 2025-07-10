"""
Model Architecture Agent for ML Pipeline

Designs and recommends optimal model architectures based on data characteristics
and problem requirements.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import json

from .base_agent import BaseAgent


class ModelArchitectureAgent(BaseAgent):
    """
    Agent responsible for model architecture design including:
    - Model selection and recommendation
    - Hyperparameter configuration
    - Architecture complexity analysis
    - Ensemble design
    - Model comparison and benchmarking
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ModelArchitectureAgent", config)
        self.candidate_models = {}
        self.model_scores = {}
        self.recommended_architecture = None
        self.ensemble_config = None
        self.complexity_analysis = {}
        
    def initialize(self) -> bool:
        """Initialize model architecture resources."""
        try:
            self.logger.info("Initializing Model Architecture Agent")
            
            # Validate required configuration
            required_keys = ["task_type", "evaluation_metric", "model_types"]
            if not self.validate_config(required_keys):
                return False
            
            # Initialize task configuration
            self.task_type = self.config.get("task_type", "regression")
            self.evaluation_metric = self.config.get("evaluation_metric", "mse")
            self.model_types = self.config.get("model_types", ["all"])
            
            # Initialize constraints
            self.constraints = self.config.get("constraints", {
                "max_training_time": 300,  # seconds
                "max_model_size": 100,  # MB
                "interpretability_required": False
            })
            
            # Initialize candidate models
            self._initialize_candidate_models()
            
            self.logger.info("Model Architecture Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False
    
    def execute(self, X_train: pd.DataFrame, y_train: pd.Series,
                X_val: Optional[pd.DataFrame] = None, 
                y_val: Optional[pd.Series] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute model architecture selection and design.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Dict containing recommended architecture and analysis
        """
        try:
            # Analyze data characteristics
            data_profile = self._analyze_data_characteristics(X_train, y_train)
            
            # Filter candidate models based on constraints
            filtered_models = self._filter_models_by_constraints(data_profile)
            
            # Quick evaluation of candidate models
            model_scores = self._evaluate_candidate_models(
                X_train, y_train, filtered_models
            )
            
            # Design optimal architecture
            architecture = self._design_optimal_architecture(
                model_scores, data_profile
            )
            
            # Design ensemble if beneficial
            ensemble = self._design_ensemble_architecture(
                model_scores, data_profile
            )
            
            # Analyze model complexity
            complexity = self._analyze_model_complexity(
                architecture, data_profile
            )
            
            # Generate hyperparameter recommendations
            hyperparams = self._recommend_hyperparameters(
                architecture, data_profile
            )
            
            # Create model comparison report
            comparison = self._create_model_comparison(model_scores)
            
            self.recommended_architecture = architecture
            self.ensemble_config = ensemble
            
            return {
                "recommended_architecture": architecture,
                "ensemble_configuration": ensemble,
                "model_scores": model_scores,
                "hyperparameter_recommendations": hyperparams,
                "complexity_analysis": complexity,
                "model_comparison": comparison,
                "data_profile": data_profile
            }
            
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            raise
    
    def _initialize_candidate_models(self):
        """Initialize candidate model configurations."""
        self.logger.info("Initializing candidate models")
        
        if self.task_type == "classification":
            self.candidate_models = {
                "logistic_regression": {
                    "model_class": LogisticRegression,
                    "complexity": "low",
                    "interpretability": "high",
                    "training_speed": "fast",
                    "default_params": {
                        "max_iter": 1000,
                        "random_state": 42
                    }
                },
                "random_forest": {
                    "model_class": RandomForestClassifier,
                    "complexity": "medium",
                    "interpretability": "medium",
                    "training_speed": "medium",
                    "default_params": {
                        "n_estimators": 100,
                        "random_state": 42
                    }
                },
                "gradient_boosting": {
                    "model_class": GradientBoostingClassifier,
                    "complexity": "high",
                    "interpretability": "low",
                    "training_speed": "slow",
                    "default_params": {
                        "n_estimators": 100,
                        "random_state": 42
                    }
                },
                "xgboost": {
                    "model_class": xgb.XGBClassifier,
                    "complexity": "high",
                    "interpretability": "low",
                    "training_speed": "medium",
                    "default_params": {
                        "n_estimators": 100,
                        "random_state": 42,
                        "use_label_encoder": False,
                        "eval_metric": 'logloss'
                    }
                },
                "lightgbm": {
                    "model_class": lgb.LGBMClassifier,
                    "complexity": "high",
                    "interpretability": "low",
                    "training_speed": "fast",
                    "default_params": {
                        "n_estimators": 100,
                        "random_state": 42,
                        "verbose": -1
                    }
                },
                "neural_network": {
                    "model_class": MLPClassifier,
                    "complexity": "very_high",
                    "interpretability": "very_low",
                    "training_speed": "slow",
                    "default_params": {
                        "hidden_layer_sizes": (100, 50),
                        "max_iter": 1000,
                        "random_state": 42
                    }
                }
            }
        else:  # regression
            self.candidate_models = {
                "linear_regression": {
                    "model_class": Ridge,
                    "complexity": "low",
                    "interpretability": "high",
                    "training_speed": "fast",
                    "default_params": {
                        "random_state": 42
                    }
                },
                "lasso": {
                    "model_class": Lasso,
                    "complexity": "low",
                    "interpretability": "high",
                    "training_speed": "fast",
                    "default_params": {
                        "random_state": 42
                    }
                },
                "elastic_net": {
                    "model_class": ElasticNet,
                    "complexity": "low",
                    "interpretability": "high",
                    "training_speed": "fast",
                    "default_params": {
                        "random_state": 42
                    }
                },
                "random_forest": {
                    "model_class": RandomForestRegressor,
                    "complexity": "medium",
                    "interpretability": "medium",
                    "training_speed": "medium",
                    "default_params": {
                        "n_estimators": 100,
                        "random_state": 42
                    }
                },
                "gradient_boosting": {
                    "model_class": GradientBoostingRegressor,
                    "complexity": "high",
                    "interpretability": "low",
                    "training_speed": "slow",
                    "default_params": {
                        "n_estimators": 100,
                        "random_state": 42
                    }
                },
                "xgboost": {
                    "model_class": xgb.XGBRegressor,
                    "complexity": "high",
                    "interpretability": "low",
                    "training_speed": "medium",
                    "default_params": {
                        "n_estimators": 100,
                        "random_state": 42
                    }
                },
                "lightgbm": {
                    "model_class": lgb.LGBMRegressor,
                    "complexity": "high",
                    "interpretability": "low",
                    "training_speed": "fast",
                    "default_params": {
                        "n_estimators": 100,
                        "random_state": 42,
                        "verbose": -1
                    }
                },
                "neural_network": {
                    "model_class": MLPRegressor,
                    "complexity": "very_high",
                    "interpretability": "very_low",
                    "training_speed": "slow",
                    "default_params": {
                        "hidden_layer_sizes": (100, 50),
                        "max_iter": 1000,
                        "random_state": 42
                    }
                }
            }
    
    def _analyze_data_characteristics(self, X: pd.DataFrame, 
                                    y: pd.Series) -> Dict[str, Any]:
        """Analyze data characteristics to inform model selection."""
        self.logger.info("Analyzing data characteristics")
        
        profile = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "feature_types": {
                "numeric": len(X.select_dtypes(include=[np.number]).columns),
                "categorical": len(X.select_dtypes(exclude=[np.number]).columns)
            },
            "target_distribution": {
                "mean": float(y.mean()) if self.task_type == "regression" else None,
                "std": float(y.std()) if self.task_type == "regression" else None,
                "unique_values": int(y.nunique()),
                "class_balance": y.value_counts().to_dict() if self.task_type == "classification" else None
            },
            "data_quality": {
                "missing_values": int(X.isnull().sum().sum()),
                "missing_percentage": float((X.isnull().sum().sum() / (X.shape[0] * X.shape[1])) * 100)
            },
            "complexity_indicators": {
                "sample_to_feature_ratio": X.shape[0] / X.shape[1],
                "high_dimensional": X.shape[1] > 100,
                "small_sample": X.shape[0] < 1000,
                "imbalanced": self._check_imbalance(y) if self.task_type == "classification" else False
            }
        }
        
        # Feature correlation analysis
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = X[numeric_cols].corr()
            profile["feature_correlations"] = {
                "max_correlation": float(corr_matrix.abs().values[~np.eye(len(corr_matrix), dtype=bool)].max()),
                "mean_correlation": float(corr_matrix.abs().values[~np.eye(len(corr_matrix), dtype=bool)].mean())
            }
        
        return profile
    
    def _check_imbalance(self, y: pd.Series) -> bool:
        """Check if target classes are imbalanced."""
        value_counts = y.value_counts()
        ratio = value_counts.min() / value_counts.max()
        return ratio < 0.3
    
    def _filter_models_by_constraints(self, data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Filter models based on constraints and data characteristics."""
        self.logger.info("Filtering models by constraints")
        
        filtered = {}
        
        for model_name, model_config in self.candidate_models.items():
            # Check model types filter
            if "all" not in self.model_types and model_name not in self.model_types:
                continue
            
            # Check interpretability constraint
            if (self.constraints.get("interpretability_required", False) and
                model_config["interpretability"] in ["low", "very_low"]):
                continue
            
            # Check complexity for small samples
            if (data_profile["complexity_indicators"]["small_sample"] and
                model_config["complexity"] in ["high", "very_high"]):
                continue
            
            # Check for high-dimensional data
            if (data_profile["complexity_indicators"]["high_dimensional"] and
                model_config["complexity"] == "low" and 
                model_name not in ["lasso", "elastic_net"]):
                continue
            
            filtered[model_name] = model_config
        
        return filtered
    
    def _evaluate_candidate_models(self, X: pd.DataFrame, y: pd.Series,
                                 candidate_models: Dict[str, Any]) -> Dict[str, Any]:
        """Quickly evaluate candidate models using cross-validation."""
        self.logger.info("Evaluating candidate models")
        
        scores = {}
        
        # Prepare data
        X_scaled = StandardScaler().fit_transform(X.select_dtypes(include=[np.number]))
        
        for model_name, model_config in candidate_models.items():
            try:
                self.logger.info(f"Evaluating {model_name}")
                
                # Create model instance
                model = model_config["model_class"](**model_config["default_params"])
                
                # Perform cross-validation
                cv_scores = cross_val_score(
                    model, X_scaled, y, 
                    cv=min(5, len(X) // 10),  # Adjust CV folds for small datasets
                    scoring=self._get_sklearn_scoring()
                )
                
                scores[model_name] = {
                    "mean_score": float(cv_scores.mean()),
                    "std_score": float(cv_scores.std()),
                    "scores": cv_scores.tolist(),
                    "model_config": model_config
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate {model_name}: {str(e)}")
                scores[model_name] = {
                    "mean_score": -np.inf,
                    "std_score": 0,
                    "error": str(e)
                }
        
        # Sort by performance
        self.model_scores = dict(sorted(
            scores.items(),
            key=lambda x: x[1]["mean_score"],
            reverse=True
        ))
        
        return self.model_scores
    
    def _get_sklearn_scoring(self) -> str:
        """Get sklearn scoring metric based on evaluation metric."""
        metric_mapping = {
            "accuracy": "accuracy",
            "mse": "neg_mean_squared_error",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc"
        }
        return metric_mapping.get(self.evaluation_metric, "neg_mean_squared_error")
    
    def _design_optimal_architecture(self, model_scores: Dict[str, Any],
                                   data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Design optimal model architecture based on evaluation results."""
        self.logger.info("Designing optimal architecture")
        
        # Get best performing model
        best_model = list(model_scores.keys())[0]
        best_config = model_scores[best_model]
        
        # Design architecture
        architecture = {
            "model_type": best_model,
            "base_architecture": best_config["model_config"]["model_class"].__name__,
            "performance_score": best_config["mean_score"],
            "recommended_config": self._generate_model_config(
                best_model, data_profile
            ),
            "preprocessing_pipeline": self._design_preprocessing_pipeline(data_profile),
            "postprocessing": self._design_postprocessing(best_model),
            "training_strategy": self._design_training_strategy(
                best_model, data_profile
            )
        }
        
        return architecture
    
    def _generate_model_config(self, model_name: str, 
                             data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model-specific configuration."""
        base_config = self.candidate_models[model_name]["default_params"].copy()
        
        # Adjust based on data size
        n_samples = data_profile["n_samples"]
        n_features = data_profile["n_features"]
        
        if model_name in ["random_forest", "gradient_boosting", "xgboost", "lightgbm"]:
            # Adjust number of estimators
            if n_samples < 1000:
                base_config["n_estimators"] = 50
            elif n_samples < 10000:
                base_config["n_estimators"] = 100
            else:
                base_config["n_estimators"] = 200
            
            # Adjust tree depth
            if model_name in ["random_forest", "gradient_boosting"]:
                if n_features < 10:
                    base_config["max_depth"] = 5
                elif n_features < 50:
                    base_config["max_depth"] = 10
                else:
                    base_config["max_depth"] = 15
        
        elif model_name == "neural_network":
            # Adjust network architecture
            if n_features < 20:
                base_config["hidden_layer_sizes"] = (50,)
            elif n_features < 100:
                base_config["hidden_layer_sizes"] = (100, 50)
            else:
                base_config["hidden_layer_sizes"] = (200, 100, 50)
        
        return base_config
    
    def _design_preprocessing_pipeline(self, data_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Design preprocessing pipeline based on data characteristics."""
        pipeline = []
        
        # Scaling
        if data_profile["feature_correlations"]["max_correlation"] > 0.8:
            pipeline.append({
                "step": "feature_scaling",
                "method": "StandardScaler",
                "reason": "High feature correlations detected"
            })
        
        # Dimensionality reduction
        if data_profile["complexity_indicators"]["high_dimensional"]:
            pipeline.append({
                "step": "dimensionality_reduction",
                "method": "PCA",
                "params": {"n_components": 0.95},
                "reason": "High dimensional data"
            })
        
        # Handle missing values
        if data_profile["data_quality"]["missing_percentage"] > 0:
            pipeline.append({
                "step": "imputation",
                "method": "SimpleImputer",
                "params": {"strategy": "mean"},
                "reason": f"{data_profile['data_quality']['missing_percentage']:.1f}% missing values"
            })
        
        return pipeline
    
    def _design_postprocessing(self, model_name: str) -> List[Dict[str, Any]]:
        """Design postprocessing steps."""
        postprocessing = []
        
        if self.task_type == "classification":
            postprocessing.append({
                "step": "probability_calibration",
                "method": "CalibratedClassifierCV",
                "reason": "Improve probability estimates"
            })
        
        return postprocessing
    
    def _design_training_strategy(self, model_name: str,
                                data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Design training strategy."""
        strategy = {
            "validation_strategy": "TimeSeriesSplit" if "time" in str(data_profile) else "StratifiedKFold",
            "n_splits": min(5, data_profile["n_samples"] // 100),
            "early_stopping": model_name in ["xgboost", "lightgbm", "neural_network"],
            "class_weight": "balanced" if data_profile["complexity_indicators"].get("imbalanced", False) else None
        }
        
        return strategy
    
    def _design_ensemble_architecture(self, model_scores: Dict[str, Any],
                                    data_profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Design ensemble architecture if beneficial."""
        self.logger.info("Designing ensemble architecture")
        
        # Check if ensemble would be beneficial
        top_models = list(model_scores.keys())[:3]
        top_scores = [model_scores[m]["mean_score"] for m in top_models]
        
        # If top models have similar performance, ensemble might help
        if len(top_scores) >= 3 and np.std(top_scores) < 0.05:
            return {
                "ensemble_type": "voting",
                "base_models": top_models,
                "weights": self._calculate_ensemble_weights(model_scores, top_models),
                "meta_learner": None,
                "expected_improvement": "3-5%"
            }
        
        # Check for stacking potential
        if len(top_models) >= 3:
            return {
                "ensemble_type": "stacking",
                "base_models": top_models,
                "meta_learner": "logistic_regression" if self.task_type == "classification" else "ridge",
                "cv_strategy": "KFold",
                "expected_improvement": "2-4%"
            }
        
        return None
    
    def _calculate_ensemble_weights(self, model_scores: Dict[str, Any],
                                  models: List[str]) -> List[float]:
        """Calculate ensemble weights based on model performance."""
        scores = [model_scores[m]["mean_score"] for m in models]
        
        # Normalize scores to sum to 1
        min_score = min(scores)
        normalized_scores = [s - min_score + 0.001 for s in scores]
        total = sum(normalized_scores)
        
        return [s / total for s in normalized_scores]
    
    def _analyze_model_complexity(self, architecture: Dict[str, Any],
                                data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model complexity and requirements."""
        self.logger.info("Analyzing model complexity")
        
        model_name = architecture["model_type"]
        model_config = self.candidate_models[model_name]
        
        # Estimate training time
        base_time = {
            "fast": 1, "medium": 5, "slow": 20
        }[model_config["training_speed"]]
        
        time_multiplier = (data_profile["n_samples"] / 1000) * (data_profile["n_features"] / 10)
        estimated_time = base_time * time_multiplier
        
        # Estimate model size
        if model_name in ["random_forest", "gradient_boosting"]:
            model_size = 0.1 * architecture["recommended_config"]["n_estimators"]
        elif model_name == "neural_network":
            layers = architecture["recommended_config"]["hidden_layer_sizes"]
            model_size = sum(layers) * 0.01
        else:
            model_size = 0.01
        
        return {
            "interpretability_score": model_config["interpretability"],
            "estimated_training_time": f"{estimated_time:.1f} seconds",
            "estimated_model_size": f"{model_size:.1f} MB",
            "computational_requirements": {
                "cpu_cores": 4 if model_name in ["xgboost", "lightgbm"] else 1,
                "memory_gb": max(1, data_profile["n_samples"] * data_profile["n_features"] / 1e8),
                "gpu_recommended": model_name == "neural_network" and data_profile["n_samples"] > 10000
            },
            "scalability": {
                "handles_large_data": model_config["training_speed"] in ["fast", "medium"],
                "incremental_learning": model_name in ["neural_network", "lightgbm"],
                "parallel_training": model_name in ["random_forest", "xgboost", "lightgbm"]
            }
        }
    
    def _recommend_hyperparameters(self, architecture: Dict[str, Any],
                                 data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend hyperparameter search space."""
        self.logger.info("Recommending hyperparameters")
        
        model_name = architecture["model_type"]
        search_space = {}
        
        if model_name == "random_forest":
            search_space = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        elif model_name in ["xgboost", "lightgbm"]:
            search_space = {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.3],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9, 1.0]
            }
        elif model_name == "neural_network":
            search_space = {
                "hidden_layer_sizes": [(50,), (100,), (100, 50), (200, 100)],
                "learning_rate_init": [0.001, 0.01, 0.1],
                "alpha": [0.0001, 0.001, 0.01]
            }
        
        return {
            "search_space": search_space,
            "search_strategy": "RandomizedSearchCV" if len(search_space) > 3 else "GridSearchCV",
            "n_iter": 20 if data_profile["n_samples"] < 10000 else 50,
            "cv_folds": min(5, data_profile["n_samples"] // 1000)
        }
    
    def _create_model_comparison(self, model_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed model comparison report."""
        comparison = {
            "rankings": [],
            "performance_summary": {},
            "recommendations": []
        }
        
        for rank, (model_name, scores) in enumerate(model_scores.items(), 1):
            if "error" not in scores:
                comparison["rankings"].append({
                    "rank": rank,
                    "model": model_name,
                    "score": scores["mean_score"],
                    "std": scores["std_score"],
                    "complexity": self.candidate_models[model_name]["complexity"],
                    "interpretability": self.candidate_models[model_name]["interpretability"]
                })
        
        # Performance summary
        if comparison["rankings"]:
            best_score = comparison["rankings"][0]["score"]
            comparison["performance_summary"] = {
                "best_model": comparison["rankings"][0]["model"],
                "best_score": best_score,
                "performance_range": f"{comparison['rankings'][-1]['score']:.4f} to {best_score:.4f}",
                "recommendation": self._generate_recommendation(comparison["rankings"])
            }
        
        return comparison
    
    def _generate_recommendation(self, rankings: List[Dict[str, Any]]) -> str:
        """Generate model recommendation based on rankings."""
        if not rankings:
            return "No models evaluated successfully"
        
        best = rankings[0]
        
        if len(rankings) > 1 and abs(rankings[0]["score"] - rankings[1]["score"]) < 0.02:
            return f"Consider ensemble of {rankings[0]['model']} and {rankings[1]['model']} for best performance"
        elif best["interpretability"] == "high":
            return f"{best['model']} offers best balance of performance and interpretability"
        else:
            return f"{best['model']} shows best performance but consider simpler models if interpretability is important"
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get recommended model configuration."""
        if self.recommended_architecture:
            return {
                "architecture": self.recommended_architecture,
                "ensemble": self.ensemble_config,
                "complexity": self.complexity_analysis
            }
        return {}