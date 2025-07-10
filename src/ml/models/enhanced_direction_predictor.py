"""
Enhanced Direction Predictor with XGBoost, CatBoost, and LightGBM Ensemble
Iteration 1: Advanced model architecture with sophisticated feature engineering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Literal
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import optuna
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class EnhancedDirectionPrediction:
    """Enhanced container for direction prediction results."""
    direction: int  # 1 for up, 0 for down
    probability: float
    confidence: float
    
    # Enhanced fields for ensemble predictions
    model_predictions: Dict[str, float] = field(default_factory=dict)
    model_weights: Dict[str, float] = field(default_factory=dict)
    ensemble_method: str = "weighted_voting"
    prediction_interval: Tuple[float, float] = (0.0, 1.0)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_interactions: Dict[str, float] = field(default_factory=dict)
    temporal_importance: Dict[str, float] = field(default_factory=dict)

class EnhancedDirectionPredictor:
    """
    Enhanced Direction Predictor with XGBoost, CatBoost, and LightGBM Ensemble.
    
    Features advanced model architecture, sophisticated feature engineering,
    and ensemble learning with optimized hyperparameters.
    """
    
    def __init__(self, 
                 ensemble_method: Literal['weighted_voting', 'stacking', 'dynamic'] = 'dynamic',
                 use_feature_interactions: bool = True,
                 use_temporal_features: bool = True,
                 use_polynomial_features: bool = True,
                 optimization_trials: int = 100,
                 random_state: int = 42):
        """
        Initialize Enhanced Direction Predictor.
        
        Args:
            ensemble_method: Method for combining model predictions
            use_feature_interactions: Whether to create feature interactions
            use_temporal_features: Whether to add temporal features
            use_polynomial_features: Whether to add polynomial features
            optimization_trials: Number of Optuna optimization trials
            random_state: Random seed
        """
        self.ensemble_method = ensemble_method
        self.use_feature_interactions = use_feature_interactions
        self.use_temporal_features = use_temporal_features
        self.use_polynomial_features = use_polynomial_features
        self.optimization_trials = optimization_trials
        self.random_state = random_state
        
        # Initialize models dictionary
        self.models = {}
        self.model_weights = {}
        self.meta_model = None
        
        # Feature engineering components
        self.feature_names = None
        self.poly_features = None
        self.scaler = StandardScaler()
        
        # Performance tracking
        self.feature_importance_history = []
        self.validation_scores = []
        self.model_performance = {}
        
        # Optimization study
        self.study = None
        
    def _optimize_xgboost(self, X: pd.DataFrame, y: pd.Series, trial: optuna.Trial) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters."""
        params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
            'gamma': trial.suggest_float('xgb_gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0, 10),
            'objective': 'binary:logistic',
            'random_state': self.random_state,
            'verbosity': 0
        }
        return params
    
    def _optimize_lightgbm(self, X: pd.DataFrame, y: pd.Series, trial: optuna.Trial) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters."""
        params = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 1000),
            'num_leaves': trial.suggest_int('lgb_num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('lgb_bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 100),
            'min_child_weight': trial.suggest_float('lgb_min_child_weight', 1e-3, 10.0),
            'reg_alpha': trial.suggest_float('lgb_reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('lgb_reg_lambda', 0, 10),
            'objective': 'binary',
            'metric': 'binary_logloss',
            'random_state': self.random_state,
            'verbosity': -1
        }
        return params
    
    def _optimize_catboost(self, X: pd.DataFrame, y: pd.Series, trial: optuna.Trial) -> Dict[str, Any]:
        """Optimize CatBoost hyperparameters."""
        params = {
            'iterations': trial.suggest_int('cb_iterations', 100, 1000),
            'depth': trial.suggest_int('cb_depth', 3, 10),
            'learning_rate': trial.suggest_float('cb_learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('cb_l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('cb_border_count', 32, 255),
            'random_strength': trial.suggest_float('cb_random_strength', 0.1, 10.0),
            'bagging_temperature': trial.suggest_float('cb_bagging_temperature', 0.1, 1.0),
            'od_type': 'Iter',
            'od_wait': 50,
            'random_state': self.random_state,
            'verbose': False
        }
        return params
    
    def _train_single_model(self, model_type: str, X: pd.DataFrame, y: pd.Series, 
                           params: Dict[str, Any]) -> Any:
        """Train a single model with given parameters."""
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(**params)
        elif model_type == 'lightgbm':
            model = lgb.LGBMClassifier(**params)
        elif model_type == 'catboost':
            model = cb.CatBoostClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X, y)
        return model
    
    def _evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_proba)
        }
    
    def _optimize_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize ensemble hyperparameters using Optuna."""
        def objective(trial):
            try:
                # Get hyperparameters for each model
                xgb_params = self._optimize_xgboost(X, y, trial)
                lgb_params = self._optimize_lightgbm(X, y, trial)
                cb_params = self._optimize_catboost(X, y, trial)
                
                # Train models
                models = {
                    'xgboost': self._train_single_model('xgboost', X, y, xgb_params),
                    'lightgbm': self._train_single_model('lightgbm', X, y, lgb_params),
                    'catboost': self._train_single_model('catboost', X, y, cb_params)
                }
                
                # Evaluate individual models
                model_scores = {}
                for name, model in models.items():
                    scores = self._evaluate_model(model, X, y)
                    model_scores[name] = scores['f1_score']
                
                # Create weighted ensemble
                if self.ensemble_method == 'weighted_voting':
                    # Optimize weights
                    weights = []
                    for name in models.keys():
                        weight = trial.suggest_float(f'weight_{name}', 0.1, 1.0)
                        weights.append(weight)
                    
                    # Normalize weights
                    total_weight = sum(weights)
                    weights = [w / total_weight for w in weights]
                    
                    # Calculate ensemble prediction
                    ensemble_proba = np.zeros(len(y))
                    for i, (name, model) in enumerate(models.items()):
                        ensemble_proba += weights[i] * model.predict_proba(X)[:, 1]
                    
                    ensemble_pred = (ensemble_proba > 0.5).astype(int)
                    return f1_score(y, ensemble_pred, zero_division=0)
                
                elif self.ensemble_method == 'stacking':
                    # Use stacking with meta-learner
                    meta_features = np.column_stack([
                        model.predict_proba(X)[:, 1] for model in models.values()
                    ])
                    
                    # Simple logistic regression as meta-learner
                    meta_model = LogisticRegression(random_state=self.random_state)
                    meta_model.fit(meta_features, y)
                    
                    ensemble_proba = meta_model.predict_proba(meta_features)[:, 1]
                    ensemble_pred = (ensemble_proba > 0.5).astype(int)
                    return f1_score(y, ensemble_pred, zero_division=0)
                
                else:  # dynamic
                    # Use performance-based dynamic weighting
                    total_score = sum(model_scores.values())
                    weights = [score / total_score for score in model_scores.values()]
                    
                    ensemble_proba = np.zeros(len(y))
                    for i, (name, model) in enumerate(models.items()):
                        ensemble_proba += weights[i] * model.predict_proba(X)[:, 1]
                    
                    ensemble_pred = (ensemble_proba > 0.5).astype(int)
                    return f1_score(y, ensemble_pred, zero_division=0)
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        # Run optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=self.optimization_trials)
        
        self.study = study
        return study.best_params
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced technical features for direction prediction."""
        features = pd.DataFrame(index=data.index)
        
        # Enhanced price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['abs_returns'] = np.abs(features['returns'])
        features['squared_returns'] = features['returns'] ** 2
        
        # Multi-timeframe returns
        for period in [1, 2, 3, 5, 10, 15, 20, 30, 60]:
            features[f'return_{period}'] = data['close'].pct_change(period)
            features[f'log_return_{period}'] = np.log(data['close'] / data['close'].shift(period))
            
        # Moving averages with multiple periods
        for period in [5, 10, 15, 20, 30, 50, 100, 200]:
            features[f'sma_{period}'] = data['close'].rolling(period).mean()
            features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            
            # Ratios and slopes
            features[f'price_to_sma_{period}'] = data['close'] / features[f'sma_{period}']
            features[f'price_to_ema_{period}'] = data['close'] / features[f'ema_{period}']
            features[f'sma_{period}_slope'] = features[f'sma_{period}'].pct_change(5)
            features[f'ema_{period}_slope'] = features[f'ema_{period}'].pct_change(5)
            
        # Advanced momentum indicators
        for period in [9, 14, 21, 30]:
            # RSI variations
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # Stochastic oscillator
            low_min = data['low'].rolling(period).min()
            high_max = data['high'].rolling(period).max()
            features[f'stoch_{period}'] = 100 * (data['close'] - low_min) / (high_max - low_min)
            
            # Williams %R
            features[f'williams_r_{period}'] = -100 * (high_max - data['close']) / (high_max - low_min)
            
        # MACD variations
        for fast, slow in [(12, 26), (8, 21), (5, 35)]:
            ema_fast = data['close'].ewm(span=fast).mean()
            ema_slow = data['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=9).mean()
            features[f'macd_{fast}_{slow}'] = macd
            features[f'macd_signal_{fast}_{slow}'] = signal
            features[f'macd_hist_{fast}_{slow}'] = macd - signal
            
        # Bollinger Bands variations
        for period in [10, 20, 50]:
            for std_dev in [1.5, 2.0, 2.5]:
                sma = data['close'].rolling(period).mean()
                std = data['close'].rolling(period).std()
                upper = sma + (std_dev * std)
                lower = sma - (std_dev * std)
                features[f'bb_upper_{period}_{std_dev}'] = upper
                features[f'bb_lower_{period}_{std_dev}'] = lower
                features[f'bb_width_{period}_{std_dev}'] = upper - lower
                features[f'bb_position_{period}_{std_dev}'] = (data['close'] - lower) / (upper - lower)
                features[f'bb_squeeze_{period}_{std_dev}'] = features[f'bb_width_{period}_{std_dev}'] / features[f'bb_width_{period}_{std_dev}'].rolling(50).mean()
                
        # Volume-based features
        for period in [5, 10, 20, 50]:
            features[f'volume_sma_{period}'] = data['volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = data['volume'] / features[f'volume_sma_{period}']
            features[f'volume_trend_{period}'] = features[f'volume_sma_{period}'].pct_change(5)
            
        # VWAP and volume indicators
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        for period in [10, 20, 50]:
            features[f'vwap_{period}'] = (typical_price * data['volume']).rolling(period).sum() / data['volume'].rolling(period).sum()
            features[f'price_to_vwap_{period}'] = data['close'] / features[f'vwap_{period}']
            
        # On Balance Volume
        features['obv'] = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
        features['obv_trend'] = features['obv'].pct_change(10)
        
        # Volatility features
        for period in [5, 10, 20, 30, 60]:
            features[f'volatility_{period}'] = data['close'].pct_change().rolling(period).std()
            features[f'volatility_ratio_{period}'] = features[f'volatility_{period}'] / features[f'volatility_{period}'].rolling(100).mean()
            
        # ATR (Average True Range)
        for period in [14, 21, 30]:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features[f'atr_{period}'] = true_range.rolling(period).mean()
            features[f'atr_ratio_{period}'] = features[f'atr_{period}'] / data['close']
            
        # Market microstructure features
        features['spread'] = data['high'] - data['low']
        features['spread_pct'] = features['spread'] / data['close']
        features['close_location'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_to_high'] = data['close'] / data['high']
        features['close_to_low'] = data['close'] / data['low']
        
        # Price patterns
        features['doji'] = (np.abs(data['close'] - data['open']) / (data['high'] - data['low'])).fillna(0)
        features['hammer'] = ((data['close'] - data['low']) / (data['high'] - data['low'])).fillna(0)
        features['shooting_star'] = ((data['high'] - data['close']) / (data['high'] - data['low'])).fillna(0)
        
        # Gap features
        features['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        features['gap_up'] = (features['gap'] > 0).astype(int)
        features['gap_down'] = (features['gap'] < 0).astype(int)
        
        # Temporal features
        if self.use_temporal_features and isinstance(data.index, pd.DatetimeIndex):
            features['hour'] = data.index.hour
            features['day_of_week'] = data.index.dayofweek
            features['day_of_month'] = data.index.day
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            features['is_month_end'] = data.index.is_month_end.astype(int)
            features['is_quarter_end'] = data.index.is_quarter_end.astype(int)
            
            # Cyclical encoding
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            
        # Lag features
        key_features = ['returns', 'volatility_20', 'rsi_14', 'volume_ratio_20']
        for feature in key_features:
            if feature in features.columns:
                for lag in [1, 2, 3, 5, 10]:
                    features[f'{feature}_lag_{lag}'] = features[feature].shift(lag)
                    
        # Feature interactions
        if self.use_feature_interactions:
            # Key feature interactions
            features['rsi_bb_interaction'] = features.get('rsi_14', 0) * features.get('bb_position_20_2.0', 0)
            features['volume_volatility_interaction'] = features.get('volume_ratio_20', 0) * features.get('volatility_20', 0)
            features['momentum_trend_interaction'] = features.get('return_10', 0) * features.get('sma_20_slope', 0)
            
        return features
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create binary labels for next period direction."""
        next_returns = data['close'].shift(-1) / data['close'] - 1
        return (next_returns > 0).astype(int)
    
    def fit(self, data: pd.DataFrame, validate: bool = True, n_splits: int = 5) -> 'EnhancedDirectionPredictor':
        """Fit the enhanced ensemble model."""
        logger.info("Starting enhanced direction predictor training...")
        
        # Create features and labels
        features = self.create_features(data)
        labels = self.create_labels(data)
        
        # Remove NaN values
        valid_mask = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Add polynomial features if requested
        if self.use_polynomial_features:
            # Select top features for polynomial expansion to avoid explosion
            top_features = features[['returns', 'volatility_20', 'rsi_14', 'volume_ratio_20']].fillna(0)
            self.poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            poly_features = self.poly_features.fit_transform(top_features)
            poly_feature_names = self.poly_features.get_feature_names_out(top_features.columns)
            
            poly_df = pd.DataFrame(poly_features, index=features.index, columns=poly_feature_names)
            features = pd.concat([features, poly_df], axis=1)
            self.feature_names.extend(poly_feature_names)
        
        # Fill remaining NaN values
        features = features.fillna(0)
        
        # Optimize ensemble
        logger.info("Optimizing ensemble hyperparameters...")
        best_params = self._optimize_ensemble(features, labels)
        
        # Train final models with best parameters
        logger.info("Training final ensemble models...")
        self.models = {}
        
        # Extract best parameters for each model
        xgb_params = {k[4:]: v for k, v in best_params.items() if k.startswith('xgb_')}
        xgb_params.update({
            'objective': 'binary:logistic',
            'random_state': self.random_state,
            'verbosity': 0
        })
        
        lgb_params = {k[4:]: v for k, v in best_params.items() if k.startswith('lgb_')}
        lgb_params.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'random_state': self.random_state,
            'verbosity': -1
        })
        
        cb_params = {k[3:]: v for k, v in best_params.items() if k.startswith('cb_')}
        cb_params.update({
            'od_type': 'Iter',
            'od_wait': 50,
            'random_state': self.random_state,
            'verbose': False
        })
        
        # Train models
        self.models['xgboost'] = self._train_single_model('xgboost', features, labels, xgb_params)
        self.models['lightgbm'] = self._train_single_model('lightgbm', features, labels, lgb_params)
        self.models['catboost'] = self._train_single_model('catboost', features, labels, cb_params)
        
        # Calculate model weights
        if self.ensemble_method == 'weighted_voting':
            # Extract weights from optimization
            weights = {}
            for name in self.models.keys():
                weights[name] = best_params.get(f'weight_{name}', 1.0)
            
            # Normalize weights
            total_weight = sum(weights.values())
            self.model_weights = {k: v / total_weight for k, v in weights.items()}
            
        elif self.ensemble_method == 'stacking':
            # Train meta-learner
            meta_features = np.column_stack([
                model.predict_proba(features)[:, 1] for model in self.models.values()
            ])
            self.meta_model = LogisticRegression(random_state=self.random_state)
            self.meta_model.fit(meta_features, labels)
            
        else:  # dynamic
            # Calculate performance-based weights
            model_scores = {}
            for name, model in self.models.items():
                scores = self._evaluate_model(model, features, labels)
                model_scores[name] = scores['f1_score']
            
            total_score = sum(model_scores.values())
            self.model_weights = {k: v / total_score for k, v in model_scores.items()}
        
        # Perform validation if requested
        if validate:
            self._perform_validation(features, labels, n_splits)
        
        logger.info("Enhanced direction predictor training completed!")
        return self
    
    def _perform_validation(self, features: pd.DataFrame, labels: pd.Series, n_splits: int):
        """Perform walk-forward validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(features)):
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
            
            # Train ensemble on fold
            fold_models = {}
            for name, model in self.models.items():
                fold_model = type(model)(**model.get_params())
                fold_model.fit(X_train, y_train)
                fold_models[name] = fold_model
            
            # Get ensemble predictions
            ensemble_pred = self._get_ensemble_prediction(fold_models, X_test)
            
            # Calculate metrics
            metrics['accuracy'].append(accuracy_score(y_test, ensemble_pred))
            metrics['precision'].append(precision_score(y_test, ensemble_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_test, ensemble_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_test, ensemble_pred, zero_division=0))
            
            # For AUC, we need probabilities
            ensemble_proba = self._get_ensemble_probability(fold_models, X_test)
            metrics['auc'].append(roc_auc_score(y_test, ensemble_proba))
            
            logger.info(f"Fold {fold + 1}: Accuracy={metrics['accuracy'][-1]:.4f}, "
                       f"F1={metrics['f1'][-1]:.4f}, AUC={metrics['auc'][-1]:.4f}")
        
        self.validation_scores = metrics
        
        # Log average performance
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        logger.info(f"Average validation metrics: {avg_metrics}")
    
    def _get_ensemble_prediction(self, models: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
        """Get ensemble prediction from models."""
        if self.ensemble_method == 'weighted_voting':
            ensemble_proba = np.zeros(len(X))
            for name, model in models.items():
                weight = self.model_weights.get(name, 1.0 / len(models))
                ensemble_proba += weight * model.predict_proba(X)[:, 1]
            return (ensemble_proba > 0.5).astype(int)
            
        elif self.ensemble_method == 'stacking':
            meta_features = np.column_stack([
                model.predict_proba(X)[:, 1] for model in models.values()
            ])
            return self.meta_model.predict(meta_features)
            
        else:  # dynamic
            ensemble_proba = np.zeros(len(X))
            for name, model in models.items():
                weight = self.model_weights.get(name, 1.0 / len(models))
                ensemble_proba += weight * model.predict_proba(X)[:, 1]
            return (ensemble_proba > 0.5).astype(int)
    
    def _get_ensemble_probability(self, models: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
        """Get ensemble probability from models."""
        if self.ensemble_method == 'weighted_voting':
            ensemble_proba = np.zeros(len(X))
            for name, model in models.items():
                weight = self.model_weights.get(name, 1.0 / len(models))
                ensemble_proba += weight * model.predict_proba(X)[:, 1]
            return ensemble_proba
            
        elif self.ensemble_method == 'stacking':
            meta_features = np.column_stack([
                model.predict_proba(X)[:, 1] for model in models.values()
            ])
            return self.meta_model.predict_proba(meta_features)[:, 1]
            
        else:  # dynamic
            ensemble_proba = np.zeros(len(X))
            for name, model in models.items():
                weight = self.model_weights.get(name, 1.0 / len(models))
                ensemble_proba += weight * model.predict_proba(X)[:, 1]
            return ensemble_proba
    
    def predict(self, data: pd.DataFrame) -> EnhancedDirectionPrediction:
        """Generate enhanced direction prediction."""
        if not self.models:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create features
        features = self.create_features(data)
        
        # Add polynomial features if used during training
        if self.use_polynomial_features and self.poly_features is not None:
            top_features = features[['returns', 'volatility_20', 'rsi_14', 'volume_ratio_20']].fillna(0)
            poly_features = self.poly_features.transform(top_features)
            poly_feature_names = self.poly_features.get_feature_names_out(top_features.columns)
            
            poly_df = pd.DataFrame(poly_features, index=features.index, columns=poly_feature_names)
            features = pd.concat([features, poly_df], axis=1)
        
        # Ensure feature order matches training
        features = features[self.feature_names].fillna(0)
        
        # Get last row (most recent)
        last_features = features.iloc[-1:].fillna(0)
        
        # Get individual model predictions
        model_predictions = {}
        for name, model in self.models.items():
            proba = model.predict_proba(last_features)[0, 1]
            model_predictions[name] = float(proba)
        
        # Get ensemble prediction
        ensemble_proba = self._get_ensemble_probability(self.models, last_features)[0]
        ensemble_pred = int(ensemble_proba > 0.5)
        
        # Calculate confidence
        confidence = abs(ensemble_proba - 0.5) * 2
        
        # Get feature importance
        feature_importance = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                weight = self.model_weights.get(name, 1.0 / len(self.models))
                for i, feat in enumerate(self.feature_names):
                    if feat not in feature_importance:
                        feature_importance[feat] = 0
                    feature_importance[feat] += weight * importance[i]
        
        # Calculate prediction interval (simple approach)
        model_probas = list(model_predictions.values())
        prediction_interval = (
            float(np.percentile(model_probas, 25)),
            float(np.percentile(model_probas, 75))
        )
        
        return EnhancedDirectionPrediction(
            direction=ensemble_pred,
            probability=float(ensemble_proba),
            confidence=float(confidence),
            model_predictions=model_predictions,
            model_weights=self.model_weights,
            ensemble_method=self.ensemble_method,
            prediction_interval=prediction_interval,
            feature_importance=feature_importance,
            feature_interactions={},  # To be implemented
            temporal_importance={}   # To be implemented
        )
    
    def get_validation_summary(self) -> Dict[str, float]:
        """Get summary of validation metrics."""
        if not self.validation_scores:
            return {}
        
        summary = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            if metric in self.validation_scores:
                summary[f'avg_{metric}'] = np.mean(self.validation_scores[metric])
                summary[f'std_{metric}'] = np.std(self.validation_scores[metric])
        
        return summary
    
    def get_feature_importance_summary(self, top_n: int = 10) -> Dict[str, float]:
        """Get summary of feature importance."""
        if not self.models:
            return {}
        
        # Aggregate feature importance across models
        feature_importance = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                weight = self.model_weights.get(name, 1.0 / len(self.models))
                for i, feat in enumerate(self.feature_names):
                    if feat not in feature_importance:
                        feature_importance[feat] = 0
                    feature_importance[feat] += weight * importance[i]
        
        # Sort and return top N
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:top_n])

# Maintain backward compatibility
DirectionPredictor = EnhancedDirectionPredictor
DirectionPrediction = EnhancedDirectionPrediction