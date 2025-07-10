"""
Model Architecture Optimization (Loop 2)

Optimizes ML model architecture, hyperparameters, and training configuration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ArchitectureOptimization:
    """
    Optimizes model architecture and hyperparameters for trading strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize architecture optimization
        
        Args:
            config: Architecture optimization configuration
        """
        self.config = config
        self.model_cache = {}
        
    def get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Get hyperparameters from Optuna trial for architecture optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of architecture parameters
        """
        # First, choose model type
        model_type = trial.suggest_categorical(
            'model_type',
            ['random_forest', 'xgboost', 'lightgbm', 'neural_network', 'ensemble']
        )
        
        params = {
            'model_architecture': {
                'type': model_type,
                'training': {
                    'epochs': trial.suggest_int('epochs', 10, 100) if model_type == 'neural_network' else None,
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]) if model_type == 'neural_network' else None,
                    'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),
                    'validation_split': trial.suggest_float('validation_split', 0.1, 0.3),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                }
            }
        }
        
        # Model-specific parameters
        if model_type == 'random_forest':
            params['model_architecture']['random_forest'] = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
                'max_depth': trial.suggest_int('rf_max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('rf_max_features', ['auto', 'sqrt', 'log2']),
                'bootstrap': trial.suggest_categorical('rf_bootstrap', [True, False]),
                'class_weight': trial.suggest_categorical('rf_class_weight', ['balanced', None]),
            }
            
        elif model_type == 'xgboost':
            params['model_architecture']['xgboost'] = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('xgb_gamma', 0.0, 1.0),
                'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 1.0, log=True),
            }
            
        elif model_type == 'lightgbm':
            params['model_architecture']['lightgbm'] = {
                'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 300),
                'num_leaves': trial.suggest_int('lgb_num_leaves', 10, 100),
                'max_depth': trial.suggest_int('lgb_max_depth', 3, 15),
                'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('lgb_bagging_freq', 1, 10),
                'lambda_l1': trial.suggest_float('lgb_lambda_l1', 1e-8, 1.0, log=True),
                'lambda_l2': trial.suggest_float('lgb_lambda_l2', 1e-8, 1.0, log=True),
                'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 30),
            }
            
        elif model_type == 'neural_network':
            n_layers = trial.suggest_int('nn_n_layers', 2, 5)
            layers = []
            
            for i in range(n_layers):
                layers.append({
                    'units': trial.suggest_int(f'nn_layer_{i}_units', 32, 256),
                    'activation': trial.suggest_categorical(f'nn_layer_{i}_activation', ['relu', 'tanh', 'sigmoid']),
                    'dropout': trial.suggest_float(f'nn_layer_{i}_dropout', 0.0, 0.5),
                })
            
            params['model_architecture']['neural_network'] = {
                'layers': layers,
                'optimizer': trial.suggest_categorical('nn_optimizer', ['adam', 'sgd', 'rmsprop']),
                'loss': trial.suggest_categorical('nn_loss', ['binary_crossentropy', 'categorical_crossentropy']),
                'batch_normalization': trial.suggest_categorical('nn_batch_norm', [True, False]),
                'weight_initialization': trial.suggest_categorical('nn_weight_init', ['glorot_uniform', 'he_normal', 'random_normal']),
            }
            
        elif model_type == 'ensemble':
            params['model_architecture']['ensemble'] = {
                'base_models': trial.suggest_categorical(
                    'ensemble_base_models',
                    [
                        ['random_forest', 'xgboost'],
                        ['random_forest', 'lightgbm'],
                        ['xgboost', 'lightgbm'],
                        ['random_forest', 'xgboost', 'lightgbm'],
                    ]
                ),
                'voting': trial.suggest_categorical('ensemble_voting', ['soft', 'hard']),
                'weights': trial.suggest_categorical('ensemble_weights', ['uniform', 'optimized']),
            }
        
        return params
    
    def evaluate(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """
        Evaluate model architecture configuration
        
        Args:
            data: Market data with features
            params: Complete parameter set including architecture params
            
        Returns:
            Performance metric (higher is better)
        """
        try:
            # Get features from previous optimization or use default
            features = self._prepare_features(data, params)
            
            if features.empty or len(features) < 100:
                return -np.inf
            
            # Create target variable
            target = self._create_target(data, features.index)
            
            # Build and evaluate model
            model = self._build_model(params.get('model_architecture', {}))
            score = self._evaluate_model(model, features, target, params.get('model_architecture', {}))
            
            return score
            
        except Exception as e:
            logger.error(f"Error in architecture evaluation: {str(e)}")
            return -np.inf
    
    def _prepare_features(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for model training"""
        # If features are provided in params, use them
        if 'features' in params:
            return params['features']
        
        # Otherwise, create basic features
        features = pd.DataFrame(index=data.index)
        
        # Basic price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_open_ratio'] = data['close'] / data['open']
        
        # Simple technical indicators
        for period in [5, 10, 20]:
            features[f'sma_{period}'] = data['close'].rolling(period).mean()
            features[f'returns_std_{period}'] = features['returns'].rolling(period).std()
            features[f'returns_mean_{period}'] = features['returns'].rolling(period).mean()
        
        # Volume features if available
        if 'volume' in data.columns:
            features['volume_ratio'] = data['volume'] / data['volume'].rolling(10).mean()
        
        # Lag features
        for lag in range(1, 6):
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        
        return features.dropna()
    
    def _create_target(self, data: pd.DataFrame, index: pd.Index) -> pd.Series:
        """Create target variable for model training"""
        # Binary classification: predict if next return is positive
        future_returns = data['close'].pct_change().shift(-1)
        target = (future_returns > 0).astype(int)
        return target.loc[index].dropna()
    
    def _build_model(self, architecture_params: Dict[str, Any]):
        """Build model based on architecture parameters"""
        model_type = architecture_params.get('type', 'random_forest')
        
        if model_type == 'random_forest':
            rf_params = architecture_params.get('random_forest', {})
            model = RandomForestClassifier(
                n_estimators=rf_params.get('n_estimators', 100),
                max_depth=rf_params.get('max_depth', 10),
                min_samples_split=rf_params.get('min_samples_split', 2),
                min_samples_leaf=rf_params.get('min_samples_leaf', 1),
                max_features=rf_params.get('max_features', 'auto'),
                bootstrap=rf_params.get('bootstrap', True),
                class_weight=rf_params.get('class_weight', None),
                random_state=42,
                n_jobs=-1
            )
            
        elif model_type == 'xgboost':
            xgb_params = architecture_params.get('xgboost', {})
            model = xgb.XGBClassifier(
                n_estimators=xgb_params.get('n_estimators', 100),
                max_depth=xgb_params.get('max_depth', 6),
                learning_rate=xgb_params.get('learning_rate', 0.1),
                subsample=xgb_params.get('subsample', 0.8),
                colsample_bytree=xgb_params.get('colsample_bytree', 0.8),
                gamma=xgb_params.get('gamma', 0),
                min_child_weight=xgb_params.get('min_child_weight', 1),
                reg_alpha=xgb_params.get('reg_alpha', 0),
                reg_lambda=xgb_params.get('reg_lambda', 1),
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
        elif model_type == 'lightgbm':
            lgb_params = architecture_params.get('lightgbm', {})
            model = lgb.LGBMClassifier(
                n_estimators=lgb_params.get('n_estimators', 100),
                num_leaves=lgb_params.get('num_leaves', 31),
                max_depth=lgb_params.get('max_depth', -1),
                learning_rate=lgb_params.get('learning_rate', 0.1),
                feature_fraction=lgb_params.get('feature_fraction', 0.8),
                bagging_fraction=lgb_params.get('bagging_fraction', 0.8),
                bagging_freq=lgb_params.get('bagging_freq', 5),
                lambda_l1=lgb_params.get('lambda_l1', 0),
                lambda_l2=lgb_params.get('lambda_l2', 0),
                min_child_samples=lgb_params.get('min_child_samples', 20),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
        elif model_type == 'neural_network':
            nn_params = architecture_params.get('neural_network', {})
            model = self._build_neural_network(nn_params)
            
        elif model_type == 'ensemble':
            ensemble_params = architecture_params.get('ensemble', {})
            model = self._build_ensemble(ensemble_params, architecture_params)
            
        else:
            # Default to Random Forest
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        return model
    
    def _build_neural_network(self, nn_params: Dict[str, Any]):
        """Build neural network model"""
        layers_config = nn_params.get('layers', [
            {'units': 64, 'activation': 'relu', 'dropout': 0.2},
            {'units': 32, 'activation': 'relu', 'dropout': 0.2}
        ])
        
        # Create a wrapper class for sklearn compatibility
        class KerasClassifier:
            def __init__(self, layers_config, nn_params):
                self.layers_config = layers_config
                self.nn_params = nn_params
                self.model = None
                self.history = None
                
            def fit(self, X, y, **kwargs):
                # Build model
                self.model = keras.Sequential()
                
                # Input layer
                self.model.add(keras.layers.Input(shape=(X.shape[1],)))
                
                # Hidden layers
                for layer_config in self.layers_config:
                    self.model.add(keras.layers.Dense(
                        layer_config['units'],
                        activation=layer_config['activation'],
                        kernel_initializer=self.nn_params.get('weight_initialization', 'glorot_uniform')
                    ))
                    
                    if self.nn_params.get('batch_normalization', False):
                        self.model.add(keras.layers.BatchNormalization())
                    
                    if layer_config.get('dropout', 0) > 0:
                        self.model.add(keras.layers.Dropout(layer_config['dropout']))
                
                # Output layer
                self.model.add(keras.layers.Dense(1, activation='sigmoid'))
                
                # Compile model
                self.model.compile(
                    optimizer=self.nn_params.get('optimizer', 'adam'),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Fit model
                self.history = self.model.fit(
                    X, y,
                    epochs=kwargs.get('epochs', 50),
                    batch_size=kwargs.get('batch_size', 32),
                    validation_split=kwargs.get('validation_split', 0.2),
                    verbose=0
                )
                
                return self
            
            def predict(self, X):
                return (self.model.predict(X) > 0.5).astype(int).flatten()
            
            def predict_proba(self, X):
                proba = self.model.predict(X)
                return np.column_stack([1 - proba, proba])
        
        return KerasClassifier(layers_config, nn_params)
    
    def _build_ensemble(self, ensemble_params: Dict[str, Any], architecture_params: Dict[str, Any]):
        """Build ensemble model"""
        from sklearn.ensemble import VotingClassifier
        
        base_models = ensemble_params.get('base_models', ['random_forest', 'xgboost'])
        estimators = []
        
        for model_name in base_models:
            if model_name == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            elif model_name == 'xgboost':
                model = xgb.XGBClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1,
                    use_label_encoder=False, eval_metric='logloss'
                )
            elif model_name == 'lightgbm':
                model = lgb.LGBMClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1, verbose=-1
                )
            
            estimators.append((model_name, model))
        
        voting = ensemble_params.get('voting', 'soft')
        
        return VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)
    
    def _evaluate_model(self, model, features: pd.DataFrame, target: pd.Series, 
                       architecture_params: Dict[str, Any]) -> float:
        """Evaluate model performance using time series cross-validation"""
        # Align features and target
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]
        
        if len(features) < 100:
            return -np.inf
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, test_idx in tscv.split(features):
            X_train = features.iloc[train_idx]
            X_test = features.iloc[test_idx]
            y_train = target.iloc[train_idx]
            y_test = target.iloc[test_idx]
            
            try:
                # Handle neural network training differently
                if architecture_params.get('type') == 'neural_network':
                    training_params = architecture_params.get('training', {})
                    model.fit(
                        X_train, y_train,
                        epochs=training_params.get('epochs', 50),
                        batch_size=training_params.get('batch_size', 32),
                        validation_split=training_params.get('validation_split', 0.2)
                    )
                else:
                    model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate multiple metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Calculate profit factor (trading-specific metric)
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    # Use probability for position sizing
                    positions = np.where(y_proba > 0.5, y_proba - 0.5, 0)
                else:
                    positions = y_pred
                
                returns = features.iloc[test_idx]['returns'].shift(-1).fillna(0)
                strategy_returns = positions * returns
                
                profit_factor = self._calculate_profit_factor(strategy_returns)
                sharpe_ratio = self._calculate_sharpe_ratio(strategy_returns)
                
                # Combine metrics
                score = (
                    0.2 * accuracy +
                    0.2 * f1 +
                    0.3 * profit_factor +
                    0.3 * sharpe_ratio
                )
                
                scores.append(score)
                
            except Exception as e:
                logger.error(f"Error in model evaluation fold: {str(e)}")
                scores.append(0)
        
        return np.mean(scores) if scores else -np.inf
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor from returns"""
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        
        if negative_returns == 0:
            return 2.0 if positive_returns > 0 else 0
        
        profit_factor = positive_returns / negative_returns
        return min(profit_factor, 2.0)  # Cap at 2.0 for normalization
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio from returns"""
        if len(returns) < 2:
            return 0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0
        
        # Annualize (assuming daily returns)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        
        # Normalize to 0-1 range
        return max(0, min(sharpe / 3, 1))  # Cap at 3 for normalization
    
    def get_optimized_model(self, best_params: Dict[str, Any]):
        """
        Build model using optimized parameters
        
        Args:
            best_params: Optimized parameters
            
        Returns:
            Trained model instance
        """
        architecture_params = best_params.get('model_architecture', {})
        return self._build_model(architecture_params)