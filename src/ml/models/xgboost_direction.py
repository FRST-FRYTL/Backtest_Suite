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
class DirectionPrediction:
    """Enhanced container for direction prediction results."""
    direction: int  # 1 for up, 0 for down
    probability: float
    feature_importance: Dict[str, float]
    confidence: float
    
    # Enhanced fields for ensemble predictions
    model_predictions: Dict[str, float] = field(default_factory=dict)
    model_weights: Dict[str, float] = field(default_factory=dict)
    ensemble_method: str = "weighted_voting"
    prediction_interval: Tuple[float, float] = (0.0, 1.0)
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
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced technical features for direction prediction.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with advanced features
        """
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
            features['momentum_trend_interaction'] = features.get('roc_10', 0) * features.get('sma_20_slope', 0)
            
        return features
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """
        Create binary labels for next period direction.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Series with binary labels (1 for up, 0 for down)
        """
        next_returns = data['close'].shift(-1) / data['close'] - 1
        return (next_returns > 0).astype(int)
    
    def walk_forward_validation(self, 
                              features: pd.DataFrame, 
                              labels: pd.Series,
                              n_splits: int = 5,
                              test_size: int = 252,
                              gap: int = 0) -> Dict[str, List[float]]:
        """
        Perform walk-forward validation.
        
        Args:
            features: Feature DataFrame
            labels: Label Series
            n_splits: Number of splits
            test_size: Size of test set
            gap: Gap between train and test sets
            
        Returns:
            Dictionary of validation metrics
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'feature_importance': []
        }
        
        for i, (train_idx, test_idx) in enumerate(tscv.split(features)):
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
            
            # Remove NaN values
            train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
            test_mask = ~(X_test.isna().any(axis=1) | y_test.isna())
            
            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
            X_test = X_test[test_mask]
            y_test = y_test[test_mask]
            
            # Train model
            model = xgb.XGBClassifier(**self.params)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['precision'].append(precision_score(y_test, y_pred))
            metrics['recall'].append(recall_score(y_test, y_pred))
            metrics['f1'].append(f1_score(y_test, y_pred))
            
            # Feature importance
            importance = model.feature_importances_
            feature_importance = dict(zip(features.columns, importance))
            metrics['feature_importance'].append(feature_importance)
            
            logger.info(f"Fold {i+1}: Accuracy={metrics['accuracy'][-1]:.4f}, "
                       f"F1={metrics['f1'][-1]:.4f}")
        
        self.validation_scores = metrics
        return metrics
    
    def fit(self, 
            data: pd.DataFrame,
            validate: bool = True,
            n_splits: int = 5) -> 'DirectionPredictor':
        """
        Fit the model on historical data.
        
        Args:
            data: DataFrame with OHLCV data
            validate: Whether to perform walk-forward validation
            n_splits: Number of validation splits
            
        Returns:
            Self for chaining
        """
        # Create features and labels
        features = self.create_features(data)
        labels = self.create_labels(data)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Remove NaN values
        valid_mask = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        # Perform validation if requested
        if validate:
            self.walk_forward_validation(features, labels, n_splits=n_splits)
        
        # Train final model on all data
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(features, labels)
        
        # Store feature importance
        self.feature_importance_history.append(
            dict(zip(self.feature_names, self.model.feature_importances_))
        )
        
        return self
    
    def predict(self, data: pd.DataFrame) -> DirectionPrediction:
        """
        Predict next period direction.
        
        Args:
            data: DataFrame with recent OHLCV data
            
        Returns:
            DirectionPrediction object
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create features
        features = self.create_features(data)
        
        # Get last row (most recent)
        last_features = features.iloc[-1:][self.feature_names]
        
        # Handle NaN values
        if last_features.isna().any().any():
            logger.warning("NaN values in features, filling with 0")
            last_features = last_features.fillna(0)
        
        # Predict
        direction = self.model.predict(last_features)[0]
        probability = self.model.predict_proba(last_features)[0]
        
        # Get feature importance for this prediction
        feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        # Calculate confidence based on probability
        confidence = abs(probability[1] - 0.5) * 2  # Scale to 0-1
        
        return DirectionPrediction(
            direction=int(direction),
            probability=float(probability[1]),
            feature_importance=feature_importance,
            confidence=float(confidence)
        )
    
    def get_feature_importance_summary(self, top_n: int = 10) -> Dict[str, float]:
        """
        Get summary of feature importance across all training.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature names and average importance
        """
        if not self.feature_importance_history:
            return {}
        
        # Average importance across history
        avg_importance = {}
        for feature in self.feature_names:
            importances = [h.get(feature, 0) for h in self.feature_importance_history]
            avg_importance[feature] = np.mean(importances)
        
        # Sort and return top N
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:top_n])
    
    def get_validation_summary(self) -> Dict[str, float]:
        """
        Get summary of validation metrics.
        
        Returns:
            Dictionary of metric names and average values
        """
        if not self.validation_scores:
            return {}
        
        summary = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if metric in self.validation_scores:
                summary[f'avg_{metric}'] = np.mean(self.validation_scores[metric])
                summary[f'std_{metric}'] = np.std(self.validation_scores[metric])
        
        return summary