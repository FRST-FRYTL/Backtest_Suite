"""
Feature Engineering Optimization (Loop 1)

Optimizes feature selection, engineering, and preprocessing for ML trading models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import optuna
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import talib
import logging

logger = logging.getLogger(__name__)


class FeatureOptimization:
    """
    Optimizes feature engineering pipeline for trading strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature optimization
        
        Args:
            config: Feature optimization configuration
        """
        self.config = config
        self.feature_cache = {}
        
    def get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Get hyperparameters from Optuna trial for feature optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of feature engineering parameters
        """
        params = {
            'feature_engineering': {
                # Technical indicators selection
                'use_sma': trial.suggest_categorical('use_sma', [True, False]),
                'sma_periods': trial.suggest_int('sma_periods', 5, 50) if trial.params.get('use_sma', False) else None,
                
                'use_ema': trial.suggest_categorical('use_ema', [True, False]),
                'ema_periods': trial.suggest_int('ema_periods', 5, 50) if trial.params.get('use_ema', False) else None,
                
                'use_rsi': trial.suggest_categorical('use_rsi', [True, False]),
                'rsi_period': trial.suggest_int('rsi_period', 7, 21) if trial.params.get('use_rsi', False) else None,
                
                'use_macd': trial.suggest_categorical('use_macd', [True, False]),
                'macd_fast': trial.suggest_int('macd_fast', 8, 15) if trial.params.get('use_macd', False) else None,
                'macd_slow': trial.suggest_int('macd_slow', 20, 30) if trial.params.get('use_macd', False) else None,
                'macd_signal': trial.suggest_int('macd_signal', 5, 12) if trial.params.get('use_macd', False) else None,
                
                'use_bollinger': trial.suggest_categorical('use_bollinger', [True, False]),
                'bb_period': trial.suggest_int('bb_period', 15, 25) if trial.params.get('use_bollinger', False) else None,
                'bb_std': trial.suggest_float('bb_std', 1.5, 3.0) if trial.params.get('use_bollinger', False) else None,
                
                # Volume indicators
                'use_volume_features': trial.suggest_categorical('use_volume_features', [True, False]),
                'volume_ma_period': trial.suggest_int('volume_ma_period', 5, 20) if trial.params.get('use_volume_features', False) else None,
                
                # Price action features
                'use_price_patterns': trial.suggest_categorical('use_price_patterns', [True, False]),
                'pattern_lookback': trial.suggest_int('pattern_lookback', 5, 20) if trial.params.get('use_price_patterns', False) else None,
                
                # Lag features
                'n_lags': trial.suggest_int('n_lags', 1, 10),
                'lag_features': trial.suggest_categorical('lag_features', ['returns', 'prices', 'both']),
                
                # Rolling statistics
                'rolling_window': trial.suggest_int('rolling_window', 5, 30),
                'rolling_features': trial.suggest_categorical(
                    'rolling_features', 
                    ['mean', 'std', 'mean_std', 'all']
                ),
            },
            
            'feature_selection': {
                'method': trial.suggest_categorical(
                    'selection_method',
                    ['kbest', 'mutual_info', 'rfe', 'tree_based', 'pca', 'none']
                ),
                'n_features': trial.suggest_int('n_features', 10, 50),
                'selection_threshold': trial.suggest_float('selection_threshold', 0.01, 0.5),
            },
            
            'preprocessing': {
                'scaler': trial.suggest_categorical(
                    'scaler',
                    ['standard', 'robust', 'minmax', 'none']
                ),
                'handle_outliers': trial.suggest_categorical('handle_outliers', [True, False]),
                'outlier_threshold': trial.suggest_float('outlier_threshold', 2.5, 4.0) if trial.params.get('handle_outliers', False) else None,
                'fill_method': trial.suggest_categorical(
                    'fill_method',
                    ['forward', 'backward', 'interpolate', 'mean']
                ),
            }
        }
        
        return params
    
    def evaluate(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """
        Evaluate feature engineering configuration
        
        Args:
            data: Market data
            params: Complete parameter set including feature params
            
        Returns:
            Performance metric (higher is better)
        """
        try:
            # Extract features
            features = self._engineer_features(data, params.get('feature_engineering', {}))
            
            # Preprocess features
            features = self._preprocess_features(features, params.get('preprocessing', {}))
            
            # Select features
            features = self._select_features(features, params.get('feature_selection', {}))
            
            # Evaluate feature quality
            score = self._evaluate_feature_quality(features, data)
            
            return score
            
        except Exception as e:
            logger.error(f"Error in feature evaluation: {str(e)}")
            return -np.inf
    
    def _engineer_features(self, data: pd.DataFrame, feature_params: Dict[str, Any]) -> pd.DataFrame:
        """Engineer features based on parameters"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_open_ratio'] = data['close'] / data['open']
        
        # Technical indicators
        if feature_params.get('use_sma'):
            period = feature_params.get('sma_periods', 20)
            features[f'sma_{period}'] = talib.SMA(data['close'].values, timeperiod=period)
            features[f'sma_ratio_{period}'] = data['close'] / features[f'sma_{period}']
        
        if feature_params.get('use_ema'):
            period = feature_params.get('ema_periods', 20)
            features[f'ema_{period}'] = talib.EMA(data['close'].values, timeperiod=period)
            features[f'ema_ratio_{period}'] = data['close'] / features[f'ema_{period}']
        
        if feature_params.get('use_rsi'):
            period = feature_params.get('rsi_period', 14)
            features[f'rsi_{period}'] = talib.RSI(data['close'].values, timeperiod=period)
        
        if feature_params.get('use_macd'):
            fast = feature_params.get('macd_fast', 12)
            slow = feature_params.get('macd_slow', 26)
            signal = feature_params.get('macd_signal', 9)
            macd, macd_signal, macd_hist = talib.MACD(
                data['close'].values,
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal
            )
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_hist'] = macd_hist
        
        if feature_params.get('use_bollinger'):
            period = feature_params.get('bb_period', 20)
            std_dev = feature_params.get('bb_std', 2.0)
            upper, middle, lower = talib.BBANDS(
                data['close'].values,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev
            )
            features['bb_upper'] = upper
            features['bb_middle'] = middle
            features['bb_lower'] = lower
            features['bb_width'] = upper - lower
            features['bb_position'] = (data['close'] - lower) / (upper - lower)
        
        # Volume features
        if feature_params.get('use_volume_features') and 'volume' in data.columns:
            vol_period = feature_params.get('volume_ma_period', 10)
            features['volume_ratio'] = data['volume'] / talib.SMA(data['volume'].values, timeperiod=vol_period)
            features['volume_roc'] = talib.ROC(data['volume'].values, timeperiod=1)
            features['price_volume'] = data['close'] * data['volume']
        
        # Price patterns
        if feature_params.get('use_price_patterns'):
            lookback = feature_params.get('pattern_lookback', 10)
            # Higher highs and lower lows
            features['higher_highs'] = (
                data['high'] > data['high'].rolling(lookback).max().shift(1)
            ).astype(int)
            features['lower_lows'] = (
                data['low'] < data['low'].rolling(lookback).min().shift(1)
            ).astype(int)
            
            # Support and resistance levels
            features['resistance_distance'] = (
                data['high'].rolling(lookback).max() - data['close']
            ) / data['close']
            features['support_distance'] = (
                data['close'] - data['low'].rolling(lookback).min()
            ) / data['close']
        
        # Lag features
        n_lags = feature_params.get('n_lags', 3)
        lag_type = feature_params.get('lag_features', 'returns')
        
        if lag_type in ['returns', 'both']:
            for i in range(1, n_lags + 1):
                features[f'returns_lag_{i}'] = features['returns'].shift(i)
        
        if lag_type in ['prices', 'both']:
            for i in range(1, n_lags + 1):
                features[f'price_lag_{i}'] = data['close'].shift(i) / data['close']
        
        # Rolling statistics
        window = feature_params.get('rolling_window', 10)
        rolling_type = feature_params.get('rolling_features', 'mean_std')
        
        if rolling_type in ['mean', 'mean_std', 'all']:
            features[f'returns_mean_{window}'] = features['returns'].rolling(window).mean()
            features[f'price_mean_{window}'] = data['close'].rolling(window).mean()
        
        if rolling_type in ['std', 'mean_std', 'all']:
            features[f'returns_std_{window}'] = features['returns'].rolling(window).std()
            features[f'price_std_{window}'] = data['close'].rolling(window).std()
        
        if rolling_type == 'all':
            features[f'returns_skew_{window}'] = features['returns'].rolling(window).skew()
            features[f'returns_kurt_{window}'] = features['returns'].rolling(window).kurt()
            features[f'returns_min_{window}'] = features['returns'].rolling(window).min()
            features[f'returns_max_{window}'] = features['returns'].rolling(window).max()
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def _preprocess_features(self, features: pd.DataFrame, preprocess_params: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess features"""
        # Handle missing values
        fill_method = preprocess_params.get('fill_method', 'forward')
        if fill_method == 'forward':
            features = features.fillna(method='ffill')
        elif fill_method == 'backward':
            features = features.fillna(method='bfill')
        elif fill_method == 'interpolate':
            features = features.interpolate()
        elif fill_method == 'mean':
            features = features.fillna(features.mean())
        
        # Handle outliers
        if preprocess_params.get('handle_outliers', False):
            threshold = preprocess_params.get('outlier_threshold', 3.0)
            for col in features.columns:
                mean = features[col].mean()
                std = features[col].std()
                features[col] = features[col].clip(
                    lower=mean - threshold * std,
                    upper=mean + threshold * std
                )
        
        # Scale features
        scaler_type = preprocess_params.get('scaler', 'standard')
        if scaler_type != 'none':
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            
            scaled_data = scaler.fit_transform(features)
            features = pd.DataFrame(
                scaled_data,
                index=features.index,
                columns=features.columns
            )
        
        return features
    
    def _select_features(self, features: pd.DataFrame, selection_params: Dict[str, Any]) -> pd.DataFrame:
        """Select best features"""
        method = selection_params.get('method', 'none')
        
        if method == 'none' or len(features.columns) <= selection_params.get('n_features', 20):
            return features
        
        # Create target variable (future returns for prediction)
        target = (features['returns'].shift(-1) > 0).astype(int).dropna()
        features_aligned = features.iloc[:-1]  # Align with target
        
        n_features = min(selection_params.get('n_features', 20), len(features.columns))
        
        if method == 'kbest':
            selector = SelectKBest(f_classif, k=n_features)
            selected = selector.fit_transform(features_aligned, target)
            selected_features = features.columns[selector.get_support()].tolist()
            
        elif method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=n_features)
            selected = selector.fit_transform(features_aligned, target)
            selected_features = features.columns[selector.get_support()].tolist()
            
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=n_features)
            selector.fit(features_aligned, target)
            selected_features = features.columns[selector.support_].tolist()
            
        elif method == 'tree_based':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            estimator.fit(features_aligned, target)
            selector = SelectFromModel(
                estimator,
                threshold=selection_params.get('selection_threshold', 'median'),
                max_features=n_features
            )
            selector.fit(features_aligned, target)
            selected_features = features.columns[selector.get_support()].tolist()
            
        elif method == 'pca':
            pca = PCA(n_components=n_features)
            pca_features = pca.fit_transform(features)
            # Create new feature names
            selected_features = [f'pca_{i}' for i in range(n_features)]
            features = pd.DataFrame(
                pca_features,
                index=features.index,
                columns=selected_features
            )
            return features
        
        return features[selected_features]
    
    def _evaluate_feature_quality(self, features: pd.DataFrame, original_data: pd.DataFrame) -> float:
        """
        Evaluate quality of engineered features
        
        Returns a score based on:
        - Feature importance
        - Predictive power
        - Feature stability
        - Information content
        """
        if features.empty or len(features) < 100:
            return -np.inf
        
        scores = []
        
        # 1. Information content (entropy)
        for col in features.columns:
            if features[col].std() > 0:
                # Normalize and compute entropy
                normalized = (features[col] - features[col].min()) / (features[col].max() - features[col].min() + 1e-8)
                hist, _ = np.histogram(normalized.dropna(), bins=20)
                hist = hist / hist.sum()
                entropy = -np.sum(hist * np.log(hist + 1e-8))
                scores.append(entropy)
        
        information_score = np.mean(scores) if scores else 0
        
        # 2. Predictive power (using simple model)
        try:
            # Create simple target
            target = (original_data['close'].pct_change().shift(-1) > 0).astype(int)
            target = target.loc[features.index].dropna()
            features_aligned = features.loc[target.index]
            
            # Train simple model
            from sklearn.model_selection import cross_val_score
            from sklearn.ensemble import RandomForestClassifier
            
            clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            cv_scores = cross_val_score(clf, features_aligned, target, cv=3, scoring='roc_auc')
            predictive_score = np.mean(cv_scores)
            
        except Exception:
            predictive_score = 0.5
        
        # 3. Feature stability (low correlation between features)
        try:
            corr_matrix = features.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr = (upper_tri > 0.95).sum().sum()
            stability_score = 1.0 - (high_corr / (len(features.columns) * (len(features.columns) - 1) / 2))
        except Exception:
            stability_score = 0.5
        
        # 4. Feature coverage (non-NaN ratio)
        coverage_score = 1.0 - (features.isna().sum().sum() / features.size)
        
        # Combine scores
        total_score = (
            0.3 * information_score +
            0.4 * predictive_score +
            0.2 * stability_score +
            0.1 * coverage_score
        )
        
        return total_score
    
    def get_optimized_features(self, data: pd.DataFrame, best_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate features using optimized parameters
        
        Args:
            data: Market data
            best_params: Optimized parameters
            
        Returns:
            DataFrame of optimized features
        """
        # Extract features
        features = self._engineer_features(data, best_params.get('feature_engineering', {}))
        
        # Preprocess
        features = self._preprocess_features(features, best_params.get('preprocessing', {}))
        
        # Select features
        features = self._select_features(features, best_params.get('feature_selection', {}))
        
        return features