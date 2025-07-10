"""
Feature engineering module for ML-enhanced backtesting.

Provides comprehensive feature extraction and transformation capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    price_features: bool = True
    volume_features: bool = True
    technical_indicators: bool = True
    statistical_features: bool = True
    market_microstructure: bool = True
    regime_features: bool = True
    lag_features: List[int] = None
    rolling_windows: List[int] = None
    
    def __post_init__(self):
        if self.lag_features is None:
            self.lag_features = [1, 2, 3, 5, 10]
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 20, 50]


class FeatureEngineer:
    """
    Comprehensive feature engineering for ML trading strategies.
    """
    
    def __init__(self):
        self.feature_names = []
        self.feature_importance = {}
    
    def engineer_features(
        self,
        data: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Engineer comprehensive feature set from OHLCV data.
        
        Args:
            data: OHLCV DataFrame
            config: Feature configuration dict
            
        Returns:
            DataFrame with engineered features
        """
        if config is None:
            config = {}
        
        # Create feature config
        feature_config = FeatureConfig(**config)
        
        # Copy data to avoid modifying original
        features = data.copy()
        
        # Price features
        if feature_config.price_features:
            features = self._add_price_features(features)
        
        # Volume features
        if feature_config.volume_features:
            features = self._add_volume_features(features)
        
        # Technical indicators
        if feature_config.technical_indicators:
            features = self._add_technical_indicators(features)
        
        # Statistical features
        if feature_config.statistical_features:
            features = self._add_statistical_features(features, feature_config.rolling_windows)
        
        # Market microstructure
        if feature_config.market_microstructure:
            features = self._add_microstructure_features(features)
        
        # Regime features
        if feature_config.regime_features:
            features = self._add_regime_features(features)
        
        # Lag features
        if feature_config.lag_features:
            features = self._add_lag_features(features, feature_config.lag_features)
        
        # Store feature names
        self.feature_names = [col for col in features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        return features.dropna()
    
    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Price ratios
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        
        # Price position in range
        data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Gaps
        data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        
        return data
    
    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume moving average
        data['volume_sma_20'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma_20']
        
        # Volume-price correlation
        data['volume_price_corr'] = data['volume'].rolling(20).corr(data['close'])
        
        # On-balance volume
        data['obv'] = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
        data['obv_change'] = data['obv'].pct_change()
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators."""
        # Moving averages
        for period in [10, 20, 50]:
            data[f'sma_{period}'] = data['close'].rolling(period).mean()
            data[f'price_to_sma_{period}'] = data['close'] / data[f'sma_{period}']
        
        # RSI
        data['rsi'] = self._calculate_rsi(data['close'], 14)
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        data['bb_middle'] = data['close'].rolling(bb_period).mean()
        rolling_std = data['close'].rolling(bb_period).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * rolling_std)
        data['bb_lower'] = data['bb_middle'] - (bb_std * rolling_std)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['close']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # ATR
        data['atr'] = self._calculate_atr(data, 14)
        data['atr_percent'] = data['atr'] / data['close']
        
        return data
    
    def _add_statistical_features(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add statistical features."""
        for window in windows:
            # Rolling statistics for returns
            data[f'return_mean_{window}'] = data['returns'].rolling(window).mean()
            data[f'return_std_{window}'] = data['returns'].rolling(window).std()
            data[f'return_skew_{window}'] = data['returns'].rolling(window).skew()
            data[f'return_kurt_{window}'] = data['returns'].rolling(window).kurt()
            
            # Rolling statistics for volume
            data[f'volume_std_{window}'] = data['volume'].rolling(window).std()
            
        return data
    
    def _add_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        # Spread
        data['spread'] = (data['high'] - data['low']) / data['close']
        
        # Amihud illiquidity
        data['illiquidity'] = abs(data['returns']) / (data['volume'] * data['close'])
        
        # Kyle's lambda (simplified)
        data['kyle_lambda'] = abs(data['returns']) / np.sqrt(data['volume'])
        
        return data
    
    def _add_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features."""
        # Volatility regime
        data['volatility_20'] = data['returns'].rolling(20).std()
        data['volatility_60'] = data['returns'].rolling(60).std()
        data['vol_regime'] = data['volatility_20'] / data['volatility_60']
        
        # Trend regime
        data['trend_strength'] = (data['close'] - data['close'].rolling(50).mean()) / data['close'].rolling(50).std()
        
        return data
    
    def _add_lag_features(self, data: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """Add lagged features."""
        for lag in lags:
            data[f'returns_lag_{lag}'] = data['returns'].shift(lag)
            data[f'volume_lag_{lag}'] = data['volume'].shift(lag)
            
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def select_features(
        self,
        data: pd.DataFrame,
        target: str,
        method: str = 'mutual_info',
        threshold: float = 0.01
    ) -> List[str]:
        """
        Select important features based on importance scores.
        
        Args:
            data: DataFrame with features
            target: Target column name
            method: Feature selection method
            threshold: Importance threshold
            
        Returns:
            List of selected feature names
        """
        from sklearn.feature_selection import mutual_info_regression
        
        # Get feature columns (exclude target and OHLCV)
        feature_cols = [col for col in data.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', target]]
        
        # Prepare data
        X = data[feature_cols].fillna(0)
        y = data[target].fillna(0)
        
        # Calculate importance scores
        if method == 'mutual_info':
            scores = mutual_info_regression(X, y)
        else:
            # Default to correlation
            scores = [abs(data[col].corr(data[target])) for col in feature_cols]
        
        # Create importance dict
        importance_dict = dict(zip(feature_cols, scores))
        self.feature_importance = importance_dict
        
        # Select features above threshold
        selected_features = [feat for feat, score in importance_dict.items() 
                           if score > threshold]
        
        return selected_features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance