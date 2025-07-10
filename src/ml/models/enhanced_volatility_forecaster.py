"""
Enhanced LSTM Volatility Forecaster with Bidirectional LSTM and Attention
Iteration 1: Advanced volatility prediction with multi-scale analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, Attention, 
    MultiHeadAttention, LayerNormalization, Concatenate, 
    Conv1D, MaxPooling1D, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class EnhancedVolatilityPrediction:
    """Enhanced container for volatility prediction results."""
    atr_prediction: float
    confidence_interval: Tuple[float, float]
    prediction_intervals: Dict[str, Tuple[float, float]]  # Multiple confidence levels
    attention_weights: Optional[np.ndarray]
    model_uncertainty: float
    volatility_regime: str  # low, medium, high
    garch_components: Dict[str, float]
    multi_horizon_predictions: Dict[int, float]  # 1, 5, 10, 20 periods ahead
    feature_contributions: Dict[str, float]
    rmse: float
    mae: float

class EnhancedVolatilityForecaster:
    """
    Enhanced LSTM-based volatility forecaster with bidirectional LSTM and attention.
    
    Features:
    - Bidirectional LSTM with multi-head attention
    - Multi-scale temporal convolutions
    - GARCH-like volatility clustering
    - Multiple horizon predictions
    - Uncertainty quantification
    - Volatility regime detection
    """
    
    def __init__(self,
                 sequence_length: int = 60,
                 lstm_units: List[int] = [128, 64, 32],
                 attention_heads: int = 8,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 use_attention: bool = True,
                 use_cnn_features: bool = True,
                 use_garch_features: bool = True,
                 prediction_horizons: List[int] = [1, 5, 10, 20],
                 random_state: int = 42):
        """
        Initialize Enhanced Volatility Forecaster.
        
        Args:
            sequence_length: Length of input sequences
            lstm_units: List of units for each LSTM layer
            attention_heads: Number of attention heads
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Maximum training epochs
            use_attention: Whether to use attention mechanism
            use_cnn_features: Whether to use CNN feature extraction
            use_garch_features: Whether to include GARCH-like features
            prediction_horizons: List of prediction horizons
            random_state: Random seed
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_attention = use_attention
        self.use_cnn_features = use_cnn_features
        self.use_garch_features = use_garch_features
        self.prediction_horizons = prediction_horizons
        self.random_state = random_state
        
        # Models for different horizons
        self.models = {}
        self.scalers = {}
        self.feature_scalers = {}
        
        # Training history
        self.training_history = {}
        self.validation_scores = {}
        
        # Volatility regime thresholds
        self.volatility_thresholds = {'low': 0.15, 'high': 0.35}
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def calculate_enhanced_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced volatility features."""
        features = pd.DataFrame(index=data.index)
        
        # Basic volatility measures
        returns = data['close'].pct_change()
        
        # Multiple ATR periods
        for period in [7, 14, 21, 30, 60]:
            atr = self._calculate_atr(data, period)
            features[f'atr_{period}'] = atr
            features[f'atr_ratio_{period}'] = atr / data['close']
            
        # Historical volatility at different horizons
        for period in [5, 10, 20, 30, 60]:
            features[f'realized_vol_{period}'] = returns.rolling(period).std() * np.sqrt(252)
            features[f'vol_ratio_{period}'] = features[f'realized_vol_{period}'] / features[f'realized_vol_{period}'].rolling(100).mean()
            
        # GARCH-like features
        if self.use_garch_features:
            # Squared returns (proxy for volatility)
            features['squared_returns'] = returns ** 2
            features['abs_returns'] = np.abs(returns)
            
            # GARCH(1,1) components
            features['garch_vol'] = self._calculate_garch_volatility(returns)
            features['garch_residual'] = features['squared_returns'] - features['garch_vol']
            
            # Volatility clustering
            features['vol_cluster'] = features['squared_returns'].rolling(20).apply(
                lambda x: np.sum(x > x.quantile(0.75)) / len(x)
            )
            
        # Parkinson volatility (using high-low)
        features['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            (np.log(data['high'] / data['low']) ** 2).rolling(20).mean()
        ) * np.sqrt(252)
        
        # Garman-Klass volatility
        features['gk_vol'] = np.sqrt(
            (0.5 * (np.log(data['high'] / data['low']) ** 2) -
             (2 * np.log(2) - 1) * (np.log(data['close'] / data['open']) ** 2)).rolling(20).mean()
        ) * np.sqrt(252)
        
        # Rogers-Satchell volatility
        features['rs_vol'] = np.sqrt(
            (np.log(data['high'] / data['close']) * np.log(data['high'] / data['open']) +
             np.log(data['low'] / data['close']) * np.log(data['low'] / data['open'])).rolling(20).mean()
        ) * np.sqrt(252)
        
        # Volume-based volatility
        features['volume_volatility'] = data['volume'].pct_change().rolling(20).std()
        features['volume_vol_correlation'] = features['realized_vol_20'].rolling(60).corr(features['volume_volatility'])
        
        # Volatility of volatility
        features['vol_of_vol'] = features['realized_vol_20'].rolling(20).std()
        
        # Jump detection
        features['jump_indicator'] = (np.abs(returns) > 3 * features['realized_vol_20'] / np.sqrt(252)).astype(int)
        features['jump_frequency'] = features['jump_indicator'].rolling(20).mean()
        
        # Volatility asymmetry (leverage effect)
        features['asymmetry'] = returns.rolling(20).corr(features['realized_vol_20'])
        
        # Volatility persistence
        features['vol_persistence'] = features['realized_vol_20'].rolling(60).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
        )
        
        # Volatility regime features
        features['vol_regime_low'] = (features['realized_vol_20'] < self.volatility_thresholds['low']).astype(int)
        features['vol_regime_high'] = (features['realized_vol_20'] > self.volatility_thresholds['high']).astype(int)
        
        # Time-based features
        if isinstance(data.index, pd.DatetimeIndex):
            features['hour'] = data.index.hour
            features['day_of_week'] = data.index.dayofweek
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            
            # Volatility seasonality
            features['weekday_effect'] = (data.index.dayofweek < 5).astype(int)
            features['month_end_effect'] = data.index.is_month_end.astype(int)
            features['quarter_end_effect'] = data.index.is_quarter_end.astype(int)
            
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            features[f'vol_lag_{lag}'] = features['realized_vol_20'].shift(lag)
            features[f'atr_lag_{lag}'] = features['atr_14'].shift(lag)
            
        return features
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def _calculate_garch_volatility(self, returns: pd.Series, alpha: float = 0.1, beta: float = 0.8) -> pd.Series:
        """Calculate GARCH(1,1) volatility."""
        # Simple GARCH(1,1) implementation
        vol = pd.Series(index=returns.index, dtype=float)
        vol.iloc[0] = returns.std()
        
        for i in range(1, len(returns)):
            vol.iloc[i] = np.sqrt(
                0.000001 +  # omega (small constant)
                alpha * (returns.iloc[i-1] ** 2) +  # alpha * lagged squared return
                beta * (vol.iloc[i-1] ** 2)  # beta * lagged volatility
            )
        
        return vol
    
    def _build_enhanced_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build enhanced volatility forecasting model."""
        inputs = Input(shape=input_shape)
        
        # CNN feature extraction branch
        if self.use_cnn_features:
            cnn_out = Conv1D(64, 3, activation='relu')(inputs)
            cnn_out = Conv1D(32, 3, activation='relu')(cnn_out)
            cnn_out = MaxPooling1D(2)(cnn_out)
            cnn_out = GlobalAveragePooling1D()(cnn_out)
            cnn_out = Dense(32, activation='relu')(cnn_out)
        
        # Bidirectional LSTM layers
        lstm_out = inputs
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1) or self.use_attention
            
            lstm_out = Bidirectional(
                LSTM(units, return_sequences=return_sequences, dropout=self.dropout_rate),
                name=f'bidirectional_lstm_{i}'
            )(lstm_out)
            
            if return_sequences:
                lstm_out = Dropout(self.dropout_rate)(lstm_out)
        
        # Multi-head attention
        if self.use_attention:
            attention_out = MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=self.lstm_units[-1] * 2,  # *2 for bidirectional
                name='multi_head_attention'
            )(lstm_out, lstm_out)
            
            # Add & Norm
            attention_out = LayerNormalization()(attention_out + lstm_out)
            
            # Global average pooling for attention output
            attention_out = GlobalAveragePooling1D()(attention_out)
            
            # Combine with LSTM output
            if len(self.lstm_units) > 1:
                # If we have LSTM output (when return_sequences=False for last layer)
                combined = Concatenate()([lstm_out, attention_out])
            else:
                combined = attention_out
        else:
            combined = lstm_out
        
        # Combine with CNN features if used
        if self.use_cnn_features:
            combined = Concatenate()([combined, cnn_out])
        
        # Dense layers
        dense_out = Dense(128, activation='relu')(combined)
        dense_out = Dropout(self.dropout_rate)(dense_out)
        dense_out = Dense(64, activation='relu')(dense_out)
        dense_out = Dropout(self.dropout_rate)(dense_out)
        dense_out = Dense(32, activation='relu')(dense_out)
        
        # Output layer
        output = Dense(1, activation='linear', name='volatility_prediction')(dense_out)
        
        model = Model(inputs=inputs, outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_sequences(self, features: pd.DataFrame, target: pd.Series, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input."""
        # Remove NaN values
        valid_mask = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_mask]
        target = target[valid_mask]
        
        # Scale features
        scaler_key = f'features_{horizon}'
        if scaler_key not in self.feature_scalers:
            self.feature_scalers[scaler_key] = StandardScaler()
        
        features_scaled = self.feature_scalers[scaler_key].fit_transform(features)
        
        # Scale target
        if horizon not in self.scalers:
            self.scalers[horizon] = MinMaxScaler()
        
        target_scaled = self.scalers[horizon].fit_transform(target.values.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(features) - horizon + 1):
            X.append(features_scaled[i-self.sequence_length:i])
            y.append(target_scaled[i + horizon - 1])
        
        return np.array(X), np.array(y)
    
    def fit(self, data: pd.DataFrame, validate: bool = True, n_splits: int = 5) -> 'EnhancedVolatilityForecaster':
        """Fit the enhanced volatility forecaster."""
        logger.info("Starting enhanced volatility forecaster training...")
        
        # Create features
        features = self.calculate_enhanced_volatility_features(data)
        
        # Train models for different horizons
        for horizon in self.prediction_horizons:
            logger.info(f"Training model for {horizon}-period horizon...")
            
            # Calculate target (ATR at horizon)
            target = self._calculate_atr(data, 14).shift(-horizon + 1)
            
            # Create sequences
            X, y = self._create_sequences(features, target, horizon)
            
            if len(X) == 0:
                logger.warning(f"No valid sequences for horizon {horizon}")
                continue
            
            # Build model
            model = self._build_enhanced_model(input_shape=(X.shape[1], X.shape[2]))
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=0.00001,
                    verbose=0
                )
            ]
            
            # Train model
            history = model.fit(
                X, y,
                validation_split=0.2,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            self.models[horizon] = model
            self.training_history[horizon] = history.history
            
            # Validation
            if validate:
                val_scores = self._validate_model(X, y, horizon, n_splits)
                self.validation_scores[horizon] = val_scores
                
                logger.info(f"Horizon {horizon} validation - RMSE: {val_scores['rmse']:.4f}, "
                           f"MAE: {val_scores['mae']:.4f}")
        
        logger.info("Enhanced volatility forecaster training completed!")
        return self
    
    def _validate_model(self, X: np.ndarray, y: np.ndarray, horizon: int, n_splits: int) -> Dict[str, float]:
        """Validate model using time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        rmse_scores = []
        mae_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Build and train model
            model = self._build_enhanced_model(input_shape=(X.shape[1], X.shape[2]))
            
            model.fit(
                X_train, y_train,
                validation_split=0.1,
                epochs=50,  # Reduced for validation
                batch_size=self.batch_size,
                verbose=0
            )
            
            # Predict
            y_pred = model.predict(X_test, verbose=0).flatten()
            
            # Inverse transform
            y_test_actual = self.scalers[horizon].inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_actual = self.scalers[horizon].inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
            mae = mean_absolute_error(y_test_actual, y_pred_actual)
            
            rmse_scores.append(rmse)
            mae_scores.append(mae)
        
        return {
            'rmse': np.mean(rmse_scores),
            'mae': np.mean(mae_scores),
            'rmse_std': np.std(rmse_scores),
            'mae_std': np.std(mae_scores)
        }
    
    def predict(self, data: pd.DataFrame, n_simulations: int = 100) -> EnhancedVolatilityPrediction:
        """Generate enhanced volatility prediction."""
        if not self.models:
            raise ValueError("Models not fitted. Call fit() first.")
        
        # Create features
        features = self.calculate_enhanced_volatility_features(data)
        
        # Get predictions for all horizons
        multi_horizon_predictions = {}
        prediction_intervals = {}
        
        for horizon in self.prediction_horizons:
            if horizon not in self.models:
                continue
                
            model = self.models[horizon]
            scaler_key = f'features_{horizon}'
            
            # Prepare features
            valid_mask = ~features.isna().any(axis=1)
            features_clean = features[valid_mask]
            
            if len(features_clean) < self.sequence_length:
                raise ValueError(f"Need at least {self.sequence_length} periods of data")
            
            # Scale features
            features_scaled = self.feature_scalers[scaler_key].transform(features_clean)
            X = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Monte Carlo predictions for uncertainty
            predictions = []
            for _ in range(n_simulations):
                pred = model.predict(X, verbose=0)
                predictions.append(pred[0, 0])
            
            predictions = np.array(predictions)
            
            # Get prediction and confidence intervals
            pred_scaled = np.mean(predictions)
            pred_std = np.std(predictions)
            
            # Inverse transform
            pred_actual = self.scalers[horizon].inverse_transform([[pred_scaled]])[0, 0]
            pred_std_actual = pred_std * self.scalers[horizon].scale_[0]
            
            multi_horizon_predictions[horizon] = float(pred_actual)
            
            # Multiple confidence intervals
            prediction_intervals[f'{horizon}_period'] = {
                '68%': (pred_actual - pred_std_actual, pred_actual + pred_std_actual),
                '95%': (pred_actual - 1.96 * pred_std_actual, pred_actual + 1.96 * pred_std_actual),
                '99%': (pred_actual - 2.58 * pred_std_actual, pred_actual + 2.58 * pred_std_actual)
            }
        
        # Primary prediction (1-period ahead)
        primary_prediction = multi_horizon_predictions.get(1, list(multi_horizon_predictions.values())[0])
        primary_interval = prediction_intervals.get('1_period', {})
        
        # Determine volatility regime
        volatility_regime = self._determine_volatility_regime(primary_prediction)
        
        # Calculate model uncertainty
        model_uncertainty = np.std(list(multi_horizon_predictions.values())) / np.mean(list(multi_horizon_predictions.values()))
        
        # GARCH components (simplified)
        garch_components = self._calculate_garch_components(features)
        
        # Get attention weights (if available)
        attention_weights = self._extract_attention_weights(features)
        
        # Feature contributions (simplified)
        feature_contributions = self._calculate_feature_contributions(features)
        
        # Get validation metrics
        val_scores = self.validation_scores.get(1, {'rmse': 0, 'mae': 0})
        
        return EnhancedVolatilityPrediction(
            atr_prediction=float(primary_prediction),
            confidence_interval=primary_interval.get('95%', (0.0, 0.0)),
            prediction_intervals=prediction_intervals,
            attention_weights=attention_weights,
            model_uncertainty=float(model_uncertainty),
            volatility_regime=volatility_regime,
            garch_components=garch_components,
            multi_horizon_predictions=multi_horizon_predictions,
            feature_contributions=feature_contributions,
            rmse=float(val_scores['rmse']),
            mae=float(val_scores['mae'])
        )
    
    def _determine_volatility_regime(self, prediction: float) -> str:
        """Determine volatility regime based on prediction."""
        if prediction < self.volatility_thresholds['low']:
            return 'low'
        elif prediction > self.volatility_thresholds['high']:
            return 'high'
        else:
            return 'medium'
    
    def _calculate_garch_components(self, features: pd.DataFrame) -> Dict[str, float]:
        """Calculate GARCH components."""
        if 'garch_vol' in features.columns:
            latest_garch = features['garch_vol'].iloc[-1]
            return {
                'garch_volatility': float(latest_garch),
                'volatility_persistence': float(features.get('vol_persistence', pd.Series([0])).iloc[-1]),
                'volatility_clustering': float(features.get('vol_cluster', pd.Series([0])).iloc[-1])
            }
        return {}
    
    def _extract_attention_weights(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract attention weights from model (placeholder)."""
        # This would require custom implementation to extract attention weights
        return None
    
    def _calculate_feature_contributions(self, features: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature contributions (simplified)."""
        # Simplified feature contribution based on recent values
        key_features = ['realized_vol_20', 'atr_14', 'vol_of_vol', 'garch_vol']
        contributions = {}
        
        for feature in key_features:
            if feature in features.columns:
                # Use recent change as proxy for contribution
                recent_change = features[feature].iloc[-1] - features[feature].iloc[-5]
                contributions[feature] = float(recent_change)
        
        return contributions
    
    def get_validation_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of validation metrics for all horizons."""
        return self.validation_scores
    
    def plot_training_history(self, horizon: int = 1):
        """Plot training history for specific horizon."""
        if horizon not in self.training_history:
            logger.warning(f"No training history available for horizon {horizon}")
            return
        
        import matplotlib.pyplot as plt
        
        history = self.training_history[horizon]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(history['loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title(f'Model Loss - {horizon} Period Horizon')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE
        ax2.plot(history['mae'], label='Training MAE')
        ax2.plot(history['val_mae'], label='Validation MAE')
        ax2.set_title(f'Model MAE - {horizon} Period Horizon')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# Maintain backward compatibility
VolatilityForecaster = EnhancedVolatilityForecaster
VolatilityPrediction = EnhancedVolatilityPrediction