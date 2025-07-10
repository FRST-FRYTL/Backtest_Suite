"""
LSTM Volatility Forecaster for next period ATR prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VolatilityPrediction:
    """Container for volatility prediction results."""
    atr_prediction: float
    confidence_interval: Tuple[float, float]
    attention_weights: Optional[np.ndarray]
    rmse: float

class VolatilityForecaster:
    """
    LSTM-based forecaster for next period ATR (Average True Range).
    
    Features attention mechanism and sequence-based prediction.
    """
    
    def __init__(self,
                 sequence_length: int = 40,
                 lstm_units: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 use_attention: bool = True,
                 random_state: int = 42):
        """
        Initialize VolatilityForecaster.
        
        Args:
            sequence_length: Length of input sequences
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Maximum training epochs
            use_attention: Whether to use attention mechanism
            random_state: Random seed
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_attention = use_attention
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.training_history = None
        self.validation_scores = []
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            data: DataFrame with OHLC data
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for volatility prediction.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=data.index)
        
        # ATR at different periods
        for period in [7, 14, 21, 28]:
            features[f'atr_{period}'] = self.calculate_atr(data, period)
        
        # Historical volatility
        returns = data['close'].pct_change()
        for period in [5, 10, 20, 30]:
            features[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252)
        
        # GARCH-like features
        features['squared_returns'] = returns ** 2
        features['abs_returns'] = abs(returns)
        
        # Volume volatility
        features['volume_volatility'] = data['volume'].pct_change().rolling(20).std()
        
        # Price range features
        features['high_low_range'] = (data['high'] - data['low']) / data['close']
        features['true_range'] = self.calculate_atr(data, 1)  # Single period TR
        
        # Volatility of volatility
        features['vol_of_vol'] = features['volatility_20'].rolling(20).std()
        
        # Market microstructure
        features['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            (np.log(data['high'] / data['low']) ** 2).rolling(20).mean()
        ) * np.sqrt(252)
        
        # Garman-Klass volatility
        features['gk_vol'] = np.sqrt(
            (0.5 * (np.log(data['high'] / data['low']) ** 2) -
             (2 * np.log(2) - 1) * (np.log(data['close'] / data['open']) ** 2)).rolling(20).mean()
        ) * np.sqrt(252)
        
        return features
    
    def create_sequences(self, 
                        features: pd.DataFrame, 
                        target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.
        
        Args:
            features: Feature DataFrame
            target: Target Series (ATR)
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        # Remove NaN values
        valid_mask = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_mask]
        target = target[valid_mask]
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.scaler.fit_transform(target.values.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(features)):
            X.append(features_scaled[i-self.sequence_length:i])
            y.append(target_scaled[i])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build LSTM model with optional attention.
        
        Args:
            input_shape: Shape of input sequences
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            self.lstm_units[0],
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:]):
            return_sequences = (i < len(self.lstm_units) - 2) or self.use_attention
            model.add(LSTM(units, return_sequences=return_sequences))
            model.add(Dropout(self.dropout_rate))
        
        # Attention mechanism
        if self.use_attention:
            # Self-attention layer
            attention_layer = Attention()
            attended_features = attention_layer([
                model.layers[-2].output,
                model.layers[-2].output
            ])
            model.add(Dense(32, activation='relu')(attended_features))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def walk_forward_validation(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              n_splits: int = 5) -> Dict[str, List[float]]:
        """
        Perform walk-forward validation.
        
        Args:
            X: Feature sequences
            y: Target values
            n_splits: Number of splits
            
        Returns:
            Dictionary of validation metrics
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        metrics = {
            'rmse': [],
            'mae': [],
            'mape': []
        }
        
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Build and train model
            model = self.build_model(input_shape=(X.shape[1], X.shape[2]))
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train
            history = model.fit(
                X_train, y_train,
                validation_split=0.1,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Predict
            y_pred = model.predict(X_test).flatten()
            
            # Calculate metrics (inverse transform to get actual values)
            y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_actual = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
            rmse = np.sqrt(np.mean((y_test_actual - y_pred_actual) ** 2))
            mae = np.mean(np.abs(y_test_actual - y_pred_actual))
            mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
            
            metrics['rmse'].append(rmse)
            metrics['mae'].append(mae)
            metrics['mape'].append(mape)
            
            logger.info(f"Fold {i+1}: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")
        
        self.validation_scores = metrics
        return metrics
    
    def fit(self,
            data: pd.DataFrame,
            validate: bool = True,
            n_splits: int = 5) -> 'VolatilityForecaster':
        """
        Fit the model on historical data.
        
        Args:
            data: DataFrame with OHLCV data
            validate: Whether to perform walk-forward validation
            n_splits: Number of validation splits
            
        Returns:
            Self for chaining
        """
        # Create features and target
        features = self.create_features(data)
        target = self.calculate_atr(data)
        
        # Create sequences
        X, y = self.create_sequences(features, target)
        
        # Perform validation if requested
        if validate:
            self.walk_forward_validation(X, y, n_splits=n_splits)
        
        # Build final model
        self.model = self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Train final model
        self.training_history = self.model.fit(
            X, y,
            validation_split=0.1,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self
    
    def predict(self, 
                data: pd.DataFrame,
                n_simulations: int = 100) -> VolatilityPrediction:
        """
        Predict next period ATR with confidence intervals.
        
        Args:
            data: DataFrame with recent OHLCV data
            n_simulations: Number of Monte Carlo simulations for confidence intervals
            
        Returns:
            VolatilityPrediction object
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create features and target
        features = self.create_features(data)
        target = self.calculate_atr(data)
        
        # Get last sequence
        valid_mask = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_mask]
        
        if len(features) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} periods of data")
        
        # Scale and create sequence
        features_scaled = self.feature_scaler.transform(features)
        X = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        # Monte Carlo dropout for uncertainty estimation
        predictions = []
        for _ in range(n_simulations):
            pred = self.model.predict(X, verbose=0)
            predictions.append(pred[0, 0])
        
        predictions = np.array(predictions)
        
        # Get prediction and confidence interval
        pred_scaled = np.mean(predictions)
        pred_std = np.std(predictions)
        
        # Inverse transform
        pred_actual = self.scaler.inverse_transform([[pred_scaled]])[0, 0]
        pred_std_actual = pred_std * self.scaler.scale_[0]
        
        # 95% confidence interval
        confidence_interval = (
            pred_actual - 1.96 * pred_std_actual,
            pred_actual + 1.96 * pred_std_actual
        )
        
        # Get attention weights if available
        attention_weights = None
        if self.use_attention:
            # This would require a custom model implementation to extract attention weights
            # For now, we'll leave it as None
            pass
        
        # Calculate RMSE from validation
        rmse = np.mean(self.validation_scores.get('rmse', [0]))
        
        return VolatilityPrediction(
            atr_prediction=float(pred_actual),
            confidence_interval=confidence_interval,
            attention_weights=attention_weights,
            rmse=float(rmse)
        )
    
    def get_validation_summary(self) -> Dict[str, float]:
        """
        Get summary of validation metrics.
        
        Returns:
            Dictionary of metric names and average values
        """
        if not self.validation_scores:
            return {}
        
        summary = {}
        for metric in ['rmse', 'mae', 'mape']:
            if metric in self.validation_scores:
                summary[f'avg_{metric}'] = np.mean(self.validation_scores[metric])
                summary[f'std_{metric}'] = np.std(self.validation_scores[metric])
        
        return summary
    
    def plot_training_history(self) -> None:
        """Plot training history."""
        if self.training_history is None:
            logger.warning("No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(self.training_history.history['loss'], label='Train')
        ax1.plot(self.training_history.history['val_loss'], label='Validation')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # MAE
        ax2.plot(self.training_history.history['mae'], label='Train')
        ax2.plot(self.training_history.history['val_mae'], label='Validation')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()