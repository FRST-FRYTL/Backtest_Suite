"""
Ensemble Model combining predictions from multiple ML models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Literal
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Import our models
from .enhanced_direction_predictor import EnhancedDirectionPredictor, EnhancedDirectionPrediction
from .enhanced_volatility_forecaster import EnhancedVolatilityForecaster, EnhancedVolatilityPrediction
from .regime_detection import MarketRegimeDetector, RegimeDetection, MarketRegime

# Maintain backward compatibility
DirectionPredictor = EnhancedDirectionPredictor
DirectionPrediction = EnhancedDirectionPrediction
VolatilityForecaster = EnhancedVolatilityForecaster
VolatilityPrediction = EnhancedVolatilityPrediction

logger = logging.getLogger(__name__)

@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results."""
    # Direction prediction
    direction: int  # 1 for up, 0 for down
    direction_probability: float
    direction_confidence: float
    
    # Volatility prediction
    atr_forecast: float
    atr_confidence_interval: Tuple[float, float]
    
    # Regime prediction
    market_regime: MarketRegime
    regime_probabilities: Dict[MarketRegime, float]
    
    # Ensemble metrics
    ensemble_confidence: float
    model_weights: Dict[str, float]
    feature_importance: Dict[str, float]
    risk_score: float  # 0-1, higher means riskier

class EnsembleModel:
    """
    Ensemble model combining XGBoost direction, LSTM volatility, and regime detection.
    
    Supports weighted voting and stacking approaches.
    """
    
    def __init__(self,
                 ensemble_method: Literal['weighted_voting', 'stacking', 'dynamic'] = 'dynamic',
                 direction_weight: float = 0.4,
                 volatility_weight: float = 0.3,
                 regime_weight: float = 0.3,
                 use_meta_learner: bool = True,
                 confidence_threshold: float = 0.6,
                 risk_adjustment: bool = True):
        """
        Initialize EnsembleModel.
        
        Args:
            ensemble_method: Method for combining predictions
            direction_weight: Weight for direction predictor
            volatility_weight: Weight for volatility forecaster
            regime_weight: Weight for regime detector
            use_meta_learner: Whether to use meta-learner for stacking
            confidence_threshold: Minimum confidence for predictions
            risk_adjustment: Whether to adjust predictions based on risk
        """
        self.ensemble_method = ensemble_method
        self.weights = {
            'direction': direction_weight,
            'volatility': volatility_weight,
            'regime': regime_weight
        }
        self.use_meta_learner = use_meta_learner
        self.confidence_threshold = confidence_threshold
        self.risk_adjustment = risk_adjustment
        
        # Initialize component models with enhanced capabilities
        self.direction_model = EnhancedDirectionPredictor(
            ensemble_method='dynamic',
            use_feature_interactions=True,
            use_temporal_features=True,
            use_polynomial_features=True,
            optimization_trials=50  # Reduced for faster training
        )
        self.volatility_model = EnhancedVolatilityForecaster(
            sequence_length=60,
            lstm_units=[128, 64, 32],
            attention_heads=8,
            use_attention=True,
            use_cnn_features=True,
            use_garch_features=True,
            epochs=50  # Reduced for faster training
        )
        self.regime_model = MarketRegimeDetector()
        
        # Meta-learner for stacking
        self.meta_learner = None
        self.meta_scaler = StandardScaler()
        
        # Performance tracking
        self.performance_history = []
        self.dynamic_weights = self.weights.copy()
        
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1."""
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
    
    def _calculate_risk_score(self,
                            volatility: float,
                            regime: MarketRegime,
                            regime_confidence: float) -> float:
        """
        Calculate risk score based on volatility and regime.
        
        Args:
            volatility: ATR prediction
            regime: Current market regime
            regime_confidence: Confidence in regime prediction
            
        Returns:
            Risk score between 0 and 1
        """
        # Base risk from volatility (normalized)
        # Assume typical ATR range is 0.5% to 5%
        volatility_risk = np.clip(volatility / 5.0, 0, 1)
        
        # Regime risk
        regime_risk_map = {
            MarketRegime.STRONG_BULL: 0.3,
            MarketRegime.BULL: 0.4,
            MarketRegime.SIDEWAYS: 0.5,
            MarketRegime.BEAR: 0.7,
            MarketRegime.STRONG_BEAR: 0.9
        }
        regime_risk = regime_risk_map.get(regime, 0.5)
        
        # Adjust for confidence
        confidence_factor = 1 - regime_confidence
        
        # Combine risks
        risk_score = (
            0.5 * volatility_risk +
            0.3 * regime_risk +
            0.2 * confidence_factor
        )
        
        return float(np.clip(risk_score, 0, 1))
    
    def _create_meta_features(self,
                            direction_pred: EnhancedDirectionPrediction,
                            volatility_pred: EnhancedVolatilityPrediction,
                            regime_pred: RegimeDetection) -> np.ndarray:
        """
        Create features for meta-learner.
        
        Args:
            direction_pred: Direction prediction
            volatility_pred: Volatility prediction
            regime_pred: Regime prediction
            
        Returns:
            Feature array for meta-learner
        """
        features = []
        
        # Enhanced direction features
        features.extend([
            direction_pred.probability,
            direction_pred.confidence,
            float(direction_pred.direction)
        ])
        
        # Add model-specific predictions
        for model_name, prediction in direction_pred.model_predictions.items():
            features.append(prediction)
        
        # Add ensemble method indicator
        ensemble_methods = ['weighted_voting', 'stacking', 'dynamic']
        for method in ensemble_methods:
            features.append(1.0 if direction_pred.ensemble_method == method else 0.0)
        
        # Enhanced volatility features
        features.extend([
            volatility_pred.atr_prediction,
            volatility_pred.confidence_interval[1] - volatility_pred.confidence_interval[0],  # Uncertainty
            volatility_pred.rmse,
            volatility_pred.mae,
            volatility_pred.model_uncertainty
        ])
        
        # Volatility regime features
        regime_map = {'low': 0, 'medium': 1, 'high': 2}
        features.append(regime_map.get(volatility_pred.volatility_regime, 1))
        
        # Multi-horizon predictions
        for horizon in [1, 5, 10, 20]:
            horizon_pred = volatility_pred.multi_horizon_predictions.get(horizon, volatility_pred.atr_prediction)
            features.append(horizon_pred)
        
        # Regime features
        features.append(regime_pred.confidence)
        for regime in MarketRegime:
            features.append(regime_pred.regime_probabilities.get(regime, 0))
        
        # Interaction features
        features.append(direction_pred.confidence * regime_pred.confidence)
        features.append(volatility_pred.atr_prediction * direction_pred.probability)
        
        return np.array(features)
    
    def _update_dynamic_weights(self, performance_metrics: Dict[str, float]) -> None:
        """
        Update weights dynamically based on recent performance.
        
        Args:
            performance_metrics: Recent performance of each model
        """
        if len(self.performance_history) < 10:
            return  # Not enough history
        
        # Calculate recent performance for each model
        recent_performance = {
            'direction': np.mean([p['direction_accuracy'] for p in self.performance_history[-10:]]),
            'volatility': 1 - np.mean([p['volatility_mae'] for p in self.performance_history[-10:]]),
            'regime': np.mean([p['regime_accuracy'] for p in self.performance_history[-10:]])
        }
        
        # Update weights based on performance
        for model, perf in recent_performance.items():
            # Increase weight for better performing models
            self.dynamic_weights[model] = self.weights[model] * (0.8 + 0.4 * perf)
        
        # Normalize weights
        self.dynamic_weights = self._normalize_weights(self.dynamic_weights)
        
        logger.info(f"Updated dynamic weights: {self.dynamic_weights}")
    
    def fit(self,
            data: pd.DataFrame,
            validate: bool = True,
            n_splits: int = 5) -> 'EnsembleModel':
        """
        Fit all component models.
        
        Args:
            data: DataFrame with OHLCV data
            validate: Whether to perform validation
            n_splits: Number of validation splits
            
        Returns:
            Self for chaining
        """
        logger.info("Fitting ensemble model components...")
        
        # Fit component models
        logger.info("Fitting direction predictor...")
        self.direction_model.fit(data, validate=validate, n_splits=n_splits)
        
        logger.info("Fitting volatility forecaster...")
        self.volatility_model.fit(data, validate=validate, n_splits=n_splits)
        
        logger.info("Fitting regime detector...")
        self.regime_model.fit(data)
        
        # Train meta-learner if using stacking
        if self.ensemble_method == 'stacking' and self.use_meta_learner:
            logger.info("Training meta-learner...")
            self._train_meta_learner(data)
        
        logger.info("Ensemble model fitting complete")
        return self
    
    def _train_meta_learner(self, data: pd.DataFrame) -> None:
        """
        Train meta-learner for stacking ensemble.
        
        Args:
            data: Training data
        """
        # Generate predictions from base models
        meta_features = []
        meta_labels = []
        
        # Use sliding window for training
        window_size = 252  # 1 year
        step_size = 20  # Refit every 20 days
        
        for i in range(window_size, len(data) - 1, step_size):
            train_data = data.iloc[:i]
            current_data = data.iloc[:i+1]
            
            try:
                # Get predictions
                direction_pred = self.direction_model.predict(current_data)
                volatility_pred = self.volatility_model.predict(current_data)
                regime_pred = self.regime_model.predict(current_data)
                
                # Create meta features
                features = self._create_meta_features(
                    direction_pred, volatility_pred, regime_pred
                )
                meta_features.append(features)
                
                # Get actual label (next day direction)
                next_return = data['close'].iloc[i+1] / data['close'].iloc[i] - 1
                meta_labels.append(1 if next_return > 0 else 0)
                
            except Exception as e:
                logger.warning(f"Error generating meta features: {e}")
                continue
        
        if len(meta_features) > 50:  # Need sufficient samples
            X_meta = np.array(meta_features)
            y_meta = np.array(meta_labels)
            
            # Scale features
            X_meta_scaled = self.meta_scaler.fit_transform(X_meta)
            
            # Train meta-learner (logistic regression for interpretability)
            self.meta_learner = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
            self.meta_learner.fit(X_meta_scaled, y_meta)
            
            logger.info(f"Meta-learner trained with {len(meta_features)} samples")
        else:
            logger.warning("Insufficient data for meta-learner training")
    
    def predict(self, data: pd.DataFrame) -> EnsemblePrediction:
        """
        Generate ensemble prediction.
        
        Args:
            data: DataFrame with recent OHLCV data
            
        Returns:
            EnsemblePrediction object
        """
        # Get predictions from all models
        direction_pred = self.direction_model.predict(data)
        volatility_pred = self.volatility_model.predict(data)
        regime_pred = self.regime_model.predict(data)
        
        # Determine ensemble method
        if self.ensemble_method == 'weighted_voting':
            return self._weighted_voting_predict(
                direction_pred, volatility_pred, regime_pred
            )
        elif self.ensemble_method == 'stacking':
            return self._stacking_predict(
                direction_pred, volatility_pred, regime_pred
            )
        elif self.ensemble_method == 'dynamic':
            return self._dynamic_predict(
                direction_pred, volatility_pred, regime_pred
            )
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _weighted_voting_predict(self,
                               direction_pred: EnhancedDirectionPrediction,
                               volatility_pred: EnhancedVolatilityPrediction,
                               regime_pred: RegimeDetection) -> EnsemblePrediction:
        """
        Weighted voting ensemble prediction.
        """
        # Adjust direction probability based on regime
        regime_adjustment = {
            MarketRegime.STRONG_BULL: 0.1,
            MarketRegime.BULL: 0.05,
            MarketRegime.SIDEWAYS: 0,
            MarketRegime.BEAR: -0.05,
            MarketRegime.STRONG_BEAR: -0.1
        }
        
        adjusted_direction_prob = direction_pred.probability + regime_adjustment.get(
            regime_pred.current_regime, 0
        )
        adjusted_direction_prob = np.clip(adjusted_direction_prob, 0, 1)
        
        # Calculate ensemble confidence with enhanced volatility features
        vol_confidence = 1 - min(volatility_pred.model_uncertainty, 1.0)  # Convert uncertainty to confidence
        ensemble_confidence = (
            self.weights['direction'] * direction_pred.confidence +
            self.weights['volatility'] * vol_confidence +
            self.weights['regime'] * regime_pred.confidence
        )
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(
            volatility_pred.atr_prediction,
            regime_pred.current_regime,
            regime_pred.confidence
        )
        
        # Adjust for risk if enabled
        if self.risk_adjustment and risk_score > 0.7:
            ensemble_confidence *= (1 - risk_score * 0.3)
        
        # Determine final direction
        final_direction = 1 if adjusted_direction_prob > 0.5 else 0
        
        # Combine feature importance
        combined_importance = {}
        for feature, importance in direction_pred.feature_importance.items():
            combined_importance[f"direction_{feature}"] = importance * self.weights['direction']
        
        return EnsemblePrediction(
            direction=final_direction,
            direction_probability=float(adjusted_direction_prob),
            direction_confidence=float(direction_pred.confidence),
            atr_forecast=float(volatility_pred.atr_prediction),
            atr_confidence_interval=volatility_pred.confidence_interval,
            market_regime=regime_pred.current_regime,
            regime_probabilities=regime_pred.regime_probabilities,
            ensemble_confidence=float(ensemble_confidence),
            model_weights=self.weights,
            feature_importance=combined_importance,
            risk_score=float(risk_score)
        )
    
    def _stacking_predict(self,
                         direction_pred: EnhancedDirectionPrediction,
                         volatility_pred: EnhancedVolatilityPrediction,
                         regime_pred: RegimeDetection) -> EnsemblePrediction:
        """
        Stacking ensemble prediction using meta-learner.
        """
        if self.meta_learner is None:
            # Fall back to weighted voting
            logger.warning("Meta-learner not trained, using weighted voting")
            return self._weighted_voting_predict(
                direction_pred, volatility_pred, regime_pred
            )
        
        # Create meta features
        meta_features = self._create_meta_features(
            direction_pred, volatility_pred, regime_pred
        )
        
        # Scale features
        meta_features_scaled = self.meta_scaler.transform(meta_features.reshape(1, -1))
        
        # Get meta prediction
        meta_direction = self.meta_learner.predict(meta_features_scaled)[0]
        meta_probability = self.meta_learner.predict_proba(meta_features_scaled)[0, 1]
        
        # Calculate ensemble confidence
        base_confidences = [
            direction_pred.confidence,
            1 - volatility_pred.rmse / 100,
            regime_pred.confidence
        ]
        ensemble_confidence = np.mean(base_confidences) * abs(meta_probability - 0.5) * 2
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(
            volatility_pred.atr_prediction,
            regime_pred.current_regime,
            regime_pred.confidence
        )
        
        # Get feature importance from meta-learner
        if hasattr(self.meta_learner, 'coef_'):
            feature_names = [
                'dir_prob', 'dir_conf', 'dir_pred',
                'vol_atr', 'vol_uncertainty', 'vol_rmse',
                'regime_conf'
            ] + [f'regime_prob_{r.value}' for r in MarketRegime] + [
                'interaction_conf', 'interaction_vol_dir'
            ]
            
            feature_importance = dict(zip(
                feature_names,
                abs(self.meta_learner.coef_[0])
            ))
        else:
            feature_importance = direction_pred.feature_importance
        
        return EnsemblePrediction(
            direction=int(meta_direction),
            direction_probability=float(meta_probability),
            direction_confidence=float(direction_pred.confidence),
            atr_forecast=float(volatility_pred.atr_prediction),
            atr_confidence_interval=volatility_pred.confidence_interval,
            market_regime=regime_pred.current_regime,
            regime_probabilities=regime_pred.regime_probabilities,
            ensemble_confidence=float(ensemble_confidence),
            model_weights=self.weights,
            feature_importance=feature_importance,
            risk_score=float(risk_score)
        )
    
    def _dynamic_predict(self,
                        direction_pred: EnhancedDirectionPrediction,
                        volatility_pred: EnhancedVolatilityPrediction,
                        regime_pred: RegimeDetection) -> EnsemblePrediction:
        """
        Dynamic ensemble prediction with adaptive weights.
        """
        # Update weights based on recent performance
        if len(self.performance_history) > 0:
            self._update_dynamic_weights(self.performance_history[-1])
        
        # Use dynamic weights for weighted voting
        original_weights = self.weights.copy()
        self.weights = self.dynamic_weights
        
        prediction = self._weighted_voting_predict(
            direction_pred, volatility_pred, regime_pred
        )
        
        # Restore original weights
        self.weights = original_weights
        
        # Update model weights in prediction
        prediction.model_weights = self.dynamic_weights
        
        return prediction
    
    def update_performance(self,
                          prediction: EnsemblePrediction,
                          actual_direction: int,
                          actual_atr: float) -> None:
        """
        Update performance history for dynamic weighting.
        
        Args:
            prediction: The ensemble prediction made
            actual_direction: Actual direction (1 or 0)
            actual_atr: Actual ATR value
        """
        performance = {
            'direction_accuracy': float(prediction.direction == actual_direction),
            'volatility_mae': abs(prediction.atr_forecast - actual_atr),
            'regime_accuracy': float(prediction.ensemble_confidence > 0.5),
            'timestamp': pd.Timestamp.now()
        }
        
        self.performance_history.append(performance)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of ensemble model performance.
        
        Returns:
            Dictionary with model summaries
        """
        summary = {
            'ensemble_method': self.ensemble_method,
            'current_weights': self.dynamic_weights if self.ensemble_method == 'dynamic' else self.weights,
            'confidence_threshold': self.confidence_threshold,
            'risk_adjustment': self.risk_adjustment
        }
        
        # Add component model summaries
        summary['direction_model'] = self.direction_model.get_validation_summary()
        summary['volatility_model'] = self.volatility_model.get_validation_summary()
        summary['regime_model'] = self.regime_model.get_regime_statistics()
        
        # Add performance history summary if available
        if self.performance_history:
            recent_perf = self.performance_history[-20:]
            summary['recent_performance'] = {
                'avg_direction_accuracy': np.mean([p['direction_accuracy'] for p in recent_perf]),
                'avg_volatility_mae': np.mean([p['volatility_mae'] for p in recent_perf]),
                'avg_regime_accuracy': np.mean([p['regime_accuracy'] for p in recent_perf])
            }
        
        return summary