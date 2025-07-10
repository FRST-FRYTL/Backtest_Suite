"""
Market Regime Detection and Adaptive Strategy System

This module implements market regime detection using machine learning
and provides adaptive strategy adjustments based on regime changes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import logging
import warnings
from pathlib import Path
import joblib
from scipy import stats

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_QUIET = "bull_quiet"           # Low volatility uptrend
    BULL_VOLATILE = "bull_volatile"     # High volatility uptrend  
    BEAR_QUIET = "bear_quiet"           # Low volatility downtrend
    BEAR_VOLATILE = "bear_volatile"     # High volatility downtrend
    SIDEWAYS = "sideways"               # Range-bound market
    CRISIS = "crisis"                   # Extreme volatility/drawdown

@dataclass
class RegimeCharacteristics:
    """Characteristics of a market regime"""
    regime: MarketRegime
    avg_return: float
    volatility: float
    trend_strength: float
    typical_duration_days: int
    confidence_threshold_adjustment: float
    position_size_multiplier: float
    stop_loss_adjustment: float

@dataclass
class RegimeDetectionResult:
    """Result from regime detection"""
    current_regime: MarketRegime
    regime_probability: float
    regime_history: pd.Series
    transition_matrix: np.ndarray
    regime_characteristics: Dict[MarketRegime, RegimeCharacteristics]
    change_points: List[pd.Timestamp]

class MarketRegimeDetector:
    """
    Detects market regimes and provides adaptive strategy parameters.
    """
    
    # Default regime characteristics
    DEFAULT_CHARACTERISTICS = {
        MarketRegime.BULL_QUIET: RegimeCharacteristics(
            regime=MarketRegime.BULL_QUIET,
            avg_return=0.0008,      # 0.08% daily
            volatility=0.01,        # 1% daily vol
            trend_strength=0.7,
            typical_duration_days=90,
            confidence_threshold_adjustment=-0.05,  # Lower threshold
            position_size_multiplier=1.2,           # Larger positions
            stop_loss_adjustment=1.5                # Wider stops
        ),
        MarketRegime.BULL_VOLATILE: RegimeCharacteristics(
            regime=MarketRegime.BULL_VOLATILE,
            avg_return=0.001,       # 0.1% daily
            volatility=0.02,        # 2% daily vol
            trend_strength=0.5,
            typical_duration_days=45,
            confidence_threshold_adjustment=0.0,
            position_size_multiplier=0.8,           # Smaller positions
            stop_loss_adjustment=2.0                # Wider stops for volatility
        ),
        MarketRegime.BEAR_QUIET: RegimeCharacteristics(
            regime=MarketRegime.BEAR_QUIET,
            avg_return=-0.0005,     # -0.05% daily
            volatility=0.012,       # 1.2% daily vol
            trend_strength=-0.6,
            typical_duration_days=60,
            confidence_threshold_adjustment=0.1,    # Higher threshold
            position_size_multiplier=0.5,           # Defensive sizing
            stop_loss_adjustment=0.8                # Tighter stops
        ),
        MarketRegime.BEAR_VOLATILE: RegimeCharacteristics(
            regime=MarketRegime.BEAR_VOLATILE,
            avg_return=-0.002,      # -0.2% daily
            volatility=0.03,        # 3% daily vol
            trend_strength=-0.7,
            typical_duration_days=30,
            confidence_threshold_adjustment=0.15,   # Much higher threshold
            position_size_multiplier=0.3,           # Very defensive
            stop_loss_adjustment=0.7                # Tight stops
        ),
        MarketRegime.SIDEWAYS: RegimeCharacteristics(
            regime=MarketRegime.SIDEWAYS,
            avg_return=0.0,
            volatility=0.015,       # 1.5% daily vol
            trend_strength=0.0,
            typical_duration_days=120,
            confidence_threshold_adjustment=0.05,
            position_size_multiplier=0.7,           # Reduced sizing
            stop_loss_adjustment=1.0                # Normal stops
        ),
        MarketRegime.CRISIS: RegimeCharacteristics(
            regime=MarketRegime.CRISIS,
            avg_return=-0.005,      # -0.5% daily
            volatility=0.05,        # 5% daily vol
            trend_strength=-0.9,
            typical_duration_days=20,
            confidence_threshold_adjustment=0.2,    # Very high threshold
            position_size_multiplier=0.1,           # Minimal exposure
            stop_loss_adjustment=0.5                # Very tight stops
        )
    }
    
    def __init__(
        self,
        n_regimes: int = 5,
        lookback_days: int = 252,
        min_regime_days: int = 20
    ):
        """
        Initialize the market regime detector.
        
        Args:
            n_regimes: Number of regimes to detect
            lookback_days: Days of history for regime detection
            min_regime_days: Minimum days to confirm regime change
        """
        self.n_regimes = n_regimes
        self.lookback_days = lookback_days
        self.min_regime_days = min_regime_days
        
        # Models
        self.hmm_model = None
        self.gmm_model = None
        self.scaler = StandardScaler()
        
        # State tracking
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_history = pd.Series(dtype=int)
        self.transition_matrix = None
        
    def fit(
        self,
        returns: pd.Series,
        volume: Optional[pd.Series] = None,
        vix: Optional[pd.Series] = None
    ) -> 'MarketRegimeDetector':
        """
        Fit regime detection models on historical data.
        
        Args:
            returns: Return series
            volume: Optional volume series
            vix: Optional VIX series
            
        Returns:
            Fitted detector
        """
        # Prepare features
        features = self._prepare_features(returns, volume, vix)
        
        # Fit HMM model
        logger.info("Fitting Hidden Markov Model...")
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit model
        self.hmm_model.fit(features_scaled)
        
        # Get transition matrix
        self.transition_matrix = self.hmm_model.transmat_
        
        # Fit GMM as backup
        logger.info("Fitting Gaussian Mixture Model...")
        self.gmm_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42
        )
        self.gmm_model.fit(features_scaled)
        
        # Predict historical regimes
        regime_predictions = self.hmm_model.predict(features_scaled)
        self.regime_history = pd.Series(regime_predictions, index=returns.index[-len(regime_predictions):])
        
        # Map numeric regimes to market regimes
        self._map_regimes_to_market_states(features, regime_predictions)
        
        logger.info("Regime detection model fitted successfully")
        return self
    
    def _prepare_features(
        self,
        returns: pd.Series,
        volume: Optional[pd.Series] = None,
        vix: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Prepare features for regime detection."""
        features = pd.DataFrame(index=returns.index)
        
        # Return features
        features['returns'] = returns
        features['returns_ma_5'] = returns.rolling(5).mean()
        features['returns_ma_20'] = returns.rolling(20).mean()
        
        # Volatility features
        features['volatility_10'] = returns.rolling(10).std()
        features['volatility_30'] = returns.rolling(30).std()
        features['volatility_ratio'] = features['volatility_10'] / features['volatility_30']
        
        # Trend features
        features['trend_20'] = returns.rolling(20).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]
        )
        features['trend_60'] = returns.rolling(60).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]
        )
        
        # Volume features (if available)
        if volume is not None:
            features['volume_ratio'] = volume / volume.rolling(20).mean()
            features['volume_trend'] = volume.rolling(20).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]
            )
        
        # VIX features (if available)
        if vix is not None:
            features['vix_level'] = vix
            features['vix_change'] = vix.pct_change()
            features['vix_percentile'] = vix.rolling(252).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 0 else 0
            )
        
        # Remove NaN values
        features = features.dropna()
        
        return features
    
    def _map_regimes_to_market_states(
        self,
        features: pd.DataFrame,
        regime_predictions: np.ndarray
    ):
        """Map numeric regimes to market regime states."""
        # Calculate regime statistics
        regime_stats = {}
        
        for regime in range(self.n_regimes):
            mask = regime_predictions == regime
            if mask.any():
                regime_features = features.iloc[-len(regime_predictions):][mask]
                
                avg_return = regime_features['returns'].mean()
                volatility = regime_features['returns'].std()
                trend = regime_features['trend_20'].mean() if 'trend_20' in regime_features else 0
                
                regime_stats[regime] = {
                    'avg_return': avg_return,
                    'volatility': volatility,
                    'trend': trend
                }
        
        # Map to market regimes based on characteristics
        self.regime_mapping = {}
        
        # Sort by average return
        sorted_regimes = sorted(regime_stats.items(), key=lambda x: x[1]['avg_return'])
        
        # Simple mapping logic
        if self.n_regimes >= 5:
            # Map based on return and volatility characteristics
            for i, (regime_id, stats) in enumerate(sorted_regimes):
                if i == 0:  # Lowest return
                    if stats['volatility'] > np.median([s['volatility'] for s in regime_stats.values()]):
                        self.regime_mapping[regime_id] = MarketRegime.BEAR_VOLATILE
                    else:
                        self.regime_mapping[regime_id] = MarketRegime.BEAR_QUIET
                elif i == len(sorted_regimes) - 1:  # Highest return
                    if stats['volatility'] > np.median([s['volatility'] for s in regime_stats.values()]):
                        self.regime_mapping[regime_id] = MarketRegime.BULL_VOLATILE
                    else:
                        self.regime_mapping[regime_id] = MarketRegime.BULL_QUIET
                else:  # Middle regimes
                    if abs(stats['avg_return']) < 0.0001:  # Near zero return
                        self.regime_mapping[regime_id] = MarketRegime.SIDEWAYS
                    elif stats['volatility'] > np.percentile([s['volatility'] for s in regime_stats.values()], 80):
                        self.regime_mapping[regime_id] = MarketRegime.CRISIS
                    else:
                        self.regime_mapping[regime_id] = MarketRegime.SIDEWAYS
    
    def detect_regime(
        self,
        recent_returns: pd.Series,
        recent_volume: Optional[pd.Series] = None,
        recent_vix: Optional[pd.Series] = None
    ) -> RegimeDetectionResult:
        """
        Detect current market regime.
        
        Args:
            recent_returns: Recent return series
            recent_volume: Recent volume series
            recent_vix: Recent VIX series
            
        Returns:
            Regime detection result
        """
        if self.hmm_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare features
        features = self._prepare_features(recent_returns, recent_volume, recent_vix)
        
        if len(features) == 0:
            logger.warning("Insufficient data for regime detection")
            return self._get_default_result()
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict regime
        try:
            # Get regime probabilities
            regime_probs = self.hmm_model.predict_proba(features_scaled[-1:])
            regime_prediction = np.argmax(regime_probs)
            regime_probability = regime_probs[0, regime_prediction]
            
            # Map to market regime
            if regime_prediction in self.regime_mapping:
                current_regime = self.regime_mapping[regime_prediction]
            else:
                current_regime = MarketRegime.SIDEWAYS
            
            # Detect regime changes
            recent_predictions = self.hmm_model.predict(features_scaled[-self.min_regime_days:])
            regime_series = pd.Series(recent_predictions, index=features.index[-self.min_regime_days:])
            
            # Find change points
            change_points = self._detect_change_points(regime_series)
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return self._get_default_result()
        
        # Get regime characteristics
        regime_characteristics = self._calculate_regime_characteristics(features, recent_predictions)
        
        return RegimeDetectionResult(
            current_regime=current_regime,
            regime_probability=regime_probability,
            regime_history=regime_series,
            transition_matrix=self.transition_matrix,
            regime_characteristics=regime_characteristics,
            change_points=change_points
        )
    
    def _detect_change_points(self, regime_series: pd.Series) -> List[pd.Timestamp]:
        """Detect regime change points."""
        change_points = []
        
        if len(regime_series) < 2:
            return change_points
        
        # Find where regime changes
        regime_changes = regime_series != regime_series.shift(1)
        change_indices = regime_series[regime_changes].index
        
        # Filter out noise (require regime to persist)
        for idx in change_indices:
            # Check if regime persists for at least a few days
            loc = regime_series.index.get_loc(idx)
            if loc + 3 < len(regime_series):
                if (regime_series.iloc[loc:loc+3] == regime_series.iloc[loc]).all():
                    change_points.append(idx)
        
        return change_points
    
    def _calculate_regime_characteristics(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray
    ) -> Dict[MarketRegime, RegimeCharacteristics]:
        """Calculate characteristics for each regime."""
        characteristics = {}
        
        # Use default characteristics as base
        for regime in MarketRegime:
            if regime in self.DEFAULT_CHARACTERISTICS:
                characteristics[regime] = self.DEFAULT_CHARACTERISTICS[regime]
        
        # Update with observed characteristics if available
        for numeric_regime, market_regime in self.regime_mapping.items():
            mask = predictions == numeric_regime
            if mask.any() and len(features) >= len(predictions):
                regime_data = features.iloc[-len(predictions):][mask]
                
                if len(regime_data) > 0:
                    # Update characteristics based on observed data
                    char = characteristics.get(market_regime, self.DEFAULT_CHARACTERISTICS[market_regime])
                    
                    # Create updated characteristics
                    updated_char = RegimeCharacteristics(
                        regime=market_regime,
                        avg_return=regime_data['returns'].mean(),
                        volatility=regime_data['returns'].std(),
                        trend_strength=regime_data['trend_20'].mean() if 'trend_20' in regime_data else char.trend_strength,
                        typical_duration_days=len(regime_data),
                        confidence_threshold_adjustment=char.confidence_threshold_adjustment,
                        position_size_multiplier=char.position_size_multiplier,
                        stop_loss_adjustment=char.stop_loss_adjustment
                    )
                    
                    characteristics[market_regime] = updated_char
        
        return characteristics
    
    def _get_default_result(self) -> RegimeDetectionResult:
        """Get default result when detection fails."""
        return RegimeDetectionResult(
            current_regime=MarketRegime.SIDEWAYS,
            regime_probability=0.5,
            regime_history=pd.Series(dtype=int),
            transition_matrix=np.eye(self.n_regimes) if self.n_regimes else np.array([[1]]),
            regime_characteristics=self.DEFAULT_CHARACTERISTICS,
            change_points=[]
        )
    
    def get_adaptive_parameters(
        self,
        base_parameters: Dict[str, float],
        current_regime: MarketRegime
    ) -> Dict[str, float]:
        """
        Get adapted strategy parameters for current regime.
        
        Args:
            base_parameters: Base strategy parameters
            current_regime: Current market regime
            
        Returns:
            Adapted parameters
        """
        characteristics = self.DEFAULT_CHARACTERISTICS.get(
            current_regime,
            self.DEFAULT_CHARACTERISTICS[MarketRegime.SIDEWAYS]
        )
        
        # Adapt parameters based on regime
        adapted_params = base_parameters.copy()
        
        # Adjust confluence threshold
        if 'confluence_threshold' in adapted_params:
            adapted_params['confluence_threshold'] += characteristics.confidence_threshold_adjustment
            adapted_params['confluence_threshold'] = max(0.4, min(0.9, adapted_params['confluence_threshold']))
        
        # Adjust position size
        if 'position_size' in adapted_params:
            adapted_params['position_size'] *= characteristics.position_size_multiplier
            adapted_params['position_size'] = max(0.01, min(0.25, adapted_params['position_size']))
        
        # Adjust stop loss
        if 'stop_loss_multiplier' in adapted_params:
            adapted_params['stop_loss_multiplier'] *= characteristics.stop_loss_adjustment
        
        # Add regime-specific parameters
        adapted_params['regime'] = current_regime.value
        adapted_params['regime_volatility'] = characteristics.volatility
        adapted_params['regime_trend'] = characteristics.trend_strength
        
        logger.info(f"Adapted parameters for {current_regime.value}: "
                   f"confluence={adapted_params.get('confluence_threshold', 0.65):.2f}, "
                   f"position_size={adapted_params.get('position_size', 0.1):.1%}")
        
        return adapted_params
    
    def calculate_regime_transition_probabilities(
        self,
        current_regime: MarketRegime,
        horizon_days: int = 20
    ) -> Dict[MarketRegime, float]:
        """
        Calculate probability of transitioning to each regime.
        
        Args:
            current_regime: Current market regime
            horizon_days: Time horizon for transition
            
        Returns:
            Transition probabilities
        """
        if self.transition_matrix is None:
            # Return uniform probabilities
            return {regime: 1.0 / len(MarketRegime) for regime in MarketRegime}
        
        # Get current regime index
        current_idx = None
        for idx, regime in self.regime_mapping.items():
            if regime == current_regime:
                current_idx = idx
                break
        
        if current_idx is None:
            # Default to equal probabilities
            return {regime: 1.0 / len(MarketRegime) for regime in MarketRegime}
        
        # Calculate multi-step transition probabilities
        multi_step_matrix = np.linalg.matrix_power(self.transition_matrix, horizon_days)
        
        # Get probabilities for current regime
        transition_probs = multi_step_matrix[current_idx]
        
        # Map to market regimes
        regime_probs = {}
        for idx, prob in enumerate(transition_probs):
            if idx in self.regime_mapping:
                market_regime = self.regime_mapping[idx]
                regime_probs[market_regime] = regime_probs.get(market_regime, 0) + prob
        
        # Ensure all regimes have probabilities
        for regime in MarketRegime:
            if regime not in regime_probs:
                regime_probs[regime] = 0.0
        
        return regime_probs
    
    def save_model(self, filepath: str):
        """Save fitted model to disk."""
        model_data = {
            'hmm_model': self.hmm_model,
            'gmm_model': self.gmm_model,
            'scaler': self.scaler,
            'regime_mapping': self.regime_mapping,
            'transition_matrix': self.transition_matrix,
            'n_regimes': self.n_regimes
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Saved regime detection model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load fitted model from disk."""
        model_data = joblib.load(filepath)
        
        self.hmm_model = model_data['hmm_model']
        self.gmm_model = model_data['gmm_model']
        self.scaler = model_data['scaler']
        self.regime_mapping = model_data['regime_mapping']
        self.transition_matrix = model_data['transition_matrix']
        self.n_regimes = model_data['n_regimes']
        
        logger.info(f"Loaded regime detection model from {filepath}")