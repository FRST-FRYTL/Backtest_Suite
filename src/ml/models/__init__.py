"""
Machine Learning models for trading strategy optimization.
"""

from .xgboost_direction import EnhancedDirectionPredictor, DirectionPrediction
from .lstm_volatility import VolatilityForecaster, VolatilityPrediction
from .regime_detection import MarketRegimeDetector, RegimeDetection, MarketRegime
from .ensemble import EnsembleModel, EnsemblePrediction

# Aliases for backward compatibility
DirectionPredictor = EnhancedDirectionPredictor

__all__ = [
    # Direction prediction
    'DirectionPredictor',
    'EnhancedDirectionPredictor',
    'DirectionPrediction',
    
    # Volatility forecasting
    'VolatilityForecaster', 
    'VolatilityPrediction',
    
    # Regime detection
    'MarketRegimeDetector',
    'RegimeDetection',
    'MarketRegime',
    
    # Ensemble
    'EnsembleModel',
    'EnsemblePrediction'
]