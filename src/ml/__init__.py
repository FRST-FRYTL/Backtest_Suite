"""
Machine Learning Module for Backtesting Suite

This module provides a comprehensive ML pipeline for developing,
training, and evaluating trading strategies using machine learning.
"""

from .agents import (
    BaseAgent,
    DataEngineeringAgent,
    FeatureAnalysisAgent,
    ModelArchitectureAgent,
    TrainingOrchestratorAgent,
    MarketRegimeAgent,
    RiskModelingAgent,
    PerformanceAnalysisAgent,
    VisualizationAgent,
    OptimizationAgent,
    IntegrationAgent,
    create_agent,
    AGENT_TYPES
)

# Import ML models
from .models.enhanced_direction_predictor import EnhancedDirectionPredictor
from .models.lstm_volatility import VolatilityForecaster
from .market_regime_detector import MarketRegimeDetector

# Create aliases for backward compatibility
DirectionPredictor = EnhancedDirectionPredictor

__version__ = "1.0.0"

__all__ = [
    # Agents
    'BaseAgent',
    'DataEngineeringAgent',
    'FeatureAnalysisAgent',
    'ModelArchitectureAgent',
    'TrainingOrchestratorAgent',
    'MarketRegimeAgent',
    'RiskModelingAgent',
    'PerformanceAnalysisAgent',
    'VisualizationAgent',
    'OptimizationAgent',
    'IntegrationAgent',
    
    # Factory functions
    'create_agent',
    'AGENT_TYPES',
    
    # ML Models
    'EnhancedDirectionPredictor',
    'DirectionPredictor',
    'VolatilityForecaster',
    'MarketRegimeDetector',
    
    # Version
    '__version__'
]