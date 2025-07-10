"""
ML Optimization Module

This module provides a 5-loop optimization system for machine learning trading strategies:
1. Feature Engineering Optimization
2. Model Architecture Optimization  
3. Market Regime Optimization
4. Risk Management Optimization
5. Integration & Ensemble Optimization
"""

from .optimization_orchestrator import OptimizationOrchestrator, OptimizationResult
from .feature_optimization import FeatureOptimization
from .architecture_optimization import ArchitectureOptimization
from .regime_optimization import RegimeOptimization
from .risk_optimization import RiskOptimization
from .integration_optimization import IntegrationOptimization

__all__ = [
    'OptimizationOrchestrator',
    'OptimizationResult',
    'FeatureOptimization',
    'ArchitectureOptimization',
    'RegimeOptimization',
    'RiskOptimization',
    'IntegrationOptimization'
]

# Version info
__version__ = '1.0.0'