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
    
    # Version
    '__version__'
]