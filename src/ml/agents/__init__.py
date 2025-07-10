"""
ML Agents Module

This module contains all specialized agents for the ML pipeline
in the backtesting system.
"""

from .base_agent import BaseAgent
from .data_engineering_agent import DataEngineeringAgent
from .feature_analysis_agent import FeatureAnalysisAgent
from .model_architecture_agent import ModelArchitectureAgent
from .training_orchestrator_agent import TrainingOrchestratorAgent
from .market_regime_agent import MarketRegimeAgent
from .risk_modeling_agent import RiskModelingAgent
from .performance_analysis_agent import PerformanceAnalysisAgent
from .visualization_agent import VisualizationAgent
from .optimization_agent import OptimizationAgent
from .integration_agent import IntegrationAgent

__all__ = [
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
    'IntegrationAgent'
]

# Agent type mapping for dynamic instantiation
AGENT_TYPES = {
    'data_engineering': DataEngineeringAgent,
    'feature_analysis': FeatureAnalysisAgent,
    'model_architecture': ModelArchitectureAgent,
    'training_orchestrator': TrainingOrchestratorAgent,
    'market_regime': MarketRegimeAgent,
    'risk_modeling': RiskModelingAgent,
    'performance_analysis': PerformanceAnalysisAgent,
    'visualization': VisualizationAgent,
    'optimization': OptimizationAgent,
    'integration': IntegrationAgent
}

def create_agent(agent_type: str, config: dict) -> BaseAgent:
    """
    Factory function to create an agent instance.
    
    Args:
        agent_type: Type of agent to create
        config: Configuration dictionary for the agent
        
    Returns:
        Instance of the requested agent type
        
    Raises:
        ValueError: If agent_type is not recognized
    """
    if agent_type not in AGENT_TYPES:
        raise ValueError(f"Unknown agent type: {agent_type}. "
                        f"Available types: {list(AGENT_TYPES.keys())}")
    
    agent_class = AGENT_TYPES[agent_type]
    return agent_class(config)