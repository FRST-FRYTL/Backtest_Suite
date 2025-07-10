"""
Integration Agent for ML Pipeline

Integrates all ML components and coordinates the complete pipeline execution
for the backtesting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import pickle
import joblib
from datetime import datetime
import os
import shutil

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


class IntegrationAgent(BaseAgent):
    """
    Agent responsible for integrating and coordinating all ML pipeline components:
    - Pipeline orchestration
    - Component communication
    - Result aggregation
    - Error handling and recovery
    - Pipeline versioning and reproducibility
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("IntegrationAgent", config)
        self.agents = {}
        self.pipeline_results = {}
        self.pipeline_version = None
        self.execution_order = []
        
    def initialize(self) -> bool:
        """Initialize integration agent and all sub-agents."""
        try:
            self.logger.info("Initializing Integration Agent")
            
            # Validate required configuration
            required_keys = ["pipeline_config", "output_dir", "agents_config"]
            if not self.validate_config(required_keys):
                return False
            
            # Initialize pipeline settings
            self.pipeline_config = self.config.get("pipeline_config", {})
            self.output_dir = self.config.get("output_dir", "ml_output")
            self.agents_config = self.config.get("agents_config", {})
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Initialize execution order
            self.execution_order = self.pipeline_config.get("execution_order", [
                "data_engineering",
                "feature_analysis",
                "market_regime",
                "model_architecture",
                "training_orchestrator",
                "risk_modeling",
                "performance_analysis",
                "optimization",
                "visualization"
            ])
            
            # Initialize all agents
            self._initialize_agents()
            
            # Create pipeline version
            self.pipeline_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            self.logger.info("Integration Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False
    
    def _initialize_agents(self):
        """Initialize all ML agents."""
        self.logger.info("Initializing ML agents")
        
        # Create agent instances
        agent_classes = {
            "data_engineering": DataEngineeringAgent,
            "feature_analysis": FeatureAnalysisAgent,
            "model_architecture": ModelArchitectureAgent,
            "training_orchestrator": TrainingOrchestratorAgent,
            "market_regime": MarketRegimeAgent,
            "risk_modeling": RiskModelingAgent,
            "performance_analysis": PerformanceAnalysisAgent,
            "visualization": VisualizationAgent,
            "optimization": OptimizationAgent
        }
        
        for agent_name, agent_class in agent_classes.items():
            agent_config = self.agents_config.get(agent_name, {})
            agent = agent_class(agent_config)
            
            if agent.initialize():
                self.agents[agent_name] = agent
                self.logger.info(f"Initialized {agent_name} agent")
            else:
                self.logger.error(f"Failed to initialize {agent_name} agent")
    
    def execute(self, data_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute complete ML pipeline.
        
        Args:
            data_path: Path to input data
            **kwargs: Additional parameters for pipeline execution
            
        Returns:
            Dict containing integrated pipeline results
        """
        try:
            self.logger.info("Starting ML pipeline execution")
            pipeline_start_time = datetime.now()
            
            # Create pipeline run directory
            run_dir = os.path.join(self.output_dir, f"run_{self.pipeline_version}")
            os.makedirs(run_dir, exist_ok=True)
            
            # Execute pipeline stages
            results = {}
            
            # Stage 1: Data Engineering
            if "data_engineering" in self.execution_order:
                results["data_engineering"] = self._execute_data_engineering(
                    data_path, **kwargs
                )
            
            # Stage 2: Feature Analysis
            if "feature_analysis" in self.execution_order:
                results["feature_analysis"] = self._execute_feature_analysis(
                    results.get("data_engineering", {})
                )
            
            # Stage 3: Market Regime Analysis
            if "market_regime" in self.execution_order:
                results["market_regime"] = self._execute_market_regime(
                    results.get("data_engineering", {})
                )
            
            # Stage 4: Model Architecture Design
            if "model_architecture" in self.execution_order:
                results["model_architecture"] = self._execute_model_architecture(
                    results.get("data_engineering", {}),
                    results.get("feature_analysis", {})
                )
            
            # Stage 5: Model Training
            if "training_orchestrator" in self.execution_order:
                results["training"] = self._execute_training(
                    results.get("data_engineering", {}),
                    results.get("model_architecture", {}),
                    results.get("feature_analysis", {})
                )
            
            # Stage 6: Risk Modeling
            if "risk_modeling" in self.execution_order:
                results["risk_modeling"] = self._execute_risk_modeling(
                    results.get("training", {}),
                    results.get("data_engineering", {})
                )
            
            # Stage 7: Performance Analysis
            if "performance_analysis" in self.execution_order:
                results["performance_analysis"] = self._execute_performance_analysis(
                    results.get("training", {}),
                    results.get("data_engineering", {})
                )
            
            # Stage 8: Optimization
            if "optimization" in self.execution_order:
                results["optimization"] = self._execute_optimization(
                    results.get("performance_analysis", {}),
                    results.get("model_architecture", {})
                )
            
            # Stage 9: Visualization
            if "visualization" in self.execution_order:
                results["visualization"] = self._execute_visualization(results)
            
            # Aggregate results
            pipeline_results = self._aggregate_results(results)
            
            # Generate pipeline report
            report = self._generate_pipeline_report(
                pipeline_results, 
                pipeline_start_time
            )
            
            # Save pipeline artifacts
            self._save_pipeline_artifacts(
                pipeline_results, 
                report, 
                run_dir
            )
            
            # Store results
            self.pipeline_results = pipeline_results
            
            return {
                "pipeline_results": pipeline_results,
                "report": report,
                "artifacts_dir": run_dir,
                "execution_time": (datetime.now() - pipeline_start_time).total_seconds(),
                "pipeline_version": self.pipeline_version
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise
    
    def _execute_data_engineering(self, data_path: Optional[str], 
                                **kwargs) -> Dict[str, Any]:
        """Execute data engineering stage."""
        self.logger.info("Executing data engineering stage")
        
        agent = self.agents.get("data_engineering")
        if not agent:
            return {"error": "Data engineering agent not available"}
        
        results = agent.execute(
            data_path=data_path,
            start_date=kwargs.get("start_date"),
            end_date=kwargs.get("end_date"),
            train_ratio=kwargs.get("train_ratio", 0.7),
            val_ratio=kwargs.get("val_ratio", 0.15)
        )
        
        return results
    
    def _execute_feature_analysis(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature analysis stage."""
        self.logger.info("Executing feature analysis stage")
        
        agent = self.agents.get("feature_analysis")
        if not agent:
            return {"error": "Feature analysis agent not available"}
        
        # Get processed data from data engineering
        data_agent = self.agents.get("data_engineering")
        if not data_agent or not data_agent.processed_data:
            return {"error": "No processed data available"}
        
        train_data = data_agent.processed_data["train"]
        
        # Separate features and target
        target_col = agent.config.get("target_variable", "returns")
        if target_col in train_data.columns:
            features = train_data.drop(columns=[target_col])
            target = train_data[target_col]
        else:
            features = train_data
            target = None
        
        results = agent.execute(features, target)
        
        return results
    
    def _execute_market_regime(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute market regime analysis stage."""
        self.logger.info("Executing market regime analysis stage")
        
        agent = self.agents.get("market_regime")
        if not agent:
            return {"error": "Market regime agent not available"}
        
        # Get raw data from data engineering
        data_agent = self.agents.get("data_engineering")
        if not data_agent or not data_agent.processed_data:
            return {"error": "No data available"}
        
        # Use training data for regime analysis
        data = data_agent.processed_data["train"]
        
        results = agent.execute(data)
        
        return results
    
    def _execute_model_architecture(self, data_results: Dict[str, Any],
                                  feature_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model architecture design stage."""
        self.logger.info("Executing model architecture design stage")
        
        agent = self.agents.get("model_architecture")
        if not agent:
            return {"error": "Model architecture agent not available"}
        
        # Get data
        data_agent = self.agents.get("data_engineering")
        if not data_agent or not data_agent.processed_data:
            return {"error": "No data available"}
        
        train_data = data_agent.processed_data["train"]
        val_data = data_agent.processed_data["val"]
        
        # Get selected features
        feature_agent = self.agents.get("feature_analysis")
        selected_features = []
        if feature_agent and hasattr(feature_agent, 'selected_features'):
            selected_features = feature_agent.selected_features
        
        # Prepare data
        target_col = agent.config.get("target_variable", "returns")
        
        if selected_features:
            X_train = train_data[selected_features]
            X_val = val_data[selected_features]
        else:
            X_train = train_data.drop(columns=[target_col], errors='ignore')
            X_val = val_data.drop(columns=[target_col], errors='ignore')
        
        y_train = train_data[target_col] if target_col in train_data.columns else None
        y_val = val_data[target_col] if target_col in val_data.columns else None
        
        results = agent.execute(X_train, y_train, X_val, y_val)
        
        return results
    
    def _execute_training(self, data_results: Dict[str, Any],
                        architecture_results: Dict[str, Any],
                        feature_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model training stage."""
        self.logger.info("Executing model training stage")
        
        agent = self.agents.get("training_orchestrator")
        if not agent:
            return {"error": "Training orchestrator agent not available"}
        
        # Get model configuration
        arch_agent = self.agents.get("model_architecture")
        if not arch_agent or not arch_agent.recommended_architecture:
            return {"error": "No model architecture available"}
        
        model_config = {
            "model_class": arch_agent.candidate_models[
                arch_agent.recommended_architecture["model_type"]
            ]["model_class"],
            "base_params": arch_agent.recommended_architecture["recommended_config"],
            "search_space": arch_agent._recommend_hyperparameters(
                arch_agent.recommended_architecture,
                {"n_samples": 1000, "n_features": 10}  # Dummy profile
            )["search_space"]
        }
        
        # Get data
        data_agent = self.agents.get("data_engineering")
        if not data_agent or not data_agent.processed_data:
            return {"error": "No data available"}
        
        train_data = data_agent.processed_data["train"]
        val_data = data_agent.processed_data["val"]
        
        # Get selected features
        feature_agent = self.agents.get("feature_analysis")
        selected_features = []
        if feature_agent and hasattr(feature_agent, 'selected_features'):
            selected_features = feature_agent.selected_features
        
        # Prepare data
        target_col = "returns"  # Default target
        
        if selected_features:
            X_train = train_data[selected_features]
            X_val = val_data[selected_features]
        else:
            X_train = train_data.drop(columns=[target_col], errors='ignore')
            X_val = val_data.drop(columns=[target_col], errors='ignore')
        
        y_train = train_data[target_col] if target_col in train_data.columns else None
        y_val = val_data[target_col] if target_col in val_data.columns else None
        
        results = agent.execute(model_config, X_train, y_train, X_val, y_val)
        
        return results
    
    def _execute_risk_modeling(self, training_results: Dict[str, Any],
                             data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute risk modeling stage."""
        self.logger.info("Executing risk modeling stage")
        
        agent = self.agents.get("risk_modeling")
        if not agent:
            return {"error": "Risk modeling agent not available"}
        
        # Get returns data
        data_agent = self.agents.get("data_engineering")
        if not data_agent or not data_agent.processed_data:
            return {"error": "No data available"}
        
        # Use test data for risk analysis
        test_data = data_agent.processed_data["test"]
        
        # Get model predictions if available
        train_agent = self.agents.get("training_orchestrator")
        if train_agent and train_agent.best_model:
            # Generate predictions
            model = train_agent.best_model
            X_test = test_data.drop(columns=["returns"], errors='ignore')
            
            try:
                predictions = model.predict(X_test)
                # Convert predictions to returns/signals
                returns = pd.Series(predictions, index=test_data.index, name="returns")
            except:
                returns = test_data[["returns"]] if "returns" in test_data.columns else test_data
        else:
            returns = test_data[["returns"]] if "returns" in test_data.columns else test_data
        
        results = agent.execute(returns)
        
        return results
    
    def _execute_performance_analysis(self, training_results: Dict[str, Any],
                                    data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute performance analysis stage."""
        self.logger.info("Executing performance analysis stage")
        
        agent = self.agents.get("performance_analysis")
        if not agent:
            return {"error": "Performance analysis agent not available"}
        
        # Get test data
        data_agent = self.agents.get("data_engineering")
        if not data_agent or not data_agent.processed_data:
            return {"error": "No data available"}
        
        test_data = data_agent.processed_data["test"]
        
        # Get model and predictions
        train_agent = self.agents.get("training_orchestrator")
        if train_agent and train_agent.best_model:
            model = train_agent.best_model
            
            # Prepare test features
            target_col = "returns"
            X_test = test_data.drop(columns=[target_col], errors='ignore')
            y_test = test_data[target_col] if target_col in test_data.columns else None
            
            # Generate predictions
            predictions = model.predict(X_test)
            
            # Convert to trading signals (simple threshold)
            positions = pd.Series(
                (predictions > 0).astype(int), 
                index=test_data.index
            )
            
            # Get price data for trading performance
            prices = test_data[["close"]] if "close" in test_data.columns else None
            
            results = agent.execute(
                predictions=predictions,
                actual=y_test,
                prices=prices,
                positions=positions,
                task_type="regression"
            )
        else:
            results = {"error": "No trained model available"}
        
        return results
    
    def _execute_optimization(self, performance_results: Dict[str, Any],
                            architecture_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimization stage."""
        self.logger.info("Executing optimization stage")
        
        agent = self.agents.get("optimization")
        if not agent:
            return {"error": "Optimization agent not available"}
        
        # Define objective function for strategy optimization
        def strategy_objective(params):
            # This is a placeholder - in practice, you'd backtest with these params
            # and return the performance metric
            return np.random.random()  # Placeholder
        
        # Define search space
        search_space = {
            "lookback_period": {"type": "int", "low": 10, "high": 100},
            "threshold": {"type": "float", "low": 0.0, "high": 1.0},
            "stop_loss": {"type": "float", "low": 0.01, "high": 0.1}
        }
        
        results = agent.execute(strategy_objective, search_space)
        
        return results
    
    def _execute_visualization(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute visualization stage."""
        self.logger.info("Executing visualization stage")
        
        agent = self.agents.get("visualization")
        if not agent:
            return {"error": "Visualization agent not available"}
        
        # Prepare data for visualization
        viz_data = {}
        
        # Get raw data
        data_agent = self.agents.get("data_engineering")
        if data_agent and data_agent.processed_data:
            viz_data["raw_data"] = data_agent.processed_data["train"]
        
        # Get features and importance
        feature_agent = self.agents.get("feature_analysis")
        if feature_agent:
            if hasattr(feature_agent, 'processed_data') and feature_agent.processed_data:
                viz_data["features"] = feature_agent.processed_data["train"]
            if hasattr(feature_agent, 'feature_importance_scores'):
                viz_data["feature_importance"] = feature_agent.feature_importance_scores
        
        # Get predictions and performance
        perf_results = all_results.get("performance_analysis", {})
        if "performance_metrics" in perf_results:
            metrics = perf_results["performance_metrics"]
            if "model_performance" in metrics:
                # Extract predictions/actual from stored results
                # This is simplified - in practice you'd store these during analysis
                viz_data["predictions"] = np.random.randn(100)  # Placeholder
                viz_data["actual"] = np.random.randn(100)  # Placeholder
        
        # Get risk metrics
        risk_results = all_results.get("risk_modeling", {})
        if "risk_metrics" in risk_results:
            viz_data["risk_metrics"] = risk_results["risk_metrics"]
        
        results = agent.execute(viz_data)
        
        return results
    
    def _aggregate_results(self, stage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from all pipeline stages."""
        self.logger.info("Aggregating pipeline results")
        
        aggregated = {
            "pipeline_version": self.pipeline_version,
            "execution_timestamp": datetime.now().isoformat(),
            "stages_completed": list(stage_results.keys()),
            "summary_metrics": {},
            "key_insights": [],
            "recommendations": []
        }
        
        # Extract key metrics from each stage
        if "data_engineering" in stage_results:
            data_res = stage_results["data_engineering"]
            if "data_shape" in data_res:
                aggregated["summary_metrics"]["data_size"] = data_res["data_shape"]
        
        if "feature_analysis" in stage_results:
            feat_res = stage_results["feature_analysis"]
            if "selected_features" in feat_res:
                aggregated["summary_metrics"]["n_features_selected"] = len(
                    feat_res["selected_features"]
                )
                aggregated["key_insights"].append(
                    f"Selected {len(feat_res['selected_features'])} features for modeling"
                )
        
        if "model_architecture" in stage_results:
            arch_res = stage_results["model_architecture"]
            if "recommended_architecture" in arch_res:
                aggregated["summary_metrics"]["model_type"] = (
                    arch_res["recommended_architecture"]["model_type"]
                )
        
        if "training" in stage_results:
            train_res = stage_results["training"]
            if "best_params" in train_res:
                aggregated["summary_metrics"]["hyperparameters_optimized"] = True
        
        if "performance_analysis" in stage_results:
            perf_res = stage_results["performance_analysis"]
            if "performance_metrics" in perf_res:
                metrics = perf_res["performance_metrics"]
                if "trading_performance" in metrics and metrics["trading_performance"]:
                    trading = metrics["trading_performance"]
                    aggregated["summary_metrics"]["sharpe_ratio"] = trading.get("sharpe_ratio")
                    aggregated["summary_metrics"]["annual_return"] = trading.get("annual_return")
                    aggregated["summary_metrics"]["max_drawdown"] = trading.get("max_drawdown")
        
        if "risk_modeling" in stage_results:
            risk_res = stage_results["risk_modeling"]
            if "risk_metrics" in risk_res:
                aggregated["key_insights"].append(
                    "Comprehensive risk analysis completed"
                )
        
        # Generate recommendations
        aggregated["recommendations"] = self._generate_pipeline_recommendations(
            stage_results, aggregated["summary_metrics"]
        )
        
        return aggregated
    
    def _generate_pipeline_recommendations(self, results: Dict[str, Any],
                                         metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on pipeline results."""
        recommendations = []
        
        # Model performance recommendations
        sharpe = metrics.get("sharpe_ratio", 0)
        if sharpe < 1:
            recommendations.append(
                "Consider alternative model architectures to improve risk-adjusted returns"
            )
        
        # Feature recommendations
        n_features = metrics.get("n_features_selected", 0)
        if n_features > 50:
            recommendations.append(
                "Large number of features selected - consider dimensionality reduction"
            )
        
        # Risk recommendations
        max_dd = abs(metrics.get("max_drawdown", 0))
        if max_dd > 0.2:
            recommendations.append(
                "High maximum drawdown detected - implement risk management strategies"
            )
        
        # Data recommendations
        if "data_size" in metrics:
            train_size = metrics["data_size"].get("train", [0])[0]
            if train_size < 1000:
                recommendations.append(
                    "Limited training data - consider data augmentation techniques"
                )
        
        return recommendations
    
    def _generate_pipeline_report(self, results: Dict[str, Any],
                                start_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive pipeline execution report."""
        self.logger.info("Generating pipeline report")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        report = {
            "executive_summary": {
                "pipeline_version": self.pipeline_version,
                "execution_date": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_execution_time": f"{execution_time:.2f} seconds",
                "stages_completed": results["stages_completed"],
                "overall_success": len(results["stages_completed"]) == len(self.execution_order)
            },
            "key_results": results["summary_metrics"],
            "insights": results["key_insights"],
            "recommendations": results["recommendations"],
            "stage_details": {},
            "configuration": {
                "pipeline_config": self.pipeline_config,
                "agents_config": {
                    agent: self.agents[agent].config 
                    for agent in self.agents
                }
            }
        }
        
        # Add stage-specific details
        for stage in results["stages_completed"]:
            agent = self.agents.get(stage)
            if agent:
                report["stage_details"][stage] = {
                    "status": agent.status,
                    "execution_time": agent.end_time - agent.start_time 
                                    if agent.start_time and agent.end_time else None,
                    "errors": agent.errors
                }
        
        return report
    
    def _save_pipeline_artifacts(self, results: Dict[str, Any],
                                report: Dict[str, Any],
                                output_dir: str):
        """Save all pipeline artifacts."""
        self.logger.info("Saving pipeline artifacts")
        
        # Save aggregated results
        with open(os.path.join(output_dir, "pipeline_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save report
        with open(os.path.join(output_dir, "pipeline_report.json"), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save trained models
        models_dir = os.path.join(output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        train_agent = self.agents.get("training_orchestrator")
        if train_agent and train_agent.best_model:
            model_path = os.path.join(models_dir, "best_model.pkl")
            joblib.dump(train_agent.best_model, model_path)
            
            # Save model parameters
            params_path = os.path.join(models_dir, "best_params.json")
            with open(params_path, 'w') as f:
                json.dump(train_agent.best_params, f, indent=2, default=str)
        
        # Save visualizations
        viz_agent = self.agents.get("visualization")
        if viz_agent and hasattr(viz_agent, 'visualizations'):
            viz_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Copy visualization files
            for category, viz_files in viz_agent.visualizations.items():
                if isinstance(viz_files, dict):
                    for viz_name, viz_path in viz_files.items():
                        if os.path.exists(viz_path):
                            dest_path = os.path.join(
                                viz_dir, 
                                f"{category}_{viz_name}.png"
                            )
                            shutil.copy2(viz_path, dest_path)
        
        # Save pipeline metadata
        metadata = {
            "pipeline_version": self.pipeline_version,
            "execution_date": datetime.now().isoformat(),
            "configuration": self.config,
            "stages_executed": list(results.get("stages_completed", [])),
            "artifacts": {
                "results": "pipeline_results.json",
                "report": "pipeline_report.json",
                "models": "models/",
                "visualizations": "visualizations/"
            }
        }
        
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Artifacts saved to {output_dir}")
    
    def load_pipeline_results(self, pipeline_version: str) -> Dict[str, Any]:
        """Load results from a previous pipeline run."""
        run_dir = os.path.join(self.output_dir, f"run_{pipeline_version}")
        
        if not os.path.exists(run_dir):
            raise ValueError(f"Pipeline version {pipeline_version} not found")
        
        # Load results
        results_path = os.path.join(run_dir, "pipeline_results.json")
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Load report
        report_path = os.path.join(run_dir, "pipeline_report.json")
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        # Load model if exists
        model_path = os.path.join(run_dir, "models", "best_model.pkl")
        model = None
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        
        return {
            "results": results,
            "report": report,
            "model": model,
            "artifacts_dir": run_dir
        }
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of current pipeline state."""
        return {
            "pipeline_version": self.pipeline_version,
            "agents_initialized": list(self.agents.keys()),
            "execution_order": self.execution_order,
            "has_results": bool(self.pipeline_results),
            "output_directory": self.output_dir
        }