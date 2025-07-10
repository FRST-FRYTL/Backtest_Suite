"""
Base ML Agent Class for Backtesting Suite

This module provides the abstract base class for all ML agents in the system.
Each specialized agent inherits from this base class and implements specific
functionality for their domain.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import logging
from datetime import datetime
import json
import traceback


class BaseAgent(ABC):
    """
    Abstract base class for all ML agents in the backtesting system.
    
    Provides common functionality for initialization, execution, status reporting,
    and result retrieval with built-in error handling and logging.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name for identification and logging
            config: Configuration dictionary containing agent-specific parameters
        """
        self.name = name
        self.config = config
        self.logger = self._setup_logger()
        self.status = "initialized"
        self.results = {}
        self.errors = []
        self.start_time = None
        self.end_time = None
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Set up agent-specific logger."""
        logger = logging.getLogger(f"ml.agents.{self.name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize agent-specific resources and dependencies.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's main functionality.
        
        Args:
            **kwargs: Agent-specific keyword arguments
            
        Returns:
            Dict containing execution results
        """
        pass
    
    def report_status(self) -> Dict[str, Any]:
        """
        Report current agent status and metrics.
        
        Returns:
            Dict containing status information
        """
        runtime = None
        if self.start_time:
            end = self.end_time or datetime.now()
            runtime = (end - self.start_time).total_seconds()
            
        return {
            "agent_name": self.name,
            "status": self.status,
            "errors": self.errors,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "runtime_seconds": runtime,
            "metadata": self.metadata,
            "has_results": bool(self.results)
        }
    
    def get_results(self) -> Dict[str, Any]:
        """
        Retrieve agent execution results.
        
        Returns:
            Dict containing execution results and metadata
        """
        return {
            "agent_name": self.name,
            "status": self.status,
            "results": self.results,
            "errors": self.errors,
            "metadata": self.metadata,
            "execution_time": {
                "start": self.start_time.isoformat() if self.start_time else None,
                "end": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": (
                    (self.end_time - self.start_time).total_seconds() 
                    if self.start_time and self.end_time else None
                )
            }
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Main execution wrapper with error handling and logging.
        
        Args:
            **kwargs: Arguments to pass to execute method
            
        Returns:
            Dict containing execution results
        """
        try:
            self.logger.info(f"Starting {self.name} agent execution")
            self.start_time = datetime.now()
            self.status = "running"
            
            # Initialize if not already done
            if self.status == "initialized":
                init_success = self.initialize()
                if not init_success:
                    raise RuntimeError(f"Failed to initialize {self.name} agent")
            
            # Execute main functionality
            self.results = self.execute(**kwargs)
            
            self.status = "completed"
            self.logger.info(f"{self.name} agent execution completed successfully")
            
        except Exception as e:
            self.status = "failed"
            error_msg = f"Error in {self.name} agent: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.errors.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
        finally:
            self.end_time = datetime.now()
            
        return self.get_results()
    
    def reset(self):
        """Reset agent state for fresh execution."""
        self.status = "initialized"
        self.results = {}
        self.errors = []
        self.start_time = None
        self.end_time = None
        self.logger.info(f"{self.name} agent state reset")
    
    def save_results(self, filepath: str):
        """
        Save agent results to JSON file.
        
        Args:
            filepath: Path to save results
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.get_results(), f, indent=2, default=str)
            self.logger.info(f"Results saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise
    
    def load_config(self, config_path: str):
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    self.config = json.load(f)
                elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    import yaml
                    self.config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path}")
            self.logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            raise
    
    def validate_config(self, required_keys: List[str]) -> bool:
        """
        Validate that required configuration keys are present.
        
        Args:
            required_keys: List of required configuration keys
            
        Returns:
            bool: True if all required keys present, False otherwise
        """
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            self.logger.error(f"Missing required config keys: {missing_keys}")
            return False
        return True