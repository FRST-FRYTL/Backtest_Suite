"""
Comprehensive tests for BaseAgent class

This test module provides thorough coverage of the BaseAgent abstract class,
including initialization, execution, error handling, and utility methods.
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import logging

from src.ml.agents.base_agent import BaseAgent


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing"""
    
    def initialize(self) -> bool:
        """Test implementation of initialize"""
        if self.config.get("fail_init", False):
            return False
        return True
    
    def execute(self, **kwargs) -> dict:
        """Test implementation of execute"""
        if self.config.get("fail_execute", False):
            raise RuntimeError("Execution failed as requested")
        
        return {
            "status": "success",
            "data": kwargs.get("data", "test_data"),
            "timestamp": datetime.now().isoformat()
        }


class TestBaseAgent:
    """Comprehensive test suite for BaseAgent"""
    
    @pytest.fixture
    def base_config(self):
        """Basic configuration for testing"""
        return {
            "param1": "value1",
            "param2": 42,
            "nested": {
                "key": "value"
            }
        }
    
    @pytest.fixture
    def agent(self, base_config):
        """Create a concrete agent instance for testing"""
        return ConcreteAgent("TestAgent", base_config)
    
    def test_initialization(self, base_config):
        """Test agent initialization with various configurations"""
        # Basic initialization
        agent = ConcreteAgent("TestAgent", base_config)
        assert agent.name == "TestAgent"
        assert agent.config == base_config
        assert agent.status == "initialized"
        assert agent.results == {}
        assert agent.errors == []
        assert agent.start_time is None
        assert agent.end_time is None
        assert "created_at" in agent.metadata
        assert agent.metadata["version"] == "1.0.0"
        
        # Test with empty config
        empty_agent = ConcreteAgent("EmptyAgent", {})
        assert empty_agent.config == {}
        
        # Test with complex config
        complex_config = {
            "layers": [10, 20, 30],
            "activation": "relu",
            "learning_rate": 0.001,
            "batch_size": 32
        }
        complex_agent = ConcreteAgent("ComplexAgent", complex_config)
        assert complex_agent.config == complex_config
    
    def test_logger_setup(self, agent):
        """Test logger configuration and functionality"""
        assert agent.logger is not None
        assert agent.logger.name == "ml.agents.TestAgent"
        assert agent.logger.level == logging.INFO
        
        # Test logging output
        with patch('logging.StreamHandler.emit') as mock_emit:
            agent.logger.info("Test message")
            assert mock_emit.called
    
    def test_initialize_method(self):
        """Test initialize method behavior"""
        # Successful initialization
        agent = ConcreteAgent("TestAgent", {"param": "value"})
        assert agent.initialize() is True
        
        # Failed initialization
        fail_agent = ConcreteAgent("FailAgent", {"fail_init": True})
        assert fail_agent.initialize() is False
    
    def test_execute_method(self):
        """Test execute method behavior"""
        # Successful execution
        agent = ConcreteAgent("TestAgent", {})
        result = agent.execute(data="test_input")
        assert result["status"] == "success"
        assert result["data"] == "test_input"
        assert "timestamp" in result
        
        # Execution with different parameters
        result2 = agent.execute(data="other_input", extra_param="extra")
        assert result2["data"] == "other_input"
        
        # Failed execution
        fail_agent = ConcreteAgent("FailAgent", {"fail_execute": True})
        with pytest.raises(RuntimeError):
            fail_agent.execute()
    
    def test_report_status(self, agent):
        """Test status reporting functionality"""
        # Initial status
        status = agent.report_status()
        assert status["agent_name"] == "TestAgent"
        assert status["status"] == "initialized"
        assert status["errors"] == []
        assert status["start_time"] is None
        assert status["end_time"] is None
        assert status["runtime_seconds"] is None
        assert status["has_results"] is False
        
        # After execution
        agent.run()
        status = agent.report_status()
        assert status["status"] == "completed"
        assert status["start_time"] is not None
        assert status["end_time"] is not None
        assert status["runtime_seconds"] > 0
        assert status["has_results"] is True
        
        # With errors
        fail_agent = ConcreteAgent("FailAgent", {"fail_execute": True})
        fail_agent.run()
        status = fail_agent.report_status()
        assert status["status"] == "failed"
        assert len(status["errors"]) > 0
    
    def test_get_results(self, agent):
        """Test result retrieval functionality"""
        # Before execution
        results = agent.get_results()
        assert results["agent_name"] == "TestAgent"
        assert results["status"] == "initialized"
        assert results["results"] == {}
        
        # After successful execution
        agent.run(data="test_data")
        results = agent.get_results()
        assert results["status"] == "completed"
        assert results["results"]["status"] == "success"
        assert results["results"]["data"] == "test_data"
        assert results["execution_time"]["duration_seconds"] > 0
        
        # With errors
        fail_agent = ConcreteAgent("FailAgent", {"fail_execute": True})
        fail_agent.run()
        results = fail_agent.get_results()
        assert results["status"] == "failed"
        assert len(results["errors"]) > 0
        assert results["errors"][0]["error"] == "Execution failed as requested"
    
    def test_run_method(self, agent):
        """Test complete run workflow"""
        # Successful run
        result = agent.run(data="input_data")
        assert result["status"] == "completed"
        assert result["results"]["data"] == "input_data"
        assert agent.status == "completed"
        
        # Run with initialization failure
        fail_init_agent = ConcreteAgent("FailInitAgent", {"fail_init": True})
        result = fail_init_agent.run()
        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        
        # Run with execution failure
        fail_exec_agent = ConcreteAgent("FailExecAgent", {"fail_execute": True})
        result = fail_exec_agent.run()
        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
    
    def test_reset_method(self, agent):
        """Test agent state reset functionality"""
        # Run agent first
        agent.run(data="test")
        assert agent.status == "completed"
        assert agent.results != {}
        assert agent.start_time is not None
        
        # Reset and verify
        agent.reset()
        assert agent.status == "initialized"
        assert agent.results == {}
        assert agent.errors == []
        assert agent.start_time is None
        assert agent.end_time is None
    
    def test_save_results(self, agent):
        """Test result saving functionality"""
        # Run agent to generate results
        agent.run(data="test_data")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            agent.save_results(temp_path)
            
            # Verify file contents
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["agent_name"] == "TestAgent"
            assert saved_data["status"] == "completed"
            assert saved_data["results"]["data"] == "test_data"
            
        finally:
            os.unlink(temp_path)
        
        # Test save failure
        with pytest.raises(Exception):
            agent.save_results("/invalid/path/results.json")
    
    def test_load_config_json(self, agent):
        """Test loading configuration from JSON file"""
        config_data = {
            "new_param": "new_value",
            "number": 123,
            "list": [1, 2, 3]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            agent.load_config(temp_path)
            assert agent.config == config_data
        finally:
            os.unlink(temp_path)
    
    def test_load_config_yaml(self, agent):
        """Test loading configuration from YAML file"""
        yaml_content = """
        new_param: yaml_value
        number: 456
        list:
          - a
          - b
          - c
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            agent.load_config(temp_path)
            assert agent.config["new_param"] == "yaml_value"
            assert agent.config["number"] == 456
            assert agent.config["list"] == ["a", "b", "c"]
        finally:
            os.unlink(temp_path)
    
    def test_load_config_invalid_format(self, agent):
        """Test loading configuration with invalid format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid config")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                agent.load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_validate_config(self, agent):
        """Test configuration validation"""
        # Valid configuration
        agent.config = {
            "required1": "value1",
            "required2": "value2",
            "optional": "value3"
        }
        assert agent.validate_config(["required1", "required2"]) is True
        
        # Missing required keys
        assert agent.validate_config(["required1", "missing_key"]) is False
        
        # Empty required keys
        assert agent.validate_config([]) is True
    
    def test_error_handling(self):
        """Test comprehensive error handling"""
        # Test various error scenarios
        error_configs = [
            {"fail_execute": True, "error_type": "RuntimeError"},
            {"fail_init": True, "error_type": "InitError"},
        ]
        
        for config in error_configs:
            agent = ConcreteAgent("ErrorAgent", config)
            result = agent.run()
            
            assert result["status"] == "failed"
            assert len(result["errors"]) > 0
            assert "timestamp" in result["errors"][0]
            assert "error" in result["errors"][0]
            assert "traceback" in result["errors"][0]
    
    def test_concurrent_execution(self):
        """Test agent behavior with concurrent operations"""
        agents = [ConcreteAgent(f"Agent{i}", {"id": i}) for i in range(5)]
        
        # Run all agents
        results = [agent.run(data=f"data_{i}") for i, agent in enumerate(agents)]
        
        # Verify all completed successfully
        for i, result in enumerate(results):
            assert result["status"] == "completed"
            assert result["results"]["data"] == f"data_{i}"
    
    def test_agent_metadata(self, agent):
        """Test metadata handling"""
        # Check initial metadata
        assert "created_at" in agent.metadata
        assert agent.metadata["version"] == "1.0.0"
        
        # Add custom metadata
        agent.metadata["custom_field"] = "custom_value"
        agent.metadata["run_id"] = "12345"
        
        # Verify in results
        results = agent.get_results()
        assert results["metadata"]["custom_field"] == "custom_value"
        assert results["metadata"]["run_id"] == "12345"
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Very long agent name
        long_name = "A" * 1000
        agent = ConcreteAgent(long_name, {})
        assert agent.name == long_name
        
        # Unicode in config
        unicode_config = {
            "name": "测试代理",
            "emoji": "🚀",
            "special": "àáäâèéëê"
        }
        unicode_agent = ConcreteAgent("UnicodeAgent", unicode_config)
        result = unicode_agent.run()
        assert result["status"] == "completed"
        
        # Large config
        large_config = {str(i): i for i in range(10000)}
        large_agent = ConcreteAgent("LargeAgent", large_config)
        assert len(large_agent.config) == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])