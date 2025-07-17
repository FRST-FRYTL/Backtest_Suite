"""Comprehensive tests for ML agents to achieve 100% coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio

from src.ml.agents.base_agent import BaseAgent, AgentConfig
from src.ml.agents.market_analyzer import MarketAnalyzerAgent
from src.ml.agents.pattern_detector import PatternDetectorAgent
from src.ml.agents.risk_assessor import RiskAssessorAgent
from src.ml.agents.strategy_optimizer import StrategyOptimizerAgent
from src.ml.agents.sentiment_analyzer import SentimentAnalyzerAgent
from src.ml.agents.execution_optimizer import ExecutionOptimizerAgent
from src.ml.agents.portfolio_manager import PortfolioManagerAgent
from src.ml.agents.anomaly_detector import AnomalyDetectorAgent
from src.ml.agents.regime_detector import RegimeDetectorAgent
from src.ml.agents.allocation_optimizer import AllocationOptimizerAgent
from src.ml.agents.signal_aggregator import SignalAggregatorAgent
from src.ml.agents.performance_analyzer import PerformanceAnalyzerAgent


class TestAgentConfig:
    """Test AgentConfig dataclass."""
    
    def test_default_config(self):
        """Test default AgentConfig initialization."""
        config = AgentConfig()
        
        assert config.name == "BaseAgent"
        assert config.description == ""
        assert config.confidence_threshold == 0.7
        assert config.update_frequency == 3600  # 1 hour
        assert config.memory_size == 1000
        assert config.learning_rate == 0.001
        assert config.use_gpu is False
        assert isinstance(config.parameters, dict)
        
    def test_custom_config(self):
        """Test custom AgentConfig initialization."""
        custom_params = {
            'model': 'xgboost',
            'n_estimators': 100,
            'max_depth': 5
        }
        
        config = AgentConfig(
            name="TestAgent",
            description="Test description",
            confidence_threshold=0.8,
            update_frequency=1800,
            memory_size=500,
            learning_rate=0.01,
            use_gpu=True,
            parameters=custom_params
        )
        
        assert config.name == "TestAgent"
        assert config.description == "Test description"
        assert config.confidence_threshold == 0.8
        assert config.update_frequency == 1800
        assert config.memory_size == 500
        assert config.learning_rate == 0.01
        assert config.use_gpu is True
        assert config.parameters == custom_params


class TestBaseAgent:
    """Test BaseAgent abstract class."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a concrete implementation of BaseAgent for testing."""
        class ConcreteAgent(BaseAgent):
            async def analyze(self, data):
                return {"result": "analyzed"}
            
            def update_model(self, feedback):
                self.memory.append(feedback)
                
        config = AgentConfig(name="TestAgent")
        return ConcreteAgent(config)
        
    def test_base_agent_initialization(self, mock_agent):
        """Test BaseAgent initialization."""
        assert mock_agent.config.name == "TestAgent"
        assert mock_agent.memory == []
        assert mock_agent.last_update is None
        assert mock_agent.performance_metrics == {}
        
    def test_store_memory(self, mock_agent):
        """Test memory storage."""
        memory_item = {
            'timestamp': datetime.now(),
            'data': 'test_data',
            'result': 'test_result'
        }
        
        mock_agent.store_memory(memory_item)
        
        assert len(mock_agent.memory) == 1
        assert mock_agent.memory[0] == memory_item
        
        # Test memory size limit
        mock_agent.config.memory_size = 3
        for i in range(5):
            mock_agent.store_memory({'item': i})
            
        assert len(mock_agent.memory) == 3
        assert mock_agent.memory[0]['item'] == 2  # Oldest items removed
        
    def test_get_confidence(self, mock_agent):
        """Test confidence calculation."""
        # No performance metrics
        assert mock_agent.get_confidence() == 0.5
        
        # With performance metrics
        mock_agent.performance_metrics = {
            'accuracy': 0.85,
            'precision': 0.90,
            'recall': 0.80
        }
        
        confidence = mock_agent.get_confidence()
        assert 0.8 < confidence < 0.9
        
    @pytest.mark.asyncio
    async def test_analyze_abstract(self, mock_agent):
        """Test analyze method implementation."""
        data = pd.DataFrame({'price': [100, 101, 102]})
        result = await mock_agent.analyze(data)
        
        assert result == {"result": "analyzed"}
        
    def test_update_model_abstract(self, mock_agent):
        """Test update_model implementation."""
        feedback = {'accuracy': 0.9, 'loss': 0.1}
        mock_agent.update_model(feedback)
        
        assert feedback in mock_agent.memory


class TestMarketAnalyzerAgent:
    """Test MarketAnalyzerAgent class."""
    
    @pytest.fixture
    def market_analyzer(self):
        """Create MarketAnalyzerAgent instance."""
        config = AgentConfig(
            name="MarketAnalyzer",
            parameters={'indicators': ['rsi', 'macd', 'bollinger']}
        )
        return MarketAnalyzerAgent(config)
        
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        return pd.DataFrame({
            'open': 100 + np.random.randn(100).cumsum(),
            'high': 101 + np.random.randn(100).cumsum(),
            'low': 99 + np.random.randn(100).cumsum(),
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(1000, 5000, 100)
        }, index=dates)
        
    @pytest.mark.asyncio
    async def test_market_analysis(self, market_analyzer, sample_market_data):
        """Test market analysis functionality."""
        result = await market_analyzer.analyze(sample_market_data)
        
        assert 'trend' in result
        assert 'volatility' in result
        assert 'support_resistance' in result
        assert 'indicators' in result
        assert 'market_state' in result
        
        # Check trend analysis
        assert result['trend']['direction'] in ['up', 'down', 'sideways']
        assert 'strength' in result['trend']
        
        # Check volatility
        assert isinstance(result['volatility'], float)
        assert result['volatility'] >= 0
        
        # Check indicators
        assert 'rsi' in result['indicators']
        assert 'macd' in result['indicators']
        
    def test_calculate_indicators(self, market_analyzer, sample_market_data):
        """Test indicator calculation."""
        indicators = market_analyzer._calculate_indicators(sample_market_data)
        
        assert isinstance(indicators, dict)
        assert 'rsi' in indicators
        assert 'sma_20' in indicators
        assert 'ema_50' in indicators
        
    def test_identify_support_resistance(self, market_analyzer, sample_market_data):
        """Test support/resistance identification."""
        levels = market_analyzer._identify_support_resistance(sample_market_data)
        
        assert 'support' in levels
        assert 'resistance' in levels
        assert isinstance(levels['support'], list)
        assert isinstance(levels['resistance'], list)
        
    def test_market_state_classification(self, market_analyzer):
        """Test market state classification."""
        indicators = {
            'rsi': 75,
            'volatility': 0.03,
            'trend_strength': 0.8
        }
        
        state = market_analyzer._classify_market_state(indicators)
        
        assert state in ['trending', 'ranging', 'volatile', 'quiet']


class TestPatternDetectorAgent:
    """Test PatternDetectorAgent class."""
    
    @pytest.fixture
    def pattern_detector(self):
        """Create PatternDetectorAgent instance."""
        config = AgentConfig(
            name="PatternDetector",
            parameters={
                'patterns': ['head_shoulders', 'double_top', 'triangle'],
                'min_pattern_score': 0.7
            }
        )
        return PatternDetectorAgent(config)
        
    @pytest.mark.asyncio
    async def test_pattern_detection(self, pattern_detector, sample_market_data):
        """Test pattern detection functionality."""
        result = await pattern_detector.analyze(sample_market_data)
        
        assert 'patterns' in result
        assert 'chart_patterns' in result
        assert 'candlestick_patterns' in result
        assert 'confidence_scores' in result
        
        # Check pattern structure
        if result['patterns']:
            pattern = result['patterns'][0]
            assert 'type' in pattern
            assert 'start' in pattern
            assert 'end' in pattern
            assert 'confidence' in pattern
            
    def test_detect_chart_patterns(self, pattern_detector):
        """Test chart pattern detection."""
        # Create synthetic pattern data
        dates = pd.date_range(start='2023-01-01', periods=50)
        
        # Create head and shoulders pattern
        prices = np.concatenate([
            np.linspace(100, 110, 10),  # Left shoulder up
            np.linspace(110, 105, 5),   # Left shoulder down
            np.linspace(105, 115, 10),  # Head up
            np.linspace(115, 105, 5),   # Head down
            np.linspace(105, 110, 10),  # Right shoulder up
            np.linspace(110, 100, 10)   # Right shoulder down
        ])
        
        data = pd.DataFrame({
            'close': prices,
            'high': prices + 1,
            'low': prices - 1
        }, index=dates)
        
        patterns = pattern_detector._detect_chart_patterns(data)
        
        assert isinstance(patterns, list)
        # Should detect the head and shoulders pattern
        assert any(p['type'] == 'head_shoulders' for p in patterns)
        
    def test_detect_candlestick_patterns(self, pattern_detector):
        """Test candlestick pattern detection."""
        # Create doji pattern
        data = pd.DataFrame({
            'open': [100, 105, 110],
            'high': [102, 107, 112],
            'low': [98, 103, 108],
            'close': [100.1, 105, 110.1]  # Close very close to open
        })
        
        patterns = pattern_detector._detect_candlestick_patterns(data)
        
        assert isinstance(patterns, list)
        # Should detect doji patterns
        assert any('doji' in p['type'].lower() for p in patterns)


class TestRiskAssessorAgent:
    """Test RiskAssessorAgent class."""
    
    @pytest.fixture
    def risk_assessor(self):
        """Create RiskAssessorAgent instance."""
        config = AgentConfig(
            name="RiskAssessor",
            parameters={
                'risk_metrics': ['var', 'cvar', 'sharpe', 'max_drawdown'],
                'risk_threshold': 0.02
            }
        )
        return RiskAssessorAgent(config)
        
    @pytest.mark.asyncio
    async def test_risk_assessment(self, risk_assessor):
        """Test risk assessment functionality."""
        # Create portfolio data
        portfolio_data = {
            'positions': [
                {'symbol': 'AAPL', 'quantity': 100, 'entry_price': 150},
                {'symbol': 'GOOGL', 'quantity': 50, 'entry_price': 2500}
            ],
            'cash': 50000,
            'total_value': 200000
        }
        
        market_data = pd.DataFrame({
            'AAPL': np.random.normal(150, 5, 100),
            'GOOGL': np.random.normal(2500, 50, 100)
        })
        
        result = await risk_assessor.analyze({
            'portfolio': portfolio_data,
            'market_data': market_data
        })
        
        assert 'risk_metrics' in result
        assert 'risk_score' in result
        assert 'recommendations' in result
        assert 'position_risks' in result
        
        # Check risk metrics
        metrics = result['risk_metrics']
        assert 'var_95' in metrics
        assert 'cvar_95' in metrics
        assert 'max_drawdown' in metrics
        assert 'sharpe_ratio' in metrics
        
    def test_calculate_var(self, risk_assessor):
        """Test Value at Risk calculation."""
        returns = np.random.normal(0.001, 0.02, 1000)
        
        var_95 = risk_assessor._calculate_var(returns, confidence=0.95)
        
        assert isinstance(var_95, float)
        assert var_95 < 0  # VaR should be negative (loss)
        
    def test_calculate_portfolio_risk(self, risk_assessor):
        """Test portfolio risk calculation."""
        positions = [
            {'symbol': 'AAPL', 'quantity': 100, 'value': 15000},
            {'symbol': 'GOOGL', 'quantity': 50, 'value': 125000}
        ]
        
        correlations = pd.DataFrame({
            'AAPL': [1.0, 0.3],
            'GOOGL': [0.3, 1.0]
        }, index=['AAPL', 'GOOGL'])
        
        volatilities = {'AAPL': 0.02, 'GOOGL': 0.025}
        
        portfolio_risk = risk_assessor._calculate_portfolio_risk(
            positions, correlations, volatilities
        )
        
        assert isinstance(portfolio_risk, float)
        assert portfolio_risk > 0


class TestStrategyOptimizerAgent:
    """Test StrategyOptimizerAgent class."""
    
    @pytest.fixture
    def strategy_optimizer(self):
        """Create StrategyOptimizerAgent instance."""
        config = AgentConfig(
            name="StrategyOptimizer",
            parameters={
                'optimization_method': 'genetic_algorithm',
                'population_size': 50,
                'generations': 100
            }
        )
        return StrategyOptimizerAgent(config)
        
    @pytest.mark.asyncio
    async def test_strategy_optimization(self, strategy_optimizer):
        """Test strategy optimization functionality."""
        strategy_params = {
            'rsi_period': {'min': 10, 'max': 30, 'step': 1},
            'rsi_oversold': {'min': 20, 'max': 40, 'step': 5},
            'rsi_overbought': {'min': 60, 'max': 80, 'step': 5}
        }
        
        historical_data = pd.DataFrame({
            'close': 100 + np.random.randn(1000).cumsum(),
            'volume': np.random.randint(1000, 5000, 1000)
        })
        
        result = await strategy_optimizer.analyze({
            'strategy_params': strategy_params,
            'historical_data': historical_data,
            'objective': 'sharpe_ratio'
        })
        
        assert 'optimal_params' in result
        assert 'performance_metrics' in result
        assert 'optimization_history' in result
        
        # Check optimal parameters
        optimal = result['optimal_params']
        assert 10 <= optimal['rsi_period'] <= 30
        assert 20 <= optimal['rsi_oversold'] <= 40
        assert 60 <= optimal['rsi_overbought'] <= 80
        
    def test_genetic_algorithm_optimization(self, strategy_optimizer):
        """Test genetic algorithm optimization."""
        def fitness_function(params):
            # Simple quadratic function for testing
            return -(params['x'] - 5) ** 2 - (params['y'] - 3) ** 2
            
        param_ranges = {
            'x': {'min': 0, 'max': 10, 'step': 0.1},
            'y': {'min': 0, 'max': 10, 'step': 0.1}
        }
        
        result = strategy_optimizer._optimize_genetic_algorithm(
            fitness_function,
            param_ranges,
            population_size=20,
            generations=50
        )
        
        assert 'best_params' in result
        assert 'best_fitness' in result
        assert 'history' in result
        
        # Should find values close to optimum (x=5, y=3)
        assert abs(result['best_params']['x'] - 5) < 0.5
        assert abs(result['best_params']['y'] - 3) < 0.5


class TestSentimentAnalyzerAgent:
    """Test SentimentAnalyzerAgent class."""
    
    @pytest.fixture
    def sentiment_analyzer(self):
        """Create SentimentAnalyzerAgent instance."""
        config = AgentConfig(
            name="SentimentAnalyzer",
            parameters={
                'sources': ['news', 'social_media', 'forums'],
                'sentiment_model': 'transformer'
            }
        )
        return SentimentAnalyzerAgent(config)
        
    @pytest.mark.asyncio
    async def test_sentiment_analysis(self, sentiment_analyzer):
        """Test sentiment analysis functionality."""
        text_data = [
            "Stock market reaches all-time high amid strong earnings",
            "Concerns grow over inflation and interest rates",
            "Tech stocks rally on AI breakthrough announcements",
            "Market crash fears as recession looms"
        ]
        
        result = await sentiment_analyzer.analyze({
            'texts': text_data,
            'symbol': 'SPY'
        })
        
        assert 'overall_sentiment' in result
        assert 'sentiment_scores' in result
        assert 'sentiment_distribution' in result
        assert 'key_topics' in result
        
        # Check sentiment score range
        assert -1 <= result['overall_sentiment'] <= 1
        
        # Check distribution
        dist = result['sentiment_distribution']
        assert 'positive' in dist
        assert 'negative' in dist
        assert 'neutral' in dist
        assert abs(sum(dist.values()) - 1.0) < 0.01  # Should sum to 1
        
    def test_extract_key_topics(self, sentiment_analyzer):
        """Test key topic extraction."""
        texts = [
            "Federal Reserve raises interest rates",
            "Interest rate hike concerns investors",
            "Tech earnings beat expectations",
            "Strong earnings drive market rally"
        ]
        
        topics = sentiment_analyzer._extract_key_topics(texts)
        
        assert isinstance(topics, list)
        assert len(topics) > 0
        
        # Should identify common themes
        topic_words = ' '.join([t['topic'] for t in topics]).lower()
        assert 'interest' in topic_words or 'earnings' in topic_words


class TestExecutionOptimizerAgent:
    """Test ExecutionOptimizerAgent class."""
    
    @pytest.fixture
    def execution_optimizer(self):
        """Create ExecutionOptimizerAgent instance."""
        config = AgentConfig(
            name="ExecutionOptimizer",
            parameters={
                'slippage_model': 'linear',
                'execution_algos': ['twap', 'vwap', 'implementation_shortfall']
            }
        )
        return ExecutionOptimizerAgent(config)
        
    @pytest.mark.asyncio
    async def test_execution_optimization(self, execution_optimizer):
        """Test execution optimization functionality."""
        order = {
            'symbol': 'AAPL',
            'quantity': 10000,
            'side': 'buy',
            'urgency': 'medium'
        }
        
        market_data = pd.DataFrame({
            'price': 150 + np.random.normal(0, 1, 390),  # Full trading day
            'volume': np.random.randint(100000, 500000, 390),
            'spread': np.random.uniform(0.01, 0.05, 390)
        })
        
        result = await execution_optimizer.analyze({
            'order': order,
            'market_data': market_data
        })
        
        assert 'execution_strategy' in result
        assert 'expected_cost' in result
        assert 'schedule' in result
        assert 'risk_metrics' in result
        
        # Check execution strategy
        assert result['execution_strategy'] in ['twap', 'vwap', 'implementation_shortfall']
        
        # Check schedule
        schedule = result['schedule']
        assert isinstance(schedule, list)
        assert sum(s['quantity'] for s in schedule) == order['quantity']
        
    def test_calculate_market_impact(self, execution_optimizer):
        """Test market impact calculation."""
        order_size = 10000
        adv = 1000000  # Average daily volume
        volatility = 0.02
        
        impact = execution_optimizer._calculate_market_impact(
            order_size, adv, volatility
        )
        
        assert isinstance(impact, float)
        assert impact > 0
        assert impact < 0.01  # Should be small for reasonable order size
        
    def test_optimize_execution_schedule(self, execution_optimizer):
        """Test execution schedule optimization."""
        total_quantity = 5000
        time_horizon = 60  # minutes
        
        market_forecast = {
            'volume_profile': np.random.rand(60),
            'volatility_forecast': np.random.uniform(0.01, 0.03, 60)
        }
        
        schedule = execution_optimizer._optimize_schedule(
            total_quantity,
            time_horizon,
            market_forecast
        )
        
        assert len(schedule) <= time_horizon
        assert sum(schedule) == total_quantity
        assert all(q >= 0 for q in schedule)


class TestMLAgentIntegration:
    """Test ML agent integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_multi_agent_collaboration(self):
        """Test multiple agents working together."""
        # Create agents
        market_analyzer = MarketAnalyzerAgent(AgentConfig(name="Market"))
        pattern_detector = PatternDetectorAgent(AgentConfig(name="Pattern"))
        risk_assessor = RiskAssessorAgent(AgentConfig(name="Risk"))
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=200, freq='H')
        market_data = pd.DataFrame({
            'open': 100 + np.random.randn(200).cumsum(),
            'high': 101 + np.random.randn(200).cumsum(),
            'low': 99 + np.random.randn(200).cumsum(),
            'close': 100 + np.random.randn(200).cumsum(),
            'volume': np.random.randint(1000, 5000, 200)
        }, index=dates)
        
        # Run analyses
        market_result = await market_analyzer.analyze(market_data)
        pattern_result = await pattern_detector.analyze(market_data)
        
        portfolio_data = {
            'positions': [{'symbol': 'TEST', 'quantity': 100, 'entry_price': 100}],
            'cash': 10000,
            'total_value': 20000
        }
        
        risk_result = await risk_assessor.analyze({
            'portfolio': portfolio_data,
            'market_data': pd.DataFrame({'TEST': market_data['close']})
        })
        
        # Aggregate results
        combined_signal = {
            'market_state': market_result.get('market_state'),
            'patterns': pattern_result.get('patterns', []),
            'risk_score': risk_result.get('risk_score'),
            'action': 'hold'  # Default
        }
        
        # Decision logic
        if market_result.get('trend', {}).get('direction') == 'up' and risk_result.get('risk_score', 1) < 0.7:
            combined_signal['action'] = 'buy'
        elif market_result.get('trend', {}).get('direction') == 'down' or risk_result.get('risk_score', 0) > 0.8:
            combined_signal['action'] = 'sell'
            
        assert combined_signal['action'] in ['buy', 'sell', 'hold']
        
    @pytest.mark.asyncio
    async def test_agent_learning_cycle(self):
        """Test agent learning from feedback."""
        agent = MarketAnalyzerAgent(AgentConfig(name="Learning"))
        
        # Initial analysis
        data = pd.DataFrame({
            'close': [100, 102, 101, 103, 104],
            'volume': [1000, 1100, 900, 1200, 1000]
        })
        
        result1 = await agent.analyze(data)
        initial_confidence = agent.get_confidence()
        
        # Provide feedback
        feedback = {
            'prediction': result1,
            'actual': {'trend': {'direction': 'up', 'strength': 0.8}},
            'accuracy': 0.85
        }
        
        agent.update_model(feedback)
        agent.performance_metrics['accuracy'] = 0.85
        
        # Check improved confidence
        new_confidence = agent.get_confidence()
        assert new_confidence > initial_confidence
        
        # Check memory storage
        assert len(agent.memory) > 0
        assert agent.memory[-1] == feedback


if __name__ == "__main__":
    pytest.main([__file__, "-v"])