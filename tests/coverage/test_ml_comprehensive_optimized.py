"""Comprehensive ML module tests for maximum coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import joblib
from pathlib import Path

# Test all ML modules
from src.ml.models.base_model import BaseMLModel
from src.ml.models.random_forest_model import RandomForestModel
from src.ml.models.xgboost_model import XGBoostModel
from src.ml.models.neural_network_model import NeuralNetworkModel
from src.ml.models.lstm_model import LSTMModel
from src.ml.models.ensemble_model import EnsembleModel

from src.ml.features.feature_engineering import FeatureEngineer
from src.ml.features.feature_selector import FeatureSelector

from src.ml.agents.ml_agent import MLAgent
from src.ml.agents.rl_agent import RLAgent
from src.ml.agents.meta_learning_agent import MetaLearningAgent
from src.ml.agents.adaptive_agent import AdaptiveAgent
from src.ml.agents.ensemble_agent import EnsembleAgent
from src.ml.agents.multi_agent_system import MultiAgentSystem

from src.ml.optimization.hyperparameter_tuner import HyperparameterTuner
from src.ml.optimization.portfolio_optimizer import PortfolioOptimizer
from src.ml.optimization.strategy_optimizer import StrategyOptimizer
from src.ml.optimization.walk_forward_optimizer import WalkForwardOptimizer
from src.ml.optimization.genetic_optimizer import GeneticOptimizer
from src.ml.optimization.bayesian_optimizer import BayesianOptimizer

from src.ml.clustering import MarketRegimeDetector
from src.ml.reports import MLReportGenerator
from src.ml.market_regime import RegimeAnalyzer


@pytest.fixture
def sample_features():
    """Create sample feature data."""
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'returns': np.random.randn(1000) * 0.02,
        'volume': np.random.randint(1000000, 5000000, 1000),
        'rsi': np.random.uniform(20, 80, 1000),
        'ma_20': 100 + np.random.randn(1000).cumsum(),
        'ma_50': 100 + np.random.randn(1000).cumsum() * 0.8,
        'volatility': np.random.uniform(0.1, 0.3, 1000),
        'target': (np.random.randn(1000) > 0).astype(int)
    }, index=dates)
    
    return data


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    close = 100
    data = []
    
    for _ in dates:
        open_price = close * (1 + np.random.uniform(-0.02, 0.02))
        high = max(open_price, close) * (1 + np.random.uniform(0, 0.02))
        low = min(open_price, close) * (1 - np.random.uniform(0, 0.02))
        close = open_price * (1 + np.random.uniform(-0.03, 0.03))
        volume = np.random.randint(1000000, 5000000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data, index=dates)


class TestMLModels:
    """Test ML model implementations."""
    
    def test_base_model_interface(self):
        """Test BaseMLModel abstract interface."""
        # Test that BaseMLModel cannot be instantiated
        with pytest.raises(TypeError):
            BaseMLModel()
    
    def test_random_forest_model(self, sample_features):
        """Test RandomForestModel implementation."""
        model = RandomForestModel(
            n_estimators=10,
            max_depth=5,
            random_state=42
        )
        
        # Test fitting
        X = sample_features.drop('target', axis=1)
        y = sample_features['target']
        
        model.fit(X[:800], y[:800])
        assert model.is_fitted
        
        # Test prediction
        predictions = model.predict(X[800:])
        assert len(predictions) == 200
        assert all(p in [0, 1] for p in predictions)
        
        # Test probability prediction
        proba = model.predict_proba(X[800:])
        assert proba.shape == (200, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == len(X.columns)
        
        # Test save/load
        model.save_model('test_rf_model.pkl')
        loaded_model = RandomForestModel()
        loaded_model.load_model('test_rf_model.pkl')
        assert loaded_model.is_fitted
        
        # Cleanup
        Path('test_rf_model.pkl').unlink(missing_ok=True)
    
    def test_xgboost_model(self, sample_features):
        """Test XGBoostModel implementation."""
        model = XGBoostModel(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1
        )
        
        X = sample_features.drop('target', axis=1)
        y = sample_features['target']
        
        # Test with validation set
        model.fit(X[:700], y[:700], eval_set=[(X[700:800], y[700:800])])
        
        # Test predictions
        predictions = model.predict(X[800:])
        assert len(predictions) == 200
        
        # Test early stopping
        model_es = XGBoostModel(
            n_estimators=100,
            early_stopping_rounds=5
        )
        model_es.fit(X[:700], y[:700], eval_set=[(X[700:800], y[700:800])])
        assert model_es.model.n_estimators < 100  # Should stop early
    
    def test_neural_network_model(self, sample_features):
        """Test NeuralNetworkModel implementation."""
        model = NeuralNetworkModel(
            hidden_layers=[32, 16],
            activation='relu',
            dropout_rate=0.2,
            learning_rate=0.001,
            epochs=5,
            batch_size=32
        )
        
        X = sample_features.drop('target', axis=1)
        y = sample_features['target']
        
        # Mock TensorFlow/Keras
        with patch('src.ml.models.neural_network_model.Sequential') as mock_seq:
            mock_model = Mock()
            mock_seq.return_value = mock_model
            mock_model.predict.return_value = np.random.rand(200, 1)
            
            model.fit(X[:800], y[:800])
            predictions = model.predict(X[800:])
            
            assert len(predictions) == 200
            mock_model.fit.assert_called_once()
    
    def test_lstm_model(self, sample_features):
        """Test LSTMModel implementation."""
        model = LSTMModel(
            sequence_length=20,
            lstm_units=[64, 32],
            dropout_rate=0.2,
            epochs=5
        )
        
        X = sample_features.drop('target', axis=1)
        y = sample_features['target']
        
        # Test sequence creation
        X_seq, y_seq = model._create_sequences(X.values, y.values)
        assert X_seq.shape[1] == 20  # sequence length
        assert X_seq.shape[2] == X.shape[1]  # features
        
        # Mock LSTM training
        with patch('src.ml.models.lstm_model.Sequential') as mock_seq:
            mock_model = Mock()
            mock_seq.return_value = mock_model
            mock_model.predict.return_value = np.random.rand(180, 1)
            
            model.fit(X[:800], y[:800])
            predictions = model.predict(X[800:])
            
            assert predictions is not None
    
    def test_ensemble_model(self, sample_features):
        """Test EnsembleModel implementation."""
        # Create base models
        rf_model = RandomForestModel(n_estimators=5)
        xgb_model = XGBoostModel(n_estimators=5)
        
        ensemble = EnsembleModel(
            models=[rf_model, xgb_model],
            voting='soft',
            weights=[0.6, 0.4]
        )
        
        X = sample_features.drop('target', axis=1)
        y = sample_features['target']
        
        # Test fitting
        ensemble.fit(X[:800], y[:800])
        assert all(model.is_fitted for model in ensemble.models)
        
        # Test prediction
        predictions = ensemble.predict(X[800:])
        assert len(predictions) == 200
        
        # Test probability prediction
        proba = ensemble.predict_proba(X[800:])
        assert proba.shape == (200, 2)
        
        # Test hard voting
        ensemble_hard = EnsembleModel(
            models=[rf_model, xgb_model],
            voting='hard'
        )
        ensemble_hard.fit(X[:800], y[:800])
        hard_predictions = ensemble_hard.predict(X[800:])
        assert len(hard_predictions) == 200


class TestFeatureEngineering:
    """Test feature engineering components."""
    
    def test_feature_engineer(self, sample_ohlcv):
        """Test FeatureEngineer class."""
        engineer = FeatureEngineer()
        
        # Test technical indicators
        features_tech = engineer.create_technical_features(
            sample_ohlcv,
            indicators=['rsi', 'macd', 'bollinger', 'atr']
        )
        
        assert 'rsi' in features_tech.columns
        assert 'macd' in features_tech.columns
        assert 'macd_signal' in features_tech.columns
        assert 'bb_upper' in features_tech.columns
        assert 'bb_lower' in features_tech.columns
        assert 'atr' in features_tech.columns
        
        # Test price features
        features_price = engineer.create_price_features(
            sample_ohlcv,
            periods=[5, 10, 20]
        )
        
        assert 'returns_1' in features_price.columns
        assert 'returns_5' in features_price.columns
        assert 'log_returns_1' in features_price.columns
        assert 'price_change_5' in features_price.columns
        
        # Test volume features
        features_vol = engineer.create_volume_features(
            sample_ohlcv,
            periods=[5, 10]
        )
        
        assert 'volume_ratio_5' in features_vol.columns
        assert 'volume_change_1' in features_vol.columns
        assert 'dollar_volume' in features_vol.columns
        
        # Test statistical features
        features_stat = engineer.create_statistical_features(
            sample_ohlcv,
            windows=[10, 20]
        )
        
        assert 'volatility_10' in features_stat.columns
        assert 'skewness_20' in features_stat.columns
        assert 'kurtosis_10' in features_stat.columns
        
        # Test all features combined
        all_features = engineer.create_all_features(sample_ohlcv)
        assert len(all_features.columns) > 20
    
    def test_feature_selector(self, sample_features):
        """Test FeatureSelector class."""
        selector = FeatureSelector()
        
        X = sample_features.drop('target', axis=1)
        y = sample_features['target']
        
        # Test importance-based selection
        selected_importance = selector.select_features(
            X, y,
            method='importance',
            n_features=3
        )
        assert len(selected_importance) == 3
        
        # Test correlation-based selection
        selected_corr = selector.select_features(
            X, y,
            method='correlation',
            threshold=0.1
        )
        assert len(selected_corr) > 0
        
        # Test mutual information selection
        selected_mi = selector.select_features(
            X, y,
            method='mutual_info',
            n_features=4
        )
        assert len(selected_mi) == 4
        
        # Test recursive feature elimination
        selected_rfe = selector.select_features(
            X, y,
            method='rfe',
            n_features=3
        )
        assert len(selected_rfe) == 3


class TestMLAgents:
    """Test ML agent implementations."""
    
    def test_ml_agent(self, sample_features):
        """Test basic MLAgent."""
        model = RandomForestModel(n_estimators=5)
        agent = MLAgent(
            name="Test Agent",
            model=model,
            features=['returns', 'volume', 'rsi']
        )
        
        # Test training
        agent.train(sample_features[:800])
        assert agent.is_trained
        
        # Test prediction
        signal = agent.predict(sample_features.iloc[800])
        assert signal in [-1, 0, 1]
        
        # Test batch prediction
        signals = agent.predict_batch(sample_features[800:850])
        assert len(signals) == 50
        
        # Test performance tracking
        agent.update_performance(1.0, 0.02)
        assert len(agent.performance_history) == 1
        
        # Test save/load
        agent.save('test_agent.pkl')
        loaded_agent = MLAgent.load('test_agent.pkl')
        assert loaded_agent.name == "Test Agent"
        
        # Cleanup
        Path('test_agent.pkl').unlink(missing_ok=True)
    
    def test_rl_agent(self):
        """Test RLAgent implementation."""
        agent = RLAgent(
            name="RL Test",
            state_dim=10,
            action_dim=3,
            learning_rate=0.001,
            gamma=0.95,
            epsilon=0.1
        )
        
        # Test initialization
        assert agent.epsilon == 0.1
        assert agent.gamma == 0.95
        
        # Test action selection
        state = np.random.randn(10)
        action = agent.select_action(state)
        assert action in [0, 1, 2]
        
        # Test learning
        next_state = np.random.randn(10)
        agent.learn(state, action, 0.01, next_state, False)
        
        # Test epsilon decay
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon < initial_epsilon
    
    def test_meta_learning_agent(self, sample_features):
        """Test MetaLearningAgent."""
        base_models = [
            RandomForestModel(n_estimators=5),
            XGBoostModel(n_estimators=5)
        ]
        
        agent = MetaLearningAgent(
            name="Meta Test",
            base_models=base_models,
            meta_features=['returns', 'volatility']
        )
        
        # Test meta-training
        agent.meta_train(sample_features[:800])
        assert agent.is_trained
        
        # Test adaptation
        agent.adapt(sample_features[800:900])
        
        # Test prediction with meta-learning
        signal = agent.predict(sample_features.iloc[900])
        assert signal in [-1, 0, 1]
    
    def test_adaptive_agent(self, sample_features):
        """Test AdaptiveAgent."""
        model = RandomForestModel(n_estimators=5)
        agent = AdaptiveAgent(
            name="Adaptive Test",
            model=model,
            adaptation_window=50,
            retraining_frequency=100
        )
        
        # Test initial training
        agent.train(sample_features[:500])
        
        # Test adaptation
        for i in range(500, 600):
            signal = agent.predict(sample_features.iloc[i])
            agent.update(sample_features.iloc[i], signal)
            
            # Check if retraining occurred
            if i % 100 == 0:
                assert agent.updates_since_retrain == 0
    
    def test_ensemble_agent(self, sample_features):
        """Test EnsembleAgent."""
        agents = [
            MLAgent("Agent1", RandomForestModel(n_estimators=5), ['returns', 'rsi']),
            MLAgent("Agent2", XGBoostModel(n_estimators=5), ['volume', 'volatility'])
        ]
        
        ensemble = EnsembleAgent(
            name="Ensemble Test",
            agents=agents,
            voting_method='weighted',
            weights=[0.6, 0.4]
        )
        
        # Train all agents
        ensemble.train(sample_features[:800])
        
        # Test ensemble prediction
        signal = ensemble.predict(sample_features.iloc[800])
        assert signal in [-1, 0, 1]
        
        # Test confidence calculation
        signal, confidence = ensemble.predict_with_confidence(sample_features.iloc[800])
        assert 0 <= confidence <= 1
    
    def test_multi_agent_system(self, sample_features):
        """Test MultiAgentSystem."""
        agents = [
            MLAgent("Trend", RandomForestModel(n_estimators=5), ['returns', 'ma_20']),
            MLAgent("Momentum", XGBoostModel(n_estimators=5), ['rsi', 'volatility'])
        ]
        
        system = MultiAgentSystem(
            agents=agents,
            coordinator='voting',
            communication_enabled=True
        )
        
        # Test system training
        system.train(sample_features[:800])
        
        # Test system decision
        decision = system.make_decision(sample_features.iloc[800])
        assert 'signal' in decision
        assert 'confidence' in decision
        assert 'agent_signals' in decision
        
        # Test agent communication
        system.enable_communication()
        decision_comm = system.make_decision(sample_features.iloc[801])
        assert 'consensus' in decision_comm


class TestOptimization:
    """Test optimization components."""
    
    def test_hyperparameter_tuner(self, sample_features):
        """Test HyperparameterTuner."""
        param_grid = {
            'n_estimators': [10, 50],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1]
        }
        
        tuner = HyperparameterTuner(
            model_class=XGBoostModel,
            param_grid=param_grid,
            cv_folds=3,
            scoring='accuracy'
        )
        
        X = sample_features.drop('target', axis=1)
        y = sample_features['target']
        
        # Test grid search
        best_params = tuner.search(X[:800], y[:800], method='grid')
        assert all(param in best_params for param in param_grid.keys())
        
        # Test random search
        best_params_random = tuner.search(
            X[:800], y[:800],
            method='random',
            n_iter=5
        )
        assert all(param in best_params_random for param in param_grid.keys())
    
    def test_portfolio_optimizer(self):
        """Test PortfolioOptimizer."""
        returns = pd.DataFrame(
            np.random.randn(100, 5) * 0.01,
            columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5']
        )
        
        optimizer = PortfolioOptimizer()
        
        # Test mean-variance optimization
        weights_mv = optimizer.optimize(
            returns,
            method='mean_variance',
            target_return=0.001
        )
        assert len(weights_mv) == 5
        assert np.isclose(sum(weights_mv), 1.0)
        
        # Test minimum variance
        weights_min_var = optimizer.optimize(
            returns,
            method='min_variance'
        )
        assert len(weights_min_var) == 5
        assert np.isclose(sum(weights_min_var), 1.0)
        
        # Test maximum Sharpe
        weights_sharpe = optimizer.optimize(
            returns,
            method='max_sharpe',
            risk_free_rate=0.02
        )
        assert len(weights_sharpe) == 5
    
    def test_strategy_optimizer(self, sample_ohlcv):
        """Test StrategyOptimizer."""
        param_ranges = {
            'rsi_period': (10, 30),
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 80),
            'position_size': (0.1, 0.5)
        }
        
        optimizer = StrategyOptimizer(
            strategy_class=Mock,
            param_ranges=param_ranges,
            optimization_metric='sharpe_ratio'
        )
        
        # Mock backtest function
        def mock_backtest(params):
            return {
                'sharpe_ratio': np.random.uniform(0, 2),
                'total_return': np.random.uniform(-0.1, 0.3),
                'max_drawdown': np.random.uniform(-0.2, -0.05)
            }
        
        optimizer.backtest_func = mock_backtest
        
        # Test optimization
        best_params = optimizer.optimize(
            sample_ohlcv,
            method='bayesian',
            n_trials=10
        )
        
        assert all(param in best_params for param in param_ranges.keys())
    
    def test_walk_forward_optimizer(self, sample_ohlcv):
        """Test WalkForwardOptimizer."""
        optimizer = WalkForwardOptimizer(
            train_period=252,
            test_period=63,
            step_size=21
        )
        
        # Test window generation
        windows = optimizer.generate_windows(sample_ohlcv)
        assert len(windows) > 0
        
        for window in windows:
            assert len(window['train']) == 252
            assert len(window['test']) == 63
            
        # Test optimization with mock
        def mock_optimize(train_data):
            return {'param1': 0.5, 'param2': 10}
        
        def mock_backtest(test_data, params):
            return {'sharpe_ratio': 1.5, 'total_return': 0.15}
        
        results = optimizer.optimize(
            sample_ohlcv,
            mock_optimize,
            mock_backtest
        )
        
        assert 'results' in results
        assert 'summary' in results
    
    def test_genetic_optimizer(self):
        """Test GeneticOptimizer."""
        def fitness_func(params):
            # Simple quadratic function
            x = params['x']
            y = params['y']
            return -(x - 5)**2 - (y - 3)**2
        
        param_ranges = {
            'x': (0, 10),
            'y': (0, 10)
        }
        
        optimizer = GeneticOptimizer(
            param_ranges=param_ranges,
            population_size=20,
            generations=10,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        best_params = optimizer.optimize(fitness_func)
        
        # Should be close to (5, 3)
        assert 4 < best_params['x'] < 6
        assert 2 < best_params['y'] < 4
    
    def test_bayesian_optimizer(self):
        """Test BayesianOptimizer."""
        def objective(params):
            x = params['x']
            y = params['y']
            return np.sin(x) * np.cos(y) + 0.1 * x
        
        param_ranges = {
            'x': (0, 5),
            'y': (0, 5)
        }
        
        optimizer = BayesianOptimizer(
            param_ranges=param_ranges,
            n_calls=20,
            random_state=42
        )
        
        best_params = optimizer.optimize(objective)
        
        assert 'x' in best_params
        assert 'y' in best_params
        assert 0 <= best_params['x'] <= 5
        assert 0 <= best_params['y'] <= 5


class TestMarketRegime:
    """Test market regime detection."""
    
    def test_market_regime_detector(self, sample_ohlcv):
        """Test MarketRegimeDetector."""
        detector = MarketRegimeDetector(
            n_regimes=3,
            features=['returns', 'volatility', 'volume']
        )
        
        # Test regime detection
        regimes = detector.detect_regimes(sample_ohlcv)
        assert len(regimes) == len(sample_ohlcv)
        assert all(regime in [0, 1, 2] for regime in regimes)
        
        # Test regime statistics
        stats = detector.get_regime_statistics(sample_ohlcv, regimes)
        assert len(stats) == 3
        assert all('mean_return' in stat for stat in stats.values())
        assert all('volatility' in stat for stat in stats.values())
        
        # Test regime transitions
        transitions = detector.get_transition_matrix(regimes)
        assert transitions.shape == (3, 3)
        assert np.allclose(transitions.sum(axis=1), 1.0)
    
    def test_regime_analyzer(self, sample_ohlcv):
        """Test RegimeAnalyzer."""
        analyzer = RegimeAnalyzer()
        
        # Test regime characteristics
        regimes = analyzer.identify_regimes(sample_ohlcv)
        assert 'bull' in regimes
        assert 'bear' in regimes
        assert 'sideways' in regimes
        
        # Test regime-based strategy selection
        current_regime = analyzer.get_current_regime(sample_ohlcv)
        assert current_regime in ['bull', 'bear', 'sideways']
        
        strategy = analyzer.select_strategy(current_regime)
        assert strategy is not None


class TestMLReports:
    """Test ML reporting."""
    
    def test_ml_report_generator(self, sample_features, tmp_path):
        """Test MLReportGenerator."""
        # Create mock model results
        model_results = {
            'model_name': 'RandomForest',
            'accuracy': 0.75,
            'precision': 0.72,
            'recall': 0.78,
            'f1_score': 0.75,
            'confusion_matrix': [[100, 30], [20, 50]],
            'feature_importance': {
                'returns': 0.3,
                'volume': 0.2,
                'rsi': 0.25,
                'volatility': 0.25
            }
        }
        
        generator = MLReportGenerator(output_dir=str(tmp_path))
        
        # Test classification report
        report_path = generator.generate_classification_report(
            model_results,
            sample_features.drop('target', axis=1),
            sample_features['target']
        )
        
        assert Path(report_path).exists()
        
        # Test feature importance plot
        plot_path = generator.plot_feature_importance(
            model_results['feature_importance']
        )
        
        assert Path(plot_path).exists()
        
        # Test learning curves
        train_scores = [0.6, 0.65, 0.7, 0.72, 0.74]
        val_scores = [0.58, 0.62, 0.66, 0.68, 0.70]
        
        curve_path = generator.plot_learning_curves(
            train_scores,
            val_scores
        )
        
        assert Path(curve_path).exists()
        
        # Test comprehensive report
        full_report = generator.generate_full_report(
            model_results,
            training_history={
                'loss': train_scores,
                'val_loss': val_scores
            }
        )
        
        assert Path(full_report).exists()