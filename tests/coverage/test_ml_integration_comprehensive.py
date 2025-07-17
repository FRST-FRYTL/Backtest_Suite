"""Comprehensive tests for ML integration to achieve 100% coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.backtesting.ml_integration import (
    MLBacktestConfig, MLBacktestEngine, MLSignalGenerator,
    MLPerformanceTracker, WalkForwardAnalyzer
)
from src.backtesting.events import SignalEvent, EventType
from src.strategies.ml_strategy import MLStrategy, MLSignal
from src.ml.features.feature_engineering import FeatureEngineer


class TestMLBacktestConfig:
    """Test MLBacktestConfig dataclass."""
    
    def test_default_config(self):
        """Test default MLBacktestConfig initialization."""
        config = MLBacktestConfig()
        
        assert config.use_walk_forward is True
        assert config.walk_forward_window == 252
        assert config.retrain_frequency == 63
        assert config.validation_split == 0.2
        assert config.min_training_samples == 500
        assert config.feature_selection is True
        assert config.feature_importance_threshold == 0.01
        assert config.ensemble_voting == 'soft'
        assert config.risk_parity is True
        assert config.max_correlation == 0.95
        
    def test_custom_config(self):
        """Test custom MLBacktestConfig initialization."""
        config = MLBacktestConfig(
            use_walk_forward=False,
            walk_forward_window=500,
            retrain_frequency=21,
            validation_split=0.3,
            min_training_samples=1000,
            feature_selection=False,
            feature_importance_threshold=0.05,
            ensemble_voting='hard',
            risk_parity=False,
            max_correlation=0.9
        )
        
        assert config.use_walk_forward is False
        assert config.walk_forward_window == 500
        assert config.retrain_frequency == 21
        assert config.validation_split == 0.3
        assert config.min_training_samples == 1000
        assert config.feature_selection is False
        assert config.feature_importance_threshold == 0.05
        assert config.ensemble_voting == 'hard'
        assert config.risk_parity is False
        assert config.max_correlation == 0.9


class TestMLBacktestEngine:
    """Test MLBacktestEngine class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic market data
        returns = np.random.normal(0.0005, 0.02, len(dates))
        price = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': price * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
            'high': price * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': price * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'close': price,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Ensure high/low are correct
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
        
    @pytest.fixture
    def ml_strategy(self):
        """Create mock ML strategy."""
        strategy = Mock(spec=MLStrategy)
        strategy.name = "Test ML Strategy"
        strategy.models = {
            'direction': Mock(),
            'volatility': Mock(),
            'regime': Mock()
        }
        strategy.feature_columns = ['close', 'volume', 'rsi', 'sma_20']
        strategy.min_confidence = 0.6
        strategy.max_positions = 5
        strategy.risk_per_trade = 0.02
        return strategy
        
    def test_engine_initialization(self):
        """Test MLBacktestEngine initialization."""
        # Default initialization
        engine = MLBacktestEngine()
        
        assert isinstance(engine.ml_config, MLBacktestConfig)
        assert isinstance(engine.feature_engineer, FeatureEngineer)
        assert engine.ml_predictions == []
        assert engine.feature_importance_history == []
        assert engine.model_performance_history == []
        assert engine.regime_history == []
        assert engine.training_windows == []
        assert engine.out_of_sample_periods == []
        
        # Custom configuration
        custom_config = MLBacktestConfig(use_walk_forward=False)
        engine = MLBacktestEngine(ml_config=custom_config)
        
        assert engine.ml_config.use_walk_forward is False
        
    def test_prepare_ml_data(self, sample_data):
        """Test ML data preparation."""
        engine = MLBacktestEngine()
        
        # Mock feature engineer
        engine.feature_engineer.create_features = Mock(return_value=sample_data.copy())
        
        prepared_data = engine._prepare_ml_data(sample_data)
        
        assert engine.feature_engineer.create_features.called
        assert isinstance(prepared_data, pd.DataFrame)
        assert len(prepared_data) == len(sample_data)
        
    def test_run_ml_backtest_walk_forward(self, sample_data, ml_strategy):
        """Test ML backtest with walk-forward analysis."""
        engine = MLBacktestEngine()
        
        # Mock internal methods
        engine._prepare_ml_data = Mock(return_value=sample_data)
        engine._run_walk_forward_backtest = Mock(return_value={'returns': 0.15})
        
        results = engine.run_ml_backtest(
            data=sample_data,
            ml_strategy=ml_strategy,
            progress_bar=False
        )
        
        assert engine._prepare_ml_data.called
        assert engine._run_walk_forward_backtest.called
        assert 'returns' in results
        
    def test_run_ml_backtest_standard(self, sample_data, ml_strategy):
        """Test ML backtest without walk-forward."""
        config = MLBacktestConfig(use_walk_forward=False)
        engine = MLBacktestEngine(ml_config=config)
        
        # Mock internal methods
        engine._prepare_ml_data = Mock(return_value=sample_data)
        engine._run_standard_ml_backtest = Mock(return_value={'returns': 0.12})
        
        results = engine.run_ml_backtest(
            data=sample_data,
            ml_strategy=ml_strategy
        )
        
        assert engine._run_standard_ml_backtest.called
        assert not engine._run_walk_forward_backtest.called
        
    def test_walk_forward_windows(self, sample_data):
        """Test walk-forward window generation."""
        engine = MLBacktestEngine()
        
        windows = engine._generate_walk_forward_windows(
            data=sample_data,
            window_size=252,
            step_size=63
        )
        
        assert isinstance(windows, list)
        assert len(windows) > 0
        
        # Check window properties
        for train_start, train_end, test_start, test_end in windows:
            assert train_start < train_end
            assert test_start == train_end
            assert test_start < test_end
            assert (train_end - train_start).days >= 252
            
    def test_train_ml_models(self, sample_data, ml_strategy):
        """Test ML model training."""
        engine = MLBacktestEngine()
        
        # Mock feature creation
        features = sample_data.copy()
        features['rsi'] = 50 + np.random.randn(len(sample_data)) * 10
        features['sma_20'] = features['close'].rolling(20).mean()
        
        # Mock model training
        for model_name, model in ml_strategy.models.items():
            model.fit = Mock()
            model.predict_proba = Mock(return_value=np.random.rand(100, 2))
            
        engine._train_ml_models(
            ml_strategy=ml_strategy,
            train_data=features.iloc[:400],
            validation_data=features.iloc[400:500]
        )
        
        # Check all models were trained
        for model in ml_strategy.models.values():
            assert model.fit.called
            
    def test_generate_ml_signals(self, sample_data, ml_strategy):
        """Test ML signal generation."""
        engine = MLBacktestEngine()
        
        # Prepare test data
        test_data = sample_data.iloc[-100:]
        test_data['rsi'] = 50 + np.random.randn(len(test_data)) * 10
        test_data['sma_20'] = test_data['close'].rolling(20).mean()
        
        # Mock model predictions
        ml_strategy.generate_signals = Mock(return_value=[
            MLSignal(
                timestamp=test_data.index[50],
                symbol='TEST',
                direction='BUY',
                confidence=0.75,
                predicted_return=0.02,
                model_name='ensemble',
                features={'rsi': 30, 'sma_20': 95}
            )
        ])
        
        signals = engine._generate_ml_signals(
            ml_strategy=ml_strategy,
            data=test_data
        )
        
        assert isinstance(signals, list)
        assert len(signals) > 0
        assert isinstance(signals[0], MLSignal)
        
    def test_calculate_ml_position_size(self, ml_strategy):
        """Test ML-based position sizing."""
        engine = MLBacktestEngine()
        
        # Test confidence-based sizing
        signal = MLSignal(
            timestamp=datetime.now(),
            symbol='TEST',
            direction='BUY',
            confidence=0.8,
            predicted_return=0.03,
            model_name='test'
        )
        
        size = engine._calculate_ml_position_size(
            signal=signal,
            portfolio_value=100000,
            current_price=100,
            ml_strategy=ml_strategy
        )
        
        assert isinstance(size, int)
        assert size > 0
        assert size <= (100000 * ml_strategy.risk_per_trade) / 100
        
    def test_update_ml_metrics(self, sample_data):
        """Test ML metrics updating."""
        engine = MLBacktestEngine()
        
        # Create mock prediction
        prediction = {
            'timestamp': datetime.now(),
            'predicted_direction': 'BUY',
            'actual_direction': 'BUY',
            'confidence': 0.75,
            'predicted_return': 0.02,
            'actual_return': 0.025,
            'model': 'ensemble'
        }
        
        engine._update_ml_metrics(prediction)
        
        assert len(engine.ml_predictions) == 1
        assert engine.ml_predictions[0] == prediction
        
    def test_calculate_feature_importance(self, ml_strategy):
        """Test feature importance calculation."""
        engine = MLBacktestEngine()
        
        # Mock models with feature importance
        for model_name, model in ml_strategy.models.items():
            if hasattr(model, 'feature_importances_'):
                model.feature_importances_ = np.random.rand(4)
                
        importance = engine._calculate_feature_importance(ml_strategy)
        
        assert isinstance(importance, dict)
        assert len(importance) == len(ml_strategy.feature_columns)
        
    def test_get_ml_performance_metrics(self):
        """Test ML performance metrics calculation."""
        engine = MLBacktestEngine()
        
        # Add mock predictions
        for i in range(100):
            correct = np.random.rand() > 0.4  # 60% accuracy
            engine.ml_predictions.append({
                'timestamp': datetime.now() + timedelta(days=i),
                'predicted_direction': 'BUY',
                'actual_direction': 'BUY' if correct else 'SELL',
                'confidence': np.random.uniform(0.5, 0.9),
                'predicted_return': np.random.uniform(-0.05, 0.05),
                'actual_return': np.random.uniform(-0.05, 0.05),
                'model': 'ensemble'
            })
        
        metrics = engine.get_ml_performance_metrics()
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'sharpe_ml' in metrics
        assert 'confidence_correlation' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1


class TestMLSignalGenerator:
    """Test MLSignalGenerator class."""
    
    def test_signal_generator_initialization(self):
        """Test MLSignalGenerator initialization."""
        generator = MLSignalGenerator(
            min_confidence=0.6,
            use_ensemble=True,
            signal_threshold=0.7
        )
        
        assert generator.min_confidence == 0.6
        assert generator.use_ensemble is True
        assert generator.signal_threshold == 0.7
        assert generator.signal_history == []
        
    def test_generate_ensemble_signal(self):
        """Test ensemble signal generation."""
        generator = MLSignalGenerator(use_ensemble=True)
        
        # Mock multiple model predictions
        predictions = {
            'model1': {'direction': 'BUY', 'confidence': 0.8, 'return': 0.02},
            'model2': {'direction': 'BUY', 'confidence': 0.7, 'return': 0.015},
            'model3': {'direction': 'SELL', 'confidence': 0.6, 'return': -0.01}
        }
        
        signal = generator.generate_ensemble_signal(
            predictions=predictions,
            timestamp=datetime.now(),
            symbol='TEST'
        )
        
        assert isinstance(signal, MLSignal)
        assert signal.direction == 'BUY'  # Majority vote
        assert signal.confidence > 0
        assert signal.model_name == 'ensemble'
        
    def test_filter_signals_by_confidence(self):
        """Test signal filtering by confidence."""
        generator = MLSignalGenerator(min_confidence=0.7)
        
        signals = [
            MLSignal(datetime.now(), 'TEST', 'BUY', 0.8, 0.02, 'model1'),
            MLSignal(datetime.now(), 'TEST', 'SELL', 0.6, -0.01, 'model2'),
            MLSignal(datetime.now(), 'TEST', 'BUY', 0.75, 0.015, 'model3')
        ]
        
        filtered = generator.filter_signals(signals)
        
        assert len(filtered) == 2
        assert all(s.confidence >= 0.7 for s in filtered)
        
    def test_combine_signals(self):
        """Test signal combination logic."""
        generator = MLSignalGenerator()
        
        signals = [
            MLSignal(datetime.now(), 'AAPL', 'BUY', 0.8, 0.02, 'model1'),
            MLSignal(datetime.now(), 'AAPL', 'BUY', 0.7, 0.025, 'model2'),
            MLSignal(datetime.now(), 'GOOGL', 'SELL', 0.75, -0.015, 'model3')
        ]
        
        combined = generator.combine_signals_by_symbol(signals)
        
        assert len(combined) == 2  # Two symbols
        assert 'AAPL' in combined
        assert 'GOOGL' in combined
        
        # AAPL should have averaged confidence and return
        aapl_signal = combined['AAPL']
        assert aapl_signal.confidence == 0.75  # Average of 0.8 and 0.7
        assert aapl_signal.predicted_return == 0.0225  # Average


class TestMLPerformanceTracker:
    """Test MLPerformanceTracker class."""
    
    def test_tracker_initialization(self):
        """Test MLPerformanceTracker initialization."""
        tracker = MLPerformanceTracker()
        
        assert tracker.predictions == []
        assert tracker.model_metrics == {}
        assert tracker.feature_importance_history == []
        assert tracker.regime_accuracy == {}
        
    def test_add_prediction(self):
        """Test adding predictions."""
        tracker = MLPerformanceTracker()
        
        prediction = {
            'timestamp': datetime.now(),
            'model': 'test_model',
            'predicted': 'BUY',
            'actual': 'BUY',
            'confidence': 0.8,
            'features': {'rsi': 30, 'volume': 1000000}
        }
        
        tracker.add_prediction(prediction)
        
        assert len(tracker.predictions) == 1
        assert tracker.predictions[0] == prediction
        
    def test_calculate_model_metrics(self):
        """Test model metrics calculation."""
        tracker = MLPerformanceTracker()
        
        # Add predictions for different models
        for i in range(50):
            for model in ['model1', 'model2']:
                correct = np.random.rand() > 0.3  # 70% accuracy
                tracker.add_prediction({
                    'timestamp': datetime.now() + timedelta(hours=i),
                    'model': model,
                    'predicted': 'BUY',
                    'actual': 'BUY' if correct else 'SELL',
                    'confidence': np.random.uniform(0.5, 0.9),
                    'predicted_return': np.random.uniform(-0.03, 0.03),
                    'actual_return': np.random.uniform(-0.03, 0.03)
                })
        
        metrics = tracker.calculate_model_metrics()
        
        assert 'model1' in metrics
        assert 'model2' in metrics
        
        for model, model_metrics in metrics.items():
            assert 'accuracy' in model_metrics
            assert 'total_predictions' in model_metrics
            assert 'avg_confidence' in model_metrics
            assert 0 <= model_metrics['accuracy'] <= 1
            
    def test_get_feature_importance_trends(self):
        """Test feature importance trend tracking."""
        tracker = MLPerformanceTracker()
        
        # Add feature importance over time
        features = ['rsi', 'volume', 'sma_20', 'bbands']
        
        for i in range(10):
            importance = {
                feat: np.random.rand() for feat in features
            }
            tracker.add_feature_importance(
                timestamp=datetime.now() + timedelta(days=i),
                importance=importance
            )
        
        trends = tracker.get_feature_importance_trends()
        
        assert isinstance(trends, pd.DataFrame)
        assert len(trends) == 10
        assert all(feat in trends.columns for feat in features)
        
    def test_calculate_regime_performance(self):
        """Test regime-specific performance calculation."""
        tracker = MLPerformanceTracker()
        
        regimes = ['bull', 'bear', 'sideways']
        
        # Add predictions with regime labels
        for i in range(90):
            regime = regimes[i // 30]
            correct = np.random.rand() > 0.4
            
            tracker.add_prediction({
                'timestamp': datetime.now() + timedelta(days=i),
                'model': 'regime_model',
                'predicted': 'BUY',
                'actual': 'BUY' if correct else 'SELL',
                'confidence': np.random.uniform(0.5, 0.9),
                'regime': regime
            })
        
        regime_metrics = tracker.calculate_regime_performance()
        
        assert all(regime in regime_metrics for regime in regimes)
        
        for regime, metrics in regime_metrics.items():
            assert 'accuracy' in metrics
            assert 'count' in metrics
            assert metrics['count'] == 30


class TestWalkForwardAnalyzer:
    """Test WalkForwardAnalyzer class."""
    
    def test_analyzer_initialization(self):
        """Test WalkForwardAnalyzer initialization."""
        analyzer = WalkForwardAnalyzer(
            window_size=252,
            step_size=63,
            min_train_size=500
        )
        
        assert analyzer.window_size == 252
        assert analyzer.step_size == 63
        assert analyzer.min_train_size == 500
        assert analyzer.results == []
        
    def test_generate_windows(self):
        """Test window generation for walk-forward analysis."""
        analyzer = WalkForwardAnalyzer(
            window_size=100,
            step_size=25,
            min_train_size=50
        )
        
        # Create sample data
        dates = pd.date_range(start='2022-01-01', periods=300, freq='D')
        data = pd.DataFrame({'close': np.random.randn(300)}, index=dates)
        
        windows = analyzer.generate_windows(data)
        
        assert isinstance(windows, list)
        assert len(windows) > 0
        
        for window in windows:
            assert 'train_start' in window
            assert 'train_end' in window
            assert 'test_start' in window
            assert 'test_end' in window
            
            # Check window constraints
            train_size = (window['train_end'] - window['train_start']).days
            assert train_size >= analyzer.min_train_size
            
    def test_analyze_window_performance(self):
        """Test single window performance analysis."""
        analyzer = WalkForwardAnalyzer()
        
        # Mock window results
        window_result = {
            'train_performance': {
                'returns': 0.15,
                'sharpe': 1.2,
                'max_drawdown': -0.08
            },
            'test_performance': {
                'returns': 0.12,
                'sharpe': 1.0,
                'max_drawdown': -0.10
            },
            'model_metrics': {
                'train_accuracy': 0.65,
                'test_accuracy': 0.58
            }
        }
        
        analysis = analyzer.analyze_window(window_result)
        
        assert 'overfitting_score' in analysis
        assert 'performance_degradation' in analysis
        assert 'is_stable' in analysis
        
    def test_aggregate_results(self):
        """Test aggregation of walk-forward results."""
        analyzer = WalkForwardAnalyzer()
        
        # Add mock results
        for i in range(5):
            analyzer.results.append({
                'window': i,
                'train_returns': 0.15 + np.random.uniform(-0.05, 0.05),
                'test_returns': 0.12 + np.random.uniform(-0.05, 0.05),
                'train_sharpe': 1.2 + np.random.uniform(-0.2, 0.2),
                'test_sharpe': 1.0 + np.random.uniform(-0.2, 0.2)
            })
        
        summary = analyzer.get_summary_statistics()
        
        assert 'avg_train_returns' in summary
        assert 'avg_test_returns' in summary
        assert 'consistency_score' in summary
        assert 'stability_ratio' in summary
        
    def test_plot_walk_forward_results(self):
        """Test walk-forward results visualization."""
        analyzer = WalkForwardAnalyzer()
        
        # Add results with timestamps
        base_date = datetime.now()
        for i in range(10):
            analyzer.results.append({
                'test_start': base_date + timedelta(days=i*30),
                'test_end': base_date + timedelta(days=(i+1)*30),
                'test_returns': 0.01 + np.random.uniform(-0.005, 0.005),
                'test_sharpe': 1.0 + np.random.uniform(-0.5, 0.5)
            })
        
        # Test that plot data can be generated
        plot_data = analyzer.prepare_plot_data()
        
        assert 'dates' in plot_data
        assert 'returns' in plot_data
        assert 'sharpe_ratios' in plot_data
        assert len(plot_data['dates']) == len(analyzer.results)


class TestMLIntegrationScenarios:
    """Test realistic ML integration scenarios."""
    
    def test_full_ml_backtest_workflow(self, sample_data):
        """Test complete ML backtest workflow."""
        # Create engine and strategy
        engine = MLBacktestEngine()
        
        # Mock ML strategy
        ml_strategy = Mock(spec=MLStrategy)
        ml_strategy.models = {
            'xgboost': Mock(),
            'random_forest': Mock()
        }
        ml_strategy.min_confidence = 0.6
        ml_strategy.use_regime_filter = True
        
        # Mock internal methods
        engine._prepare_ml_data = Mock(return_value=sample_data)
        engine._train_ml_models = Mock()
        engine._generate_ml_signals = Mock(return_value=[
            MLSignal(
                timestamp=sample_data.index[100],
                symbol='TEST',
                direction='BUY',
                confidence=0.75,
                predicted_return=0.02,
                model_name='ensemble'
            )
        ])
        
        # Run backtest
        with patch.object(engine, 'run') as mock_run:
            mock_run.return_value = {
                'total_return': 0.25,
                'sharpe_ratio': 1.5
            }
            
            results = engine.run_ml_backtest(
                data=sample_data,
                ml_strategy=ml_strategy,
                progress_bar=False
            )
        
        assert 'total_return' in results
        assert results['total_return'] == 0.25
        
    def test_regime_based_ml_trading(self):
        """Test ML trading with market regime detection."""
        engine = MLBacktestEngine()
        
        # Create regime-aware strategy
        strategy = Mock(spec=MLStrategy)
        strategy.use_regime_filter = True
        strategy.regime_models = {
            'bull': Mock(),
            'bear': Mock(),
            'sideways': Mock()
        }
        
        # Test regime detection
        current_regime = engine._detect_market_regime(
            data=pd.DataFrame({
                'close': [100, 102, 104, 103, 105],
                'volume': [1000000] * 5
            })
        )
        
        assert current_regime in ['bull', 'bear', 'sideways']
        
    def test_ensemble_model_predictions(self):
        """Test ensemble model prediction aggregation."""
        engine = MLBacktestEngine()
        
        # Mock multiple models
        models = {
            'model1': Mock(predict_proba=Mock(return_value=[[0.3, 0.7]])),
            'model2': Mock(predict_proba=Mock(return_value=[[0.4, 0.6]])),
            'model3': Mock(predict_proba=Mock(return_value=[[0.2, 0.8]]))
        }
        
        # Test soft voting
        ensemble_pred = engine._ensemble_predict(
            models=models,
            features=pd.DataFrame({'feature1': [1], 'feature2': [2]}),
            voting='soft'
        )
        
        assert len(ensemble_pred) == 1
        assert 0 <= ensemble_pred[0] <= 1
        
    def test_adaptive_position_sizing(self):
        """Test ML-based adaptive position sizing."""
        engine = MLBacktestEngine()
        
        # Test different confidence levels
        test_cases = [
            (0.9, 0.05, 10000),   # High confidence, high predicted return
            (0.6, 0.02, 5000),    # Medium confidence, medium return
            (0.51, 0.001, 1000)   # Low confidence, low return
        ]
        
        for confidence, predicted_return, expected_min_size in test_cases:
            signal = MLSignal(
                timestamp=datetime.now(),
                symbol='TEST',
                direction='BUY',
                confidence=confidence,
                predicted_return=predicted_return,
                model_name='test'
            )
            
            size = engine._calculate_adaptive_position_size(
                signal=signal,
                portfolio_value=1000000,
                base_risk=0.02,
                current_price=100
            )
            
            assert size >= expected_min_size
            assert size <= 200000  # Max 20% of portfolio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])