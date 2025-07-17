"""
Comprehensive tests for ML integration modules.

This module tests the machine learning integration components including
direction predictors, volatility forecasters, regime detectors, and ensemble models.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import joblib
from pathlib import Path

from src.ml.models.enhanced_direction_predictor import EnhancedDirectionPredictor
from src.ml.models.enhanced_volatility_forecaster import EnhancedVolatilityForecaster
from src.ml.models.regime_detection import RegimeDetector
from src.ml.models.ensemble import EnsembleModel
from src.ml.features.feature_engineering import FeatureEngineering
from src.ml.features.feature_selector import FeatureSelector
from src.ml.clustering.kmeans_optimizer import KMeansOptimizer
from src.backtesting.ml_integration import MLIntegration


class TestEnhancedDirectionPredictor:
    """Test Enhanced Direction Predictor functionality."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')
        
        features = pd.DataFrame({
            'returns': np.random.normal(0.001, 0.02, 1000),
            'volatility': np.random.uniform(0.01, 0.05, 1000),
            'volume_ratio': np.random.uniform(0.5, 2.0, 1000),
            'rsi': np.random.uniform(20, 80, 1000),
            'macd_signal': np.random.normal(0, 0.1, 1000),
            'momentum': np.random.normal(0, 0.02, 1000),
            'trend_strength': np.random.uniform(0, 1, 1000),
            'support_resistance': np.random.uniform(0, 1, 1000)
        }, index=dates)
        
        return features
    
    @pytest.fixture
    def sample_targets(self):
        """Create sample targets for testing."""
        np.random.seed(42)
        # Create realistic direction targets (1 for up, -1 for down)
        targets = np.random.choice([-1, 1], 1000, p=[0.45, 0.55])
        return targets
    
    def test_initialization(self):
        """Test model initialization."""
        predictor = EnhancedDirectionPredictor()
        
        assert predictor.model_type == 'xgboost'
        assert predictor.feature_importance_threshold == 0.01
        assert predictor.cross_validation_folds == 5
        assert predictor.random_state == 42
        assert predictor.enable_feature_selection == True
        assert predictor.enable_hyperparameter_tuning == True
    
    def test_custom_initialization(self):
        """Test model initialization with custom parameters."""
        predictor = EnhancedDirectionPredictor(
            model_type='random_forest',
            feature_importance_threshold=0.05,
            cross_validation_folds=3,
            random_state=123,
            enable_feature_selection=False
        )
        
        assert predictor.model_type == 'random_forest'
        assert predictor.feature_importance_threshold == 0.05
        assert predictor.cross_validation_folds == 3
        assert predictor.random_state == 123
        assert predictor.enable_feature_selection == False
    
    def test_feature_engineering(self, sample_features):
        """Test feature engineering pipeline."""
        predictor = EnhancedDirectionPredictor()
        
        # Test feature engineering
        engineered_features = predictor._engineer_features(sample_features)
        
        assert len(engineered_features.columns) > len(sample_features.columns)
        assert 'returns_rolling_mean_5' in engineered_features.columns
        assert 'volatility_rolling_std_10' in engineered_features.columns
        assert 'rsi_normalized' in engineered_features.columns
    
    def test_feature_selection(self, sample_features, sample_targets):
        """Test feature selection process."""
        predictor = EnhancedDirectionPredictor(enable_feature_selection=True)
        
        # Engineer features first
        engineered_features = predictor._engineer_features(sample_features)
        
        # Test feature selection
        selected_features = predictor._select_features(engineered_features, sample_targets)
        
        assert len(selected_features.columns) <= len(engineered_features.columns)
        assert len(selected_features.columns) > 0
    
    def test_model_training(self, sample_features, sample_targets):
        """Test model training process."""
        predictor = EnhancedDirectionPredictor()
        
        # Train model
        predictor.train(sample_features, sample_targets)
        
        assert predictor.model is not None
        assert predictor.is_trained == True
        assert hasattr(predictor, 'feature_names_')
        assert hasattr(predictor, 'training_score_')
    
    def test_model_prediction(self, sample_features, sample_targets):
        """Test model prediction functionality."""
        predictor = EnhancedDirectionPredictor()
        
        # Train model
        predictor.train(sample_features, sample_targets)
        
        # Make predictions
        predictions = predictor.predict(sample_features[:100])
        
        assert len(predictions) == 100
        assert 'prediction' in predictions.columns
        assert 'probability' in predictions.columns
        assert 'confidence' in predictions.columns
        
        # Check prediction values
        assert all(pred in [-1, 1] for pred in predictions['prediction'])
        assert all(0 <= prob <= 1 for prob in predictions['probability'])
        assert all(0 <= conf <= 1 for conf in predictions['confidence'])
    
    def test_model_evaluation(self, sample_features, sample_targets):
        """Test model evaluation metrics."""
        predictor = EnhancedDirectionPredictor()
        
        # Train model
        predictor.train(sample_features, sample_targets)
        
        # Evaluate model
        evaluation = predictor.evaluate(sample_features, sample_targets)
        
        assert 'accuracy' in evaluation
        assert 'precision' in evaluation
        assert 'recall' in evaluation
        assert 'f1_score' in evaluation
        assert 'roc_auc' in evaluation
        assert 'confusion_matrix' in evaluation
        
        # Check metric ranges
        assert 0 <= evaluation['accuracy'] <= 1
        assert 0 <= evaluation['precision'] <= 1
        assert 0 <= evaluation['recall'] <= 1
        assert 0 <= evaluation['f1_score'] <= 1
        assert 0 <= evaluation['roc_auc'] <= 1
    
    def test_hyperparameter_tuning(self, sample_features, sample_targets):
        """Test hyperparameter tuning process."""
        predictor = EnhancedDirectionPredictor(enable_hyperparameter_tuning=True)
        
        # Train with hyperparameter tuning
        predictor.train(sample_features, sample_targets)
        
        assert hasattr(predictor, 'best_params_')
        assert hasattr(predictor, 'tuning_score_')
        assert predictor.tuning_score_ > 0
    
    def test_model_persistence(self, sample_features, sample_targets):
        """Test model saving and loading."""
        predictor = EnhancedDirectionPredictor()
        
        # Train model
        predictor.train(sample_features, sample_targets)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "direction_predictor.pkl"
            
            # Save model
            predictor.save(model_path)
            assert model_path.exists()
            
            # Load model
            loaded_predictor = EnhancedDirectionPredictor.load(model_path)
            
            # Test loaded model
            assert loaded_predictor.is_trained == True
            assert loaded_predictor.model_type == predictor.model_type
            
            # Test predictions are consistent
            original_pred = predictor.predict(sample_features[:10])
            loaded_pred = loaded_predictor.predict(sample_features[:10])
            
            pd.testing.assert_frame_equal(original_pred, loaded_pred)
    
    def test_empty_features(self):
        """Test handling of empty features."""
        predictor = EnhancedDirectionPredictor()
        
        empty_features = pd.DataFrame()
        empty_targets = np.array([])
        
        with pytest.raises(ValueError, match="Empty features"):
            predictor.train(empty_features, empty_targets)
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        predictor = EnhancedDirectionPredictor()
        
        # Very small dataset
        small_features = pd.DataFrame({
            'returns': [0.01, 0.02],
            'volatility': [0.1, 0.2]
        })
        small_targets = np.array([1, -1])
        
        with pytest.raises(ValueError, match="Insufficient data"):
            predictor.train(small_features, small_targets)
    
    def test_prediction_without_training(self, sample_features):
        """Test prediction without training."""
        predictor = EnhancedDirectionPredictor()
        
        with pytest.raises(ValueError, match="Model not trained"):
            predictor.predict(sample_features)


class TestEnhancedVolatilityForecaster:
    """Test Enhanced Volatility Forecaster functionality."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')
        
        # Generate returns with changing volatility
        vol_regime = np.sin(np.linspace(0, 10, 1000)) * 0.01 + 0.02
        returns = np.random.normal(0, vol_regime)
        
        return pd.Series(returns, index=dates)
    
    def test_initialization(self):
        """Test forecaster initialization."""
        forecaster = EnhancedVolatilityForecaster()
        
        assert forecaster.model_type == 'lstm'
        assert forecaster.sequence_length == 30
        assert forecaster.forecast_horizon == 5
        assert forecaster.validation_split == 0.2
        assert forecaster.enable_ensemble == True
    
    def test_volatility_calculation(self, sample_returns):
        """Test volatility calculation methods."""
        forecaster = EnhancedVolatilityForecaster()
        
        # Test realized volatility
        realized_vol = forecaster._calculate_realized_volatility(sample_returns)
        
        assert len(realized_vol) == len(sample_returns)
        assert all(vol >= 0 for vol in realized_vol.dropna())
        
        # Test GARCH volatility
        garch_vol = forecaster._calculate_garch_volatility(sample_returns)
        
        assert len(garch_vol) == len(sample_returns)
        assert all(vol >= 0 for vol in garch_vol.dropna())
    
    def test_feature_engineering(self, sample_returns):
        """Test feature engineering for volatility forecasting."""
        forecaster = EnhancedVolatilityForecaster()
        
        features = forecaster._engineer_volatility_features(sample_returns)
        
        assert len(features.columns) > 5  # Should have multiple features
        assert 'returns_abs' in features.columns
        assert 'returns_squared' in features.columns
        assert 'rolling_vol_5' in features.columns
        assert 'rolling_vol_21' in features.columns
    
    def test_model_training(self, sample_returns):
        """Test model training process."""
        forecaster = EnhancedVolatilityForecaster()
        
        # Train model
        forecaster.train(sample_returns)
        
        assert forecaster.model is not None
        assert forecaster.is_trained == True
        assert hasattr(forecaster, 'scaler_')
        assert hasattr(forecaster, 'training_loss_')
    
    def test_volatility_forecasting(self, sample_returns):
        """Test volatility forecasting functionality."""
        forecaster = EnhancedVolatilityForecaster()
        
        # Train model
        forecaster.train(sample_returns)
        
        # Make forecast
        forecast = forecaster.forecast(sample_returns, horizon=5)
        
        assert 'volatility_forecast' in forecast.columns
        assert 'confidence_lower' in forecast.columns
        assert 'confidence_upper' in forecast.columns
        assert len(forecast) == 5
        
        # Check forecast values
        assert all(vol >= 0 for vol in forecast['volatility_forecast'])
        assert all(forecast['confidence_lower'] <= forecast['volatility_forecast'])
        assert all(forecast['volatility_forecast'] <= forecast['confidence_upper'])
    
    def test_ensemble_forecasting(self, sample_returns):
        """Test ensemble forecasting."""
        forecaster = EnhancedVolatilityForecaster(enable_ensemble=True)
        
        # Train model
        forecaster.train(sample_returns)
        
        # Make ensemble forecast
        forecast = forecaster.forecast(sample_returns, horizon=3)
        
        assert 'ensemble_forecast' in forecast.columns
        assert 'model_weights' in forecast.columns
        assert len(forecast) == 3


class TestRegimeDetector:
    """Test Market Regime Detection functionality."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=500, freq='D')
        
        # Create different market regimes
        regime1 = np.random.normal(0.002, 0.01, 150)  # Bull market
        regime2 = np.random.normal(-0.001, 0.02, 200)  # Bear market
        regime3 = np.random.normal(0.0001, 0.005, 150)  # Sideways market
        
        returns = np.concatenate([regime1, regime2, regime3])
        
        data = pd.DataFrame({
            'returns': returns,
            'close': (1 + returns).cumprod() * 100,
            'volume': np.random.randint(1000000, 5000000, 500),
            'volatility': np.abs(returns)
        }, index=dates)
        
        return data
    
    def test_initialization(self):
        """Test regime detector initialization."""
        detector = RegimeDetector()
        
        assert detector.n_regimes == 3
        assert detector.lookback_window == 252
        assert detector.min_regime_length == 20
        assert detector.enable_transition_probabilities == True
    
    def test_regime_detection(self, sample_market_data):
        """Test regime detection process."""
        detector = RegimeDetector()
        
        # Detect regimes
        regimes = detector.detect_regimes(sample_market_data)
        
        assert 'regime' in regimes.columns
        assert 'regime_probability' in regimes.columns
        assert 'regime_confidence' in regimes.columns
        assert len(regimes) == len(sample_market_data)
        
        # Check regime values
        unique_regimes = regimes['regime'].dropna().unique()
        assert len(unique_regimes) <= detector.n_regimes
        assert all(0 <= prob <= 1 for prob in regimes['regime_probability'].dropna())
    
    def test_transition_matrix(self, sample_market_data):
        """Test transition matrix calculation."""
        detector = RegimeDetector(enable_transition_probabilities=True)
        
        # Detect regimes
        regimes = detector.detect_regimes(sample_market_data)
        
        # Get transition matrix
        transition_matrix = detector.get_transition_matrix()
        
        assert transition_matrix.shape == (detector.n_regimes, detector.n_regimes)
        
        # Check that rows sum to 1 (probability)
        row_sums = transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)
    
    def test_regime_characteristics(self, sample_market_data):
        """Test regime characteristics analysis."""
        detector = RegimeDetector()
        
        # Detect regimes
        regimes = detector.detect_regimes(sample_market_data)
        
        # Get regime characteristics
        characteristics = detector.get_regime_characteristics(sample_market_data, regimes)
        
        assert 'regime_stats' in characteristics
        assert 'regime_descriptions' in characteristics
        
        # Check that all regimes have characteristics
        for regime in regimes['regime'].dropna().unique():
            assert regime in characteristics['regime_stats']
            assert regime in characteristics['regime_descriptions']


class TestEnsembleModel:
    """Test Ensemble Model functionality."""
    
    @pytest.fixture
    def mock_models(self):
        """Create mock models for ensemble testing."""
        models = []
        
        for i in range(3):
            model = Mock()
            model.predict.return_value = pd.DataFrame({
                'prediction': np.random.choice([-1, 1], 100),
                'probability': np.random.uniform(0.5, 1.0, 100)
            })
            model.is_trained = True
            models.append(model)
        
        return models
    
    def test_initialization(self):
        """Test ensemble model initialization."""
        ensemble = EnsembleModel()
        
        assert ensemble.voting_method == 'soft'
        assert ensemble.weight_by_performance == True
        assert ensemble.enable_stacking == False
        assert ensemble.models == []
    
    def test_add_model(self, mock_models):
        """Test adding models to ensemble."""
        ensemble = EnsembleModel()
        
        # Add models
        for model in mock_models:
            ensemble.add_model(model, weight=1.0)
        
        assert len(ensemble.models) == 3
        assert len(ensemble.weights) == 3
        assert all(weight == 1.0 for weight in ensemble.weights)
    
    def test_ensemble_prediction(self, mock_models):
        """Test ensemble prediction process."""
        ensemble = EnsembleModel()
        
        # Add models
        for model in mock_models:
            ensemble.add_model(model, weight=1.0)
        
        # Create mock features
        features = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        
        # Make ensemble prediction
        predictions = ensemble.predict(features)
        
        assert 'ensemble_prediction' in predictions.columns
        assert 'ensemble_probability' in predictions.columns
        assert 'prediction_confidence' in predictions.columns
        assert len(predictions) == 100
    
    def test_model_weighting(self, mock_models):
        """Test model weighting based on performance."""
        ensemble = EnsembleModel(weight_by_performance=True)
        
        # Add models with different weights
        weights = [0.5, 0.3, 0.2]
        for model, weight in zip(mock_models, weights):
            ensemble.add_model(model, weight=weight)
        
        # Test that weights are normalized
        assert abs(sum(ensemble.weights) - 1.0) < 1e-10


class TestFeatureEngineering:
    """Test Feature Engineering functionality."""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        
        close_prices = 100 + np.cumsum(np.random.randn(200) * 0.02)
        
        data = pd.DataFrame({
            'open': close_prices + np.random.randn(200) * 0.1,
            'high': close_prices + np.abs(np.random.randn(200)) * 0.3,
            'low': close_prices - np.abs(np.random.randn(200)) * 0.3,
            'close': close_prices,
            'volume': np.random.randint(1000000, 5000000, 200)
        }, index=dates)
        
        return data
    
    def test_technical_indicators(self, sample_ohlcv_data):
        """Test technical indicator feature engineering."""
        fe = FeatureEngineering()
        
        features = fe.create_technical_features(sample_ohlcv_data)
        
        # Check that common indicators are present
        expected_indicators = [
            'sma_20', 'ema_12', 'rsi_14', 'macd', 'bollinger_upper',
            'bollinger_lower', 'atr_14', 'adx_14', 'stoch_k', 'stoch_d'
        ]
        
        for indicator in expected_indicators:
            assert indicator in features.columns
    
    def test_price_features(self, sample_ohlcv_data):
        """Test price-based feature engineering."""
        fe = FeatureEngineering()
        
        features = fe.create_price_features(sample_ohlcv_data)
        
        # Check price-based features
        expected_features = [
            'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio',
            'upper_shadow', 'lower_shadow', 'body_size'
        ]
        
        for feature in expected_features:
            assert feature in features.columns
    
    def test_volume_features(self, sample_ohlcv_data):
        """Test volume-based feature engineering."""
        fe = FeatureEngineering()
        
        features = fe.create_volume_features(sample_ohlcv_data)
        
        # Check volume-based features
        expected_features = [
            'volume_ratio', 'volume_sma', 'volume_momentum',
            'price_volume', 'volume_breakout'
        ]
        
        for feature in expected_features:
            assert feature in features.columns
    
    def test_lag_features(self, sample_ohlcv_data):
        """Test lag feature creation."""
        fe = FeatureEngineering()
        
        features = fe.create_lag_features(sample_ohlcv_data['close'], lags=[1, 2, 3, 5])
        
        # Check lag features
        expected_features = ['close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5']
        
        for feature in expected_features:
            assert feature in features.columns
    
    def test_rolling_features(self, sample_ohlcv_data):
        """Test rolling feature creation."""
        fe = FeatureEngineering()
        
        features = fe.create_rolling_features(
            sample_ohlcv_data['close'], 
            windows=[5, 10, 20],
            functions=['mean', 'std', 'min', 'max']
        )
        
        # Check rolling features
        expected_features = [
            'close_rolling_mean_5', 'close_rolling_std_5',
            'close_rolling_min_10', 'close_rolling_max_20'
        ]
        
        for feature in expected_features:
            assert feature in features.columns


class TestFeatureSelector:
    """Test Feature Selector functionality."""
    
    @pytest.fixture
    def sample_features_and_target(self):
        """Create sample features and target for testing."""
        np.random.seed(42)
        
        # Create correlated features
        features = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
            'feature4': np.random.randn(1000) * 0.1,  # Low variance
            'feature5': np.random.randn(1000),
        })
        
        # Create target correlated with some features
        target = (features['feature1'] + features['feature2'] * 0.5 + 
                 np.random.randn(1000) * 0.3 > 0).astype(int)
        
        # Make feature3 identical to feature1 (high correlation)
        features['feature3'] = features['feature1'] + np.random.randn(1000) * 0.01
        
        return features, target
    
    def test_variance_threshold_selection(self, sample_features_and_target):
        """Test variance threshold feature selection."""
        features, target = sample_features_and_target
        
        selector = FeatureSelector(method='variance_threshold', threshold=0.01)
        selected_features = selector.select_features(features, target)
        
        # Low variance feature should be removed
        assert 'feature4' not in selected_features.columns
        assert len(selected_features.columns) < len(features.columns)
    
    def test_correlation_selection(self, sample_features_and_target):
        """Test correlation-based feature selection."""
        features, target = sample_features_and_target
        
        selector = FeatureSelector(method='correlation', threshold=0.95)
        selected_features = selector.select_features(features, target)
        
        # Highly correlated feature should be removed
        assert len(selected_features.columns) < len(features.columns)
    
    def test_mutual_information_selection(self, sample_features_and_target):
        """Test mutual information feature selection."""
        features, target = sample_features_and_target
        
        selector = FeatureSelector(method='mutual_info', k=3)
        selected_features = selector.select_features(features, target)
        
        # Should select top 3 features
        assert len(selected_features.columns) == 3
    
    def test_lasso_selection(self, sample_features_and_target):
        """Test LASSO-based feature selection."""
        features, target = sample_features_and_target
        
        selector = FeatureSelector(method='lasso', alpha=0.1)
        selected_features = selector.select_features(features, target)
        
        # Should select subset of features
        assert len(selected_features.columns) <= len(features.columns)
        assert len(selected_features.columns) > 0


class TestKMeansOptimizer:
    """Test K-Means Optimizer functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for clustering."""
        np.random.seed(42)
        
        # Create three clusters
        cluster1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
        cluster2 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 100)
        cluster3 = np.random.multivariate_normal([0, 5], [[1, 0], [0, 1]], 100)
        
        data = np.vstack([cluster1, cluster2, cluster3])
        
        return pd.DataFrame(data, columns=['feature1', 'feature2'])
    
    def test_optimal_k_selection(self, sample_data):
        """Test optimal K selection."""
        optimizer = KMeansOptimizer()
        
        optimal_k = optimizer.find_optimal_k(sample_data, k_range=(2, 8))
        
        # Should find 3 clusters (or close to it)
        assert 2 <= optimal_k <= 5
    
    def test_clustering_performance(self, sample_data):
        """Test clustering performance."""
        optimizer = KMeansOptimizer()
        
        # Perform clustering
        result = optimizer.cluster(sample_data, k=3)
        
        assert 'cluster_labels' in result
        assert 'cluster_centers' in result
        assert 'inertia' in result
        assert 'silhouette_score' in result
        
        # Check cluster labels
        assert len(result['cluster_labels']) == len(sample_data)
        assert len(np.unique(result['cluster_labels'])) == 3
    
    def test_cluster_stability(self, sample_data):
        """Test cluster stability across runs."""
        optimizer = KMeansOptimizer(random_state=42)
        
        # Run clustering multiple times
        results = []
        for _ in range(5):
            result = optimizer.cluster(sample_data, k=3)
            results.append(result['cluster_labels'])
        
        # Results should be stable (identical)
        for i in range(1, 5):
            np.testing.assert_array_equal(results[0], results[i])


class TestMLIntegration:
    """Test ML Integration functionality."""
    
    @pytest.fixture
    def sample_backtest_data(self):
        """Create sample backtest data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Create price data
        returns = np.random.normal(0.001, 0.02, 252)
        close_prices = 100 * (1 + returns).cumprod()
        
        data = pd.DataFrame({
            'open': close_prices + np.random.randn(252) * 0.1,
            'high': close_prices + np.abs(np.random.randn(252)) * 0.3,
            'low': close_prices - np.abs(np.random.randn(252)) * 0.3,
            'close': close_prices,
            'volume': np.random.randint(1000000, 5000000, 252)
        }, index=dates)
        
        return data
    
    def test_ml_integration_initialization(self):
        """Test ML integration initialization."""
        ml_integration = MLIntegration()
        
        assert ml_integration.direction_predictor is not None
        assert ml_integration.volatility_forecaster is not None
        assert ml_integration.regime_detector is not None
        assert ml_integration.feature_engineering is not None
    
    def test_feature_preparation(self, sample_backtest_data):
        """Test feature preparation for ML models."""
        ml_integration = MLIntegration()
        
        features = ml_integration.prepare_features(sample_backtest_data)
        
        assert len(features.columns) > 5  # Should have multiple features
        assert len(features) == len(sample_backtest_data)
        assert not features.isna().all().any()  # No columns should be all NaN
    
    def test_ml_signal_generation(self, sample_backtest_data):
        """Test ML signal generation."""
        ml_integration = MLIntegration()
        
        # Mock trained models
        with patch.object(ml_integration.direction_predictor, 'is_trained', True):
            with patch.object(ml_integration.direction_predictor, 'predict') as mock_predict:
                mock_predict.return_value = pd.DataFrame({
                    'prediction': np.random.choice([-1, 1], len(sample_backtest_data)),
                    'probability': np.random.uniform(0.5, 1.0, len(sample_backtest_data))
                })
                
                signals = ml_integration.generate_ml_signals(sample_backtest_data)
                
                assert 'ml_direction' in signals.columns
                assert 'ml_confidence' in signals.columns
                assert len(signals) == len(sample_backtest_data)
    
    def test_ensemble_prediction(self, sample_backtest_data):
        """Test ensemble prediction functionality."""
        ml_integration = MLIntegration()
        
        # Mock multiple models
        models = [Mock() for _ in range(3)]
        for i, model in enumerate(models):
            model.predict.return_value = pd.DataFrame({
                'prediction': np.random.choice([-1, 1], len(sample_backtest_data)),
                'probability': np.random.uniform(0.5, 1.0, len(sample_backtest_data))
            })
            model.is_trained = True
        
        ensemble_pred = ml_integration.ensemble_predict(sample_backtest_data, models)
        
        assert 'ensemble_prediction' in ensemble_pred.columns
        assert 'ensemble_confidence' in ensemble_pred.columns
        assert len(ensemble_pred) == len(sample_backtest_data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])