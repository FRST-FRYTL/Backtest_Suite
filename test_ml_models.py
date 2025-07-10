#!/usr/bin/env python3
"""
Test ML models with real data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import the actual ML models with correct names
from src.ml.models.xgboost_direction import EnhancedDirectionPredictor, DirectionPrediction
from src.ml.models.lstm_volatility import VolatilityForecaster, VolatilityPrediction
from src.ml.models.regime_detection import MarketRegimeDetector, RegimeDetection, MarketRegime
from src.ml.models.ensemble import EnsembleModel, EnsemblePrediction

def load_real_data():
    """Load real market data if available."""
    try:
        # Try to load downloaded data
        data = pd.read_csv('data/crypto/BTCUSDT_1d.csv', parse_dates=['timestamp'])
        data.set_index('timestamp', inplace=True)
        print(f"Loaded real data: {len(data)} days of BTCUSDT")
        return data
    except:
        print("No real data found, generating sample data...")
        return generate_sample_data()

def generate_sample_data(n_days: int = 1000) -> pd.DataFrame:
    """Generate sample OHLCV data for demonstration."""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Generate realistic price movements
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, n_days)
    price = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame(index=dates)
    data['close'] = price
    data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, 0.005, n_days))
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.01, n_days)))
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.01, n_days)))
    data['volume'] = np.random.lognormal(15, 0.5, n_days)
    
    # Fill first row
    data.iloc[0, data.columns.get_loc('open')] = 100
    
    return data.dropna()

def test_direction_predictor(data: pd.DataFrame):
    """Test Enhanced Direction Predictor."""
    print("\n=== Testing Enhanced Direction Predictor ===")
    
    # Initialize predictor
    predictor = EnhancedDirectionPredictor(
        ensemble_method='dynamic',
        use_feature_interactions=True,
        use_temporal_features=True,
        optimization_trials=5  # Reduced for testing
    )
    
    # Split data
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # Fit model
    print("Fitting direction predictor...")
    # Use fewer splits for smaller datasets
    n_splits = min(2, max(1, (len(train_data) - 252) // 100))
    predictor.fit(train_data, validate=True, n_splits=n_splits)
    
    # Show validation results
    validation_summary = predictor.get_validation_summary()
    print("\nValidation Results:")
    for metric, value in validation_summary.items():
        print(f"  {metric}: {value:.4f}")
    
    # Make prediction on last data point
    prediction = predictor.predict(test_data.tail(50))  # Need enough data for features
    print(f"\nNext period prediction:")
    print(f"  Direction: {'UP' if prediction.direction == 1 else 'DOWN'}")
    print(f"  Probability: {prediction.probability:.4f}")
    print(f"  Confidence: {prediction.confidence:.4f}")
    
    # Show model predictions
    if prediction.model_predictions:
        print(f"\nIndividual Model Predictions:")
        for model, prob in prediction.model_predictions.items():
            print(f"  {model}: {prob:.4f}")
    
    # Show top features
    print("\nTop 5 Features:")
    top_features = predictor.get_feature_importance_summary(top_n=5)
    for feature, importance in top_features.items():
        print(f"  {feature}: {importance:.4f}")
    
    return predictor

def test_volatility_forecaster(data: pd.DataFrame):
    """Test LSTM volatility forecaster."""
    print("\n=== Testing Volatility Forecaster ===")
    
    # Initialize forecaster
    forecaster = VolatilityForecaster(
        sequence_length=30,
        lstm_units=[32, 16],
        epochs=10,  # Reduced for testing
        use_attention=True
    )
    
    # Split data
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # Fit model
    print("Fitting volatility forecaster...")
    forecaster.fit(train_data, validate=True, n_splits=3)
    
    # Show validation results
    validation_summary = forecaster.get_validation_summary()
    print("\nValidation Results:")
    for metric, value in validation_summary.items():
        print(f"  {metric}: {value:.4f}")
    
    # Make prediction
    prediction = forecaster.predict(test_data.tail(50), n_simulations=20)
    print(f"\nNext period volatility prediction:")
    print(f"  ATR: {prediction.atr_prediction:.4f}")
    print(f"  95% CI: [{prediction.confidence_interval[0]:.4f}, {prediction.confidence_interval[1]:.4f}]")
    print(f"  RMSE: {prediction.rmse:.4f}")
    
    return forecaster

def test_regime_detector(data: pd.DataFrame):
    """Test market regime detector."""
    print("\n=== Testing Regime Detector ===")
    
    # Initialize detector
    detector = MarketRegimeDetector(
        method='ensemble',
        n_regimes=5,
        lookback_period=252
    )
    
    # Fit model
    print("Fitting regime detector...")
    detector.fit(data)
    
    # Make prediction
    prediction = detector.predict(data)
    print(f"\nCurrent Market Regime: {prediction.current_regime.value}")
    print(f"Confidence: {prediction.confidence:.4f}")
    
    print("\nRegime Probabilities:")
    for regime, prob in prediction.regime_probabilities.items():
        print(f"  {regime.value}: {prob:.4f}")
    
    # Show regime statistics
    print("\nHistorical Regime Statistics:")
    stats = detector.get_regime_statistics()
    for regime, regime_stats in stats.items():
        print(f"\n{regime.value}:")
        for stat, value in regime_stats.items():
            if isinstance(value, (int, float)):
                print(f"  {stat}: {value:.4f}")
    
    return detector

def test_ensemble_model(data: pd.DataFrame):
    """Test ensemble model."""
    print("\n=== Testing Ensemble Model ===")
    
    # Initialize ensemble model
    ensemble = EnsembleModel(
        ensemble_method='dynamic',
        direction_weight=0.4,
        volatility_weight=0.3,
        regime_weight=0.3,
        use_meta_learner=True,
        risk_adjustment=True
    )
    
    # Split data
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # Fit ensemble
    print("Fitting ensemble model...")
    ensemble.fit(train_data, validate=True, n_splits=3)
    
    # Make prediction
    prediction = ensemble.predict(test_data.tail(50))
    
    print(f"\n=== Ensemble Prediction ===")
    print(f"Direction: {'UP' if prediction.direction == 1 else 'DOWN'}")
    print(f"Direction Probability: {prediction.direction_probability:.4f}")
    print(f"Direction Confidence: {prediction.direction_confidence:.4f}")
    print(f"\nVolatility Forecast: {prediction.atr_forecast:.4f}")
    print(f"ATR 95% CI: [{prediction.atr_confidence_interval[0]:.4f}, {prediction.atr_confidence_interval[1]:.4f}]")
    print(f"\nMarket Regime: {prediction.market_regime.value}")
    print(f"Ensemble Confidence: {prediction.ensemble_confidence:.4f}")
    print(f"Risk Score: {prediction.risk_score:.4f}")
    
    print(f"\nModel Weights:")
    for model, weight in prediction.model_weights.items():
        print(f"  {model}: {weight:.4f}")
    
    # Get model summary
    summary = ensemble.get_model_summary()
    print(f"\nEnsemble Method: {summary['ensemble_method']}")
    
    return ensemble

def main():
    """Run all ML model tests."""
    print("ML Model Testing Suite")
    print("=" * 50)
    
    # Load data
    data = load_real_data()
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Store results in memory
    from src.ml.models import xgboost_direction
    
    # Test individual models
    direction_predictor = test_direction_predictor(data)
    volatility_forecaster = test_volatility_forecaster(data)
    regime_detector = test_regime_detector(data)
    
    # Test ensemble model
    ensemble_model = test_ensemble_model(data)
    
    print("\n=== All tests completed successfully! ===")
    
    # Return results for further analysis
    return {
        'data': data,
        'direction_predictor': direction_predictor,
        'volatility_forecaster': volatility_forecaster,
        'regime_detector': regime_detector,
        'ensemble_model': ensemble_model
    }

if __name__ == "__main__":
    results = main()