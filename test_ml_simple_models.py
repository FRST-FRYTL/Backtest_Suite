#!/usr/bin/env python3
"""
Simple ML model test without full optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def generate_sample_data(n_days: int = 500) -> pd.DataFrame:
    """Generate sample OHLCV data."""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, n_days)
    price = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame(index=dates)
    data['close'] = price
    data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, 0.005, n_days))
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.01, n_days)))
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.01, n_days)))
    data['volume'] = np.random.lognormal(15, 0.5, n_days)
    
    data.iloc[0, data.columns.get_loc('open')] = 100
    
    return data.dropna()

def test_basic_models():
    """Test basic model functionality."""
    print("Testing Basic ML Model Functionality")
    print("=" * 50)
    
    # Generate data
    data = generate_sample_data(500)
    print(f"Generated {len(data)} days of data")
    
    # Test 1: Import and check models
    print("\n1. Testing Model Imports...")
    try:
        from src.ml.models import DirectionPredictor, VolatilityForecaster, MarketRegimeDetector, EnsembleModel
        print("✓ All models imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return
    
    # Test 2: Check feature engineering
    print("\n2. Testing Feature Engineering...")
    try:
        from src.ml.features.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        features = engineer.create_features(data)
        print(f"✓ Created {len(features.columns)} features from OHLCV data")
        print(f"  Feature shape: {features.shape}")
        print(f"  Sample features: {list(features.columns[:5])}")
    except Exception as e:
        print(f"✗ Feature engineering error: {e}")
    
    # Test 3: Check ML integration
    print("\n3. Testing ML Integration...")
    try:
        from src.backtesting.ml_integration import MLIntegration
        ml_integration = MLIntegration({})
        print("✓ ML Integration module loaded")
    except Exception as e:
        print(f"✗ ML Integration error: {e}")
    
    # Test 4: Check ML strategy
    print("\n4. Testing ML Strategy...")
    try:
        from src.strategies.ml_strategy import MLStrategy
        print("✓ ML Strategy module loaded")
    except Exception as e:
        print(f"✗ ML Strategy error: {e}")
    
    # Test 5: Check report generator
    print("\n5. Testing Report Generator...")
    try:
        from src.ml.reports.report_generator import MLReportGenerator
        report_gen = MLReportGenerator()
        print("✓ Report generator initialized")
        print(f"  Output directory: {report_gen.output_dir}")
    except Exception as e:
        print(f"✗ Report generator error: {e}")
    
    # Test 6: Basic model instantiation
    print("\n6. Testing Model Instantiation...")
    
    try:
        # Test Direction Predictor
        print("  - Direction Predictor...")
        from src.ml.models.xgboost_direction import EnhancedDirectionPredictor
        predictor = EnhancedDirectionPredictor(optimization_trials=1)
        print("    ✓ EnhancedDirectionPredictor created")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    try:
        # Test Volatility Forecaster
        print("  - Volatility Forecaster...")
        from src.ml.models.lstm_volatility import VolatilityForecaster
        forecaster = VolatilityForecaster(epochs=1)
        print("    ✓ VolatilityForecaster created")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    try:
        # Test Regime Detector
        print("  - Regime Detector...")
        from src.ml.models.regime_detection import MarketRegimeDetector
        detector = MarketRegimeDetector()
        print("    ✓ MarketRegimeDetector created")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    try:
        # Test Ensemble Model
        print("  - Ensemble Model...")
        from src.ml.models.ensemble import EnsembleModel
        ensemble = EnsembleModel()
        print("    ✓ EnsembleModel created")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    print("\n" + "=" * 50)
    print("Basic ML Model Testing Complete!")

if __name__ == "__main__":
    test_basic_models()