#!/usr/bin/env python3
"""Test ML Integration - Verify all models work together"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from src.ml.models.enhanced_direction_predictor import EnhancedDirectionPredictor
from src.ml.models.enhanced_volatility_forecaster import EnhancedVolatilityForecaster
from src.ml.models.regime_detection import MarketRegimeDetector as RegimeDetector, MarketRegime
from src.ml.models.ensemble import EnsembleModel

def test_ml_integration():
    """Test that all ML models work together"""
    print("🧪 Testing ML Integration...")
    
    # Create sample OHLCV data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    # Generate realistic OHLC data
    daily_range = np.random.uniform(0.5, 2.5, 100)
    high_prices = close_prices + daily_range * 0.5
    low_prices = close_prices - daily_range * 0.5
    open_prices = close_prices - np.random.randn(100) * 0.5
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 100),
        'rsi': np.random.uniform(20, 80, 100),
        'bb_width': np.random.uniform(0.01, 0.05, 100),
        'macd_signal': np.random.uniform(-2, 2, 100)
    }, index=dates)
    
    try:
        # Test Direction Predictor
        print("  Testing Direction Predictor...")
        dir_model = EnhancedDirectionPredictor()
        y = (data['close'].shift(-1) > data['close']).astype(int)[:-1]
        dir_model.fit(data[:-1], y)
        dir_pred = dir_model.predict(data.iloc[-1:])
        print(f"  ✅ Direction prediction: {dir_pred}")
        
        # Test Volatility Forecaster
        print("  Testing Volatility Forecaster...")
        vol_model = EnhancedVolatilityForecaster()
        vol_target = data['close'].pct_change().rolling(20).std().shift(-1)[:-1]
        vol_model.fit(data[:-1], vol_target)
        vol_pred = vol_model.predict(data.iloc[-1:])
        print(f"  ✅ Volatility forecast: {vol_pred}")
        
        # Test Regime Detector
        print("  Testing Regime Detector...")
        regime_model = RegimeDetector()
        regime_model.fit(data[['close', 'volume']].values)
        current_regime = regime_model.predict(data[['close', 'volume']].iloc[-1:].values)
        print(f"  ✅ Current regime: {MarketRegime(current_regime[0])}")
        
        # Test Ensemble
        print("  Testing Ensemble Model...")
        ensemble = EnsembleModel(
            direction_model=dir_model,
            volatility_model=vol_model,
            regime_model=regime_model
        )
        ensemble_pred = ensemble.predict(data.iloc[-1:])
        print(f"  ✅ Ensemble prediction: {ensemble_pred}")
        
        print("\n✅ All ML models integrated successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ ML Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ml_integration()
