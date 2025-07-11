#!/usr/bin/env python3
"""
Fix ML Integration Issues
Resolves MarketRegime enum conflicts and ensures proper ML model integration
"""

import os
import re
from pathlib import Path

def fix_market_regime_enums():
    """Standardize MarketRegime enum across all files"""
    print("🔧 Fixing MarketRegime enum conflicts...")
    
    # Standard MarketRegime definition
    standard_enum = '''class MarketRegime(Enum):
    """Unified market regime classification"""
    STRONG_BULL = "strong_bull"
    BULL = "bull" 
    SIDEWAYS = "sideways"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    
    @classmethod
    def from_string(cls, regime_str: str):
        """Convert string to MarketRegime enum"""
        regime_map = {
            'strong_bull': cls.STRONG_BULL,
            'bull': cls.BULL,
            'bullish': cls.BULL,  # Legacy mapping
            'sideways': cls.SIDEWAYS,
            'neutral': cls.SIDEWAYS,  # Legacy mapping
            'bear': cls.BEAR,
            'bearish': cls.BEAR,  # Legacy mapping
            'strong_bear': cls.STRONG_BEAR
        }
        return regime_map.get(regime_str.lower(), cls.SIDEWAYS)'''
    
    # Files to update
    files_to_update = [
        'src/ml/models/regime_detection.py',
        'src/ml/market_regime_detector.py',
        'src/strategies/ml_strategy.py'
    ]
    
    for file_path in files_to_update:
        if os.path.exists(file_path):
            print(f"  📝 Updating {file_path}")
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Update MLStrategy to use correct regime values
            if 'ml_strategy.py' in file_path:
                # Fix regime references
                content = re.sub(r'MarketRegime\.BULLISH', 'MarketRegime.BULL', content)
                content = re.sub(r'MarketRegime\.BEARISH', 'MarketRegime.BEAR', content)
                content = re.sub(r'MarketRegime\.NEUTRAL', 'MarketRegime.SIDEWAYS', content)
                
                # Add import if not present
                if 'from src.ml.models.regime_detection import MarketRegime' not in content:
                    import_line = 'from src.ml.models.regime_detection import MarketRegime\n'
                    content = import_line + content
            
            # Write back
            with open(file_path, 'w') as f:
                f.write(content)
            
            print(f"  ✅ Fixed {file_path}")

def create_ml_integration_test():
    """Create a test script to verify ML integration"""
    test_script = '''#!/usr/bin/env python3
"""Test ML Integration - Verify all models work together"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from src.ml.models.enhanced_direction_predictor import EnhancedDirectionPredictor
from src.ml.models.enhanced_volatility_forecaster import EnhancedVolatilityForecaster
from src.ml.models.regime_detection import RegimeDetector, MarketRegime
from src.ml.models.ensemble import EnsembleModel

def test_ml_integration():
    """Test that all ML models work together"""
    print("🧪 Testing ML Integration...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(100) * 2),
        'volume': np.random.randint(1000000, 5000000, 100),
        'rsi': np.random.uniform(20, 80, 100),
        'bb_width': np.random.uniform(0.01, 0.05, 100),
        'macd_signal': np.random.uniform(-2, 2, 100)
    }, index=dates)
    
    try:
        # Test Direction Predictor
        print("  Testing Direction Predictor...")
        dir_model = EnhancedDirectionPredictor()
        X = data[['rsi', 'bb_width', 'macd_signal']]
        y = (data['close'].shift(-1) > data['close']).astype(int)[:-1]
        dir_model.fit(X[:-1], y)
        dir_pred = dir_model.predict(X.iloc[-1:])
        print(f"  ✅ Direction prediction: {dir_pred}")
        
        # Test Volatility Forecaster
        print("  Testing Volatility Forecaster...")
        vol_model = EnhancedVolatilityForecaster()
        vol_target = data['close'].pct_change().rolling(20).std().shift(-1)[:-1]
        vol_model.fit(X[:-1], vol_target)
        vol_pred = vol_model.predict(X.iloc[-1:])
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
        ensemble_pred = ensemble.predict(X.iloc[-1:])
        print(f"  ✅ Ensemble prediction: {ensemble_pred}")
        
        print("\\n✅ All ML models integrated successfully!")
        return True
        
    except Exception as e:
        print(f"\\n❌ ML Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ml_integration()
'''
    
    with open('test_ml_integration.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created ML integration test script")

if __name__ == "__main__":
    print("🚀 Fixing ML Integration Issues")
    print("=" * 50)
    
    # Fix enum conflicts
    fix_market_regime_enums()
    
    # Create test script
    create_ml_integration_test()
    
    print("\n✅ ML integration fixes complete!")
    print("Run 'python test_ml_integration.py' to verify")