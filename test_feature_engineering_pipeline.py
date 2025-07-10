"""
Comprehensive test script for feature engineering pipeline.
Tests technical indicators, multi-timeframe indicators, and ML feature engineering.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import pickle
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.indicators.technical_indicators import TechnicalIndicators
from src.indicators.multi_timeframe_indicators import MultiTimeframeIndicators
from src.ml.features.feature_engineering import FeatureEngineer
import yaml


class FeatureEngineeringTester:
    """Test all feature engineering components."""
    
    def __init__(self):
        self.results = {
            'technical_indicators': {},
            'multi_timeframe': {},
            'ml_features': {},
            'real_data_tests': {},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.errors = []
    
    def load_real_data(self, symbol='SPY', timeframe='1d'):
        """Load real market data from cache."""
        cache_path = Path(f'data/cache/{symbol}_{timeframe}_2019-01-01_2025-07-10.pkl')
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ Loaded {symbol} {timeframe} data: {len(data)} rows")
            return data
        else:
            print(f"✗ Data not found: {cache_path}")
            return None
    
    def test_technical_indicators(self):
        """Test all technical indicators."""
        print("\n=== Testing Technical Indicators ===")
        
        # Generate test data
        n_points = 500
        dates = pd.date_range('2024-01-01', periods=n_points, freq='H')
        
        # Create realistic price data
        np.random.seed(42)
        close = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.01, n_points)))
        
        data = pd.DataFrame({
            'Open': close * (1 + np.random.uniform(-0.001, 0.001, n_points)),
            'High': close * (1 + np.abs(np.random.normal(0, 0.003, n_points))),
            'Low': close * (1 - np.abs(np.random.normal(0, 0.003, n_points))),
            'Close': close,
            'Volume': np.random.lognormal(10, 1, n_points)
        }, index=dates)
        
        # Test each indicator
        indicators = TechnicalIndicators()
        tests = []
        
        # 1. SMA
        try:
            sma = indicators.sma(data['Close'], 20)
            tests.append(('SMA', len(sma.dropna()) > 0, f"Generated {len(sma.dropna())} values"))
        except Exception as e:
            tests.append(('SMA', False, str(e)))
            
        # 2. EMA
        try:
            ema = indicators.ema(data['Close'], 20)
            tests.append(('EMA', len(ema.dropna()) > 0, f"Generated {len(ema.dropna())} values"))
        except Exception as e:
            tests.append(('EMA', False, str(e)))
            
        # 3. RSI
        try:
            rsi = indicators.rsi(data['Close'], 14)
            valid_rsi = (rsi.dropna() >= 0) & (rsi.dropna() <= 100)
            tests.append(('RSI', valid_rsi.all(), f"Valid range: {rsi.dropna().min():.2f} - {rsi.dropna().max():.2f}"))
        except Exception as e:
            tests.append(('RSI', False, str(e)))
            
        # 4. Bollinger Bands
        try:
            bb = indicators.bollinger_bands(data['Close'], 20, [2.0])
            has_bands = all(key in bb for key in ['middle', 'upper_2.0', 'lower_2.0'])
            tests.append(('Bollinger Bands', has_bands, f"Generated {len(bb)} band types"))
        except Exception as e:
            tests.append(('Bollinger Bands', False, str(e)))
            
        # 5. ATR
        try:
            atr = indicators.atr(data['High'], data['Low'], data['Close'], 14)
            tests.append(('ATR', (atr.dropna() > 0).all(), f"Average ATR: {atr.dropna().mean():.4f}"))
        except Exception as e:
            tests.append(('ATR', False, str(e)))
            
        # 6. VWAP
        try:
            vwap = indicators.vwap(data['High'], data['Low'], data['Close'], data['Volume'])
            tests.append(('VWAP', len(vwap.dropna()) > 0, f"Generated {len(vwap.dropna())} values"))
        except Exception as e:
            tests.append(('VWAP', False, str(e)))
            
        # 7. MACD
        try:
            macd = indicators.macd(data['Close'])
            has_all = all(key in macd for key in ['macd', 'signal', 'histogram'])
            tests.append(('MACD', has_all, f"Generated {len(macd)} components"))
        except Exception as e:
            tests.append(('MACD', False, str(e)))
            
        # 8. Stochastic
        try:
            stoch = indicators.stochastic(data['High'], data['Low'], data['Close'])
            has_kd = all(key in stoch for key in ['k', 'd'])
            tests.append(('Stochastic', has_kd, f"Generated K and D lines"))
        except Exception as e:
            tests.append(('Stochastic', False, str(e)))
            
        # 9. OBV
        try:
            obv = indicators.obv(data['Close'], data['Volume'])
            tests.append(('OBV', len(obv.dropna()) > 0, f"Generated {len(obv.dropna())} values"))
        except Exception as e:
            tests.append(('OBV', False, str(e)))
            
        # 10. Rolling VWAP
        try:
            rvwap = indicators.rolling_vwap(data['High'], data['Low'], data['Close'], data['Volume'], 20)
            has_vwap = all(key in rvwap for key in ['vwap', 'std'])
            tests.append(('Rolling VWAP', has_vwap, f"Generated VWAP with std dev"))
        except Exception as e:
            tests.append(('Rolling VWAP', False, str(e)))
            
        # 11. ADX
        try:
            adx = indicators.adx(data['High'], data['Low'], data['Close'], 14)
            has_all = all(key in adx for key in ['adx', 'plus_di', 'minus_di'])
            tests.append(('ADX', has_all, f"Generated ADX with DI lines"))
        except Exception as e:
            tests.append(('ADX', False, str(e)))
        
        # Store results
        passed = sum(1 for _, success, _ in tests if success)
        self.results['technical_indicators'] = {
            'tests': tests,
            'passed': passed,
            'total': len(tests),
            'success_rate': f"{(passed/len(tests)*100):.1f}%"
        }
        
        # Print results
        print(f"\nTechnical Indicators Test Results:")
        for name, success, msg in tests:
            status = "✓" if success else "✗"
            print(f"  {status} {name}: {msg}")
        print(f"\nPassed: {passed}/{len(tests)} ({(passed/len(tests)*100):.1f}%)")
        
        return passed == len(tests)
    
    def test_multi_timeframe_indicators(self):
        """Test multi-timeframe indicator calculations."""
        print("\n=== Testing Multi-Timeframe Indicators ===")
        
        # Load config
        config_path = Path('config/strategy_config.yaml')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default config
            config = {
                'indicators': {
                    'sma': {'periods': [20, 50, 200]},
                    'bollinger_bands': {'period': 20, 'std_devs': [1.25, 2.2, 3.2]},
                    'vwap': {'std_devs': [1, 2, 3]},
                    'rsi': {'period': 14},
                    'atr': {'period': 14},
                    'rolling_vwap': {'periods': [5, 10, 20], 'std_devs': [1, 2]}
                }
            }
        
        # Load real data
        data = self.load_real_data('SPY', '1h')
        if data is None:
            # Use generated data
            n_points = 1000
            dates = pd.date_range('2024-01-01', periods=n_points, freq='H')
            data = pd.DataFrame({
                'Open': 100 + np.random.randn(n_points).cumsum(),
                'High': 101 + np.random.randn(n_points).cumsum(),
                'Low': 99 + np.random.randn(n_points).cumsum(),
                'Close': 100 + np.random.randn(n_points).cumsum(),
                'Volume': np.random.lognormal(10, 1, n_points)
            }, index=dates)
        
        mtf = MultiTimeframeIndicators(config)
        tests = []
        
        # Test individual indicator methods
        try:
            # Test SMA
            data_sma = mtf.calculate_sma(data.copy(), [20, 50])
            has_smas = all(f'SMA_{p}' in data_sma.columns for p in [20, 50])
            tests.append(('SMA Calculation', has_smas, f"Generated {sum(1 for col in data_sma.columns if 'SMA' in col)} SMAs"))
        except Exception as e:
            tests.append(('SMA Calculation', False, str(e)))
            
        try:
            # Test Bollinger Bands
            data_bb = mtf.calculate_bollinger_bands(data.copy())
            has_bb = 'BB_Middle' in data_bb.columns and 'BB_Position' in data_bb.columns
            tests.append(('Bollinger Bands', has_bb, f"Generated {sum(1 for col in data_bb.columns if 'BB' in col)} BB features"))
        except Exception as e:
            tests.append(('Bollinger Bands', False, str(e)))
            
        try:
            # Test Rolling VWAP
            data_rvwap = mtf.calculate_rolling_vwap(data.copy(), periods=[10, 20])
            has_rvwap = any('Rolling_VWAP' in col for col in data_rvwap.columns)
            tests.append(('Rolling VWAP', has_rvwap, f"Generated {sum(1 for col in data_rvwap.columns if 'Rolling_VWAP' in col)} RVWAP features"))
        except Exception as e:
            tests.append(('Rolling VWAP', False, str(e)))
            
        try:
            # Test all indicators
            data_all = mtf.calculate_all_indicators(data.copy(), '1H')
            indicator_cols = [col for col in data_all.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            tests.append(('All Indicators', len(indicator_cols) > 20, f"Generated {len(indicator_cols)} total indicators"))
        except Exception as e:
            tests.append(('All Indicators', False, str(e)))
        
        # Store results
        passed = sum(1 for _, success, _ in tests if success)
        self.results['multi_timeframe'] = {
            'tests': tests,
            'passed': passed,
            'total': len(tests),
            'success_rate': f"{(passed/len(tests)*100):.1f}%"
        }
        
        # Print results
        print(f"\nMulti-Timeframe Test Results:")
        for name, success, msg in tests:
            status = "✓" if success else "✗"
            print(f"  {status} {name}: {msg}")
        print(f"\nPassed: {passed}/{len(tests)} ({(passed/len(tests)*100):.1f}%)")
        
        return passed == len(tests)
    
    def test_ml_feature_engineering(self):
        """Test ML feature engineering."""
        print("\n=== Testing ML Feature Engineering ===")
        
        # Generate test data
        n_points = 500
        dates = pd.date_range('2024-01-01', periods=n_points, freq='H')
        data = pd.DataFrame({
            'open': 100 + np.random.randn(n_points).cumsum(),
            'high': 101 + np.random.randn(n_points).cumsum(),
            'low': 99 + np.random.randn(n_points).cumsum(),
            'close': 100 + np.random.randn(n_points).cumsum(),
            'volume': np.random.lognormal(10, 1, n_points)
        }, index=dates)
        
        engineer = FeatureEngineer()
        tests = []
        
        # Test feature engineering with different configs
        configs = [
            {'price_features': True, 'volume_features': False},
            {'technical_indicators': True, 'statistical_features': True},
            {'market_microstructure': True, 'regime_features': True},
            {}  # Default config
        ]
        
        for i, config in enumerate(configs):
            try:
                features = engineer.engineer_features(data.copy(), config)
                n_features = len([col for col in features.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
                tests.append((f'Config {i+1}', n_features > 0, f"Generated {n_features} features"))
            except Exception as e:
                tests.append((f'Config {i+1}', False, str(e)))
        
        # Test feature selection
        try:
            # Engineer all features
            all_features = engineer.engineer_features(data.copy())
            
            # Create target
            target = data['close'].pct_change().shift(-1).fillna(0)
            target = target.loc[all_features.index]
            
            # Select features
            selected = engineer.select_features(all_features, 'target', threshold=0.001)
            all_features['target'] = target
            
            tests.append(('Feature Selection', len(selected) > 0, f"Selected {len(selected)} important features"))
        except Exception as e:
            tests.append(('Feature Selection', False, str(e)))
        
        # Store results
        passed = sum(1 for _, success, _ in tests if success)
        self.results['ml_features'] = {
            'tests': tests,
            'passed': passed,
            'total': len(tests),
            'success_rate': f"{(passed/len(tests)*100):.1f}%"
        }
        
        # Print results
        print(f"\nML Feature Engineering Test Results:")
        for name, success, msg in tests:
            status = "✓" if success else "✗"
            print(f"  {status} {name}: {msg}")
        print(f"\nPassed: {passed}/{len(tests)} ({(passed/len(tests)*100):.1f}%)")
        
        return passed == len(tests)
    
    def test_real_data_pipeline(self):
        """Test complete pipeline with real market data."""
        print("\n=== Testing Real Data Pipeline ===")
        
        symbols = ['AAPL', 'SPY', 'QQQ']
        tests = []
        
        for symbol in symbols:
            try:
                # Load data
                data = self.load_real_data(symbol, '1d')
                if data is None:
                    tests.append((f'{symbol} Pipeline', False, 'Data not found'))
                    continue
                
                # Ensure correct column names
                data = data.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Volume': 'volume'
                })
                
                # Test technical indicators
                indicators = TechnicalIndicators()
                data['rsi'] = indicators.rsi(data['close'])
                data['sma_20'] = indicators.sma(data['close'], 20)
                
                # Test ML features
                engineer = FeatureEngineer()
                features = engineer.engineer_features(data.head(1000))  # Use subset for speed
                
                n_features = len([col for col in features.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
                tests.append((f'{symbol} Pipeline', n_features > 50, f"Generated {n_features} features"))
                
            except Exception as e:
                tests.append((f'{symbol} Pipeline', False, str(e)))
        
        # Store results
        passed = sum(1 for _, success, _ in tests if success)
        self.results['real_data_tests'] = {
            'tests': tests,
            'passed': passed,
            'total': len(tests),
            'success_rate': f"{(passed/len(tests)*100):.1f}%"
        }
        
        # Print results
        print(f"\nReal Data Pipeline Test Results:")
        for name, success, msg in tests:
            status = "✓" if success else "✗"
            print(f"  {status} {name}: {msg}")
        print(f"\nPassed: {passed}/{len(tests)} ({(passed/len(tests)*100):.1f}%)")
        
        return passed == len(tests)
    
    def run_feature_demo(self):
        """Run the feature engineering demo."""
        print("\n=== Running Feature Engineering Demo ===")
        
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, 'examples/feature_engineering_demo.py'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✓ Demo completed successfully")
                print("\nDemo Output:")
                print(result.stdout)
                return True
            else:
                print("✗ Demo failed")
                print(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"✗ Failed to run demo: {str(e)}")
            return False
    
    def save_results(self):
        """Save test results."""
        output_path = Path('test_results/feature_engineering_test_results.json')
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def run_all_tests(self):
        """Run all feature engineering tests."""
        print("=" * 60)
        print("FEATURE ENGINEERING PIPELINE TEST SUITE")
        print("=" * 60)
        
        # Run tests
        tech_passed = self.test_technical_indicators()
        mtf_passed = self.test_multi_timeframe_indicators()
        ml_passed = self.test_ml_feature_engineering()
        real_passed = self.test_real_data_pipeline()
        demo_passed = self.run_feature_demo()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        all_tests = [
            ('Technical Indicators', tech_passed),
            ('Multi-Timeframe', mtf_passed),
            ('ML Features', ml_passed),
            ('Real Data Pipeline', real_passed),
            ('Feature Demo', demo_passed)
        ]
        
        for name, passed in all_tests:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{name}: {status}")
        
        total_passed = sum(1 for _, passed in all_tests if passed)
        print(f"\nOverall: {total_passed}/{len(all_tests)} test suites passed")
        
        # Save results
        self.save_results()
        
        return total_passed == len(all_tests)


if __name__ == "__main__":
    tester = FeatureEngineeringTester()
    success = tester.run_all_tests()
    exit(0 if success else 1)