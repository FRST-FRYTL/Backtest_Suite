"""
Simple feature engineering test that doesn't require ta-lib.
Tests technical indicators and ML feature engineering using pure Python implementations.
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
from src.ml.features.feature_engineering import FeatureEngineer


class SimpleFeatureEngineeringTester:
    """Test feature engineering components without ta-lib dependency."""
    
    def __init__(self):
        self.results = {
            'technical_indicators': {},
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
        print("\n=== Testing Technical Indicators (Pure Python) ===")
        
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
        
        # Test feature details
        try:
            # Engineer all features
            all_features = engineer.engineer_features(data.copy())
            feature_cols = [col for col in all_features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            # Categorize features
            price_features = [f for f in feature_cols if 'return' in f or 'ratio' in f or 'gap' in f]
            volume_features = [f for f in feature_cols if 'volume' in f or 'obv' in f]
            technical_features = [f for f in feature_cols if 'rsi' in f or 'bb_' in f or 'sma' in f or 'atr' in f]
            statistical_features = [f for f in feature_cols if 'mean' in f or 'std' in f or 'skew' in f or 'kurt' in f]
            
            print(f"\n  Feature Breakdown:")
            print(f"    - Price features: {len(price_features)}")
            print(f"    - Volume features: {len(volume_features)}")
            print(f"    - Technical indicators: {len(technical_features)}")
            print(f"    - Statistical features: {len(statistical_features)}")
            print(f"    - Total features: {len(feature_cols)}")
            
            tests.append(('Feature Categories', len(feature_cols) > 50, f"Total {len(feature_cols)} features"))
        except Exception as e:
            tests.append(('Feature Categories', False, str(e)))
        
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
                
                # Test size
                print(f"\n  Testing {symbol}:")
                print(f"    - Data shape: {data.shape}")
                print(f"    - Date range: {data.index[0]} to {data.index[-1]}")
                
                # Test technical indicators
                indicators = TechnicalIndicators()
                
                # Calculate indicators
                data['RSI'] = indicators.rsi(data['Close'])
                data['SMA_20'] = indicators.sma(data['Close'], 20)
                data['SMA_50'] = indicators.sma(data['Close'], 50)
                bb = indicators.bollinger_bands(data['Close'])
                data['BB_Upper'] = bb['upper_2.0']
                data['BB_Lower'] = bb['lower_2.0']
                
                # Calculate rolling VWAP
                rvwap = indicators.rolling_vwap(
                    data['High'], 
                    data['Low'], 
                    data['Close'], 
                    data['Volume'], 
                    20
                )
                data['Rolling_VWAP'] = rvwap['vwap']
                
                # Count valid indicator values
                valid_indicators = sum([
                    (data['RSI'].dropna().shape[0] > 0),
                    (data['SMA_20'].dropna().shape[0] > 0),
                    (data['SMA_50'].dropna().shape[0] > 0),
                    (data['BB_Upper'].dropna().shape[0] > 0),
                    (data['Rolling_VWAP'].dropna().shape[0] > 0)
                ])
                
                tests.append((f'{symbol} Indicators', valid_indicators == 5, f"Valid indicators: {valid_indicators}/5"))
                
                # Test ML features on subset
                data_subset = data.head(500).copy()
                data_subset.columns = [c.lower() for c in data_subset.columns]
                
                engineer = FeatureEngineer()
                features = engineer.engineer_features(data_subset)
                
                n_features = len([col for col in features.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
                tests.append((f'{symbol} ML Features', n_features > 30, f"Generated {n_features} ML features"))
                
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
    
    def test_feature_demo(self):
        """Test if the feature engineering demo can run with available modules."""
        print("\n=== Testing Feature Engineering Demo (Modified) ===")
        
        try:
            # Create a simple test similar to the demo
            from src.ml.features import FeatureEngineer
            
            # Generate sample data
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
            n_points = len(dates)
            
            np.random.seed(42)
            base_price = 100
            returns = np.random.normal(0.0001, 0.01, n_points)
            close_prices = base_price * np.exp(np.cumsum(returns))
            
            data = pd.DataFrame({
                'open': close_prices * (1 + np.random.uniform(-0.002, 0.002, n_points)),
                'high': close_prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
                'low': close_prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
                'close': close_prices,
                'volume': np.random.lognormal(10, 1, n_points)
            }, index=dates)
            
            # Engineer features
            engineer = FeatureEngineer()
            features = engineer.engineer_features(data.head(500))  # Use subset
            
            n_features = len([col for col in features.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
            
            print(f"✓ Demo test completed")
            print(f"  - Generated {n_features} features from {len(data.head(500))} data points")
            
            return n_features > 30
            
        except Exception as e:
            print(f"✗ Demo test failed: {str(e)}")
            return False
    
    def save_results(self):
        """Save test results."""
        output_path = Path('test_results/feature_engineering_simple_test_results.json')
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert test results to JSON-serializable format
        json_results = {}
        for key, value in self.results.items():
            if key == 'timestamp':
                json_results[key] = value
            elif isinstance(value, dict) and 'tests' in value:
                json_results[key] = {
                    'passed': value['passed'],
                    'total': value['total'],
                    'success_rate': value['success_rate'],
                    'tests': [
                        {'name': t[0], 'success': bool(t[1]), 'message': t[2]}
                        for t in value['tests']
                    ]
                }
            else:
                json_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
        # Also save a summary report
        summary_path = Path('test_results/feature_engineering_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("FEATURE ENGINEERING TEST SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Test Date: {self.results['timestamp']}\n\n")
            
            for category, results in self.results.items():
                if category != 'timestamp' and isinstance(results, dict) and 'tests' in results:
                    f.write(f"{category.upper()}:\n")
                    f.write(f"  Passed: {results['passed']}/{results['total']} ({results['success_rate']})\n")
                    f.write(f"  Tests:\n")
                    for test_item in results['tests']:
                        if isinstance(test_item, (list, tuple)) and len(test_item) >= 3:
                            name, success, msg = test_item[0], test_item[1], test_item[2]
                            status = "PASS" if success else "FAIL"
                            f.write(f"    [{status}] {name}: {msg}\n")
                    f.write("\n")
        
        print(f"Summary saved to: {summary_path}")
    
    def run_all_tests(self):
        """Run all feature engineering tests."""
        print("=" * 60)
        print("FEATURE ENGINEERING TEST SUITE (Without TA-Lib)")
        print("=" * 60)
        
        # Run tests
        tech_passed = self.test_technical_indicators()
        ml_passed = self.test_ml_feature_engineering()
        real_passed = self.test_real_data_pipeline()
        demo_passed = self.test_feature_demo()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        all_tests = [
            ('Technical Indicators', tech_passed),
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
    tester = SimpleFeatureEngineeringTester()
    success = tester.run_all_tests()
    exit(0 if success else 1)