#!/usr/bin/env python3
"""
Quick Performance Validation Script
Tests basic functionality and generates performance metrics.
"""

import sys
import time
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test basic imports work."""
    try:
        from src.backtesting.engine import BacktestEngine
        from src.indicators.supertrend_ai import SuperTrendAI
        from src.strategies.supertrend_ai_strategy import SuperTrendAIStrategy
        print("✓ Basic imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_indicator_functionality():
    """Test indicator calculation."""
    try:
        from src.indicators.supertrend_ai import SuperTrendAI
        
        # Create test data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 105,
            'low': np.random.randn(100).cumsum() + 95,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Fix column names to lowercase
        data.columns = data.columns.str.lower()
        
        # Create indicator
        indicator = SuperTrendAI(
            atr_periods=[10],
            multipliers=[2.0],
            n_clusters=3
        )
        
        # Test calculation
        start_time = time.time()
        result = indicator.calculate(data)
        end_time = time.time()
        
        print(f"✓ Indicator calculation successful in {end_time - start_time:.3f}s")
        print(f"  Result type: {type(result)}")
        if hasattr(result, 'trend'):
            print(f"  Trend length: {len(result.trend)}")
        return True
        
    except Exception as e:
        print(f"✗ Indicator test failed: {e}")
        return False

def test_strategy_basic():
    """Test strategy basic functionality."""
    try:
        from src.strategies.supertrend_ai_strategy import SuperTrendAIStrategy, SuperTrendConfig
        
        # Create strategy
        config = SuperTrendConfig(
            atr_length=10,
            min_factor=2.0,
            max_factor=2.0
        )
        strategy = SuperTrendAIStrategy(config=config)
        
        # Test initialization
        print(f"✓ Strategy created successfully")
        print(f"  Strategy config: {strategy.config}")
        print(f"  Factors: {strategy.factors}")
        return True
        
    except Exception as e:
        print(f"✗ Strategy test failed: {e}")
        return False

def test_memory_usage():
    """Test basic memory usage."""
    try:
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Create larger dataset
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        data = pd.DataFrame({
            'open': np.random.randn(1000).cumsum() + 100,
            'high': np.random.randn(1000).cumsum() + 105,
            'low': np.random.randn(1000).cumsum() + 95,
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - initial_memory
        
        print(f"✓ Memory usage test completed")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Current memory: {current_memory:.1f}MB")
        print(f"  Memory increase: {memory_increase:.1f}MB")
        
        return {
            'initial_memory': initial_memory,
            'current_memory': current_memory,
            'memory_increase': memory_increase
        }
        
    except Exception as e:
        print(f"✗ Memory test failed: {e}")
        return None

def run_basic_coverage_check():
    """Run basic coverage check."""
    try:
        import subprocess
        result = subprocess.run([
            'python', '-m', 'pytest', 
            'tests/test_supertrend_ai.py',
            '-v', '--tb=short'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Basic SuperTrend AI tests passed")
            return True
        else:
            print("✗ Some SuperTrend AI tests failed")
            print(f"  Return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"✗ Coverage check failed: {e}")
        return False

def generate_performance_summary():
    """Generate performance summary."""
    print("\n" + "="*60)
    print("PERFORMANCE VALIDATION SUMMARY")
    print("="*60)
    
    results = {}
    
    # Test 1: Basic imports
    results['imports'] = test_basic_imports()
    
    # Test 2: Indicator functionality
    results['indicator'] = test_indicator_functionality()
    
    # Test 3: Strategy basic functionality
    results['strategy'] = test_strategy_basic()
    
    # Test 4: Memory usage
    results['memory'] = test_memory_usage()
    
    # Test 5: Basic coverage
    results['coverage'] = run_basic_coverage_check()
    
    # Calculate overall score
    passed_tests = sum(1 for v in results.values() if v not in [False, None])
    total_tests = len(results)
    score = (passed_tests / total_tests) * 100
    
    print(f"\n📊 Overall Score: {score:.1f}% ({passed_tests}/{total_tests} tests passed)")
    
    # System information
    print(f"\n🖥️  System Information:")
    print(f"  CPU Count: {psutil.cpu_count()}")
    print(f"  Memory Available: {psutil.virtual_memory().available / 1024 / 1024:.1f}MB")
    print(f"  Memory Used: {psutil.virtual_memory().used / 1024 / 1024:.1f}MB")
    print(f"  Memory Percent: {psutil.virtual_memory().percent:.1f}%")
    
    # Save results
    with open('quick_performance_results.json', 'w') as f:
        json.dump({
            'results': results,
            'score': score,
            'timestamp': time.time(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
                'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
                'memory_percent': psutil.virtual_memory().percent
            }
        }, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: quick_performance_results.json")
    
    return results

if __name__ == "__main__":
    generate_performance_summary()