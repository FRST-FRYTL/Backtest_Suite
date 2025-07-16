#!/usr/bin/env python3
"""
Performance Validation Script for 100% Functional Coverage
Validates all critical performance requirements for production deployment
"""

import time
import pandas as pd
import numpy as np
import psutil
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from backtesting.engine import BacktestEngine
from indicators.supertrend_ai import SuperTrendAI
from strategies.supertrend_ai_strategy import SuperTrendAIStrategy
from data.spx_multi_timeframe_fetcher import SPXMultiTimeframeFetcher
from reporting.standard_report_generator import StandardReportGenerator

def create_test_data(size=1000):
    """Create synthetic test data for performance testing"""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=size, freq='1H'),
        'open': np.random.randn(size).cumsum() + 100,
        'high': np.random.randn(size).cumsum() + 102,
        'low': np.random.randn(size).cumsum() + 98,
        'close': np.random.randn(size).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, size)
    })

def test_indicator_performance():
    """Test 1: Indicator Performance - SuperTrend AI"""
    print('\n📊 TEST 1: INDICATOR PERFORMANCE')
    print('-' * 40)
    
    test_data = create_test_data(1000)
    
    # Performance test SuperTrend AI
    start_time = time.time()
    supertrend = SuperTrendAI()
    result = supertrend.calculate(test_data)
    indicator_time = time.time() - start_time
    
    status = "✅ PASS" if indicator_time < 0.1 else "❌ FAIL"
    print(f'✅ SuperTrend AI (1000 bars): {indicator_time:.3f}s')
    print(f'   Target: <0.100s | Status: {status}')
    
    return indicator_time < 0.1

def test_backtesting_performance():
    """Test 2: Backtesting Engine Performance"""
    print('\n⚡ TEST 2: BACKTESTING ENGINE PERFORMANCE')
    print('-' * 40)
    
    # Create larger test data - 10k bars
    large_data = create_test_data(10000)
    
    start_time = time.time()
    engine = BacktestEngine(initial_capital=10000)
    strategy = SuperTrendAIStrategy()
    results = engine.backtest(large_data, strategy)
    backtest_time = time.time() - start_time
    
    status = "✅ PASS" if backtest_time < 1.0 else "❌ FAIL"
    print(f'✅ Backtesting Engine (10k bars): {backtest_time:.3f}s')
    print(f'   Target: <1.000s | Status: {status}')
    
    return backtest_time < 1.0

def test_memory_usage():
    """Test 3: Memory Usage Analysis"""
    print('\n🧠 TEST 3: MEMORY USAGE ANALYSIS')
    print('-' * 40)
    
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run multiple backtests to check for memory leaks
    test_data = create_test_data(1000)
    strategy = SuperTrendAIStrategy()
    
    for i in range(5):
        engine = BacktestEngine(initial_capital=10000)
        results = engine.backtest(test_data, strategy)
        
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_growth = memory_after - memory_before
    
    status = "✅ PASS" if memory_growth < 50 else "❌ FAIL"
    print(f'✅ Memory before: {memory_before:.1f} MB')
    print(f'✅ Memory after: {memory_after:.1f} MB')
    print(f'✅ Memory growth: {memory_growth:.1f} MB')
    print(f'   Target: <50MB growth | Status: {status}')
    
    return memory_growth < 50

def test_portfolio_performance():
    """Test 4: Portfolio Management Performance"""
    print('\n💼 TEST 4: PORTFOLIO MANAGEMENT PERFORMANCE')
    print('-' * 40)
    
    # Test position updates
    engine = BacktestEngine(initial_capital=10000)
    
    start_time = time.time()
    for i in range(100):
        engine.update_position('TEST', 100 + i, 10, 'BUY')
    portfolio_time = time.time() - start_time
    
    status = "✅ PASS" if portfolio_time < 0.05 else "❌ FAIL"
    print(f'✅ Portfolio Updates (100 operations): {portfolio_time:.3f}s')
    print(f'   Target: <0.050s | Status: {status}')
    
    return portfolio_time < 0.05

def test_data_fetching_performance():
    """Test 5: Data Fetching Performance"""
    print('\n📈 TEST 5: DATA FETCHING PERFORMANCE')
    print('-' * 40)
    
    # Test with existing data files
    data_files = [
        'data/SPX/1D/SPY_1D_latest.csv',
        'data/SPX/1H/SPY_1H_latest.csv',
        'data/SPX/5min/SPY_5min_latest.csv'
    ]
    
    total_time = 0
    files_tested = 0
    
    for file_path in data_files:
        if os.path.exists(file_path):
            start_time = time.time()
            data = pd.read_csv(file_path)
            load_time = time.time() - start_time
            total_time += load_time
            files_tested += 1
            
            print(f'✅ {file_path}: {load_time:.3f}s ({len(data)} rows)')
    
    avg_time = total_time / files_tested if files_tested > 0 else 0
    status = "✅ PASS" if avg_time < 5.0 else "❌ FAIL"
    print(f'✅ Average data loading time: {avg_time:.3f}s')
    print(f'   Target: <5.000s | Status: {status}')
    
    return avg_time < 5.0

def test_report_generation_performance():
    """Test 6: Report Generation Performance"""
    print('\n📊 TEST 6: REPORT GENERATION PERFORMANCE')
    print('-' * 40)
    
    # Create sample backtest results
    test_data = create_test_data(1000)
    engine = BacktestEngine(initial_capital=10000)
    strategy = SuperTrendAIStrategy()
    results = engine.backtest(test_data, strategy)
    
    # Test report generation
    start_time = time.time()
    report_gen = StandardReportGenerator()
    report = report_gen.generate_report(results, 'TEST')
    report_time = time.time() - start_time
    
    status = "✅ PASS" if report_time < 10.0 else "❌ FAIL"
    print(f'✅ Report generation: {report_time:.3f}s')
    print(f'   Target: <10.000s | Status: {status}')
    
    return report_time < 10.0

def stress_test_large_dataset():
    """Test 7: Stress Testing with Large Datasets"""
    print('\n🔥 TEST 7: STRESS TESTING - LARGE DATASETS')
    print('-' * 40)
    
    # Create very large dataset - 50k bars
    large_data = create_test_data(50000)
    
    start_time = time.time()
    try:
        engine = BacktestEngine(initial_capital=10000)
        strategy = SuperTrendAIStrategy()
        results = engine.backtest(large_data, strategy)
        stress_time = time.time() - start_time
        
        status = "✅ PASS" if stress_time < 30.0 else "❌ FAIL"
        print(f'✅ Stress test (50k bars): {stress_time:.3f}s')
        print(f'   Target: <30.000s | Status: {status}')
        
        return stress_time < 30.0
    except Exception as e:
        print(f'❌ Stress test failed: {e}')
        return False

def main():
    """Run all performance validation tests"""
    print('🚀 COMPREHENSIVE PERFORMANCE VALIDATION STARTING')
    print('=' * 60)
    
    # Run all performance tests
    test_results = []
    
    test_results.append(test_indicator_performance())
    test_results.append(test_backtesting_performance())
    test_results.append(test_memory_usage())
    test_results.append(test_portfolio_performance())
    test_results.append(test_data_fetching_performance())
    test_results.append(test_report_generation_performance())
    test_results.append(stress_test_large_dataset())
    
    # Summary
    print('\n🎯 PERFORMANCE VALIDATION SUMMARY')
    print('=' * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f'✅ Tests Passed: {passed}/{total}')
    print(f'📊 Success Rate: {passed/total*100:.1f}%')
    
    if passed == total:
        print('🎉 ALL PERFORMANCE TARGETS MET - PRODUCTION READY!')
    else:
        print('⚠️  Some performance targets not met - optimization needed')
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)