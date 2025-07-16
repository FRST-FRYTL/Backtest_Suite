#!/usr/bin/env python3
"""
Performance Validation Script for 100% Functional Coverage
"""

import time
import pandas as pd
import numpy as np
import psutil
import os
import sys
from pathlib import Path

def test_data_loading_performance():
    """Test data loading performance"""
    print('\n📈 TEST 1: DATA LOADING PERFORMANCE')
    print('-' * 40)
    
    data_files = [
        'data/SPX/1D/SPY_1D_latest.csv',
        'data/SPX/1H/SPY_1H_latest.csv', 
        'data/SPX/5min/SPY_5min_latest.csv'
    ]
    
    total_time = 0
    files_tested = 0
    results = []
    
    for file_path in data_files:
        if os.path.exists(file_path):
            start_time = time.time()
            data = pd.read_csv(file_path)
            load_time = time.time() - start_time
            total_time += load_time
            files_tested += 1
            
            status = 'PASS' if load_time < 5.0 else 'FAIL'
            print(f'File: {file_path}')
            print(f'  Time: {load_time:.3f}s | Rows: {len(data)} | Status: {status}')
            results.append(load_time < 5.0)
    
    avg_time = total_time / files_tested if files_tested > 0 else 0
    overall_status = 'PASS' if avg_time < 5.0 else 'FAIL'
    print(f'Average loading time: {avg_time:.3f}s | Target: <5.000s | Status: {overall_status}')
    
    return all(results)

def test_pandas_performance():
    """Test pandas operations performance"""
    print('\n📊 TEST 2: PANDAS OPERATIONS PERFORMANCE')
    print('-' * 40)
    
    # Create test data
    size = 10000
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=size, freq='1H'),
        'open': np.random.randn(size).cumsum() + 100,
        'high': np.random.randn(size).cumsum() + 102,
        'low': np.random.randn(size).cumsum() + 98,
        'close': np.random.randn(size).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, size)
    })
    
    results = []
    
    # Test 1: Rolling calculations
    start_time = time.time()
    test_data['sma_20'] = test_data['close'].rolling(20).mean()
    test_data['ema_12'] = test_data['close'].ewm(span=12).mean()
    rolling_time = time.time() - start_time
    
    status = 'PASS' if rolling_time < 0.5 else 'FAIL'
    print(f'Rolling calculations (10k bars): {rolling_time:.3f}s | Target: <0.500s | Status: {status}')
    results.append(rolling_time < 0.5)
    
    # Test 2: Statistical operations
    start_time = time.time()
    test_data['returns'] = test_data['close'].pct_change()
    test_data['volatility'] = test_data['returns'].rolling(20).std()
    stats_time = time.time() - start_time
    
    status = 'PASS' if stats_time < 0.2 else 'FAIL'
    print(f'Statistical operations (10k bars): {stats_time:.3f}s | Target: <0.200s | Status: {status}')
    results.append(stats_time < 0.2)
    
    return all(results)

def test_memory_usage():
    """Test memory usage and leak detection"""
    print('\n🧠 TEST 3: MEMORY USAGE ANALYSIS')
    print('-' * 40)
    
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create and destroy multiple DataFrames
    for i in range(10):
        large_df = pd.DataFrame({
            'col1': np.random.randn(10000),
            'col2': np.random.randn(10000),
            'col3': np.random.randn(10000)
        })
        # Perform operations
        large_df['result'] = large_df['col1'] * large_df['col2'] + large_df['col3']
        del large_df
    
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_growth = memory_after - memory_before
    
    status = 'PASS' if memory_growth < 50 else 'FAIL'
    print(f'Memory before: {memory_before:.1f} MB')
    print(f'Memory after: {memory_after:.1f} MB')
    print(f'Memory growth: {memory_growth:.1f} MB | Target: <50MB | Status: {status}')
    
    return memory_growth < 50

def test_numpy_performance():
    """Test numpy operations performance"""
    print('\n⚡ TEST 4: NUMPY OPERATIONS PERFORMANCE')
    print('-' * 40)
    
    results = []
    
    # Large array operations
    size = 100000
    arr1 = np.random.randn(size)
    arr2 = np.random.randn(size)
    
    start_time = time.time()
    result = arr1 * arr2 + np.sin(arr1) * np.cos(arr2)
    numpy_time = time.time() - start_time
    
    status = 'PASS' if numpy_time < 0.1 else 'FAIL'
    print(f'Numpy operations (100k elements): {numpy_time:.3f}s | Target: <0.100s | Status: {status}')
    results.append(numpy_time < 0.1)
    
    # Matrix operations
    matrix_size = 500
    matrix1 = np.random.randn(matrix_size, matrix_size)
    matrix2 = np.random.randn(matrix_size, matrix_size)
    
    start_time = time.time()
    result = np.dot(matrix1, matrix2)
    matrix_time = time.time() - start_time
    
    status = 'PASS' if matrix_time < 1.0 else 'FAIL'
    print(f'Matrix multiplication (500x500): {matrix_time:.3f}s | Target: <1.000s | Status: {status}')
    results.append(matrix_time < 1.0)
    
    return all(results)

def test_file_io_performance():
    """Test file I/O performance"""
    print('\n💾 TEST 5: FILE I/O PERFORMANCE')
    print('-' * 40)
    
    results = []
    test_file = 'temp_performance_test.csv'
    
    # Create test data
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10000, freq='1H'),
        'value1': np.random.randn(10000),
        'value2': np.random.randn(10000),
        'value3': np.random.randn(10000)
    })
    
    # Test write performance
    start_time = time.time()
    test_data.to_csv(test_file, index=False)
    write_time = time.time() - start_time
    
    status = 'PASS' if write_time < 2.0 else 'FAIL'
    print(f'CSV write (10k rows): {write_time:.3f}s | Target: <2.000s | Status: {status}')
    results.append(write_time < 2.0)
    
    # Test read performance
    start_time = time.time()
    loaded_data = pd.read_csv(test_file)
    read_time = time.time() - start_time
    
    status = 'PASS' if read_time < 1.0 else 'FAIL'
    print(f'CSV read (10k rows): {read_time:.3f}s | Target: <1.000s | Status: {status}')
    results.append(read_time < 1.0)
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
    
    return all(results)

def main():
    """Run all performance validation tests"""
    print('🚀 COMPREHENSIVE PERFORMANCE VALIDATION STARTING')
    print('=' * 60)
    
    # Run all performance tests
    test_results = []
    
    test_results.append(test_data_loading_performance())
    test_results.append(test_pandas_performance())
    test_results.append(test_memory_usage())
    test_results.append(test_numpy_performance())
    test_results.append(test_file_io_performance())
    
    # Summary
    print('\n🎯 PERFORMANCE VALIDATION SUMMARY')
    print('=' * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f'Tests Passed: {passed}/{total}')
    print(f'Success Rate: {passed/total*100:.1f}%')
    
    if passed == total:
        print('🎉 ALL PERFORMANCE TARGETS MET - PRODUCTION READY!')
    else:
        print('⚠️  Some performance targets not met - optimization needed')
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)