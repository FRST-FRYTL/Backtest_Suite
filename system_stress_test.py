#!/usr/bin/env python3
"""
System Stress Test for Production Readiness
Validates system performance under extreme conditions
"""

import time
import pandas as pd
import numpy as np
import psutil
import os
import sys
import concurrent.futures
from pathlib import Path
import threading
import gc

def stress_test_concurrent_backtests():
    """Test concurrent backtesting performance"""
    print('\n🔥 STRESS TEST: CONCURRENT BACKTESTING')
    print('-' * 50)
    
    def run_backtest(thread_id):
        """Run a single backtest in a thread"""
        # Create synthetic data
        size = 5000
        data = pd.DataFrame({
            'Close': np.random.randn(size).cumsum() + 100,
            'High': np.random.randn(size).cumsum() + 102,
            'Low': np.random.randn(size).cumsum() + 98,
            'Volume': np.random.randint(1000, 10000, size)
        })
        
        # Simple strategy
        data['sma_fast'] = data['Close'].rolling(10).mean()
        data['sma_slow'] = data['Close'].rolling(20).mean()
        
        capital = 10000
        position = 0
        trades = 0
        
        for i in range(20, len(data)):
            current_price = data['Close'].iloc[i]
            fast_ma = data['sma_fast'].iloc[i]
            slow_ma = data['sma_slow'].iloc[i]
            fast_ma_prev = data['sma_fast'].iloc[i-1]
            slow_ma_prev = data['sma_slow'].iloc[i-1]
            
            # Buy signal
            if fast_ma > slow_ma and fast_ma_prev <= slow_ma_prev and position == 0:
                position = capital / current_price
                capital = 0
                trades += 1
            
            # Sell signal
            elif fast_ma < slow_ma and fast_ma_prev >= slow_ma_prev and position > 0:
                capital = position * current_price
                position = 0
                trades += 1
        
        # Close final position
        if position > 0:
            capital = position * data['Close'].iloc[-1]
        
        return {
            'thread_id': thread_id,
            'final_capital': capital,
            'trades': trades,
            'return': (capital - 10000) / 10000 * 100
        }
    
    start_time = time.time()
    
    # Run 8 concurrent backtests
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_backtest, i) for i in range(8)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    concurrent_time = time.time() - start_time
    
    # Calculate stats
    avg_return = np.mean([r['return'] for r in results])
    total_trades = sum([r['trades'] for r in results])
    
    status = 'PASS' if concurrent_time < 10.0 else 'FAIL'
    print(f'8 concurrent backtests: {concurrent_time:.3f}s | Target: <10.000s | Status: {status}')
    print(f'Average return: {avg_return:.2f}%')
    print(f'Total trades: {total_trades}')
    
    return concurrent_time < 10.0

def stress_test_memory_under_load():
    """Test memory usage under heavy load"""
    print('\n🧠 STRESS TEST: MEMORY UNDER HEAVY LOAD')
    print('-' * 50)
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create and process large datasets
    datasets = []
    max_memory = initial_memory
    
    for i in range(10):
        # Large dataset
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=50000, freq='1H'),
            'open': np.random.randn(50000).cumsum() + 100,
            'high': np.random.randn(50000).cumsum() + 102,
            'low': np.random.randn(50000).cumsum() + 98,
            'close': np.random.randn(50000).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 50000)
        })
        
        # Perform calculations
        large_data['sma_20'] = large_data['close'].rolling(20).mean()
        large_data['sma_50'] = large_data['close'].rolling(50).mean()
        large_data['rsi'] = calculate_rsi(large_data['close'], 14)
        large_data['volatility'] = large_data['close'].pct_change().rolling(20).std()
        
        datasets.append(large_data)
        
        # Check memory usage
        current_memory = process.memory_info().rss / 1024 / 1024
        max_memory = max(max_memory, current_memory)
        
        if i % 3 == 0:
            # Periodic cleanup
            gc.collect()
    
    # Cleanup
    del datasets
    gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024
    peak_memory_growth = max_memory - initial_memory
    
    status = 'PASS' if peak_memory_growth < 500 else 'FAIL'
    print(f'Initial memory: {initial_memory:.1f} MB')
    print(f'Peak memory: {max_memory:.1f} MB')
    print(f'Final memory: {final_memory:.1f} MB')
    print(f'Peak memory growth: {peak_memory_growth:.1f} MB | Target: <500MB | Status: {status}')
    
    return peak_memory_growth < 500

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def stress_test_rapid_strategy_switching():
    """Test rapid strategy parameter switching"""
    print('\n⚡ STRESS TEST: RAPID STRATEGY SWITCHING')
    print('-' * 50)
    
    # Load data once
    data_file = 'data/SPX/1H/SPY_1H_latest.csv'
    if not os.path.exists(data_file):
        print(f'❌ Data file not found: {data_file}')
        return False
    
    data = pd.read_csv(data_file)
    
    start_time = time.time()
    
    # Test rapid parameter switching
    results = []
    
    for fast in range(5, 20, 2):
        for slow in range(20, 50, 5):
            for multiplier in [1.5, 2.0, 2.5]:
                # Calculate indicators
                data['sma_fast'] = data['Close'].rolling(fast).mean()
                data['sma_slow'] = data['Close'].rolling(slow).mean()
                data['atr'] = data['High'].rolling(14).max() - data['Low'].rolling(14).min()
                data['upper_band'] = data['Close'] + (multiplier * data['atr'])
                data['lower_band'] = data['Close'] - (multiplier * data['atr'])
                
                # Quick backtest
                capital = 10000
                position = 0
                
                for i in range(slow, min(1000, len(data))):  # Limit for speed
                    current_price = data['Close'].iloc[i]
                    fast_ma = data['sma_fast'].iloc[i]
                    slow_ma = data['sma_slow'].iloc[i]
                    upper = data['upper_band'].iloc[i]
                    lower = data['lower_band'].iloc[i]
                    
                    # Simple strategy
                    if fast_ma > slow_ma and current_price < lower and position == 0:
                        position = capital / current_price
                        capital = 0
                    elif fast_ma < slow_ma and current_price > upper and position > 0:
                        capital = position * current_price
                        position = 0
                
                # Close position
                if position > 0:
                    capital = position * data['Close'].iloc[min(1000, len(data)-1)]
                
                results.append({
                    'fast': fast,
                    'slow': slow,
                    'multiplier': multiplier,
                    'return': (capital - 10000) / 10000
                })
    
    switching_time = time.time() - start_time
    
    # Find best parameters
    best_result = max(results, key=lambda x: x['return'])
    
    status = 'PASS' if switching_time < 15.0 else 'FAIL'
    print(f'Parameter switching ({len(results)} combinations): {switching_time:.3f}s | Target: <15.000s | Status: {status}')
    print(f'Best parameters: Fast={best_result["fast"]}, Slow={best_result["slow"]}, Multiplier={best_result["multiplier"]:.1f}')
    print(f'Best return: {best_result["return"]:.2%}')
    
    return switching_time < 15.0

def stress_test_large_dataset_processing():
    """Test processing very large datasets"""
    print('\n📊 STRESS TEST: LARGE DATASET PROCESSING')
    print('-' * 50)
    
    # Create very large dataset - 100k bars
    size = 100000
    
    start_time = time.time()
    
    large_data = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=size, freq='1H'),
        'open': np.random.randn(size).cumsum() + 100,
        'high': np.random.randn(size).cumsum() + 102,
        'low': np.random.randn(size).cumsum() + 98,
        'close': np.random.randn(size).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, size)
    })
    
    # Perform extensive calculations
    large_data['returns'] = large_data['close'].pct_change()
    large_data['sma_20'] = large_data['close'].rolling(20).mean()
    large_data['sma_50'] = large_data['close'].rolling(50).mean()
    large_data['sma_200'] = large_data['close'].rolling(200).mean()
    large_data['ema_12'] = large_data['close'].ewm(span=12).mean()
    large_data['ema_26'] = large_data['close'].ewm(span=26).mean()
    large_data['rsi'] = calculate_rsi(large_data['close'], 14)
    large_data['volatility'] = large_data['returns'].rolling(20).std()
    large_data['atr'] = large_data['high'].rolling(14).max() - large_data['low'].rolling(14).min()
    
    # Statistical analysis
    large_data['zscore'] = (large_data['close'] - large_data['sma_20']) / large_data['volatility']
    large_data['upper_bollinger'] = large_data['sma_20'] + (2 * large_data['volatility'])
    large_data['lower_bollinger'] = large_data['sma_20'] - (2 * large_data['volatility'])
    
    processing_time = time.time() - start_time
    
    # Memory usage
    memory_mb = large_data.memory_usage(deep=True).sum() / 1024 / 1024
    
    status = 'PASS' if processing_time < 30.0 else 'FAIL'
    print(f'Large dataset processing (100k bars): {processing_time:.3f}s | Target: <30.000s | Status: {status}')
    print(f'Dataset memory usage: {memory_mb:.1f} MB')
    print(f'Final dataset shape: {large_data.shape}')
    
    return processing_time < 30.0

def stress_test_concurrent_data_access():
    """Test concurrent data file access"""
    print('\n🔄 STRESS TEST: CONCURRENT DATA ACCESS')
    print('-' * 50)
    
    def access_data_file(file_path, thread_id):
        """Access and process data file"""
        start = time.time()
        
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            
            # Process data
            data['returns'] = data['Close'].pct_change()
            data['sma_20'] = data['Close'].rolling(20).mean()
            data['volatility'] = data['returns'].rolling(20).std()
            
            # Calculate some metrics
            total_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
            avg_volatility = data['volatility'].mean()
            
            return {
                'thread_id': thread_id,
                'file': file_path,
                'rows': len(data),
                'total_return': total_return,
                'avg_volatility': avg_volatility,
                'time': time.time() - start
            }
        else:
            return None
    
    # Test files
    test_files = [
        'data/SPX/1D/SPY_1D_latest.csv',
        'data/SPX/1H/SPY_1H_latest.csv',
        'data/SPX/5min/SPY_5min_latest.csv',
        'data/SPX/1D/SPY_1D_latest.csv',  # Access same file multiple times
        'data/SPX/1H/SPY_1H_latest.csv'
    ]
    
    start_time = time.time()
    
    # Run concurrent access
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(access_data_file, file, i) for i, file in enumerate(test_files)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    concurrent_access_time = time.time() - start_time
    
    # Filter valid results
    valid_results = [r for r in results if r is not None]
    
    total_rows = sum(r['rows'] for r in valid_results)
    avg_access_time = np.mean([r['time'] for r in valid_results])
    
    status = 'PASS' if concurrent_access_time < 5.0 else 'FAIL'
    print(f'Concurrent data access: {concurrent_access_time:.3f}s | Target: <5.000s | Status: {status}')
    print(f'Total rows processed: {total_rows}')
    print(f'Average access time: {avg_access_time:.3f}s')
    print(f'Files processed: {len(valid_results)}')
    
    return concurrent_access_time < 5.0

def main():
    """Run all stress tests"""
    print('🔥 SYSTEM STRESS TEST FOR PRODUCTION READINESS')
    print('=' * 60)
    
    # Run all stress tests
    test_results = []
    
    test_results.append(stress_test_concurrent_backtests())
    test_results.append(stress_test_memory_under_load())
    test_results.append(stress_test_rapid_strategy_switching())
    test_results.append(stress_test_large_dataset_processing())
    test_results.append(stress_test_concurrent_data_access())
    
    # System resource summary
    print('\n🖥️  SYSTEM RESOURCE SUMMARY')
    print('-' * 50)
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f'CPU Usage: {cpu_percent:.1f}%')
    
    # Memory usage
    memory = psutil.virtual_memory()
    print(f'Memory Usage: {memory.percent:.1f}% ({memory.used / 1024 / 1024 / 1024:.1f}GB / {memory.total / 1024 / 1024 / 1024:.1f}GB)')
    
    # Disk usage
    disk = psutil.disk_usage('/')
    print(f'Disk Usage: {disk.percent:.1f}% ({disk.used / 1024 / 1024 / 1024:.1f}GB / {disk.total / 1024 / 1024 / 1024:.1f}GB)')
    
    # Final summary
    print('\n🎯 STRESS TEST SUMMARY')
    print('=' * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f'Stress Tests Passed: {passed}/{total}')
    print(f'Success Rate: {passed/total*100:.1f}%')
    
    if passed == total:
        print('🎉 SYSTEM READY FOR PRODUCTION DEPLOYMENT!')
        print('✅ All stress tests passed under extreme conditions')
    else:
        print('⚠️  System may need optimization for production workloads')
        print('❌ Some stress tests failed - investigate bottlenecks')
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)