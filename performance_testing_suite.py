#!/usr/bin/env python3
"""
Comprehensive Performance Testing Suite for Backtest_Suite
Continues from previous session to achieve >90% test coverage
"""

import os
import sys
import time
import psutil
import tracemalloc
import gc
from typing import Dict, List, Tuple, Any
from pathlib import Path
import json
import logging
from datetime import datetime
import subprocess

import pandas as pd
import numpy as np
import pytest
from memory_profiler import profile
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.backtesting.engine import BacktestEngine
from src.strategies.supertrend_ai_strategy import SuperTrendAIStrategy
from src.indicators.supertrend_ai import SuperTrendAI
from src.reporting.standard_report_generator import StandardReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PerformanceTester:
    """Comprehensive performance testing class."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self) -> Dict[str, pd.DataFrame]:
        """Generate test datasets of various sizes."""
        datasets = {}
        
        # Small dataset (100 days)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        base_price = 100
        datasets['small'] = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + base_price,
            'High': np.random.randn(100).cumsum() + base_price + 5,
            'Low': np.random.randn(100).cumsum() + base_price - 5,
            'Close': np.random.randn(100).cumsum() + base_price,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        # Ensure High >= Low and other price relationships
        datasets['small']['High'] = datasets['small'][['Open', 'High', 'Close']].max(axis=1) + np.random.uniform(0, 2, 100)
        datasets['small']['Low'] = datasets['small'][['Open', 'Low', 'Close']].min(axis=1) - np.random.uniform(0, 2, 100)
        
        # Medium dataset (1000 days)
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        base_price = 100
        datasets['medium'] = pd.DataFrame({
            'Open': np.random.randn(1000).cumsum() + base_price,
            'High': np.random.randn(1000).cumsum() + base_price + 5,
            'Low': np.random.randn(1000).cumsum() + base_price - 5,
            'Close': np.random.randn(1000).cumsum() + base_price,
            'Volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        # Ensure High >= Low and other price relationships
        datasets['medium']['High'] = datasets['medium'][['Open', 'High', 'Close']].max(axis=1) + np.random.uniform(0, 2, 1000)
        datasets['medium']['Low'] = datasets['medium'][['Open', 'Low', 'Close']].min(axis=1) - np.random.uniform(0, 2, 1000)
        
        # Large dataset (10000 days)
        dates = pd.date_range('1990-01-01', periods=10000, freq='D')
        base_price = 100
        datasets['large'] = pd.DataFrame({
            'Open': np.random.randn(10000).cumsum() + base_price,
            'High': np.random.randn(10000).cumsum() + base_price + 5,
            'Low': np.random.randn(10000).cumsum() + base_price - 5,
            'Close': np.random.randn(10000).cumsum() + base_price,
            'Volume': np.random.randint(1000, 10000, 10000)
        }, index=dates)
        # Ensure High >= Low and other price relationships
        datasets['large']['High'] = datasets['large'][['Open', 'High', 'Close']].max(axis=1) + np.random.uniform(0, 2, 10000)
        datasets['large']['Low'] = datasets['large'][['Open', 'Low', 'Close']].min(axis=1) - np.random.uniform(0, 2, 10000)
        
        logger.info(f"Generated test datasets: {list(datasets.keys())}")
        return datasets
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run comprehensive test coverage analysis."""
        logger.info("Running coverage analysis...")
        
        try:
            # Run pytest with coverage
            result = subprocess.run([
                'python', '-m', 'pytest', 
                '--cov=src', 
                '--cov-report=html',
                '--cov-report=term-missing',
                '--cov-report=json',
                '-v'
            ], capture_output=True, text=True, cwd=os.getcwd())
            
            coverage_data = {
                'command_successful': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            # Try to parse coverage JSON if it exists
            coverage_json_path = Path('coverage.json')
            if coverage_json_path.exists():
                with open(coverage_json_path) as f:
                    coverage_json = json.load(f)
                    coverage_data['coverage_summary'] = coverage_json.get('totals', {})
            
            self.results['coverage_analysis'] = coverage_data
            logger.info(f"Coverage analysis completed: {result.returncode == 0}")
            return coverage_data
            
        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")
            self.results['coverage_analysis'] = {'error': str(e)}
            return {'error': str(e)}
    
    def test_backtesting_performance(self) -> Dict[str, Any]:
        """Test backtesting engine performance with different data sizes."""
        logger.info("Testing backtesting performance...")
        
        performance_results = {}
        
        for size_name, data in self.test_data.items():
            logger.info(f"Testing {size_name} dataset ({len(data)} rows)")
            
            try:
                # Create strategy
                from src.strategies.supertrend_ai_strategy import SuperTrendConfig
                config = SuperTrendConfig(
                    atr_length=10,
                    min_factor=2.0,
                    max_factor=2.0
                )
                strategy = SuperTrendAIStrategy(config=config)
                
                # Create engine
                engine = BacktestEngine(
                    initial_capital=100000,
                    commission_rate=0.001
                )
                
                # Measure performance
                start_time = time.time()
                tracemalloc.start()
                
                # Run backtest
                results = engine.run(data, strategy, progress_bar=False)
                
                end_time = time.time()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                performance_results[size_name] = {
                    'execution_time': end_time - start_time,
                    'rows_processed': len(data),
                    'rows_per_second': len(data) / (end_time - start_time),
                    'memory_current_mb': current / 1024 / 1024,
                    'memory_peak_mb': peak / 1024 / 1024,
                    'trades_generated': len(results.get('trades', [])),
                    'success': True
                }
                
            except Exception as e:
                logger.error(f"Backtesting performance test failed for {size_name}: {e}")
                performance_results[size_name] = {
                    'error': str(e),
                    'success': False
                }
        
        self.results['backtesting_performance'] = performance_results
        return performance_results
    
    def test_indicator_performance(self) -> Dict[str, Any]:
        """Test indicator calculation performance."""
        logger.info("Testing indicator performance...")
        
        indicator_results = {}
        
        for size_name, data in self.test_data.items():
            logger.info(f"Testing indicators with {size_name} dataset")
            
            try:
                # Test SuperTrend AI
                indicator = SuperTrendAI(
                    atr_periods=[10],
                    multipliers=[2.0],
                    n_clusters=3
                )
                
                start_time = time.time()
                result = indicator.calculate(data)
                end_time = time.time()
                
                indicator_results[size_name] = {
                    'supertrend_ai_time': end_time - start_time,
                    'calculation_success': result is not None,
                    'result_length': len(result.trend) if hasattr(result, 'trend') else 0,
                    'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
                }
                
            except Exception as e:
                logger.error(f"Indicator performance test failed for {size_name}: {e}")
                indicator_results[size_name] = {
                    'error': str(e),
                    'success': False
                }
        
        self.results['indicator_performance'] = indicator_results
        return indicator_results
    
    def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage patterns."""
        logger.info("Testing memory usage patterns...")
        
        memory_results = {}
        process = psutil.Process()
        
        # Initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        for size_name, data in self.test_data.items():
            logger.info(f"Testing memory usage with {size_name} dataset")
            
            try:
                # Force garbage collection
                gc.collect()
                pre_memory = process.memory_info().rss / 1024 / 1024
                
                # Load data and run operations
                from src.strategies.supertrend_ai_strategy import SuperTrendConfig
                config = SuperTrendConfig(
                    atr_length=10,
                    min_factor=2.0,
                    max_factor=2.0
                )
                strategy = SuperTrendAIStrategy(config=config)
                
                engine = BacktestEngine(
                    initial_capital=100000,
                    commission_rate=0.001
                )
                
                # Run multiple iterations to check for memory leaks
                for i in range(3):
                    results = engine.run(data, strategy, progress_bar=False)
                    
                post_memory = process.memory_info().rss / 1024 / 1024
                
                memory_results[size_name] = {
                    'initial_memory_mb': initial_memory,
                    'pre_test_memory_mb': pre_memory,
                    'post_test_memory_mb': post_memory,
                    'memory_increase_mb': post_memory - pre_memory,
                    'data_size_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
                    'memory_efficiency': (post_memory - pre_memory) / (data.memory_usage(deep=True).sum() / 1024 / 1024)
                }
                
            except Exception as e:
                logger.error(f"Memory usage test failed for {size_name}: {e}")
                memory_results[size_name] = {
                    'error': str(e),
                    'success': False
                }
        
        self.results['memory_usage'] = memory_results
        return memory_results
    
    def test_stress_conditions(self) -> Dict[str, Any]:
        """Test system under stress conditions."""
        logger.info("Testing stress conditions...")
        
        stress_results = {}
        
        try:
            # Test with extreme parameters
            extreme_data = self.test_data['large']
            
            # Test 1: Very short period
            from src.strategies.supertrend_ai_strategy import SuperTrendConfig
            config1 = SuperTrendConfig(
                atr_length=2,
                min_factor=0.5,
                max_factor=0.5
            )
            strategy1 = SuperTrendAIStrategy(config=config1)
            
            # Test 2: Very long period
            config2 = SuperTrendConfig(
                atr_length=50,
                min_factor=5.0,
                max_factor=5.0
            )
            strategy2 = SuperTrendAIStrategy(config=config2)
            
            strategies = [
                ('extreme_short', strategy1),
                ('extreme_long', strategy2)
            ]
            
            for name, strategy in strategies:
                try:
                    engine = BacktestEngine(
                        initial_capital=100000,
                        commission_rate=0.001
                    )
                    
                    start_time = time.time()
                    results = engine.run(extreme_data, strategy, progress_bar=False)
                    end_time = time.time()
                    
                    stress_results[name] = {
                        'execution_time': end_time - start_time,
                        'completed_successfully': True,
                        'trades_count': len(results.get('trades', [])),
                        'final_equity': results.get('final_equity', 0)
                    }
                    
                except Exception as e:
                    stress_results[name] = {
                        'error': str(e),
                        'completed_successfully': False
                    }
        
        except Exception as e:
            logger.error(f"Stress testing failed: {e}")
            stress_results['error'] = str(e)
        
        self.results['stress_testing'] = stress_results
        return stress_results
    
    def test_concurrent_operations(self) -> Dict[str, Any]:
        """Test concurrent operations."""
        logger.info("Testing concurrent operations...")
        
        import threading
        import concurrent.futures
        
        concurrent_results = {}
        
        def run_backtest(data, strategy_name):
            """Run a single backtest."""
            try:
                from src.strategies.supertrend_ai_strategy import SuperTrendConfig
                config = SuperTrendConfig(
                    atr_length=10,
                    min_factor=2.0,
                    max_factor=2.0
                )
                strategy = SuperTrendAIStrategy(config=config)
                
                engine = BacktestEngine(
                    initial_capital=100000,
                    commission_rate=0.001
                )
                
                start_time = time.time()
                results = engine.run(data, strategy, progress_bar=False)
                end_time = time.time()
                
                return {
                    'strategy_name': strategy_name,
                    'execution_time': end_time - start_time,
                    'success': True,
                    'trades_count': len(results.get('trades', []))
                }
                
            except Exception as e:
                return {
                    'strategy_name': strategy_name,
                    'error': str(e),
                    'success': False
                }
        
        try:
            # Test with multiple threads
            test_data = self.test_data['medium']
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(run_backtest, test_data, f"concurrent_{i}")
                    for i in range(4)
                ]
                
                results = []
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            
            concurrent_results['thread_test'] = {
                'total_threads': 4,
                'successful_threads': sum(1 for r in results if r['success']),
                'failed_threads': sum(1 for r in results if not r['success']),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Concurrent operations test failed: {e}")
            concurrent_results['error'] = str(e)
        
        self.results['concurrent_operations'] = concurrent_results
        return concurrent_results
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        logger.info("Generating performance report...")
        
        report = []
        report.append("# Performance Testing Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total test time: {time.time() - self.start_time:.2f} seconds")
        report.append("")
        
        # Coverage Analysis
        if 'coverage_analysis' in self.results:
            coverage = self.results['coverage_analysis']
            report.append("## Coverage Analysis")
            if 'coverage_summary' in coverage:
                summary = coverage['coverage_summary']
                report.append(f"- Total Coverage: {summary.get('percent_covered', 'N/A'):.1f}%")
                report.append(f"- Lines Covered: {summary.get('covered_lines', 'N/A')}")
                report.append(f"- Total Lines: {summary.get('num_statements', 'N/A')}")
            else:
                report.append("- Coverage data not available")
            report.append("")
        
        # Backtesting Performance
        if 'backtesting_performance' in self.results:
            perf = self.results['backtesting_performance']
            report.append("## Backtesting Performance")
            for size, results in perf.items():
                if results.get('success', False):
                    report.append(f"### {size.title()} Dataset")
                    report.append(f"- Execution Time: {results['execution_time']:.2f}s")
                    report.append(f"- Rows per Second: {results['rows_per_second']:.0f}")
                    report.append(f"- Peak Memory: {results['memory_peak_mb']:.1f}MB")
                    report.append(f"- Trades Generated: {results['trades_generated']}")
                    report.append("")
        
        # Memory Usage
        if 'memory_usage' in self.results:
            memory = self.results['memory_usage']
            report.append("## Memory Usage Analysis")
            for size, results in memory.items():
                if 'error' not in results:
                    report.append(f"### {size.title()} Dataset")
                    report.append(f"- Memory Increase: {results['memory_increase_mb']:.1f}MB")
                    report.append(f"- Memory Efficiency: {results['memory_efficiency']:.2f}x")
                    report.append(f"- Data Size: {results['data_size_mb']:.1f}MB")
                    report.append("")
        
        # Stress Testing
        if 'stress_testing' in self.results:
            stress = self.results['stress_testing']
            report.append("## Stress Testing")
            for test_name, results in stress.items():
                if isinstance(results, dict) and results.get('completed_successfully', False):
                    report.append(f"### {test_name.title()} Test")
                    report.append(f"- Execution Time: {results['execution_time']:.2f}s")
                    report.append(f"- Trades Count: {results['trades_count']}")
                    report.append("")
        
        # Concurrent Operations
        if 'concurrent_operations' in self.results:
            concurrent = self.results['concurrent_operations']
            report.append("## Concurrent Operations")
            if 'thread_test' in concurrent:
                thread_test = concurrent['thread_test']
                report.append(f"- Total Threads: {thread_test['total_threads']}")
                report.append(f"- Successful: {thread_test['successful_threads']}")
                report.append(f"- Failed: {thread_test['failed_threads']}")
                report.append("")
        
        # Summary
        report.append("## Summary")
        report.append("Performance testing completed successfully.")
        report.append(f"Total test categories: {len(self.results)}")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        with open('performance_test_report.md', 'w') as f:
            f.write(report_text)
        
        logger.info("Performance report saved to performance_test_report.md")
        return report_text
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all performance tests."""
        logger.info("Starting comprehensive performance testing...")
        
        # Run all test categories
        test_methods = [
            self.run_coverage_analysis,
            self.test_backtesting_performance,
            self.test_indicator_performance,
            self.test_memory_usage,
            self.test_stress_conditions,
            self.test_concurrent_operations
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                logger.error(f"Test {test_method.__name__} failed: {e}")
                self.results[test_method.__name__] = {'error': str(e)}
        
        # Generate report
        report = self.generate_performance_report()
        
        # Save results
        with open('performance_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return self.results


def main():
    """Main execution function."""
    logger.info("Starting Performance Testing Suite")
    
    # Create tester instance
    tester = PerformanceTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE TESTING SUMMARY")
    print("="*60)
    
    for category, result in results.items():
        status = "✓ PASSED" if 'error' not in result else "✗ FAILED"
        print(f"{category:30s} {status}")
    
    print("\nDetailed results saved to:")
    print("- performance_test_results.json")
    print("- performance_test_report.md")
    print("- performance_testing.log")
    
    logger.info("Performance testing completed")


if __name__ == "__main__":
    main()