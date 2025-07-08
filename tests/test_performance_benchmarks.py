"""Performance benchmark tests for the Backtest Suite."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import psutil
import gc
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from unittest.mock import patch

from src.backtesting import BacktestEngine
from src.strategies import StrategyBuilder
from src.indicators import RSI, BollingerBands, VWAP, TSV, VWMABands
from src.data import DataFetcher, CacheManager
from src.optimization import Optimizer
from src.utils import PerformanceMetrics


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.results = {}
    
    def start(self):
        """Start benchmark measurement."""
        gc.collect()
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def stop(self):
        """Stop benchmark measurement."""
        self.end_time = time.time()
        self.end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        gc.collect()
    
    def get_results(self):
        """Get benchmark results."""
        return {
            'name': self.name,
            'duration': self.end_time - self.start_time,
            'memory_used': self.end_memory - self.start_memory,
            'memory_peak': self.end_memory,
            **self.results
        }


class TestBacktestingPerformance:
    """Test backtesting engine performance."""
    
    def test_baseline_performance(self):
        """Establish baseline performance metrics."""
        benchmark = PerformanceBenchmark("Baseline Backtest")
        
        # Generate 1 year of daily data
        data = self._generate_test_data(periods=252)
        
        # Simple strategy
        strategy = StrategyBuilder("Baseline")
        strategy.add_entry_rule("close > close.shift(1)")
        strategy.add_exit_rule("close < close.shift(1)")
        strategy = strategy.build()
        
        engine = BacktestEngine()
        
        benchmark.start()
        results = engine.run(data, strategy, progress_bar=False)
        benchmark.stop()
        
        perf = benchmark.get_results()
        
        # Baseline should complete quickly
        assert perf['duration'] < 1.0  # Less than 1 second
        assert perf['memory_used'] < 50  # Less than 50MB
        
        # Store baseline for comparison
        self.baseline_duration = perf['duration']
        self.baseline_memory = perf['memory_used']
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        dataset_sizes = [
            (252, "1 Year Daily"),
            (252 * 5, "5 Years Daily"),
            (252 * 10, "10 Years Daily"),
            (252 * 390, "1 Year Minute"),  # Approx minute bars
        ]
        
        results = []
        
        for periods, description in dataset_sizes:
            benchmark = PerformanceBenchmark(f"Large Dataset - {description}")
            
            # Generate data
            data = self._generate_test_data(periods=min(periods, 5000))  # Cap for test speed
            
            # Add indicators
            data['sma_20'] = data['close'].rolling(20).mean()
            data['sma_50'] = data['close'].rolling(50).mean()
            
            # Strategy with indicators
            strategy = StrategyBuilder("Large Dataset Test")
            strategy.add_entry_rule("close > sma_20 and sma_20 > sma_50")
            strategy.add_exit_rule("close < sma_20")
            strategy = strategy.build()
            
            engine = BacktestEngine()
            
            benchmark.start()
            backtest_results = engine.run(data, strategy, progress_bar=False)
            benchmark.stop()
            
            perf = benchmark.get_results()
            perf['periods'] = min(periods, 5000)
            perf['trades'] = len(backtest_results.trades) if not backtest_results.trades.empty else 0
            
            results.append(perf)
        
        # Performance should scale reasonably
        for result in results:
            # Time complexity should be roughly O(n)
            expected_duration = (result['periods'] / 252) * 0.5  # 0.5s per year
            assert result['duration'] < expected_duration * 2  # Allow 2x margin
            
            # Memory should not grow excessively
            assert result['memory_used'] < 200  # Less than 200MB
    
    def test_complex_strategy_performance(self):
        """Test performance with complex strategies."""
        data = self._generate_test_data(periods=252 * 2)  # 2 years
        
        # Add multiple indicators
        indicators = [
            ('rsi', RSI(period=14)),
            ('bb', BollingerBands(period=20)),
            ('vwap', VWAP(window=20)),
            ('tsv', TSV(period=13)),
            ('vwma', VWMABands(period=20))
        ]
        
        for name, indicator in indicators:
            if name == 'rsi':
                data[name] = indicator.calculate(data)
            else:
                result = indicator.calculate(data)
                if isinstance(result, pd.DataFrame):
                    data = pd.concat([data, result], axis=1)
                else:
                    data[name] = result
        
        # Complex strategy with multiple conditions
        strategy = StrategyBuilder("Complex Strategy")
        strategy.add_entry_rule(
            "(rsi < 30 and close < bb_lower) or "
            "(close > vwap and tsv > 0 and close > vwma)"
        )
        strategy.add_exit_rule(
            "(rsi > 70 and close > bb_upper) or "
            "(close < vwap * 0.98)"
        )
        strategy.set_risk_management(
            stop_loss=0.03,
            take_profit=0.06,
            trailing_stop=0.02,
            max_positions=5
        )
        strategy = strategy.build()
        
        engine = BacktestEngine()
        
        benchmark = PerformanceBenchmark("Complex Strategy")
        benchmark.start()
        results = engine.run(data, strategy, progress_bar=False)
        benchmark.stop()
        
        perf = benchmark.get_results()
        
        # Complex strategy should still perform well
        assert perf['duration'] < 5.0  # Less than 5 seconds
        assert perf['memory_used'] < 150  # Less than 150MB
    
    def test_multi_symbol_performance(self):
        """Test performance with multiple symbols."""
        n_symbols = [5, 10, 20, 50]
        
        results = []
        
        for n in n_symbols:
            benchmark = PerformanceBenchmark(f"Multi-Symbol - {n} symbols")
            
            # Generate data for each symbol
            symbol_data = {}
            for i in range(min(n, 20)):  # Cap at 20 for test speed
                symbol_data[f'SYMBOL_{i}'] = self._generate_test_data(periods=252)
            
            # Simple momentum strategy
            strategy = StrategyBuilder("Multi-Symbol Test")
            strategy.add_entry_rule("close > close.rolling(20).mean()")
            strategy.add_exit_rule("close < close.rolling(20).mean()")
            strategy.set_risk_management(position_size=1.0/n)
            strategy = strategy.build()
            
            engine = BacktestEngine(max_positions=n)
            
            benchmark.start()
            backtest_results = engine.run(symbol_data, strategy, progress_bar=False)
            benchmark.stop()
            
            perf = benchmark.get_results()
            perf['n_symbols'] = min(n, 20)
            
            results.append(perf)
        
        # Performance should scale linearly with symbols
        for i in range(1, len(results)):
            ratio = results[i]['n_symbols'] / results[0]['n_symbols']
            expected_duration = results[0]['duration'] * ratio
            
            # Allow 50% margin for overhead
            assert results[i]['duration'] < expected_duration * 1.5
    
    def test_event_processing_performance(self):
        """Test event processing performance."""
        # Generate high-frequency data
        data = self._generate_test_data(periods=1000)
        
        # Track event processing
        events_processed = []
        
        class PerformanceTrackingEngine(BacktestEngine):
            def _process_event(self, event):
                start = time.time()
                result = super()._process_event(event)
                duration = time.time() - start
                events_processed.append({
                    'type': type(event).__name__,
                    'duration': duration
                })
                return result
        
        engine = PerformanceTrackingEngine()
        
        strategy = StrategyBuilder("Event Test")
        strategy.add_entry_rule("close > open")
        strategy.add_exit_rule("close < open")
        strategy = strategy.build()
        
        benchmark = PerformanceBenchmark("Event Processing")
        benchmark.start()
        results = engine.run(data, strategy, progress_bar=False)
        benchmark.stop()
        
        # Analyze event processing
        if events_processed:
            avg_duration = np.mean([e['duration'] for e in events_processed])
            max_duration = np.max([e['duration'] for e in events_processed])
            
            # Events should process quickly
            assert avg_duration < 0.001  # Less than 1ms average
            assert max_duration < 0.01   # Less than 10ms max
    
    def _generate_test_data(self, periods=252):
        """Generate test data for benchmarks."""
        dates = pd.date_range('2020-01-01', periods=periods, freq='D')
        
        # Realistic price movement
        returns = np.random.normal(0.0001, 0.02, periods)
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, periods)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, periods)
        }, index=dates)
        
        # Fix high/low
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data


class TestIndicatorPerformance:
    """Test indicator calculation performance."""
    
    def test_indicator_calculation_speed(self):
        """Test speed of indicator calculations."""
        data_sizes = [252, 252 * 5, 252 * 10]  # 1, 5, 10 years
        
        indicators = [
            ('RSI', RSI(period=14)),
            ('Bollinger Bands', BollingerBands(period=20)),
            ('VWAP', VWAP(window=20)),
            ('TSV', TSV(period=13)),
            ('VWMA Bands', VWMABands(period=20))
        ]
        
        results = []
        
        for periods in data_sizes:
            data = self._generate_test_data(periods=periods)
            
            for name, indicator in indicators:
                benchmark = PerformanceBenchmark(f"{name} - {periods} periods")
                
                benchmark.start()
                result = indicator.calculate(data)
                benchmark.stop()
                
                perf = benchmark.get_results()
                perf['indicator'] = name
                perf['periods'] = periods
                
                results.append(perf)
        
        # All indicators should calculate quickly
        for result in results:
            # Should process at least 1000 bars per second
            bars_per_second = result['periods'] / result['duration']
            assert bars_per_second > 1000
            
            # Memory usage should be minimal
            assert result['memory_used'] < 50  # Less than 50MB
    
    def test_indicator_chaining_performance(self):
        """Test performance of chained indicator calculations."""
        data = self._generate_test_data(periods=252 * 5)
        
        benchmark = PerformanceBenchmark("Indicator Chaining")
        
        benchmark.start()
        
        # Calculate multiple indicators in sequence
        rsi = RSI(period=14)
        data['rsi'] = rsi.calculate(data)
        
        bb = BollingerBands(period=20)
        bb_data = bb.calculate(data)
        data = pd.concat([data, bb_data], axis=1)
        
        vwap = VWAP(window=20)
        vwap_data = vwap.calculate(data)
        data = pd.concat([data, vwap_data], axis=1)
        
        # Create derived indicators
        data['rsi_bb_signal'] = (data['rsi'] < 30) & (data['close'] < data['bb_lower'])
        data['vwap_bb_signal'] = (data['close'] > data['vwap']) & (data['close'] > data['bb_upper'])
        
        benchmark.stop()
        
        perf = benchmark.get_results()
        
        # Chaining should still be efficient
        assert perf['duration'] < 2.0  # Less than 2 seconds
        assert perf['memory_used'] < 100  # Less than 100MB
    
    def _generate_test_data(self, periods=252):
        """Generate test data for benchmarks."""
        dates = pd.date_range('2020-01-01', periods=periods, freq='D')
        
        # Realistic price movement
        returns = np.random.normal(0.0001, 0.02, periods)
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, periods)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, periods)
        }, index=dates)
        
        # Fix high/low
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data


class TestOptimizationPerformance:
    """Test optimization performance."""
    
    def test_parameter_optimization_speed(self):
        """Test speed of parameter optimization."""
        data = self._generate_test_data(periods=252)
        
        # Add indicators
        data['rsi'] = RSI(period=14).calculate(data)
        
        # Parameter grid
        param_grid = {
            'rsi_oversold': [25, 30, 35],
            'rsi_overbought': [65, 70, 75],
            'stop_loss': [0.02, 0.03, 0.05],
            'take_profit': [0.04, 0.06, 0.08]
        }
        
        # Total combinations
        n_combinations = np.prod([len(v) for v in param_grid.values()])
        
        optimizer = Optimizer()
        
        def objective(params):
            strategy = StrategyBuilder("Opt Test")
            strategy.add_entry_rule(f"rsi < {params['rsi_oversold']}")
            strategy.add_exit_rule(f"rsi > {params['rsi_overbought']}")
            strategy.set_risk_management(
                stop_loss=params['stop_loss'],
                take_profit=params['take_profit']
            )
            
            engine = BacktestEngine()
            results = engine.run(data, strategy.build(), progress_bar=False)
            
            return results.performance_metrics.sharpe_ratio
        
        benchmark = PerformanceBenchmark("Parameter Optimization")
        
        benchmark.start()
        
        # Grid search
        best_score = -np.inf
        best_params = None
        
        for rsi_os in param_grid['rsi_oversold']:
            for rsi_ob in param_grid['rsi_overbought']:
                for sl in param_grid['stop_loss']:
                    for tp in param_grid['take_profit']:
                        params = {
                            'rsi_oversold': rsi_os,
                            'rsi_overbought': rsi_ob,
                            'stop_loss': sl,
                            'take_profit': tp
                        }
                        
                        score = objective(params)
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
        
        benchmark.stop()
        
        perf = benchmark.get_results()
        perf['combinations_tested'] = n_combinations
        perf['time_per_combination'] = perf['duration'] / n_combinations
        
        # Should test many combinations quickly
        assert perf['time_per_combination'] < 0.5  # Less than 0.5s per combination
    
    @pytest.mark.asyncio
    async def test_parallel_optimization(self):
        """Test parallel optimization performance."""
        data = self._generate_test_data(periods=252)
        data['rsi'] = RSI(period=14).calculate(data)
        
        # Smaller parameter grid for parallel test
        param_grid = {
            'rsi_oversold': [25, 30],
            'rsi_overbought': [70, 75],
            'stop_loss': [0.03, 0.05]
        }
        
        n_combinations = np.prod([len(v) for v in param_grid.values()])
        
        async def optimize_parallel(n_workers):
            benchmark = PerformanceBenchmark(f"Parallel Optimization - {n_workers} workers")
            
            # Create parameter combinations
            from itertools import product
            combinations = list(product(*param_grid.values()))
            param_list = [
                dict(zip(param_grid.keys(), combo))
                for combo in combinations
            ]
            
            benchmark.start()
            
            # Simulate parallel execution
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                for params in param_list:
                    future = executor.submit(self._evaluate_strategy, data, params)
                    futures.append(future)
                
                # Wait for all to complete
                scores = [f.result() for f in futures]
            
            benchmark.stop()
            
            return benchmark.get_results()
        
        # Test with different worker counts
        results = []
        for n_workers in [1, 2, 4]:
            result = await optimize_parallel(n_workers)
            result['n_workers'] = n_workers
            results.append(result)
        
        # Parallel should be faster
        if len(results) > 1:
            # 2 workers should be faster than 1
            assert results[1]['duration'] < results[0]['duration'] * 0.7
            
            # 4 workers should be faster than 2 (with diminishing returns)
            if len(results) > 2:
                assert results[2]['duration'] < results[1]['duration'] * 0.8
    
    def _evaluate_strategy(self, data, params):
        """Evaluate a single strategy configuration."""
        strategy = StrategyBuilder("Eval")
        strategy.add_entry_rule(f"rsi < {params['rsi_oversold']}")
        strategy.add_exit_rule(f"rsi > {params['rsi_overbought']}")
        strategy.set_risk_management(stop_loss=params['stop_loss'])
        
        engine = BacktestEngine()
        results = engine.run(data, strategy.build(), progress_bar=False)
        
        return results.performance_metrics.sharpe_ratio
    
    def _generate_test_data(self, periods=252):
        """Generate test data for benchmarks."""
        dates = pd.date_range('2020-01-01', periods=periods, freq='D')
        
        # Realistic price movement
        returns = np.random.normal(0.0001, 0.02, periods)
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, periods)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, periods)
        }, index=dates)
        
        # Fix high/low
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data


class TestMemoryEfficiency:
    """Test memory efficiency of the backtesting system."""
    
    def test_memory_usage_scaling(self):
        """Test how memory usage scales with data size."""
        data_sizes = [1000, 5000, 10000, 20000]
        
        results = []
        
        for size in data_sizes:
            benchmark = PerformanceBenchmark(f"Memory Scaling - {size} bars")
            
            # Generate data
            data = self._generate_test_data(periods=size)
            
            # Add indicators
            data['sma_20'] = data['close'].rolling(20).mean()
            data['sma_50'] = data['close'].rolling(50).mean()
            
            strategy = StrategyBuilder("Memory Test")
            strategy.add_entry_rule("close > sma_20 and sma_20 > sma_50")
            strategy.add_exit_rule("close < sma_20")
            strategy = strategy.build()
            
            engine = BacktestEngine()
            
            # Force garbage collection
            gc.collect()
            
            benchmark.start()
            backtest_results = engine.run(data, strategy, progress_bar=False)
            benchmark.stop()
            
            perf = benchmark.get_results()
            perf['data_size'] = size
            
            # Clean up
            del data
            del backtest_results
            gc.collect()
            
            results.append(perf)
        
        # Memory usage should scale linearly or better
        for i in range(1, len(results)):
            size_ratio = results[i]['data_size'] / results[0]['data_size']
            memory_ratio = results[i]['memory_used'] / results[0]['memory_used']
            
            # Memory should not grow faster than data
            assert memory_ratio <= size_ratio * 1.5  # Allow 50% overhead
    
    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated operations."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Run multiple backtests
        for i in range(10):
            data = self._generate_test_data(periods=1000)
            
            strategy = StrategyBuilder(f"Leak Test {i}")
            strategy.add_entry_rule("close > close.shift(1)")
            strategy.add_exit_rule("close < close.shift(1)")
            strategy = strategy.build()
            
            engine = BacktestEngine()
            results = engine.run(data, strategy, progress_bar=False)
            
            # Clean up
            del data
            del results
            del engine
            gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal
        assert memory_growth < 50  # Less than 50MB growth
    
    def test_large_result_handling(self):
        """Test handling of large result sets."""
        # Generate data that will produce many trades
        data = self._generate_test_data(periods=5000)
        
        # High-frequency strategy
        strategy = StrategyBuilder("High Frequency")
        strategy.add_entry_rule("close > open and volume > volume.shift(1)")
        strategy.add_exit_rule("close < open or volume < volume.shift(1)")
        strategy = strategy.build()
        
        engine = BacktestEngine()
        
        benchmark = PerformanceBenchmark("Large Results")
        
        benchmark.start()
        results = engine.run(data, strategy, progress_bar=False)
        benchmark.stop()
        
        perf = benchmark.get_results()
        perf['n_trades'] = len(results.trades) if not results.trades.empty else 0
        
        # Should handle many trades efficiently
        if perf['n_trades'] > 0:
            memory_per_trade = perf['memory_used'] / perf['n_trades']
            assert memory_per_trade < 0.1  # Less than 0.1MB per trade
    
    def _generate_test_data(self, periods=252):
        """Generate test data for benchmarks."""
        dates = pd.date_range('2020-01-01', periods=periods, freq='D')
        
        # Realistic price movement
        returns = np.random.normal(0.0001, 0.02, periods)
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, periods)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, periods)
        }, index=dates)
        
        # Fix high/low
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data


class TestConcurrencyPerformance:
    """Test concurrent operations performance."""
    
    @pytest.mark.asyncio
    async def test_concurrent_data_fetching(self):
        """Test performance of concurrent data fetching."""
        symbols = [f'SYMBOL_{i}' for i in range(20)]
        
        fetcher = DataFetcher()
        
        # Mock the fetch function
        async def mock_fetch(symbol, start, end):
            # Simulate network delay
            await asyncio.sleep(0.1)
            return self._generate_test_data(periods=252)
        
        fetcher.fetch_stock_data = mock_fetch
        
        benchmark = PerformanceBenchmark("Concurrent Fetching")
        
        benchmark.start()
        
        # Fetch all symbols concurrently
        tasks = [
            fetcher.fetch_stock_data(symbol, '2023-01-01', '2023-12-31')
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks)
        
        benchmark.stop()
        
        perf = benchmark.get_results()
        perf['n_symbols'] = len(symbols)
        
        # Concurrent fetching should be much faster than sequential
        sequential_time = len(symbols) * 0.1  # 0.1s per symbol
        assert perf['duration'] < sequential_time * 0.3  # At least 3x faster
    
    def test_concurrent_backtesting(self):
        """Test running multiple backtests concurrently."""
        n_backtests = 10
        
        # Create different strategies
        strategies = []
        for i in range(n_backtests):
            builder = StrategyBuilder(f"Concurrent {i}")
            builder.add_entry_rule(f"close > close.shift({i+1})")
            builder.add_exit_rule(f"close < close.shift({i+1})")
            strategies.append(builder.build())
        
        data = self._generate_test_data(periods=252)
        
        benchmark = PerformanceBenchmark("Concurrent Backtesting")
        
        benchmark.start()
        
        # Run backtests in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for strategy in strategies:
                future = executor.submit(self._run_backtest, data, strategy)
                futures.append(future)
            
            results = [f.result() for f in futures]
        
        benchmark.stop()
        
        perf = benchmark.get_results()
        perf['n_backtests'] = n_backtests
        
        # Parallel execution should provide speedup
        sequential_estimate = n_backtests * 0.2  # Estimate 0.2s per backtest
        assert perf['duration'] < sequential_estimate * 0.5  # At least 2x faster
    
    def _run_backtest(self, data, strategy):
        """Run a single backtest."""
        engine = BacktestEngine()
        return engine.run(data, strategy, progress_bar=False)
    
    def _generate_test_data(self, periods=252):
        """Generate test data for benchmarks."""
        dates = pd.date_range('2020-01-01', periods=periods, freq='D')
        
        # Realistic price movement
        returns = np.random.normal(0.0001, 0.02, periods)
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, periods)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, periods)
        }, index=dates)
        
        # Fix high/low
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])