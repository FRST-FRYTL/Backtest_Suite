"""Performance benchmarks for SuperTrend AI indicator and strategy."""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from datetime import datetime, timedelta
from memory_profiler import memory_usage

from src.indicators.supertrend_ai import SuperTrendAI
from src.backtesting import BacktestEngine
from tests.integration.test_supertrend_ai_strategy import SuperTrendAIStrategy


class TestSuperTrendAIPerformance:
    """Performance benchmarks for SuperTrend AI."""
    
    @pytest.fixture
    def large_dataset(self):
        """Create a large dataset for performance testing."""
        # 10 years of minute data (~2.5M bars)
        dates = pd.date_range(start='2014-01-01', end='2024-01-01', freq='T')
        dates = dates[dates.dayofweek < 5]  # Weekdays only
        dates = dates[(dates.hour >= 9) & (dates.hour < 16)]  # Market hours
        
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.0001))
        
        data = pd.DataFrame({
            'open': prices + np.random.randn(len(dates)) * 0.1,
            'high': prices + np.abs(np.random.randn(len(dates))) * 0.5,
            'low': prices - np.abs(np.random.randn(len(dates))) * 0.5,
            'close': prices,
            'volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)
        
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
    
    @pytest.fixture
    def medium_dataset(self):
        """Create a medium dataset for performance testing."""
        # 5 years of 5-minute data (~130K bars)
        dates = pd.date_range(start='2019-01-01', end='2024-01-01', freq='5T')
        dates = dates[dates.dayofweek < 5]
        dates = dates[(dates.hour >= 9) & (dates.hour < 16)]
        
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.0005))
        
        data = pd.DataFrame({
            'open': prices + np.random.randn(len(dates)) * 0.2,
            'high': prices + np.abs(np.random.randn(len(dates))) * 0.8,
            'low': prices - np.abs(np.random.randn(len(dates))) * 0.8,
            'close': prices,
            'volume': np.random.randint(50000, 500000, len(dates))
        }, index=dates)
        
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
    
    def test_indicator_calculation_speed(self, medium_dataset):
        """Test SuperTrend AI calculation speed."""
        st = SuperTrendAI()
        
        # Warm up
        _ = st.calculate(medium_dataset[:1000])
        
        # Time the calculation
        start_time = time.time()
        result = st.calculate(medium_dataset)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        bars_per_second = len(medium_dataset) / calculation_time
        
        print(f"\nIndicator Calculation Performance:")
        print(f"  Dataset size: {len(medium_dataset):,} bars")
        print(f"  Calculation time: {calculation_time:.2f} seconds")
        print(f"  Bars per second: {bars_per_second:,.0f}")
        
        # Performance assertions
        assert calculation_time < 10  # Should process medium dataset in < 10 seconds
        assert bars_per_second > 10000  # Should process > 10K bars per second
    
    def test_memory_usage(self, medium_dataset):
        """Test memory usage of SuperTrend AI calculation."""
        st = SuperTrendAI()
        
        # Get baseline memory
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate with memory profiling
        def calculate_supertrend():
            return st.calculate(medium_dataset)
        
        mem_usage = memory_usage(calculate_supertrend, interval=0.1)
        peak_memory = max(mem_usage)
        memory_increase = peak_memory - baseline_memory
        
        print(f"\nMemory Usage:")
        print(f"  Baseline memory: {baseline_memory:.1f} MB")
        print(f"  Peak memory: {peak_memory:.1f} MB")
        print(f"  Memory increase: {memory_increase:.1f} MB")
        print(f"  Memory per bar: {memory_increase / len(medium_dataset) * 1000:.3f} KB")
        
        # Memory assertions
        assert memory_increase < 500  # Should use < 500MB additional memory
        assert memory_increase / len(medium_dataset) < 0.01  # < 10KB per bar
    
    def test_incremental_updates(self, medium_dataset):
        """Test performance of incremental updates."""
        st = SuperTrendAI()
        
        # Initial calculation
        initial_data = medium_dataset[:-1000]
        result = st.calculate(initial_data)
        
        # Time incremental updates
        update_times = []
        for i in range(0, 1000, 100):
            new_data = medium_dataset[len(initial_data) + i:len(initial_data) + i + 100]
            
            start_time = time.time()
            updated_result = st.update(new_data, result)
            end_time = time.time()
            
            update_times.append(end_time - start_time)
            result = updated_result
        
        avg_update_time = np.mean(update_times)
        print(f"\nIncremental Update Performance:")
        print(f"  Average update time (100 bars): {avg_update_time*1000:.2f} ms")
        print(f"  Updates per second: {1/avg_update_time:.0f}")
        
        # Should handle updates quickly
        assert avg_update_time < 0.1  # < 100ms per 100-bar update
    
    def test_backtest_performance(self, medium_dataset):
        """Test full backtest performance."""
        # Add ATR for the strategy
        high_low = medium_dataset['high'] - medium_dataset['low']
        high_close = (medium_dataset['high'] - medium_dataset['close'].shift()).abs()
        low_close = (medium_dataset['low'] - medium_dataset['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        medium_dataset['atr'] = true_range.rolling(14).mean()
        
        strategy = SuperTrendAIStrategy()
        engine = BacktestEngine(initial_capital=100000)
        
        # Time the backtest
        start_time = time.time()
        results = engine.run(data=medium_dataset, strategy=strategy)
        end_time = time.time()
        
        backtest_time = end_time - start_time
        bars_per_second = len(medium_dataset) / backtest_time
        
        print(f"\nBacktest Performance:")
        print(f"  Dataset size: {len(medium_dataset):,} bars")
        print(f"  Backtest time: {backtest_time:.2f} seconds")
        print(f"  Bars per second: {bars_per_second:,.0f}")
        print(f"  Total trades: {len(results['trades'])}")
        
        # Performance assertions
        assert backtest_time < 30  # Should complete in < 30 seconds
        assert bars_per_second > 4000  # Should process > 4K bars per second
    
    def test_parameter_optimization_performance(self, medium_dataset):
        """Test parameter optimization performance."""
        # Prepare data
        high_low = medium_dataset['high'] - medium_dataset['low']
        high_close = (medium_dataset['high'] - medium_dataset['close'].shift()).abs()
        low_close = (medium_dataset['low'] - medium_dataset['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        medium_dataset['atr'] = true_range.rolling(14).mean()
        
        # Define parameter grid
        param_grid = []
        for signal_strength in [3, 4, 5, 6]:
            for sl_mult in [1.5, 2.0, 2.5]:
                param_grid.append({
                    'min_signal_strength': signal_strength,
                    'stop_loss_atr_mult': sl_mult
                })
        
        # Time optimization
        start_time = time.time()
        
        results = []
        for params in param_grid:
            strategy = SuperTrendAIStrategy(params)
            engine = BacktestEngine(initial_capital=100000)
            result = engine.run(data=medium_dataset, strategy=strategy)
            results.append({
                'params': params,
                'sharpe': result['metrics']['sharpe_ratio']
            })
        
        end_time = time.time()
        
        optimization_time = end_time - start_time
        time_per_combo = optimization_time / len(param_grid)
        
        print(f"\nOptimization Performance:")
        print(f"  Parameter combinations: {len(param_grid)}")
        print(f"  Total time: {optimization_time:.2f} seconds")
        print(f"  Time per combination: {time_per_combo:.2f} seconds")
        
        # Should optimize efficiently
        assert time_per_combo < 3  # < 3 seconds per parameter combination
    
    def test_concurrent_calculations(self, medium_dataset):
        """Test concurrent calculation performance."""
        import concurrent.futures
        
        # Split data into chunks
        chunk_size = len(medium_dataset) // 4
        chunks = [
            medium_dataset[i:i+chunk_size] 
            for i in range(0, len(medium_dataset), chunk_size)
        ]
        
        def calculate_chunk(data_chunk):
            st = SuperTrendAI()
            return st.calculate(data_chunk)
        
        # Time concurrent execution
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(calculate_chunk, chunk) for chunk in chunks]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        
        concurrent_time = end_time - start_time
        
        # Time sequential execution for comparison
        start_time = time.time()
        sequential_results = [calculate_chunk(chunk) for chunk in chunks]
        sequential_time = time.time() - start_time
        
        speedup = sequential_time / concurrent_time
        
        print(f"\nConcurrent Execution Performance:")
        print(f"  Sequential time: {sequential_time:.2f} seconds")
        print(f"  Concurrent time: {concurrent_time:.2f} seconds")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Should show some speedup (though limited by GIL)
        assert speedup > 1.2  # At least 20% speedup
    
    def test_vectorization_efficiency(self, medium_dataset):
        """Test vectorization efficiency of calculations."""
        st = SuperTrendAI()
        
        # Test with different data sizes
        sizes = [1000, 5000, 10000, 50000]
        times = []
        
        for size in sizes:
            data_subset = medium_dataset[:size]
            
            start_time = time.time()
            _ = st.calculate(data_subset)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Calculate scaling factor
        scaling_factors = []
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = times[i] / times[i-1]
            scaling_factors.append(time_ratio / size_ratio)
        
        avg_scaling = np.mean(scaling_factors)
        
        print(f"\nVectorization Efficiency:")
        for i, (size, t) in enumerate(zip(sizes, times)):
            print(f"  {size:,} bars: {t:.3f} seconds ({size/t:,.0f} bars/sec)")
        print(f"  Average scaling factor: {avg_scaling:.3f}")
        
        # Should scale sub-linearly (good vectorization)
        assert avg_scaling < 1.2  # Less than linear scaling
    
    def test_edge_case_performance(self):
        """Test performance with edge cases."""
        # Very volatile data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        volatile_data = pd.DataFrame({
            'open': 100 + np.random.randn(len(dates)) * 10,
            'high': 100 + np.abs(np.random.randn(len(dates))) * 20,
            'low': 100 - np.abs(np.random.randn(len(dates))) * 20,
            'close': 100 + np.random.randn(len(dates)) * 10,
            'volume': np.random.randint(100000, 10000000, len(dates))
        }, index=dates)
        volatile_data['high'] = volatile_data[['open', 'high', 'close']].max(axis=1)
        volatile_data['low'] = volatile_data[['open', 'low', 'close']].min(axis=1)
        
        st = SuperTrendAI()
        
        # Test with extreme parameters
        test_cases = [
            {'atr_length': 5, 'factor_min': 0.5, 'factor_max': 10, 'factor_step': 0.1},
            {'atr_length': 50, 'factor_min': 1, 'factor_max': 3, 'factor_step': 0.5},
            {'max_iter': 5000, 'max_data': 100}
        ]
        
        for params in test_cases:
            st_edge = SuperTrendAI(**params)
            
            start_time = time.time()
            result = st_edge.calculate(volatile_data)
            end_time = time.time()
            
            calc_time = end_time - start_time
            print(f"\nEdge case {params}:")
            print(f"  Calculation time: {calc_time:.3f} seconds")
            
            # Should handle edge cases reasonably
            assert calc_time < 5  # < 5 seconds even for edge cases
    
    def test_strategy_state_performance(self, medium_dataset):
        """Test strategy state management performance."""
        # Add required data
        medium_dataset['atr'] = 2.0  # Simplified
        
        strategy = SuperTrendAIStrategy()
        
        # Process data in streaming fashion
        chunk_size = 1000
        processing_times = []
        
        for i in range(0, len(medium_dataset) - chunk_size, chunk_size):
            chunk = medium_dataset[i:i+chunk_size]
            
            start_time = time.time()
            signals = strategy.generate_signals(chunk)
            end_time = time.time()
            
            processing_times.append(end_time - start_time)
        
        avg_time = np.mean(processing_times)
        std_time = np.std(processing_times)
        
        print(f"\nStrategy State Management:")
        print(f"  Average chunk processing: {avg_time*1000:.2f} ms")
        print(f"  Std deviation: {std_time*1000:.2f} ms")
        print(f"  Chunks per second: {1/avg_time:.0f}")
        
        # Should maintain consistent performance
        assert std_time < avg_time * 0.5  # Low variance
        assert avg_time < 0.1  # < 100ms per chunk


@pytest.mark.benchmark
class TestSuperTrendAIBenchmarks:
    """Comparative benchmarks against other indicators."""
    
    def test_vs_simple_supertrend(self, medium_dataset):
        """Compare performance vs simple SuperTrend."""
        from src.indicators.technical_indicators import SuperTrend
        
        # Our AI version
        st_ai = SuperTrendAI()
        start_time = time.time()
        result_ai = st_ai.calculate(medium_dataset)
        ai_time = time.time() - start_time
        
        # Simple version
        st_simple = SuperTrend(period=10, multiplier=3)
        start_time = time.time()
        result_simple = st_simple.calculate(medium_dataset)
        simple_time = time.time() - start_time
        
        overhead = (ai_time - simple_time) / simple_time
        
        print(f"\nSuperTrend AI vs Simple SuperTrend:")
        print(f"  Simple SuperTrend: {simple_time:.2f} seconds")
        print(f"  SuperTrend AI: {ai_time:.2f} seconds")
        print(f"  AI overhead: {overhead:.1%}")
        
        # AI version should not be too much slower
        assert overhead < 10  # Less than 10x overhead for AI features
    
    def test_vs_other_indicators(self, medium_dataset):
        """Compare performance against other complex indicators."""
        from src.indicators import RSI, BollingerBands, VWAP
        from src.indicators.technical_indicators import MACD
        
        indicators = {
            'SuperTrend AI': SuperTrendAI(),
            'RSI': RSI(period=14),
            'Bollinger Bands': BollingerBands(period=20),
            'MACD': MACD(),
            'VWAP': VWAP()
        }
        
        results = {}
        for name, indicator in indicators.items():
            start_time = time.time()
            _ = indicator.calculate(medium_dataset)
            calc_time = time.time() - start_time
            results[name] = calc_time
        
        print(f"\nIndicator Performance Comparison:")
        for name, calc_time in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {name}: {calc_time:.3f} seconds")
        
        # SuperTrend AI should be in reasonable range
        ai_time = results['SuperTrend AI']
        median_time = np.median(list(results.values()))
        assert ai_time < median_time * 5  # Not more than 5x median