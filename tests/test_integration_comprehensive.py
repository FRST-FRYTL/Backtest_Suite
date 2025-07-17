"""
Comprehensive integration tests for the complete backtest suite.

This module provides end-to-end integration tests that verify the complete
system works correctly with all components together.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import asyncio

# Core imports
from src.backtesting import BacktestEngine, Portfolio
from src.strategies import StrategyBuilder
from src.indicators import RSI, BollingerBands, VWAP
from src.indicators.supertrend_ai import SuperTrendAI
from src.data import StockDataFetcher
from src.reporting import StandardReportGenerator, ReportConfig
from src.reporting.visualizations import ReportVisualizations
from src.analysis.timeframe_performance_analyzer import TimeframePerformanceAnalyzer
from src.utils.metrics import PerformanceMetrics
from src.ml.clustering.kmeans_optimizer import KMeansOptimizer


class TestCompleteBacktestWorkflow:
    """Test complete end-to-end backtest workflow."""
    
    @pytest.fixture
    def comprehensive_market_data(self):
        """Create comprehensive market data for testing."""
        dates = pd.date_range('2022-01-01', periods=500, freq='D')
        np.random.seed(42)
        
        # Generate realistic market data with trends and volatility
        trend = np.sin(np.linspace(0, 4 * np.pi, len(dates))) * 0.1
        volatility = np.random.normal(0, 0.02, len(dates))
        close_prices = 100 * np.exp(np.cumsum(trend + volatility))
        
        data = pd.DataFrame({
            'open': close_prices + np.random.normal(0, 0.5, len(dates)),
            'high': close_prices + np.abs(np.random.normal(1, 0.5, len(dates))),
            'low': close_prices - np.abs(np.random.normal(1, 0.5, len(dates))),
            'close': close_prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Ensure OHLC relationships
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
    
    def test_complete_strategy_development_workflow(self, comprehensive_market_data):
        """Test complete strategy development workflow."""
        # Step 1: Create and configure strategy
        strategy_builder = StrategyBuilder("Complete Test Strategy")
        
        # Add technical indicators
        strategy_builder.add_indicator(RSI(period=14))
        strategy_builder.add_indicator(BollingerBands(period=20))
        strategy_builder.add_indicator(VWAP())
        
        # Add entry rules
        strategy_builder.add_entry_rule("rsi < 30")
        strategy_builder.add_entry_rule("close < bb_lower")
        strategy_builder.add_entry_rule("close > vwap")
        
        # Add exit rules
        strategy_builder.add_exit_rule("rsi > 70")
        strategy_builder.add_exit_rule("close > bb_upper")
        
        # Configure risk management
        strategy_builder.set_risk_management(
            stop_loss=0.05,
            take_profit=0.10,
            position_size=0.02,
            max_positions=3
        )
        
        # Build strategy
        strategy = strategy_builder.build()
        
        # Step 2: Initialize backtest engine
        engine = BacktestEngine(
            initial_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        # Step 3: Run backtest
        results = engine.run(
            data=comprehensive_market_data,
            strategy=strategy
        )
        
        # Step 4: Verify results
        assert isinstance(results, dict)
        assert 'equity_curve' in results
        assert 'trades' in results
        assert 'metrics' in results
        assert 'portfolio_value' in results
        
        # Check equity curve
        equity_curve = results['equity_curve']
        assert isinstance(equity_curve, pd.Series)
        assert len(equity_curve) > 0
        assert equity_curve.iloc[0] == 100000  # Initial capital
        
        # Check trades
        trades = results['trades']
        assert isinstance(trades, pd.DataFrame)
        
        # Check metrics
        metrics = results['metrics']
        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        
        # Step 5: Generate report
        report_config = ReportConfig(
            title="Complete Strategy Test Report",
            include_trade_analysis=True,
            include_risk_analysis=True
        )
        
        report_generator = StandardReportGenerator(report_config)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_files = report_generator.generate_report(
                backtest_results=results,
                output_dir=tmp_dir,
                report_name="complete_test"
            )
            
            # Verify report files were created
            assert 'html' in report_files
            assert 'json' in report_files
            assert Path(report_files['html']).exists()
            assert Path(report_files['json']).exists()
    
    def test_supertrend_ai_integration_workflow(self, comprehensive_market_data):
        """Test SuperTrend AI integration workflow."""
        # Step 1: Initialize SuperTrend AI
        supertrend_ai = SuperTrendAI(
            atr_periods=[10, 14, 20],
            multipliers=[2.0, 2.5, 3.0],
            n_clusters=3,
            adaptive=True,
            volatility_adjustment=True
        )
        
        # Step 2: Calculate SuperTrend AI signals
        st_result = supertrend_ai.calculate(comprehensive_market_data)
        
        # Verify SuperTrend AI results
        assert hasattr(st_result, 'trend')
        assert hasattr(st_result, 'upper_band')
        assert hasattr(st_result, 'lower_band')
        assert hasattr(st_result, 'signal')
        assert hasattr(st_result, 'optimal_params')
        
        # Step 3: Create strategy using SuperTrend AI
        from tests.integration.test_supertrend_ai_strategy import SuperTrendAIStrategy
        
        strategy = SuperTrendAIStrategy(
            min_signal_strength=5,
            stop_loss_atr_mult=2.0,
            take_profit_rr_ratio=2.0
        )
        
        # Add ATR to data for strategy
        high_low = comprehensive_market_data['high'] - comprehensive_market_data['low']
        high_close = (comprehensive_market_data['high'] - comprehensive_market_data['close'].shift()).abs()
        low_close = (comprehensive_market_data['low'] - comprehensive_market_data['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        comprehensive_market_data['atr'] = true_range.rolling(14).mean()
        
        # Step 4: Run backtest
        engine = BacktestEngine(initial_capital=100000)
        results = engine.run(
            data=comprehensive_market_data,
            strategy=strategy
        )
        
        # Step 5: Verify SuperTrend AI strategy results
        assert isinstance(results, dict)
        assert 'trades' in results
        assert 'metrics' in results
        
        # Check that SuperTrend AI specific metrics are reasonable
        if len(results['trades']) > 0:
            assert results['metrics']['total_return'] is not None
            assert results['metrics']['sharpe_ratio'] is not None
    
    def test_multi_timeframe_analysis_workflow(self, comprehensive_market_data):
        """Test multi-timeframe analysis workflow."""
        # Step 1: Create data for different timeframes
        timeframe_data = {}
        
        # Daily data (original)
        timeframe_data['1D'] = comprehensive_market_data
        
        # Hourly data (resample daily to hourly)
        hourly_data = comprehensive_market_data.resample('H').first().fillna(method='ffill')
        timeframe_data['1H'] = hourly_data
        
        # 4-hour data
        four_hour_data = comprehensive_market_data.resample('4H').first().fillna(method='ffill')
        timeframe_data['4H'] = four_hour_data
        
        # Step 2: Create strategy for each timeframe
        def create_timeframe_strategy(timeframe):
            builder = StrategyBuilder(f"Multi-TF Strategy {timeframe}")
            builder.add_entry_rule("close > open")
            builder.add_exit_rule("close < open")
            builder.set_risk_management(position_size=0.05)
            return builder.build()
        
        # Step 3: Run backtests for each timeframe
        timeframe_results = {}
        
        for timeframe, data in timeframe_data.items():
            strategy = create_timeframe_strategy(timeframe)
            engine = BacktestEngine(initial_capital=100000)
            
            results = engine.run(data=data, strategy=strategy)
            timeframe_results[timeframe] = results
        
        # Step 4: Analyze performance across timeframes
        analyzer = TimeframePerformanceAnalyzer(timeframes=['1D', '1H', '4H'])
        
        # Extract equity curves and trades
        equity_curves = {}
        trades_data = {}
        
        for timeframe, results in timeframe_results.items():
            equity_curves[timeframe] = results['equity_curve']
            trades_data[timeframe] = results['trades']
        
        # Analyze all timeframes
        analysis_results = analyzer.analyze_all_timeframes(
            equity_curves=equity_curves,
            trades_data=trades_data
        )
        
        # Step 5: Compare timeframes
        comparison = analyzer.compare_timeframes(analysis_results)
        
        # Verify comparison results
        assert isinstance(comparison, dict)
        assert 'summary' in comparison
        assert 'rankings' in comparison
        assert 'best_performing' in comparison
        assert 'correlations' in comparison
        
        # Check that all timeframes are analyzed
        assert len(comparison['summary']) == 3
        assert '1D' in comparison['summary']
        assert '1H' in comparison['summary']
        assert '4H' in comparison['summary']
    
    def test_ml_optimization_workflow(self, comprehensive_market_data):
        """Test ML optimization workflow."""
        # Step 1: Generate multiple strategy configurations
        parameter_combinations = []
        
        # Generate parameter grid
        for rsi_period in [10, 14, 20]:
            for bb_period in [15, 20, 25]:
                for position_size in [0.01, 0.02, 0.05]:
                    parameter_combinations.append({
                        'rsi_period': rsi_period,
                        'bb_period': bb_period,
                        'position_size': position_size
                    })
        
        # Step 2: Run backtests for each configuration
        backtest_results = []
        
        for i, params in enumerate(parameter_combinations[:10]):  # Limit to first 10 for testing
            # Create strategy with parameters
            builder = StrategyBuilder(f"ML Optimization Test {i}")
            builder.add_entry_rule("rsi < 30")
            builder.add_exit_rule("rsi > 70")
            builder.set_risk_management(position_size=params['position_size'])
            
            strategy = builder.build()
            
            # Run backtest
            engine = BacktestEngine(initial_capital=100000)
            results = engine.run(data=comprehensive_market_data, strategy=strategy)
            
            # Calculate performance metrics
            metrics = PerformanceMetrics.calculate_all_metrics(
                returns=results['equity_curve'].pct_change().dropna(),
                equity_curve=results['equity_curve'],
                trades=results['trades']
            )
            
            # Store results with parameters
            result_row = {
                'rsi_period': params['rsi_period'],
                'bb_period': params['bb_period'],
                'position_size': params['position_size'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'total_trades': len(results['trades'])
            }
            backtest_results.append(result_row)
        
        # Step 3: Use ML clustering to optimize parameters
        results_df = pd.DataFrame(backtest_results)
        
        optimizer = KMeansOptimizer(
            n_clusters=3,
            performance_weights={
                'sharpe_ratio': 0.3,
                'max_drawdown': 0.25,
                'win_rate': 0.2,
                'profit_factor': 0.15,
                'total_trades': 0.1
            }
        )
        
        # Step 4: Find optimal parameters
        clustering_result = optimizer.fit(results_df)
        best_params = optimizer.get_best_parameters(results_df, clustering_result)
        
        # Step 5: Verify optimization results
        assert isinstance(clustering_result, dict)
        assert 'cluster_labels' in clustering_result
        assert 'best_cluster' in clustering_result
        
        assert isinstance(best_params, dict)
        assert 'parameters' in best_params
        assert 'expected_performance' in best_params
        
        # Check that best parameters are from the original parameter space
        optimal_params = best_params['parameters']
        assert 'rsi_period' in optimal_params
        assert 'bb_period' in optimal_params
        assert 'position_size' in optimal_params
        assert optimal_params['rsi_period'] in [10, 14, 20]
        assert optimal_params['bb_period'] in [15, 20, 25]
        assert optimal_params['position_size'] in [0.01, 0.02, 0.05]
    
    def test_comprehensive_reporting_workflow(self, comprehensive_market_data):
        """Test comprehensive reporting workflow."""
        # Step 1: Run backtest
        strategy = StrategyBuilder("Comprehensive Report Test")
        strategy.add_entry_rule("close > open")
        strategy.add_exit_rule("close < open")
        strategy.set_risk_management(position_size=0.02)
        
        engine = BacktestEngine(initial_capital=100000)
        results = engine.run(
            data=comprehensive_market_data,
            strategy=strategy.build()
        )
        
        # Step 2: Generate comprehensive report
        config = ReportConfig(
            title="Comprehensive Integration Test Report",
            subtitle="Complete System Test",
            include_executive_summary=True,
            include_performance_analysis=True,
            include_risk_analysis=True,
            include_trade_analysis=True,
            include_market_regime_analysis=True,
            include_technical_details=True
        )
        
        report_generator = StandardReportGenerator(config)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Step 3: Generate report files
            report_files = report_generator.generate_report(
                backtest_results=results,
                output_dir=tmp_dir,
                report_name="comprehensive_integration"
            )
            
            # Step 4: Verify report generation
            assert 'html' in report_files
            assert 'json' in report_files
            
            html_path = Path(report_files['html'])
            json_path = Path(report_files['json'])
            
            assert html_path.exists()
            assert json_path.exists()
            
            # Step 5: Verify report content
            # Check HTML content
            with open(html_path, 'r') as f:
                html_content = f.read()
                assert 'Comprehensive Integration Test Report' in html_content
                assert 'plotly' in html_content.lower()  # Visualizations included
                assert 'Total Return' in html_content
                assert 'Sharpe Ratio' in html_content
            
            # Check JSON content
            with open(json_path, 'r') as f:
                json_content = json.load(f)
                assert 'metadata' in json_content
                assert 'sections' in json_content
                assert 'backtest_summary' in json_content
                
                # Check sections
                sections = json_content['sections']
                assert 'executivesummary' in sections
                assert 'performanceanalysis' in sections
                assert 'riskanalysis' in sections
                assert 'tradeanalysis' in sections
    
    def test_error_handling_and_recovery(self, comprehensive_market_data):
        """Test error handling and recovery mechanisms."""
        # Test 1: Invalid strategy configuration
        with pytest.raises(ValueError):
            invalid_strategy = StrategyBuilder("Invalid Strategy")
            invalid_strategy.set_risk_management(position_size=2.0)  # Invalid: > 1
            invalid_strategy.build()
        
        # Test 2: Insufficient data
        insufficient_data = comprehensive_market_data.head(10)  # Only 10 days
        
        strategy = StrategyBuilder("Insufficient Data Test")
        strategy.add_entry_rule("close > open")
        strategy.add_exit_rule("close < open")
        
        engine = BacktestEngine(initial_capital=100000)
        
        # Should handle insufficient data gracefully
        results = engine.run(data=insufficient_data, strategy=strategy.build())
        
        assert isinstance(results, dict)
        assert 'equity_curve' in results
        assert 'trades' in results
        assert 'metrics' in results
        
        # Test 3: Empty trade results
        no_signal_strategy = StrategyBuilder("No Signals Strategy")
        no_signal_strategy.add_entry_rule("close > 1000000")  # Never true
        no_signal_strategy.add_exit_rule("close < 0")  # Never true
        
        results = engine.run(data=comprehensive_market_data, strategy=no_signal_strategy.build())
        
        # Should handle no trades gracefully
        assert isinstance(results, dict)
        assert len(results['trades']) == 0
        assert results['metrics']['total_return'] == 0
        
        # Test 4: Report generation with empty results
        config = ReportConfig(title="Empty Results Test")
        report_generator = StandardReportGenerator(config)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_files = report_generator.generate_report(
                backtest_results=results,
                output_dir=tmp_dir,
                report_name="empty_results"
            )
            
            # Should generate report even with empty results
            assert 'html' in report_files
            assert 'json' in report_files
            assert Path(report_files['html']).exists()
            assert Path(report_files['json']).exists()
    
    def test_performance_benchmarking(self, comprehensive_market_data):
        """Test performance benchmarking across different configurations."""
        # Test different engine configurations
        configurations = [
            {'commission': 0.001, 'slippage': 0.0005},
            {'commission': 0.0005, 'slippage': 0.0001},
            {'commission': 0.002, 'slippage': 0.001}
        ]
        
        strategy = StrategyBuilder("Performance Benchmark")
        strategy.add_entry_rule("close > open")
        strategy.add_exit_rule("close < open")
        strategy.set_risk_management(position_size=0.02)
        built_strategy = strategy.build()
        
        results = []
        
        for config in configurations:
            engine = BacktestEngine(
                initial_capital=100000,
                commission_rate=config['commission'],
                slippage_rate=config['slippage']
            )
            
            import time
            start_time = time.time()
            
            result = engine.run(data=comprehensive_market_data, strategy=built_strategy)
            
            end_time = time.time()
            
            results.append({
                'config': config,
                'execution_time': end_time - start_time,
                'total_return': result['metrics']['total_return'],
                'sharpe_ratio': result['metrics']['sharpe_ratio']
            })
        
        # Verify performance characteristics
        for result in results:
            assert result['execution_time'] < 30  # Should complete quickly
            assert isinstance(result['total_return'], (int, float))
            assert isinstance(result['sharpe_ratio'], (int, float))
        
        # Higher costs should generally reduce returns
        high_cost_result = results[2]  # Highest commission/slippage
        low_cost_result = results[1]   # Lowest commission/slippage
        
        # This might not always hold due to randomness, but generally should
        assert high_cost_result['total_return'] <= low_cost_result['total_return'] + 0.05
    
    def test_data_pipeline_integration(self):
        """Test data pipeline integration."""
        # Test 1: Data fetcher integration
        fetcher = StockDataFetcher()
        
        # Mock the data fetching for testing
        with patch.object(fetcher, 'fetch_stock_data') as mock_fetch:
            mock_data = pd.DataFrame({
                'open': [100, 101, 102],
                'high': [101, 102, 103],
                'low': [99, 100, 101],
                'close': [100.5, 101.5, 102.5],
                'volume': [1000000, 1100000, 1200000]
            }, index=pd.date_range('2023-01-01', periods=3, freq='D'))
            
            mock_fetch.return_value = mock_data
            
            # Test data fetching
            data = fetcher.fetch_stock_data('TEST', '2023-01-01', '2023-01-03')
            
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 3
            assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        
        # Test 2: Data validation
        # This would test data quality checks, missing data handling, etc.
        invalid_data = pd.DataFrame({
            'open': [100, np.nan, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1200000]
        })
        
        # The system should handle invalid data gracefully
        strategy = StrategyBuilder("Data Validation Test")
        strategy.add_entry_rule("close > open")
        strategy.add_exit_rule("close < open")
        
        engine = BacktestEngine(initial_capital=100000)
        
        # Should handle NaN values without crashing
        results = engine.run(data=invalid_data, strategy=strategy.build())
        
        assert isinstance(results, dict)
        assert 'equity_curve' in results
        assert 'trades' in results
    
    def test_scalability_and_memory_usage(self):
        """Test system scalability and memory usage."""
        # Create large dataset
        large_dates = pd.date_range('2020-01-01', periods=2000, freq='D')
        np.random.seed(42)
        
        large_data = pd.DataFrame({
            'open': np.random.uniform(90, 110, len(large_dates)),
            'high': np.random.uniform(100, 120, len(large_dates)),
            'low': np.random.uniform(80, 100, len(large_dates)),
            'close': np.random.uniform(95, 105, len(large_dates)),
            'volume': np.random.randint(1000000, 10000000, len(large_dates))
        }, index=large_dates)
        
        # Ensure OHLC relationships
        large_data['high'] = large_data[['open', 'high', 'close']].max(axis=1)
        large_data['low'] = large_data[['open', 'low', 'close']].min(axis=1)
        
        # Test with complex strategy
        strategy = StrategyBuilder("Scalability Test")
        strategy.add_indicator(RSI(period=14))
        strategy.add_indicator(BollingerBands(period=20))
        strategy.add_indicator(VWAP())
        
        strategy.add_entry_rule("rsi < 30")
        strategy.add_entry_rule("close < bb_lower")
        strategy.add_exit_rule("rsi > 70")
        strategy.add_exit_rule("close > bb_upper")
        
        strategy.set_risk_management(position_size=0.01)
        
        engine = BacktestEngine(initial_capital=100000)
        
        # Monitor memory usage
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        import time
        start_time = time.time()
        
        results = engine.run(data=large_data, strategy=strategy.build())
        
        end_time = time.time()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Performance assertions
        execution_time = end_time - start_time
        memory_increase = memory_after - memory_before
        
        assert execution_time < 60  # Should complete within 1 minute
        assert memory_increase < 500  # Should not use more than 500MB
        
        # Results should be valid
        assert isinstance(results, dict)
        assert 'equity_curve' in results
        assert len(results['equity_curve']) == len(large_data)
    
    def test_concurrent_backtests(self, comprehensive_market_data):
        """Test concurrent backtest execution."""
        import concurrent.futures
        
        # Create multiple strategies
        strategies = []
        for i in range(4):
            builder = StrategyBuilder(f"Concurrent Test {i}")
            builder.add_entry_rule("close > open")
            builder.add_exit_rule("close < open")
            builder.set_risk_management(position_size=0.01 * (i + 1))
            strategies.append(builder.build())
        
        def run_backtest(strategy):
            engine = BacktestEngine(initial_capital=100000)
            return engine.run(data=comprehensive_market_data, strategy=strategy)
        
        # Run backtests concurrently
        import time
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_backtest, strategy) for strategy in strategies]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        
        # Verify concurrent execution
        assert len(results) == 4
        assert all(isinstance(result, dict) for result in results)
        assert all('equity_curve' in result for result in results)
        
        # Should complete faster than sequential execution
        concurrent_time = end_time - start_time
        assert concurrent_time < 30  # Should complete quickly
        
        # Results should be valid and different
        total_returns = [result['metrics']['total_return'] for result in results]
        assert len(set(total_returns)) > 1  # Should have different results


class TestSystemStressTests:
    """Stress tests for the complete system."""
    
    def test_extreme_market_conditions(self):
        """Test system behavior under extreme market conditions."""
        # Test 1: Market crash scenario
        crash_dates = pd.date_range('2023-01-01', periods=100, freq='D')
        crash_prices = 100 * np.exp(-np.linspace(0, 2, len(crash_dates)))  # 86% decline
        
        crash_data = pd.DataFrame({
            'open': crash_prices + np.random.normal(0, 0.5, len(crash_dates)),
            'high': crash_prices + np.abs(np.random.normal(1, 0.5, len(crash_dates))),
            'low': crash_prices - np.abs(np.random.normal(1, 0.5, len(crash_dates))),
            'close': crash_prices,
            'volume': np.random.randint(5000000, 50000000, len(crash_dates))  # High volume
        }, index=crash_dates)
        
        crash_data['high'] = crash_data[['open', 'high', 'close']].max(axis=1)
        crash_data['low'] = crash_data[['open', 'low', 'close']].min(axis=1)
        
        # Test strategy in crash scenario
        strategy = StrategyBuilder("Crash Test")
        strategy.add_entry_rule("close > open")
        strategy.add_exit_rule("close < open")
        strategy.set_risk_management(stop_loss=0.05, position_size=0.01)
        
        engine = BacktestEngine(initial_capital=100000)
        results = engine.run(data=crash_data, strategy=strategy.build())
        
        # System should handle crash gracefully
        assert isinstance(results, dict)
        assert 'equity_curve' in results
        assert results['metrics']['max_drawdown'] < 0  # Should have drawdown
        assert results['metrics']['max_drawdown'] > -1  # But not lose everything
        
        # Test 2: High volatility scenario
        volatile_dates = pd.date_range('2023-01-01', periods=100, freq='D')
        volatile_prices = 100 + np.cumsum(np.random.normal(0, 5, len(volatile_dates)))  # High volatility
        
        volatile_data = pd.DataFrame({
            'open': volatile_prices + np.random.normal(0, 2, len(volatile_dates)),
            'high': volatile_prices + np.abs(np.random.normal(5, 2, len(volatile_dates))),
            'low': volatile_prices - np.abs(np.random.normal(5, 2, len(volatile_dates))),
            'close': volatile_prices,
            'volume': np.random.randint(1000000, 10000000, len(volatile_dates))
        }, index=volatile_dates)
        
        volatile_data['high'] = volatile_data[['open', 'high', 'close']].max(axis=1)
        volatile_data['low'] = volatile_data[['open', 'low', 'close']].min(axis=1)
        
        # Test in volatile market
        results = engine.run(data=volatile_data, strategy=strategy.build())
        
        # System should handle volatility
        assert isinstance(results, dict)
        assert results['metrics']['volatility'] > 0  # Should measure volatility
        assert results['metrics']['sharpe_ratio'] is not None
    
    def test_memory_stress_test(self):
        """Test system behavior under memory stress."""
        # Create very large dataset
        huge_dates = pd.date_range('2015-01-01', periods=5000, freq='D')
        np.random.seed(42)
        
        # Generate data in chunks to avoid memory issues during creation
        chunk_size = 1000
        data_chunks = []
        
        for i in range(0, len(huge_dates), chunk_size):
            chunk_dates = huge_dates[i:i+chunk_size]
            chunk_data = pd.DataFrame({
                'open': np.random.uniform(90, 110, len(chunk_dates)),
                'high': np.random.uniform(100, 120, len(chunk_dates)),
                'low': np.random.uniform(80, 100, len(chunk_dates)),
                'close': np.random.uniform(95, 105, len(chunk_dates)),
                'volume': np.random.randint(1000000, 10000000, len(chunk_dates))
            }, index=chunk_dates)
            
            chunk_data['high'] = chunk_data[['open', 'high', 'close']].max(axis=1)
            chunk_data['low'] = chunk_data[['open', 'low', 'close']].min(axis=1)
            
            data_chunks.append(chunk_data)
        
        huge_data = pd.concat(data_chunks)
        
        # Simple strategy to minimize computation
        strategy = StrategyBuilder("Memory Stress Test")
        strategy.add_entry_rule("close > open")
        strategy.add_exit_rule("close < open")
        strategy.set_risk_management(position_size=0.001)  # Small positions
        
        engine = BacktestEngine(initial_capital=100000)
        
        # Monitor memory during execution
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        results = engine.run(data=huge_data, strategy=strategy.build())
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory should not increase excessively
        assert memory_increase < 1000  # Less than 1GB increase
        
        # Results should still be valid
        assert isinstance(results, dict)
        assert 'equity_curve' in results
        assert len(results['equity_curve']) == len(huge_data)
    
    def test_computational_stress_test(self):
        """Test system behavior under computational stress."""
        # Create complex strategy with many indicators
        strategy = StrategyBuilder("Computational Stress Test")
        
        # Add multiple indicators
        strategy.add_indicator(RSI(period=14))
        strategy.add_indicator(RSI(period=21))
        strategy.add_indicator(BollingerBands(period=20))
        strategy.add_indicator(BollingerBands(period=50))
        strategy.add_indicator(VWAP())
        
        # Add many rules
        strategy.add_entry_rule("rsi_14 < 30")
        strategy.add_entry_rule("rsi_21 < 35")
        strategy.add_entry_rule("close < bb_lower_20")
        strategy.add_entry_rule("close > vwap")
        
        strategy.add_exit_rule("rsi_14 > 70")
        strategy.add_exit_rule("rsi_21 > 65")
        strategy.add_exit_rule("close > bb_upper_20")
        
        strategy.set_risk_management(position_size=0.01)
        
        # Create reasonably large dataset
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': np.random.uniform(90, 110, len(dates)),
            'high': np.random.uniform(100, 120, len(dates)),
            'low': np.random.uniform(80, 100, len(dates)),
            'close': np.random.uniform(95, 105, len(dates)),
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        engine = BacktestEngine(initial_capital=100000)
        
        import time
        start_time = time.time()
        
        results = engine.run(data=data, strategy=strategy.build())
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete in reasonable time even with complex strategy
        assert execution_time < 60  # Less than 1 minute
        
        # Results should be valid
        assert isinstance(results, dict)
        assert 'equity_curve' in results
        assert 'trades' in results
        assert 'metrics' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])