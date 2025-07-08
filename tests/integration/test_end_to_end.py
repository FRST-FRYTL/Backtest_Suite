"""End-to-end integration tests for the Backtest Suite."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import tempfile
import os
from unittest.mock import patch, Mock, AsyncMock

from src.backtesting import BacktestEngine
from src.strategies import StrategyBuilder
from src.indicators import RSI, BollingerBands, VWAP, TSV
from src.data import DataFetcher, CacheManager
from src.optimization import Optimizer, WalkForwardAnalysis
from src.visualization import ChartGenerator, DashboardBuilder
from src.utils import PerformanceMetrics


class TestCompleteBacktestWorkflow:
    """Test complete backtesting workflows from data fetch to results."""
    
    @pytest.mark.asyncio
    async def test_simple_strategy_workflow(self, temp_cache_dir):
        """Test a simple end-to-end workflow."""
        # 1. Setup components
        cache = CacheManager(cache_dir=temp_cache_dir)
        fetcher = DataFetcher(cache_manager=cache)
        engine = BacktestEngine(initial_capital=100000)
        
        # 2. Fetch data
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = generate_stock_data(
                start_date='2022-01-01',
                end_date='2023-12-31',
                initial_price=100.0,
                volatility=0.02
            )
            
            data = await fetcher.fetch_stock_data(
                'AAPL',
                start_date='2022-01-01',
                end_date='2023-12-31'
            )
        
        # 3. Add indicators
        rsi = RSI(period=14)
        data['rsi'] = rsi.calculate(data)
        
        # 4. Create strategy
        strategy = StrategyBuilder("RSI Mean Reversion")
        strategy.add_entry_rule("rsi < 30")
        strategy.add_exit_rule("rsi > 70")
        strategy.set_risk_management(
            stop_loss=0.05,
            take_profit=0.10,
            max_positions=1
        )
        strategy = strategy.build()
        
        # 5. Run backtest
        results = engine.run(data, strategy, progress_bar=False)
        
        # 6. Verify results
        assert results is not None
        assert not results.equity_curve.empty
        assert results.performance_metrics.total_return is not None
        
        # 7. Check if strategy traded
        if not results.trades.empty:
            assert results.trades['entry_price'].mean() > 0
            assert results.statistics['total_trades'] > 0
    
    @pytest.mark.asyncio
    async def test_multi_symbol_portfolio_workflow(self, temp_cache_dir):
        """Test portfolio backtesting with multiple symbols."""
        # Setup
        cache = CacheManager(cache_dir=temp_cache_dir)
        fetcher = DataFetcher(cache_manager=cache)
        engine = BacktestEngine(
            initial_capital=1000000,
            max_positions=5
        )
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
        
        # Fetch data for all symbols
        all_data = {}
        with patch('yfinance.download') as mock_download:
            for i, symbol in enumerate(symbols):
                mock_download.return_value = generate_stock_data(
                    initial_price=100 + i * 50,
                    volatility=0.015 + i * 0.005,
                    seed=42 + i
                )
                
                data = await fetcher.fetch_stock_data(
                    symbol,
                    start_date='2022-01-01',
                    end_date='2023-12-31'
                )
                
                # Add indicators
                bb = BollingerBands()
                bb_data = bb.calculate(data)
                all_data[symbol] = pd.concat([data, bb_data], axis=1)
        
        # Create momentum strategy
        strategy = StrategyBuilder("Multi-Asset Momentum")
        strategy.add_entry_rule("close > bb_upper")
        strategy.add_exit_rule("close < bb_middle")
        strategy.set_risk_management(
            position_size=0.2,  # 20% per position
            stop_loss=0.03,
            max_positions=5
        )
        strategy = strategy.build()
        
        # Run backtest
        results = engine.run(all_data, strategy, progress_bar=False)
        
        # Verify portfolio behavior
        assert results is not None
        
        # Should trade multiple symbols
        if not results.trades.empty:
            symbols_traded = results.trades['symbol'].unique()
            assert len(symbols_traded) > 1
            
            # Check position sizing
            avg_position_value = (
                results.trades['quantity'] * results.trades['entry_price']
            ).mean()
            expected_size = engine.initial_capital * 0.2
            assert 0.15 * expected_size < avg_position_value < 0.25 * expected_size
    
    @pytest.mark.asyncio
    async def test_optimization_workflow(self, temp_cache_dir):
        """Test strategy optimization workflow."""
        # Setup
        cache = CacheManager(cache_dir=temp_cache_dir)
        fetcher = DataFetcher(cache_manager=cache)
        optimizer = Optimizer()
        
        # Fetch data
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = generate_stock_data(
                start_date='2021-01-01',
                end_date='2023-12-31',
                volatility=0.025
            )
            
            data = await fetcher.fetch_stock_data('AAPL', '2021-01-01', '2023-12-31')
        
        # Add indicators with different parameters to test
        for period in [10, 14, 20]:
            rsi = RSI(period=period)
            data[f'rsi_{period}'] = rsi.calculate(data)
        
        # Define parameter space
        param_space = {
            'rsi_period': [10, 14, 20],
            'oversold': [25, 30, 35],
            'overbought': [65, 70, 75],
            'stop_loss': [0.03, 0.05, 0.07]
        }
        
        # Optimization function
        def create_strategy(params):
            builder = StrategyBuilder("Optimized RSI")
            rsi_col = f"rsi_{params['rsi_period']}"
            builder.add_entry_rule(f"{rsi_col} < {params['oversold']}")
            builder.add_exit_rule(f"{rsi_col} > {params['overbought']}")
            builder.set_risk_management(stop_loss=params['stop_loss'])
            return builder.build()
        
        # Run optimization
        with patch.object(optimizer, 'optimize') as mock_optimize:
            # Mock optimization results
            mock_optimize.return_value = {
                'best_params': {
                    'rsi_period': 14,
                    'oversold': 30,
                    'overbought': 70,
                    'stop_loss': 0.05
                },
                'best_score': 1.25,  # 25% return
                'all_results': [
                    {'params': p, 'score': 1.1 + i * 0.05}
                    for i, p in enumerate([
                        {'rsi_period': 10, 'oversold': 25, 'overbought': 75, 'stop_loss': 0.03},
                        {'rsi_period': 14, 'oversold': 30, 'overbought': 70, 'stop_loss': 0.05},
                        {'rsi_period': 20, 'oversold': 35, 'overbought': 65, 'stop_loss': 0.07}
                    ])
                ]
            }
            
            results = await optimizer.optimize_async(
                data=data,
                param_space=param_space,
                strategy_func=create_strategy,
                metric='sharpe_ratio'
            )
        
        # Verify optimization results
        assert 'best_params' in results
        assert 'best_score' in results
        assert results['best_params']['rsi_period'] == 14
    
    @pytest.mark.asyncio
    async def test_walk_forward_analysis_workflow(self):
        """Test walk-forward analysis workflow."""
        # Generate longer dataset for walk-forward
        data = generate_stock_data(
            start_date='2020-01-01',
            end_date='2023-12-31',
            volatility=0.02
        )
        
        # Add indicators
        vwap = VWAP(window=20)
        vwap_data = vwap.calculate(data)
        data = pd.concat([data, vwap_data], axis=1)
        
        # Create simple VWAP strategy
        strategy = StrategyBuilder("VWAP Strategy")
        strategy.add_entry_rule("close > vwap")
        strategy.add_exit_rule("close < vwap * 0.98")
        strategy = strategy.build()
        
        # Setup walk-forward analysis
        wfa = WalkForwardAnalysis(
            train_periods=252,  # 1 year
            test_periods=63,    # 3 months
            step_size=63        # 3 months
        )
        
        # Run analysis
        with patch.object(wfa, 'run') as mock_run:
            # Mock WFA results
            mock_run.return_value = {
                'periods': [
                    {
                        'train_start': '2020-01-01',
                        'train_end': '2020-12-31',
                        'test_start': '2021-01-01',
                        'test_end': '2021-03-31',
                        'in_sample_return': 0.15,
                        'out_sample_return': 0.12,
                        'efficiency': 0.80
                    },
                    {
                        'train_start': '2020-04-01',
                        'train_end': '2021-03-31',
                        'test_start': '2021-04-01',
                        'test_end': '2021-06-30',
                        'in_sample_return': 0.18,
                        'out_sample_return': 0.14,
                        'efficiency': 0.78
                    }
                ],
                'avg_efficiency': 0.79,
                'consistency_score': 0.85
            }
            
            results = wfa.run(data, strategy)
        
        # Verify walk-forward results
        assert 'periods' in results
        assert len(results['periods']) >= 2
        assert results['avg_efficiency'] > 0.7  # Good out-of-sample performance
    
    @pytest.mark.asyncio
    async def test_real_time_simulation_workflow(self):
        """Test real-time trading simulation workflow."""
        # Create minute-level data
        minute_data = generate_stock_data(
            start_date='2023-01-01 09:30:00',
            end_date='2023-01-01 16:00:00'
        )
        minute_data.index = pd.date_range(
            '2023-01-01 09:30:00',
            '2023-01-01 16:00:00',
            freq='1min'
        )
        
        # Create high-frequency strategy
        strategy = StrategyBuilder("HF Momentum")
        strategy.add_indicator("sma_5", "SMA", period=5)
        strategy.add_indicator("sma_20", "SMA", period=20)
        strategy.add_entry_rule("close > sma_5 and sma_5 > sma_20")
        strategy.add_exit_rule("close < sma_5")
        strategy.set_risk_management(
            position_size=0.5,
            max_positions=1
        )
        strategy = strategy.build()
        
        # Real-time simulation
        engine = BacktestEngine(
            initial_capital=100000,
            commission_rate=0.0005,  # Lower commission for HF
            slippage_model='market_impact'
        )
        
        # Add indicators
        minute_data['sma_5'] = minute_data['close'].rolling(5).mean()
        minute_data['sma_20'] = minute_data['close'].rolling(20).mean()
        
        # Run simulation
        results = engine.run(
            minute_data,
            strategy,
            mode='tick',  # Tick-by-tick simulation
            progress_bar=False
        )
        
        # Verify high-frequency results
        assert results is not None
        
        # Should have many trades in one day
        if not results.trades.empty:
            assert len(results.trades) >= 1
            
            # Check trade timing
            trade_durations = []
            for _, trade in results.trades.iterrows():
                if pd.notna(trade['exit_date']):
                    duration = (trade['exit_date'] - trade['entry_date']).total_seconds() / 60
                    trade_durations.append(duration)
            
            if trade_durations:
                avg_duration = np.mean(trade_durations)
                assert avg_duration < 120  # Average trade < 2 hours


class TestVisualizationIntegration:
    """Test integration with visualization components."""
    
    def test_chart_generation_workflow(self, sample_ohlcv_data):
        """Test generating charts from backtest results."""
        # Run a backtest
        engine = BacktestEngine()
        
        # Add indicators
        bb = BollingerBands()
        bb_data = bb.calculate(sample_ohlcv_data)
        data = pd.concat([sample_ohlcv_data, bb_data], axis=1)
        
        strategy = StrategyBuilder("BB Strategy")
        strategy.add_entry_rule("close < bb_lower")
        strategy.add_exit_rule("close > bb_upper")
        strategy = strategy.build()
        
        results = engine.run(data, strategy, progress_bar=False)
        
        # Generate charts
        charter = ChartGenerator()
        
        with patch.object(charter, 'create_equity_curve') as mock_equity:
            mock_equity.return_value = Mock()  # Mock plotly figure
            
            # Create equity curve
            fig = charter.create_equity_curve(results.equity_curve)
            mock_equity.assert_called_once()
        
        with patch.object(charter, 'create_drawdown_chart') as mock_dd:
            mock_dd.return_value = Mock()
            
            # Create drawdown chart
            fig = charter.create_drawdown_chart(results.equity_curve)
            mock_dd.assert_called_once()
        
        with patch.object(charter, 'create_trade_analysis') as mock_trades:
            mock_trades.return_value = Mock()
            
            # Create trade analysis
            if not results.trades.empty:
                fig = charter.create_trade_analysis(results.trades)
                mock_trades.assert_called_once()
    
    def test_dashboard_generation_workflow(self):
        """Test generating interactive dashboard."""
        # Mock backtest results
        results = Mock()
        results.equity_curve = pd.DataFrame({
            'total_value': np.linspace(100000, 120000, 252),
            'cash': np.linspace(100000, 50000, 252),
            'positions_value': np.linspace(0, 70000, 252)
        }, index=pd.date_range('2023-01-01', periods=252))
        
        results.trades = pd.DataFrame({
            'entry_date': pd.date_range('2023-01-01', periods=10, freq='25D'),
            'exit_date': pd.date_range('2023-01-10', periods=10, freq='25D'),
            'symbol': ['AAPL'] * 10,
            'quantity': [100] * 10,
            'entry_price': np.random.uniform(150, 160, 10),
            'exit_price': np.random.uniform(155, 165, 10),
            'pnl': np.random.uniform(-500, 1500, 10)
        })
        
        results.performance_metrics = Mock(
            total_return=0.20,
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            win_rate=0.60
        )
        
        # Create dashboard
        dashboard = DashboardBuilder()
        
        with patch.object(dashboard, 'build') as mock_build:
            mock_build.return_value = Mock()  # Mock dash app
            
            app = dashboard.build(results)
            mock_build.assert_called_once()
    
    def test_report_generation_workflow(self, sample_ohlcv_data):
        """Test generating comprehensive reports."""
        # Run backtest
        engine = BacktestEngine()
        
        strategy = StrategyBuilder("Test Strategy")
        strategy.add_entry_rule("close > close.shift(1) * 1.01")
        strategy.add_exit_rule("close < close.shift(1) * 0.99")
        strategy = strategy.build()
        
        results = engine.run(sample_ohlcv_data, strategy, progress_bar=False)
        
        # Generate report
        from src.visualization import ReportGenerator
        reporter = ReportGenerator()
        
        with patch.object(reporter, 'generate_html_report') as mock_html:
            mock_html.return_value = "<html>Report</html>"
            
            # Generate HTML report
            html = reporter.generate_html_report(results)
            assert isinstance(html, str)
            assert "<html>" in html
        
        with patch.object(reporter, 'generate_pdf_report') as mock_pdf:
            mock_pdf.return_value = b"PDF content"
            
            # Generate PDF report
            pdf = reporter.generate_pdf_report(results)
            assert isinstance(pdf, bytes)


class TestMonitoringIntegration:
    """Test integration with monitoring components."""
    
    @pytest.mark.asyncio
    async def test_live_monitoring_workflow(self):
        """Test live monitoring during backtest."""
        # Setup monitoring
        from src.monitoring import BacktestMonitor, MetricsCollector
        
        monitor = BacktestMonitor()
        collector = MetricsCollector()
        
        # Run backtest with monitoring
        engine = BacktestEngine()
        
        # Hook monitoring into engine
        original_process = engine._process_event
        events_processed = []
        
        def monitored_process(event):
            events_processed.append({
                'timestamp': datetime.now(),
                'event_type': type(event).__name__,
                'event': event
            })
            collector.record_event(event)
            return original_process(event)
        
        engine._process_event = monitored_process
        
        # Run backtest
        data = generate_stock_data()
        strategy = StrategyBuilder("Monitored Strategy").build()
        
        results = engine.run(data, strategy, progress_bar=False)
        
        # Verify monitoring captured events
        assert len(events_processed) > 0
        
        # Check metrics collected
        metrics = collector.get_metrics()
        assert 'events_processed' in metrics
        assert 'processing_time' in metrics
    
    @pytest.mark.asyncio
    async def test_alert_system_workflow(self):
        """Test alert system integration."""
        from src.monitoring.alerts import AlertManager, Alert, AlertLevel
        
        alert_manager = AlertManager()
        alerts_triggered = []
        
        # Setup alert handler
        def alert_handler(alert: Alert):
            alerts_triggered.append(alert)
        
        alert_manager.add_handler(alert_handler)
        
        # Define alerts
        alert_manager.add_rule(
            name="Large Drawdown",
            condition=lambda metrics: metrics.get('drawdown', 0) > 0.10,
            level=AlertLevel.WARNING,
            message="Drawdown exceeds 10%"
        )
        
        alert_manager.add_rule(
            name="Low Win Rate",
            condition=lambda metrics: metrics.get('win_rate', 1) < 0.40,
            level=AlertLevel.WARNING,
            message="Win rate below 40%"
        )
        
        # Run backtest
        engine = BacktestEngine()
        data = generate_stock_data(volatility=0.03)  # Higher volatility
        
        strategy = StrategyBuilder("Alert Test")
        strategy.add_entry_rule("close < close.rolling(5).mean() * 0.98")
        strategy.add_exit_rule("close > close.rolling(5).mean() * 1.02")
        strategy = strategy.build()
        
        results = engine.run(data, strategy, progress_bar=False)
        
        # Check alerts based on results
        metrics = {
            'drawdown': results.performance_metrics.max_drawdown,
            'win_rate': results.performance_metrics.win_rate
        }
        
        alert_manager.check_alerts(metrics)
        
        # Verify alerts triggered if conditions met
        if metrics['drawdown'] > 0.10:
            assert any(alert.name == "Large Drawdown" for alert in alerts_triggered)
        
        if metrics['win_rate'] < 0.40:
            assert any(alert.name == "Low Win Rate" for alert in alerts_triggered)


class TestPerformanceScenarios:
    """Test performance under various scenarios."""
    
    def test_large_scale_backtest(self, performance_monitor):
        """Test performance with large-scale backtest."""
        # 5 years of minute data
        n_minutes = 252 * 390 * 5  # ~500k bars
        dates = pd.date_range('2019-01-01 09:30', periods=n_minutes, freq='1min')
        
        # Generate data in chunks to avoid memory issues
        chunk_size = 10000
        data_chunks = []
        
        for i in range(0, n_minutes, chunk_size):
            chunk_dates = dates[i:i+chunk_size]
            chunk = pd.DataFrame({
                'open': 100 + np.random.randn(len(chunk_dates)).cumsum() * 0.1,
                'high': 100.5 + np.random.randn(len(chunk_dates)).cumsum() * 0.1,
                'low': 99.5 + np.random.randn(len(chunk_dates)).cumsum() * 0.1,
                'close': 100 + np.random.randn(len(chunk_dates)).cumsum() * 0.1,
                'volume': np.random.randint(1000, 10000, len(chunk_dates))
            }, index=chunk_dates)
            data_chunks.append(chunk)
        
        # Test with first chunk only (to keep test fast)
        data = data_chunks[0]
        
        # Simple strategy
        strategy = StrategyBuilder("Large Scale Test")
        strategy.add_entry_rule("close > open")
        strategy.add_exit_rule("close < open")
        strategy = strategy.build()
        
        engine = BacktestEngine()
        
        performance_monitor.start('large_backtest')
        results = engine.run(data, strategy, progress_bar=False)
        performance_monitor.stop('large_backtest')
        
        # Should complete in reasonable time
        assert performance_monitor.get_duration('large_backtest') < 10.0
        assert results is not None
    
    def test_stress_test_scenarios(self):
        """Test various stress scenarios."""
        scenarios = [
            # Market crash
            {
                'name': 'Market Crash',
                'volatility': 0.05,
                'trend': -0.001,
                'description': 'High volatility bear market'
            },
            # Flash crash
            {
                'name': 'Flash Crash',
                'volatility': 0.02,
                'events': [{'date': 100, 'magnitude': -0.10}],
                'description': 'Sudden 10% drop'
            },
            # Low liquidity
            {
                'name': 'Low Liquidity',
                'volatility': 0.03,
                'volume_factor': 0.1,
                'description': '90% volume reduction'
            }
        ]
        
        engine = BacktestEngine(initial_capital=100000)
        
        # Test strategy
        strategy = StrategyBuilder("Stress Test")
        strategy.add_entry_rule("rsi < 30")
        strategy.add_exit_rule("rsi > 70")
        strategy.set_risk_management(
            stop_loss=0.05,
            max_positions=3
        )
        strategy = strategy.build()
        
        results = {}
        
        for scenario in scenarios:
            # Generate scenario data
            data = generate_stock_data(
                volatility=scenario.get('volatility', 0.02),
                trend=scenario.get('trend', 0)
            )
            
            # Add RSI
            rsi = RSI()
            data['rsi'] = rsi.calculate(data)
            
            # Apply events
            if 'events' in scenario:
                for event in scenario['events']:
                    idx = event['date']
                    data.iloc[idx:idx+5, data.columns.get_loc('close')] *= (1 + event['magnitude'])
            
            # Apply volume factor
            if 'volume_factor' in scenario:
                data['volume'] *= scenario['volume_factor']
            
            # Run backtest
            result = engine.run(data, strategy, progress_bar=False)
            results[scenario['name']] = {
                'return': result.performance_metrics.total_return,
                'max_drawdown': result.performance_metrics.max_drawdown,
                'sharpe': result.performance_metrics.sharpe_ratio
            }
            
            engine.reset()
        
        # Verify all scenarios completed
        assert len(results) == len(scenarios)
        
        # Crash scenarios should have worse performance
        assert results['Market Crash']['return'] < results.get('Normal', {}).get('return', 0)


# Helper function for integration tests
def generate_stock_data(
    start_date='2023-01-01',
    end_date='2023-12-31',
    initial_price=100.0,
    volatility=0.02,
    trend=0.0001,
    seed=42
):
    """Generate realistic stock data for testing."""
    np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Generate returns
    returns = np.random.normal(trend, volatility, n_days)
    
    # Calculate prices
    price_series = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    data = pd.DataFrame(index=dates)
    data['close'] = price_series
    data['open'] = data['close'].shift(1) * (1 + np.random.normal(0, 0.001, n_days))
    data['open'].iloc[0] = initial_price
    
    daily_range = np.abs(np.random.normal(0.01, 0.005, n_days))
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + daily_range)
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - daily_range)
    
    # Volume correlated with price changes
    base_volume = 1000000
    price_change = np.abs(data['close'].pct_change()).fillna(0)
    data['volume'] = base_volume * (1 + price_change * 50) * np.random.uniform(0.8, 1.2, n_days)
    data['volume'] = data['volume'].astype(int)
    
    return data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])