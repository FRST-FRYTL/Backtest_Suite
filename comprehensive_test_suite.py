"""
Comprehensive Test Suite for Backtest Suite
Tests all major components and generates detailed reports
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import all major components
from src.data.download_historical_data import load_cached_data, MarketDataDownloader
from src.backtesting.engine import BacktestEngine
from src.strategies.builder import StrategyBuilder
from src.indicators.technical_indicators import SMA, RSI, BollingerBands, ATR, MACD
from src.utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown
from src.visualization.charts import create_candlestick_chart


class ComprehensiveTestSuite:
    """Run comprehensive tests on all components"""
    
    def __init__(self):
        self.test_results = {}
        self.report_dir = Path("reports")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_all_tests(self):
        """Run all test suites"""
        print("=" * 80)
        print("COMPREHENSIVE BACKTEST SUITE TESTING")
        print("=" * 80)
        print(f"Started at: {datetime.now()}\n")
        
        # 1. Data Quality Tests
        self.test_data_quality()
        
        # 2. Indicator Tests
        self.test_indicators()
        
        # 3. Strategy Tests
        self.test_strategies()
        
        # 4. Performance Tests
        self.test_performance_metrics()
        
        # 5. Backtest Engine Tests
        self.test_backtest_engine()
        
        # 6. Generate Summary Report
        self.generate_summary_report()
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETE")
        print("=" * 80)
        print(f"Finished at: {datetime.now()}")
        print(f"\nReports saved to: {self.report_dir}")
        
    def test_data_quality(self):
        """Test data quality and completeness"""
        print("\n📊 Testing Data Quality...")
        print("-" * 40)
        
        results = {
            'test_name': 'Data Quality',
            'timestamp': datetime.now().isoformat(),
            'assets_tested': [],
            'issues_found': [],
            'summary': {}
        }
        
        # Test each asset
        assets = ['SPY', 'QQQ', 'AAPL', 'MSFT']
        timeframes = ['1D', '1W', '1M']
        
        for asset in assets:
            asset_results = {
                'symbol': asset,
                'timeframes': {}
            }
            
            for tf in timeframes:
                print(f"Testing {asset} - {tf}...")
                data = load_cached_data(asset, tf)
                
                if data is not None:
                    # Check for missing data
                    missing_count = data.isnull().sum().sum()
                    
                    # Check for data anomalies
                    anomalies = self._check_data_anomalies(data)
                    
                    # Check date continuity
                    gaps = self._check_date_gaps(data)
                    
                    asset_results['timeframes'][tf] = {
                        'rows': len(data),
                        'date_range': f"{data.index[0]} to {data.index[-1]}",
                        'missing_values': int(missing_count),
                        'anomalies': len(anomalies),
                        'date_gaps': len(gaps),
                        'quality_score': self._calculate_quality_score(data, missing_count, anomalies, gaps)
                    }
                else:
                    asset_results['timeframes'][tf] = {
                        'error': 'Data not found'
                    }
            
            results['assets_tested'].append(asset_results)
        
        # Save report
        self._save_report(results, 'data_quality', f"data_quality_test_{self.timestamp}.json")
        self.test_results['data_quality'] = results
        
        print("✅ Data Quality Test Complete")
        
    def test_indicators(self):
        """Test all technical indicators"""
        print("\n📈 Testing Technical Indicators...")
        print("-" * 40)
        
        results = {
            'test_name': 'Technical Indicators',
            'timestamp': datetime.now().isoformat(),
            'indicators_tested': [],
            'performance_metrics': {}
        }
        
        # Load test data
        data = load_cached_data('SPY', '1D')
        if data is None:
            print("❌ No data available for indicator testing")
            return
        
        # Test each indicator
        indicators = [
            ('SMA', SMA(period=20)),
            ('RSI', RSI(period=14)),
            ('Bollinger Bands', BollingerBands(period=20, std_dev=2)),
            ('ATR', ATR(period=14)),
            ('MACD', MACD(fast_period=12, slow_period=26, signal_period=9))
        ]
        
        for name, indicator in indicators:
            print(f"Testing {name}...")
            start_time = time.time()
            
            try:
                # Calculate indicator
                result = indicator.calculate(data)
                calc_time = time.time() - start_time
                
                # Validate results
                validation = self._validate_indicator_output(result, name)
                
                indicator_result = {
                    'name': name,
                    'calculation_time': calc_time,
                    'output_columns': list(result.columns),
                    'valid_values': len(result.dropna()),
                    'validation': validation,
                    'status': 'success'
                }
                
            except Exception as e:
                indicator_result = {
                    'name': name,
                    'status': 'error',
                    'error': str(e)
                }
            
            results['indicators_tested'].append(indicator_result)
        
        # Save report
        self._save_report(results, 'indicators', f"indicator_test_{self.timestamp}.json")
        self.test_results['indicators'] = results
        
        print("✅ Indicator Test Complete")
        
    def test_strategies(self):
        """Test various trading strategies"""
        print("\n🎯 Testing Trading Strategies...")
        print("-" * 40)
        
        results = {
            'test_name': 'Strategy Testing',
            'timestamp': datetime.now().isoformat(),
            'strategies_tested': []
        }
        
        # Load test data
        data = load_cached_data('SPY', '1D')
        if data is None:
            print("❌ No data available for strategy testing")
            return
        
        # Test different strategies
        strategies = [
            self._create_sma_crossover_strategy(),
            self._create_rsi_strategy(),
            self._create_bollinger_strategy()
        ]
        
        engine = BacktestEngine(
            initial_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        for strategy in strategies:
            print(f"Testing {strategy.name}...")
            
            try:
                # Run backtest
                backtest_results = engine.run(
                    data=data.iloc[-252:],  # Last year
                    strategy=strategy,
                    progress_bar=False
                )
                
                strategy_result = {
                    'name': strategy.name,
                    'status': 'success',
                    'performance': backtest_results['performance'],
                    'trade_count': len(backtest_results.get('trades', [])),
                    'final_equity': backtest_results['equity_curve'].iloc[-1] if 'equity_curve' in backtest_results else None
                }
                
            except Exception as e:
                strategy_result = {
                    'name': strategy.name,
                    'status': 'error',
                    'error': str(e)
                }
            
            results['strategies_tested'].append(strategy_result)
        
        # Save report
        self._save_report(results, 'strategies', f"strategy_test_{self.timestamp}.json")
        self.test_results['strategies'] = results
        
        print("✅ Strategy Test Complete")
        
    def test_performance_metrics(self):
        """Test performance calculation functions"""
        print("\n📊 Testing Performance Metrics...")
        print("-" * 40)
        
        results = {
            'test_name': 'Performance Metrics',
            'timestamp': datetime.now().isoformat(),
            'metrics_tested': []
        }
        
        # Create sample returns data
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        equity = (1 + returns).cumprod() * 100000
        
        # Test each metric
        metrics = [
            ('Sharpe Ratio', lambda: calculate_sharpe_ratio(returns)),
            ('Max Drawdown', lambda: calculate_max_drawdown(equity)),
            ('Volatility', lambda: returns.std() * np.sqrt(252)),
            ('Annual Return', lambda: (1 + returns.mean()) ** 252 - 1),
            ('Calmar Ratio', lambda: (((1 + returns.mean()) ** 252 - 1) / abs(calculate_max_drawdown(equity))))
        ]
        
        for name, calc_func in metrics:
            print(f"Testing {name}...")
            
            try:
                result = calc_func()
                metric_result = {
                    'name': name,
                    'status': 'success',
                    'value': float(result) if not isinstance(result, (list, dict)) else result
                }
            except Exception as e:
                metric_result = {
                    'name': name,
                    'status': 'error',
                    'error': str(e)
                }
            
            results['metrics_tested'].append(metric_result)
        
        # Save report
        self._save_report(results, 'performance', f"performance_metrics_test_{self.timestamp}.json")
        self.test_results['performance_metrics'] = results
        
        print("✅ Performance Metrics Test Complete")
        
    def test_backtest_engine(self):
        """Test the backtest engine with various scenarios"""
        print("\n⚙️ Testing Backtest Engine...")
        print("-" * 40)
        
        results = {
            'test_name': 'Backtest Engine',
            'timestamp': datetime.now().isoformat(),
            'scenarios_tested': []
        }
        
        # Load test data
        data = load_cached_data('SPY', '1D')
        if data is None:
            print("❌ No data available for backtest engine testing")
            return
        
        # Test scenarios
        scenarios = [
            {
                'name': 'High Frequency',
                'initial_capital': 10000,
                'commission_rate': 0.0001,
                'max_positions': 10
            },
            {
                'name': 'Conservative',
                'initial_capital': 100000,
                'commission_rate': 0.005,
                'max_positions': 1
            },
            {
                'name': 'No Commission',
                'initial_capital': 50000,
                'commission_rate': 0,
                'max_positions': 5
            }
        ]
        
        strategy = self._create_simple_strategy()
        
        for scenario in scenarios:
            print(f"Testing scenario: {scenario['name']}...")
            
            try:
                engine = BacktestEngine(
                    initial_capital=scenario['initial_capital'],
                    commission_rate=scenario['commission_rate'],
                    max_positions=scenario['max_positions']
                )
                
                result = engine.run(
                    data=data.iloc[-100:],  # Last 100 days
                    strategy=strategy,
                    progress_bar=False
                )
                
                scenario_result = {
                    'scenario': scenario,
                    'status': 'success',
                    'final_equity': result['equity_curve'].iloc[-1],
                    'total_return': result['performance']['total_return'],
                    'trades': result['performance']['total_trades']
                }
                
            except Exception as e:
                scenario_result = {
                    'scenario': scenario,
                    'status': 'error',
                    'error': str(e)
                }
            
            results['scenarios_tested'].append(scenario_result)
        
        # Save report
        self._save_report(results, 'backtest', f"backtest_engine_test_{self.timestamp}.json")
        self.test_results['backtest_engine'] = results
        
        print("✅ Backtest Engine Test Complete")
        
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n📝 Generating Summary Report...")
        print("-" * 40)
        
        summary = {
            'test_suite': 'Comprehensive Backtest Suite Test',
            'timestamp': self.timestamp,
            'total_tests': len(self.test_results),
            'test_summary': {}
        }
        
        # Summarize each test
        for test_name, results in self.test_results.items():
            if isinstance(results, dict):
                summary['test_summary'][test_name] = {
                    'status': 'completed',
                    'items_tested': len(results.get('assets_tested', results.get('indicators_tested', results.get('strategies_tested', []))))
                }
        
        # Create HTML summary
        html_content = self._generate_html_summary(summary)
        
        # Save reports
        self._save_report(summary, 'summary', f"test_summary_{self.timestamp}.json")
        
        html_path = self.report_dir / 'summary' / f"test_summary_{self.timestamp}.html"
        html_path.write_text(html_content)
        
        # Create latest symlinks
        latest_json = self.report_dir / 'summary' / 'latest_summary.json'
        latest_html = self.report_dir / 'summary' / 'latest_summary.html'
        
        if latest_json.exists():
            latest_json.unlink()
        if latest_html.exists():
            latest_html.unlink()
            
        latest_json.symlink_to(f"test_summary_{self.timestamp}.json")
        latest_html.symlink_to(f"test_summary_{self.timestamp}.html")
        
        print("✅ Summary Report Generated")
        print(f"📄 View report at: {latest_html}")
        
    # Helper methods
    def _check_data_anomalies(self, data):
        """Check for data anomalies"""
        anomalies = []
        
        # Check for extreme price movements (>20% in a day)
        if 'close' in data.columns:
            returns = data['close'].pct_change()
            extreme_moves = returns[abs(returns) > 0.2]
            anomalies.extend(extreme_moves.index.tolist())
        
        # Check for zero volume
        if 'volume' in data.columns:
            zero_volume = data[data['volume'] == 0]
            anomalies.extend(zero_volume.index.tolist())
        
        return anomalies
    
    def _check_date_gaps(self, data):
        """Check for gaps in dates"""
        gaps = []
        
        # Check for missing trading days
        date_diff = data.index.to_series().diff()
        
        # Assuming daily data, gaps > 3 days are suspicious (weekends + holiday)
        large_gaps = date_diff[date_diff > pd.Timedelta(days=3)]
        gaps = large_gaps.index.tolist()
        
        return gaps
    
    def _calculate_quality_score(self, data, missing_count, anomalies, gaps):
        """Calculate data quality score (0-100)"""
        total_issues = missing_count + len(anomalies) + len(gaps)
        total_points = len(data) * len(data.columns)
        
        score = max(0, 100 - (total_issues / total_points * 100))
        return round(score, 2)
    
    def _validate_indicator_output(self, result, indicator_name):
        """Validate indicator output"""
        validation = {
            'has_data': not result.empty,
            'no_infinities': not result.isin([np.inf, -np.inf]).any().any(),
            'reasonable_values': True
        }
        
        # Check for reasonable values based on indicator type
        if indicator_name == 'RSI' and 'rsi' in result.columns:
            validation['reasonable_values'] = result['rsi'].dropna().between(0, 100).all()
        
        return validation
    
    def _create_sma_crossover_strategy(self):
        """Create SMA crossover strategy"""
        builder = StrategyBuilder("SMA Crossover")
        builder.add_entry_rule("sma_20 > sma_50")
        builder.add_exit_rule("sma_20 < sma_50")
        return builder.build()
    
    def _create_rsi_strategy(self):
        """Create RSI strategy"""
        builder = StrategyBuilder("RSI Mean Reversion")
        builder.add_entry_rule("rsi < 30")
        builder.add_exit_rule("rsi > 70")
        return builder.build()
    
    def _create_bollinger_strategy(self):
        """Create Bollinger Bands strategy"""
        builder = StrategyBuilder("Bollinger Breakout")
        builder.add_entry_rule("close > bb_upper")
        builder.add_exit_rule("close < bb_middle")
        return builder.build()
    
    def _create_simple_strategy(self):
        """Create simple test strategy"""
        builder = StrategyBuilder("Simple Test Strategy")
        builder.add_entry_rule("close > open")
        builder.add_exit_rule("close < open")
        return builder.build()
    
    def _save_report(self, data, category, filename):
        """Save report to appropriate directory"""
        filepath = self.report_dir / category / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _generate_html_summary(self, summary):
        """Generate HTML summary report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Suite Test Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; text-align: center; }}
        .summary-box {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .test-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .success {{ color: #28a745; }}
        .error {{ color: #dc3545; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Backtest Suite Test Summary</h1>
        <div class="summary-box">
            <h2>Test Overview</h2>
            <p><strong>Test Suite:</strong> {summary['test_suite']}</p>
            <p><strong>Timestamp:</strong> <span class="timestamp">{summary['timestamp']}</span></p>
            <p><strong>Total Tests:</strong> {summary['total_tests']}</p>
        </div>
        
        <h2>Test Results</h2>
"""
        
        for test_name, test_info in summary['test_summary'].items():
            status_class = 'success' if test_info['status'] == 'completed' else 'error'
            html += f"""
        <div class="test-section">
            <h3>{test_name.replace('_', ' ').title()}</h3>
            <p>Status: <span class="{status_class}">{test_info['status'].upper()}</span></p>
            <p>Items Tested: {test_info['items_tested']}</p>
        </div>
"""
        
        html += """
        <div class="summary-box">
            <h3>📁 Report Locations</h3>
            <ul>
                <li>Data Quality: <a href="../data_quality/">reports/data_quality/</a></li>
                <li>Indicators: <a href="../indicators/">reports/indicators/</a></li>
                <li>Strategies: <a href="../strategies/">reports/strategies/</a></li>
                <li>Performance: <a href="../performance/">reports/performance/</a></li>
                <li>Backtest Engine: <a href="../backtest/">reports/backtest/</a></li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    # Run comprehensive test suite
    test_suite = ComprehensiveTestSuite()
    test_suite.run_all_tests()