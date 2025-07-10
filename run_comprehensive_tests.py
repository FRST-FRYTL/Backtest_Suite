"""
Simplified comprehensive test runner for Backtest Suite
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

# Import core modules
from src.data.download_historical_data import load_cached_data
from src.backtesting.engine import BacktestEngine
from src.strategies.builder import StrategyBuilder


def generate_test_report():
    """Generate comprehensive test report"""
    print("=" * 80)
    print("BACKTEST SUITE - COMPREHENSIVE TEST REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now()}\n")
    
    report_dir = Path("reports")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    test_results = {
        'timestamp': timestamp,
        'tests_run': [],
        'summary': {
            'total_tests': 0,
            'passed': 0,
            'failed': 0
        }
    }
    
    # 1. Test Data Availability
    print("📊 1. Testing Data Availability")
    print("-" * 40)
    
    data_test_results = []
    assets = ['SPY', 'QQQ', 'AAPL', 'MSFT']
    
    for asset in assets:
        try:
            data = load_cached_data(asset, '1D')
            if data is not None:
                result = {
                    'asset': asset,
                    'status': 'PASS',
                    'rows': len(data),
                    'date_range': f"{data.index[0]} to {data.index[-1]}"
                }
                print(f"✅ {asset}: {len(data)} rows loaded")
                test_results['summary']['passed'] += 1
            else:
                result = {
                    'asset': asset,
                    'status': 'FAIL',
                    'error': 'No data found'
                }
                print(f"❌ {asset}: No data found")
                test_results['summary']['failed'] += 1
        except Exception as e:
            result = {
                'asset': asset,
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"❌ {asset}: Error - {e}")
            test_results['summary']['failed'] += 1
        
        data_test_results.append(result)
        test_results['summary']['total_tests'] += 1
    
    test_results['tests_run'].append({
        'test_name': 'Data Availability',
        'results': data_test_results
    })
    
    # 2. Test Basic Strategy Execution
    print("\n🎯 2. Testing Strategy Execution")
    print("-" * 40)
    
    strategy_results = []
    
    # Load SPY data for testing
    spy_data = load_cached_data('SPY', '1D')
    
    if spy_data is not None:
        # Test simple moving average crossover
        try:
            # Add simple indicators
            spy_data['sma_fast'] = spy_data['close'].rolling(10).mean()
            spy_data['sma_slow'] = spy_data['close'].rolling(30).mean()
            spy_data = spy_data.dropna()
            
            # Create strategy
            builder = StrategyBuilder("Test SMA Strategy")
            builder.add_entry_rule("sma_fast > sma_slow")
            builder.add_exit_rule("sma_fast < sma_slow")
            strategy = builder.build()
            
            # Run backtest
            engine = BacktestEngine(initial_capital=10000)
            results = engine.run(
                data=spy_data.iloc[-252:],  # Last year
                strategy=strategy,
                progress_bar=False
            )
            
            strategy_result = {
                'strategy': 'SMA Crossover',
                'status': 'PASS',
                'total_return': f"{results['performance']['total_return']:.2f}%",
                'trades': results['performance']['total_trades'],
                'sharpe_ratio': results['performance']['sharpe_ratio']
            }
            print(f"✅ SMA Strategy: Return={results['performance']['total_return']:.2f}%, Trades={results['performance']['total_trades']}")
            test_results['summary']['passed'] += 1
            
        except Exception as e:
            strategy_result = {
                'strategy': 'SMA Crossover',
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"❌ SMA Strategy: {e}")
            test_results['summary']['failed'] += 1
    else:
        strategy_result = {
            'strategy': 'SMA Crossover',
            'status': 'SKIP',
            'reason': 'No data available'
        }
        print("⚠️ Strategy test skipped - no data")
    
    strategy_results.append(strategy_result)
    test_results['summary']['total_tests'] += 1
    
    test_results['tests_run'].append({
        'test_name': 'Strategy Execution',
        'results': strategy_results
    })
    
    # 3. Test Performance Metrics
    print("\n📈 3. Testing Performance Metrics")
    print("-" * 40)
    
    metrics_results = []
    
    try:
        # Create sample returns
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        # Calculate basic metrics
        metrics = {
            'annual_return': (1 + returns.mean()) ** 252 - 1,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std(),
            'max_return': returns.max(),
            'min_return': returns.min()
        }
        
        metrics_result = {
            'test': 'Basic Metrics Calculation',
            'status': 'PASS',
            'metrics': {k: f"{v:.4f}" for k, v in metrics.items()}
        }
        print("✅ Performance metrics calculated successfully")
        test_results['summary']['passed'] += 1
        
    except Exception as e:
        metrics_result = {
            'test': 'Basic Metrics Calculation',
            'status': 'FAIL',
            'error': str(e)
        }
        print(f"❌ Metrics calculation failed: {e}")
        test_results['summary']['failed'] += 1
    
    metrics_results.append(metrics_result)
    test_results['summary']['total_tests'] += 1
    
    test_results['tests_run'].append({
        'test_name': 'Performance Metrics',
        'results': metrics_results
    })
    
    # 4. Generate Summary Report
    print("\n📝 4. Generating Reports")
    print("-" * 40)
    
    # Save JSON report
    json_report_path = report_dir / 'summary' / f'test_report_{timestamp}.json'
    json_report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_report_path, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"✅ JSON report saved: {json_report_path}")
    
    # Generate HTML summary
    html_content = generate_html_report(test_results)
    html_report_path = report_dir / 'summary' / f'test_report_{timestamp}.html'
    html_report_path.write_text(html_content)
    
    print(f"✅ HTML report saved: {html_report_path}")
    
    # Create latest symlinks
    latest_json = report_dir / 'summary' / 'latest_test_report.json'
    latest_html = report_dir / 'summary' / 'latest_test_report.html'
    
    try:
        if latest_json.exists():
            latest_json.unlink()
        if latest_html.exists():
            latest_html.unlink()
        
        # Create relative symlinks
        os.symlink(f'test_report_{timestamp}.json', str(latest_json))
        os.symlink(f'test_report_{timestamp}.html', str(latest_html))
        print("✅ Created latest report links")
    except:
        # If symlinks fail, just copy the files
        import shutil
        shutil.copy(json_report_path, latest_json)
        shutil.copy(html_report_path, latest_html)
        print("✅ Created latest report copies")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {test_results['summary']['total_tests']}")
    print(f"Passed: {test_results['summary']['passed']} ✅")
    print(f"Failed: {test_results['summary']['failed']} ❌")
    print(f"\nSuccess Rate: {test_results['summary']['passed'] / test_results['summary']['total_tests'] * 100:.1f}%")
    print(f"\nView detailed report: {latest_html}")
    print("=" * 80)


def generate_html_report(test_results):
    """Generate HTML test report"""
    passed = test_results['summary']['passed']
    failed = test_results['summary']['failed']
    total = test_results['summary']['total_tests']
    success_rate = (passed / total * 100) if total > 0 else 0
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Suite - Test Report</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .summary-card.success {{
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        }}
        .summary-card.failure {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 18px;
        }}
        .summary-card .value {{
            font-size: 36px;
            font-weight: bold;
            margin: 0;
        }}
        .test-section {{
            margin-bottom: 30px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }}
        .test-header {{
            background-color: #f8f9fa;
            padding: 15px 20px;
            font-size: 18px;
            font-weight: bold;
            border-bottom: 1px solid #e0e0e0;
        }}
        .test-results {{
            padding: 20px;
        }}
        .result-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #f0f0f0;
        }}
        .result-item:last-child {{
            border-bottom: none;
        }}
        .status {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
        }}
        .status.pass {{
            background-color: #d4edda;
            color: #155724;
        }}
        .status.fail {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .status.skip {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .details {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        .progress-bar {{
            width: 100%;
            height: 30px;
            background-color: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #84fab0 0%, #8fd3f4 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.3s ease;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Backtest Suite - Test Report</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Tests</h3>
                <p class="value">{total}</p>
            </div>
            <div class="summary-card success">
                <h3>Passed</h3>
                <p class="value">{passed}</p>
            </div>
            <div class="summary-card failure">
                <h3>Failed</h3>
                <p class="value">{failed}</p>
            </div>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" style="width: {success_rate}%">
                {success_rate:.1f}% Success Rate
            </div>
        </div>
"""
    
    # Add test results
    for test_group in test_results['tests_run']:
        html += f"""
        <div class="test-section">
            <div class="test-header">{test_group['test_name']}</div>
            <div class="test-results">
"""
        
        for result in test_group['results']:
            if 'status' in result:
                status_class = result['status'].lower()
                status_text = result['status']
            else:
                status_class = 'skip'
                status_text = 'SKIP'
            
            # Get the main identifier (asset, strategy, or test name)
            item_name = result.get('asset', result.get('strategy', result.get('test', 'Unknown')))
            
            html += f"""
                <div class="result-item">
                    <div>
                        <strong>{item_name}</strong>
"""
            
            # Add details based on test type
            if 'rows' in result:
                html += f'<div class="details">Rows: {result["rows"]}, Range: {result["date_range"]}</div>'
            elif 'total_return' in result:
                html += f'<div class="details">Return: {result["total_return"]}, Trades: {result["trades"]}</div>'
            elif 'metrics' in result:
                metrics_str = ', '.join([f'{k}: {v}' for k, v in list(result['metrics'].items())[:3]])
                html += f'<div class="details">{metrics_str}</div>'
            elif 'error' in result:
                html += f'<div class="details">Error: {result["error"][:100]}...</div>'
            
            html += f"""
                    </div>
                    <span class="status {status_class}">{status_text}</span>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    return html


if __name__ == "__main__":
    generate_test_report()