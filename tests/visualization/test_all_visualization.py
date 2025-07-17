"""
Comprehensive test suite runner for all visualization modules.

This module runs all visualization tests and provides coverage reporting.
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import all test modules
from .test_charts import TestChartGenerator
from .test_dashboard import TestDashboard
from .test_export_utils import TestExportManager
from .test_comprehensive_trading_dashboard import TestComprehensiveTradingDashboard
from .test_performance_report import TestPerformanceAnalysisReport
from .test_trade_explorer import TestInteractiveTradeExplorer


def run_all_visualization_tests():
    """Run all visualization tests with coverage."""
    
    # Configure pytest arguments
    pytest_args = [
        '-v',  # Verbose output
        '--tb=short',  # Short traceback format
        '--cov=src.visualization',  # Coverage for visualization module
        '--cov-report=term-missing',  # Show missing lines
        '--cov-report=html:coverage_html/visualization',  # HTML report
        '--cov-report=json:coverage.json',  # JSON report
        '--cov-config=.coveragerc',  # Coverage configuration
        str(Path(__file__).parent),  # Test directory
    ]
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    return exit_code


def run_specific_test_class(test_class_name):
    """Run a specific test class."""
    
    pytest_args = [
        '-v',
        '--tb=short',
        '--cov=src.visualization',
        '--cov-report=term-missing',
        f'{Path(__file__).parent}::{test_class_name}',
    ]
    
    return pytest.main(pytest_args)


def run_test_by_module(module_name):
    """Run tests for a specific visualization module."""
    
    module_mapping = {
        'charts': 'test_charts.py',
        'dashboard': 'test_dashboard.py',
        'export': 'test_export_utils.py',
        'comprehensive': 'test_comprehensive_trading_dashboard.py',
        'performance': 'test_performance_report.py',
        'trade_explorer': 'test_trade_explorer.py',
    }
    
    if module_name not in module_mapping:
        print(f"Unknown module: {module_name}")
        print(f"Available modules: {', '.join(module_mapping.keys())}")
        return 1
    
    test_file = Path(__file__).parent / module_mapping[module_name]
    
    pytest_args = [
        '-v',
        '--tb=short',
        '--cov=src.visualization',
        '--cov-report=term-missing',
        str(test_file),
    ]
    
    return pytest.main(pytest_args)


def generate_coverage_report():
    """Generate detailed coverage report for visualization module."""
    
    print("Generating comprehensive coverage report for visualization module...")
    
    # Run all tests with detailed coverage
    pytest_args = [
        '-v',
        '--cov=src.visualization',
        '--cov-report=term-missing:skip-covered',
        '--cov-report=html:coverage_html/visualization',
        '--cov-report=json:coverage_visualization.json',
        '--cov-report=xml:coverage_visualization.xml',
        '--cov-branch',  # Branch coverage
        '--cov-context=test',  # Context for each test
        str(Path(__file__).parent),
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
        print("📊 Coverage reports generated:")
        print("   - HTML: coverage_html/visualization/index.html")
        print("   - JSON: coverage_visualization.json")
        print("   - XML: coverage_visualization.xml")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    return exit_code


def check_coverage_threshold(threshold=80):
    """Check if coverage meets the specified threshold."""
    
    import json
    
    # Run tests and generate coverage
    generate_coverage_report()
    
    # Read coverage report
    try:
        with open('coverage_visualization.json', 'r') as f:
            coverage_data = json.load(f)
        
        total_coverage = coverage_data['totals']['percent_covered']
        
        print(f"\n📊 Total Coverage: {total_coverage:.2f}%")
        print(f"📊 Threshold: {threshold}%")
        
        if total_coverage >= threshold:
            print(f"✅ Coverage meets threshold!")
            return True
        else:
            print(f"❌ Coverage below threshold!")
            return False
            
    except FileNotFoundError:
        print("❌ Coverage report not found!")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run visualization module tests')
    parser.add_argument(
        '--all', 
        action='store_true', 
        help='Run all visualization tests'
    )
    parser.add_argument(
        '--module', 
        type=str, 
        help='Run tests for specific module (charts, dashboard, export, etc.)'
    )
    parser.add_argument(
        '--class', 
        type=str, 
        dest='test_class',
        help='Run specific test class'
    )
    parser.add_argument(
        '--coverage', 
        action='store_true', 
        help='Generate detailed coverage report'
    )
    parser.add_argument(
        '--threshold', 
        type=int, 
        default=80,
        help='Coverage threshold percentage (default: 80)'
    )
    
    args = parser.parse_args()
    
    if args.coverage:
        exit_code = generate_coverage_report()
    elif args.all:
        exit_code = run_all_visualization_tests()
    elif args.module:
        exit_code = run_test_by_module(args.module)
    elif args.test_class:
        exit_code = run_specific_test_class(args.test_class)
    else:
        # Default: run all tests
        exit_code = run_all_visualization_tests()
    
    sys.exit(exit_code)