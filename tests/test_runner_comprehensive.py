"""
Comprehensive test runner for the complete test suite.

This module provides utilities to run all tests systematically and generate
comprehensive test reports with coverage analysis.
"""

import pytest
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class ComprehensiveTestRunner:
    """Comprehensive test runner for the backtest suite."""
    
    def __init__(self, test_dir: str = None):
        """Initialize the test runner.
        
        Args:
            test_dir: Directory containing tests (defaults to current directory)
        """
        self.test_dir = Path(test_dir) if test_dir else Path(__file__).parent
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_test_category(self, category: str, verbose: bool = True) -> Dict[str, Any]:
        """Run tests for a specific category.
        
        Args:
            category: Test category to run
            verbose: Whether to show verbose output
            
        Returns:
            Test results dictionary
        """
        test_patterns = {
            'unit': 'test_*_unit.py',
            'integration': 'test_*_integration.py',
            'performance': 'test_*_performance.py',
            'comprehensive': 'test_*_comprehensive.py',
            'ml': 'test_ml_*.py',
            'visualization': 'test_visualization_*.py',
            'reporting': 'test_reporting/test_*.py',
            'coverage': 'coverage/test_*.py'
        }
        
        pattern = test_patterns.get(category, f'test_{category}*.py')
        test_files = list(self.test_dir.glob(pattern))
        
        if not test_files:
            return {
                'category': category,
                'status': 'no_tests',
                'message': f'No test files found for pattern: {pattern}'
            }
        
        # Run pytest with appropriate options
        pytest_args = [
            '-v' if verbose else '-q',
            '--tb=short',
            '--durations=10',
            '--strict-markers'
        ]
        
        # Add test files
        pytest_args.extend([str(f) for f in test_files])
        
        try:
            result = pytest.main(pytest_args)
            
            return {
                'category': category,
                'status': 'passed' if result == 0 else 'failed',
                'exit_code': result,
                'test_files': [str(f) for f in test_files],
                'file_count': len(test_files)
            }
        except Exception as e:
            return {
                'category': category,
                'status': 'error',
                'error': str(e),
                'test_files': [str(f) for f in test_files]
            }
    
    def run_all_tests(self, categories: List[str] = None, verbose: bool = True) -> Dict[str, Any]:
        """Run all test categories.
        
        Args:
            categories: List of categories to run (defaults to all)
            verbose: Whether to show verbose output
            
        Returns:
            Complete test results
        """
        if categories is None:
            categories = [
                'unit',
                'integration', 
                'performance',
                'comprehensive',
                'ml',
                'visualization',
                'reporting',
                'coverage'
            ]
        
        self.start_time = datetime.now()
        results = {}
        
        for category in categories:
            print(f"\n{'='*60}")
            print(f"Running {category.upper()} tests...")
            print(f"{'='*60}")
            
            category_result = self.run_test_category(category, verbose)
            results[category] = category_result
            
            # Print summary
            status = category_result['status']
            if status == 'passed':
                print(f"✅ {category.upper()} tests: PASSED")
            elif status == 'failed':
                print(f"❌ {category.upper()} tests: FAILED")
            elif status == 'error':
                print(f"💥 {category.upper()} tests: ERROR - {category_result.get('error', 'Unknown')}")
            else:
                print(f"⚠️  {category.upper()} tests: {status.upper()}")
        
        self.end_time = datetime.now()
        
        # Generate summary
        summary = self.generate_test_summary(results)
        
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration': str(self.end_time - self.start_time),
            'categories': results,
            'summary': summary
        }
    
    def generate_test_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of test results.
        
        Args:
            results: Test results by category
            
        Returns:
            Summary statistics
        """
        total_categories = len(results)
        passed_categories = sum(1 for r in results.values() if r['status'] == 'passed')
        failed_categories = sum(1 for r in results.values() if r['status'] == 'failed')
        error_categories = sum(1 for r in results.values() if r['status'] == 'error')
        no_test_categories = sum(1 for r in results.values() if r['status'] == 'no_tests')
        
        total_files = sum(r.get('file_count', 0) for r in results.values())
        
        return {
            'total_categories': total_categories,
            'passed_categories': passed_categories,
            'failed_categories': failed_categories,
            'error_categories': error_categories,
            'no_test_categories': no_test_categories,
            'total_test_files': total_files,
            'pass_rate': passed_categories / total_categories if total_categories > 0 else 0,
            'overall_status': 'passed' if failed_categories == 0 and error_categories == 0 else 'failed'
        }
    
    def run_coverage_analysis(self, coverage_report_dir: str = None) -> Dict[str, Any]:
        """Run coverage analysis on the test suite.
        
        Args:
            coverage_report_dir: Directory to store coverage reports
            
        Returns:
            Coverage analysis results
        """
        if coverage_report_dir is None:
            coverage_report_dir = self.test_dir.parent / 'htmlcov'
        
        coverage_report_dir = Path(coverage_report_dir)
        coverage_report_dir.mkdir(exist_ok=True)
        
        # Run coverage
        cmd = [
            sys.executable, '-m', 'coverage', 'run',
            '--source=src',
            '--omit=*/tests/*,*/test_*',
            '-m', 'pytest',
            str(self.test_dir)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.test_dir.parent)
            
            if result.returncode != 0:
                return {
                    'status': 'error',
                    'error': result.stderr,
                    'stdout': result.stdout
                }
            
            # Generate HTML report
            html_cmd = [
                sys.executable, '-m', 'coverage', 'html',
                '--directory', str(coverage_report_dir)
            ]
            
            html_result = subprocess.run(html_cmd, capture_output=True, text=True, cwd=self.test_dir.parent)
            
            # Generate JSON report for parsing
            json_cmd = [
                sys.executable, '-m', 'coverage', 'json',
                '--output', str(coverage_report_dir / 'coverage.json')
            ]
            
            json_result = subprocess.run(json_cmd, capture_output=True, text=True, cwd=self.test_dir.parent)
            
            # Parse coverage data
            coverage_data = self.parse_coverage_report(coverage_report_dir / 'coverage.json')
            
            return {
                'status': 'success',
                'html_report': str(coverage_report_dir / 'index.html'),
                'json_report': str(coverage_report_dir / 'coverage.json'),
                'coverage_data': coverage_data
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def parse_coverage_report(self, coverage_json_path: Path) -> Dict[str, Any]:
        """Parse coverage JSON report.
        
        Args:
            coverage_json_path: Path to coverage JSON file
            
        Returns:
            Parsed coverage data
        """
        try:
            with open(coverage_json_path, 'r') as f:
                coverage_data = json.load(f)
            
            # Extract summary
            totals = coverage_data.get('totals', {})
            
            # Extract file-level coverage
            files = coverage_data.get('files', {})
            
            # Calculate module-level coverage
            modules = {}
            for file_path, file_data in files.items():
                # Extract module name from file path
                if 'src/' in file_path:
                    module_parts = file_path.split('src/')[1].split('/')
                    module_name = module_parts[0] if module_parts else 'unknown'
                    
                    if module_name not in modules:
                        modules[module_name] = {
                            'num_statements': 0,
                            'covered_lines': 0,
                            'missing_lines': 0,
                            'files': []
                        }
                    
                    summary = file_data.get('summary', {})
                    modules[module_name]['num_statements'] += summary.get('num_statements', 0)
                    modules[module_name]['covered_lines'] += summary.get('covered_lines', 0)
                    modules[module_name]['missing_lines'] += summary.get('missing_lines', 0)
                    modules[module_name]['files'].append(file_path)
            
            # Calculate coverage percentages for modules
            for module_name, module_data in modules.items():
                if module_data['num_statements'] > 0:
                    module_data['coverage_percent'] = (
                        module_data['covered_lines'] / module_data['num_statements'] * 100
                    )
                else:
                    module_data['coverage_percent'] = 100.0
            
            return {
                'overall_coverage': totals.get('percent_covered', 0),
                'total_statements': totals.get('num_statements', 0),
                'covered_statements': totals.get('covered_lines', 0),
                'missing_statements': totals.get('missing_lines', 0),
                'module_coverage': modules,
                'file_coverage': files
            }
            
        except Exception as e:
            return {
                'error': f'Failed to parse coverage report: {str(e)}'
            }
    
    def generate_html_report(self, results: Dict[str, Any], output_path: str = None) -> str:
        """Generate HTML report of test results.
        
        Args:
            results: Test results from run_all_tests
            output_path: Path to save HTML report
            
        Returns:
            Path to generated HTML report
        """
        if output_path is None:
            output_path = self.test_dir / 'test_report.html'
        
        output_path = Path(output_path)
        
        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Suite Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .summary { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .category { margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }
                .category-header { background: #f8f8f8; padding: 10px; font-weight: bold; }
                .category-content { padding: 15px; }
                .passed { color: #008000; }
                .failed { color: #d00000; }
                .error { color: #ff8800; }
                .no-tests { color: #666; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧪 Backtest Suite Test Report</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Duration:</strong> {duration}</p>
            </div>
            
            <div class="summary">
                <h2>📊 Test Summary</h2>
                <div class="metric">
                    <strong>Total Categories:</strong> {total_categories}
                </div>
                <div class="metric">
                    <strong>Passed:</strong> <span class="passed">{passed_categories}</span>
                </div>
                <div class="metric">
                    <strong>Failed:</strong> <span class="failed">{failed_categories}</span>
                </div>
                <div class="metric">
                    <strong>Errors:</strong> <span class="error">{error_categories}</span>
                </div>
                <div class="metric">
                    <strong>Pass Rate:</strong> {pass_rate:.1%}
                </div>
                <div class="metric">
                    <strong>Total Files:</strong> {total_files}
                </div>
            </div>
            
            <h2>📋 Category Results</h2>
            {category_results}
            
            <div class="summary">
                <h2>🎯 Overall Status</h2>
                <p><strong>Result:</strong> <span class="{overall_status}">{overall_status_text}</span></p>
            </div>
        </body>
        </html>
        """
        
        # Generate category results HTML
        category_html = []
        for category, result in results['categories'].items():
            status = result['status']
            status_class = status.replace('_', '-')
            
            if status == 'passed':
                status_text = '✅ PASSED'
            elif status == 'failed':
                status_text = '❌ FAILED'
            elif status == 'error':
                status_text = '💥 ERROR'
            else:
                status_text = f'⚠️ {status.upper()}'
            
            test_files = result.get('test_files', [])
            file_list = '<br>'.join(test_files) if test_files else 'No test files'
            
            category_html.append(f"""
            <div class="category">
                <div class="category-header">
                    <span class="{status_class}">{status_text}</span> {category.upper()}
                </div>
                <div class="category-content">
                    <p><strong>Files:</strong> {result.get('file_count', 0)}</p>
                    <p><strong>Test Files:</strong><br>{file_list}</p>
                    {f'<p><strong>Error:</strong> {result.get("error", "")}</p>' if status == 'error' else ''}
                </div>
            </div>
            """)
        
        # Fill template
        summary = results['summary']
        html_content = html_template.format(
            timestamp=results['start_time'],
            duration=results['duration'],
            total_categories=summary['total_categories'],
            passed_categories=summary['passed_categories'],
            failed_categories=summary['failed_categories'],
            error_categories=summary['error_categories'],
            pass_rate=summary['pass_rate'],
            total_files=summary['total_test_files'],
            category_results=''.join(category_html),
            overall_status=summary['overall_status'],
            overall_status_text='✅ ALL TESTS PASSED' if summary['overall_status'] == 'passed' else '❌ SOME TESTS FAILED'
        )
        
        # Write HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def save_results_json(self, results: Dict[str, Any], output_path: str = None) -> str:
        """Save test results as JSON.
        
        Args:
            results: Test results from run_all_tests
            output_path: Path to save JSON file
            
        Returns:
            Path to saved JSON file
        """
        if output_path is None:
            output_path = self.test_dir / 'test_results.json'
        
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return str(output_path)


def main():
    """Main function to run comprehensive tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive test suite')
    parser.add_argument('--categories', nargs='*', help='Test categories to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--coverage', action='store_true', help='Run coverage analysis')
    parser.add_argument('--output-dir', help='Output directory for reports')
    parser.add_argument('--html-report', action='store_true', help='Generate HTML report')
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = ComprehensiveTestRunner()
    
    # Run tests
    print("🚀 Starting comprehensive test suite...")
    results = runner.run_all_tests(
        categories=args.categories,
        verbose=args.verbose
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    summary = results['summary']
    print(f"Total Categories: {summary['total_categories']}")
    print(f"Passed: {summary['passed_categories']}")
    print(f"Failed: {summary['failed_categories']}")
    print(f"Errors: {summary['error_categories']}")
    print(f"Pass Rate: {summary['pass_rate']:.1%}")
    print(f"Duration: {results['duration']}")
    
    # Run coverage analysis
    if args.coverage:
        print(f"\n{'='*60}")
        print("📈 COVERAGE ANALYSIS")
        print(f"{'='*60}")
        
        coverage_result = runner.run_coverage_analysis(
            coverage_report_dir=args.output_dir
        )
        
        if coverage_result['status'] == 'success':
            coverage_data = coverage_result['coverage_data']
            print(f"Overall Coverage: {coverage_data['overall_coverage']:.1f}%")
            print(f"Total Statements: {coverage_data['total_statements']}")
            print(f"Covered Statements: {coverage_data['covered_statements']}")
            print(f"HTML Report: {coverage_result['html_report']}")
        else:
            print(f"Coverage analysis failed: {coverage_result.get('error', 'Unknown error')}")
    
    # Generate HTML report
    if args.html_report:
        print(f"\n{'='*60}")
        print("📝 GENERATING HTML REPORT")
        print(f"{'='*60}")
        
        output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
        html_path = runner.generate_html_report(
            results,
            output_path=output_dir / 'test_report.html'
        )
        print(f"HTML Report: {html_path}")
        
        # Save JSON results
        json_path = runner.save_results_json(
            results,
            output_path=output_dir / 'test_results.json'
        )
        print(f"JSON Results: {json_path}")
    
    # Final status
    print(f"\n{'='*60}")
    if summary['overall_status'] == 'passed':
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    sys.exit(0 if summary['overall_status'] == 'passed' else 1)


if __name__ == '__main__':
    main()