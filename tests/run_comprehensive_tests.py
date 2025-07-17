#!/usr/bin/env python3
"""
Comprehensive test runner for the Backtest Suite.

This script runs all test modules and provides detailed reporting
on test coverage, performance, and quality metrics.
"""

import subprocess
import sys
import time
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import argparse


class ComprehensiveTestRunner:
    """Comprehensive test runner with detailed reporting."""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    def run_test_module(self, module_name: str, verbose: bool = False) -> Dict[str, Any]:
        """Run a specific test module and capture results."""
        print(f"\n{'='*60}")
        print(f"Running {module_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Prepare pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            f"tests/{module_name}",
            "--tb=short",
            "--durations=10",
            f"--json-report={self.output_dir}/{module_name}_report.json",
            "--json-report-summary"
        ]
        
        if verbose:
            cmd.append("-v")
        
        # Run the test
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per module
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse JSON report if available
            json_report_path = self.output_dir / f"{module_name}_report.json"
            test_details = {}
            if json_report_path.exists():
                try:
                    with open(json_report_path, 'r') as f:
                        test_details = json.load(f)
                except json.JSONDecodeError:
                    test_details = {}
            
            return {
                'module': module_name,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'duration': duration,
                'passed': result.returncode == 0,
                'details': test_details
            }
            
        except subprocess.TimeoutExpired:
            return {
                'module': module_name,
                'returncode': -1,
                'stdout': "",
                'stderr': "Test timed out after 5 minutes",
                'duration': 300,
                'passed': False,
                'details': {}
            }
        except Exception as e:
            return {
                'module': module_name,
                'returncode': -1,
                'stdout': "",
                'stderr': str(e),
                'duration': 0,
                'passed': False,
                'details': {}
            }
    
    def run_all_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run all test modules."""
        print("Starting comprehensive test suite...")
        self.start_time = time.time()
        
        # Define test modules in order of execution
        test_modules = [
            # Core functionality tests
            "test_supertrend_ai.py",
            "test_supertrend_ai_performance.py",
            
            # Integration tests
            "integration/test_supertrend_ai_strategy.py",
            
            # Reporting tests
            "test_reporting/test_standard_report_generator.py",
            "test_reporting/test_enhanced_trade_reporting.py",
            "test_reporting/test_visualization_types.py",
            
            # New comprehensive tests
            "test_ml_integration.py",
            "test_visualization_comprehensive.py",
            "test_portfolio_risk_management.py",
            
            # Coverage tests
            "coverage/test_technical_indicators_comprehensive.py",
            "coverage/test_backtesting_engine_comprehensive.py",
            "coverage/test_strategy_framework_comprehensive.py",
            "coverage/test_portfolio_management_comprehensive.py",
            
            # Additional tests
            "test_backtesting.py",
            "test_strategies.py",
            "test_indicators.py",
            "test_portfolio.py",
            "test_data.py",
            "test_feature_engineering.py",
            "test_performance_benchmarks.py"
        ]
        
        results = {}
        
        for module in test_modules:
            if Path(f"tests/{module}").exists():
                result = self.run_test_module(module, verbose)
                results[module] = result
                
                # Print immediate summary
                status = "PASSED" if result['passed'] else "FAILED"
                print(f"{module}: {status} ({result['duration']:.2f}s)")
                
                if not result['passed']:
                    print(f"  Error: {result['stderr']}")
            else:
                print(f"Skipping {module} - file not found")
        
        self.end_time = time.time()
        self.test_results = results
        
        return results
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report."""
        total_duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
        
        passed_tests = sum(1 for r in self.test_results.values() if r['passed'])
        failed_tests = len(self.test_results) - passed_tests
        
        # Extract detailed metrics from JSON reports
        total_test_cases = 0
        total_assertions = 0
        coverage_data = {}
        performance_metrics = {}
        
        for module, result in self.test_results.items():
            details = result.get('details', {})
            
            if 'summary' in details:
                summary = details['summary']
                total_test_cases += summary.get('total', 0)
                
            # Extract performance data
            if 'duration' in result:
                performance_metrics[module] = {
                    'duration': result['duration'],
                    'tests_per_second': total_test_cases / result['duration'] if result['duration'] > 0 else 0
                }
        
        summary = {
            'execution_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
                'total_duration': total_duration,
                'test_modules': len(self.test_results)
            },
            'test_results': {
                'passed_modules': passed_tests,
                'failed_modules': failed_tests,
                'success_rate': passed_tests / len(self.test_results) if self.test_results else 0,
                'total_test_cases': total_test_cases
            },
            'performance_metrics': performance_metrics,
            'module_details': {}
        }
        
        # Add detailed module information
        for module, result in self.test_results.items():
            details = result.get('details', {})
            module_summary = {
                'passed': result['passed'],
                'duration': result['duration'],
                'returncode': result['returncode']
            }
            
            if 'summary' in details:
                module_summary.update(details['summary'])
            
            summary['module_details'][module] = module_summary
        
        return summary
    
    def save_reports(self, summary: Dict[str, Any]):
        """Save detailed reports to files."""
        # Save summary report
        summary_path = self.output_dir / "test_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        detailed_path = self.output_dir / "detailed_results.json"
        with open(detailed_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Generate HTML report
        self.generate_html_report(summary)
        
        # Generate markdown report
        self.generate_markdown_report(summary)
    
    def generate_html_report(self, summary: Dict[str, Any]):
        """Generate HTML test report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Suite - Test Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ text-align: center; padding: 10px; background-color: #e9ecef; border-radius: 5px; }}
        .metric h3 {{ margin: 0; color: #333; }}
        .metric p {{ margin: 5px 0; font-size: 24px; font-weight: bold; }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .module-results {{ margin: 20px 0; }}
        .module {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007bff; background-color: #f8f9fa; }}
        .module.passed {{ border-left-color: #28a745; }}
        .module.failed {{ border-left-color: #dc3545; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Backtest Suite - Test Results</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>Total Modules</h3>
            <p>{summary['execution_info']['test_modules']}</p>
        </div>
        <div class="metric">
            <h3>Passed</h3>
            <p class="passed">{summary['test_results']['passed_modules']}</p>
        </div>
        <div class="metric">
            <h3>Failed</h3>
            <p class="failed">{summary['test_results']['failed_modules']}</p>
        </div>
        <div class="metric">
            <h3>Success Rate</h3>
            <p>{summary['test_results']['success_rate']:.1%}</p>
        </div>
        <div class="metric">
            <h3>Total Duration</h3>
            <p>{summary['execution_info']['total_duration']:.2f}s</p>
        </div>
    </div>
    
    <h2>Module Results</h2>
    <div class="module-results">
"""
        
        for module, details in summary['module_details'].items():
            status_class = "passed" if details['passed'] else "failed"
            status_text = "PASSED" if details['passed'] else "FAILED"
            
            html_content += f"""
        <div class="module {status_class}">
            <h3>{module} - {status_text}</h3>
            <p>Duration: {details['duration']:.2f}s</p>
            <p>Return Code: {details['returncode']}</p>
        </div>
"""
        
        html_content += """
    </div>
    
    <h2>Performance Metrics</h2>
    <table>
        <tr>
            <th>Module</th>
            <th>Duration (s)</th>
            <th>Tests/Second</th>
        </tr>
"""
        
        for module, metrics in summary['performance_metrics'].items():
            html_content += f"""
        <tr>
            <td>{module}</td>
            <td>{metrics['duration']:.2f}</td>
            <td>{metrics['tests_per_second']:.2f}</td>
        </tr>
"""
        
        html_content += """
    </table>
</body>
</html>
"""
        
        html_path = self.output_dir / "test_report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
    
    def generate_markdown_report(self, summary: Dict[str, Any]):
        """Generate Markdown test report."""
        markdown_content = f"""# Backtest Suite - Test Results

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| Total Modules | {summary['execution_info']['test_modules']} |
| Passed Modules | {summary['test_results']['passed_modules']} |
| Failed Modules | {summary['test_results']['failed_modules']} |
| Success Rate | {summary['test_results']['success_rate']:.1%} |
| Total Duration | {summary['execution_info']['total_duration']:.2f}s |

## Module Results

"""
        
        for module, details in summary['module_details'].items():
            status_emoji = "✅" if details['passed'] else "❌"
            status_text = "PASSED" if details['passed'] else "FAILED"
            
            markdown_content += f"""### {status_emoji} {module} - {status_text}

- **Duration**: {details['duration']:.2f}s
- **Return Code**: {details['returncode']}

"""
        
        markdown_content += """## Performance Metrics

| Module | Duration (s) | Tests/Second |
|--------|--------------|--------------|
"""
        
        for module, metrics in summary['performance_metrics'].items():
            markdown_content += f"| {module} | {metrics['duration']:.2f} | {metrics['tests_per_second']:.2f} |\n"
        
        markdown_path = self.output_dir / "test_report.md"
        with open(markdown_path, 'w') as f:
            f.write(markdown_content)
    
    def print_final_summary(self, summary: Dict[str, Any]):
        """Print final summary to console."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST SUITE SUMMARY")
        print("="*80)
        
        print(f"Total Modules: {summary['execution_info']['test_modules']}")
        print(f"Passed: {summary['test_results']['passed_modules']}")
        print(f"Failed: {summary['test_results']['failed_modules']}")
        print(f"Success Rate: {summary['test_results']['success_rate']:.1%}")
        print(f"Total Duration: {summary['execution_info']['total_duration']:.2f}s")
        
        print("\nFAILED MODULES:")
        for module, details in summary['module_details'].items():
            if not details['passed']:
                print(f"  ❌ {module}")
                # Show error details from original results
                if module in self.test_results:
                    stderr = self.test_results[module].get('stderr', '')
                    if stderr:
                        print(f"     Error: {stderr[:200]}...")
        
        print("\nReports saved to:")
        print(f"  - HTML: {self.output_dir}/test_report.html")
        print(f"  - Markdown: {self.output_dir}/test_report.md")
        print(f"  - JSON Summary: {self.output_dir}/test_summary.json")
        print(f"  - Detailed Results: {self.output_dir}/detailed_results.json")
        
        print("\n" + "="*80)


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description='Run comprehensive test suite')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output-dir', '-o', default='test_results', help='Output directory for results')
    parser.add_argument('--module', '-m', help='Run specific test module only')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = ComprehensiveTestRunner(args.output_dir)
    
    try:
        if args.module:
            # Run specific module
            result = runner.run_test_module(args.module, args.verbose)
            print(f"\nModule {args.module}: {'PASSED' if result['passed'] else 'FAILED'}")
            print(f"Duration: {result['duration']:.2f}s")
            
            if not result['passed']:
                print(f"Error: {result['stderr']}")
                sys.exit(1)
        else:
            # Run all tests
            results = runner.run_all_tests(args.verbose)
            summary = runner.generate_summary_report()
            runner.save_reports(summary)
            runner.print_final_summary(summary)
            
            # Exit with error code if any tests failed
            if summary['test_results']['failed_modules'] > 0:
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()