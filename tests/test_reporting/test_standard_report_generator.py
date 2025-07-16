"""
Tests for StandardReportGenerator

This module tests the core functionality of the standard report generator.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

from src.reporting import StandardReportGenerator, ReportConfig
from src.reporting.report_sections import (
    ExecutiveSummary,
    PerformanceAnalysis,
    RiskAnalysis,
    TradeAnalysis
)


class TestStandardReportGenerator:
    """Test cases for StandardReportGenerator"""
    
    @pytest.fixture
    def sample_backtest_results(self):
        """Create sample backtest results for testing"""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Equity curve
        returns = np.random.normal(0.0003, 0.01, len(dates))
        equity_curve = pd.Series(
            (1 + returns).cumprod() * 100000,
            index=dates,
            name='equity'
        )
        
        # Trades
        trades_data = []
        for i in range(50):
            entry_date = dates[i * 7]
            exit_date = entry_date + timedelta(days=3)
            trades_data.append({
                'entry_time': entry_date,
                'exit_time': exit_date,
                'side': 'long',
                'size': 10000,
                'pnl': np.random.normal(100, 500),
                'duration': 72  # hours
            })
        
        trades = pd.DataFrame(trades_data)
        
        # Metrics
        metrics = {
            'total_return': 0.15,
            'annual_return': 0.15,
            'volatility': 0.16,
            'sharpe_ratio': 0.94,
            'max_drawdown': -0.08,
            'win_rate': 0.58,
            'profit_factor': 1.5
        }
        
        # Strategy params
        strategy_params = {
            'name': 'Test Strategy',
            'lookback': 20,
            'threshold': 2.0
        }
        
        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'metrics': metrics,
            'strategy_params': strategy_params
        }
    
    def test_initialization_default_config(self):
        """Test initialization with default configuration"""
        generator = StandardReportGenerator()
        
        assert generator.config.title == "Backtest Results Report"
        assert generator.config.include_executive_summary == True
        assert len(generator.sections) == 6  # All sections enabled by default
    
    def test_initialization_custom_config(self):
        """Test initialization with custom configuration"""
        config = ReportConfig(
            title="Custom Report",
            include_trade_analysis=False,
            include_market_regime_analysis=False
        )
        
        generator = StandardReportGenerator(config)
        
        assert generator.config.title == "Custom Report"
        assert len(generator.sections) == 4  # Two sections disabled
    
    def test_generate_report_basic(self, sample_backtest_results):
        """Test basic report generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = StandardReportGenerator()
            
            output_files = generator.generate_report(
                backtest_results=sample_backtest_results,
                output_dir=temp_dir,
                report_name="test_report"
            )
            
            # Check that files were created
            assert 'html' in output_files
            assert 'json' in output_files
            assert Path(output_files['html']).exists()
            assert Path(output_files['json']).exists()
    
    def test_validate_backtest_results(self):
        """Test validation of backtest results"""
        generator = StandardReportGenerator()
        
        # Test missing required fields
        invalid_results = {'equity_curve': pd.Series([1, 2, 3])}
        
        with pytest.raises(ValueError, match="Missing required fields"):
            generator._validate_backtest_results(invalid_results)
        
        # Test invalid data types
        invalid_results = {
            'equity_curve': [1, 2, 3],  # Should be Series/DataFrame
            'trades': pd.DataFrame(),
            'metrics': {},
            'strategy_params': {}
        }
        
        with pytest.raises(TypeError, match="must be a pandas"):
            generator._validate_backtest_results(invalid_results)
    
    def test_create_backtest_summary(self, sample_backtest_results):
        """Test backtest summary creation"""
        generator = StandardReportGenerator()
        summary = generator._create_backtest_summary(sample_backtest_results)
        
        assert 'performance' in summary
        assert 'risk' in summary
        assert 'trading' in summary
        assert 'evaluation' in summary
        
        # Check performance metrics
        assert summary['performance']['total_return'] == 0.15
        assert summary['performance']['sharpe_ratio'] == 0.94
        
        # Check evaluation
        assert 'overall_rating' in summary['evaluation']
        assert 'strengths' in summary['evaluation']
        assert 'weaknesses' in summary['evaluation']
    
    def test_evaluate_performance(self):
        """Test performance evaluation logic"""
        generator = StandardReportGenerator()
        
        # Test excellent performance
        excellent_metrics = {
            'sharpe_ratio': 2.5,
            'max_drawdown': -0.08,
            'win_rate': 0.65,
            'profit_factor': 2.5,
            'consistency_score': 0.9
        }
        
        evaluation = generator._evaluate_performance(excellent_metrics)
        assert evaluation['overall_rating'] == "Excellent"
        assert len(evaluation['strengths']) > len(evaluation['weaknesses'])
        
        # Test poor performance
        poor_metrics = {
            'sharpe_ratio': 0.5,
            'max_drawdown': -0.35,
            'win_rate': 0.35,
            'profit_factor': 0.8,
            'consistency_score': 0.3
        }
        
        evaluation = generator._evaluate_performance(poor_metrics)
        assert evaluation['overall_rating'] == "Needs Improvement"
        assert len(evaluation['weaknesses']) > len(evaluation['strengths'])
    
    def test_save_json_report(self, sample_backtest_results):
        """Test JSON report saving"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = StandardReportGenerator()
            
            report_data = {
                'metadata': generator.metadata,
                'sections': {'test': {'data': 'value'}},
                'backtest_summary': generator._create_backtest_summary(sample_backtest_results)
            }
            
            output_file = generator._save_json_report(
                report_data,
                Path(temp_dir),
                "test_json"
            )
            
            # Verify file exists and can be loaded
            assert output_file.exists()
            
            with open(output_file, 'r') as f:
                loaded_data = json.load(f)
            
            assert 'metadata' in loaded_data
            assert 'sections' in loaded_data
            assert loaded_data['sections']['test']['data'] == 'value'
    
    def test_add_custom_section(self):
        """Test adding custom sections"""
        generator = StandardReportGenerator()
        initial_sections = len(generator.sections)
        
        custom_content = {
            'custom_metric': 42,
            'custom_analysis': 'Test analysis'
        }
        
        generator.add_custom_section('Custom Section', custom_content)
        
        assert len(generator.sections) == initial_sections + 1
        assert generator.sections[-1].name == 'Custom Section'
    
    def test_set_theme(self):
        """Test theme setting"""
        generator = StandardReportGenerator()
        
        # Test valid theme
        generator.set_theme('minimal')
        assert generator.config.color_scheme['primary'] == '#000000'
        
        # Test invalid theme (should not raise error)
        generator.set_theme('invalid_theme')  # Should log warning
    
    def test_config_to_dict(self):
        """Test configuration serialization"""
        config = ReportConfig(
            title="Test Config",
            min_sharpe_ratio=1.5,
            include_trade_analysis=False
        )
        
        generator = StandardReportGenerator(config)
        config_dict = generator._config_to_dict()
        
        assert config_dict['title'] == "Test Config"
        assert config_dict['thresholds']['min_sharpe'] == 1.5
        assert config_dict['sections']['trade_analysis'] == False


class TestReportSections:
    """Test individual report sections"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return ReportConfig()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for section tests"""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        equity = pd.Series(np.linspace(100000, 115000, len(dates)), index=dates)
        
        return {
            'equity_curve': equity,
            'metrics': {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.10,
                'win_rate': 0.55
            },
            'trades': pd.DataFrame({
                'pnl': np.random.normal(100, 50, 100),
                'duration': np.random.uniform(1, 48, 100)
            })
        }
    
    def test_executive_summary_generation(self, config, sample_data):
        """Test executive summary generation"""
        section = ExecutiveSummary(config)
        result = section.generate(sample_data)
        
        assert 'key_metrics' in result
        assert 'performance_summary' in result
        assert 'risk_summary' in result
        assert 'strategy_assessment' in result
        assert 'recommendations' in result
        
        # Check key metrics formatting
        assert 'Total Return' in result['key_metrics']
        assert '%' in result['key_metrics']['Total Return']
    
    def test_performance_analysis_generation(self, config, sample_data):
        """Test performance analysis generation"""
        section = PerformanceAnalysis(config)
        result = section.generate(sample_data)
        
        assert 'return_analysis' in result
        assert 'risk_adjusted_metrics' in result
        assert 'rolling_performance' in result
        assert 'statistical_significance' in result
        
        # Check risk-adjusted metrics
        sharpe_data = result['risk_adjusted_metrics']['sharpe_ratio']
        assert 'value' in sharpe_data
        assert 'interpretation' in sharpe_data
    
    def test_risk_analysis_generation(self, config, sample_data):
        """Test risk analysis generation"""
        section = RiskAnalysis(config)
        result = section.generate(sample_data)
        
        assert 'drawdown_analysis' in result
        assert 'volatility_analysis' in result
        assert 'var_analysis' in result
        assert 'stress_testing' in result
        assert 'risk_metrics' in result
        
        # Check drawdown analysis
        dd_analysis = result['drawdown_analysis']
        assert 'maximum_drawdown' in dd_analysis
        assert 'drawdown_duration' in dd_analysis
    
    def test_trade_analysis_empty_trades(self, config):
        """Test trade analysis with no trades"""
        section = TradeAnalysis(config)
        result = section.generate({'trades': pd.DataFrame()})
        
        assert 'message' in result
        assert 'No trades' in result['message']
    
    def test_format_number(self, config):
        """Test number formatting"""
        section = ExecutiveSummary(config)
        
        # Test different format types
        assert section.format_number(0.1523, 'percentage') == '15.23%'
        assert section.format_number(1250000, 'currency') == '$1,250,000.00'
        assert section.format_number(1.853, 'ratio') == '1.85'
        assert section.format_number(1250, 'integer') == '1,250'
        
        # Test NaN handling
        assert section.format_number(np.nan, 'percentage') == 'N/A'


class TestReportIntegration:
    """Integration tests for the complete reporting system"""
    
    def test_full_report_generation(self):
        """Test complete report generation workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create comprehensive test data
            dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
            np.random.seed(42)
            
            # Generate realistic equity curve
            returns = np.random.normal(0.0004, 0.012, len(dates))
            equity = pd.Series((1 + returns).cumprod() * 100000, index=dates)
            
            # Generate trades
            n_trades = 200
            trades = pd.DataFrame({
                'entry_time': pd.to_datetime(np.random.choice(dates[:-5], n_trades)),
                'exit_time': pd.to_datetime(np.random.choice(dates[5:], n_trades)),
                'side': np.random.choice(['long', 'short'], n_trades),
                'size': np.random.uniform(5000, 20000, n_trades),
                'pnl': np.concatenate([
                    np.random.normal(150, 100, n_trades // 2),  # Winners
                    np.random.normal(-100, 80, n_trades // 2)   # Losers
                ]),
                'entry_reason': np.random.choice(['signal1', 'signal2'], n_trades),
                'exit_reason': np.random.choice(['target', 'stop', 'time'], n_trades)
            })
            
            trades['duration'] = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 3600
            
            # Calculate comprehensive metrics
            daily_returns = equity.pct_change().dropna()
            
            metrics = {
                'total_return': (equity.iloc[-1] / equity.iloc[0] - 1),
                'annual_return': daily_returns.mean() * 252,
                'volatility': daily_returns.std() * np.sqrt(252),
                'sharpe_ratio': (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)),
                'sortino_ratio': (daily_returns.mean() * 252) / (daily_returns[daily_returns < 0].std() * np.sqrt(252)),
                'max_drawdown': ((equity / equity.cummax()) - 1).min(),
                'win_rate': (trades['pnl'] > 0).mean(),
                'profit_factor': trades[trades['pnl'] > 0]['pnl'].sum() / abs(trades[trades['pnl'] < 0]['pnl'].sum()),
                'var_95': np.percentile(daily_returns, 5),
                'cvar_95': daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean()
            }
            
            # Create backtest results
            backtest_results = {
                'equity_curve': equity,
                'trades': trades,
                'metrics': metrics,
                'returns': daily_returns,
                'strategy_params': {
                    'name': 'Integration Test Strategy',
                    'version': '1.0',
                    'parameters': {'lookback': 20, 'threshold': 2.0}
                },
                'market_data': pd.DataFrame({
                    'close': equity * 1.1,
                    'volume': np.random.randint(1e6, 5e6, len(dates))
                }, index=dates)
            }
            
            # Configure and generate report
            config = ReportConfig(
                title="Integration Test Report",
                subtitle="Complete System Test",
                output_formats=['html', 'json']
            )
            
            generator = StandardReportGenerator(config)
            output_files = generator.generate_report(
                backtest_results=backtest_results,
                output_dir=temp_dir,
                report_name="integration_test"
            )
            
            # Verify outputs
            assert all(Path(f).exists() for f in output_files.values())
            
            # Verify JSON content
            with open(output_files['json'], 'r') as f:
                report_data = json.load(f)
            
            assert 'metadata' in report_data
            assert 'sections' in report_data
            assert 'backtest_summary' in report_data
            
            # Verify all sections were generated
            expected_sections = [
                'executivesummary',
                'performanceanalysis',
                'riskanalysis',
                'tradeanalysis',
                'marketregimeanalysis',
                'technicaldetails'
            ]
            
            for section in expected_sections:
                assert section in report_data['sections']
                assert 'error' not in report_data['sections'][section]
    
    def test_report_generation_with_errors(self):
        """Test report generation handles errors gracefully"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid data that will cause some sections to fail
            invalid_results = {
                'equity_curve': pd.Series([100000, 110000, 105000]),
                'trades': pd.DataFrame(),  # Empty trades
                'metrics': {'sharpe_ratio': 'invalid'},  # Invalid metric type
                'strategy_params': {}
            }
            
            generator = StandardReportGenerator()
            
            # Should complete without raising exception
            output_files = generator.generate_report(
                backtest_results=invalid_results,
                output_dir=temp_dir,
                report_name="error_test"
            )
            
            # Report should still be generated
            assert 'json' in output_files
            assert Path(output_files['json']).exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])