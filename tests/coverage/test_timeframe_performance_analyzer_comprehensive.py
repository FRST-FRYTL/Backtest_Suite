"""
Comprehensive test suite for Timeframe Performance Analyzer module.

This test suite aims for 100% coverage of the timeframe_performance_analyzer.py module
by testing all methods, visualizations, report generation, and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, mock_open
import json
import tempfile
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Import modules to test
from src.analysis.timeframe_performance_analyzer import (
    TimeframePerformanceAnalyzer, PerformanceMetrics, TimeframeResult
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass"""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation and properties"""
        metrics = PerformanceMetrics(
            total_return=0.25,
            annualized_return=0.15,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            max_drawdown=-0.15,
            calmar_ratio=1.0,
            win_rate=0.65,
            profit_factor=1.8,
            volatility=0.10,
            var_95=-0.02,
            cvar_95=-0.03,
            total_trades=100,
            avg_trade_duration=5.5,
            best_trade=0.10,
            worst_trade=-0.05,
            recovery_time=20.0,
            beta=0.9,
            alpha=0.05,
            information_ratio=0.8
        )
        
        # Test all attributes
        assert metrics.total_return == 0.25
        assert metrics.sharpe_ratio == 1.5
        assert metrics.total_trades == 100
        assert metrics.avg_trade_duration == 5.5
        
        # Test risk_adjusted_return property
        expected_rar = 0.15 / 0.10  # annualized_return / volatility
        assert metrics.risk_adjusted_return == expected_rar
        
        # Test with zero volatility
        metrics_zero_vol = PerformanceMetrics(
            total_return=0.25,
            annualized_return=0.15,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=-0.15,
            calmar_ratio=0,
            win_rate=0.5,
            profit_factor=1.0,
            volatility=0.0,
            var_95=0,
            cvar_95=0,
            total_trades=10
        )
        assert metrics_zero_vol.risk_adjusted_return == 0.0
    
    def test_consistency_score_calculation(self):
        """Test consistency score calculation"""
        # High performance metrics
        high_perf = PerformanceMetrics(
            total_return=0.50,
            annualized_return=0.30,
            sharpe_ratio=2.5,
            sortino_ratio=3.0,
            max_drawdown=-0.10,
            calmar_ratio=3.0,
            win_rate=0.75,
            profit_factor=2.5,
            volatility=0.12,
            var_95=-0.015,
            cvar_95=-0.02,
            total_trades=150
        )
        
        # Win rate: 0.75 * 40 = 30 points
        # Sharpe: min(2.5 * 10, 30) = 25 points
        # Drawdown: max(0, 30 * (1 - 0.10/0.5)) = 24 points
        expected_score = 30 + 25 + 24
        assert high_perf.consistency_score == expected_score
        
        # Low performance metrics
        low_perf = PerformanceMetrics(
            total_return=-0.20,
            annualized_return=-0.15,
            sharpe_ratio=-0.5,
            sortino_ratio=-0.3,
            max_drawdown=-0.60,
            calmar_ratio=-0.25,
            win_rate=0.30,
            profit_factor=0.8,
            volatility=0.20,
            var_95=-0.05,
            cvar_95=-0.07,
            total_trades=50
        )
        
        # Win rate: 0.30 * 40 = 12 points
        # Sharpe: 0 points (negative)
        # Drawdown: max(0, 30 * (1 - 0.60/0.5)) = 0 points (capped at 0)
        expected_score_low = 12 + 0 + 0
        assert low_perf.consistency_score == expected_score_low


class TestTimeframeResult:
    """Test TimeframeResult dataclass"""
    
    def test_timeframe_result_creation(self):
        """Test TimeframeResult creation and parameter hash"""
        metrics = PerformanceMetrics(
            total_return=0.20,
            annualized_return=0.12,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown=-0.12,
            calmar_ratio=1.0,
            win_rate=0.60,
            profit_factor=1.5,
            volatility=0.10,
            var_95=-0.02,
            cvar_95=-0.03,
            total_trades=80
        )
        
        result = TimeframeResult(
            timeframe='1H',
            symbol='SPY',
            parameters={'sma_period': 20, 'rsi_period': 14, 'threshold': 0.65},
            metrics=metrics,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        assert result.timeframe == '1H'
        assert result.symbol == 'SPY'
        assert result.parameters['sma_period'] == 20
        assert result.metrics.total_return == 0.20
        
        # Test parameter hash
        expected_hash = 'SPY_1H_rsi_period=14_sma_period=20_threshold=0.65'
        assert result.parameter_hash == expected_hash
        
        # Test with empty parameters
        result_empty = TimeframeResult(
            timeframe='1D',
            symbol='QQQ',
            parameters={},
            metrics=metrics,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        assert result_empty.parameter_hash == 'QQQ_1D_'


class TestTimeframePerformanceAnalyzer:
    """Comprehensive tests for TimeframePerformanceAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create TimeframePerformanceAnalyzer instance"""
        return TimeframePerformanceAnalyzer()
    
    @pytest.fixture
    def sample_results(self):
        """Create sample timeframe results"""
        results = []
        
        timeframes = ['1H', '4H', '1D']
        symbols = ['SPY', 'QQQ']
        
        for tf in timeframes:
            for symbol in symbols:
                for i in range(3):  # 3 parameter configs per timeframe/symbol
                    metrics = PerformanceMetrics(
                        total_return=np.random.uniform(-0.1, 0.5),
                        annualized_return=np.random.uniform(-0.05, 0.3),
                        sharpe_ratio=np.random.uniform(-0.5, 2.5),
                        sortino_ratio=np.random.uniform(-0.3, 3.0),
                        max_drawdown=np.random.uniform(-0.30, -0.05),
                        calmar_ratio=np.random.uniform(0.5, 3.0),
                        win_rate=np.random.uniform(0.4, 0.7),
                        profit_factor=np.random.uniform(0.8, 2.0),
                        volatility=np.random.uniform(0.05, 0.20),
                        var_95=np.random.uniform(-0.05, -0.01),
                        cvar_95=np.random.uniform(-0.07, -0.02),
                        total_trades=np.random.randint(50, 200),
                        avg_trade_duration=np.random.uniform(1, 10),
                        best_trade=np.random.uniform(0.05, 0.15),
                        worst_trade=np.random.uniform(-0.10, -0.02),
                        recovery_time=np.random.uniform(5, 30)
                    )
                    
                    result = TimeframeResult(
                        timeframe=tf,
                        symbol=symbol,
                        parameters={
                            'sma_period': 20 + i * 10,
                            'rsi_period': 14 + i * 2,
                            'threshold': 0.65 + i * 0.05
                        },
                        metrics=metrics,
                        start_date='2023-01-01',
                        end_date='2023-12-31'
                    )
                    results.append(result)
        
        return results
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert isinstance(analyzer.results, list)
        assert len(analyzer.results) == 0
        assert isinstance(analyzer.analysis_results, dict)
        assert analyzer.results_dir == Path("backtest_results")
    
    def test_initialization_with_custom_dir(self):
        """Test analyzer initialization with custom directory"""
        custom_dir = Path("/tmp/custom_results")
        analyzer = TimeframePerformanceAnalyzer(results_dir=custom_dir)
        assert analyzer.results_dir == custom_dir
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"test": "data"}')
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_results_single_file(self, mock_exists, mock_file, analyzer):
        """Test loading results from a single file"""
        with patch.object(analyzer, '_parse_results') as mock_parse:
            analyzer.load_results(Path("test_results.json"))
            mock_parse.assert_called_once_with({"test": "data"})
    
    @patch('pathlib.Path.glob')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_results_directory(self, mock_file, mock_glob, analyzer):
        """Test loading results from directory"""
        # Mock glob to return test files
        mock_glob.return_value = [Path("file1.json"), Path("file2.json")]
        
        # Mock file contents
        mock_file.return_value.read.side_effect = [
            '{"data": "file1"}',
            '{"data": "file2"}'
        ]
        
        with patch.object(analyzer, '_parse_results') as mock_parse:
            analyzer.load_results()
            assert mock_parse.call_count == 2
    
    @patch('pathlib.Path.glob')
    @patch('builtins.open', side_effect=Exception("File error"))
    def test_load_results_with_error(self, mock_file, mock_glob, analyzer, capsys):
        """Test load results error handling"""
        mock_glob.return_value = [Path("error_file.json")]
        
        analyzer.load_results()
        captured = capsys.readouterr()
        assert "Error loading" in captured.out
    
    def test_parse_results(self, analyzer):
        """Test _parse_results method"""
        # This is implementation-specific and would need actual data structure
        data = {"results": []}
        analyzer._parse_results(data)
        # Since implementation is empty, just verify it doesn't crash
        assert True
    
    def test_analyze_by_timeframe(self, analyzer, sample_results):
        """Test timeframe analysis"""
        analyzer.results = sample_results
        
        analysis = analyzer.analyze_by_timeframe()
        
        assert isinstance(analysis, dict)
        assert '1H' in analysis
        assert '4H' in analysis
        assert '1D' in analysis
        
        # Check structure of each timeframe analysis
        for tf, data in analysis.items():
            assert 'count' in data
            assert 'avg_return' in data
            assert 'avg_sharpe' in data
            assert 'avg_drawdown' in data
            assert 'best_sharpe' in data
            assert 'best_return' in data
            assert 'worst_drawdown' in data
            assert 'avg_trades' in data
            assert 'consistency' in data
            assert 'risk_adjusted_return' in data
            assert 'best_config' in data
            
            # Verify best config structure
            if data.get('best_config'):
                assert 'parameters' in data['best_config']
                assert 'sharpe_ratio' in data['best_config']
                assert 'total_return' in data['best_config']
                assert 'max_drawdown' in data['best_config']
    
    def test_analyze_parameter_sensitivity(self, analyzer, sample_results):
        """Test parameter sensitivity analysis"""
        analyzer.results = sample_results
        
        sensitivity = analyzer.analyze_parameter_sensitivity()
        
        assert isinstance(sensitivity, dict)
        # Should have parameter columns
        assert 'param_sma_period' in sensitivity
        assert 'param_rsi_period' in sensitivity
        assert 'param_threshold' in sensitivity
        
        # Check correlation structure
        for param, correlations in sensitivity.items():
            assert 'sharpe_ratio' in correlations
            assert 'total_return' in correlations
            assert 'max_drawdown' in correlations
            assert 'consistency_score' in correlations
            
            # All correlations should be floats
            for metric, corr in correlations.items():
                assert isinstance(corr, float)
                assert -1 <= corr <= 1 or corr == 0.0
    
    def test_find_robust_configurations(self, analyzer, sample_results):
        """Test finding robust configurations"""
        analyzer.results = sample_results
        
        robust_configs = analyzer.find_robust_configurations(
            min_sharpe=0.5,
            max_drawdown=-0.25
        )
        
        assert isinstance(robust_configs, list)
        
        # Check each robust config
        for config in robust_configs:
            assert 'parameters' in config
            assert 'avg_sharpe' in config
            assert 'avg_return' in config
            assert 'worst_drawdown' in config
            assert 'timeframe_count' in config
            assert 'sharpe_std' in config
            assert 'return_std' in config
            
            # Verify constraints
            assert config['avg_sharpe'] >= 0.5
            assert config['worst_drawdown'] >= -0.25
        
        # Should be sorted by avg_sharpe descending
        if len(robust_configs) > 1:
            for i in range(len(robust_configs) - 1):
                assert robust_configs[i]['avg_sharpe'] >= robust_configs[i+1]['avg_sharpe']
    
    def test_results_to_dataframe(self, analyzer, sample_results):
        """Test conversion of results to DataFrame"""
        analyzer.results = sample_results
        
        df = analyzer._results_to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_results)
        
        # Check required columns
        required_cols = [
            'timeframe', 'symbol', 'parameters', 'parameter_hash',
            'start_date', 'end_date', 'total_return', 'annualized_return',
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'calmar_ratio',
            'win_rate', 'profit_factor', 'volatility', 'var_95', 'cvar_95',
            'total_trades', 'risk_adjusted_return', 'consistency_score'
        ]
        for col in required_cols:
            assert col in df.columns
        
        # Check parameter columns
        assert 'param_sma_period' in df.columns
        assert 'param_rsi_period' in df.columns
        assert 'param_threshold' in df.columns
        
        # Check optional columns
        assert 'avg_trade_duration' in df.columns
        assert 'best_trade' in df.columns
        assert 'worst_trade' in df.columns
        assert 'recovery_time' in df.columns
    
    @patch('plotly.graph_objects.Figure')
    def test_create_performance_heatmap(self, mock_fig, analyzer, sample_results):
        """Test performance heatmap creation"""
        analyzer.results = sample_results
        
        fig = analyzer.create_performance_heatmap()
        
        # Should create subplots
        assert isinstance(fig, go.Figure)
    
    @patch('plotly.graph_objects.Figure')
    def test_create_timeframe_comparison_chart(self, mock_fig, analyzer, sample_results):
        """Test timeframe comparison chart"""
        analyzer.results = sample_results
        
        fig = analyzer.create_timeframe_comparison_chart()
        
        assert isinstance(fig, go.Figure)
    
    @patch('plotly.graph_objects.Figure')
    def test_create_drawdown_analysis(self, mock_fig, analyzer, sample_results):
        """Test drawdown analysis visualization"""
        analyzer.results = sample_results
        
        fig = analyzer.create_drawdown_analysis()
        
        assert isinstance(fig, go.Figure)
    
    @patch('plotly.graph_objects.Figure')
    def test_create_return_distribution_plots(self, mock_fig, analyzer, sample_results):
        """Test return distribution plots"""
        analyzer.results = sample_results
        
        fig = analyzer.create_return_distribution_plots()
        
        assert isinstance(fig, go.Figure)
    
    def test_generate_summary_cards(self, analyzer, sample_results):
        """Test summary cards HTML generation"""
        analyzer.results = sample_results
        
        html = analyzer._generate_summary_cards()
        
        assert isinstance(html, str)
        assert 'summary-card' in html
        assert 'Best Sharpe Ratio' in html
        assert 'Best Total Return' in html
        assert 'Best Risk-Adjusted Return' in html
        assert 'Most Consistent Strategy' in html
    
    def test_generate_timeframe_table(self, analyzer, sample_results):
        """Test timeframe table HTML generation"""
        analyzer.results = sample_results
        analyzer.analyze_by_timeframe()
        
        html = analyzer._generate_timeframe_table()
        
        assert isinstance(html, str)
        assert '<table>' in html
        assert 'Timeframe' in html
        assert '1H' in html
        assert '4H' in html
        assert '1D' in html
    
    def test_generate_timeframe_table_empty(self, analyzer):
        """Test timeframe table with no analysis"""
        html = analyzer._generate_timeframe_table()
        assert 'No timeframe analysis available' in html
    
    def test_generate_robust_configs_table(self, analyzer, sample_results):
        """Test robust configurations table generation"""
        analyzer.results = sample_results
        robust_configs = analyzer.find_robust_configurations()
        
        html = analyzer._generate_robust_configs_table(robust_configs[:5])
        
        assert isinstance(html, str)
        assert '<table>' in html
        assert 'Parameters' in html
        assert 'Avg Sharpe' in html
        
        # Test with empty configs
        html_empty = analyzer._generate_robust_configs_table([])
        assert 'No robust configurations found' in html_empty
        
        # Test with string parameters
        configs_str = [{
            'parameters': 'sma=20,rsi=14',
            'avg_sharpe': 1.5,
            'avg_return': 0.20,
            'worst_drawdown': -0.15,
            'timeframe_count': 3,
            'sharpe_std': 0.2
        }]
        html_str = analyzer._generate_robust_configs_table(configs_str)
        assert 'sma=20,rsi=14' in html_str
    
    def test_generate_sensitivity_table(self, analyzer, sample_results):
        """Test sensitivity table generation"""
        analyzer.results = sample_results
        analyzer.analyze_parameter_sensitivity()
        
        html = analyzer._generate_sensitivity_table()
        
        assert isinstance(html, str)
        assert '<table>' in html
        assert 'sma_period' in html  # Should remove 'param_' prefix
        assert 'Sharpe Correlation' in html
        
        # Test with no sensitivity analysis
        analyzer.analysis_results = {}
        html_empty = analyzer._generate_sensitivity_table()
        assert 'No parameter sensitivity analysis available' in html_empty
    
    def test_generate_recommendations(self, analyzer, sample_results):
        """Test recommendations generation"""
        analyzer.results = sample_results
        analyzer.analyze_by_timeframe()
        analyzer.find_robust_configurations()
        analyzer.analyze_parameter_sensitivity()
        
        html = analyzer._generate_recommendations()
        
        assert isinstance(html, str)
        assert '<ul>' in html
        assert 'Best Average Performance' in html
        assert 'Risk Management' in html
        assert 'Diversification' in html
        
        # Test with high impact parameters
        analyzer.analysis_results['parameter_sensitivity'] = {
            'param_sma_period': {'sharpe_ratio': 0.5, 'total_return': 0.4}
        }
        html_high_impact = analyzer._generate_recommendations()
        assert 'High Impact Parameters' in html_high_impact
        assert 'sma_period' in html_high_impact
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('plotly.graph_objects.Figure.to_json', return_value='{}')
    def test_generate_html_report(self, mock_to_json, mock_file, analyzer, sample_results):
        """Test HTML report generation"""
        analyzer.results = sample_results
        
        output_path = Path("test_report.html")
        analyzer.generate_html_report(output_path)
        
        # Verify file was written
        mock_file.assert_called_once_with(output_path, 'w')
        handle = mock_file()
        
        # Get the written content
        written_content = ''.join(call.args[0] for call in handle.write.call_args_list)
        
        # Verify HTML structure
        assert '<!DOCTYPE html>' in written_content
        assert 'Multi-Timeframe Performance Analysis' in written_content
        assert 'Executive Summary' in written_content
        assert 'Performance Visualizations' in written_content
        assert 'Key Findings and Recommendations' in written_content
        assert datetime.now().strftime('%Y-%m-%d') in written_content
    
    def test_main_function(self):
        """Test main function"""
        from src.analysis.timeframe_performance_analyzer import main
        
        # Should run without errors
        main()
        assert True
    
    def test_edge_cases(self, analyzer):
        """Test edge cases and error handling"""
        # Test with single result
        metrics = PerformanceMetrics(
            total_return=0.10,
            annualized_return=0.06,
            sharpe_ratio=0.8,
            sortino_ratio=1.0,
            max_drawdown=-0.08,
            calmar_ratio=0.75,
            win_rate=0.55,
            profit_factor=1.2,
            volatility=0.075,
            var_95=-0.015,
            cvar_95=-0.02,
            total_trades=40
        )
        
        single_result = TimeframeResult(
            timeframe='1H',
            symbol='SPY',
            parameters={'sma': 20},
            metrics=metrics,
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        analyzer.results = [single_result]
        
        # Should handle single result gracefully
        analysis = analyzer.analyze_by_timeframe()
        assert '1H' in analysis
        assert analysis['1H']['count'] == 1
        
        # Test with results having NaN values
        metrics_nan = PerformanceMetrics(
            total_return=np.nan,
            annualized_return=0.06,
            sharpe_ratio=np.nan,
            sortino_ratio=1.0,
            max_drawdown=-0.08,
            calmar_ratio=0.75,
            win_rate=0.55,
            profit_factor=1.2,
            volatility=0.075,
            var_95=-0.015,
            cvar_95=-0.02,
            total_trades=40
        )
        
        result_nan = TimeframeResult(
            timeframe='1D',
            symbol='QQQ',
            parameters={'rsi': 14},
            metrics=metrics_nan,
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        analyzer.results = [result_nan]
        df = analyzer._results_to_dataframe()
        
        # Should handle NaN values
        assert pd.isna(df['total_return'].iloc[0])
        assert pd.isna(df['sharpe_ratio'].iloc[0])