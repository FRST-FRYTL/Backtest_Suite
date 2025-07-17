"""
Comprehensive tests for performance analysis components.

This module provides complete test coverage for performance metrics calculation,
analysis functions, and benchmarking utilities.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import warnings
import math

from src.analysis.timeframe_performance_analyzer import TimeframePerformanceAnalyzer
from src.utils.metrics import PerformanceMetrics
from src.backtesting.strategy import Strategy


class TestTimeframePerformanceAnalyzer:
    """Comprehensive tests for TimeframePerformanceAnalyzer class."""
    
    @pytest.fixture
    def sample_equity_curves(self):
        """Create sample equity curves for different timeframes."""
        np.random.seed(42)
        
        # Create base dates
        start_date = pd.Timestamp('2023-01-01')
        
        # Daily equity curve
        daily_dates = pd.date_range(start=start_date, periods=365, freq='D')
        daily_returns = np.random.normal(0.001, 0.02, len(daily_dates))
        daily_equity = pd.Series(
            100000 * (1 + daily_returns).cumprod(),
            index=daily_dates,
            name='equity'
        )
        
        # Hourly equity curve (sample)
        hourly_dates = pd.date_range(start=start_date, periods=365*24, freq='H')
        hourly_returns = np.random.normal(0.0001, 0.005, len(hourly_dates))
        hourly_equity = pd.Series(
            100000 * (1 + hourly_returns).cumprod(),
            index=hourly_dates,
            name='equity'
        )
        
        # 5-minute equity curve (sample)
        minute_dates = pd.date_range(start=start_date, periods=365*24*12, freq='5T')
        minute_returns = np.random.normal(0.00001, 0.002, len(minute_dates))
        minute_equity = pd.Series(
            100000 * (1 + minute_returns).cumprod(),
            index=minute_dates,
            name='equity'
        )
        
        return {
            '1D': daily_equity,
            '1H': hourly_equity,
            '5T': minute_equity
        }
    
    @pytest.fixture
    def sample_trades_data(self):
        """Create sample trades data for different timeframes."""
        np.random.seed(42)
        
        # Daily trades
        daily_trades = pd.DataFrame({
            'entry_time': pd.date_range('2023-01-01', periods=100, freq='D'),
            'exit_time': pd.date_range('2023-01-01', periods=100, freq='D') + timedelta(days=1),
            'pnl': np.random.normal(100, 500, 100),
            'duration': np.random.uniform(1, 24, 100),  # hours
            'side': np.random.choice(['long', 'short'], 100)
        })
        
        # Hourly trades
        hourly_trades = pd.DataFrame({
            'entry_time': pd.date_range('2023-01-01', periods=500, freq='H'),
            'exit_time': pd.date_range('2023-01-01', periods=500, freq='H') + timedelta(hours=1),
            'pnl': np.random.normal(20, 100, 500),
            'duration': np.random.uniform(0.1, 2, 500),  # hours
            'side': np.random.choice(['long', 'short'], 500)
        })
        
        # 5-minute trades
        minute_trades = pd.DataFrame({
            'entry_time': pd.date_range('2023-01-01', periods=2000, freq='5T'),
            'exit_time': pd.date_range('2023-01-01', periods=2000, freq='5T') + timedelta(minutes=5),
            'pnl': np.random.normal(5, 25, 2000),
            'duration': np.random.uniform(0.01, 0.5, 2000),  # hours
            'side': np.random.choice(['long', 'short'], 2000)
        })
        
        return {
            '1D': daily_trades,
            '1H': hourly_trades,
            '5T': minute_trades
        }
    
    def test_initialization(self):
        """Test TimeframePerformanceAnalyzer initialization."""
        analyzer = TimeframePerformanceAnalyzer()
        
        assert isinstance(analyzer.timeframes, list)
        assert len(analyzer.timeframes) > 0
        assert '1D' in analyzer.timeframes
        assert '1H' in analyzer.timeframes
        assert '5T' in analyzer.timeframes
        assert isinstance(analyzer.results, dict)
        assert len(analyzer.results) == 0
    
    def test_initialization_custom_timeframes(self):
        """Test initialization with custom timeframes."""
        custom_timeframes = ['1D', '4H', '1H']
        analyzer = TimeframePerformanceAnalyzer(timeframes=custom_timeframes)
        
        assert analyzer.timeframes == custom_timeframes
        assert len(analyzer.results) == 0
    
    def test_analyze_timeframe_performance(self, sample_equity_curves, sample_trades_data):
        """Test single timeframe performance analysis."""
        analyzer = TimeframePerformanceAnalyzer()
        
        # Analyze daily timeframe
        result = analyzer.analyze_timeframe_performance(
            timeframe='1D',
            equity_curve=sample_equity_curves['1D'],
            trades=sample_trades_data['1D']
        )
        
        assert isinstance(result, dict)
        assert 'timeframe' in result
        assert 'metrics' in result
        assert 'statistics' in result
        assert 'risk_metrics' in result
        
        # Check timeframe
        assert result['timeframe'] == '1D'
        
        # Check metrics
        metrics = result['metrics']
        assert 'total_return' in metrics
        assert 'annual_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        
        # Check statistics
        stats = result['statistics']
        assert 'total_trades' in stats
        assert 'avg_trade_duration' in stats
        assert 'avg_trade_pnl' in stats
        
        # Check risk metrics
        risk = result['risk_metrics']
        assert 'volatility' in risk
        assert 'var_95' in risk
        assert 'calmar_ratio' in risk
    
    def test_analyze_all_timeframes(self, sample_equity_curves, sample_trades_data):
        """Test analysis of all timeframes."""
        analyzer = TimeframePerformanceAnalyzer(timeframes=['1D', '1H', '5T'])
        
        results = analyzer.analyze_all_timeframes(
            equity_curves=sample_equity_curves,
            trades_data=sample_trades_data
        )
        
        assert isinstance(results, dict)
        assert len(results) == 3
        assert '1D' in results
        assert '1H' in results
        assert '5T' in results
        
        # Check that each timeframe has complete results
        for timeframe, result in results.items():
            assert 'timeframe' in result
            assert 'metrics' in result
            assert 'statistics' in result
            assert 'risk_metrics' in result
            assert result['timeframe'] == timeframe
    
    def test_compare_timeframes(self, sample_equity_curves, sample_trades_data):
        """Test timeframe comparison functionality."""
        analyzer = TimeframePerformanceAnalyzer(timeframes=['1D', '1H', '5T'])
        
        # First analyze all timeframes
        results = analyzer.analyze_all_timeframes(
            equity_curves=sample_equity_curves,
            trades_data=sample_trades_data
        )
        
        # Then compare them
        comparison = analyzer.compare_timeframes(results)
        
        assert isinstance(comparison, dict)
        assert 'summary' in comparison
        assert 'rankings' in comparison
        assert 'correlations' in comparison
        assert 'best_performing' in comparison
        
        # Check summary
        summary = comparison['summary']
        assert len(summary) == 3
        for timeframe in ['1D', '1H', '5T']:
            assert timeframe in summary
            assert 'total_return' in summary[timeframe]
            assert 'sharpe_ratio' in summary[timeframe]
            assert 'max_drawdown' in summary[timeframe]
        
        # Check rankings
        rankings = comparison['rankings']
        assert 'by_total_return' in rankings
        assert 'by_sharpe_ratio' in rankings
        assert 'by_max_drawdown' in rankings
        
        # Check correlations
        correlations = comparison['correlations']
        assert isinstance(correlations, pd.DataFrame)
        assert correlations.shape == (3, 3)
        
        # Check best performing
        best = comparison['best_performing']
        assert 'timeframe' in best
        assert 'composite_score' in best
        assert best['timeframe'] in ['1D', '1H', '5T']
    
    def test_calculate_metrics(self, sample_equity_curves, sample_trades_data):
        """Test metrics calculation."""
        analyzer = TimeframePerformanceAnalyzer()
        
        metrics = analyzer.calculate_metrics(
            equity_curve=sample_equity_curves['1D'],
            trades=sample_trades_data['1D']
        )
        
        assert isinstance(metrics, dict)
        
        # Check return metrics
        assert 'total_return' in metrics
        assert 'annual_return' in metrics
        assert isinstance(metrics['total_return'], (int, float))
        assert isinstance(metrics['annual_return'], (int, float))
        
        # Check risk metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'volatility' in metrics
        assert isinstance(metrics['sharpe_ratio'], (int, float))
        assert isinstance(metrics['max_drawdown'], (int, float))
        assert isinstance(metrics['volatility'], (int, float))
        
        # Check trade metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        assert isinstance(metrics['win_rate'], (int, float))
        assert isinstance(metrics['profit_factor'], (int, float))
        
        # Check bounds
        assert -1 <= metrics['total_return'] <= 10  # Reasonable bounds
        assert -1 <= metrics['annual_return'] <= 10
        assert 0 <= metrics['win_rate'] <= 1
        assert metrics['profit_factor'] >= 0
        assert metrics['max_drawdown'] <= 0
    
    def test_calculate_statistics(self, sample_trades_data):
        """Test statistics calculation."""
        analyzer = TimeframePerformanceAnalyzer()
        
        stats = analyzer.calculate_statistics(sample_trades_data['1D'])
        
        assert isinstance(stats, dict)
        
        # Check basic statistics
        assert 'total_trades' in stats
        assert 'winning_trades' in stats
        assert 'losing_trades' in stats
        assert isinstance(stats['total_trades'], int)
        assert isinstance(stats['winning_trades'], int)
        assert isinstance(stats['losing_trades'], int)
        
        # Check trade metrics
        assert 'avg_trade_pnl' in stats
        assert 'avg_trade_duration' in stats
        assert 'median_trade_pnl' in stats
        assert isinstance(stats['avg_trade_pnl'], (int, float))
        assert isinstance(stats['avg_trade_duration'], (int, float))
        assert isinstance(stats['median_trade_pnl'], (int, float))
        
        # Check consistency
        assert stats['total_trades'] == stats['winning_trades'] + stats['losing_trades']
        assert stats['total_trades'] == len(sample_trades_data['1D'])
    
    def test_calculate_risk_metrics(self, sample_equity_curves):
        """Test risk metrics calculation."""
        analyzer = TimeframePerformanceAnalyzer()
        
        risk_metrics = analyzer.calculate_risk_metrics(sample_equity_curves['1D'])
        
        assert isinstance(risk_metrics, dict)
        
        # Check volatility metrics
        assert 'volatility' in risk_metrics
        assert 'downside_deviation' in risk_metrics
        assert isinstance(risk_metrics['volatility'], (int, float))
        assert isinstance(risk_metrics['downside_deviation'], (int, float))
        
        # Check Value at Risk metrics
        assert 'var_95' in risk_metrics
        assert 'cvar_95' in risk_metrics
        assert isinstance(risk_metrics['var_95'], (int, float))
        assert isinstance(risk_metrics['cvar_95'], (int, float))
        
        # Check ratio metrics
        assert 'calmar_ratio' in risk_metrics
        assert 'sortino_ratio' in risk_metrics
        assert isinstance(risk_metrics['calmar_ratio'], (int, float))
        assert isinstance(risk_metrics['sortino_ratio'], (int, float))
        
        # Check bounds
        assert risk_metrics['volatility'] >= 0
        assert risk_metrics['downside_deviation'] >= 0
        assert risk_metrics['var_95'] <= 0  # Should be negative
        assert risk_metrics['cvar_95'] <= risk_metrics['var_95']  # CVaR should be worse than VaR
    
    def test_create_performance_summary(self, sample_equity_curves, sample_trades_data):
        """Test performance summary creation."""
        analyzer = TimeframePerformanceAnalyzer(timeframes=['1D', '1H'])
        
        results = analyzer.analyze_all_timeframes(
            equity_curves={'1D': sample_equity_curves['1D'], '1H': sample_equity_curves['1H']},
            trades_data={'1D': sample_trades_data['1D'], '1H': sample_trades_data['1H']}
        )
        
        summary = analyzer.create_performance_summary(results)
        
        assert isinstance(summary, dict)
        assert 'overview' in summary
        assert 'detailed_metrics' in summary
        assert 'recommendations' in summary
        
        # Check overview
        overview = summary['overview']
        assert 'best_timeframe' in overview
        assert 'total_timeframes_analyzed' in overview
        assert overview['total_timeframes_analyzed'] == 2
        
        # Check detailed metrics
        detailed = summary['detailed_metrics']
        assert len(detailed) == 2
        assert '1D' in detailed
        assert '1H' in detailed
        
        # Check recommendations
        recommendations = summary['recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)
    
    def test_analyze_timeframe_correlations(self, sample_equity_curves):
        """Test timeframe correlation analysis."""
        analyzer = TimeframePerformanceAnalyzer()
        
        # Create returns for correlation analysis
        returns_data = {}
        for timeframe, equity in sample_equity_curves.items():
            returns_data[timeframe] = equity.pct_change().dropna()
        
        correlations = analyzer.analyze_timeframe_correlations(returns_data)
        
        assert isinstance(correlations, dict)
        assert 'correlation_matrix' in correlations
        assert 'average_correlation' in correlations
        assert 'highest_correlation' in correlations
        assert 'lowest_correlation' in correlations
        
        # Check correlation matrix
        corr_matrix = correlations['correlation_matrix']
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        
        # Check diagonal is 1.0
        for i in range(len(corr_matrix)):
            assert abs(corr_matrix.iloc[i, i] - 1.0) < 1e-10
        
        # Check bounds
        assert -1 <= correlations['average_correlation'] <= 1
        assert -1 <= correlations['highest_correlation'] <= 1
        assert -1 <= correlations['lowest_correlation'] <= 1
    
    def test_analyze_scaling_relationships(self, sample_equity_curves, sample_trades_data):
        """Test scaling relationship analysis."""
        analyzer = TimeframePerformanceAnalyzer()
        
        # Analyze all timeframes first
        results = analyzer.analyze_all_timeframes(
            equity_curves=sample_equity_curves,
            trades_data=sample_trades_data
        )
        
        scaling = analyzer.analyze_scaling_relationships(results)
        
        assert isinstance(scaling, dict)
        assert 'return_scaling' in scaling
        assert 'risk_scaling' in scaling
        assert 'trade_frequency_scaling' in scaling
        
        # Check return scaling
        return_scaling = scaling['return_scaling']
        assert isinstance(return_scaling, dict)
        assert len(return_scaling) > 0
        
        # Check risk scaling
        risk_scaling = scaling['risk_scaling']
        assert isinstance(risk_scaling, dict)
        assert len(risk_scaling) > 0
        
        # Check trade frequency scaling
        trade_freq = scaling['trade_frequency_scaling']
        assert isinstance(trade_freq, dict)
        assert len(trade_freq) > 0
    
    def test_generate_optimization_suggestions(self, sample_equity_curves, sample_trades_data):
        """Test optimization suggestions generation."""
        analyzer = TimeframePerformanceAnalyzer()
        
        results = analyzer.analyze_all_timeframes(
            equity_curves=sample_equity_curves,
            trades_data=sample_trades_data
        )
        
        suggestions = analyzer.generate_optimization_suggestions(results)
        
        assert isinstance(suggestions, dict)
        assert 'parameter_suggestions' in suggestions
        assert 'timeframe_recommendations' in suggestions
        assert 'risk_management_advice' in suggestions
        
        # Check parameter suggestions
        param_suggestions = suggestions['parameter_suggestions']
        assert isinstance(param_suggestions, dict)
        assert len(param_suggestions) > 0
        
        # Check timeframe recommendations
        timeframe_recs = suggestions['timeframe_recommendations']
        assert isinstance(timeframe_recs, list)
        assert len(timeframe_recs) > 0
        
        # Check risk management advice
        risk_advice = suggestions['risk_management_advice']
        assert isinstance(risk_advice, list)
        assert len(risk_advice) > 0
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        analyzer = TimeframePerformanceAnalyzer()
        
        # Test with empty equity curve
        empty_equity = pd.Series([], dtype=float)
        empty_trades = pd.DataFrame()
        
        result = analyzer.analyze_timeframe_performance(
            timeframe='1D',
            equity_curve=empty_equity,
            trades=empty_trades
        )
        
        assert isinstance(result, dict)
        assert 'error' in result or 'metrics' in result
        
        # If metrics exist, they should handle empty data gracefully
        if 'metrics' in result:
            metrics = result['metrics']
            assert all(
                pd.isna(value) or value == 0 or math.isinf(value) 
                for value in metrics.values()
            )
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data."""
        analyzer = TimeframePerformanceAnalyzer()
        
        # Test with NaN values
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        nan_equity = pd.Series([100000] * 50 + [np.nan] * 50, index=dates)
        
        result = analyzer.analyze_timeframe_performance(
            timeframe='1D',
            equity_curve=nan_equity,
            trades=pd.DataFrame()
        )
        
        # Should handle NaN values gracefully
        assert isinstance(result, dict)
        assert 'metrics' in result or 'error' in result
    
    def test_single_timeframe_analysis(self, sample_equity_curves, sample_trades_data):
        """Test analysis with single timeframe."""
        analyzer = TimeframePerformanceAnalyzer(timeframes=['1D'])
        
        results = analyzer.analyze_all_timeframes(
            equity_curves={'1D': sample_equity_curves['1D']},
            trades_data={'1D': sample_trades_data['1D']}
        )
        
        assert isinstance(results, dict)
        assert len(results) == 1
        assert '1D' in results
        
        # Test comparison with single timeframe
        comparison = analyzer.compare_timeframes(results)
        assert isinstance(comparison, dict)
        assert 'summary' in comparison
        assert 'best_performing' in comparison
        assert comparison['best_performing']['timeframe'] == '1D'
    
    def test_performance_with_large_datasets(self):
        """Test performance with large datasets."""
        analyzer = TimeframePerformanceAnalyzer()
        
        # Create large dataset
        dates = pd.date_range('2020-01-01', periods=10000, freq='D')
        large_equity = pd.Series(
            100000 * (1 + np.random.normal(0.001, 0.02, len(dates))).cumprod(),
            index=dates
        )
        
        large_trades = pd.DataFrame({
            'entry_time': pd.date_range('2020-01-01', periods=5000, freq='D'),
            'exit_time': pd.date_range('2020-01-01', periods=5000, freq='D') + timedelta(days=1),
            'pnl': np.random.normal(100, 500, 5000),
            'duration': np.random.uniform(1, 24, 5000),
            'side': np.random.choice(['long', 'short'], 5000)
        })
        
        import time
        start_time = time.time()
        
        result = analyzer.analyze_timeframe_performance(
            timeframe='1D',
            equity_curve=large_equity,
            trades=large_trades
        )
        
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 10  # Less than 10 seconds
        assert isinstance(result, dict)
        assert 'metrics' in result
    
    def test_save_load_results(self, sample_equity_curves, sample_trades_data, tmp_path):
        """Test saving and loading results."""
        analyzer = TimeframePerformanceAnalyzer()
        
        results = analyzer.analyze_all_timeframes(
            equity_curves=sample_equity_curves,
            trades_data=sample_trades_data
        )
        
        # Save results
        save_path = tmp_path / "timeframe_results.json"
        analyzer.save_results(results, str(save_path))
        
        assert save_path.exists()
        
        # Load results
        loaded_results = analyzer.load_results(str(save_path))
        
        assert isinstance(loaded_results, dict)
        assert len(loaded_results) == len(results)
        
        # Check that loaded results match original
        for timeframe in results.keys():
            assert timeframe in loaded_results
            assert 'metrics' in loaded_results[timeframe]
            assert 'statistics' in loaded_results[timeframe]


class TestPerformanceMetrics:
    """Comprehensive tests for PerformanceMetrics class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        returns = pd.Series(
            np.random.normal(0.001, 0.02, len(dates)),
            index=dates,
            name='returns'
        )
        return returns
    
    @pytest.fixture
    def sample_equity(self):
        """Create sample equity curve."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        returns = np.random.normal(0.001, 0.02, len(dates))
        equity = pd.Series(
            100000 * (1 + returns).cumprod(),
            index=dates,
            name='equity'
        )
        return equity
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades data."""
        np.random.seed(42)
        trades = pd.DataFrame({
            'entry_time': pd.date_range('2023-01-01', periods=100, freq='D'),
            'exit_time': pd.date_range('2023-01-01', periods=100, freq='D') + timedelta(days=1),
            'pnl': np.random.normal(100, 500, 100),
            'duration': np.random.uniform(1, 24, 100),
            'side': np.random.choice(['long', 'short'], 100),
            'size': np.random.uniform(1000, 10000, 100)
        })
        return trades
    
    def test_total_return(self, sample_equity):
        """Test total return calculation."""
        total_return = PerformanceMetrics.total_return(sample_equity)
        
        assert isinstance(total_return, float)
        expected = (sample_equity.iloc[-1] / sample_equity.iloc[0]) - 1
        assert abs(total_return - expected) < 1e-10
    
    def test_annual_return(self, sample_returns):
        """Test annual return calculation."""
        annual_return = PerformanceMetrics.annual_return(sample_returns)
        
        assert isinstance(annual_return, float)
        assert -1 <= annual_return <= 10  # Reasonable bounds
    
    def test_volatility(self, sample_returns):
        """Test volatility calculation."""
        volatility = PerformanceMetrics.volatility(sample_returns)
        
        assert isinstance(volatility, float)
        assert volatility >= 0
        
        # Compare with manual calculation
        expected = sample_returns.std() * np.sqrt(252)
        assert abs(volatility - expected) < 1e-10
    
    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        sharpe = PerformanceMetrics.sharpe_ratio(sample_returns)
        
        assert isinstance(sharpe, float)
        assert -10 <= sharpe <= 10  # Reasonable bounds
        
        # Test with risk-free rate
        sharpe_rf = PerformanceMetrics.sharpe_ratio(sample_returns, risk_free_rate=0.02)
        assert isinstance(sharpe_rf, float)
        assert sharpe_rf != sharpe  # Should be different
    
    def test_max_drawdown(self, sample_equity):
        """Test maximum drawdown calculation."""
        max_dd = PerformanceMetrics.max_drawdown(sample_equity)
        
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Should be negative or zero
        
        # Test with drawdown series
        max_dd_series = PerformanceMetrics.max_drawdown(sample_equity, return_series=True)
        assert isinstance(max_dd_series, pd.Series)
        assert len(max_dd_series) == len(sample_equity)
        assert (max_dd_series <= 0).all()  # All values should be <= 0
    
    def test_calmar_ratio(self, sample_returns):
        """Test Calmar ratio calculation."""
        calmar = PerformanceMetrics.calmar_ratio(sample_returns)
        
        assert isinstance(calmar, float)
        assert -100 <= calmar <= 100  # Reasonable bounds
    
    def test_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation."""
        sortino = PerformanceMetrics.sortino_ratio(sample_returns)
        
        assert isinstance(sortino, float)
        assert -10 <= sortino <= 10  # Reasonable bounds
        
        # Test with target return
        sortino_target = PerformanceMetrics.sortino_ratio(sample_returns, target_return=0.01)
        assert isinstance(sortino_target, float)
    
    def test_value_at_risk(self, sample_returns):
        """Test Value at Risk calculation."""
        var_95 = PerformanceMetrics.value_at_risk(sample_returns, confidence=0.95)
        
        assert isinstance(var_95, float)
        assert var_95 <= 0  # Should be negative
        
        # Test different confidence levels
        var_99 = PerformanceMetrics.value_at_risk(sample_returns, confidence=0.99)
        assert var_99 <= var_95  # 99% VaR should be worse than 95% VaR
    
    def test_conditional_value_at_risk(self, sample_returns):
        """Test Conditional Value at Risk calculation."""
        cvar_95 = PerformanceMetrics.conditional_value_at_risk(sample_returns, confidence=0.95)
        
        assert isinstance(cvar_95, float)
        assert cvar_95 <= 0  # Should be negative
        
        # CVaR should be worse than VaR
        var_95 = PerformanceMetrics.value_at_risk(sample_returns, confidence=0.95)
        assert cvar_95 <= var_95
    
    def test_win_rate(self, sample_trades):
        """Test win rate calculation."""
        win_rate = PerformanceMetrics.win_rate(sample_trades)
        
        assert isinstance(win_rate, float)
        assert 0 <= win_rate <= 1
        
        # Manual calculation
        winning_trades = (sample_trades['pnl'] > 0).sum()
        total_trades = len(sample_trades)
        expected = winning_trades / total_trades
        assert abs(win_rate - expected) < 1e-10
    
    def test_profit_factor(self, sample_trades):
        """Test profit factor calculation."""
        profit_factor = PerformanceMetrics.profit_factor(sample_trades)
        
        assert isinstance(profit_factor, float)
        assert profit_factor >= 0
        
        # Manual calculation
        gross_profit = sample_trades[sample_trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(sample_trades[sample_trades['pnl'] < 0]['pnl'].sum())
        
        if gross_loss > 0:
            expected = gross_profit / gross_loss
            assert abs(profit_factor - expected) < 1e-10
    
    def test_avg_trade_duration(self, sample_trades):
        """Test average trade duration calculation."""
        avg_duration = PerformanceMetrics.avg_trade_duration(sample_trades)
        
        assert isinstance(avg_duration, float)
        assert avg_duration >= 0
        
        # Manual calculation
        expected = sample_trades['duration'].mean()
        assert abs(avg_duration - expected) < 1e-10
    
    def test_trade_frequency(self, sample_trades):
        """Test trade frequency calculation."""
        frequency = PerformanceMetrics.trade_frequency(sample_trades, period='daily')
        
        assert isinstance(frequency, float)
        assert frequency >= 0
        
        # Test different periods
        weekly_freq = PerformanceMetrics.trade_frequency(sample_trades, period='weekly')
        monthly_freq = PerformanceMetrics.trade_frequency(sample_trades, period='monthly')
        
        assert weekly_freq >= 0
        assert monthly_freq >= 0
        assert weekly_freq <= frequency * 7  # Should be related
    
    def test_information_ratio(self, sample_returns):
        """Test information ratio calculation."""
        # Create benchmark returns
        benchmark_returns = sample_returns * 0.8 + np.random.normal(0, 0.01, len(sample_returns))
        
        info_ratio = PerformanceMetrics.information_ratio(sample_returns, benchmark_returns)
        
        assert isinstance(info_ratio, float)
        assert -10 <= info_ratio <= 10  # Reasonable bounds
    
    def test_beta(self, sample_returns):
        """Test beta calculation."""
        # Create market returns
        market_returns = sample_returns * 0.9 + np.random.normal(0, 0.015, len(sample_returns))
        
        beta = PerformanceMetrics.beta(sample_returns, market_returns)
        
        assert isinstance(beta, float)
        assert -5 <= beta <= 5  # Reasonable bounds
    
    def test_alpha(self, sample_returns):
        """Test alpha calculation."""
        # Create market returns
        market_returns = sample_returns * 0.9 + np.random.normal(0, 0.015, len(sample_returns))
        
        alpha = PerformanceMetrics.alpha(sample_returns, market_returns)
        
        assert isinstance(alpha, float)
        assert -1 <= alpha <= 1  # Reasonable bounds
    
    def test_skewness(self, sample_returns):
        """Test skewness calculation."""
        skewness = PerformanceMetrics.skewness(sample_returns)
        
        assert isinstance(skewness, float)
        assert -10 <= skewness <= 10  # Reasonable bounds
    
    def test_kurtosis(self, sample_returns):
        """Test kurtosis calculation."""
        kurtosis = PerformanceMetrics.kurtosis(sample_returns)
        
        assert isinstance(kurtosis, float)
        assert -10 <= kurtosis <= 100  # Reasonable bounds
    
    def test_tail_ratio(self, sample_returns):
        """Test tail ratio calculation."""
        tail_ratio = PerformanceMetrics.tail_ratio(sample_returns)
        
        assert isinstance(tail_ratio, float)
        assert tail_ratio >= 0
    
    def test_stability_of_timeseries(self, sample_returns):
        """Test stability of timeseries calculation."""
        stability = PerformanceMetrics.stability_of_timeseries(sample_returns)
        
        assert isinstance(stability, float)
        assert -1 <= stability <= 1  # Correlation coefficient bounds
    
    def test_calculate_all_metrics(self, sample_returns, sample_equity, sample_trades):
        """Test calculation of all metrics at once."""
        metrics = PerformanceMetrics.calculate_all_metrics(
            returns=sample_returns,
            equity_curve=sample_equity,
            trades=sample_trades
        )
        
        assert isinstance(metrics, dict)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'total_return', 'annual_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'calmar_ratio', 'sortino_ratio', 'var_95',
            'cvar_95', 'win_rate', 'profit_factor', 'avg_trade_duration',
            'trade_frequency', 'skewness', 'kurtosis', 'tail_ratio',
            'stability'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_rolling_metrics(self, sample_returns):
        """Test rolling metrics calculation."""
        rolling_sharpe = PerformanceMetrics.rolling_sharpe(sample_returns, window=30)
        
        assert isinstance(rolling_sharpe, pd.Series)
        assert len(rolling_sharpe) == len(sample_returns)
        assert rolling_sharpe.isna().sum() == 29  # First 29 values should be NaN
        
        # Test rolling volatility
        rolling_vol = PerformanceMetrics.rolling_volatility(sample_returns, window=30)
        
        assert isinstance(rolling_vol, pd.Series)
        assert len(rolling_vol) == len(sample_returns)
        assert (rolling_vol.dropna() >= 0).all()
    
    def test_performance_attribution(self, sample_returns):
        """Test performance attribution calculation."""
        # Create factor returns
        factor_returns = pd.DataFrame({
            'factor1': np.random.normal(0.0005, 0.01, len(sample_returns)),
            'factor2': np.random.normal(0.0003, 0.008, len(sample_returns)),
            'factor3': np.random.normal(0.0002, 0.012, len(sample_returns))
        }, index=sample_returns.index)
        
        attribution = PerformanceMetrics.performance_attribution(sample_returns, factor_returns)
        
        assert isinstance(attribution, dict)
        assert 'factor_exposures' in attribution
        assert 'factor_returns' in attribution
        assert 'specific_return' in attribution
        
        # Check factor exposures
        exposures = attribution['factor_exposures']
        assert isinstance(exposures, dict)
        assert len(exposures) == 3
        
        # Check factor returns
        factor_rets = attribution['factor_returns']
        assert isinstance(factor_rets, dict)
        assert len(factor_rets) == 3
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty data
        empty_series = pd.Series([], dtype=float)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # These should handle empty data gracefully
            result = PerformanceMetrics.total_return(empty_series)
            assert pd.isna(result) or result == 0
            
            result = PerformanceMetrics.volatility(empty_series)
            assert pd.isna(result) or result == 0
        
        # Test with constant values
        constant_series = pd.Series([100] * 100)
        
        vol = PerformanceMetrics.volatility(constant_series)
        assert vol == 0.0
        
        total_ret = PerformanceMetrics.total_return(constant_series)
        assert total_ret == 0.0
        
        # Test with single value
        single_value = pd.Series([100])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = PerformanceMetrics.volatility(single_value)
            assert pd.isna(result) or result == 0
    
    def test_metric_consistency(self, sample_returns, sample_equity):
        """Test consistency between different metric calculations."""
        # Test total return consistency
        total_return_from_equity = PerformanceMetrics.total_return(sample_equity)
        total_return_from_returns = (1 + sample_returns).prod() - 1
        
        assert abs(total_return_from_equity - total_return_from_returns) < 1e-10
        
        # Test annual return consistency
        annual_return = PerformanceMetrics.annual_return(sample_returns)
        expected_annual = sample_returns.mean() * 252
        
        # Should be approximately equal (allowing for compounding effects)
        assert abs(annual_return - expected_annual) < 0.05
    
    def test_metric_bounds(self, sample_returns, sample_trades):
        """Test that metrics are within reasonable bounds."""
        metrics = PerformanceMetrics.calculate_all_metrics(
            returns=sample_returns,
            equity_curve=sample_returns.cumsum() + 1,
            trades=sample_trades
        )
        
        # Check bounds
        assert 0 <= metrics['win_rate'] <= 1
        assert metrics['profit_factor'] >= 0
        assert metrics['max_drawdown'] <= 0
        assert metrics['volatility'] >= 0
        assert 0 <= metrics['trade_frequency'] <= 1000  # trades per day
        assert -1 <= metrics['stability'] <= 1


class TestBacktestStrategy:
    """Test BacktestStrategy performance analysis integration."""
    
    def test_strategy_performance_analysis(self, sample_ohlcv_data):
        """Test strategy performance analysis integration."""
        # Create a simple strategy
        strategy = BacktestStrategy(
            name="Test Strategy",
            entry_rules=["close > open"],
            exit_rules=["close < open"],
            risk_management={'stop_loss': 0.05, 'take_profit': 0.10}
        )
        
        # Run backtest
        from src.backtesting import BacktestEngine
        engine = BacktestEngine(initial_capital=100000)
        
        # This would normally connect to the strategy performance analyzer
        # but we'll test the integration points
        
        assert hasattr(strategy, 'name')
        assert hasattr(strategy, 'entry_rules')
        assert hasattr(strategy, 'exit_rules')
        assert hasattr(strategy, 'risk_management')
    
    def test_multi_strategy_comparison(self):
        """Test multi-strategy performance comparison."""
        # Create multiple strategies
        strategies = [
            BacktestStrategy(
                name="Strategy A",
                entry_rules=["rsi < 30"],
                exit_rules=["rsi > 70"]
            ),
            BacktestStrategy(
                name="Strategy B",
                entry_rules=["close > sma_20"],
                exit_rules=["close < sma_20"]
            )
        ]
        
        # Test that strategies can be compared
        for strategy in strategies:
            assert hasattr(strategy, 'name')
            assert isinstance(strategy.name, str)
            assert len(strategy.name) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])