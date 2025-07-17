"""
Comprehensive test suite for Performance Attribution module.

This test suite aims for 100% coverage of the performance_attribution.py module
by testing all methods, edge cases, error conditions, and data flows.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import warnings
from typing import Dict, List, Any

# Import modules to test
from src.analysis.performance_attribution import (
    PerformanceAttributor, AttributionResult, TimeSeriesAttribution
)


class TestPerformanceAttributor:
    """Comprehensive tests for PerformanceAttributor class"""
    
    @pytest.fixture
    def attributor(self):
        """Create PerformanceAttributor instance"""
        return PerformanceAttributor()
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data"""
        return [
            {
                'entry_date': '2023-01-01',
                'exit_date': '2023-01-15',
                'return': 0.05,
                'position_weight': 0.2,
                'confluence_score': 0.85,
                'exit_reason': 'take_profit',
                'max_risk': 0.02,
                'timeframe_scores': {'1H': 0.8, '4H': 0.9, '1D': 0.7}
            },
            {
                'entry_date': '2023-02-01',
                'exit_date': '2023-02-10',
                'return': -0.02,
                'position_weight': 0.15,
                'confluence_score': 0.72,
                'exit_reason': 'stop_loss',
                'max_risk': 0.03,
                'timeframe_scores': {'1H': 0.6, '4H': 0.7, '1D': 0.8}
            },
            {
                'entry_date': '2023-03-01',
                'exit_date': '2023-03-20',
                'return': 0.08,
                'position_weight': 0.25,
                'confluence_score': 0.55,
                'exit_reason': 'signal',
                'max_risk': 0.025,
                'timeframe_scores': {'1H': 0.5, '4H': 0.6, '1D': 0.7}
            }
        ]
    
    @pytest.fixture
    def portfolio_returns(self):
        """Create sample portfolio returns"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        returns = np.random.normal(0.0005, 0.01, len(dates))
        return pd.Series(returns, index=dates)
    
    @pytest.fixture
    def market_returns(self):
        """Create sample market returns"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        returns = np.random.normal(0.0003, 0.008, len(dates))
        return pd.Series(returns, index=dates)
    
    @pytest.fixture
    def factor_returns(self):
        """Create sample factor returns"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        return {
            'momentum': pd.Series(np.random.normal(0.0002, 0.005, len(dates)), index=dates),
            'value': pd.Series(np.random.normal(0.0001, 0.004, len(dates)), index=dates),
            'volatility': pd.Series(np.random.normal(-0.0001, 0.006, len(dates)), index=dates)
        }
    
    def test_initialization(self, attributor):
        """Test PerformanceAttributor initialization"""
        assert isinstance(attributor.attribution_history, list)
        assert len(attributor.attribution_history) == 0
        assert isinstance(attributor.factor_exposures, dict)
        assert attributor.benchmark_data is None
    
    def test_calculate_return_attribution_complete(self, attributor, sample_trades, 
                                                   portfolio_returns, market_returns, 
                                                   factor_returns):
        """Test complete return attribution calculation"""
        result = attributor.calculate_return_attribution(
            trades=sample_trades,
            portfolio_returns=portfolio_returns,
            market_returns=market_returns,
            factor_returns=factor_returns
        )
        
        # Verify result structure
        assert isinstance(result, AttributionResult)
        assert isinstance(result.total_return, float)
        assert isinstance(result.attribution_components, dict)
        assert isinstance(result.factor_contributions, dict)
        assert isinstance(result.timing_contribution, float)
        assert isinstance(result.selection_contribution, float)
        assert isinstance(result.interaction_effect, float)
        assert isinstance(result.risk_adjusted_attribution, dict)
        
        # Verify components
        assert 'timing' in result.attribution_components
        assert 'selection' in result.attribution_components
        assert 'confluence' in result.attribution_components
        assert 'risk_management' in result.attribution_components
        assert 'tf_1H' in result.attribution_components
        assert 'tf_4H' in result.attribution_components
        
        # Verify factor contributions
        assert 'momentum' in result.factor_contributions
        assert 'value' in result.factor_contributions
        assert 'volatility' in result.factor_contributions
        
        # Verify history
        assert len(attributor.attribution_history) == 1
        assert attributor.attribution_history[0] == result
    
    def test_calculate_return_attribution_no_factors(self, attributor, sample_trades,
                                                     portfolio_returns, market_returns):
        """Test return attribution without factor returns"""
        result = attributor.calculate_return_attribution(
            trades=sample_trades,
            portfolio_returns=portfolio_returns,
            market_returns=market_returns,
            factor_returns=None
        )
        
        assert isinstance(result, AttributionResult)
        assert len(result.factor_contributions) == 0
    
    def test_calculate_return_attribution_empty_trades(self, attributor, 
                                                       portfolio_returns, market_returns):
        """Test return attribution with empty trades"""
        result = attributor.calculate_return_attribution(
            trades=[],
            portfolio_returns=portfolio_returns,
            market_returns=market_returns
        )
        
        assert result.timing_contribution == 0.0
        assert result.selection_contribution == 0.0
        assert result.attribution_components['confluence'] == 0.0
        assert result.attribution_components['risk_management'] == 0.0
    
    def test_calculate_timing_attribution(self, attributor, sample_trades, market_returns):
        """Test timing attribution calculation"""
        timing = attributor._calculate_timing_attribution(sample_trades, market_returns)
        assert isinstance(timing, float)
        
        # Test with empty trades
        timing_empty = attributor._calculate_timing_attribution([], market_returns)
        assert timing_empty == 0.0
        
        # Test with no matching dates
        future_trades = [{
            'entry_date': '2025-01-01',
            'exit_date': '2025-01-15',
            'position_weight': 0.1
        }]
        timing_future = attributor._calculate_timing_attribution(future_trades, market_returns)
        assert timing_future == 0.0  # Should be 0 for no matching dates
    
    def test_calculate_selection_attribution(self, attributor, sample_trades, 
                                           portfolio_returns, market_returns):
        """Test selection attribution calculation"""
        selection = attributor._calculate_selection_attribution(
            sample_trades, portfolio_returns, market_returns
        )
        assert isinstance(selection, float)
        
        # Test with empty trades
        selection_empty = attributor._calculate_selection_attribution(
            [], portfolio_returns, market_returns
        )
        assert selection_empty == 0.0
    
    def test_calculate_confluence_attribution(self, attributor, sample_trades):
        """Test confluence attribution calculation"""
        confluence = attributor._calculate_confluence_attribution(sample_trades)
        assert isinstance(confluence, float)
        
        # Test with empty trades
        confluence_empty = attributor._calculate_confluence_attribution([])
        assert confluence_empty == 0.0
        
        # Test with different confluence scores
        high_confluence_trades = [
            {'confluence_score': 0.9, 'return': 0.05},
            {'confluence_score': 0.85, 'return': 0.03}
        ]
        confluence_high = attributor._calculate_confluence_attribution(high_confluence_trades)
        assert isinstance(confluence_high, float)
        
        # Test with only low confluence
        low_confluence_trades = [
            {'confluence_score': 0.5, 'return': -0.02},
            {'confluence_score': 0.4, 'return': -0.03}
        ]
        confluence_low = attributor._calculate_confluence_attribution(low_confluence_trades)
        assert isinstance(confluence_low, float)
    
    def test_calculate_risk_attribution(self, attributor, sample_trades):
        """Test risk management attribution calculation"""
        risk = attributor._calculate_risk_attribution(sample_trades)
        assert isinstance(risk, float)
        
        # Test with empty trades
        risk_empty = attributor._calculate_risk_attribution([])
        assert risk_empty == 0.0
        
        # Test with various exit reasons
        risk_trades = [
            {'exit_reason': 'stop_loss', 'return': -0.01, 'max_risk': 0.02},
            {'exit_reason': 'take_profit', 'return': 0.03, 'max_risk': 0.02},
            {'exit_reason': 'signal', 'return': 0.01, 'max_risk': 0.02}
        ]
        risk_mixed = attributor._calculate_risk_attribution(risk_trades)
        assert isinstance(risk_mixed, float)
    
    def test_calculate_timeframe_attribution(self, attributor, sample_trades):
        """Test timeframe attribution calculation"""
        tf_attr = attributor._calculate_timeframe_attribution(sample_trades)
        assert isinstance(tf_attr, dict)
        assert 'tf_1H' in tf_attr
        assert 'tf_4H' in tf_attr
        assert 'tf_1D' in tf_attr
        assert 'tf_1W' in tf_attr
        assert 'tf_1M' in tf_attr
        
        # Test with trades missing timeframe scores
        trades_no_tf = [{'return': 0.05}]
        tf_attr_empty = attributor._calculate_timeframe_attribution(trades_no_tf)
        assert all(v == 0.0 for v in tf_attr_empty.values())
    
    def test_calculate_factor_attribution(self, attributor, portfolio_returns, factor_returns):
        """Test factor attribution calculation"""
        factor_attr = attributor._calculate_factor_attribution(
            portfolio_returns, factor_returns
        )
        assert isinstance(factor_attr, dict)
        assert 'momentum' in factor_attr
        assert 'value' in factor_attr
        assert 'volatility' in factor_attr
        
        # Test with misaligned dates
        short_portfolio = portfolio_returns[:10]
        factor_attr_short = attributor._calculate_factor_attribution(
            short_portfolio, factor_returns
        )
        assert all(v == 0.0 for v in factor_attr_short.values())
        
        # Test with empty factor returns
        empty_factors = {
            'empty': pd.Series([], dtype=float)
        }
        factor_attr_empty = attributor._calculate_factor_attribution(
            portfolio_returns, empty_factors
        )
        assert 'empty' in factor_attr_empty
        assert factor_attr_empty['empty'] == 0.0
    
    def test_calculate_risk_adjusted_attribution(self, attributor):
        """Test risk-adjusted attribution calculation"""
        portfolio_returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        components = {
            'timing': 0.05,
            'selection': 0.03,
            'confluence': 0.02
        }
        
        risk_adj = attributor._calculate_risk_adjusted_attribution(
            components, portfolio_returns
        )
        assert isinstance(risk_adj, dict)
        assert 'timing_risk_adj' in risk_adj
        assert 'selection_risk_adj' in risk_adj
        assert 'confluence_risk_adj' in risk_adj
        
        # Test with zero volatility
        zero_vol_returns = pd.Series(np.zeros(252))
        risk_adj_zero = attributor._calculate_risk_adjusted_attribution(
            components, zero_vol_returns
        )
        assert all(v == 0 for v in risk_adj_zero.values())
    
    def test_calculate_time_series_attribution(self, attributor, sample_trades):
        """Test time series attribution calculation"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        portfolio_values = pd.Series(10000 + np.cumsum(np.random.randn(len(dates)) * 10), 
                                   index=dates)
        
        ts_attr = attributor.calculate_time_series_attribution(
            trades=sample_trades,
            portfolio_values=portfolio_values,
            window=20
        )
        
        assert isinstance(ts_attr, TimeSeriesAttribution)
        assert isinstance(ts_attr.dates, pd.DatetimeIndex)
        assert isinstance(ts_attr.cumulative_attribution, pd.DataFrame)
        assert isinstance(ts_attr.period_attribution, pd.DataFrame)
        assert isinstance(ts_attr.rolling_attribution, pd.DataFrame)
        
        # Verify columns
        expected_cols = ['selection', 'timing', 'confluence', 'trade_count']
        for col in expected_cols:
            assert col in ts_attr.period_attribution.columns
    
    def test_decompose_alpha(self, attributor, portfolio_returns, market_returns):
        """Test alpha decomposition"""
        decomposition = attributor.decompose_alpha(
            strategy_returns=portfolio_returns,
            benchmark_returns=market_returns,
            risk_free_rate=0.02
        )
        
        assert isinstance(decomposition, dict)
        assert 'total_alpha' in decomposition
        assert 'selection_alpha' in decomposition
        assert 'timing_alpha' in decomposition
        assert 'risk_alpha' in decomposition
        assert 'information_ratio' in decomposition
        assert 'treynor_ratio' in decomposition
        assert 'm_squared' in decomposition
        assert 'beta' in decomposition
        assert 'tracking_error' in decomposition
        assert 'active_return' in decomposition
        
        # Test with zero variance benchmark
        zero_benchmark = pd.Series(np.zeros(len(market_returns)), index=market_returns.index)
        decomp_zero = attributor.decompose_alpha(
            strategy_returns=portfolio_returns,
            benchmark_returns=zero_benchmark
        )
        assert decomp_zero['beta'] == 1.0
        
        # Test with zero volatility strategy
        zero_strategy = pd.Series(np.zeros(len(portfolio_returns)), index=portfolio_returns.index)
        decomp_zero_strat = attributor.decompose_alpha(
            strategy_returns=zero_strategy,
            benchmark_returns=market_returns
        )
        assert decomp_zero_strat['treynor_ratio'] == 0
    
    def test_analyze_factor_exposures(self, attributor, portfolio_returns, factor_returns):
        """Test factor exposure analysis"""
        exposures = attributor.analyze_factor_exposures(
            returns=portfolio_returns,
            factors=factor_returns,
            lookback=60
        )
        
        assert isinstance(exposures, dict)
        for factor_name in factor_returns.keys():
            assert factor_name in exposures
            factor_exp = exposures[factor_name]
            assert 'current_beta' in factor_exp
            assert 'avg_beta' in factor_exp
            assert 'beta_stability' in factor_exp
            assert 'avg_correlation' in factor_exp
            assert 'max_correlation' in factor_exp
        
        # Test with short data
        short_returns = portfolio_returns[:30]
        exposures_short = attributor.analyze_factor_exposures(
            returns=short_returns,
            factors=factor_returns,
            lookback=60
        )
        assert len(exposures_short) == 0
        
        # Test with zero variance factor
        zero_factor = {
            'zero': pd.Series(np.zeros(len(portfolio_returns)), index=portfolio_returns.index)
        }
        exposures_zero = attributor.analyze_factor_exposures(
            returns=portfolio_returns,
            factors=zero_factor,
            lookback=60
        )
        assert exposures_zero['zero']['current_beta'] == 0
    
    def test_calculate_performance_consistency(self, attributor, portfolio_returns, 
                                             market_returns):
        """Test performance consistency calculation"""
        consistency = attributor.calculate_performance_consistency(
            returns=portfolio_returns,
            benchmark_returns=market_returns,
            periods=['daily', 'weekly', 'monthly']
        )
        
        assert isinstance(consistency, dict)
        for period in ['daily', 'weekly', 'monthly']:
            assert period in consistency
            metrics = consistency[period]
            assert 'win_rate' in metrics
            assert 'outperformance_rate' in metrics
            assert 'avg_return' in metrics
            assert 'return_volatility' in metrics
            assert 'downside_deviation' in metrics
            assert 'best_period' in metrics
            assert 'worst_period' in metrics
            assert 'positive_periods' in metrics
            assert 'negative_periods' in metrics
        
        # Test with unknown period
        consistency_unknown = attributor.calculate_performance_consistency(
            returns=portfolio_returns,
            benchmark_returns=market_returns,
            periods=['quarterly']
        )
        assert 'quarterly' not in consistency_unknown
        
        # Test with all negative returns
        negative_returns = pd.Series(np.random.uniform(-0.02, -0.001, len(portfolio_returns)), 
                                   index=portfolio_returns.index)
        consistency_neg = attributor.calculate_performance_consistency(
            returns=negative_returns,
            benchmark_returns=market_returns,
            periods=['daily']
        )
        assert consistency_neg['daily']['positive_periods'] == 0
    
    def test_generate_attribution_report(self, attributor, sample_trades, 
                                       portfolio_returns, market_returns):
        """Test attribution report generation"""
        # First create some attribution history
        attributor.calculate_return_attribution(
            trades=sample_trades,
            portfolio_returns=portfolio_returns,
            market_returns=market_returns
        )
        
        report = attributor.generate_attribution_report()
        
        assert isinstance(report, dict)
        assert 'total_return' in report
        assert 'attribution_components' in report
        assert 'component_analysis' in report
        assert 'factor_contributions' in report
        assert 'risk_adjusted_attribution' in report
        assert 'summary' in report
        
        # Check component analysis
        comp_analysis = report['component_analysis']
        for component in report['attribution_components'].keys():
            assert component in comp_analysis
            comp_metrics = comp_analysis[component]
            assert 'current' in comp_metrics
            assert 'average' in comp_metrics
            assert 'volatility' in comp_metrics
            assert 'trend' in comp_metrics
            assert 'contribution_pct' in comp_metrics
        
        # Check summary
        summary = report['summary']
        assert 'primary_return_driver' in summary
        assert 'selection_vs_timing' in summary
        
        # Test with empty history
        empty_attributor = PerformanceAttributor()
        empty_report = empty_attributor.generate_attribution_report()
        assert empty_report == {}
    
    def test_edge_cases_and_error_handling(self, attributor):
        """Test edge cases and error handling"""
        # Test with single trade
        single_trade = [{
            'entry_date': '2023-01-01',
            'exit_date': '2023-01-02',
            'return': 0.01,
            'position_weight': 1.0,
            'confluence_score': 0.7,
            'exit_reason': 'signal',
            'max_risk': 0.02,
            'timeframe_scores': {'1H': 0.8}
        }]
        
        dates = pd.date_range(start='2023-01-01', end='2023-01-02', freq='D')
        returns = pd.Series([0.01, 0.01], index=dates)
        
        result = attributor.calculate_return_attribution(
            trades=single_trade,
            portfolio_returns=returns,
            market_returns=returns
        )
        assert isinstance(result, AttributionResult)
        
        # Test with trades having missing fields - make sure entry_date and exit_date exist
        incomplete_trades = [
            {'entry_date': '2023-01-01', 'exit_date': '2023-01-02'},
            {'entry_date': '2023-01-01', 'exit_date': '2023-01-02', 'return': 0.05},
            {'entry_date': '2023-01-01', 'exit_date': '2023-01-02'}
        ]
        
        result = attributor.calculate_return_attribution(
            trades=incomplete_trades,
            portfolio_returns=returns,
            market_returns=returns
        )
        assert isinstance(result, AttributionResult)
        
        # Test with extreme values
        extreme_trades = [{
            'entry_date': '2023-01-01',
            'exit_date': '2023-01-02',
            'return': 10.0,  # 1000% return
            'position_weight': 0.1,
            'confluence_score': 1.0,
            'exit_reason': 'take_profit',
            'max_risk': 0.01,
            'timeframe_scores': {'1H': 1.0}
        }]
        
        result = attributor.calculate_return_attribution(
            trades=extreme_trades,
            portfolio_returns=returns,
            market_returns=returns
        )
        assert isinstance(result, AttributionResult)


def test_dataclass_properties():
    """Test dataclass properties and methods"""
    # Test AttributionResult
    result = AttributionResult(
        total_return=0.15,
        attribution_components={'timing': 0.05, 'selection': 0.08},
        factor_contributions={'momentum': 0.02},
        timing_contribution=0.05,
        selection_contribution=0.08,
        interaction_effect=0.02,
        risk_adjusted_attribution={'timing_risk_adj': 0.25}
    )
    
    assert result.total_return == 0.15
    assert result.timing_contribution == 0.05
    assert len(result.attribution_components) == 2
    
    # Test TimeSeriesAttribution
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    ts_attr = TimeSeriesAttribution(
        dates=dates,
        cumulative_attribution=pd.DataFrame({'selection': np.arange(10)}, index=dates),
        period_attribution=pd.DataFrame({'selection': np.ones(10)}, index=dates),
        rolling_attribution=pd.DataFrame({'selection': np.ones(10) * 0.5}, index=dates)
    )
    
    assert len(ts_attr.dates) == 10
    assert ts_attr.cumulative_attribution.shape[0] == 10
    assert ts_attr.period_attribution.shape[0] == 10
    assert ts_attr.rolling_attribution.shape[0] == 10