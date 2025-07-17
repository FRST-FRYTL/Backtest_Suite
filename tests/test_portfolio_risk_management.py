"""
Comprehensive tests for portfolio and risk management modules.

This module tests portfolio optimization, risk management, position sizing,
rebalancing, and stress testing functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings

from src.portfolio.portfolio_optimizer import PortfolioOptimizer
from src.portfolio.risk_manager import RiskManager
from src.portfolio.position_sizer import PositionSizer
from src.portfolio.rebalancer import PortfolioRebalancer, RebalanceFrequency, RebalanceMethod
from src.portfolio.stress_testing import StressTester
from src.portfolio.risk_dashboard import RiskDashboard
from src.risk_management.enhanced_risk_manager import EnhancedRiskManager


class TestPortfolioOptimizer:
    """Test portfolio optimization functionality."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for multiple assets."""
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=252, freq='D')
        
        # Create correlated returns for realistic portfolio
        n_assets = 5
        correlation_matrix = np.random.uniform(0.3, 0.7, (n_assets, n_assets))
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Generate returns
        returns = np.random.multivariate_normal(
            mean=[0.001, 0.0008, 0.0012, 0.0015, 0.0005],
            cov=correlation_matrix * 0.0004,
            size=252
        )
        
        return pd.DataFrame(
            returns,
            index=dates,
            columns=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        )
    
    @pytest.fixture
    def sample_prices(self, sample_returns):
        """Create sample price data from returns."""
        return (1 + sample_returns).cumprod() * 100
    
    def test_optimizer_initialization(self):
        """Test portfolio optimizer initialization."""
        optimizer = PortfolioOptimizer()
        
        assert optimizer.risk_free_rate == 0.02
        assert optimizer.optimization_method == 'mean_variance'
        assert optimizer.constraints == {}
        assert optimizer.bounds == (0.0, 1.0)
    
    def test_custom_initialization(self):
        """Test optimizer with custom parameters."""
        optimizer = PortfolioOptimizer(
            risk_free_rate=0.03,
            optimization_method='risk_parity',
            bounds=(0.05, 0.4)
        )
        
        assert optimizer.risk_free_rate == 0.03
        assert optimizer.optimization_method == 'risk_parity'
        assert optimizer.bounds == (0.05, 0.4)
    
    def test_mean_variance_optimization(self, sample_returns):
        """Test mean-variance optimization."""
        optimizer = PortfolioOptimizer(optimization_method='mean_variance')
        
        weights = optimizer.optimize(sample_returns)
        
        assert len(weights) == len(sample_returns.columns)
        assert abs(weights.sum() - 1.0) < 1e-6  # Weights sum to 1
        assert all(w >= 0 for w in weights)  # Long-only constraint
        assert all(w <= 1 for w in weights)  # Individual asset constraint
    
    def test_risk_parity_optimization(self, sample_returns):
        """Test risk parity optimization."""
        optimizer = PortfolioOptimizer(optimization_method='risk_parity')
        
        weights = optimizer.optimize(sample_returns)
        
        assert len(weights) == len(sample_returns.columns)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)
        
        # Check that risk contributions are approximately equal
        cov_matrix = sample_returns.cov()
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        risk_contributions = weights * (cov_matrix @ weights) / portfolio_vol
        
        # Risk contributions should be similar for risk parity
        assert np.std(risk_contributions) < 0.05
    
    def test_minimum_variance_optimization(self, sample_returns):
        """Test minimum variance optimization."""
        optimizer = PortfolioOptimizer(optimization_method='min_variance')
        
        weights = optimizer.optimize(sample_returns)
        
        assert len(weights) == len(sample_returns.columns)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)
        
        # Check that this is indeed minimum variance
        cov_matrix = sample_returns.cov()
        portfolio_variance = weights.T @ cov_matrix @ weights
        
        # Should be lower than equal-weight portfolio
        equal_weights = np.array([0.2] * 5)
        equal_variance = equal_weights.T @ cov_matrix @ equal_weights
        assert portfolio_variance <= equal_variance
    
    def test_maximum_sharpe_optimization(self, sample_returns):
        """Test maximum Sharpe ratio optimization."""
        optimizer = PortfolioOptimizer(optimization_method='max_sharpe')
        
        weights = optimizer.optimize(sample_returns)
        
        assert len(weights) == len(sample_returns.columns)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)
        
        # Calculate Sharpe ratio
        portfolio_return = weights @ sample_returns.mean() * 252
        portfolio_vol = np.sqrt(weights.T @ sample_returns.cov() @ weights * 252)
        sharpe_ratio = (portfolio_return - optimizer.risk_free_rate) / portfolio_vol
        
        assert sharpe_ratio > 0  # Should be positive
    
    def test_constraints_application(self, sample_returns):
        """Test portfolio constraints."""
        # Test sector constraints
        sector_constraints = {
            'AAPL': {'min': 0.1, 'max': 0.3},
            'GOOGL': {'min': 0.0, 'max': 0.25}
        }
        
        optimizer = PortfolioOptimizer(constraints=sector_constraints)
        weights = optimizer.optimize(sample_returns)
        
        assert 0.1 <= weights[0] <= 0.3  # AAPL constraint
        assert 0.0 <= weights[1] <= 0.25  # GOOGL constraint
    
    def test_turnover_constraint(self, sample_returns):
        """Test turnover constraint."""
        current_weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        
        optimizer = PortfolioOptimizer(
            constraints={'turnover': 0.1},
            current_weights=current_weights
        )
        
        new_weights = optimizer.optimize(sample_returns)
        
        # Calculate turnover
        turnover = np.sum(np.abs(new_weights - current_weights))
        assert turnover <= 0.1
    
    def test_portfolio_metrics(self, sample_returns):
        """Test portfolio metrics calculation."""
        optimizer = PortfolioOptimizer()
        weights = optimizer.optimize(sample_returns)
        
        metrics = optimizer.calculate_portfolio_metrics(weights, sample_returns)
        
        assert 'expected_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'var_95' in metrics
        assert 'cvar_95' in metrics
        
        # Check metric ranges
        assert metrics['expected_return'] > -1  # Reasonable return
        assert metrics['volatility'] > 0  # Positive volatility
        assert metrics['var_95'] < 0  # VaR should be negative
        assert metrics['cvar_95'] < metrics['var_95']  # CVaR worse than VaR
    
    def test_efficient_frontier(self, sample_returns):
        """Test efficient frontier calculation."""
        optimizer = PortfolioOptimizer()
        
        frontier = optimizer.calculate_efficient_frontier(sample_returns, n_points=10)
        
        assert len(frontier) == 10
        assert all('weights' in point for point in frontier)
        assert all('return' in point for point in frontier)
        assert all('volatility' in point for point in frontier)
        assert all('sharpe_ratio' in point for point in frontier)
        
        # Check that frontier is monotonic
        returns = [point['return'] for point in frontier]
        volatilities = [point['volatility'] for point in frontier]
        
        assert all(returns[i] <= returns[i+1] for i in range(len(returns)-1))
        assert all(volatilities[i] <= volatilities[i+1] for i in range(len(volatilities)-1))
    
    def test_black_litterman_optimization(self, sample_returns):
        """Test Black-Litterman optimization."""
        # Create sample views
        views = pd.DataFrame({
            'asset': ['AAPL', 'GOOGL'],
            'view_return': [0.12, 0.08],
            'confidence': [0.8, 0.6]
        })
        
        optimizer = PortfolioOptimizer(optimization_method='black_litterman')
        weights = optimizer.optimize(sample_returns, views=views)
        
        assert len(weights) == len(sample_returns.columns)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)
    
    def test_resampled_efficiency(self, sample_returns):
        """Test resampled efficiency optimization."""
        optimizer = PortfolioOptimizer(optimization_method='resampled_efficiency')
        
        weights = optimizer.optimize(sample_returns, n_simulations=100)
        
        assert len(weights) == len(sample_returns.columns)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)
    
    def test_optimization_with_missing_data(self, sample_returns):
        """Test optimization with missing data."""
        # Introduce missing data
        sample_returns_missing = sample_returns.copy()
        sample_returns_missing.iloc[10:20, 0] = np.nan
        sample_returns_missing.iloc[50:60, 2] = np.nan
        
        optimizer = PortfolioOptimizer()
        weights = optimizer.optimize(sample_returns_missing)
        
        assert len(weights) == len(sample_returns.columns)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)
    
    def test_optimization_failure_handling(self, sample_returns):
        """Test handling of optimization failures."""
        # Create problematic constraints
        bad_constraints = {
            'AAPL': {'min': 0.8, 'max': 0.9},
            'GOOGL': {'min': 0.8, 'max': 0.9}
        }
        
        optimizer = PortfolioOptimizer(constraints=bad_constraints)
        
        # Should handle infeasible constraints gracefully
        with pytest.raises(ValueError):
            optimizer.optimize(sample_returns)


class TestRiskManager:
    """Test risk management functionality."""
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Portfolio returns
        returns = np.random.normal(0.001, 0.02, 252)
        portfolio_value = pd.Series(100000 * (1 + returns).cumprod(), index=dates)
        
        # Holdings
        holdings = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
            'quantity': [100, 50, 80, 30, 40],
            'price': [150, 2800, 300, 250, 3200],
            'value': [15000, 140000, 24000, 7500, 128000]
        })
        
        return {
            'portfolio_value': portfolio_value,
            'holdings': holdings,
            'returns': pd.Series(returns, index=dates)
        }
    
    def test_risk_manager_initialization(self):
        """Test risk manager initialization."""
        risk_manager = RiskManager()
        
        assert risk_manager.var_confidence == 0.95
        assert risk_manager.max_position_size == 0.05
        assert risk_manager.max_sector_exposure == 0.3
        assert risk_manager.stop_loss_threshold == 0.02
    
    def test_var_calculation(self, sample_portfolio):
        """Test Value at Risk calculation."""
        risk_manager = RiskManager()
        
        var_95 = risk_manager.calculate_var(
            sample_portfolio['returns'],
            confidence=0.95,
            method='historical'
        )
        
        assert var_95 < 0  # VaR should be negative
        assert -0.1 < var_95 < 0  # Reasonable range
        
        # Test parametric VaR
        var_parametric = risk_manager.calculate_var(
            sample_portfolio['returns'],
            confidence=0.95,
            method='parametric'
        )
        
        assert var_parametric < 0
        assert abs(var_95 - var_parametric) < 0.01  # Should be similar
    
    def test_cvar_calculation(self, sample_portfolio):
        """Test Conditional Value at Risk calculation."""
        risk_manager = RiskManager()
        
        cvar_95 = risk_manager.calculate_cvar(
            sample_portfolio['returns'],
            confidence=0.95
        )
        
        assert cvar_95 < 0  # CVaR should be negative
        
        # CVaR should be worse than VaR
        var_95 = risk_manager.calculate_var(
            sample_portfolio['returns'],
            confidence=0.95
        )
        assert cvar_95 < var_95
    
    def test_maximum_drawdown(self, sample_portfolio):
        """Test maximum drawdown calculation."""
        risk_manager = RiskManager()
        
        max_dd = risk_manager.calculate_max_drawdown(
            sample_portfolio['portfolio_value']
        )
        
        assert max_dd <= 0  # Drawdown should be negative or zero
        assert max_dd >= -1  # Shouldn't exceed -100%
    
    def test_position_size_check(self, sample_portfolio):
        """Test position size checking."""
        risk_manager = RiskManager(max_position_size=0.1)
        
        total_value = sample_portfolio['holdings']['value'].sum()
        
        violations = risk_manager.check_position_sizes(
            sample_portfolio['holdings'],
            total_value
        )
        
        assert isinstance(violations, list)
        
        # GOOGL should violate (140000 / 314500 > 0.1)
        googl_violations = [v for v in violations if v['symbol'] == 'GOOGL']
        assert len(googl_violations) > 0
    
    def test_sector_exposure_check(self, sample_portfolio):
        """Test sector exposure checking."""
        risk_manager = RiskManager(max_sector_exposure=0.4)
        
        # Add sector information
        holdings_with_sectors = sample_portfolio['holdings'].copy()
        holdings_with_sectors['sector'] = ['Technology', 'Technology', 'Technology', 'Consumer', 'Consumer']
        
        total_value = holdings_with_sectors['value'].sum()
        
        violations = risk_manager.check_sector_exposure(
            holdings_with_sectors,
            total_value
        )
        
        assert isinstance(violations, list)
        
        # Technology sector should violate (179000 / 314500 > 0.4)
        tech_violations = [v for v in violations if v['sector'] == 'Technology']
        assert len(tech_violations) > 0
    
    def test_correlation_analysis(self, sample_portfolio):
        """Test correlation analysis."""
        risk_manager = RiskManager()
        
        # Create multi-asset returns
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        asset_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'GOOGL': np.random.normal(0.0008, 0.025, 252),
            'MSFT': np.random.normal(0.0012, 0.018, 252)
        }, index=dates)
        
        # Make AAPL and MSFT highly correlated
        asset_returns['MSFT'] = asset_returns['AAPL'] * 0.8 + asset_returns['MSFT'] * 0.2
        
        high_corr_pairs = risk_manager.find_high_correlations(
            asset_returns,
            threshold=0.7
        )
        
        assert len(high_corr_pairs) > 0
        
        # Should find AAPL-MSFT pair
        aapl_msft_pair = [
            pair for pair in high_corr_pairs 
            if set(pair['assets']) == {'AAPL', 'MSFT'}
        ]
        assert len(aapl_msft_pair) > 0
    
    def test_stress_testing(self, sample_portfolio):
        """Test stress testing scenarios."""
        risk_manager = RiskManager()
        
        # Define stress scenarios
        stress_scenarios = {
            'market_crash': {'factor': -0.2, 'description': 'Market crash scenario'},
            'volatility_spike': {'factor': 2.0, 'description': 'Volatility spike'},
            'sector_rotation': {'factor': -0.15, 'description': 'Sector rotation'}
        }
        
        stress_results = risk_manager.run_stress_tests(
            sample_portfolio['portfolio_value'],
            stress_scenarios
        )
        
        assert len(stress_results) == len(stress_scenarios)
        
        for scenario, result in stress_results.items():
            assert 'portfolio_value' in result
            assert 'pnl' in result
            assert 'pnl_pct' in result
            assert result['pnl'] < 0  # Stress scenarios should be negative
    
    def test_concentration_risk(self, sample_portfolio):
        """Test concentration risk measurement."""
        risk_manager = RiskManager()
        
        concentration_metrics = risk_manager.calculate_concentration_risk(
            sample_portfolio['holdings']
        )
        
        assert 'herfindahl_index' in concentration_metrics
        assert 'effective_number_of_assets' in concentration_metrics
        assert 'top_5_concentration' in concentration_metrics
        
        # Check ranges
        assert 0 < concentration_metrics['herfindahl_index'] <= 1
        assert concentration_metrics['effective_number_of_assets'] > 0
        assert 0 < concentration_metrics['top_5_concentration'] <= 1
    
    def test_liquidity_risk(self, sample_portfolio):
        """Test liquidity risk assessment."""
        risk_manager = RiskManager()
        
        # Add liquidity information
        holdings_with_liquidity = sample_portfolio['holdings'].copy()
        holdings_with_liquidity['avg_daily_volume'] = [50000000, 2000000, 30000000, 10000000, 3000000]
        holdings_with_liquidity['bid_ask_spread'] = [0.01, 0.02, 0.01, 0.03, 0.02]
        
        liquidity_metrics = risk_manager.assess_liquidity_risk(
            holdings_with_liquidity
        )
        
        assert 'liquidity_score' in liquidity_metrics
        assert 'illiquid_positions' in liquidity_metrics
        assert 'liquidity_concentration' in liquidity_metrics
        
        # Check that illiquid positions are identified
        assert isinstance(liquidity_metrics['illiquid_positions'], list)
    
    def test_portfolio_beta(self, sample_portfolio):
        """Test portfolio beta calculation."""
        risk_manager = RiskManager()
        
        # Create benchmark returns
        np.random.seed(42)
        benchmark_returns = np.random.normal(0.0008, 0.015, 252)
        
        beta = risk_manager.calculate_portfolio_beta(
            sample_portfolio['returns'],
            benchmark_returns
        )
        
        assert isinstance(beta, float)
        assert 0 < beta < 3  # Reasonable range for equity portfolio
    
    def test_risk_attribution(self, sample_portfolio):
        """Test risk attribution analysis."""
        risk_manager = RiskManager()
        
        # Create multi-asset returns and weights
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        asset_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'GOOGL': np.random.normal(0.0008, 0.025, 252),
            'MSFT': np.random.normal(0.0012, 0.018, 252)
        }, index=dates)
        
        weights = np.array([0.4, 0.35, 0.25])
        
        attribution = risk_manager.calculate_risk_attribution(
            asset_returns,
            weights
        )
        
        assert 'total_risk' in attribution
        assert 'component_contributions' in attribution
        assert 'diversification_ratio' in attribution
        
        # Component contributions should sum to total risk
        total_contrib = sum(attribution['component_contributions'].values())
        assert abs(total_contrib - attribution['total_risk']) < 1e-6


class TestPositionSizer:
    """Test position sizing functionality."""
    
    @pytest.fixture
    def sample_account(self):
        """Create sample account for testing."""
        return {
            'total_equity': 100000,
            'available_cash': 20000,
            'margin_used': 5000,
            'max_leverage': 2.0
        }
    
    @pytest.fixture
    def sample_signal(self):
        """Create sample trading signal."""
        return {
            'symbol': 'AAPL',
            'signal_strength': 0.8,
            'entry_price': 150.0,
            'stop_loss': 140.0,
            'take_profit': 165.0,
            'expected_return': 0.05,
            'volatility': 0.25
        }
    
    def test_position_sizer_initialization(self):
        """Test position sizer initialization."""
        sizer = PositionSizer()
        
        assert sizer.max_position_size == 0.05
        assert sizer.max_risk_per_trade == 0.02
        assert sizer.volatility_lookback == 20
        assert sizer.sizing_method == 'fixed_fractional'
    
    def test_fixed_fractional_sizing(self, sample_account, sample_signal):
        """Test fixed fractional position sizing."""
        sizer = PositionSizer(sizing_method='fixed_fractional')
        
        position_size = sizer.calculate_position_size(
            sample_account,
            sample_signal
        )
        
        assert position_size['shares'] > 0
        assert position_size['notional_value'] <= sample_account['total_equity'] * sizer.max_position_size
        assert position_size['risk_amount'] <= sample_account['total_equity'] * sizer.max_risk_per_trade
    
    def test_volatility_adjusted_sizing(self, sample_account, sample_signal):
        """Test volatility-adjusted position sizing."""
        sizer = PositionSizer(sizing_method='volatility_adjusted')
        
        position_size = sizer.calculate_position_size(
            sample_account,
            sample_signal
        )
        
        assert position_size['shares'] > 0
        assert position_size['volatility_adjustment'] is not None
        
        # Higher volatility should result in smaller position
        high_vol_signal = sample_signal.copy()
        high_vol_signal['volatility'] = 0.5
        
        high_vol_position = sizer.calculate_position_size(
            sample_account,
            high_vol_signal
        )
        
        assert high_vol_position['shares'] < position_size['shares']
    
    def test_kelly_criterion_sizing(self, sample_account, sample_signal):
        """Test Kelly criterion position sizing."""
        sizer = PositionSizer(sizing_method='kelly_criterion')
        
        # Add probability and odds to signal
        signal_with_stats = sample_signal.copy()
        signal_with_stats['win_probability'] = 0.6
        signal_with_stats['avg_win'] = 0.08
        signal_with_stats['avg_loss'] = -0.05
        
        position_size = sizer.calculate_position_size(
            sample_account,
            signal_with_stats
        )
        
        assert position_size['shares'] > 0
        assert position_size['kelly_fraction'] is not None
        assert 0 <= position_size['kelly_fraction'] <= 1
    
    def test_risk_parity_sizing(self, sample_account):
        """Test risk parity position sizing."""
        sizer = PositionSizer(sizing_method='risk_parity')
        
        # Multiple signals for risk parity
        signals = [
            {
                'symbol': 'AAPL',
                'volatility': 0.25,
                'entry_price': 150.0,
                'stop_loss': 140.0
            },
            {
                'symbol': 'GOOGL',
                'volatility': 0.30,
                'entry_price': 2800.0,
                'stop_loss': 2600.0
            },
            {
                'symbol': 'MSFT',
                'volatility': 0.20,
                'entry_price': 300.0,
                'stop_loss': 285.0
            }
        ]
        
        position_sizes = sizer.calculate_risk_parity_sizes(
            sample_account,
            signals
        )
        
        assert len(position_sizes) == len(signals)
        
        # Risk contributions should be approximately equal
        risk_contributions = [
            pos['risk_amount'] for pos in position_sizes.values()
        ]
        
        assert np.std(risk_contributions) < np.mean(risk_contributions) * 0.2
    
    def test_optimal_f_sizing(self, sample_account, sample_signal):
        """Test Optimal F position sizing."""
        sizer = PositionSizer(sizing_method='optimal_f')
        
        # Add historical trade data
        signal_with_history = sample_signal.copy()
        signal_with_history['historical_returns'] = [
            0.05, -0.02, 0.08, -0.03, 0.12, -0.01, 0.06, -0.04
        ]
        
        position_size = sizer.calculate_position_size(
            sample_account,
            signal_with_history
        )
        
        assert position_size['shares'] > 0
        assert position_size['optimal_f'] is not None
        assert 0 <= position_size['optimal_f'] <= 1
    
    def test_correlation_adjusted_sizing(self, sample_account):
        """Test correlation-adjusted position sizing."""
        sizer = PositionSizer(sizing_method='correlation_adjusted')
        
        # Create correlated signals
        signals = [
            {
                'symbol': 'AAPL',
                'entry_price': 150.0,
                'stop_loss': 140.0,
                'correlation_to_portfolio': 0.8
            },
            {
                'symbol': 'GOOGL',
                'entry_price': 2800.0,
                'stop_loss': 2600.0,
                'correlation_to_portfolio': 0.3
            }
        ]
        
        position_sizes = sizer.calculate_correlation_adjusted_sizes(
            sample_account,
            signals
        )
        
        assert len(position_sizes) == len(signals)
        
        # Lower correlation should allow larger position
        assert position_sizes['GOOGL']['shares'] > position_sizes['AAPL']['shares']
    
    def test_drawdown_adjusted_sizing(self, sample_account, sample_signal):
        """Test drawdown-adjusted position sizing."""
        sizer = PositionSizer(sizing_method='drawdown_adjusted')
        
        # Test with different drawdown levels
        for current_drawdown in [0.0, 0.05, 0.15, 0.25]:
            account_with_dd = sample_account.copy()
            account_with_dd['current_drawdown'] = current_drawdown
            
            position_size = sizer.calculate_position_size(
                account_with_dd,
                sample_signal
            )
            
            assert position_size['shares'] > 0
            assert position_size['drawdown_adjustment'] is not None
            
            # Higher drawdown should result in smaller position
            if current_drawdown > 0.1:
                assert position_size['drawdown_adjustment'] < 1.0
    
    def test_monte_carlo_sizing(self, sample_account, sample_signal):
        """Test Monte Carlo position sizing."""
        sizer = PositionSizer(sizing_method='monte_carlo')
        
        # Add distribution parameters
        signal_with_dist = sample_signal.copy()
        signal_with_dist['return_distribution'] = 'normal'
        signal_with_dist['return_params'] = {'mean': 0.05, 'std': 0.15}
        
        position_size = sizer.calculate_position_size(
            sample_account,
            signal_with_dist,
            n_simulations=1000
        )
        
        assert position_size['shares'] > 0
        assert position_size['expected_return'] is not None
        assert position_size['var_95'] is not None
        assert position_size['probability_of_loss'] is not None
    
    def test_position_size_constraints(self, sample_account, sample_signal):
        """Test position size constraints."""
        sizer = PositionSizer(
            max_position_size=0.03,
            max_risk_per_trade=0.01
        )
        
        position_size = sizer.calculate_position_size(
            sample_account,
            sample_signal
        )
        
        # Check constraints are respected
        assert position_size['notional_value'] <= sample_account['total_equity'] * 0.03
        assert position_size['risk_amount'] <= sample_account['total_equity'] * 0.01
    
    def test_leverage_adjustment(self, sample_account, sample_signal):
        """Test leverage adjustment in position sizing."""
        sizer = PositionSizer()
        
        # High leverage account
        high_leverage_account = sample_account.copy()
        high_leverage_account['margin_used'] = 80000
        high_leverage_account['available_cash'] = 10000
        
        position_size = sizer.calculate_position_size(
            high_leverage_account,
            sample_signal
        )
        
        # Should reduce position size due to high leverage
        normal_position = sizer.calculate_position_size(
            sample_account,
            sample_signal
        )
        
        assert position_size['shares'] < normal_position['shares']
    
    def test_liquidity_adjusted_sizing(self, sample_account, sample_signal):
        """Test liquidity-adjusted position sizing."""
        sizer = PositionSizer(sizing_method='liquidity_adjusted')
        
        # Add liquidity information
        signal_with_liquidity = sample_signal.copy()
        signal_with_liquidity['avg_daily_volume'] = 50000000
        signal_with_liquidity['bid_ask_spread'] = 0.01
        
        position_size = sizer.calculate_position_size(
            sample_account,
            signal_with_liquidity
        )
        
        assert position_size['shares'] > 0
        assert position_size['liquidity_adjustment'] is not None
        
        # Low liquidity should reduce position size
        low_liquidity_signal = signal_with_liquidity.copy()
        low_liquidity_signal['avg_daily_volume'] = 1000000
        low_liquidity_signal['bid_ask_spread'] = 0.05
        
        low_liq_position = sizer.calculate_position_size(
            sample_account,
            low_liquidity_signal
        )
        
        assert low_liq_position['shares'] < position_size['shares']


class TestRebalancer:
    """Test portfolio rebalancing functionality."""
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio for rebalancing."""
        return {
            'current_weights': {
                'AAPL': 0.35,
                'GOOGL': 0.25,
                'MSFT': 0.20,
                'TSLA': 0.15,
                'AMZN': 0.05
            },
            'target_weights': {
                'AAPL': 0.30,
                'GOOGL': 0.25,
                'MSFT': 0.25,
                'TSLA': 0.10,
                'AMZN': 0.10
            },
            'current_values': {
                'AAPL': 35000,
                'GOOGL': 25000,
                'MSFT': 20000,
                'TSLA': 15000,
                'AMZN': 5000
            },
            'prices': {
                'AAPL': 150.0,
                'GOOGL': 2800.0,
                'MSFT': 300.0,
                'TSLA': 250.0,
                'AMZN': 3200.0
            }
        }
    
    def test_rebalancer_initialization(self):
        """Test rebalancer initialization."""
        rebalancer = PortfolioRebalancer(target_weights={"AAPL": 0.5, "GOOGL": 0.5})
        
        assert rebalancer.rebalance_threshold == 0.05
        assert rebalancer.rebalance_frequency == 'monthly'
        assert rebalancer.transaction_cost == 0.001
        assert rebalancer.min_trade_size == 100
    
    def test_rebalance_calculation(self, sample_portfolio):
        """Test rebalance calculation."""
        rebalancer = PortfolioRebalancer(target_weights={"AAPL": 0.5, "GOOGL": 0.5})
        
        rebalance_trades = rebalancer.calculate_rebalance_trades(
            sample_portfolio['current_weights'],
            sample_portfolio['target_weights'],
            sample_portfolio['current_values'],
            sample_portfolio['prices']
        )
        
        assert isinstance(rebalance_trades, dict)
        assert all(asset in rebalance_trades for asset in sample_portfolio['current_weights'].keys())
        
        # Check that trades move toward target weights
        for asset, trade in rebalance_trades.items():
            if trade['shares'] > 0:  # Buy
                assert sample_portfolio['current_weights'][asset] < sample_portfolio['target_weights'][asset]
            elif trade['shares'] < 0:  # Sell
                assert sample_portfolio['current_weights'][asset] > sample_portfolio['target_weights'][asset]
    
    def test_threshold_based_rebalancing(self, sample_portfolio):
        """Test threshold-based rebalancing."""
        rebalancer = PortfolioRebalancer(target_weights={"AAPL": 0.5, "GOOGL": 0.5}, threshold=0.1)
        
        # Small deviation - should not rebalance
        small_target = sample_portfolio['current_weights'].copy()
        small_target['AAPL'] = 0.37  # Only 2% deviation
        
        small_trades = rebalancer.calculate_rebalance_trades(
            sample_portfolio['current_weights'],
            small_target,
            sample_portfolio['current_values'],
            sample_portfolio['prices']
        )
        
        # Should have no trades due to threshold
        assert all(trade['shares'] == 0 for trade in small_trades.values())
        
        # Large deviation - should rebalance
        large_target = sample_portfolio['current_weights'].copy()
        large_target['AAPL'] = 0.20  # 15% deviation
        large_target['MSFT'] = 0.35  # 15% deviation
        
        large_trades = rebalancer.calculate_rebalance_trades(
            sample_portfolio['current_weights'],
            large_target,
            sample_portfolio['current_values'],
            sample_portfolio['prices']
        )
        
        # Should have trades
        assert any(trade['shares'] != 0 for trade in large_trades.values())
    
    def test_transaction_cost_optimization(self, sample_portfolio):
        """Test transaction cost optimization."""
        rebalancer = PortfolioRebalancer(target_weights={"AAPL": 0.5, "GOOGL": 0.5}, transaction_cost=0.01)  # High transaction cost
        
        trades = rebalancer.calculate_rebalance_trades(
            sample_portfolio['current_weights'],
            sample_portfolio['target_weights'],
            sample_portfolio['current_values'],
            sample_portfolio['prices']
        )
        
        # Calculate total transaction costs
        total_costs = sum(
            abs(trade['shares'] * sample_portfolio['prices'][asset]) * rebalancer.transaction_cost
            for asset, trade in trades.items()
        )
        
        # Should consider transaction costs in optimization
        assert total_costs > 0
        
        # Compare with zero transaction cost
        no_cost_rebalancer = PortfolioRebalancer(target_weights={"AAPL": 0.5, "GOOGL": 0.5}, transaction_cost=0.0)
        no_cost_trades = no_cost_rebalancer.calculate_rebalance_trades(
            sample_portfolio['current_weights'],
            sample_portfolio['target_weights'],
            sample_portfolio['current_values'],
            sample_portfolio['prices']
        )
        
        # High cost should result in fewer/smaller trades
        high_cost_volume = sum(abs(trade['shares']) for trade in trades.values())
        no_cost_volume = sum(abs(trade['shares']) for trade in no_cost_trades.values())
        
        assert high_cost_volume <= no_cost_volume
    
    def test_minimum_trade_size(self, sample_portfolio):
        """Test minimum trade size constraint."""
        rebalancer = PortfolioRebalancer(target_weights={"AAPL": 0.5, "GOOGL": 0.5}, min_trade_size=0.01)
        
        # Create scenario with very small required trades
        close_target = sample_portfolio['current_weights'].copy()
        close_target['AAPL'] = 0.345  # Very small change
        
        trades = rebalancer.calculate_rebalance_trades(
            sample_portfolio['current_weights'],
            close_target,
            sample_portfolio['current_values'],
            sample_portfolio['prices']
        )
        
        # Small trades should be eliminated
        for asset, trade in trades.items():
            if trade['shares'] != 0:
                trade_value = abs(trade['shares'] * sample_portfolio['prices'][asset])
                assert trade_value >= rebalancer.min_trade_size
    
    def test_calendar_rebalancing(self, sample_portfolio):
        """Test calendar-based rebalancing."""
        rebalancer = PortfolioRebalancer(target_weights={"AAPL": 0.5, "GOOGL": 0.5}, rebalance_frequency=RebalanceFrequency.QUARTERLY)
        
        # Test rebalancing schedule
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        rebalance_dates = rebalancer.get_rebalance_dates(start_date, end_date)
        
        assert len(rebalance_dates) == 4  # Quarterly
        
        # Check dates are approximately quarterly
        for i in range(1, len(rebalance_dates)):
            days_diff = (rebalance_dates[i] - rebalance_dates[i-1]).days
            assert 80 <= days_diff <= 100  # Approximately 3 months
    
    def test_volatility_based_rebalancing(self, sample_portfolio):
        """Test volatility-based rebalancing."""
        rebalancer = PortfolioRebalancer(target_weights={"AAPL": 0.5, "GOOGL": 0.5}, rebalance_method=RebalanceMethod.DYNAMIC)
        
        # Create volatility data
        np.random.seed(42)
        volatility_data = {
            'AAPL': np.random.uniform(0.15, 0.35, 252),
            'GOOGL': np.random.uniform(0.20, 0.40, 252),
            'MSFT': np.random.uniform(0.12, 0.28, 252),
            'TSLA': np.random.uniform(0.30, 0.60, 252),
            'AMZN': np.random.uniform(0.18, 0.38, 252)
        }
        
        should_rebalance = rebalancer.check_volatility_trigger(
            volatility_data,
            volatility_threshold=0.25
        )
        
        assert isinstance(should_rebalance, bool)
    
    def test_momentum_based_rebalancing(self, sample_portfolio):
        """Test momentum-based rebalancing."""
        rebalancer = PortfolioRebalancer(target_weights={"AAPL": 0.5, "GOOGL": 0.5}, rebalance_method=RebalanceMethod.TACTICAL)
        
        # Create momentum data
        np.random.seed(42)
        momentum_data = {
            'AAPL': np.random.uniform(-0.1, 0.15, 252),
            'GOOGL': np.random.uniform(-0.08, 0.12, 252),
            'MSFT': np.random.uniform(-0.06, 0.10, 252),
            'TSLA': np.random.uniform(-0.2, 0.25, 252),
            'AMZN': np.random.uniform(-0.12, 0.18, 252)
        }
        
        should_rebalance = rebalancer.check_momentum_trigger(
            momentum_data,
            momentum_threshold=0.1
        )
        
        assert isinstance(should_rebalance, bool)
    
    def test_rebalance_execution(self, sample_portfolio):
        """Test rebalance execution."""
        rebalancer = PortfolioRebalancer(target_weights={"AAPL": 0.5, "GOOGL": 0.5})
        
        trades = rebalancer.calculate_rebalance_trades(
            sample_portfolio['current_weights'],
            sample_portfolio['target_weights'],
            sample_portfolio['current_values'],
            sample_portfolio['prices']
        )
        
        # Execute rebalance
        execution_result = rebalancer.execute_rebalance(
            trades,
            sample_portfolio['prices']
        )
        
        assert 'executed_trades' in execution_result
        assert 'total_cost' in execution_result
        assert 'new_weights' in execution_result
        assert 'tracking_error' in execution_result
        
        # Check that new weights are closer to target
        for asset in sample_portfolio['current_weights'].keys():
            current_error = abs(
                sample_portfolio['current_weights'][asset] - 
                sample_portfolio['target_weights'][asset]
            )
            new_error = abs(
                execution_result['new_weights'][asset] - 
                sample_portfolio['target_weights'][asset]
            )
            assert new_error <= current_error
    
    def test_tax_efficient_rebalancing(self, sample_portfolio):
        """Test tax-efficient rebalancing."""
        rebalancer = Rebalancer(tax_efficient=True)
        
        # Add tax information
        tax_info = {
            'AAPL': {'unrealized_gains': 5000, 'holding_period': 400},
            'GOOGL': {'unrealized_gains': -2000, 'holding_period': 200},
            'MSFT': {'unrealized_gains': 3000, 'holding_period': 300},
            'TSLA': {'unrealized_gains': -1000, 'holding_period': 100},
            'AMZN': {'unrealized_gains': 500, 'holding_period': 500}
        }
        
        trades = rebalancer.calculate_tax_efficient_trades(
            sample_portfolio['current_weights'],
            sample_portfolio['target_weights'],
            sample_portfolio['current_values'],
            sample_portfolio['prices'],
            tax_info
        )
        
        # Should prefer selling losses and holding gains
        for asset, trade in trades.items():
            if trade['shares'] < 0:  # Selling
                # Should prefer selling positions with losses
                if tax_info[asset]['unrealized_gains'] > 0:
                    # If selling a gain, should be long-term (>365 days)
                    assert tax_info[asset]['holding_period'] > 365


class TestStressTesting:
    """Test stress testing functionality."""
    
    @pytest.fixture
    def sample_portfolio_for_stress(self):
        """Create sample portfolio for stress testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Portfolio returns
        returns = np.random.normal(0.001, 0.02, 252)
        portfolio_value = pd.Series(100000 * (1 + returns).cumprod(), index=dates)
        
        # Asset returns
        asset_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.025, 252),
            'GOOGL': np.random.normal(0.0008, 0.028, 252),
            'MSFT': np.random.normal(0.0012, 0.020, 252),
            'TSLA': np.random.normal(0.0015, 0.045, 252),
            'AMZN': np.random.normal(0.0005, 0.030, 252)
        }, index=dates)
        
        # Holdings
        holdings = pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
            'quantity': [100, 20, 80, 30, 15],
            'price': [150, 2800, 300, 250, 3200],
            'weight': [0.3, 0.25, 0.2, 0.15, 0.1]
        })
        
        return {
            'portfolio_value': portfolio_value,
            'asset_returns': asset_returns,
            'holdings': holdings,
            'returns': pd.Series(returns, index=dates)
        }
    
    def test_stress_testing_initialization(self):
        """Test stress testing initialization."""
        stress_tester = StressTester(portfolio_data=pd.DataFrame(columns=['returns']))
        
        assert stress_tester.confidence_levels == [0.95, 0.99]
        assert stress_tester.time_horizons == [1, 5, 10, 22]
        assert stress_tester.n_simulations == 10000
    
    def test_historical_stress_scenarios(self, sample_portfolio_for_stress):
        """Test historical stress scenarios."""
        stress_tester = StressTester(portfolio_data=pd.DataFrame(columns=['returns']))
        
        # Define historical scenarios
        scenarios = {
            'dot_com_crash': {
                'start_date': '2000-03-01',
                'end_date': '2002-10-01',
                'description': 'Dot-com bubble crash'
            },
            'financial_crisis': {
                'start_date': '2007-10-01',
                'end_date': '2009-03-01',
                'description': '2008 Financial Crisis'
            },
            'covid_crash': {
                'start_date': '2020-02-01',
                'end_date': '2020-04-01',
                'description': 'COVID-19 Market Crash'
            }
        }
        
        stress_results = stress_tester.run_historical_scenarios(
            sample_portfolio_for_stress['asset_returns'],
            sample_portfolio_for_stress['holdings'],
            scenarios
        )
        
        assert len(stress_results) == len(scenarios)
        
        for scenario, result in stress_results.items():
            assert 'portfolio_return' in result
            assert 'portfolio_volatility' in result
            assert 'max_drawdown' in result
            assert 'var_95' in result
            assert 'cvar_95' in result
            
            # Stress scenarios should show negative returns
            assert result['portfolio_return'] < 0
    
    def test_parametric_stress_testing(self, sample_portfolio_for_stress):
        """Test parametric stress testing."""
        stress_tester = StressTester(portfolio_data=pd.DataFrame(columns=['returns']))
        
        # Define parametric scenarios
        scenarios = {
            'market_crash': {
                'type': 'parametric',
                'market_shock': -0.30,
                'volatility_multiplier': 2.0,
                'correlation_increase': 0.2
            },
            'interest_rate_shock': {
                'type': 'parametric',
                'interest_rate_change': 0.03,
                'duration': 5.0
            },
            'sector_rotation': {
                'type': 'parametric',
                'sector_shocks': {
                    'technology': -0.25,
                    'healthcare': 0.10,
                    'financials': -0.15
                }
            }
        }
        
        stress_results = stress_tester.run_parametric_scenarios(
            sample_portfolio_for_stress['asset_returns'],
            sample_portfolio_for_stress['holdings'],
            scenarios
        )
        
        assert len(stress_results) == len(scenarios)
        
        for scenario, result in stress_results.items():
            assert 'portfolio_pnl' in result
            assert 'portfolio_pnl_pct' in result
            assert 'component_contributions' in result
    
    def test_monte_carlo_stress_testing(self, sample_portfolio_for_stress):
        """Test Monte Carlo stress testing."""
        stress_tester = StressTester(portfolio_data=pd.DataFrame(columns=['returns']))
        
        monte_carlo_results = stress_tester.run_monte_carlo_stress(
            sample_portfolio_for_stress['asset_returns'],
            sample_portfolio_for_stress['holdings'],
            n_simulations=1000,
            time_horizon=22  # 1 month
        )
        
        assert 'simulation_results' in monte_carlo_results
        assert 'percentiles' in monte_carlo_results
        assert 'tail_expectations' in monte_carlo_results
        assert 'worst_case_scenarios' in monte_carlo_results
        
        # Check simulation results
        sim_results = monte_carlo_results['simulation_results']
        assert len(sim_results) == 1000
        
        # Check percentiles
        percentiles = monte_carlo_results['percentiles']
        assert percentiles['p1'] < percentiles['p5'] < percentiles['p95'] < percentiles['p99']
    
    def test_tail_risk_analysis(self, sample_portfolio_for_stress):
        """Test tail risk analysis."""
        stress_tester = StressTester(portfolio_data=pd.DataFrame(columns=['returns']))
        
        tail_risk = stress_tester.analyze_tail_risk(
            sample_portfolio_for_stress['returns'],
            confidence_levels=[0.95, 0.99, 0.999]
        )
        
        assert 'var' in tail_risk
        assert 'cvar' in tail_risk
        assert 'extreme_value_theory' in tail_risk
        assert 'tail_dependence' in tail_risk
        
        # Check that higher confidence levels give worse tail risk
        assert tail_risk['var'][0.95] > tail_risk['var'][0.99]
        assert tail_risk['cvar'][0.95] > tail_risk['cvar'][0.99]
    
    def test_correlation_breakdown_stress(self, sample_portfolio_for_stress):
        """Test correlation breakdown stress testing."""
        stress_tester = StressTester(portfolio_data=pd.DataFrame(columns=['returns']))
        
        breakdown_results = stress_tester.test_correlation_breakdown(
            sample_portfolio_for_stress['asset_returns'],
            sample_portfolio_for_stress['holdings']
        )
        
        assert 'normal_correlations' in breakdown_results
        assert 'stressed_correlations' in breakdown_results
        assert 'portfolio_impact' in breakdown_results
        assert 'diversification_ratio' in breakdown_results
        
        # Stressed correlations should be higher
        normal_avg = np.mean(breakdown_results['normal_correlations'].values)
        stressed_avg = np.mean(breakdown_results['stressed_correlations'].values)
        assert stressed_avg > normal_avg
    
    def test_liquidity_stress_testing(self, sample_portfolio_for_stress):
        """Test liquidity stress testing."""
        stress_tester = StressTester(portfolio_data=pd.DataFrame(columns=['returns']))
        
        # Add liquidity information
        liquidity_info = {
            'AAPL': {'daily_volume': 50000000, 'bid_ask_spread': 0.01},
            'GOOGL': {'daily_volume': 2000000, 'bid_ask_spread': 0.02},
            'MSFT': {'daily_volume': 30000000, 'bid_ask_spread': 0.01},
            'TSLA': {'daily_volume': 25000000, 'bid_ask_spread': 0.03},
            'AMZN': {'daily_volume': 3000000, 'bid_ask_spread': 0.02}
        }
        
        liquidity_stress = stress_tester.test_liquidity_stress(
            sample_portfolio_for_stress['holdings'],
            liquidity_info,
            liquidation_timeframe=5  # 5 days
        )
        
        assert 'liquidation_costs' in liquidity_stress
        assert 'market_impact' in liquidity_stress
        assert 'liquidation_schedule' in liquidity_stress
        assert 'liquidity_score' in liquidity_stress
        
        # Check that illiquid assets have higher costs
        total_cost = liquidity_stress['liquidation_costs']['total_cost']
        assert total_cost > 0
    
    def test_regime_change_stress(self, sample_portfolio_for_stress):
        """Test regime change stress testing."""
        stress_tester = StressTester(portfolio_data=pd.DataFrame(columns=['returns']))
        
        regime_scenarios = {
            'low_vol_to_high_vol': {
                'from_regime': {'volatility': 0.15, 'correlation': 0.3},
                'to_regime': {'volatility': 0.35, 'correlation': 0.7}
            },
            'growth_to_value': {
                'from_regime': {'growth_premium': 0.05, 'value_discount': -0.02},
                'to_regime': {'growth_premium': -0.03, 'value_discount': 0.04}
            }
        }
        
        regime_stress = stress_tester.test_regime_changes(
            sample_portfolio_for_stress['asset_returns'],
            sample_portfolio_for_stress['holdings'],
            regime_scenarios
        )
        
        assert len(regime_stress) == len(regime_scenarios)
        
        for scenario, result in regime_stress.items():
            assert 'portfolio_impact' in result
            assert 'transition_probability' in result
            assert 'expected_duration' in result
    
    def test_multi_factor_stress_testing(self, sample_portfolio_for_stress):
        """Test multi-factor stress testing."""
        stress_tester = StressTester(portfolio_data=pd.DataFrame(columns=['returns']))
        
        factor_scenarios = {
            'recession_scenario': {
                'gdp_growth': -0.03,
                'unemployment_rate': 0.02,
                'inflation_rate': -0.01,
                'interest_rates': -0.02
            },
            'stagflation_scenario': {
                'gdp_growth': -0.01,
                'unemployment_rate': 0.01,
                'inflation_rate': 0.04,
                'interest_rates': 0.03
            }
        }
        
        factor_stress = stress_tester.test_multi_factor_scenarios(
            sample_portfolio_for_stress['asset_returns'],
            sample_portfolio_for_stress['holdings'],
            factor_scenarios
        )
        
        assert len(factor_stress) == len(factor_scenarios)
        
        for scenario, result in factor_stress.items():
            assert 'portfolio_return' in result
            assert 'factor_contributions' in result
            assert 'sensitivity_analysis' in result
    
    def test_stress_testing_reporting(self, sample_portfolio_for_stress):
        """Test stress testing reporting."""
        stress_tester = StressTester(portfolio_data=pd.DataFrame(columns=['returns']))
        
        # Run various stress tests
        results = {}
        
        # Historical scenario
        results['historical'] = stress_tester.run_historical_scenarios(
            sample_portfolio_for_stress['asset_returns'],
            sample_portfolio_for_stress['holdings'],
            {'test_scenario': {
                'start_date': '2023-01-01',
                'end_date': '2023-03-01',
                'description': 'Test scenario'
            }}
        )
        
        # Monte Carlo
        results['monte_carlo'] = stress_tester.run_monte_carlo_stress(
            sample_portfolio_for_stress['asset_returns'],
            sample_portfolio_for_stress['holdings'],
            n_simulations=100
        )
        
        # Generate comprehensive report
        report = stress_tester.generate_stress_report(results)
        
        assert 'executive_summary' in report
        assert 'detailed_results' in report
        assert 'risk_metrics' in report
        assert 'recommendations' in report
        
        # Check executive summary
        exec_summary = report['executive_summary']
        assert 'worst_case_loss' in exec_summary
        assert 'probability_of_loss' in exec_summary
        assert 'risk_rating' in exec_summary


class TestEnhancedRiskManager:
    """Test enhanced risk manager functionality."""
    
    @pytest.fixture
    def comprehensive_portfolio(self):
        """Create comprehensive portfolio for enhanced risk testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Multi-asset portfolio
        assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'SPY', 'QQQ', 'IWM']
        
        returns = pd.DataFrame({
            asset: np.random.normal(0.001, 0.02 + np.random.uniform(-0.005, 0.015), 252)
            for asset in assets
        }, index=dates)
        
        # Make some correlations realistic
        returns['GOOGL'] = returns['AAPL'] * 0.6 + returns['GOOGL'] * 0.4
        returns['MSFT'] = returns['AAPL'] * 0.5 + returns['MSFT'] * 0.5
        
        portfolio_value = 100000 * (1 + returns.mean(axis=1)).cumprod()
        
        return {
            'returns': returns,
            'portfolio_value': portfolio_value,
            'weights': np.array([0.15, 0.15, 0.15, 0.1, 0.1, 0.15, 0.1, 0.1])
        }
    
    def test_enhanced_risk_manager_initialization(self):
        """Test enhanced risk manager initialization."""
        risk_manager = EnhancedRiskManager()
        
        assert risk_manager.confidence_levels == [0.95, 0.99]
        assert risk_manager.time_horizons == [1, 5, 10, 22]
        assert risk_manager.enable_ml_models == True
        assert risk_manager.dynamic_hedging == True
    
    def test_dynamic_var_calculation(self, comprehensive_portfolio):
        """Test dynamic VaR calculation."""
        risk_manager = EnhancedRiskManager()
        
        dynamic_var = risk_manager.calculate_dynamic_var(
            comprehensive_portfolio['returns'],
            comprehensive_portfolio['weights']
        )
        
        assert 'garch_var' in dynamic_var
        assert 'ewma_var' in dynamic_var
        assert 'filtered_var' in dynamic_var
        assert 'regime_adjusted_var' in dynamic_var
        
        # All VaR measures should be negative
        for var_type, var_value in dynamic_var.items():
            assert var_value < 0
    
    def test_copula_based_risk_modeling(self, comprehensive_portfolio):
        """Test copula-based risk modeling."""
        risk_manager = EnhancedRiskManager()
        
        copula_results = risk_manager.model_copula_dependencies(
            comprehensive_portfolio['returns']
        )
        
        assert 'copula_type' in copula_results
        assert 'tail_dependence' in copula_results
        assert 'joint_var' in copula_results
        assert 'marginal_contributions' in copula_results
        
        # Check tail dependence structure
        tail_dep = copula_results['tail_dependence']
        assert isinstance(tail_dep, dict)
        assert all(0 <= dep <= 1 for dep in tail_dep.values())
    
    def test_machine_learning_risk_models(self, comprehensive_portfolio):
        """Test ML-based risk models."""
        risk_manager = EnhancedRiskManager(enable_ml_models=True)
        
        ml_risk_models = risk_manager.train_ml_risk_models(
            comprehensive_portfolio['returns'],
            comprehensive_portfolio['weights']
        )
        
        assert 'lstm_volatility' in ml_risk_models
        assert 'xgboost_var' in ml_risk_models
        assert 'ensemble_risk' in ml_risk_models
        assert 'feature_importance' in ml_risk_models
        
        # Check model performance
        for model_name, model in ml_risk_models.items():
            if model_name != 'feature_importance':
                assert 'predictions' in model
                assert 'accuracy' in model
                assert model['accuracy'] > 0
    
    def test_regime_aware_risk_management(self, comprehensive_portfolio):
        """Test regime-aware risk management."""
        risk_manager = EnhancedRiskManager()
        
        regime_risk = risk_manager.analyze_regime_risk(
            comprehensive_portfolio['returns'],
            comprehensive_portfolio['weights']
        )
        
        assert 'regime_probabilities' in regime_risk
        assert 'regime_var' in regime_risk
        assert 'transition_risks' in regime_risk
        assert 'regime_hedging_strategies' in regime_risk
        
        # Check regime probabilities sum to 1
        regime_probs = regime_risk['regime_probabilities']
        assert abs(sum(regime_probs.values()) - 1.0) < 1e-6
    
    def test_dynamic_hedging_strategies(self, comprehensive_portfolio):
        """Test dynamic hedging strategies."""
        risk_manager = EnhancedRiskManager(dynamic_hedging=True)
        
        hedging_strategies = risk_manager.generate_hedging_strategies(
            comprehensive_portfolio['returns'],
            comprehensive_portfolio['weights']
        )
        
        assert 'options_hedge' in hedging_strategies
        assert 'futures_hedge' in hedging_strategies
        assert 'pairs_hedge' in hedging_strategies
        assert 'dynamic_overlay' in hedging_strategies
        
        # Check hedging effectiveness
        for strategy_name, strategy in hedging_strategies.items():
            assert 'hedge_ratio' in strategy
            assert 'cost' in strategy
            assert 'effectiveness' in strategy
            assert 0 <= strategy['effectiveness'] <= 1
    
    def test_real_time_risk_monitoring(self, comprehensive_portfolio):
        """Test real-time risk monitoring."""
        risk_manager = EnhancedRiskManager()
        
        # Simulate real-time data
        current_prices = pd.Series({
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 300.0,
            'TSLA': 250.0,
            'AMZN': 3200.0,
            'SPY': 450.0,
            'QQQ': 380.0,
            'IWM': 200.0
        })
        
        risk_alerts = risk_manager.monitor_real_time_risk(
            comprehensive_portfolio['returns'],
            comprehensive_portfolio['weights'],
            current_prices
        )
        
        assert 'risk_level' in risk_alerts
        assert 'alerts' in risk_alerts
        assert 'suggested_actions' in risk_alerts
        assert 'portfolio_health' in risk_alerts
        
        # Check risk level is valid
        assert risk_alerts['risk_level'] in ['low', 'medium', 'high', 'critical']
    
    def test_portfolio_optimization_with_risk_constraints(self, comprehensive_portfolio):
        """Test portfolio optimization with risk constraints."""
        risk_manager = EnhancedRiskManager()
        
        risk_constraints = {
            'max_var': 0.02,
            'max_cvar': 0.03,
            'max_correlation': 0.8,
            'min_diversification_ratio': 0.6
        }
        
        optimized_weights = risk_manager.optimize_with_risk_constraints(
            comprehensive_portfolio['returns'],
            risk_constraints
        )
        
        assert len(optimized_weights) == len(comprehensive_portfolio['weights'])
        assert abs(optimized_weights.sum() - 1.0) < 1e-6
        assert all(w >= 0 for w in optimized_weights)
        
        # Verify risk constraints are met
        portfolio_var = risk_manager.calculate_portfolio_var(
            comprehensive_portfolio['returns'],
            optimized_weights
        )
        assert portfolio_var >= -risk_constraints['max_var']
    
    def test_risk_budgeting(self, comprehensive_portfolio):
        """Test risk budgeting and attribution."""
        risk_manager = EnhancedRiskManager()
        
        risk_budget = risk_manager.calculate_risk_budget(
            comprehensive_portfolio['returns'],
            comprehensive_portfolio['weights']
        )
        
        assert 'component_var' in risk_budget
        assert 'marginal_var' in risk_budget
        assert 'risk_contributions' in risk_budget
        assert 'diversification_ratio' in risk_budget
        
        # Risk contributions should sum to portfolio VaR
        total_contribution = sum(risk_budget['risk_contributions'].values())
        portfolio_var = risk_manager.calculate_portfolio_var(
            comprehensive_portfolio['returns'],
            comprehensive_portfolio['weights']
        )
        assert abs(total_contribution - abs(portfolio_var)) < 1e-6
    
    def test_extreme_risk_scenarios(self, comprehensive_portfolio):
        """Test extreme risk scenario analysis."""
        risk_manager = EnhancedRiskManager()
        
        extreme_scenarios = risk_manager.analyze_extreme_scenarios(
            comprehensive_portfolio['returns'],
            comprehensive_portfolio['weights']
        )
        
        assert 'black_swan_events' in extreme_scenarios
        assert 'fat_tail_analysis' in extreme_scenarios
        assert 'jump_risk' in extreme_scenarios
        assert 'contagion_risk' in extreme_scenarios
        
        # Check black swan analysis
        black_swan = extreme_scenarios['black_swan_events']
        assert 'probability' in black_swan
        assert 'expected_loss' in black_swan
        assert 'recovery_time' in black_swan


if __name__ == '__main__':
    pytest.main([__file__, '-v'])