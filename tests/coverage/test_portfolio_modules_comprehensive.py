"""Comprehensive tests for portfolio management modules to achieve 100% coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.portfolio.portfolio_optimizer import (
    PortfolioOptimizer, OptimizationObjective, OptimizationConstraints
)
from src.portfolio.position_sizer import (
    PositionSizer, SizingMethod, RiskParity, KellyCriterion
)
from src.portfolio.rebalancer import (
    PortfolioRebalancer, RebalancingStrategy, RebalanceSignal
)
from src.portfolio.risk_manager import (
    RiskManager, RiskMetrics, RiskLimits, RiskAlert
)
from src.portfolio.stress_testing import (
    StressTester, StressScenario, StressTestResult
)
from src.portfolio.allocation import (
    AssetAllocator, AllocationStrategy, AllocationConstraints
)
from src.portfolio.drawdown_analyzer import (
    DrawdownAnalyzer, DrawdownMetrics, RecoveryAnalysis
)
from src.portfolio.performance_attribution import (
    PerformanceAttributor, AttributionResult, FactorAnalysis
)


class TestPortfolioOptimizer:
    """Test PortfolioOptimizer class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return data."""
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate correlated returns for 5 assets
        n_assets = 5
        mean_returns = np.array([0.0005, 0.0008, 0.0003, 0.0006, 0.0004])
        
        # Create correlation matrix
        correlation = np.array([
            [1.0, 0.3, 0.1, 0.2, 0.0],
            [0.3, 1.0, 0.4, 0.1, 0.2],
            [0.1, 0.4, 1.0, 0.3, 0.1],
            [0.2, 0.1, 0.3, 1.0, 0.4],
            [0.0, 0.2, 0.1, 0.4, 1.0]
        ])
        
        # Generate returns
        cov_matrix = correlation * 0.01  # Scale to realistic volatility
        returns = np.random.multivariate_normal(mean_returns, cov_matrix, len(dates))
        
        return pd.DataFrame(
            returns,
            index=dates,
            columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5']
        )
        
    def test_optimizer_initialization(self):
        """Test PortfolioOptimizer initialization."""
        optimizer = PortfolioOptimizer(
            risk_free_rate=0.02,
            rebalance_frequency='monthly'
        )
        
        assert optimizer.risk_free_rate == 0.02
        assert optimizer.rebalance_frequency == 'monthly'
        assert optimizer.optimization_history == []
        
    def test_mean_variance_optimization(self, sample_returns):
        """Test mean-variance optimization."""
        optimizer = PortfolioOptimizer()
        
        constraints = OptimizationConstraints(
            min_weight=0.0,
            max_weight=0.4,
            target_return=None,
            max_volatility=0.15
        )
        
        result = optimizer.optimize(
            returns=sample_returns,
            objective=OptimizationObjective.MAX_SHARPE,
            constraints=constraints
        )
        
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'volatility' in result
        assert 'sharpe_ratio' in result
        
        # Check constraints
        weights = result['weights']
        assert all(w >= 0 for w in weights.values())
        assert all(w <= 0.4 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        
    def test_min_variance_portfolio(self, sample_returns):
        """Test minimum variance portfolio optimization."""
        optimizer = PortfolioOptimizer()
        
        result = optimizer.optimize(
            returns=sample_returns,
            objective=OptimizationObjective.MIN_VARIANCE
        )
        
        # Should have lower volatility than equal weight
        equal_weight_vol = optimizer._calculate_portfolio_volatility(
            np.ones(5) / 5,
            sample_returns.cov()
        )
        
        assert result['volatility'] <= equal_weight_vol
        
    def test_risk_parity_optimization(self, sample_returns):
        """Test risk parity optimization."""
        optimizer = PortfolioOptimizer()
        
        result = optimizer.optimize(
            returns=sample_returns,
            objective=OptimizationObjective.RISK_PARITY
        )
        
        # Check risk contributions are roughly equal
        weights = np.array(list(result['weights'].values()))
        cov_matrix = sample_returns.cov().values
        
        risk_contributions = optimizer._calculate_risk_contributions(
            weights, cov_matrix
        )
        
        # Risk contributions should be approximately equal
        assert np.std(risk_contributions) < 0.01
        
    def test_black_litterman_optimization(self, sample_returns):
        """Test Black-Litterman optimization."""
        optimizer = PortfolioOptimizer()
        
        # Define views
        views = {
            'views': np.array([[1, -1, 0, 0, 0]]),  # Asset1 outperforms Asset2
            'view_returns': np.array([0.02]),
            'view_confidence': np.array([0.8])
        }
        
        result = optimizer.optimize_black_litterman(
            returns=sample_returns,
            market_cap_weights={'Asset1': 0.3, 'Asset2': 0.3, 'Asset3': 0.2, 
                              'Asset4': 0.1, 'Asset5': 0.1},
            views=views
        )
        
        assert 'weights' in result
        assert 'posterior_returns' in result
        
        # Asset1 should have higher weight than Asset2 given the view
        assert result['weights']['Asset1'] > result['weights']['Asset2']
        
    def test_optimization_with_constraints(self, sample_returns):
        """Test optimization with various constraints."""
        optimizer = PortfolioOptimizer()
        
        # Sector constraints
        sector_constraints = {
            'Tech': {'assets': ['Asset1', 'Asset2'], 'max_weight': 0.4},
            'Finance': {'assets': ['Asset3', 'Asset4'], 'max_weight': 0.3}
        }
        
        constraints = OptimizationConstraints(
            min_weight=0.05,
            max_weight=0.3,
            sector_constraints=sector_constraints,
            target_volatility=0.12
        )
        
        result = optimizer.optimize(
            returns=sample_returns,
            objective=OptimizationObjective.MAX_RETURN,
            constraints=constraints
        )
        
        weights = result['weights']
        
        # Check individual constraints
        assert all(w >= 0.05 for w in weights.values())
        assert all(w <= 0.3 for w in weights.values())
        
        # Check sector constraints
        tech_weight = weights['Asset1'] + weights['Asset2']
        assert tech_weight <= 0.4
        
        finance_weight = weights['Asset3'] + weights['Asset4']
        assert finance_weight <= 0.3


class TestPositionSizer:
    """Test PositionSizer class."""
    
    @pytest.fixture
    def position_sizer(self):
        """Create PositionSizer instance."""
        return PositionSizer(
            base_position_size=0.02,
            max_position_size=0.10,
            use_volatility_sizing=True
        )
        
    def test_fixed_sizing(self, position_sizer):
        """Test fixed position sizing."""
        size = position_sizer.calculate_position_size(
            method=SizingMethod.FIXED,
            portfolio_value=100000,
            price=50
        )
        
        # 2% of 100k = 2000, at $50 = 40 shares
        assert size == 40
        
    def test_volatility_based_sizing(self, position_sizer):
        """Test volatility-based sizing."""
        size = position_sizer.calculate_position_size(
            method=SizingMethod.VOLATILITY,
            portfolio_value=100000,
            price=50,
            volatility=0.02,
            target_risk=0.01
        )
        
        # Lower volatility should result in larger position
        high_vol_size = position_sizer.calculate_position_size(
            method=SizingMethod.VOLATILITY,
            portfolio_value=100000,
            price=50,
            volatility=0.05,
            target_risk=0.01
        )
        
        assert size > high_vol_size
        
    def test_kelly_criterion_sizing(self, position_sizer):
        """Test Kelly Criterion sizing."""
        kelly = KellyCriterion(kelly_fraction=0.25)  # 25% Kelly
        
        size = kelly.calculate_size(
            win_probability=0.6,
            win_size=0.02,
            loss_size=0.01,
            portfolio_value=100000,
            price=50
        )
        
        # Kelly formula: f = (p*b - q) / b
        # where p=0.6, q=0.4, b=2 (win/loss ratio)
        expected_kelly = (0.6 * 2 - 0.4) / 2  # = 0.4
        expected_size = int(100000 * expected_kelly * 0.25 / 50)  # 25% Kelly
        
        assert abs(size - expected_size) < 5
        
    def test_risk_parity_sizing(self, position_sizer):
        """Test risk parity position sizing."""
        risk_parity = RiskParity()
        
        current_positions = {
            'AAPL': {'value': 20000, 'volatility': 0.02},
            'GOOGL': {'value': 30000, 'volatility': 0.025},
            'MSFT': {'value': 25000, 'volatility': 0.018}
        }
        
        size = risk_parity.calculate_size(
            symbol='TSLA',
            volatility=0.03,
            portfolio_value=100000,
            current_positions=current_positions,
            price=200
        )
        
        # Should size to balance risk contribution
        assert size > 0
        assert size < 100  # Reasonable size for high volatility asset
        
    def test_atr_based_sizing(self, position_sizer):
        """Test ATR-based position sizing."""
        size = position_sizer.calculate_position_size(
            method=SizingMethod.ATR,
            portfolio_value=100000,
            price=50,
            atr=2.5,
            risk_per_trade=0.01
        )
        
        # Risk 1% of portfolio, with 2.5 ATR stop
        # $1000 risk / $2.5 ATR = 400 shares max
        # But also check against max position size
        expected = min(400, int(100000 * 0.10 / 50))  # Max 10% position
        
        assert size == expected
        
    def test_position_size_limits(self, position_sizer):
        """Test position size limits."""
        # Test max position size limit
        size = position_sizer.calculate_position_size(
            method=SizingMethod.FIXED,
            portfolio_value=100000,
            price=10,  # Low price would give large share count
            position_size_pct=0.20  # Request 20% position
        )
        
        # Should be capped at 10% max
        max_shares = int(100000 * 0.10 / 10)
        assert size == max_shares


class TestPortfolioRebalancer:
    """Test PortfolioRebalancer class."""
    
    @pytest.fixture
    def rebalancer(self):
        """Create PortfolioRebalancer instance."""
        return PortfolioRebalancer(
            rebalance_threshold=0.05,
            min_rebalance_interval=30,
            transaction_cost=0.001
        )
        
    @pytest.fixture
    def portfolio_data(self):
        """Create sample portfolio data."""
        return {
            'positions': {
                'AAPL': {'shares': 100, 'price': 150},
                'GOOGL': {'shares': 20, 'price': 2500},
                'MSFT': {'shares': 80, 'price': 300},
                'AMZN': {'shares': 30, 'price': 3000}
            },
            'target_weights': {
                'AAPL': 0.20,
                'GOOGL': 0.30,
                'MSFT': 0.25,
                'AMZN': 0.25
            },
            'last_rebalance': datetime.now() - timedelta(days=45)
        }
        
    def test_threshold_rebalancing(self, rebalancer, portfolio_data):
        """Test threshold-based rebalancing."""
        signal = rebalancer.check_rebalance_signal(
            strategy=RebalancingStrategy.THRESHOLD,
            current_positions=portfolio_data['positions'],
            target_weights=portfolio_data['target_weights'],
            last_rebalance_date=portfolio_data['last_rebalance']
        )
        
        assert isinstance(signal, RebalanceSignal)
        assert signal.should_rebalance in [True, False]
        
        if signal.should_rebalance:
            assert signal.reason is not None
            assert signal.positions_to_adjust is not None
            
    def test_calendar_rebalancing(self, rebalancer, portfolio_data):
        """Test calendar-based rebalancing."""
        # Force rebalance by setting old date
        old_date = datetime.now() - timedelta(days=100)
        
        signal = rebalancer.check_rebalance_signal(
            strategy=RebalancingStrategy.CALENDAR,
            current_positions=portfolio_data['positions'],
            target_weights=portfolio_data['target_weights'],
            last_rebalance_date=old_date,
            rebalance_frequency='quarterly'
        )
        
        assert signal.should_rebalance is True
        assert 'calendar' in signal.reason.lower()
        
    def test_calculate_rebalance_trades(self, rebalancer, portfolio_data):
        """Test rebalance trade calculation."""
        trades = rebalancer.calculate_rebalance_trades(
            current_positions=portfolio_data['positions'],
            target_weights=portfolio_data['target_weights'],
            portfolio_value=179000  # Sum of position values
        )
        
        assert isinstance(trades, list)
        
        for trade in trades:
            assert 'symbol' in trade
            assert 'action' in trade
            assert 'quantity' in trade
            assert trade['action'] in ['buy', 'sell']
            
        # After trades, should be close to target weights
        new_positions = rebalancer._apply_trades(
            portfolio_data['positions'].copy(),
            trades
        )
        
        new_weights = rebalancer._calculate_weights(new_positions)
        
        for symbol, target in portfolio_data['target_weights'].items():
            assert abs(new_weights.get(symbol, 0) - target) < 0.02
            
    def test_tax_aware_rebalancing(self, rebalancer):
        """Test tax-aware rebalancing."""
        positions_with_tax = {
            'AAPL': {
                'shares': 100,
                'price': 150,
                'cost_basis': 100,
                'holding_period': 400  # Long-term
            },
            'GOOGL': {
                'shares': 20,
                'price': 2500,
                'cost_basis': 2600,  # Loss
                'holding_period': 200  # Short-term
            }
        }
        
        trades = rebalancer.calculate_tax_aware_trades(
            current_positions=positions_with_tax,
            target_weights={'AAPL': 0.4, 'GOOGL': 0.6},
            tax_rates={'long_term': 0.15, 'short_term': 0.35}
        )
        
        # Should prefer selling positions with losses
        googl_trade = next((t for t in trades if t['symbol'] == 'GOOGL'), None)
        if googl_trade and googl_trade['action'] == 'sell':
            assert googl_trade['tax_impact'] <= 0  # Should be a loss


class TestRiskManager:
    """Test RiskManager class."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create RiskManager instance."""
        limits = RiskLimits(
            max_position_size=0.15,
            max_sector_exposure=0.40,
            max_correlation=0.80,
            max_leverage=1.5,
            max_var=0.02,
            max_drawdown=0.20
        )
        
        return RiskManager(
            risk_limits=limits,
            confidence_level=0.95
        )
        
    def test_position_risk_check(self, risk_manager):
        """Test position-level risk checks."""
        position = {
            'symbol': 'AAPL',
            'value': 25000,
            'portfolio_value': 200000,
            'volatility': 0.025,
            'beta': 1.2
        }
        
        result = risk_manager.check_position_risk(position)
        
        assert isinstance(result, dict)
        assert 'passed' in result
        assert 'violations' in result
        
        # Position is 12.5% of portfolio, should pass
        assert result['passed'] is True
        
        # Test violation
        position['value'] = 35000  # 17.5% of portfolio
        result = risk_manager.check_position_risk(position)
        
        assert result['passed'] is False
        assert 'max_position_size' in result['violations']
        
    def test_portfolio_risk_metrics(self, risk_manager):
        """Test portfolio risk metrics calculation."""
        portfolio = {
            'positions': [
                {'symbol': 'AAPL', 'value': 30000, 'returns': np.random.normal(0.001, 0.02, 252)},
                {'symbol': 'GOOGL', 'value': 40000, 'returns': np.random.normal(0.0008, 0.025, 252)},
                {'symbol': 'MSFT', 'value': 30000, 'returns': np.random.normal(0.0012, 0.018, 252)}
            ],
            'cash': 0,
            'leverage': 1.0
        }
        
        metrics = risk_manager.calculate_risk_metrics(portfolio)
        
        assert isinstance(metrics, RiskMetrics)
        assert hasattr(metrics, 'var_95')
        assert hasattr(metrics, 'cvar_95')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'sortino_ratio')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'beta')
        
        # Check reasonable values
        assert -0.10 < metrics.var_95 < 0
        assert metrics.cvar_95 <= metrics.var_95
        assert 0 < metrics.volatility < 0.50
        
    def test_correlation_risk(self, risk_manager):
        """Test correlation risk detection."""
        correlation_matrix = pd.DataFrame({
            'AAPL': [1.0, 0.85, 0.3],
            'MSFT': [0.85, 1.0, 0.4],
            'JPM': [0.3, 0.4, 1.0]
        }, index=['AAPL', 'MSFT', 'JPM'])
        
        violations = risk_manager.check_correlation_risk(
            correlation_matrix,
            position_weights={'AAPL': 0.4, 'MSFT': 0.4, 'JPM': 0.2}
        )
        
        assert isinstance(violations, list)
        assert len(violations) > 0  # Should detect high correlation between AAPL and MSFT
        
        violation = violations[0]
        assert 'AAPL' in violation['pair'] or 'MSFT' in violation['pair']
        assert violation['correlation'] == 0.85
        
    def test_risk_alerts(self, risk_manager):
        """Test risk alert generation."""
        current_metrics = RiskMetrics(
            var_95=-0.025,
            cvar_95=-0.035,
            volatility=0.20,
            sharpe_ratio=0.8,
            sortino_ratio=1.0,
            max_drawdown=-0.18,
            beta=1.1
        )
        
        alerts = risk_manager.generate_risk_alerts(current_metrics)
        
        assert isinstance(alerts, list)
        
        # Should have alerts for VaR and drawdown close to limits
        var_alert = next((a for a in alerts if 'var' in a.metric.lower()), None)
        assert var_alert is not None
        assert var_alert.severity in ['warning', 'critical']
        
    def test_stress_test_integration(self, risk_manager):
        """Test stress test integration with risk management."""
        portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 500))
        
        stress_scenarios = [
            {'name': 'Market Crash', 'shock': -0.20},
            {'name': 'Interest Rate Spike', 'shock': -0.10},
            {'name': 'Flash Crash', 'shock': -0.30}
        ]
        
        results = risk_manager.run_stress_tests(
            portfolio_returns,
            stress_scenarios
        )
        
        assert len(results) == len(stress_scenarios)
        
        for result in results:
            assert 'scenario' in result
            assert 'impact' in result
            assert 'new_var' in result
            assert 'breaches_limits' in result


class TestStressTester:
    """Test StressTester class."""
    
    @pytest.fixture
    def stress_tester(self):
        """Create StressTester instance."""
        return StressTester(
            historical_scenarios=[
                StressScenario(
                    name='2008 Financial Crisis',
                    market_shock=-0.37,
                    volatility_multiplier=3.0,
                    correlation_increase=0.3,
                    duration_days=180
                ),
                StressScenario(
                    name='COVID-19 Crash',
                    market_shock=-0.34,
                    volatility_multiplier=4.0,
                    correlation_increase=0.4,
                    duration_days=30
                )
            ]
        )
        
    def test_historical_stress_test(self, stress_tester):
        """Test historical scenario stress testing."""
        portfolio = pd.DataFrame({
            'AAPL': 100 + np.random.randn(1000).cumsum(),
            'GOOGL': 2500 + 5 * np.random.randn(1000).cumsum(),
            'JPM': 100 + 2 * np.random.randn(1000).cumsum()
        })
        
        weights = {'AAPL': 0.3, 'GOOGL': 0.4, 'JPM': 0.3}
        
        results = stress_tester.run_historical_scenarios(portfolio, weights)
        
        assert len(results) == 2  # Two scenarios
        
        for result in results:
            assert isinstance(result, StressTestResult)
            assert result.portfolio_impact < 0  # Should be negative
            assert result.worst_asset is not None
            assert result.var_impact is not None
            
    def test_monte_carlo_stress_test(self, stress_tester):
        """Test Monte Carlo stress testing."""
        portfolio_stats = {
            'expected_return': 0.08,
            'volatility': 0.15,
            'var_95': -0.025
        }
        
        results = stress_tester.run_monte_carlo(
            portfolio_stats,
            n_simulations=1000,
            time_horizon=252,
            shock_probability=0.05
        )
        
        assert 'simulated_returns' in results
        assert 'var_distribution' in results
        assert 'probability_of_loss' in results
        assert 'expected_shortfall' in results
        
        # Check reasonable values
        assert len(results['simulated_returns']) == 1000
        assert 0 < results['probability_of_loss'] < 1
        
    def test_factor_stress_test(self, stress_tester):
        """Test factor-based stress testing."""
        factor_exposures = {
            'AAPL': {'market': 1.1, 'tech': 0.8, 'growth': 0.6},
            'JPM': {'market': 0.9, 'financial': 0.9, 'value': 0.7},
            'XOM': {'market': 0.8, 'energy': 0.95, 'value': 0.8}
        }
        
        factor_shocks = {
            'market': -0.15,
            'tech': -0.20,
            'financial': -0.25,
            'energy': 0.10,  # Energy might rise in crisis
            'growth': -0.30,
            'value': -0.10
        }
        
        weights = {'AAPL': 0.4, 'JPM': 0.3, 'XOM': 0.3}
        
        result = stress_tester.run_factor_stress_test(
            factor_exposures,
            factor_shocks,
            weights
        )
        
        assert 'portfolio_impact' in result
        assert 'asset_impacts' in result
        assert 'factor_contributions' in result
        
        # AAPL should be hit hardest (tech and growth exposure)
        assert result['asset_impacts']['AAPL'] < result['asset_impacts']['XOM']
        
    def test_liquidity_stress_test(self, stress_tester):
        """Test liquidity stress testing."""
        positions = {
            'AAPL': {'value': 100000, 'avg_daily_volume': 50000000, 'shares': 650},
            'SMALL': {'value': 50000, 'avg_daily_volume': 100000, 'shares': 5000},
            'MICRO': {'value': 30000, 'avg_daily_volume': 10000, 'shares': 10000}
        }
        
        liquidity_shock = {
            'volume_reduction': 0.8,  # 80% reduction
            'spread_widening': 5.0,   # 5x spread
            'market_impact_multiplier': 3.0
        }
        
        result = stress_tester.test_liquidity_stress(
            positions,
            liquidity_shock,
            liquidation_horizon=5  # 5 days to liquidate
        )
        
        assert 'total_cost' in result
        assert 'days_to_liquidate' in result
        assert 'position_impacts' in result
        
        # Small and micro cap should have higher costs
        assert result['position_impacts']['MICRO']['cost_pct'] > \
               result['position_impacts']['AAPL']['cost_pct']


class TestAllocationOptimizer:
    """Test allocation optimization scenarios."""
    
    def test_strategic_allocation(self):
        """Test strategic asset allocation."""
        allocator = AssetAllocator(
            risk_tolerance='moderate',
            investment_horizon=10,
            rebalance_frequency='quarterly'
        )
        
        asset_classes = {
            'stocks': {'expected_return': 0.08, 'volatility': 0.16},
            'bonds': {'expected_return': 0.04, 'volatility': 0.05},
            'reits': {'expected_return': 0.06, 'volatility': 0.19},
            'commodities': {'expected_return': 0.05, 'volatility': 0.20}
        }
        
        correlation_matrix = pd.DataFrame({
            'stocks': [1.0, -0.1, 0.6, 0.2],
            'bonds': [-0.1, 1.0, 0.1, 0.0],
            'reits': [0.6, 0.1, 1.0, 0.3],
            'commodities': [0.2, 0.0, 0.3, 1.0]
        }, index=['stocks', 'bonds', 'reits', 'commodities'])
        
        allocation = allocator.optimize_strategic_allocation(
            asset_classes,
            correlation_matrix,
            constraints=AllocationConstraints(
                min_allocation={'bonds': 0.2},  # At least 20% bonds
                max_allocation={'commodities': 0.15}  # Max 15% commodities
            )
        )
        
        assert sum(allocation.values()) == pytest.approx(1.0)
        assert allocation['bonds'] >= 0.2
        assert allocation['commodities'] <= 0.15
        
        # Moderate risk should have balanced allocation
        assert 0.3 < allocation['stocks'] < 0.6
        assert 0.2 < allocation['bonds'] < 0.5


class TestDrawdownAnalyzer:
    """Test DrawdownAnalyzer class."""
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        analyzer = DrawdownAnalyzer()
        
        # Create price series with clear drawdown
        prices = pd.Series([
            100, 105, 110, 108, 95, 92, 98, 102, 107, 103, 111
        ])
        
        drawdowns = analyzer.calculate_drawdowns(prices)
        
        assert isinstance(drawdowns, pd.Series)
        assert (drawdowns <= 0).all()  # Drawdowns are negative
        
        # Maximum drawdown should be from 110 to 92
        max_dd = drawdowns.min()
        expected_max_dd = (92 - 110) / 110
        assert abs(max_dd - expected_max_dd) < 0.001
        
    def test_drawdown_metrics(self):
        """Test comprehensive drawdown metrics."""
        analyzer = DrawdownAnalyzer()
        
        # Create returns with multiple drawdown periods
        dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
        returns = pd.Series(
            np.random.normal(0.0005, 0.02, 500),
            index=dates
        )
        
        # Insert some drawdown periods
        returns.iloc[100:120] = -0.02  # 20-day drawdown
        returns.iloc[200:210] = -0.03  # 10-day severe drawdown
        
        prices = (1 + returns).cumprod() * 100
        
        metrics = analyzer.analyze_drawdowns(prices)
        
        assert isinstance(metrics, DrawdownMetrics)
        assert metrics.max_drawdown < 0
        assert metrics.max_duration > 0
        assert len(metrics.top_drawdowns) <= 5
        assert metrics.avg_drawdown < 0
        assert 0 <= metrics.recovery_factor <= 1
        
    def test_recovery_analysis(self):
        """Test drawdown recovery analysis."""
        analyzer = DrawdownAnalyzer()
        
        # Create price series with recovery
        prices = pd.Series([
            100, 110, 105, 90, 85, 88, 92, 96, 100, 105, 112
        ])
        
        recovery = analyzer.analyze_recovery(prices, drawdown_threshold=-0.10)
        
        assert isinstance(recovery, RecoveryAnalysis)
        assert recovery.time_to_recovery > 0
        assert recovery.recovery_rate > 0
        assert recovery.max_recovery_drawdown <= 0


class TestPerformanceAttribution:
    """Test performance attribution analysis."""
    
    def test_sector_attribution(self):
        """Test sector-based performance attribution."""
        attributor = PerformanceAttributor()
        
        portfolio_weights = {
            'AAPL': 0.15, 'MSFT': 0.10,  # Tech: 25%
            'JPM': 0.10, 'BAC': 0.08,     # Finance: 18%
            'JNJ': 0.12, 'PFE': 0.08,     # Healthcare: 20%
            'XOM': 0.07                    # Energy: 7%
            # Cash: 30%
        }
        
        sector_mapping = {
            'AAPL': 'Tech', 'MSFT': 'Tech',
            'JPM': 'Finance', 'BAC': 'Finance',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare',
            'XOM': 'Energy'
        }
        
        asset_returns = {
            'AAPL': 0.15, 'MSFT': 0.12,
            'JPM': 0.08, 'BAC': 0.06,
            'JNJ': 0.10, 'PFE': 0.09,
            'XOM': -0.05
        }
        
        benchmark_sector_weights = {
            'Tech': 0.30,
            'Finance': 0.15,
            'Healthcare': 0.15,
            'Energy': 0.10,
            'Cash': 0.30
        }
        
        benchmark_sector_returns = {
            'Tech': 0.14,
            'Finance': 0.07,
            'Healthcare': 0.08,
            'Energy': -0.02,
            'Cash': 0.02
        }
        
        result = attributor.sector_attribution(
            portfolio_weights,
            sector_mapping,
            asset_returns,
            benchmark_sector_weights,
            benchmark_sector_returns
        )
        
        assert isinstance(result, AttributionResult)
        assert hasattr(result, 'allocation_effect')
        assert hasattr(result, 'selection_effect')
        assert hasattr(result, 'interaction_effect')
        assert hasattr(result, 'total_effect')
        
        # Total effect should equal sum of components
        total_calculated = (
            sum(result.allocation_effect.values()) +
            sum(result.selection_effect.values()) +
            sum(result.interaction_effect.values())
        )
        
        assert abs(result.total_effect - total_calculated) < 0.001
        
    def test_factor_attribution(self):
        """Test factor-based performance attribution."""
        attributor = PerformanceAttributor()
        
        factor_exposures = pd.DataFrame({
            'Market': [1.1, 0.9, 1.0, 0.8],
            'Size': [0.3, -0.2, 0.0, -0.4],
            'Value': [-0.2, 0.4, 0.3, 0.5],
            'Momentum': [0.4, 0.1, -0.1, -0.3]
        }, index=['AAPL', 'JPM', 'JNJ', 'XOM'])
        
        factor_returns = {
            'Market': 0.08,
            'Size': 0.02,
            'Value': 0.04,
            'Momentum': -0.01
        }
        
        asset_returns = pd.Series({
            'AAPL': 0.12,
            'JPM': 0.09,
            'JNJ': 0.08,
            'XOM': 0.06
        })
        
        weights = pd.Series({
            'AAPL': 0.3,
            'JPM': 0.3,
            'JNJ': 0.2,
            'XOM': 0.2
        })
        
        result = attributor.factor_attribution(
            factor_exposures,
            factor_returns,
            asset_returns,
            weights
        )
        
        assert isinstance(result, FactorAnalysis)
        assert hasattr(result, 'factor_contributions')
        assert hasattr(result, 'specific_returns')
        assert hasattr(result, 'r_squared')
        
        # Factor contributions should sum close to total return
        total_factor_contribution = sum(result.factor_contributions.values())
        portfolio_return = (asset_returns * weights).sum()
        
        # Difference is specific return
        specific = portfolio_return - total_factor_contribution
        assert abs(specific - result.specific_returns) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])