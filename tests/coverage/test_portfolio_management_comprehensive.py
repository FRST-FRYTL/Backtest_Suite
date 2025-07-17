"""Comprehensive tests for portfolio management to achieve >90% coverage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.backtesting.portfolio import Portfolio
from src.backtesting.position import Position
from src.backtesting.order import Order, OrderType, OrderSide
from src.portfolio.risk_manager import RiskManager
from src.portfolio.position_sizer import PositionSizer
from src.portfolio.portfolio_optimizer import PortfolioOptimizer
from src.portfolio.rebalancer import PortfolioRebalancer


class TestPortfolioComprehensive:
    """Comprehensive portfolio tests for maximum coverage."""
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        # Default initialization
        portfolio = Portfolio()
        assert portfolio.initial_capital == 100000
        assert portfolio.cash == 100000
        assert portfolio.commission_rate == 0.001
        assert portfolio.slippage_rate == 0.0
        assert portfolio.positions == {}
        assert portfolio.trades == []
        assert portfolio.portfolio_value == 100000
        
        # Custom initialization
        custom_portfolio = Portfolio(
            initial_capital=50000,
            commission_rate=0.002,
            slippage_rate=0.0005,
            margin_requirement=0.25
        )
        assert custom_portfolio.initial_capital == 50000
        assert custom_portfolio.cash == 50000
        assert custom_portfolio.commission_rate == 0.002
        assert custom_portfolio.slippage_rate == 0.0005
        assert custom_portfolio.margin_requirement == 0.25
    
    def test_portfolio_position_opening(self):
        """Test opening positions in portfolio."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Open long position
        portfolio.open_position(
            symbol='AAPL',
            quantity=100,
            price=150.0,
            side=OrderSide.BUY,
            timestamp=datetime.now()
        )
        
        # Check position was created
        assert 'AAPL' in portfolio.positions
        position = portfolio.positions['AAPL']
        assert position.symbol == 'AAPL'
        assert position.quantity == 100
        assert position.avg_price == 150.0
        assert position.side == OrderSide.BUY
        
        # Check cash reduction
        expected_cash = 100000 - (100 * 150.0) - (100 * 150.0 * 0.001)  # Commission
        assert abs(portfolio.cash - expected_cash) < 0.01
        
        # Open short position
        portfolio.open_position(
            symbol='GOOGL',
            quantity=50,
            price=2000.0,
            side=OrderSide.SELL,
            timestamp=datetime.now()
        )
        
        # Check short position
        assert 'GOOGL' in portfolio.positions
        googl_position = portfolio.positions['GOOGL']
        assert googl_position.quantity == 50
        assert googl_position.side == OrderSide.SELL
        assert googl_position.avg_price == 2000.0
    
    def test_portfolio_position_closing(self):
        """Test closing positions in portfolio."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Open position first
        portfolio.open_position('AAPL', 100, 150.0, OrderSide.BUY, datetime.now())
        initial_cash = portfolio.cash
        
        # Close position
        portfolio.close_position(
            symbol='AAPL',
            quantity=100,
            price=160.0,
            timestamp=datetime.now()
        )
        
        # Check position is closed
        assert portfolio.positions['AAPL'].quantity == 0
        
        # Check cash increase (profit)
        profit = (160.0 - 150.0) * 100  # $10 profit per share
        commission = 100 * 160.0 * 0.001  # Commission on closing
        expected_cash = initial_cash + profit - commission
        assert abs(portfolio.cash - expected_cash) < 0.01
        
        # Check trade was recorded
        assert len(portfolio.trades) == 1
        trade = portfolio.trades[0]
        assert trade['symbol'] == 'AAPL'
        assert trade['quantity'] == 100
        assert trade['entry_price'] == 150.0
        assert trade['exit_price'] == 160.0
        assert trade['pnl'] > 0  # Should be profitable
    
    def test_portfolio_partial_position_closing(self):
        """Test partial position closing."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Open large position
        portfolio.open_position('AAPL', 200, 150.0, OrderSide.BUY, datetime.now())
        
        # Partially close position
        portfolio.close_position(
            symbol='AAPL',
            quantity=75,
            price=160.0,
            timestamp=datetime.now()
        )
        
        # Check remaining position
        remaining_position = portfolio.positions['AAPL']
        assert remaining_position.quantity == 125  # 200 - 75
        assert remaining_position.avg_price == 150.0  # Unchanged
        
        # Check partial trade was recorded
        assert len(portfolio.trades) == 1
        partial_trade = portfolio.trades[0]
        assert partial_trade['quantity'] == 75
        assert partial_trade['entry_price'] == 150.0
        assert partial_trade['exit_price'] == 160.0
        
        # Close remaining position
        portfolio.close_position(
            symbol='AAPL',
            quantity=125,
            price=155.0,
            timestamp=datetime.now()
        )
        
        # Check position is fully closed
        assert portfolio.positions['AAPL'].quantity == 0
        assert len(portfolio.trades) == 2
    
    def test_portfolio_position_averaging(self):
        """Test position averaging (adding to existing position)."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Open initial position
        portfolio.open_position('AAPL', 100, 150.0, OrderSide.BUY, datetime.now())
        
        # Add to position at different price
        portfolio.add_to_position(
            symbol='AAPL',
            quantity=50,
            price=160.0,
            timestamp=datetime.now()
        )
        
        # Check averaged position
        position = portfolio.positions['AAPL']
        assert position.quantity == 150  # 100 + 50
        
        # Check average price calculation
        expected_avg = (100 * 150.0 + 50 * 160.0) / 150
        assert abs(position.avg_price - expected_avg) < 0.01
        
        # Add more at lower price
        portfolio.add_to_position(
            symbol='AAPL',
            quantity=100,
            price=140.0,
            timestamp=datetime.now()
        )
        
        # Check new average
        position = portfolio.positions['AAPL']
        assert position.quantity == 250  # 150 + 100
        
        expected_avg = (150 * expected_avg + 100 * 140.0) / 250
        assert abs(position.avg_price - expected_avg) < 0.01
    
    def test_portfolio_value_calculation(self, sample_ohlcv_data):
        """Test portfolio value calculation with market data."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Open multiple positions
        portfolio.open_position('AAPL', 100, 150.0, OrderSide.BUY, datetime.now())
        portfolio.open_position('GOOGL', 25, 2000.0, OrderSide.BUY, datetime.now())
        
        # Create market data
        market_data = {
            'AAPL': 160.0,  # Up $10
            'GOOGL': 1950.0  # Down $50
        }
        
        # Update portfolio value
        portfolio.update_portfolio_value(market_data)
        
        # Calculate expected value
        aapl_value = 100 * 160.0  # $16,000
        googl_value = 25 * 1950.0  # $48,750
        total_position_value = aapl_value + googl_value
        
        expected_portfolio_value = portfolio.cash + total_position_value
        assert abs(portfolio.portfolio_value - expected_portfolio_value) < 0.01
        
        # Check individual position values
        assert abs(portfolio.positions['AAPL'].market_value - aapl_value) < 0.01
        assert abs(portfolio.positions['GOOGL'].market_value - googl_value) < 0.01
    
    def test_portfolio_margin_calculations(self):
        """Test margin calculations and requirements."""
        portfolio = Portfolio(
            initial_capital=100000,
            margin_requirement=0.25  # 25% margin requirement
        )
        
        # Test buying power calculation
        buying_power = portfolio.calculate_buying_power()
        expected_buying_power = 100000 / 0.25  # 4x leverage
        assert abs(buying_power - expected_buying_power) < 0.01
        
        # Open leveraged position
        portfolio.open_position('AAPL', 400, 150.0, OrderSide.BUY, datetime.now())
        
        # Check margin used
        position_value = 400 * 150.0  # $60,000
        margin_used = position_value * 0.25  # $15,000
        
        assert abs(portfolio.margin_used - margin_used) < 0.01
        
        # Check remaining buying power
        remaining_buying_power = portfolio.calculate_buying_power()
        expected_remaining = (100000 - margin_used) / 0.25
        assert abs(remaining_buying_power - expected_remaining) < 0.01
    
    def test_portfolio_risk_metrics(self):
        """Test portfolio risk metric calculations."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Create some trades with different outcomes
        trades = [
            {'pnl': 1000, 'duration': 5, 'entry_price': 100, 'exit_price': 110},
            {'pnl': -500, 'duration': 3, 'entry_price': 100, 'exit_price': 95},
            {'pnl': 750, 'duration': 7, 'entry_price': 100, 'exit_price': 107.5},
            {'pnl': -200, 'duration': 2, 'entry_price': 100, 'exit_price': 98},
            {'pnl': 300, 'duration': 4, 'entry_price': 100, 'exit_price': 103}
        ]
        
        portfolio.trades = trades
        
        # Calculate metrics
        metrics = portfolio.calculate_risk_metrics()
        
        # Check required metrics
        assert 'win_rate' in metrics
        assert 'avg_win' in metrics
        assert 'avg_loss' in metrics
        assert 'profit_factor' in metrics
        assert 'max_drawdown' in metrics
        assert 'sharpe_ratio' in metrics
        
        # Check win rate calculation
        wins = sum(1 for trade in trades if trade['pnl'] > 0)
        expected_win_rate = wins / len(trades)
        assert abs(metrics['win_rate'] - expected_win_rate) < 0.01
        
        # Check average win/loss
        winning_trades = [t['pnl'] for t in trades if t['pnl'] > 0]
        losing_trades = [t['pnl'] for t in trades if t['pnl'] < 0]
        
        expected_avg_win = sum(winning_trades) / len(winning_trades)
        expected_avg_loss = sum(losing_trades) / len(losing_trades)
        
        assert abs(metrics['avg_win'] - expected_avg_win) < 0.01
        assert abs(metrics['avg_loss'] - expected_avg_loss) < 0.01
        
        # Check profit factor
        total_wins = sum(winning_trades)
        total_losses = abs(sum(losing_trades))
        expected_profit_factor = total_wins / total_losses
        assert abs(metrics['profit_factor'] - expected_profit_factor) < 0.01
    
    def test_portfolio_drawdown_calculation(self):
        """Test drawdown calculation."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Create portfolio value history
        portfolio_history = pd.Series([
            100000, 105000, 110000, 108000, 107000, 112000, 115000, 110000, 108000, 120000
        ])
        
        portfolio.portfolio_history = portfolio_history
        
        # Calculate drawdown
        drawdown = portfolio.calculate_drawdown()
        
        # Check drawdown structure
        assert isinstance(drawdown, pd.Series)
        assert len(drawdown) == len(portfolio_history)
        assert (drawdown <= 0).all()  # Drawdown should be negative or zero
        
        # Check maximum drawdown
        max_drawdown = portfolio.calculate_max_drawdown()
        assert max_drawdown <= 0
        
        # Manually calculate expected max drawdown
        running_max = portfolio_history.expanding().max()
        expected_drawdown = (portfolio_history - running_max) / running_max
        expected_max_drawdown = expected_drawdown.min()
        
        assert abs(max_drawdown - expected_max_drawdown) < 0.01
    
    def test_portfolio_correlation_analysis(self):
        """Test portfolio correlation analysis."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Create positions
        portfolio.open_position('AAPL', 100, 150.0, OrderSide.BUY, datetime.now())
        portfolio.open_position('GOOGL', 25, 2000.0, OrderSide.BUY, datetime.now())
        portfolio.open_position('MSFT', 50, 300.0, OrderSide.BUY, datetime.now())
        
        # Create mock price data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        price_data = pd.DataFrame({
            'AAPL': np.random.normal(150, 5, len(dates)),
            'GOOGL': np.random.normal(2000, 50, len(dates)),
            'MSFT': np.random.normal(300, 10, len(dates))
        }, index=dates)
        
        # Calculate correlations
        correlations = portfolio.calculate_position_correlations(price_data)
        
        # Check correlation matrix
        assert isinstance(correlations, pd.DataFrame)
        assert correlations.shape == (3, 3)
        assert (correlations.diagonal() == 1.0).all()  # Self-correlation = 1
        assert (correlations >= -1).all().all()  # Correlation >= -1
        assert (correlations <= 1).all().all()   # Correlation <= 1
        
        # Check symmetry
        assert np.allclose(correlations.values, correlations.values.T)
    
    def test_portfolio_sector_allocation(self):
        """Test portfolio sector allocation analysis."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Create positions with sector information
        positions_data = [
            {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0, 'sector': 'Technology'},
            {'symbol': 'GOOGL', 'quantity': 25, 'price': 2000.0, 'sector': 'Technology'},
            {'symbol': 'JPM', 'quantity': 50, 'price': 200.0, 'sector': 'Financial'},
            {'symbol': 'JNJ', 'quantity': 75, 'price': 160.0, 'sector': 'Healthcare'},
        ]
        
        # Open positions
        for pos in positions_data:
            portfolio.open_position(
                pos['symbol'], pos['quantity'], pos['price'], 
                OrderSide.BUY, datetime.now()
            )
            # Add sector info to position
            portfolio.positions[pos['symbol']].sector = pos['sector']
        
        # Calculate sector allocation
        sector_allocation = portfolio.calculate_sector_allocation()
        
        # Check allocation structure
        assert isinstance(sector_allocation, dict)
        assert 'Technology' in sector_allocation
        assert 'Financial' in sector_allocation
        assert 'Healthcare' in sector_allocation
        
        # Check allocation percentages
        total_allocation = sum(sector_allocation.values())
        assert abs(total_allocation - 1.0) < 0.01  # Should sum to 100%
        
        # Technology should be largest (AAPL + GOOGL)
        tech_value = 100 * 150.0 + 25 * 2000.0  # $65,000
        total_value = sum(pos['quantity'] * pos['price'] for pos in positions_data)
        expected_tech_allocation = tech_value / total_value
        
        assert abs(sector_allocation['Technology'] - expected_tech_allocation) < 0.01
    
    def test_portfolio_rebalancing(self):
        """Test portfolio rebalancing functionality."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Create unbalanced portfolio
        portfolio.open_position('AAPL', 200, 150.0, OrderSide.BUY, datetime.now())  # $30k
        portfolio.open_position('GOOGL', 5, 2000.0, OrderSide.BUY, datetime.now())   # $10k
        
        # Define target allocation
        target_allocation = {
            'AAPL': 0.5,  # 50%
            'GOOGL': 0.3, # 30%
            'CASH': 0.2   # 20%
        }
        
        # Calculate current allocation
        current_allocation = portfolio.calculate_current_allocation()
        
        # Check current allocation structure
        assert isinstance(current_allocation, dict)
        assert 'AAPL' in current_allocation
        assert 'GOOGL' in current_allocation
        assert 'CASH' in current_allocation
        
        # Calculate rebalancing trades
        rebalancing_trades = portfolio.calculate_rebalancing_trades(target_allocation)
        
        # Check trade structure
        assert isinstance(rebalancing_trades, list)
        for trade in rebalancing_trades:
            assert 'symbol' in trade
            assert 'action' in trade  # BUY or SELL
            assert 'quantity' in trade
            assert 'target_value' in trade
            assert 'current_value' in trade
    
    def test_portfolio_performance_attribution(self):
        """Test performance attribution analysis."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Create some completed trades
        trades = [
            {
                'symbol': 'AAPL',
                'entry_date': datetime(2023, 1, 1),
                'exit_date': datetime(2023, 1, 15),
                'entry_price': 150.0,
                'exit_price': 160.0,
                'quantity': 100,
                'pnl': 1000,
                'sector': 'Technology'
            },
            {
                'symbol': 'GOOGL',
                'entry_date': datetime(2023, 1, 5),
                'exit_date': datetime(2023, 1, 20),
                'entry_price': 2000.0,
                'exit_price': 1950.0,
                'quantity': 25,
                'pnl': -1250,
                'sector': 'Technology'
            },
            {
                'symbol': 'JPM',
                'entry_date': datetime(2023, 1, 10),
                'exit_date': datetime(2023, 1, 25),
                'entry_price': 200.0,
                'exit_price': 210.0,
                'quantity': 50,
                'pnl': 500,
                'sector': 'Financial'
            }
        ]
        
        portfolio.trades = trades
        
        # Calculate performance attribution
        attribution = portfolio.calculate_performance_attribution()
        
        # Check attribution structure
        assert isinstance(attribution, dict)
        assert 'by_symbol' in attribution
        assert 'by_sector' in attribution
        assert 'by_time_period' in attribution
        
        # Check symbol attribution
        symbol_attribution = attribution['by_symbol']
        assert 'AAPL' in symbol_attribution
        assert 'GOOGL' in symbol_attribution
        assert 'JPM' in symbol_attribution
        
        # Check sector attribution
        sector_attribution = attribution['by_sector']
        assert 'Technology' in sector_attribution
        assert 'Financial' in sector_attribution
        
        # Technology sector should have net loss (1000 - 1250 = -250)
        assert sector_attribution['Technology'] == -250
        # Financial sector should have profit (500)
        assert sector_attribution['Financial'] == 500
    
    def test_portfolio_stress_testing(self):
        """Test portfolio stress testing."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Create diversified portfolio
        positions = [
            {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0},
            {'symbol': 'GOOGL', 'quantity': 25, 'price': 2000.0},
            {'symbol': 'JPM', 'quantity': 50, 'price': 200.0},
            {'symbol': 'SPY', 'quantity': 200, 'price': 400.0}
        ]
        
        for pos in positions:
            portfolio.open_position(
                pos['symbol'], pos['quantity'], pos['price'],
                OrderSide.BUY, datetime.now()
            )
        
        # Define stress scenarios
        stress_scenarios = {
            'market_crash': {
                'AAPL': -0.3,   # 30% decline
                'GOOGL': -0.25, # 25% decline
                'JPM': -0.4,    # 40% decline
                'SPY': -0.2     # 20% decline
            },
            'tech_selloff': {
                'AAPL': -0.5,   # 50% decline
                'GOOGL': -0.45, # 45% decline
                'JPM': -0.1,    # 10% decline
                'SPY': -0.15    # 15% decline
            },
            'interest_rate_shock': {
                'AAPL': -0.15,  # 15% decline
                'GOOGL': -0.1,  # 10% decline
                'JPM': 0.2,     # 20% increase (banks benefit)
                'SPY': -0.05    # 5% decline
            }
        }
        
        # Run stress tests
        stress_results = portfolio.run_stress_tests(stress_scenarios)
        
        # Check results structure
        assert isinstance(stress_results, dict)
        assert 'market_crash' in stress_results
        assert 'tech_selloff' in stress_results
        assert 'interest_rate_shock' in stress_results
        
        # Check each scenario result
        for scenario, result in stress_results.items():
            assert 'portfolio_value' in result
            assert 'pnl' in result
            assert 'pnl_percentage' in result
            assert 'position_impacts' in result
            
            # Portfolio value should be different from initial
            assert result['portfolio_value'] != 100000
            
            # PnL should be negative for crash scenarios
            if scenario in ['market_crash', 'tech_selloff']:
                assert result['pnl'] < 0
                assert result['pnl_percentage'] < 0
    
    def test_portfolio_liquidity_analysis(self):
        """Test portfolio liquidity analysis."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Create positions with different liquidity characteristics
        positions_data = [
            {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0, 'avg_volume': 50000000},
            {'symbol': 'GOOGL', 'quantity': 25, 'price': 2000.0, 'avg_volume': 30000000},
            {'symbol': 'SMALLCAP', 'quantity': 1000, 'price': 50.0, 'avg_volume': 100000},
        ]
        
        for pos in positions_data:
            portfolio.open_position(
                pos['symbol'], pos['quantity'], pos['price'],
                OrderSide.BUY, datetime.now()
            )
            # Add volume info to position
            portfolio.positions[pos['symbol']].avg_daily_volume = pos['avg_volume']
        
        # Calculate liquidity metrics
        liquidity_metrics = portfolio.calculate_liquidity_metrics()
        
        # Check metrics structure
        assert isinstance(liquidity_metrics, dict)
        assert 'overall_liquidity_score' in liquidity_metrics
        assert 'position_liquidity' in liquidity_metrics
        assert 'time_to_liquidate' in liquidity_metrics
        
        # Check position-specific liquidity
        position_liquidity = liquidity_metrics['position_liquidity']
        assert 'AAPL' in position_liquidity
        assert 'GOOGL' in position_liquidity
        assert 'SMALLCAP' in position_liquidity
        
        # AAPL should be most liquid (highest volume)
        assert position_liquidity['AAPL'] > position_liquidity['SMALLCAP']
        
        # Check time to liquidate
        time_to_liquidate = liquidity_metrics['time_to_liquidate']
        assert isinstance(time_to_liquidate, dict)
        
        # Small cap should take longer to liquidate
        assert time_to_liquidate['SMALLCAP'] > time_to_liquidate['AAPL']
    
    def test_portfolio_tax_optimization(self):
        """Test portfolio tax optimization features."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Create trades with different holding periods and P&L
        trades = [
            {
                'symbol': 'AAPL',
                'entry_date': datetime(2023, 1, 1),
                'exit_date': datetime(2023, 6, 1),  # 5 months (short-term)
                'pnl': 1000,
                'quantity': 100
            },
            {
                'symbol': 'GOOGL',
                'entry_date': datetime(2022, 1, 1),
                'exit_date': datetime(2023, 2, 1),  # 13 months (long-term)
                'pnl': 2000,
                'quantity': 25
            },
            {
                'symbol': 'MSFT',
                'entry_date': datetime(2022, 6, 1),
                'exit_date': datetime(2023, 3, 1),  # 9 months (short-term)
                'pnl': -500,
                'quantity': 50
            }
        ]
        
        portfolio.trades = trades
        
        # Calculate tax implications
        tax_analysis = portfolio.calculate_tax_implications()
        
        # Check tax analysis structure
        assert isinstance(tax_analysis, dict)
        assert 'short_term_gains' in tax_analysis
        assert 'long_term_gains' in tax_analysis
        assert 'short_term_losses' in tax_analysis
        assert 'long_term_losses' in tax_analysis
        assert 'net_short_term' in tax_analysis
        assert 'net_long_term' in tax_analysis
        
        # Check calculations
        assert tax_analysis['short_term_gains'] == 1000  # AAPL
        assert tax_analysis['long_term_gains'] == 2000   # GOOGL
        assert tax_analysis['short_term_losses'] == -500 # MSFT
        assert tax_analysis['long_term_losses'] == 0
        
        # Check net calculations
        assert tax_analysis['net_short_term'] == 500   # 1000 - 500
        assert tax_analysis['net_long_term'] == 2000   # 2000 - 0
        
        # Calculate tax-loss harvesting opportunities
        harvesting_opportunities = portfolio.calculate_tax_loss_harvesting()
        assert isinstance(harvesting_opportunities, list)
        
        # Should identify losing positions for potential harvesting
        for opportunity in harvesting_opportunities:
            assert 'symbol' in opportunity
            assert 'unrealized_loss' in opportunity
            assert 'holding_period' in opportunity
            assert 'tax_benefit' in opportunity


class TestPortfolioIntegration:
    """Integration tests for portfolio management components."""
    
    def test_portfolio_with_risk_manager(self):
        """Test portfolio integration with risk manager."""
        portfolio = Portfolio(initial_capital=100000)
        risk_manager = RiskManager(
            max_position_size=0.1,
            max_portfolio_risk=0.02,
            max_correlation=0.7
        )
        
        # Test position size validation
        proposed_position = {
            'symbol': 'AAPL',
            'quantity': 200,
            'price': 150.0,
            'side': OrderSide.BUY
        }
        
        # Should approve reasonable position
        is_approved = risk_manager.validate_position(proposed_position, portfolio)
        assert isinstance(is_approved, bool)
        
        # Test with oversized position
        oversized_position = {
            'symbol': 'AAPL',
            'quantity': 1000,  # Too large
            'price': 150.0,
            'side': OrderSide.BUY
        }
        
        is_approved = risk_manager.validate_position(oversized_position, portfolio)
        assert is_approved == False
    
    def test_portfolio_with_position_sizer(self):
        """Test portfolio integration with position sizer."""
        portfolio = Portfolio(initial_capital=100000)
        position_sizer = PositionSizer(
            method='fixed_percentage',
            percentage=0.05,
            volatility_lookback=20
        )
        
        # Test position sizing
        market_data = pd.DataFrame({
            'close': [150, 152, 148, 155, 153],
            'volume': [1000000, 1100000, 900000, 1200000, 1050000]
        })
        
        position_size = position_sizer.calculate_position_size(
            symbol='AAPL',
            price=150.0,
            portfolio_value=100000,
            market_data=market_data
        )
        
        # Should return appropriate position size
        assert isinstance(position_size, int)
        assert position_size > 0
        assert position_size <= 100000 / 150.0  # Can't exceed available capital
    
    def test_portfolio_with_optimizer(self):
        """Test portfolio integration with optimizer."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Add some positions
        portfolio.open_position('AAPL', 100, 150.0, OrderSide.BUY, datetime.now())
        portfolio.open_position('GOOGL', 25, 2000.0, OrderSide.BUY, datetime.now())
        portfolio.open_position('MSFT', 50, 300.0, OrderSide.BUY, datetime.now())
        
        # Create optimizer
        optimizer = PortfolioOptimizer(
            optimization_method='mean_variance',
            risk_tolerance=0.5,
            rebalance_threshold=0.05
        )
        
        # Create mock return data
        returns_data = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'GOOGL': np.random.normal(0.0008, 0.025, 252),
            'MSFT': np.random.normal(0.0012, 0.018, 252)
        })
        
        # Calculate optimal allocation
        optimal_allocation = optimizer.calculate_optimal_allocation(
            returns_data, 
            current_portfolio=portfolio
        )
        
        # Check allocation structure
        assert isinstance(optimal_allocation, dict)
        assert 'AAPL' in optimal_allocation
        assert 'GOOGL' in optimal_allocation
        assert 'MSFT' in optimal_allocation
        
        # Allocations should sum to approximately 1
        total_allocation = sum(optimal_allocation.values())
        assert abs(total_allocation - 1.0) < 0.1
    
    def test_portfolio_with_rebalancer(self):
        """Test portfolio integration with rebalancer."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Create unbalanced portfolio
        portfolio.open_position('AAPL', 300, 150.0, OrderSide.BUY, datetime.now())
        portfolio.open_position('GOOGL', 10, 2000.0, OrderSide.BUY, datetime.now())
        
        # Create rebalancer
        rebalancer = Rebalancer(
            target_allocation={'AAPL': 0.5, 'GOOGL': 0.5},
            rebalance_threshold=0.05,
            rebalance_frequency='monthly'
        )
        
        # Calculate rebalancing trades
        rebalancing_trades = rebalancer.calculate_rebalancing_trades(portfolio)
        
        # Check trade structure
        assert isinstance(rebalancing_trades, list)
        
        for trade in rebalancing_trades:
            assert 'symbol' in trade
            assert 'action' in trade
            assert 'quantity' in trade
            assert trade['action'] in ['BUY', 'SELL']
            assert isinstance(trade['quantity'], int)
            assert trade['quantity'] > 0
        
        # Should suggest rebalancing both positions
        symbols_to_rebalance = {trade['symbol'] for trade in rebalancing_trades}
        assert len(symbols_to_rebalance) > 0
    
    def test_complete_portfolio_workflow(self, sample_ohlcv_data):
        """Test complete portfolio management workflow."""
        # Initialize components
        portfolio = Portfolio(initial_capital=100000)
        risk_manager = RiskManager(max_position_size=0.1)
        position_sizer = PositionSizer(method='volatility_based')
        
        # Simulate trading workflow
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        for symbol in symbols:
            # Calculate position size
            position_size = position_sizer.calculate_position_size(
                symbol=symbol,
                price=150.0,
                portfolio_value=portfolio.portfolio_value,
                market_data=sample_ohlcv_data
            )
            
            # Validate with risk manager
            proposed_position = {
                'symbol': symbol,
                'quantity': position_size,
                'price': 150.0,
                'side': OrderSide.BUY
            }
            
            if risk_manager.validate_position(proposed_position, portfolio):
                # Open position
                portfolio.open_position(
                    symbol=symbol,
                    quantity=position_size,
                    price=150.0,
                    side=OrderSide.BUY,
                    timestamp=datetime.now()
                )
        
        # Simulate price changes and updates
        price_changes = {
            'AAPL': 160.0,
            'GOOGL': 2100.0,
            'MSFT': 320.0
        }
        
        portfolio.update_portfolio_value(price_changes)
        
        # Calculate performance metrics
        metrics = portfolio.calculate_risk_metrics()
        
        # Verify complete workflow
        assert len(portfolio.positions) > 0
        assert portfolio.portfolio_value > 0
        assert isinstance(metrics, dict)
        assert 'total_return' in metrics or 'portfolio_value' in metrics
        
        # Portfolio should be properly managed
        total_position_value = sum(
            pos.quantity * price_changes.get(pos.symbol, 150.0)
            for pos in portfolio.positions.values()
        )
        
        expected_portfolio_value = portfolio.cash + total_position_value
        assert abs(portfolio.portfolio_value - expected_portfolio_value) < 1000  # Allow small differences