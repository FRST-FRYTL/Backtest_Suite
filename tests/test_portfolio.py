"""Comprehensive tests for portfolio management module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.backtesting.portfolio import Portfolio, PortfolioSnapshot
from src.backtesting.position import Position
from src.backtesting.order import Order, OrderType, OrderStatus


class TestPortfolioInitialization:
    """Test portfolio initialization and configuration."""
    
    def test_default_initialization(self):
        """Test portfolio with default parameters."""
        portfolio = Portfolio()
        
        assert portfolio.initial_capital == 100000
        assert portfolio.cash == 100000
        assert portfolio.commission_rate == 0.001
        assert portfolio.slippage_rate == 0.0005
        assert len(portfolio.positions) == 0
        assert len(portfolio.orders) == 0
        assert len(portfolio.trades) == 0
        
    def test_custom_initialization(self):
        """Test portfolio with custom parameters."""
        portfolio = Portfolio(
            initial_capital=50000,
            commission_rate=0.002,
            slippage_rate=0.001,
            min_commission=5.0
        )
        
        assert portfolio.initial_capital == 50000
        assert portfolio.cash == 50000
        assert portfolio.commission_rate == 0.002
        assert portfolio.slippage_rate == 0.001
        assert portfolio.min_commission == 5.0
        
    def test_invalid_initialization(self):
        """Test portfolio initialization with invalid parameters."""
        with pytest.raises(ValueError):
            Portfolio(initial_capital=-1000)
        
        with pytest.raises(ValueError):
            Portfolio(commission_rate=-0.001)
        
        with pytest.raises(ValueError):
            Portfolio(slippage_rate=1.5)


class TestOrderManagement:
    """Test order placement and management."""
    
    def test_market_order_placement(self, sample_portfolio):
        """Test placing market orders."""
        order = sample_portfolio.place_order(
            symbol="AAPL",
            quantity=100,
            direction="BUY",
            order_type=OrderType.MARKET
        )
        
        assert order.symbol == "AAPL"
        assert order.quantity == 100
        assert order.direction == "BUY"
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.SUBMITTED
        assert order in sample_portfolio.orders
        
    def test_limit_order_placement(self, sample_portfolio):
        """Test placing limit orders."""
        order = sample_portfolio.place_order(
            symbol="GOOGL",
            quantity=50,
            direction="SELL",
            order_type=OrderType.LIMIT,
            limit_price=150.0
        )
        
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 150.0
        assert order.status == OrderStatus.SUBMITTED
        
    def test_stop_order_placement(self, sample_portfolio):
        """Test placing stop orders."""
        order = sample_portfolio.place_order(
            symbol="MSFT",
            quantity=75,
            direction="SELL",
            order_type=OrderType.STOP,
            stop_price=95.0
        )
        
        assert order.order_type == OrderType.STOP
        assert order.stop_price == 95.0
        
    def test_order_validation(self, sample_portfolio):
        """Test order validation."""
        # Test invalid quantity
        with pytest.raises(ValueError):
            sample_portfolio.place_order("AAPL", 0, "BUY")
        
        # Test invalid direction
        with pytest.raises(ValueError):
            sample_portfolio.place_order("AAPL", 100, "INVALID")
        
        # Test missing limit price
        with pytest.raises(ValueError):
            sample_portfolio.place_order(
                "AAPL", 100, "BUY", 
                order_type=OrderType.LIMIT
            )
    
    def test_order_cancellation(self, sample_portfolio):
        """Test order cancellation."""
        order = sample_portfolio.place_order("AAPL", 100, "BUY")
        
        # Cancel order
        success = sample_portfolio.cancel_order(order.id)
        assert success is True
        assert order.status == OrderStatus.CANCELLED
        
        # Try to cancel already cancelled order
        success = sample_portfolio.cancel_order(order.id)
        assert success is False


class TestOrderExecution:
    """Test order execution logic."""
    
    def test_market_buy_execution(self, sample_portfolio):
        """Test executing market buy orders."""
        order = sample_portfolio.place_order("AAPL", 100, "BUY")
        
        # Execute at market price
        success = sample_portfolio.execute_order(
            order, 150.0, datetime.now()
        )
        
        assert success is True
        assert order.status == OrderStatus.FILLED
        assert order.fill_price == 150.0
        
        # Check position created
        assert "AAPL" in sample_portfolio.positions
        position = sample_portfolio.positions["AAPL"]
        assert position.quantity == 100
        assert position.avg_price == pytest.approx(150.0)
        
        # Check cash reduced
        expected_cash = 100000 - (100 * 150.0) - sample_portfolio.calculate_commission(100 * 150.0)
        assert sample_portfolio.cash == pytest.approx(expected_cash)
    
    def test_market_sell_execution(self, sample_portfolio):
        """Test executing market sell orders."""
        # First buy some shares
        buy_order = sample_portfolio.place_order("AAPL", 100, "BUY")
        sample_portfolio.execute_order(buy_order, 150.0, datetime.now())
        
        # Now sell
        sell_order = sample_portfolio.place_order("AAPL", 50, "SELL")
        success = sample_portfolio.execute_order(
            sell_order, 160.0, datetime.now()
        )
        
        assert success is True
        assert sell_order.status == OrderStatus.FILLED
        
        # Check position reduced
        position = sample_portfolio.positions["AAPL"]
        assert position.quantity == 50
        
        # Check realized P&L
        assert position.realized_pnl > 0
    
    def test_limit_order_execution(self, sample_portfolio):
        """Test limit order execution logic."""
        order = sample_portfolio.place_order(
            "AAPL", 100, "BUY",
            order_type=OrderType.LIMIT,
            limit_price=150.0
        )
        
        # Try to execute above limit - should fail
        success = sample_portfolio.execute_order(
            order, 151.0, datetime.now()
        )
        assert success is False
        assert order.status == OrderStatus.SUBMITTED
        
        # Execute at or below limit - should succeed
        success = sample_portfolio.execute_order(
            order, 149.0, datetime.now()
        )
        assert success is True
        assert order.status == OrderStatus.FILLED
        assert order.fill_price == 149.0
    
    def test_stop_order_execution(self, sample_portfolio):
        """Test stop order execution logic."""
        # Buy shares first
        buy_order = sample_portfolio.place_order("AAPL", 100, "BUY")
        sample_portfolio.execute_order(buy_order, 100.0, datetime.now())
        
        # Place stop loss order
        stop_order = sample_portfolio.place_order(
            "AAPL", 100, "SELL",
            order_type=OrderType.STOP,
            stop_price=95.0
        )
        
        # Price above stop - should not execute
        success = sample_portfolio.execute_order(
            stop_order, 96.0, datetime.now()
        )
        assert success is False
        
        # Price at or below stop - should execute
        success = sample_portfolio.execute_order(
            stop_order, 94.0, datetime.now()
        )
        assert success is True
        assert stop_order.fill_price == 94.0
    
    def test_insufficient_cash(self, sample_portfolio):
        """Test order execution with insufficient cash."""
        # Try to buy more than we can afford
        order = sample_portfolio.place_order("AAPL", 10000, "BUY")
        
        success = sample_portfolio.execute_order(
            order, 150.0, datetime.now()
        )
        
        assert success is False
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == "Insufficient cash"
    
    def test_insufficient_shares(self, sample_portfolio):
        """Test selling more shares than owned."""
        # Try to sell shares we don't own
        order = sample_portfolio.place_order("AAPL", 100, "SELL")
        
        success = sample_portfolio.execute_order(
            order, 150.0, datetime.now()
        )
        
        assert success is False
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == "Insufficient shares"


class TestPositionManagement:
    """Test position tracking and management."""
    
    def test_position_creation(self, sample_portfolio):
        """Test position creation through trades."""
        # Execute a buy order
        order = sample_portfolio.place_order("AAPL", 100, "BUY")
        sample_portfolio.execute_order(order, 150.0, datetime.now())
        
        position = sample_portfolio.positions["AAPL"]
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.avg_price == pytest.approx(150.0)
        assert position.is_long() is True
        assert position.is_open() is True
    
    def test_position_averaging(self, sample_portfolio):
        """Test position averaging on multiple buys."""
        # First buy
        order1 = sample_portfolio.place_order("AAPL", 100, "BUY")
        sample_portfolio.execute_order(order1, 150.0, datetime.now())
        
        # Second buy at different price
        order2 = sample_portfolio.place_order("AAPL", 50, "BUY")
        sample_portfolio.execute_order(order2, 160.0, datetime.now())
        
        position = sample_portfolio.positions["AAPL"]
        assert position.quantity == 150
        # Average price = (100 * 150 + 50 * 160) / 150 = 153.33
        assert position.avg_price == pytest.approx(153.33, rel=0.01)
    
    def test_position_closing(self, sample_portfolio):
        """Test closing positions."""
        # Buy shares
        buy_order = sample_portfolio.place_order("AAPL", 100, "BUY")
        sample_portfolio.execute_order(buy_order, 150.0, datetime.now())
        
        # Sell all shares
        sell_order = sample_portfolio.place_order("AAPL", 100, "SELL")
        sample_portfolio.execute_order(sell_order, 160.0, datetime.now())
        
        position = sample_portfolio.positions["AAPL"]
        assert position.quantity == 0
        assert position.is_open() is False
        assert position.realized_pnl > 0  # Should have profit
    
    def test_short_position(self, sample_portfolio):
        """Test short selling."""
        # Enable short selling
        sample_portfolio.allow_short = True
        
        # Short sell
        order = sample_portfolio.place_order("AAPL", 100, "SELL")
        success = sample_portfolio.execute_order(order, 150.0, datetime.now())
        
        assert success is True
        position = sample_portfolio.positions["AAPL"]
        assert position.quantity == -100
        assert position.is_short() is True
        assert position.avg_price == pytest.approx(150.0)
    
    def test_position_stops(self, sample_portfolio):
        """Test position stop loss and take profit."""
        # Buy with stops
        order = sample_portfolio.place_order("AAPL", 100, "BUY")
        sample_portfolio.execute_order(order, 100.0, datetime.now())
        
        # Set stops
        position = sample_portfolio.positions["AAPL"]
        position.stop_loss = 95.0
        position.take_profit = 110.0
        
        # Update price and check stops
        sample_portfolio.update_prices({"AAPL": 94.0})
        stop_hit = position.check_stops()
        assert stop_hit == "stop_loss"
        
        sample_portfolio.update_prices({"AAPL": 111.0})
        stop_hit = position.check_stops()
        assert stop_hit == "take_profit"


class TestPortfolioValuation:
    """Test portfolio valuation and P&L calculations."""
    
    def test_portfolio_value(self, sample_portfolio):
        """Test total portfolio value calculation."""
        # Initial value should equal cash
        assert sample_portfolio.current_value() == sample_portfolio.initial_capital
        
        # Buy some stocks
        order1 = sample_portfolio.place_order("AAPL", 100, "BUY")
        sample_portfolio.execute_order(order1, 150.0, datetime.now())
        
        order2 = sample_portfolio.place_order("GOOGL", 50, "BUY")
        sample_portfolio.execute_order(order2, 200.0, datetime.now())
        
        # Update prices
        sample_portfolio.update_prices({
            "AAPL": 160.0,
            "GOOGL": 210.0
        })
        
        # Calculate expected value
        cash = sample_portfolio.cash
        aapl_value = 100 * 160.0
        googl_value = 50 * 210.0
        expected_value = cash + aapl_value + googl_value
        
        assert sample_portfolio.current_value() == pytest.approx(expected_value)
    
    def test_unrealized_pnl(self, sample_portfolio):
        """Test unrealized P&L calculation."""
        # Buy stocks
        order = sample_portfolio.place_order("AAPL", 100, "BUY")
        sample_portfolio.execute_order(order, 150.0, datetime.now())
        
        # Price goes up
        sample_portfolio.update_prices({"AAPL": 160.0})
        
        unrealized_pnl = sample_portfolio.get_unrealized_pnl()
        assert unrealized_pnl == pytest.approx(1000.0)  # 100 * (160 - 150)
    
    def test_realized_pnl(self, sample_portfolio):
        """Test realized P&L tracking."""
        # Buy and sell for profit
        buy_order = sample_portfolio.place_order("AAPL", 100, "BUY")
        sample_portfolio.execute_order(buy_order, 150.0, datetime.now())
        
        sell_order = sample_portfolio.place_order("AAPL", 100, "SELL")
        sample_portfolio.execute_order(sell_order, 160.0, datetime.now())
        
        # Calculate expected P&L (including commissions)
        gross_pnl = 100 * (160.0 - 150.0)
        buy_commission = sample_portfolio.calculate_commission(100 * 150.0)
        sell_commission = sample_portfolio.calculate_commission(100 * 160.0)
        expected_pnl = gross_pnl - buy_commission - sell_commission
        
        assert sample_portfolio.get_realized_pnl() == pytest.approx(expected_pnl)
    
    def test_return_calculation(self, sample_portfolio):
        """Test return percentage calculation."""
        # Make some trades
        order1 = sample_portfolio.place_order("AAPL", 100, "BUY")
        sample_portfolio.execute_order(order1, 150.0, datetime.now())
        
        # Update price for unrealized gain
        sample_portfolio.update_prices({"AAPL": 165.0})
        
        # Sell half for realized gain
        order2 = sample_portfolio.place_order("AAPL", 50, "SELL")
        sample_portfolio.execute_order(order2, 165.0, datetime.now())
        
        returns = sample_portfolio.get_returns()
        assert returns['total_return'] > 0
        assert returns['total_return_pct'] > 0
        assert 'realized_return' in returns
        assert 'unrealized_return' in returns


class TestPositionSizing:
    """Test position sizing calculations."""
    
    def test_fixed_position_sizing(self, sample_portfolio):
        """Test fixed percentage position sizing."""
        # 10% of portfolio
        size = sample_portfolio.calculate_position_size(
            "AAPL", 150.0, position_pct=0.1
        )
        
        # 10% of 100k = 10k, at $150/share = 66.67 shares
        assert size == 66  # Rounds down
    
    def test_risk_based_sizing(self, sample_portfolio):
        """Test risk-based position sizing."""
        # Risk $1000 with 5% stop loss
        size = sample_portfolio.calculate_position_size(
            "AAPL", 100.0, risk_amount=1000, stop_loss_pct=0.05
        )
        
        # $1000 risk / ($100 * 0.05) = 200 shares
        assert size == 200
    
    def test_kelly_criterion_sizing(self, sample_portfolio):
        """Test Kelly criterion position sizing."""
        # Mock historical performance
        win_rate = 0.6
        avg_win = 0.1
        avg_loss = 0.05
        
        size = sample_portfolio.calculate_position_size(
            "AAPL", 100.0,
            sizing_method="kelly",
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss
        )
        
        # Kelly formula: f = p - q/b
        # where p = win_rate, q = 1-p, b = avg_win/avg_loss
        kelly_pct = win_rate - (1 - win_rate) / (avg_win / avg_loss)
        expected_size = int(sample_portfolio.cash * kelly_pct / 100.0)
        
        assert size == expected_size
    
    def test_max_position_limit(self, sample_portfolio):
        """Test maximum position size limits."""
        # Try to size for 50% of portfolio
        size = sample_portfolio.calculate_position_size(
            "AAPL", 100.0, position_pct=0.5, max_position_pct=0.2
        )
        
        # Should be capped at 20%
        expected_size = int(sample_portfolio.cash * 0.2 / 100.0)
        assert size == expected_size


class TestRiskManagement:
    """Test risk management features."""
    
    def test_portfolio_heat(self, sample_portfolio):
        """Test portfolio heat (total risk) calculation."""
        # Buy multiple positions with stop losses
        positions = [
            ("AAPL", 100, 150.0, 142.5),  # 5% stop
            ("GOOGL", 50, 200.0, 190.0),   # 5% stop
            ("MSFT", 75, 180.0, 171.0)     # 5% stop
        ]
        
        for symbol, qty, price, stop in positions:
            order = sample_portfolio.place_order(symbol, qty, "BUY")
            sample_portfolio.execute_order(order, price, datetime.now())
            sample_portfolio.positions[symbol].stop_loss = stop
        
        # Calculate portfolio heat
        heat = sample_portfolio.calculate_portfolio_heat()
        
        # Each position risks 5%, so total heat should be ~15%
        assert heat == pytest.approx(0.15, rel=0.1)
    
    def test_correlation_risk(self, sample_portfolio, sample_ohlcv_data):
        """Test correlation-based risk assessment."""
        # Create correlated price data
        data = {}
        base_prices = sample_ohlcv_data['close']
        
        # Highly correlated stocks
        data['AAPL'] = base_prices * 1.0
        data['GOOGL'] = base_prices * 1.2 + np.random.randn(len(base_prices)) * 0.5
        data['MSFT'] = base_prices * 0.9 + np.random.randn(len(base_prices)) * 0.3
        
        correlation_matrix = sample_portfolio.calculate_correlation_risk(data)
        
        # Should show high correlation
        assert correlation_matrix.loc['AAPL', 'GOOGL'] > 0.8
        assert correlation_matrix.loc['AAPL', 'MSFT'] > 0.8
    
    def test_max_drawdown_tracking(self, sample_portfolio):
        """Test maximum drawdown tracking."""
        # Simulate portfolio value changes
        values = [100000, 110000, 105000, 95000, 100000, 90000, 95000]
        timestamps = pd.date_range(start='2023-01-01', periods=len(values), freq='D')
        
        for ts, value in zip(timestamps, values):
            snapshot = PortfolioSnapshot(
                timestamp=ts,
                cash=value * 0.2,  # 20% cash
                positions_value=value * 0.8,  # 80% invested
                total_value=value,
                unrealized_pnl=0,
                realized_pnl=0
            )
            sample_portfolio.equity_curve.append(snapshot)
        
        max_dd = sample_portfolio.calculate_max_drawdown()
        
        # Max drawdown from 110k to 90k = 18.18%
        assert max_dd == pytest.approx(0.1818, rel=0.01)


class TestPerformanceTracking:
    """Test performance metrics and tracking."""
    
    def test_trade_statistics(self, sample_portfolio):
        """Test trade statistics calculation."""
        # Make several trades
        trades = [
            ("AAPL", 100, 150.0, 160.0),   # Win
            ("GOOGL", 50, 200.0, 190.0),    # Loss
            ("MSFT", 75, 180.0, 185.0),     # Win
            ("AMZN", 40, 250.0, 245.0),     # Loss
            ("TSLA", 60, 300.0, 320.0)      # Win
        ]
        
        for symbol, qty, buy_price, sell_price in trades:
            # Buy
            buy_order = sample_portfolio.place_order(symbol, qty, "BUY")
            sample_portfolio.execute_order(buy_order, buy_price, datetime.now())
            
            # Sell
            sell_order = sample_portfolio.place_order(symbol, qty, "SELL")
            sample_portfolio.execute_order(sell_order, sell_price, datetime.now())
        
        stats = sample_portfolio.get_trade_statistics()
        
        assert stats['total_trades'] == 10  # 5 buys + 5 sells
        assert stats['winning_trades'] == 3
        assert stats['losing_trades'] == 2
        assert stats['win_rate'] == 0.6
        assert stats['avg_win'] > 0
        assert stats['avg_loss'] < 0
        assert stats['profit_factor'] > 1  # Total wins > total losses
    
    def test_sharpe_ratio(self, sample_portfolio):
        """Test Sharpe ratio calculation."""
        # Create daily returns
        returns = pd.Series([0.01, -0.005, 0.008, -0.002, 0.015, -0.003, 0.007])
        
        sharpe = sample_portfolio.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        
        # Should be positive with these returns
        assert sharpe > 0
    
    def test_sortino_ratio(self, sample_portfolio):
        """Test Sortino ratio calculation."""
        # Create returns with different upside/downside volatility
        returns = pd.Series([0.01, -0.005, 0.008, -0.002, 0.015, -0.001, 0.007])
        
        sortino = sample_portfolio.calculate_sortino_ratio(returns, risk_free_rate=0.02)
        
        # Sortino should be higher than Sharpe for same returns
        sharpe = sample_portfolio.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        assert sortino > sharpe


class TestPortfolioSnapshot:
    """Test portfolio snapshot functionality."""
    
    def test_snapshot_creation(self, sample_portfolio):
        """Test creating portfolio snapshots."""
        # Make some trades
        order = sample_portfolio.place_order("AAPL", 100, "BUY")
        sample_portfolio.execute_order(order, 150.0, datetime.now())
        
        # Update price
        sample_portfolio.update_prices({"AAPL": 155.0})
        
        # Take snapshot
        snapshot = sample_portfolio.take_snapshot(datetime.now())
        
        assert isinstance(snapshot, PortfolioSnapshot)
        assert snapshot.cash == sample_portfolio.cash
        assert snapshot.positions_value == 100 * 155.0
        assert snapshot.total_value == sample_portfolio.current_value()
        assert snapshot.unrealized_pnl == 100 * (155.0 - 150.0)
    
    def test_equity_curve_generation(self, sample_portfolio, sample_ohlcv_data):
        """Test equity curve generation from snapshots."""
        dates = sample_ohlcv_data.index[:30]
        
        # Simulate trading over time
        for i, date in enumerate(dates):
            if i % 5 == 0 and i > 0:  # Trade every 5 days
                # Buy
                order = sample_portfolio.place_order("AAPL", 10, "BUY")
                sample_portfolio.execute_order(
                    order, sample_ohlcv_data.loc[date, 'close'], date
                )
            
            # Update prices and take snapshot
            sample_portfolio.update_prices({"AAPL": sample_ohlcv_data.loc[date, 'close']})
            sample_portfolio.take_snapshot(date)
        
        # Generate equity curve
        equity_df = sample_portfolio.get_equity_curve()
        
        assert len(equity_df) == 30
        assert 'total_value' in equity_df.columns
        assert 'cash' in equity_df.columns
        assert 'positions_value' in equity_df.columns
        assert equity_df.index.equals(dates)


class TestPortfolioConstraints:
    """Test portfolio constraints and limits."""
    
    def test_max_positions_limit(self, sample_portfolio):
        """Test maximum positions constraint."""
        sample_portfolio.max_positions = 3
        
        # Open 3 positions
        for i, symbol in enumerate(['AAPL', 'GOOGL', 'MSFT']):
            order = sample_portfolio.place_order(symbol, 100, "BUY")
            sample_portfolio.execute_order(order, 100.0 + i * 10, datetime.now())
        
        # Try to open 4th position
        order = sample_portfolio.place_order("AMZN", 100, "BUY")
        success = sample_portfolio.execute_order(order, 200.0, datetime.now())
        
        assert success is False
        assert order.rejection_reason == "Maximum positions limit reached"
    
    def test_sector_concentration_limit(self, sample_portfolio):
        """Test sector concentration limits."""
        sample_portfolio.max_sector_exposure = 0.4  # 40% max per sector
        sample_portfolio.sector_map = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'JPM': 'Finance'
        }
        
        # Buy tech stocks up to limit
        tech_value = sample_portfolio.cash * 0.35
        shares = int(tech_value / 150.0)
        order = sample_portfolio.place_order("AAPL", shares, "BUY")
        sample_portfolio.execute_order(order, 150.0, datetime.now())
        
        # Try to exceed sector limit
        order2 = sample_portfolio.place_order("MSFT", 100, "BUY")
        success = sample_portfolio.execute_order(order2, 180.0, datetime.now())
        
        assert success is False
        assert "Sector concentration limit" in order2.rejection_reason
    
    def test_liquidity_constraints(self, sample_portfolio, sample_ohlcv_data):
        """Test liquidity-based position limits."""
        sample_portfolio.max_volume_pct = 0.1  # Max 10% of daily volume
        
        # Try to buy more than 10% of daily volume
        daily_volume = sample_ohlcv_data.iloc[-1]['volume']
        max_shares = int(daily_volume * 0.1)
        excessive_shares = max_shares * 2
        
        order = sample_portfolio.place_order("AAPL", excessive_shares, "BUY")
        order.market_data = sample_ohlcv_data.iloc[-1]
        
        success = sample_portfolio.execute_order(
            order, sample_ohlcv_data.iloc[-1]['close'], datetime.now()
        )
        
        assert success is False
        assert "Exceeds liquidity limit" in order.rejection_reason


class TestCommissionAndSlippage:
    """Test commission and slippage calculations."""
    
    def test_fixed_commission(self, sample_portfolio):
        """Test fixed commission calculation."""
        commission = sample_portfolio.calculate_commission(10000)
        expected = max(10000 * 0.001, sample_portfolio.min_commission)
        assert commission == expected
    
    def test_tiered_commission(self):
        """Test tiered commission structure."""
        portfolio = Portfolio(
            commission_structure="tiered",
            commission_tiers=[
                (10000, 0.002),
                (50000, 0.0015),
                (100000, 0.001)
            ]
        )
        
        # Small trade
        assert portfolio.calculate_commission(5000) == 10.0  # 0.2%
        
        # Medium trade
        assert portfolio.calculate_commission(30000) == 45.0  # 0.15%
        
        # Large trade
        assert portfolio.calculate_commission(150000) == 150.0  # 0.1%
    
    def test_slippage_calculation(self, sample_portfolio):
        """Test slippage calculation."""
        # Buy order - price should be worse (higher)
        buy_slippage = sample_portfolio.calculate_slippage(100.0, "BUY")
        assert buy_slippage > 100.0
        assert buy_slippage == pytest.approx(100.05)  # 0.05% slippage
        
        # Sell order - price should be worse (lower)
        sell_slippage = sample_portfolio.calculate_slippage(100.0, "SELL")
        assert sell_slippage < 100.0
        assert sell_slippage == pytest.approx(99.95)  # 0.05% slippage
    
    def test_market_impact_model(self):
        """Test market impact slippage model."""
        portfolio = Portfolio(
            slippage_model="market_impact",
            impact_coefficient=0.1
        )
        
        # Larger orders should have more slippage
        small_impact = portfolio.calculate_slippage(100.0, "BUY", quantity=100)
        large_impact = portfolio.calculate_slippage(100.0, "BUY", quantity=10000)
        
        assert large_impact > small_impact


if __name__ == "__main__":
    pytest.main([__file__, "-v"])