"""Portfolio management for backtesting."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np

from .position import Position
from .order import Order, OrderType, OrderStatus


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state."""
    
    timestamp: datetime
    cash: float
    positions_value: float
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    commission: float
    positions: Dict[str, dict]


class Portfolio:
    """Portfolio tracking and management."""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        max_positions: int = 10,
        min_commission: float = 0.0,
        commission_structure: Optional[Dict] = None,
        slippage_model: Optional[str] = None
    ):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
            commission_rate: Commission rate (e.g., 0.001 = 0.1%)
            slippage_rate: Slippage rate (e.g., 0.0005 = 0.05%)
            max_positions: Maximum number of concurrent positions
        """
        # Validate inputs
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if commission_rate < 0:
            raise ValueError("Commission rate cannot be negative")
        if slippage_rate < 0 or slippage_rate >= 1:
            raise ValueError("Slippage rate must be between 0 and 1")
        if max_positions <= 0:
            raise ValueError("Max positions must be positive")
            
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_positions = max_positions
        self.min_commission = min_commission
        self.commission_structure = commission_structure or {}
        self.slippage_model = slippage_model or 'linear'
        
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.orders: List[Order] = []
        self.trades: List[dict] = []  # Trade history
        self.history: List[PortfolioSnapshot] = []
        
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_trades = 0
        
        # Short selling settings
        self.allow_short = False
        
    def current_value(self) -> float:
        """Calculate current portfolio value."""
        positions_value = sum(
            pos.market_value() for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    def get_total_value(self, data: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            data: Optional DataFrame with current prices
            
        Returns:
            Total portfolio value
        """
        if data is not None:
            # Update prices from data
            prices = {symbol: data[symbol].iloc[-1] for symbol in data.columns}
            self.update_prices(prices)
        
        return self.current_value()
    
    def get_position_weight(self, symbol: str) -> float:
        """
        Get position weight as percentage of portfolio.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            Position weight (0-1)
        """
        total_value = self.current_value()
        if total_value == 0:
            return 0.0
            
        if symbol not in self.positions:
            return 0.0
            
        position_value = self.positions[symbol].market_value()
        return position_value / total_value
        
    def unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L."""
        return sum(
            pos.unrealized_pnl() for pos in self.positions.values()
        )
        
    def realized_pnl(self) -> float:
        """Calculate total realized P&L."""
        return sum(
            pos.realized_pnl for pos in self.closed_positions
        )
        
    def total_pnl(self) -> float:
        """Calculate total P&L."""
        return self.current_value() - self.initial_capital
        
    def return_pct(self) -> float:
        """Calculate total return percentage."""
        return ((self.current_value() - self.initial_capital) / 
                self.initial_capital * 100)
                
    def position_count(self) -> int:
        """Get number of open positions."""
        return sum(1 for pos in self.positions.values() if pos.is_open())
        
    def can_open_position(self) -> bool:
        """Check if new position can be opened."""
        return self.position_count() < self.max_positions
        
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        risk_amount: Optional[float] = None,
        position_pct: float = 0.1
    ) -> int:
        """
        Calculate position size.
        
        Args:
            symbol: Symbol to trade
            price: Current price
            risk_amount: Fixed risk amount per trade
            position_pct: Percentage of portfolio for position
            
        Returns:
            Number of shares
        """
        if risk_amount:
            # Risk-based sizing
            shares = int(risk_amount / price)
        else:
            # Percentage-based sizing
            position_value = self.current_value() * position_pct
            shares = int(position_value / price)
            
        # Ensure we have enough cash
        required_cash = shares * price * (1 + self.commission_rate)
        if required_cash > self.cash:
            shares = int(self.cash / (price * (1 + self.commission_rate)))
            
        return max(0, shares)
        
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        risk_amount: Optional[float] = None,
        position_pct: float = 0.1,
        sizing_method: str = 'percentage',
        stop_loss_pct: Optional[float] = None,
        max_position_pct: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> int:
        """
        Enhanced position sizing with multiple methods.
        
        Args:
            symbol: Symbol to trade
            price: Current price
            risk_amount: Fixed risk amount per trade
            position_pct: Percentage of portfolio for position
            sizing_method: 'percentage', 'risk', 'kelly'
            stop_loss_pct: Stop loss percentage for risk calculation
            max_position_pct: Maximum position percentage
            
        Returns:
            Number of shares
        """
        if sizing_method == 'kelly':
            # Kelly criterion sizing - f = p - q/b
            win_rate = win_rate or 0.6  # Default assumption
            avg_win = avg_win or 0.1   # 10% average win
            avg_loss = avg_loss or 0.05 # 5% average loss
            kelly_pct = win_rate - (1 - win_rate) / (avg_win / avg_loss)
            shares = int(self.cash * kelly_pct / price)
        elif sizing_method == 'risk' and stop_loss_pct:
            # Risk-based sizing with stop loss
            risk_per_share = price * stop_loss_pct
            max_risk = self.current_value() * 0.02  # 2% max risk
            shares = int(max_risk / risk_per_share)
        elif risk_amount:
            # Fixed risk amount
            if stop_loss_pct:
                # Calculate shares based on risk amount and stop loss
                risk_per_share = price * stop_loss_pct
                shares = int(risk_amount / risk_per_share)
            else:
                shares = int(risk_amount / price)
        else:
            # Percentage-based sizing
            position_value = self.current_value() * position_pct
            shares = int(position_value / price)
            
        # Apply maximum position limit
        if max_position_pct:
            max_position_value = self.current_value() * max_position_pct
            max_shares = int(max_position_value / price)
            shares = min(shares, max_shares)
            
        # Ensure we have enough cash
        required_cash = shares * price * (1 + self.commission_rate)
        if required_cash > self.cash:
            shares = int(self.cash / (price * (1 + self.commission_rate)))
            
        return max(0, shares)
        
    def place_order(
        self,
        symbol: str,
        quantity: int,
        direction: str,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        limit_price: Optional[float] = None
    ) -> Order:
        """
        Place an order.
        
        Args:
            symbol: Symbol to trade
            quantity: Number of shares
            direction: 'BUY' or 'SELL'
            order_type: Type of order
            price: Limit price
            stop_price: Stop price
            limit_price: Limit price (alias for price)
            
        Returns:
            Order object
        """
        # Validate inputs
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if direction not in ['BUY', 'SELL']:
            raise ValueError("Direction must be 'BUY' or 'SELL'")
        if order_type == OrderType.LIMIT and not (price or limit_price):
            raise ValueError("Limit orders require a price")
        if order_type == OrderType.STOP and not stop_price:
            raise ValueError("Stop orders require a stop price")
            
        # Use limit_price if provided, otherwise use price
        effective_price = limit_price or price
        
        order_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            quantity=quantity,
            direction=direction,
            order_type=order_type,
            created_time=datetime.now(),
            price=effective_price,
            stop_price=stop_price,
            limit_price=limit_price
        )
        
        self.orders.append(order)
        order.status = OrderStatus.SUBMITTED
        
        return order
        
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if order was cancelled successfully
        """
        for order in self.orders:
            if order.order_id == order_id and order.is_active():
                order.cancel()
                return True
        return False
        
    def execute_order(
        self,
        order: Order,
        fill_price: float,
        timestamp: datetime
    ) -> bool:
        """
        Execute an order.
        
        Args:
            order: Order to execute
            fill_price: Execution price
            timestamp: Execution timestamp
            
        Returns:
            True if successful
        """
        if not order.is_active():
            return False
        
        # Check order type constraints
        if order.order_type == OrderType.LIMIT:
            if order.is_buy() and fill_price > order.price:
                return False  # Can't buy above limit price
            elif order.is_sell() and fill_price < order.price:
                return False  # Can't sell below limit price
                
        elif order.order_type == OrderType.STOP:
            if order.is_buy() and fill_price < order.stop_price:
                return False  # Stop buy only executes at or above stop price
            elif order.is_sell() and fill_price > order.stop_price:
                return False  # Stop sell only executes at or below stop price
                
        # Calculate slippage (only for market orders)
        if order.order_type == OrderType.MARKET:
            if order.is_buy():
                fill_price *= (1 + self.slippage_rate)
            else:
                fill_price *= (1 - self.slippage_rate)
                
        # Calculate commission
        trade_value = order.quantity * fill_price
        commission = self.calculate_commission(trade_value)
        
        # Check if we have enough cash for buys
        if order.is_buy():
            required_cash = order.quantity * fill_price + commission
            if required_cash > self.cash:
                order.reject("Insufficient cash")
                return False
                
        # Check if we have enough shares for sells (unless short selling is allowed)
        if order.is_sell():
            if order.symbol not in self.positions:
                if not self.allow_short:
                    order.reject("Insufficient shares")
                    return False
            else:
                position = self.positions[order.symbol]
                if position.quantity < order.quantity and not self.allow_short:
                    order.reject("Insufficient shares")
                    return False
                
        # Update order
        order.fill(order.quantity, fill_price, commission)
        
        # Update portfolio
        self._process_fill(order, fill_price, commission, timestamp)
        
        return True
        
    def _process_fill(
        self,
        order: Order,
        fill_price: float,
        commission: float,
        timestamp: datetime
    ) -> None:
        """Process order fill and update positions."""
        symbol = order.symbol
        quantity = order.quantity if order.is_buy() else -order.quantity
        
        # Update or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
            
        position = self.positions[symbol]
        position.add_trade(timestamp, quantity, fill_price, commission)
        
        # Update cash
        if order.is_buy():
            self.cash -= (order.quantity * fill_price + commission)
        else:
            self.cash += (order.quantity * fill_price - commission)
            
        # Update totals
        self.total_commission += commission
        self.total_trades += 1
        
        # Add to closed positions if position is closed but keep in positions
        if not position.is_open():
            self.closed_positions.append(position)
            
    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update position prices.
        
        Args:
            prices: Dictionary of symbol -> price
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])
                
    def check_stops(self) -> List[Order]:
        """
        Check stop conditions and generate exit orders.
        
        Returns:
            List of stop orders to execute
        """
        stop_orders = []
        
        for symbol, position in self.positions.items():
            stop_hit = position.check_stops()
            
            if stop_hit:
                # Create market order to close position
                order = self.place_order(
                    symbol=symbol,
                    quantity=abs(position.quantity),
                    direction='SELL' if position.is_long() else 'BUY',
                    order_type=OrderType.MARKET
                )
                order.notes = f"Stop hit: {stop_hit}"
                stop_orders.append(order)
                
        return stop_orders
        
    def take_snapshot(self, timestamp: datetime) -> PortfolioSnapshot:
        """Take a snapshot of current portfolio state."""
        positions_data = {
            symbol: pos.to_dict() 
            for symbol, pos in self.positions.items()
        }
        
        positions_value = sum(
            pos.market_value() for pos in self.positions.values()
        )
        
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions_value=positions_value,
            total_value=self.cash + positions_value,
            unrealized_pnl=self.unrealized_pnl(),
            realized_pnl=self.realized_pnl(),
            commission=self.total_commission,
            positions=positions_data
        )
        
        self.history.append(snapshot)
        return snapshot
        
    def get_performance_summary(self) -> Dict:
        """Get portfolio performance summary."""
        if not self.history:
            return {}
            
        equity_curve = [s.total_value for s in self.history]
        returns = pd.Series(equity_curve).pct_change().dropna()
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': self.current_value(),
            'total_return': self.return_pct(),
            'total_pnl': self.total_pnl(),
            'unrealized_pnl': self.unrealized_pnl(),
            'realized_pnl': self.realized_pnl(),
            'total_commission': self.total_commission,
            'total_trades': self.total_trades,
            'winning_trades': sum(1 for p in self.closed_positions if p.realized_pnl > 0),
            'losing_trades': sum(1 for p in self.closed_positions if p.realized_pnl <= 0),
            'avg_win': np.mean([p.realized_pnl for p in self.closed_positions if p.realized_pnl > 0]) if self.closed_positions else 0,
            'avg_loss': np.mean([p.realized_pnl for p in self.closed_positions if p.realized_pnl <= 0]) if self.closed_positions else 0,
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        }
        
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown percentage."""
        if not equity_curve:
            return 0.0
            
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            max_dd = max(max_dd, dd)
            
        return max_dd

    # Missing methods required by tests
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L (alias for unrealized_pnl)."""
        return self.unrealized_pnl()

    def get_realized_pnl(self) -> float:
        """Get total realized P&L (alias for realized_pnl)."""
        return self.realized_pnl()

    def calculate_commission(self, trade_value: float) -> float:
        """Calculate commission for a trade."""
        if self.commission_structure:
            # Tiered commission structure
            commission = 0.0
            remaining_value = trade_value
            
            for tier in sorted(self.commission_structure.keys()):
                if remaining_value <= 0:
                    break
                    
                tier_value = min(remaining_value, tier)
                commission += tier_value * self.commission_structure[tier]
                remaining_value -= tier_value
                
            return max(commission, self.min_commission)
        else:
            # Simple percentage commission
            commission = trade_value * self.commission_rate
            return max(commission, self.min_commission)

    def get_returns(self, period: str = 'daily') -> Dict:
        """Get portfolio returns summary."""
        current_total = self.current_value()
        total_return = current_total - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'realized_return': self.realized_pnl(),
            'unrealized_return': self.unrealized_pnl(),
            'initial_capital': self.initial_capital,
            'current_value': current_total
        }

    def calculate_portfolio_heat(self) -> float:
        """Calculate portfolio heat (percentage of capital at risk)."""
        total_risk = 0.0
        for position in self.positions.values():
            # Estimate risk as potential loss to stop loss
            if position.stop_loss:
                risk_per_share = abs(position.avg_price - position.stop_loss)
                total_risk += risk_per_share * abs(position.quantity)
        
        return total_risk / self.current_value()

    def calculate_correlation_risk(self) -> float:
        """Calculate correlation risk between positions."""
        # Simplified correlation risk calculation
        return len(self.positions) * 0.1  # Placeholder

    def get_trade_statistics(self) -> Dict:
        """Get detailed trade statistics."""
        if not self.closed_positions:
            return {}
        
        winning_trades = [p for p in self.closed_positions if p.realized_pnl > 0]
        losing_trades = [p for p in self.closed_positions if p.realized_pnl <= 0]
        
        return {
            'total_trades': len(self.closed_positions),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.closed_positions) * 100,
            'avg_win': np.mean([p.realized_pnl for p in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([p.realized_pnl for p in losing_trades]) if losing_trades else 0,
            'profit_factor': (sum(p.realized_pnl for p in winning_trades) / 
                            abs(sum(p.realized_pnl for p in losing_trades))) if losing_trades else 0,
            'total_pnl': sum(p.realized_pnl for p in self.closed_positions)
        }

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if not self.history:
            return 0.0
        
        equity_curve = [s.total_value for s in self.history]
        returns = pd.Series(equity_curve).pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        if not self.history:
            return 0.0
        
        equity_curve = [s.total_value for s in self.history]
        returns = pd.Series(equity_curve).pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = downside_returns.std()
        return excess_returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0

    def get_equity_curve(self) -> pd.Series:
        """Get portfolio equity curve."""
        if not self.history:
            return pd.Series()
        
        values = [s.total_value for s in self.history]
        timestamps = [s.timestamp for s in self.history]
        
        return pd.Series(values, index=timestamps)

    def calculate_slippage(self, symbol: str, quantity: int, price: float) -> float:
        """Calculate slippage for a trade."""
        return quantity * price * self.slippage_rate