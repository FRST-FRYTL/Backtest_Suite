"""Portfolio management for backtesting."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

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
        max_positions: int = 10
    ):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
            commission_rate: Commission rate (e.g., 0.001 = 0.1%)
            slippage_rate: Slippage rate (e.g., 0.0005 = 0.05%)
            max_positions: Maximum number of concurrent positions
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_positions = max_positions
        
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.orders: List[Order] = []
        self.history: List[PortfolioSnapshot] = []
        
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_trades = 0
        
    def current_value(self) -> float:
        """Calculate current portfolio value."""
        positions_value = sum(
            pos.market_value() for pos in self.positions.values()
        )
        return self.cash + positions_value
        
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
        
    def place_order(
        self,
        symbol: str,
        quantity: int,
        direction: str,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
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
            
        Returns:
            Order object
        """
        order_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            quantity=quantity,
            direction=direction,
            order_type=order_type,
            created_time=datetime.now(),
            price=price,
            stop_price=stop_price
        )
        
        self.orders.append(order)
        order.status = OrderStatus.SUBMITTED
        
        return order
        
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
            
        # Calculate slippage
        if order.order_type == OrderType.MARKET:
            if order.is_buy():
                fill_price *= (1 + self.slippage_rate)
            else:
                fill_price *= (1 - self.slippage_rate)
                
        # Calculate commission
        commission = order.quantity * fill_price * self.commission_rate
        
        # Check if we have enough cash for buys
        if order.is_buy():
            required_cash = order.quantity * fill_price + commission
            if required_cash > self.cash:
                order.reject("Insufficient funds")
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
        
        # Move to closed positions if position is closed
        if not position.is_open():
            self.closed_positions.append(position)
            del self.positions[symbol]
            
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