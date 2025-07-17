"""
Trade Fixtures for Testing

Provides comprehensive trade-related fixtures including trades,
positions, orders, and execution data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TradeStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


def create_trade_data(
    n_trades: int = 50,
    start_date: str = '2023-01-01',
    end_date: str = '2023-12-31',
    symbols: List[str] = ['AAPL', 'GOOGL', 'MSFT'],
    win_rate: float = 0.55,
    avg_win_pnl: float = 500,
    avg_loss_pnl: float = -300,
    include_metadata: bool = True,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Create comprehensive trade data.
    
    Args:
        n_trades: Number of trades
        start_date: Start date for trades
        end_date: End date for trades
        symbols: List of symbols to trade
        win_rate: Percentage of winning trades
        avg_win_pnl: Average profit for winning trades
        avg_loss_pnl: Average loss for losing trades
        include_metadata: Include additional trade metadata
        seed: Random seed
        
    Returns:
        DataFrame with trade data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate trade dates
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    # Entry dates
    entry_dates = []
    for _ in range(n_trades):
        entry_idx = np.random.randint(0, n_days - 5)  # Leave room for trade duration
        entry_dates.append(date_range[entry_idx])
    
    entry_dates = sorted(entry_dates)
    
    # Trade durations (1-20 days)
    durations = np.random.randint(1, 21, n_trades)
    exit_dates = [entry + timedelta(days=int(dur)) for entry, dur in zip(entry_dates, durations)]
    
    # Generate trade outcomes
    n_wins = int(n_trades * win_rate)
    is_winner = np.array([True] * n_wins + [False] * (n_trades - n_wins))
    np.random.shuffle(is_winner)
    
    # Generate P&L
    pnl = np.where(
        is_winner,
        np.random.normal(avg_win_pnl, avg_win_pnl * 0.3, n_trades),
        np.random.normal(avg_loss_pnl, abs(avg_loss_pnl) * 0.3, n_trades)
    )
    
    # Generate prices
    entry_prices = np.random.uniform(50, 500, n_trades)
    price_changes = pnl / np.random.uniform(100, 1000, n_trades)  # Based on position size
    exit_prices = entry_prices * (1 + price_changes)
    
    # Generate position sizes
    position_sizes = np.random.uniform(100, 1000, n_trades)
    
    # Basic trade data
    trades_df = pd.DataFrame({
        'trade_id': range(1, n_trades + 1),
        'symbol': np.random.choice(symbols, n_trades),
        'side': np.random.choice(['long', 'short'], n_trades),
        'entry_date': entry_dates,
        'exit_date': exit_dates,
        'entry_price': entry_prices,
        'exit_price': exit_prices,
        'position_size': position_sizes,
        'pnl': pnl,
        'pnl_pct': (exit_prices / entry_prices - 1) * 100,
        'duration_days': durations,
        'is_winner': is_winner
    })
    
    # Add metadata if requested
    if include_metadata:
        # Stop loss and take profit levels
        trades_df['stop_loss'] = np.where(
            trades_df['side'] == 'long',
            trades_df['entry_price'] * (1 - np.random.uniform(0.01, 0.05, n_trades)),
            trades_df['entry_price'] * (1 + np.random.uniform(0.01, 0.05, n_trades))
        )
        
        trades_df['take_profit'] = np.where(
            trades_df['side'] == 'long',
            trades_df['entry_price'] * (1 + np.random.uniform(0.02, 0.10, n_trades)),
            trades_df['entry_price'] * (1 - np.random.uniform(0.02, 0.10, n_trades))
        )
        
        # Exit reasons
        exit_reasons = []
        for idx, row in trades_df.iterrows():
            if row['is_winner']:
                if np.random.random() > 0.3:
                    exit_reasons.append('take_profit')
                else:
                    exit_reasons.append('signal')
            else:
                if np.random.random() > 0.4:
                    exit_reasons.append('stop_loss')
                else:
                    exit_reasons.append('signal')
        
        trades_df['exit_reason'] = exit_reasons
        
        # Commission and slippage
        trades_df['commission'] = trades_df['position_size'] * 0.001  # 0.1% commission
        trades_df['slippage'] = np.abs(trades_df['entry_price'] * np.random.normal(0, 0.0005, n_trades))
        
        # Net P&L
        trades_df['net_pnl'] = trades_df['pnl'] - trades_df['commission'] - trades_df['slippage']
        
        # Strategy name
        strategies = ['momentum', 'mean_reversion', 'breakout', 'pairs']
        trades_df['strategy'] = np.random.choice(strategies, n_trades)
        
        # Maximum adverse/favorable excursion
        trades_df['mae'] = -np.abs(np.random.normal(0, abs(avg_loss_pnl) * 0.5, n_trades))
        trades_df['mfe'] = np.abs(np.random.normal(0, avg_win_pnl * 0.5, n_trades))
    
    return trades_df


def create_trade_history(
    trade_id: int,
    symbol: str,
    entry_date: str,
    n_updates: int = 10,
    initial_size: float = 1000,
    initial_price: float = 100,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Create detailed trade history with position updates.
    
    Args:
        trade_id: Trade identifier
        symbol: Trading symbol
        entry_date: Trade entry date
        n_updates: Number of position updates
        initial_size: Initial position size
        initial_price: Initial entry price
        seed: Random seed
        
    Returns:
        DataFrame with trade history
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate timestamps
    start_time = pd.Timestamp(entry_date)
    timestamps = [start_time]
    
    for i in range(1, n_updates):
        # Random time intervals (minutes to hours)
        time_delta = timedelta(minutes=np.random.randint(5, 240))
        timestamps.append(timestamps[-1] + time_delta)
    
    # Generate position updates
    actions = ['entry'] + np.random.choice(
        ['add', 'reduce', 'partial_close'], 
        n_updates - 2
    ).tolist() + ['exit']
    
    # Track position size
    current_size = initial_size
    sizes = [current_size]
    
    for action in actions[1:-1]:
        if action == 'add':
            add_size = np.random.uniform(100, 500)
            current_size += add_size
        elif action == 'reduce':
            reduce_size = min(current_size * 0.5, np.random.uniform(100, 300))
            current_size -= reduce_size
        elif action == 'partial_close':
            current_size *= 0.5
        
        sizes.append(max(0, current_size))
    
    sizes.append(0)  # Final exit
    
    # Generate prices with realistic movement
    prices = [initial_price]
    for i in range(1, n_updates):
        price_change = np.random.normal(0, 0.005)
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    # Calculate P&L for each update
    avg_entry_price = initial_price
    pnl_values = []
    realized_pnl = 0
    
    for i, (action, size, price) in enumerate(zip(actions, sizes, prices)):
        if action == 'entry':
            pnl_values.append(0)
        elif action in ['reduce', 'partial_close', 'exit']:
            # Realize some P&L
            closed_size = sizes[i-1] - size
            realized = closed_size * (price - avg_entry_price)
            realized_pnl += realized
            pnl_values.append(realized_pnl)
        else:
            # Unrealized P&L
            unrealized = size * (price - avg_entry_price)
            pnl_values.append(realized_pnl + unrealized)
    
    # Create history DataFrame
    history_df = pd.DataFrame({
        'timestamp': timestamps,
        'trade_id': trade_id,
        'symbol': symbol,
        'action': actions,
        'position_size': sizes,
        'price': prices,
        'pnl': pnl_values,
        'realized_pnl': [realized_pnl if a in ['reduce', 'partial_close', 'exit'] else 0 
                         for a in actions]
    })
    
    return history_df


def create_position_data(
    symbols: List[str] = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
    date: str = '2023-12-29',
    include_closed: bool = False,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Create current position data.
    
    Args:
        symbols: List of symbols
        date: Current date
        include_closed: Include closed positions
        seed: Random seed
        
    Returns:
        DataFrame with position data
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_positions = len(symbols)
    
    # Generate position data
    positions = []
    
    for i, symbol in enumerate(symbols):
        # Some positions might be closed
        is_open = True if not include_closed else np.random.random() > 0.3
        
        if is_open or include_closed:
            entry_price = np.random.uniform(50, 500)
            current_price = entry_price * (1 + np.random.normal(0.01, 0.05))
            position_size = np.random.uniform(100, 2000)
            
            # Calculate metrics
            market_value = position_size * current_price
            cost_basis = position_size * entry_price
            unrealized_pnl = market_value - cost_basis
            unrealized_pct = (current_price / entry_price - 1) * 100
            
            # Entry date (random within last 60 days)
            entry_date = pd.Timestamp(date) - timedelta(days=np.random.randint(1, 60))
            
            position = {
                'symbol': symbol,
                'position_size': position_size if is_open else 0,
                'entry_price': entry_price,
                'current_price': current_price,
                'market_value': market_value if is_open else 0,
                'cost_basis': cost_basis,
                'unrealized_pnl': unrealized_pnl if is_open else 0,
                'unrealized_pct': unrealized_pct if is_open else 0,
                'realized_pnl': 0 if is_open else np.random.normal(100, 500),
                'entry_date': entry_date,
                'days_held': (pd.Timestamp(date) - entry_date).days,
                'status': 'open' if is_open else 'closed',
                'side': np.random.choice(['long', 'short'])
            }
            
            positions.append(position)
    
    positions_df = pd.DataFrame(positions)
    
    # Add portfolio summary row
    if len(positions_df) > 0 and positions_df['status'].eq('open').any():
        open_positions = positions_df[positions_df['status'] == 'open']
        
        portfolio_summary = {
            'symbol': 'PORTFOLIO_TOTAL',
            'position_size': 0,
            'entry_price': 0,
            'current_price': 0,
            'market_value': open_positions['market_value'].sum(),
            'cost_basis': open_positions['cost_basis'].sum(),
            'unrealized_pnl': open_positions['unrealized_pnl'].sum(),
            'unrealized_pct': (open_positions['unrealized_pnl'].sum() / 
                              open_positions['cost_basis'].sum() * 100),
            'realized_pnl': positions_df['realized_pnl'].sum(),
            'entry_date': pd.Timestamp(date),
            'days_held': 0,
            'status': 'summary',
            'side': 'portfolio'
        }
        
        positions_df = pd.concat([positions_df, pd.DataFrame([portfolio_summary])], 
                                ignore_index=True)
    
    return positions_df


def create_order_data(
    n_orders: int = 20,
    start_time: str = '2023-12-29 09:30:00',
    symbols: List[str] = ['AAPL', 'GOOGL', 'MSFT'],
    include_cancelled: bool = True,
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Create order execution data.
    
    Args:
        n_orders: Number of orders
        start_time: Start timestamp
        symbols: List of symbols
        include_cancelled: Include cancelled/rejected orders
        seed: Random seed
        
    Returns:
        DataFrame with order data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate order timestamps
    start = pd.Timestamp(start_time)
    timestamps = [start]
    
    for i in range(1, n_orders):
        # Random intervals (seconds to minutes)
        time_delta = timedelta(seconds=np.random.randint(5, 300))
        timestamps.append(timestamps[-1] + time_delta)
    
    # Generate order data
    orders = []
    
    for i, timestamp in enumerate(timestamps):
        order_type = np.random.choice(list(OrderType))
        symbol = np.random.choice(symbols)
        side = np.random.choice(['buy', 'sell'])
        
        # Base price
        base_price = np.random.uniform(50, 500)
        
        # Order details based on type
        if order_type == OrderType.MARKET:
            limit_price = None
            stop_price = None
            fill_price = base_price * (1 + np.random.normal(0, 0.0005))
        
        elif order_type == OrderType.LIMIT:
            limit_price = base_price * (0.995 if side == 'buy' else 1.005)
            stop_price = None
            # Might fill at limit or better
            if np.random.random() > 0.2:
                fill_price = limit_price * (1 - np.random.uniform(0, 0.001) 
                                          if side == 'buy' else 
                                          1 + np.random.uniform(0, 0.001))
            else:
                fill_price = None  # Unfilled
        
        elif order_type == OrderType.STOP:
            limit_price = None
            stop_price = base_price * (1.01 if side == 'sell' else 0.99)
            # Might trigger and fill
            if np.random.random() > 0.3:
                fill_price = stop_price * (1 + np.random.normal(0, 0.001))
            else:
                fill_price = None
        
        else:  # STOP_LIMIT
            stop_price = base_price * (1.01 if side == 'sell' else 0.99)
            limit_price = stop_price * (1.005 if side == 'sell' else 0.995)
            if np.random.random() > 0.4:
                fill_price = limit_price
            else:
                fill_price = None
        
        # Order size
        order_size = np.random.uniform(100, 1000)
        
        # Status
        if fill_price is not None:
            if np.random.random() > 0.1:
                status = OrderStatus.FILLED
                filled_size = order_size
            else:
                status = OrderStatus.PARTIAL
                filled_size = order_size * np.random.uniform(0.2, 0.8)
        else:
            if include_cancelled and np.random.random() > 0.5:
                status = np.random.choice([OrderStatus.CANCELLED, OrderStatus.REJECTED])
            else:
                status = OrderStatus.PENDING
            filled_size = 0
        
        order = {
            'order_id': f'ORD_{i+1:06d}',
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side,
            'order_type': order_type.value,
            'order_size': order_size,
            'limit_price': limit_price,
            'stop_price': stop_price,
            'fill_price': fill_price,
            'filled_size': filled_size,
            'status': status.value,
            'commission': filled_size * 0.001 if filled_size > 0 else 0,
            'remaining_size': order_size - filled_size
        }
        
        orders.append(order)
    
    orders_df = pd.DataFrame(orders)
    
    return orders_df