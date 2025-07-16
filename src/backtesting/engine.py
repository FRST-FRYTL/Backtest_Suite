"""Main backtesting engine implementation."""

import queue
from datetime import datetime
from typing import Dict, List, Optional, Callable
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from .events import Event, MarketEvent, SignalEvent, OrderEvent, FillEvent, EventType
from .portfolio import Portfolio
from .position import Position
from .order import Order, OrderType
from ..strategies.builder import Strategy
from ..strategies.signals import SignalGenerator
from ..reporting import StandardReportGenerator, ReportConfig


class BacktestEngine:
    """Event-driven backtesting engine."""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        max_positions: int = 10,
        generate_report: bool = True,
        report_config: Optional[ReportConfig] = None,
        report_dir: str = "reports"
    ):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting capital
            commission_rate: Commission rate
            slippage_rate: Slippage rate
            max_positions: Maximum concurrent positions
            generate_report: Whether to generate standard report after backtest
            report_config: Configuration for report generation
            report_dir: Directory to save reports
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_positions = max_positions
        self.generate_report = generate_report
        self.report_config = report_config
        self.report_dir = Path(report_dir)
        
        self.events_queue = queue.Queue()
        self.portfolio = None
        self.strategy = None
        self.data = None
        self.signal_generator = SignalGenerator()
        
        self.market_data_index = 0
        self.current_time = None
        self.is_running = False
        
        # Performance tracking
        self.processed_events = 0
        self.generated_signals = 0
        self.executed_orders = 0
        self.start_time = None
        self.end_time = None
        
    def run(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        progress_bar: bool = True
    ) -> Dict:
        """
        Run backtest.
        
        Args:
            data: Market data DataFrame with OHLCV
            strategy: Trading strategy
            start_date: Start date for backtest
            end_date: End date for backtest
            progress_bar: Show progress bar
            
        Returns:
            Backtest results dictionary
        """
        # Initialize
        self._initialize_backtest(data, strategy, start_date, end_date)
        
        # Create progress bar
        pbar = None
        if progress_bar:
            total_bars = len(self.data)
            pbar = tqdm(total=total_bars, desc="Backtesting")
        
        # Track timing
        self.start_time = datetime.now()
            
        # Main event loop
        self.is_running = True
        while self.is_running:
            try:
                # Process events in queue
                while not self.events_queue.empty():
                    event = self.events_queue.get(False)
                    self._process_event(event)
                    
                # Generate next market event
                if self.market_data_index < len(self.data):
                    self._generate_market_event()
                    if pbar:
                        pbar.update(1)
                else:
                    # End of data
                    self.is_running = False
                    
            except queue.Empty:
                continue
                
        if pbar:
            pbar.close()
        
        # Track end time
        self.end_time = datetime.now()
            
        # Generate results
        results = self._generate_results()
        
        # Generate report if enabled
        if self.generate_report:
            self._generate_standard_report(results)
            
        return results
        
    def _initialize_backtest(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> None:
        """Initialize backtest components."""
        # Filter data by date range
        self.data = data.copy()
        if start_date:
            self.data = self.data[self.data.index >= start_date]
        if end_date:
            self.data = self.data[self.data.index <= end_date]
            
        # Reset indices
        self.market_data_index = 0
        
        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_capital=self.initial_capital,
            commission_rate=self.commission_rate,
            slippage_rate=self.slippage_rate,
            max_positions=self.max_positions
        )
        
        # Set strategy
        self.strategy = strategy
        
        # Clear event queue
        self.events_queue = queue.Queue()
        
        # Reset counters
        self.processed_events = 0
        self.generated_signals = 0
        self.executed_orders = 0
        
    def _generate_market_event(self) -> None:
        """Generate market event from data."""
        if self.market_data_index >= len(self.data):
            return
            
        # Get current bar
        bar = self.data.iloc[self.market_data_index]
        self.current_time = self.data.index[self.market_data_index]
        
        # Create market event
        event = MarketEvent(
            timestamp=self.current_time,
            symbol=self.data.attrs.get('symbol', 'UNKNOWN'),
            open=bar['open'] if 'open' in bar else bar['Open'],
            high=bar['high'] if 'high' in bar else bar['High'],
            low=bar['low'] if 'low' in bar else bar['Low'],
            close=bar['close'] if 'close' in bar else bar['Close'],
            volume=bar['volume'] if 'volume' in bar else bar['Volume']
        )
        
        self.events_queue.put(event)
        self.market_data_index += 1
        
    def _process_event(self, event: Event) -> None:
        """Process a single event."""
        self.processed_events += 1
        
        if event.get_type() == EventType.MARKET:
            self._handle_market_event(event)
        elif event.get_type() == EventType.SIGNAL:
            self._handle_signal_event(event)
        elif event.get_type() == EventType.ORDER:
            self._handle_order_event(event)
        elif event.get_type() == EventType.FILL:
            self._handle_fill_event(event)
            
    def _handle_market_event(self, event: MarketEvent) -> None:
        """Handle market data update."""
        # Update portfolio prices
        self.portfolio.update_prices({event.symbol: event.close})
        
        # Check stops
        stop_orders = self.portfolio.check_stops()
        for order in stop_orders:
            self.events_queue.put(OrderEvent(
                timestamp=event.timestamp,
                symbol=order.symbol,
                order_type=order.order_type.value,
                quantity=order.quantity,
                direction=order.direction
            ))
            
        # Generate signals using current data up to this point
        current_data = self.data.iloc[:self.market_data_index]
        
        # Check if we should generate signals
        if self._should_generate_signal(current_data):
            signals = self.signal_generator.generate(self.strategy, current_data)
            
            if not signals.empty:
                latest_signal = signals.iloc[-1]
                
                if latest_signal['entry'] and self.portfolio.can_open_position():
                    # Generate entry signal
                    signal_event = SignalEvent(
                        timestamp=event.timestamp,
                        symbol=event.symbol,
                        signal_type='LONG',  # Simplified - could be from strategy
                        strength=latest_signal.get('signal_strength', 1.0),
                        stop_loss=latest_signal.get('stop_loss'),
                        take_profit=latest_signal.get('take_profit')
                    )
                    self.events_queue.put(signal_event)
                    self.generated_signals += 1
                    
                elif latest_signal['exit'] and event.symbol in self.portfolio.positions:
                    # Generate exit signal
                    signal_event = SignalEvent(
                        timestamp=event.timestamp,
                        symbol=event.symbol,
                        signal_type='EXIT',
                        strength=1.0
                    )
                    self.events_queue.put(signal_event)
                    self.generated_signals += 1
                    
        # Take portfolio snapshot
        self.portfolio.take_snapshot(event.timestamp)
        
    def _should_generate_signal(self, data: pd.DataFrame) -> bool:
        """Check if we have enough data to generate signals."""
        # Need at least some minimum data for indicators
        min_bars = 50  # Adjust based on strategy requirements
        return len(data) >= min_bars
        
    def _handle_signal_event(self, event: SignalEvent) -> None:
        """Handle trading signal."""
        # Generate order from signal
        if event.signal_type == 'LONG':
            # Calculate position size
            current_price = self.portfolio.positions.get(
                event.symbol, 
                type('', (), {'current_price': self.data.iloc[self.market_data_index-1]['close']})()
            ).current_price
            
            quantity = self.portfolio.calculate_position_size(
                event.symbol,
                current_price,
                position_pct=0.1  # 10% of portfolio
            )
            
            if quantity > 0:
                order = self.portfolio.place_order(
                    symbol=event.symbol,
                    quantity=quantity,
                    direction='BUY',
                    order_type=OrderType.MARKET
                )
                
                # Set stops if provided
                if event.symbol not in self.portfolio.positions:
                    self.portfolio.positions[event.symbol] = Position(symbol=event.symbol)
                    
                position = self.portfolio.positions[event.symbol]
                position.stop_loss = event.stop_loss
                position.take_profit = event.take_profit
                
                # Create order event
                order_event = OrderEvent(
                    timestamp=event.timestamp,
                    symbol=event.symbol,
                    order_type='MARKET',
                    quantity=quantity,
                    direction='BUY'
                )
                self.events_queue.put(order_event)
                
        elif event.signal_type == 'EXIT':
            # Close position
            if event.symbol in self.portfolio.positions:
                position = self.portfolio.positions[event.symbol]
                if position.is_open():
                    order = self.portfolio.place_order(
                        symbol=event.symbol,
                        quantity=abs(position.quantity),
                        direction='SELL' if position.is_long() else 'BUY',
                        order_type=OrderType.MARKET
                    )
                    
                    order_event = OrderEvent(
                        timestamp=event.timestamp,
                        symbol=event.symbol,
                        order_type='MARKET',
                        quantity=abs(position.quantity),
                        direction='SELL' if position.is_long() else 'BUY'
                    )
                    self.events_queue.put(order_event)
                    
    def _handle_order_event(self, event: OrderEvent) -> None:
        """Handle order placement."""
        # In a real system, this would send to broker
        # For backtest, we immediately fill at market
        
        # Get current price
        current_bar = self.data.iloc[self.market_data_index-1]
        
        # Simple fill logic - in reality would be more complex
        if event.direction == 'BUY':
            fill_price = current_bar['high'] if 'high' in current_bar else current_bar['High']
        else:
            fill_price = current_bar['low'] if 'low' in current_bar else current_bar['Low']
            
        # Create fill event
        fill_event = FillEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            quantity=event.quantity,
            direction=event.direction,
            fill_price=fill_price,
            commission=event.quantity * fill_price * self.commission_rate,
            slippage=fill_price * self.slippage_rate
        )
        
        self.events_queue.put(fill_event)
        
    def _handle_fill_event(self, event: FillEvent) -> None:
        """Handle order fill."""
        # Find the order (simplified - assumes we can match)
        matching_orders = [
            o for o in self.portfolio.orders 
            if o.symbol == event.symbol and 
            o.direction == event.direction and
            o.is_active()
        ]
        
        if matching_orders:
            order = matching_orders[0]
            
            # Execute the order
            success = self.portfolio.execute_order(
                order,
                event.fill_price,
                event.timestamp
            )
            
            if success:
                self.executed_orders += 1
                
    def _generate_results(self) -> Dict:
        """Generate backtest results."""
        # Get portfolio performance
        performance = self.portfolio.get_performance_summary()
        
        # Create equity curve
        equity_curve = pd.DataFrame([
            {
                'timestamp': s.timestamp,
                'total_value': s.total_value,
                'cash': s.cash,
                'positions_value': s.positions_value,
                'unrealized_pnl': s.unrealized_pnl,
                'realized_pnl': s.realized_pnl
            }
            for s in self.portfolio.history
        ])
        
        if not equity_curve.empty:
            equity_curve.set_index('timestamp', inplace=True)
            
        # Calculate returns for the report
        if not equity_curve.empty:
            equity_series = equity_curve['total_value']
            returns = equity_series.pct_change().dropna()
        else:
            equity_series = pd.Series()
            returns = pd.Series()
            
        # Get trade list with additional fields for reporting
        trades = []
        for position in self.portfolio.closed_positions:
            entry_time = None
            entry_price = None
            
            for i, trade in enumerate(position.trades):
                if trade.trade_type == 'OPEN' and entry_time is None:
                    entry_time = trade.timestamp
                    entry_price = trade.price
                elif trade.trade_type == 'CLOSE' and entry_time is not None:
                    trades.append({
                        'symbol': position.symbol,
                        'entry_time': entry_time,
                        'exit_time': trade.timestamp,
                        'entry_price': entry_price,
                        'exit_price': trade.price,
                        'quantity': abs(trade.quantity),
                        'side': 'long' if trade.quantity > 0 else 'short',
                        'pnl': position.realized_pnl,
                        'duration': (trade.timestamp - entry_time).total_seconds() / 3600,  # hours
                        'commission': trade.commission,
                        'entry_reason': 'signal',  # Could be enhanced
                        'exit_reason': 'signal'    # Could be enhanced
                    })
                    entry_time = None
                    entry_price = None
                
        trades_df = pd.DataFrame(trades)
        
        # Calculate additional metrics for reporting
        if not equity_series.empty and len(trades_df) > 0:
            total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1)
            annual_return = returns.mean() * 252 if len(returns) > 0 else 0
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Calculate drawdown
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Trade metrics
            win_rate = (trades_df['pnl'] > 0).mean() if len(trades_df) > 0 else 0
            profit_factor = (
                trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
                if len(trades_df[trades_df['pnl'] < 0]) > 0 else float('inf')
            )
            
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sharpe_ratio * 1.2,  # Simplified
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(trades_df)
            }
        else:
            metrics = {
                'total_return': 0,
                'annual_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0
            }
        
        # Compile results
        results = {
            'performance': performance,
            'equity_curve': equity_series,  # Use Series for report compatibility
            'trades': trades_df,
            'metrics': metrics,
            'returns': returns,
            'strategy_params': {
                'name': getattr(self.strategy, 'name', 'Unknown Strategy'),
                'initial_capital': self.initial_capital,
                'commission_rate': self.commission_rate,
                'slippage_rate': self.slippage_rate,
                'max_positions': self.max_positions
            },
            'final_portfolio': {
                'cash': self.portfolio.cash,
                'positions': {
                    symbol: pos.to_dict() 
                    for symbol, pos in self.portfolio.positions.items()
                },
                'total_value': self.portfolio.current_value()
            },
            'statistics': {
                'total_events': self.processed_events,
                'signals_generated': self.generated_signals,
                'orders_executed': self.executed_orders,
                'data_points': len(self.data)
            },
            'execution_stats': {
                'total_orders': self.executed_orders,
                'filled_orders': self.executed_orders,
                'rejected_orders': 0,
                'avg_fill_time': 0,  # Instant for backtest
                'avg_slippage': self.slippage_rate,
                'max_slippage': self.slippage_rate * 2,
                'slippage_cost': trades_df['pnl'].sum() * self.slippage_rate if len(trades_df) > 0 else 0,
                'total_commission': trades_df['commission'].sum() if 'commission' in trades_df else 0
            },
            'performance_stats': {
                'total_time': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0,
                'bars_processed': len(self.data),
                'avg_time_per_bar': (
                    (self.end_time - self.start_time).total_seconds() / len(self.data) 
                    if self.start_time and self.end_time and len(self.data) > 0 else 0
                )
            }
        }
        
        return results
        
    def plot_results(self, results: Dict) -> None:
        """Plot backtest results (placeholder for visualization)."""
        # This would be implemented in the visualization module
        pass
    
    def _generate_standard_report(self, results: Dict) -> Optional[Dict[str, Path]]:
        """
        Generate standard report for backtest results.
        
        Args:
            results: Backtest results dictionary
            
        Returns:
            Dictionary of output file paths or None if report generation fails
        """
        try:
            # Create report generator
            if self.report_config is None:
                # Use default configuration
                self.report_config = ReportConfig(
                    title=f"{results['strategy_params'].get('name', 'Strategy')} Backtest Report",
                    subtitle=f"Period: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}",
                    author="Backtest Suite"
                )
            
            generator = StandardReportGenerator(self.report_config)
            
            # Generate timestamp for unique naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = results['strategy_params'].get('name', 'strategy').lower().replace(' ', '_')
            report_name = f"{strategy_name}_{timestamp}"
            
            # Ensure report directory exists
            self.report_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate report
            print(f"\nGenerating standard report...")
            output_files = generator.generate_report(
                backtest_results=results,
                output_dir=self.report_dir,
                report_name=report_name
            )
            
            print(f"Report saved to:")
            for format_type, filepath in output_files.items():
                print(f"  - {format_type.upper()}: {filepath}")
                
            return output_files
            
        except Exception as e:
            print(f"Warning: Failed to generate report: {str(e)}")
            return None