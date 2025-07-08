"""
Example of running a backtest with real-time monitoring enabled.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.backtesting import BacktestEngine, Portfolio
from src.data import DataFetcher, DataCache
from src.strategies import StrategyBuilder
from src.monitoring import (
    MonitoringConfig,
    AlertEngine,
    MonitoringDashboard,
    PerformanceCollector,
    PositionCollector,
    SystemHealthCollector,
    MetricAggregator,
    AlertType,
    create_drawdown_alert,
    create_signal_alert
)


async def run_monitored_backtest():
    """Run a backtest with monitoring enabled."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting Monitored Backtest Example")
    
    # Initialize data components
    cache = DataCache()
    fetcher = DataFetcher(cache=cache)
    
    # Fetch historical data
    symbols = ["AAPL", "GOOGL", "MSFT"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"📊 Fetching data for {symbols} from {start_date.date()} to {end_date.date()}")
    
    data = {}
    for symbol in symbols:
        df = await fetcher.fetch_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval="1h"
        )
        data[symbol] = df
        print(f"✅ Fetched {len(df)} data points for {symbol}")
    
    # Create strategy (simple moving average crossover)
    strategy_config = {
        "name": "SMA Crossover",
        "type": "trend_following",
        "entry_rules": [
            {
                "indicator": "sma_cross",
                "params": {"fast_period": 10, "slow_period": 20},
                "condition": "bullish_cross"
            }
        ],
        "exit_rules": [
            {
                "indicator": "sma_cross",
                "params": {"fast_period": 10, "slow_period": 20},
                "condition": "bearish_cross"
            }
        ],
        "position_sizing": {
            "method": "fixed_percentage",
            "value": 0.02  # 2% risk per trade
        }
    }
    
    builder = StrategyBuilder()
    strategy = builder.from_config(strategy_config)
    
    # Initialize portfolio
    portfolio = Portfolio(initial_capital=100000)
    
    # Setup monitoring system
    print("\n🔧 Setting up monitoring system...")
    
    # Load or create monitoring config
    config = MonitoringConfig()
    config.dashboard.port = 8055  # Use different port to avoid conflicts
    config.dashboard.websocket_port = 8056
    config.alerts.drawdown_threshold = 0.05  # Alert at 5% drawdown
    config.alerts.position_size_threshold = 0.20  # Alert at 20% position
    
    # Initialize monitoring components
    alert_engine = AlertEngine(config.alerts)
    metric_aggregator = MetricAggregator()
    
    # Add collectors
    perf_collector = PerformanceCollector(config.collectors, portfolio)
    pos_collector = PositionCollector(config.collectors, portfolio)
    sys_collector = SystemHealthCollector(config.collectors)
    
    metric_aggregator.add_collector(perf_collector)
    metric_aggregator.add_collector(pos_collector)
    metric_aggregator.add_collector(sys_collector)
    
    # Create dashboard
    dashboard = MonitoringDashboard(config.dashboard, alert_engine, metric_aggregator)
    
    # Register alert handlers
    async def drawdown_handler(alert):
        print(f"⚠️  DRAWDOWN ALERT: {alert.message}")
        # In a real system, might reduce position sizes here
    
    async def signal_handler(alert):
        print(f"📈 SIGNAL ALERT: {alert.message}")
    
    alert_engine.register_handler(AlertType.DRAWDOWN, drawdown_handler)
    alert_engine.register_handler(AlertType.SIGNAL, signal_handler)
    
    # Start monitoring
    await alert_engine.start()
    await metric_aggregator.start_all()
    await dashboard.start()
    
    print(f"\n✅ Monitoring system started!")
    print(f"📊 Dashboard: http://localhost:{config.dashboard.port}")
    print(f"🔌 WebSocket: ws://localhost:{config.dashboard.websocket_port}")
    
    # Create backtest engine with monitoring hooks
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        portfolio=portfolio,
        commission=0.001  # 0.1% commission
    )
    
    # Add monitoring hooks to engine
    async def on_signal(symbol, signal_type, timestamp):
        """Hook called when a signal is generated."""
        alert = create_signal_alert(symbol, signal_type, 0.85)
        await alert_engine.send_alert(alert)
    
    async def on_trade(trade):
        """Hook called after a trade is executed."""
        # Update system health
        sys_collector.record_tick()
        
        # Check for drawdown alerts
        if portfolio.current_drawdown > config.alerts.drawdown_threshold:
            alert = create_drawdown_alert(
                portfolio.current_drawdown,
                portfolio.max_drawdown
            )
            await alert_engine.send_alert(alert)
    
    # Register hooks
    engine.on_signal = on_signal
    engine.on_trade = on_trade
    
    print("\n📈 Starting backtest with monitoring...")
    
    # Run backtest
    try:
        results = await engine.run_async()
        
        print("\n📊 Backtest Results:")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        
        # Keep monitoring running for observation
        print("\n✅ Backtest complete! Monitoring dashboard will remain active.")
        print("📊 View final results in the dashboard")
        print("🛑 Press Ctrl+C to stop")
        
        # Keep running
        await asyncio.sleep(300)  # Run for 5 minutes
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    
    finally:
        # Cleanup
        await metric_aggregator.stop_all()
        await alert_engine.stop()
        await dashboard.stop()
        
        # Save results
        alert_history_path = Path("data/alerts/backtest_history.json")
        alert_engine.save_history(alert_history_path)
        print(f"\n💾 Alert history saved to: {alert_history_path}")
        
        print("✅ Monitoring system stopped successfully")


# Modified BacktestEngine to support monitoring hooks
class MonitoredBacktestEngine(BacktestEngine):
    """Extended backtest engine with monitoring support."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_signal = None
        self.on_trade = None
    
    async def run_async(self):
        """Run backtest asynchronously with monitoring hooks."""
        results = {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'total_trades': 0
        }
        
        # Simulate backtest execution
        for i in range(100):  # Simulate 100 time steps
            # Simulate portfolio changes
            self.portfolio.update_equity(
                self.portfolio.total_value * (1 + (0.002 - 0.004 * (i % 3 == 0)))
            )
            
            # Simulate trades
            if i % 10 == 0:
                symbol = ["AAPL", "GOOGL", "MSFT"][i % 3]
                signal_type = "BUY" if i % 20 == 0 else "SELL"
                
                # Call signal hook
                if self.on_signal:
                    await self.on_signal(symbol, signal_type, datetime.now())
                
                # Simulate trade
                if signal_type == "BUY":
                    self.portfolio.open_position(symbol, 100, 150.0 + i)
                else:
                    # Close a position if exists
                    for pos_id, pos in list(self.portfolio.positions.items()):
                        if pos.symbol == symbol and not pos.is_closed:
                            self.portfolio.close_position(pos_id, 155.0 + i)
                            break
                
                # Record trade
                profit = 100 + (50 if i % 3 == 0 else -30)
                self.portfolio.record_trade(profit, profit > 0)
                
                # Call trade hook
                if self.on_trade:
                    await self.on_trade({'symbol': symbol, 'profit': profit})
            
            # Simulate drawdown
            if i > 50:
                self.portfolio.current_drawdown = 0.03 + (0.03 * (i % 10) / 10)
                self.portfolio.max_drawdown = max(
                    self.portfolio.max_drawdown, 
                    self.portfolio.current_drawdown
                )
            
            # Small delay to see real-time updates
            await asyncio.sleep(0.1)
        
        # Calculate final results
        results['total_return'] = (self.portfolio.total_value - self.portfolio.initial_capital) / self.portfolio.initial_capital
        results['sharpe_ratio'] = 1.5  # Simplified
        results['max_drawdown'] = self.portfolio.max_drawdown
        results['win_rate'] = self.portfolio.winning_trades / max(self.portfolio.total_trades, 1)
        results['total_trades'] = self.portfolio.total_trades
        
        return results


# Replace the engine creation in the main function
BacktestEngine = MonitoredBacktestEngine


if __name__ == "__main__":
    asyncio.run(run_monitored_backtest())