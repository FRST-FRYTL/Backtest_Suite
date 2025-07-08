"""
Example usage of the monitoring system with backtesting.
"""

import asyncio
import logging
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.monitoring import (
    MonitoringConfig,
    AlertEngine,
    MonitoringDashboard,
    PerformanceCollector,
    PositionCollector,
    SystemHealthCollector,
    MetricAggregator,
    Alert,
    AlertType,
    AlertPriority,
    create_drawdown_alert,
    create_position_size_alert,
    create_signal_alert
)
from src.backtesting import Portfolio, BacktestEngine
from src.data import DataFetcher, DataCache
from src.strategies import StrategyBuilder


async def setup_monitoring_system(portfolio: Portfolio):
    """Setup and configure the monitoring system."""
    
    # Load configuration
    config_path = Path("src/monitoring/config/default_config.json")
    if config_path.exists():
        config = MonitoringConfig.from_file(config_path)
    else:
        config = MonitoringConfig()
    
    # Initialize alert engine
    alert_engine = AlertEngine(config.alerts)
    
    # Initialize collectors
    metric_aggregator = MetricAggregator()
    
    # Add collectors
    performance_collector = PerformanceCollector(config.collectors, portfolio)
    position_collector = PositionCollector(config.collectors, portfolio)
    system_collector = SystemHealthCollector(config.collectors)
    
    metric_aggregator.add_collector(performance_collector)
    metric_aggregator.add_collector(position_collector)
    metric_aggregator.add_collector(system_collector)
    
    # Initialize dashboard
    dashboard = MonitoringDashboard(config.dashboard, alert_engine, metric_aggregator)
    
    # Register custom alert handlers
    async def drawdown_handler(alert: Alert):
        """Custom handler for drawdown alerts."""
        print(f"⚠️  DRAWDOWN ALERT: {alert.message}")
        # Could trigger risk reduction here
    
    alert_engine.register_handler(AlertType.DRAWDOWN, drawdown_handler)
    
    return config, alert_engine, metric_aggregator, dashboard


async def monitor_backtest_performance(alert_engine: AlertEngine, portfolio: Portfolio):
    """Monitor performance and generate alerts during backtesting."""
    
    while True:
        # Check for drawdown
        if portfolio.current_drawdown > 0.08:  # 8% drawdown
            alert = create_drawdown_alert(
                portfolio.current_drawdown,
                portfolio.max_drawdown
            )
            await alert_engine.send_alert(alert)
        
        # Check position sizes
        for position in portfolio.positions.values():
            if not position.is_closed:
                position_size = abs(position.market_value) / portfolio.total_value
                if position_size > 0.20:  # 20% of portfolio
                    alert = create_position_size_alert(
                        position.symbol,
                        position_size,
                        0.20
                    )
                    await alert_engine.send_alert(alert)
        
        # Simulate signal alerts
        if portfolio.total_trades > 0 and portfolio.total_trades % 10 == 0:
            alert = create_signal_alert(
                "AAPL",
                "BUY",
                0.85
            )
            await alert_engine.send_alert(alert)
        
        await asyncio.sleep(5)  # Check every 5 seconds


async def main():
    """Main example function."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting Monitoring System Example")
    
    # Initialize portfolio
    portfolio = Portfolio(initial_capital=100000)
    
    # Setup monitoring system
    config, alert_engine, metric_aggregator, dashboard = await setup_monitoring_system(portfolio)
    
    # Start all components
    await alert_engine.start()
    await metric_aggregator.start_all()
    await dashboard.start()
    
    print(f"📊 Dashboard available at: http://{config.dashboard.host}:{config.dashboard.port}")
    print(f"🔌 WebSocket available at: ws://{config.dashboard.host}:{config.dashboard.websocket_port}")
    
    # Create monitoring task
    monitor_task = asyncio.create_task(monitor_backtest_performance(alert_engine, portfolio))
    
    # Simulate some portfolio activity
    print("\n📈 Simulating portfolio activity...")
    
    # Add some fake positions
    portfolio.open_position("AAPL", 100, 150.0)
    portfolio.open_position("GOOGL", 50, 2800.0)
    portfolio.open_position("MSFT", 75, 350.0)
    
    # Update portfolio value
    portfolio.update_equity(portfolio.total_value * 1.05)  # 5% gain
    
    # Record some trades
    portfolio.record_trade(500, True)  # Winning trade
    portfolio.record_trade(-200, False)  # Losing trade
    portfolio.record_trade(300, True)  # Winning trade
    
    # Simulate drawdown
    portfolio.current_drawdown = 0.09  # 9% drawdown
    
    try:
        # Keep running for demonstration
        print("\n✅ Monitoring system is running. Press Ctrl+C to stop.")
        print("📊 View real-time metrics in the dashboard")
        print("🔔 Alerts will appear as they are triggered")
        
        await asyncio.sleep(60)  # Run for 60 seconds
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitoring system...")
    
    finally:
        # Cleanup
        monitor_task.cancel()
        await metric_aggregator.stop_all()
        await alert_engine.stop()
        await dashboard.stop()
        
        # Save alert history
        alert_history_path = Path("data/alerts/history.json")
        alert_engine.save_history(alert_history_path)
        print(f"💾 Alert history saved to: {alert_history_path}")
        
        print("✅ Monitoring system stopped successfully")


if __name__ == "__main__":
    asyncio.run(main())