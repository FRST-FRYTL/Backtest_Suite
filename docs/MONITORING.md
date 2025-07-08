# Monitoring System Documentation

The Backtest Suite includes a comprehensive real-time monitoring and alerting system designed to track backtesting performance, system health, and trading signals.

## Overview

The monitoring system provides:
- **Real-time Performance Tracking**: Monitor key metrics like returns, Sharpe ratio, and drawdowns
- **Alert System**: Configurable alerts for trading signals, risk events, and system issues
- **Position Monitoring**: Track open positions, exposure, and P&L in real-time
- **System Health**: Monitor CPU, memory, and processing performance
- **Dashboard**: Web-based dashboard with real-time updates via WebSocket

## Architecture

```
monitoring/
├── config.py          # Configuration management
├── alerts.py          # Alert engine and notifications
├── collectors.py      # Metric collection system
├── dashboard.py       # Real-time web dashboard
├── alerts/            # Alert-specific modules
├── collectors/        # Collector-specific modules  
├── config/            # Configuration files
│   └── default_config.json
└── dashboard/         # Dashboard components
```

## Quick Start

```python
import asyncio
from src.monitoring import (
    MonitoringConfig,
    AlertEngine,
    MonitoringDashboard,
    PerformanceCollector,
    MetricAggregator
)

async def main():
    # Load configuration
    config = MonitoringConfig.from_file("config/monitoring.json")
    
    # Initialize components
    alert_engine = AlertEngine(config.alerts)
    aggregator = MetricAggregator()
    
    # Add collectors
    perf_collector = PerformanceCollector(config.collectors, portfolio)
    aggregator.add_collector(perf_collector)
    
    # Create dashboard
    dashboard = MonitoringDashboard(config.dashboard, alert_engine, aggregator)
    
    # Start monitoring
    await alert_engine.start()
    await aggregator.start_all()
    await dashboard.start()
    
    print(f"Dashboard: http://localhost:{config.dashboard.port}")

asyncio.run(main())
```

## Configuration

### Main Configuration Options

```json
{
  "enabled": true,
  "log_level": "INFO",
  "metric_retention_days": 30,
  "alert_history_days": 7,
  "alerts": { ... },
  "collectors": { ... },
  "dashboard": { ... }
}
```

### Alert Configuration

```json
{
  "alerts": {
    "enabled": true,
    "email_enabled": false,
    "webhook_enabled": true,
    "webhook_urls": ["https://hooks.slack.com/..."],
    "drawdown_threshold": 0.10,
    "position_size_threshold": 0.25,
    "win_rate_threshold": 0.30,
    "max_alerts_per_minute": 10,
    "alert_cooldown_seconds": 60
  }
}
```

### Collector Configuration

```json
{
  "collectors": {
    "collection_interval_seconds": 1.0,
    "buffer_size": 1000,
    "persist_to_disk": true,
    "persistence_path": "data/metrics",
    "performance_metrics": [
      "total_return", "sharpe_ratio", "win_rate", 
      "profit_factor", "max_drawdown"
    ]
  }
}
```

### Dashboard Configuration

```json
{
  "dashboard": {
    "enabled": true,
    "host": "localhost",
    "port": 8050,
    "websocket_enabled": true,
    "websocket_port": 8051,
    "theme": "dark",
    "max_chart_points": 500
  }
}
```

## Alert System

### Alert Types

- **DRAWDOWN**: Triggered when drawdown exceeds threshold
- **POSITION_SIZE**: Triggered when position size exceeds limit
- **WIN_RATE**: Triggered when win rate drops below threshold
- **SIGNAL**: Trading signal alerts
- **RISK**: Risk management alerts
- **SYSTEM_ERROR**: System and technical alerts

### Creating Custom Alerts

```python
from src.monitoring import Alert, AlertType, AlertPriority

# Create custom alert
alert = Alert(
    id="custom_001",
    type=AlertType.CUSTOM,
    priority=AlertPriority.HIGH,
    title="Large Position Opened",
    message="Position in AAPL exceeds 30% of portfolio",
    data={"symbol": "AAPL", "size": 0.32}
)

# Send alert
await alert_engine.send_alert(alert)
```

### Alert Handlers

Register custom handlers for specific alert types:

```python
async def position_alert_handler(alert: Alert):
    """Custom handler for position alerts."""
    if alert.data['size'] > 0.40:
        # Trigger risk reduction
        await reduce_position(alert.data['symbol'])

alert_engine.register_handler(AlertType.POSITION_SIZE, position_alert_handler)
```

## Metric Collectors

### Performance Collector

Tracks trading performance metrics:
- Total return
- Sharpe ratio
- Win rate
- Profit factor
- Maximum drawdown
- Current drawdown

### Position Collector

Monitors position-related metrics:
- Open positions count
- Position concentrations
- Total exposure
- Unrealized P&L
- Long vs short exposure

### System Health Collector

Monitors system performance:
- CPU usage
- Memory usage
- Processing tick rate
- Thread count
- Network connections

### Creating Custom Collectors

```python
from src.monitoring import MetricCollector, MetricPoint

class CustomCollector(MetricCollector):
    async def collect(self) -> List[MetricPoint]:
        metrics = []
        
        # Collect custom metric
        value = await self.get_custom_value()
        metric = MetricPoint(
            timestamp=time.time(),
            name="custom_metric",
            value=value,
            tags={"source": "custom"}
        )
        metrics.append(metric)
        
        return metrics
```

## Dashboard

The web-based dashboard provides real-time visualization of:

### Key Metrics Display
- Total Return (with color coding)
- Sharpe Ratio
- Win Rate
- Maximum Drawdown

### Charts
- **Equity Curve**: Real-time portfolio value
- **Position Exposure**: Pie chart of position concentrations
- **System Metrics**: CPU, memory, and performance graphs

### Tables
- **Recent Alerts**: Latest alerts with priority indicators
- **Open Positions**: Current positions with P&L

### WebSocket Integration

Real-time updates via WebSocket:

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8051');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'metrics') {
        // Update metrics display
        updateMetrics(data.data);
    } else if (data.type === 'alert') {
        // Show new alert
        showAlert(data.data);
    }
};
```

## Integration with Backtesting

### Automatic Integration

The monitoring system can be automatically integrated with backtesting:

```python
from src.backtesting import BacktestEngine
from src.monitoring import setup_monitoring

# Create backtest with monitoring
engine = BacktestEngine(
    data=data,
    strategy=strategy,
    portfolio=portfolio,
    enable_monitoring=True
)

# Monitoring will automatically track all metrics
results = engine.run()
```

### Manual Integration

For more control, manually integrate monitoring:

```python
# During backtest loop
for timestamp, data in market_data:
    # Process trades
    signals = strategy.generate_signals(data)
    
    # Update monitoring
    if monitoring_enabled:
        # Record tick for performance tracking
        system_collector.record_tick()
        
        # Check for alerts
        if portfolio.current_drawdown > config.alerts.drawdown_threshold:
            alert = create_drawdown_alert(
                portfolio.current_drawdown,
                portfolio.max_drawdown
            )
            await alert_engine.send_alert(alert)
```

## Performance Considerations

### Optimization Tips

1. **Adjust Collection Intervals**: Increase intervals for better performance
   ```python
   config.collectors.collection_interval_seconds = 5.0  # Every 5 seconds
   ```

2. **Limit Chart Points**: Reduce memory usage
   ```python
   config.dashboard.max_chart_points = 200  # Keep last 200 points
   ```

3. **Selective Metrics**: Only collect needed metrics
   ```python
   config.collectors.performance_metrics = ["total_return", "sharpe_ratio"]
   ```

4. **Disable Persistence**: For testing/development
   ```python
   config.collectors.persist_to_disk = False
   ```

### Resource Usage

Typical resource usage:
- **CPU**: < 5% overhead
- **Memory**: ~50-100MB for standard configuration
- **Disk**: ~1MB/hour with default persistence

## Troubleshooting

### Common Issues

1. **Dashboard Not Loading**
   - Check if port is available: `netstat -an | grep 8050`
   - Verify dashboard is enabled in config
   - Check firewall settings

2. **No Metrics Appearing**
   - Ensure collectors are started: `await aggregator.start_all()`
   - Check collection intervals
   - Verify portfolio is being updated

3. **Alerts Not Triggering**
   - Check alert thresholds in configuration
   - Verify alert engine is started
   - Check rate limiting settings

4. **WebSocket Connection Failed**
   - Ensure WebSocket port is available
   - Check CORS settings if connecting from different origin
   - Verify WebSocket is enabled in config

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in config
config.log_level = "DEBUG"
```

## Advanced Features

### Metric Persistence

Metrics are automatically persisted to disk:

```python
# Load historical metrics
from src.monitoring.collectors import load_metrics

historical = load_metrics(
    path="data/metrics/PerformanceCollector_*.json",
    start_time=datetime.now() - timedelta(days=7)
)
```

### Alert History

Access and analyze alert history:

```python
# Save alert history
alert_engine.save_history(Path("data/alerts/history.json"))

# Load and analyze
alert_engine.load_history(Path("data/alerts/history.json"))
alerts_by_type = alert_engine.get_alerts_by_type(AlertType.DRAWDOWN)
```

### Custom Dashboard Components

Extend the dashboard with custom components:

```python
# Add custom chart to dashboard
@app.callback(Output('custom-chart', 'figure'), 
              [Input('metrics-store', 'children')])
def update_custom_chart(metrics_json):
    # Create custom visualization
    fig = create_custom_chart(metrics_json)
    return fig
```

## Example Usage

See `examples/monitoring_example.py` for a complete working example demonstrating:
- System setup and configuration
- Custom alert handlers
- Integration with backtesting
- Dashboard usage
- Metric persistence

## API Reference

For detailed API documentation, see the docstrings in:
- `src/monitoring/config.py` - Configuration classes
- `src/monitoring/alerts.py` - Alert system
- `src/monitoring/collectors.py` - Metric collectors
- `src/monitoring/dashboard.py` - Dashboard components