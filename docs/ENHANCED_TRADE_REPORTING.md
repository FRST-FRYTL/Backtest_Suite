# Enhanced Trade Reporting Documentation

## Overview

The Enhanced Trade Reporting system provides comprehensive analysis and visualization of trading performance with detailed price-level information, stop loss analysis, and risk metrics. This system extends the standard reporting framework to include advanced trade analytics that help traders understand their strategy performance at a granular level.

## Key Features

### 1. **Detailed Trade Price Analysis**
- Entry and exit price tracking
- Stop loss and take profit level visualization
- Price movement analysis and slippage tracking
- Trade execution quality metrics

### 2. **Stop Loss Analysis**
- Stop loss effectiveness measurement
- Stop hit rate analysis
- Optimal stop placement recommendations
- Stop distance distribution analysis

### 3. **Risk Per Trade Analysis**
- Risk/reward ratio calculation
- Position sizing analysis
- Risk consistency measurement
- Portfolio risk metrics

### 4. **Enhanced Visualizations**
- Interactive trade price charts
- Stop loss effectiveness dashboards
- Risk distribution analysis
- Trade timeline visualization

### 5. **Comprehensive Data Export**
- Enhanced JSON export with full trade details
- Detailed trade tables in reports
- Exportable visualization data
- Performance metrics with trade-level granularity

## Configuration

### Basic Configuration

```python
from src.reporting.report_config import ReportConfig, TradeReportingConfig

# Configure trade reporting features
trade_config = TradeReportingConfig(
    enable_detailed_trade_prices=True,
    price_display_format="absolute",  # or "percentage"
    show_entry_exit_prices=True,
    show_stop_loss_prices=True,
    show_take_profit_prices=True,
    enable_stop_loss_analysis=True,
    enable_risk_per_trade_analysis=True,
    max_trades_in_detailed_table=100,
    include_trade_timing_analysis=True,
    show_trade_price_charts=True
)

# Create main report config
config = ReportConfig()
config.trade_reporting = trade_config
```

### Advanced Configuration

```python
# Customize visualization settings
config.style.chart_height = 600
config.style.chart_width = 1000
config.style.primary_color = "#1f77b4"
config.style.success_color = "#2ca02c"
config.style.danger_color = "#d62728"

# Add custom trade metrics
config.add_custom_metric(
    key="custom_risk_metric",
    name="Custom Risk Metric",
    format="percentage",
    higher_is_better=False,
    description="Custom risk calculation"
)
```

## Data Requirements

### Required Trade Data Fields

For basic trade analysis:
- `entry_time`: Trade entry timestamp
- `exit_time`: Trade exit timestamp  
- `entry_price`: Price at which trade was entered
- `exit_price`: Price at which trade was exited
- `side`: Trade direction ('long' or 'short')
- `size`: Trade size/quantity
- `pnl`: Profit/loss for the trade

### Optional Enhanced Fields

For advanced analysis:
- `stop_loss`: Stop loss price level
- `take_profit`: Take profit price level
- `exit_reason`: Reason for exit ('target', 'stop', 'time', etc.)
- `commission`: Commission paid
- `slippage`: Slippage experienced
- `trade_id`: Unique identifier for the trade
- `duration`: Trade duration in hours

### Example Trade Data Structure

```python
import pandas as pd
from datetime import datetime, timedelta

trades = pd.DataFrame({
    'trade_id': [1, 2, 3],
    'entry_time': [
        datetime(2023, 1, 1, 9, 30),
        datetime(2023, 1, 2, 10, 0),
        datetime(2023, 1, 3, 11, 15)
    ],
    'exit_time': [
        datetime(2023, 1, 1, 15, 30),
        datetime(2023, 1, 2, 14, 0),
        datetime(2023, 1, 3, 16, 45)
    ],
    'side': ['long', 'short', 'long'],
    'entry_price': [150.00, 148.50, 152.25],
    'exit_price': [155.00, 145.00, 149.75],
    'stop_loss': [145.00, 152.00, 147.00],
    'take_profit': [160.00, 140.00, 158.00],
    'size': [100, 200, 150],
    'pnl': [500.00, 700.00, -375.00],
    'exit_reason': ['target', 'target', 'stop'],
    'commission': [5.00, 5.00, 5.00],
    'slippage': [0.05, 0.03, 0.02],
    'duration': [6.0, 4.0, 5.5]
})
```

## Usage Examples

### Basic Report Generation

```python
from src.reporting.standard_report_generator import StandardReportGenerator
from src.reporting.report_config import ReportConfig, TradeReportingConfig

# Configure enhanced trade reporting
config = ReportConfig()
config.trade_reporting = TradeReportingConfig(
    enable_detailed_trade_prices=True,
    enable_stop_loss_analysis=True,
    enable_risk_per_trade_analysis=True
)

# Generate report
generator = StandardReportGenerator(config)
report_paths = generator.generate_report(
    backtest_results=backtest_results,
    output_dir="reports/enhanced_trade_analysis",
    report_name="my_strategy_enhanced"
)

print(f"HTML Report: {report_paths['html']}")
print(f"JSON Export: {report_paths['json']}")
```

### Custom Visualization Generation

```python
from src.reporting.visualizations import ReportVisualizations

# Create visualizations
viz = ReportVisualizations()

# Trade price chart
price_chart = viz.create_trade_price_chart(
    trades=trades_data,
    price_data=price_data
)

# Stop loss analysis chart
stop_chart = viz.create_stop_loss_analysis(trades_data)

# Risk analysis chart
risk_chart = viz.create_trade_risk_chart(trades_data)

# Save charts
price_chart.write_html("trade_prices.html")
stop_chart.write_html("stop_analysis.html")
risk_chart.write_html("risk_analysis.html")
```

### Enhanced JSON Export

```python
from src.reporting.enhanced_json_export import create_enhanced_json_export

# Configure export
export_config = {
    'include_detailed_trades': True,
    'include_price_analysis': True,
    'include_stop_loss_analysis': True,
    'include_risk_analysis': True,
    'max_trades_export': 500,
    'decimal_places': 4
}

# Create enhanced export
json_path = create_enhanced_json_export(
    report_data=report_data,
    output_path="enhanced_report.json",
    config=export_config
)

print(f"Enhanced JSON export saved to: {json_path}")
```

### Markdown Report Generation

```python
from src.reporting.markdown_template import generate_markdown_report

# Generate markdown report
markdown_content = generate_markdown_report(report_data, config)

# Save to file
with open("enhanced_trade_report.md", "w") as f:
    f.write(markdown_content)
```

## Enhanced Visualizations

### Trade Price Chart

The trade price chart provides a comprehensive view of trade execution:

- **Price line**: Shows the underlying asset price
- **Entry points**: Triangle-up markers in primary color
- **Exit points**: Triangle-down markers (green for profit, red for loss)
- **Stop loss levels**: X markers in red
- **Take profit levels**: Star markers in green

```python
# Create trade price chart
fig = viz.create_trade_price_chart(trades, price_data)
fig.show()
```

### Stop Loss Analysis Chart

Multi-panel analysis of stop loss effectiveness:

- **Panel 1**: Stop hit rate pie chart
- **Panel 2**: Stop distance distribution histogram
- **Panel 3**: P&L comparison (stop hit vs no stop hit)
- **Panel 4**: Stop distance vs P&L scatter plot

```python
# Create stop loss analysis
fig = viz.create_stop_loss_analysis(trades)
fig.show()
```

### Trade Risk Chart

Comprehensive risk analysis across four panels:

- **Panel 1**: Risk per trade distribution
- **Panel 2**: Risk vs P&L scatter plot
- **Panel 3**: Risk over time line chart
- **Panel 4**: Risk by trade size scatter plot

```python
# Create trade risk chart
fig = viz.create_trade_risk_chart(trades)
fig.show()
```

## Report Sections

### Enhanced Trade Analysis Section

The trade analysis section now includes:

1. **Basic Trade Statistics**
   - Total trades, win rate, profit factor
   - Average trade P&L and duration

2. **Win/Loss Analysis**
   - Detailed winner vs loser comparison
   - Distribution analysis

3. **Price Analysis**
   - Entry/exit price statistics
   - Price movement patterns
   - Slippage analysis

4. **Stop Loss Analysis**
   - Stop loss usage rate
   - Stop hit rate and effectiveness
   - Optimal stop placement analysis

5. **Risk Analysis**
   - Risk per trade metrics
   - Position sizing analysis
   - Risk consistency measurement

6. **Detailed Trades Table**
   - Complete trade-by-trade breakdown
   - All price levels and execution details
   - Calculated metrics (risk %, P&L %, etc.)

### Detailed Trades Table

The detailed trades table includes:

| Column | Description |
|--------|-------------|
| Trade ID | Unique identifier |
| Entry Time | Trade entry timestamp |
| Exit Time | Trade exit timestamp |
| Side | Long or short position |
| Entry Price | Execution price at entry |
| Exit Price | Execution price at exit |
| Stop Loss | Stop loss price level |
| Take Profit | Take profit price level |
| P&L | Profit/loss amount |
| Duration | Trade duration in hours |
| Risk % | Risk as percentage of entry price |
| P&L % | P&L as percentage change |

## JSON Export Schema

### Enhanced Export Structure

```json
{
  "metadata": {
    "enhanced_export_version": "1.0.0",
    "export_timestamp": "2023-12-01T12:00:00",
    "trade_reporting_enabled": true,
    "features_included": ["detailed_trades", "price_analysis", "stop_loss_analysis"]
  },
  "trade_analysis": {
    "basic_statistics": {...},
    "win_loss_analysis": {...},
    "profitability_analysis": {...}
  },
  "price_analysis": {
    "entry_price_statistics": {...},
    "exit_price_statistics": {...},
    "price_movement_analysis": {...},
    "slippage_analysis": {...}
  },
  "stop_loss_analysis": {
    "stop_loss_usage": {...},
    "stop_loss_effectiveness": {...},
    "stop_distance_analysis": {...}
  },
  "risk_analysis": {
    "risk_per_trade": {...},
    "position_sizing": {...},
    "risk_reward_ratios": {...}
  },
  "detailed_trades": [...],
  "performance_metrics": {...},
  "visualizations_data": {...}
}
```

### Trade Record Schema

```json
{
  "trade_id": 1,
  "entry_time": "2023-01-01T09:30:00",
  "exit_time": "2023-01-01T15:30:00",
  "side": "long",
  "size": 100,
  "entry_price": 150.00,
  "exit_price": 155.00,
  "stop_loss": 145.00,
  "take_profit": 160.00,
  "pnl": 500.00,
  "duration_hours": 6.0,
  "exit_reason": "target",
  "commission": 5.00,
  "slippage": 0.05,
  "price_change_pct": 3.33,
  "stop_distance_pct": 3.33,
  "trade_value": 15000.00
}
```

## Best Practices

### 1. **Data Quality**
- Ensure all required fields are present
- Validate price data consistency
- Check for missing or invalid timestamps
- Verify trade P&L calculations

### 2. **Performance Optimization**
- Limit detailed trades table to reasonable size
- Use appropriate decimal precision
- Consider memory usage for large datasets
- Implement data sampling for very large trade sets

### 3. **Visualization Guidelines**
- Use consistent color schemes
- Ensure charts are readable at different sizes
- Include appropriate legends and labels
- Test interactive features

### 4. **Report Configuration**
- Customize metrics based on strategy type
- Set appropriate thresholds for alerts
- Configure display formats for your audience
- Enable relevant analysis sections

### 5. **Security Considerations**
- Sanitize sensitive data in exports
- Implement access controls for detailed reports
- Consider data retention policies
- Use secure file handling practices

## Troubleshooting

### Common Issues

1. **Missing Trade Data**
   - Ensure all required fields are present
   - Check data types and formats
   - Validate timestamp formats

2. **Visualization Errors**
   - Verify Plotly installation
   - Check data completeness
   - Ensure sufficient data points

3. **Performance Issues**
   - Reduce trade data size
   - Limit visualization complexity
   - Use appropriate sampling

4. **Export Failures**
   - Check file permissions
   - Verify output directory exists
   - Validate JSON serialization

### Error Messages

- **"No trades available for analysis"**: Check that trades DataFrame is not empty
- **"Insufficient price data"**: Ensure entry_price and exit_price columns exist
- **"No stop loss data available"**: Verify stop_loss column exists and has data
- **"Visualization creation failed"**: Check Plotly installation and data format

## API Reference

### TradeReportingConfig

```python
class TradeReportingConfig:
    enable_detailed_trade_prices: bool = True
    price_display_format: str = "absolute"  # "absolute" or "percentage"
    show_entry_exit_prices: bool = True
    show_stop_loss_prices: bool = True
    show_take_profit_prices: bool = True
    enable_stop_loss_analysis: bool = True
    enable_risk_per_trade_analysis: bool = True
    max_trades_in_detailed_table: int = 100
    include_trade_timing_analysis: bool = True
    show_trade_price_charts: bool = True
```

### Key Methods

```python
# Visualization creation
viz.create_trade_price_chart(trades, price_data)
viz.create_stop_loss_analysis(trades)
viz.create_trade_risk_chart(trades)

# Enhanced export
create_enhanced_json_export(report_data, output_path, config)

# Markdown generation
generate_markdown_report(report_data, config)
```

## Examples and Templates

See the `examples/` directory for:
- `enhanced_trade_reporting_example.py`: Complete usage example
- `custom_trade_visualizations.py`: Custom visualization creation
- `trade_analysis_templates.py`: Template configurations
- `batch_trade_analysis.py`: Processing multiple strategies

## Future Enhancements

Planned features for future versions:
- Real-time trade monitoring
- Machine learning-based trade analysis
- Advanced risk attribution
- Multi-timeframe trade analysis
- Integration with external data sources
- Advanced portfolio-level risk metrics

---

The Enhanced Trade Reporting system provides comprehensive insights into trading performance, helping traders optimize their strategies through detailed price-level analysis, stop loss effectiveness measurement, and risk management assessment.