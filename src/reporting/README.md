# Standardized Reporting System

The Backtest Suite Standardized Reporting System provides a consistent, professional framework for generating comprehensive backtest reports. This system ensures all strategies and analyses follow the same high-quality reporting standards.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Report Configuration](#report-configuration)
4. [Report Sections](#report-sections)
5. [Visualization Types](#visualization-types)
6. [Customization](#customization)
7. [Best Practices](#best-practices)
8. [API Reference](#api-reference)

## Overview

The reporting system consists of several key components:

- **StandardReportGenerator**: Main class that orchestrates report generation
- **Report Sections**: Modular components for different analysis areas
- **Visualization Types**: Consistent, professional charts and graphs
- **HTML Template**: Professional report layout with interactive elements

### Key Features

- 📊 **Consistent Format**: All reports follow the same professional structure
- 🎨 **Professional Visuals**: High-quality charts with consistent styling
- 📈 **Comprehensive Analysis**: Covers performance, risk, trades, and market regimes
- 🔧 **Highly Customizable**: Configure sections, metrics, and thresholds
- 📱 **Responsive Design**: Reports work on all devices
- 🖨️ **Print-Friendly**: Optimized for PDF export and printing

## Quick Start

### Basic Usage

```python
from src.reporting import StandardReportGenerator, ReportConfig

# Create report generator with default config
generator = StandardReportGenerator()

# Generate report from backtest results
output_files = generator.generate_report(
    backtest_results=results,
    output_dir="reports/",
    report_name="my_strategy_report"
)

print(f"Report saved to: {output_files['html']}")
```

### Custom Configuration

```python
from src.reporting import ReportConfig

# Create custom configuration
config = ReportConfig(
    title="Advanced Trading Strategy Analysis",
    subtitle="Q4 2024 Performance Review",
    author="Trading Team",
    
    # Customize thresholds
    min_sharpe_ratio=1.5,
    max_drawdown_limit=0.15,
    min_win_rate=0.55,
    
    # Select output formats
    output_formats=["html", "pdf"],
    
    # Customize colors
    color_scheme={
        "primary": "#1E88E5",
        "secondary": "#FFA726",
        "success": "#43A047",
        "warning": "#FB8C00",
        "danger": "#E53935"
    }
)

generator = StandardReportGenerator(config)
```

## Report Configuration

### ReportConfig Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | str | "Backtest Results Report" | Main report title |
| `subtitle` | str | "" | Optional subtitle |
| `author` | str | "Backtest Suite" | Report author |
| `include_executive_summary` | bool | True | Include executive summary section |
| `include_performance_analysis` | bool | True | Include performance analysis |
| `include_risk_analysis` | bool | True | Include risk analysis |
| `include_trade_analysis` | bool | True | Include trade analysis |
| `include_market_regime_analysis` | bool | True | Include regime analysis |
| `include_technical_details` | bool | True | Include technical details |
| `chart_style` | str | "professional" | Chart styling theme |
| `figure_dpi` | int | 300 | DPI for saved figures |
| `figure_size` | tuple | (10, 6) | Default figure size |
| `output_formats` | list | ["html", "pdf", "json"] | Output formats to generate |
| `confidence_level` | float | 0.95 | Confidence level for statistics |
| `risk_free_rate` | float | 0.02 | Risk-free rate for calculations |
| `benchmark_symbol` | str | "SPY" | Benchmark for comparison |

### Performance Thresholds

Configure thresholds for performance evaluation:

```python
config = ReportConfig(
    min_sharpe_ratio=1.0,      # Minimum acceptable Sharpe ratio
    max_drawdown_limit=0.20,   # Maximum acceptable drawdown
    min_win_rate=0.40,         # Minimum acceptable win rate
)
```

## Report Sections

### 1. Executive Summary

Provides a high-level overview of strategy performance:

- Key performance metrics
- Overall strategy assessment
- Performance and risk summaries
- Key findings and recommendations

```python
# The executive summary automatically evaluates performance
# based on configured thresholds and provides actionable insights
```

### 2. Performance Analysis

Detailed performance metrics and analysis:

- Return analysis (total, annualized, distribution)
- Risk-adjusted metrics (Sharpe, Sortino, Calmar)
- Benchmark comparison
- Performance attribution
- Rolling performance metrics
- Statistical significance tests

### 3. Risk Analysis

Comprehensive risk assessment:

- Drawdown analysis (depth, duration, recovery)
- Volatility analysis (current, historical, regimes)
- Value at Risk (VaR) and CVaR
- Stress testing scenarios
- Risk decomposition
- Concentration analysis

### 4. Trade Analysis

Detailed examination of trading activity:

- Trade statistics (count, frequency, size)
- Win/loss analysis
- Trade duration analysis
- Entry/exit analysis
- Trade clustering patterns
- Performance by various factors

### 5. Market Regime Analysis

Analysis of performance across market conditions:

- Regime identification
- Performance by regime
- Regime transitions
- Adaptive behavior
- Correlation analysis

### 6. Technical Details

Implementation and configuration details:

- Strategy parameters
- Execution statistics
- Computational performance
- Data quality assessment
- Backtest assumptions
- Implementation notes

## Visualization Types

### Available Visualizations

1. **EquityCurveChart**: Portfolio value over time with drawdown shading
2. **DrawdownChart**: Underwater plot and duration analysis
3. **ReturnsDistribution**: Histogram, Q-Q plot, and volatility analysis
4. **TradeScatterPlot**: Trade analysis and P&L distribution
5. **RollingMetricsChart**: Rolling performance metrics
6. **HeatmapVisualization**: Correlation matrices and parameter sensitivity

### Creating Custom Visualizations

```python
from src.reporting.visualization_types import EquityCurveChart

# Create equity curve visualization
chart = EquityCurveChart(config)
result = chart.create(
    equity_curve=equity_series,
    benchmark=benchmark_series,
    save_path="charts/equity_curve.png"
)

# Access the Plotly figure
fig = result['figure']
```

### Visualization Customization

```python
# Customize chart appearance
config = ReportConfig(
    chart_style="professional",  # or "minimal", "colorful"
    figure_dpi=300,
    figure_size=(12, 8),
    color_scheme={
        "primary": "#2E86AB",
        "success": "#27AE60",
        "danger": "#E74C3C"
    }
)
```

## Customization

### Adding Custom Sections

```python
# Add a custom section to the report
generator.add_custom_section(
    section_name="Custom Analysis",
    section_content={
        "metric1": "value1",
        "metric2": "value2",
        "analysis": "Custom analysis text"
    }
)
```

### Using Predefined Themes

```python
# Apply a predefined theme
generator.set_theme("minimal")  # Options: "professional", "minimal", "colorful"
```

### Custom Metric Formatting

```python
# All sections use consistent formatting
section = PerformanceAnalysis(config)

# Format numbers consistently
formatted_value = section.format_number(0.1523, "percentage")  # "15.23%"
formatted_value = section.format_number(1250000, "currency")   # "$1,250,000"
formatted_value = section.format_number(1.85, "ratio")         # "1.85"
```

## Best Practices

### 1. Data Preparation

Ensure your backtest results dictionary contains:

```python
backtest_results = {
    "equity_curve": pd.Series,      # Required: Portfolio value over time
    "trades": pd.DataFrame,         # Required: Trade records
    "metrics": dict,                # Required: Performance metrics
    "strategy_params": dict,        # Required: Strategy parameters
    
    # Optional but recommended
    "returns": pd.Series,           # Daily returns
    "benchmark": dict,              # Benchmark comparison data
    "market_data": pd.DataFrame,    # Market data for regime analysis
    "execution_stats": dict,        # Execution statistics
    "performance_stats": dict,      # Computational performance
    "data_statistics": dict        # Data quality metrics
}
```

### 2. Trade Data Format

Trades DataFrame should include:

```python
trades_df = pd.DataFrame({
    "entry_time": pd.DatetimeIndex,
    "exit_time": pd.DatetimeIndex,
    "side": str,                    # "long" or "short"
    "size": float,                  # Position size
    "pnl": float,                   # Profit/loss
    "duration": float,              # Hours
    "entry_reason": str,            # Optional: Entry signal
    "exit_reason": str,             # Optional: Exit signal
})
```

### 3. Metrics Dictionary

Standard metrics to include:

```python
metrics = {
    # Returns
    "total_return": 0.45,
    "annual_return": 0.22,
    "monthly_return": 0.018,
    
    # Risk
    "volatility": 0.16,
    "max_drawdown": -0.15,
    "downside_deviation": 0.10,
    
    # Risk-adjusted
    "sharpe_ratio": 1.38,
    "sortino_ratio": 1.85,
    "calmar_ratio": 1.47,
    
    # Trading
    "win_rate": 0.58,
    "profit_factor": 1.75,
    "avg_win_loss_ratio": 1.5,
    
    # VaR
    "var_95": -0.025,
    "cvar_95": -0.035
}
```

### 4. Performance Tips

- Pre-calculate metrics before report generation
- Use appropriate data types (Series for time series, DataFrame for tabular data)
- Ensure date indices are properly formatted
- Include all optional data for comprehensive analysis

## API Reference

### StandardReportGenerator

```python
class StandardReportGenerator:
    def __init__(self, config: Optional[ReportConfig] = None)
    
    def generate_report(
        self,
        backtest_results: Dict[str, Any],
        output_dir: Union[str, Path],
        report_name: Optional[str] = None
    ) -> Dict[str, Path]
    
    def add_custom_section(self, section_name: str, section_content: Any)
    
    def set_theme(self, theme_name: str)
```

### ReportSection Base Class

```python
class ReportSection(ABC):
    def __init__(self, config)
    
    @abstractmethod
    def generate(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]
    
    def format_number(self, value: float, format_type: str = "general") -> str
```

### Visualization Base Class

```python
class BaseVisualization:
    def __init__(self, config)
    
    def create(self, data, save_path: Optional[Path] = None) -> Dict[str, Any]
    
    def save_figure(self, fig, save_path: Optional[Path] = None, format: str = "png")
```

## Examples

### Complete Example: Strategy Report

```python
import pandas as pd
from datetime import datetime, timedelta
from src.reporting import StandardReportGenerator, ReportConfig

# Load backtest results
equity_curve = pd.read_csv("results/equity_curve.csv", index_col=0, parse_dates=True)
trades = pd.read_csv("results/trades.csv", parse_dates=["entry_time", "exit_time"])

# Calculate metrics
returns = equity_curve.pct_change()
metrics = {
    "total_return": (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1),
    "annual_return": returns.mean() * 252,
    "volatility": returns.std() * np.sqrt(252),
    "sharpe_ratio": (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
    "max_drawdown": ((equity_curve / equity_curve.cummax()) - 1).min(),
    "win_rate": (trades["pnl"] > 0).mean(),
    "profit_factor": trades[trades["pnl"] > 0]["pnl"].sum() / abs(trades[trades["pnl"] < 0]["pnl"].sum())
}

# Prepare results dictionary
backtest_results = {
    "equity_curve": equity_curve,
    "trades": trades,
    "metrics": metrics,
    "returns": returns,
    "strategy_params": {
        "lookback_period": 20,
        "entry_threshold": 2.0,
        "stop_loss": 0.02,
        "position_sizing": "risk_parity"
    }
}

# Configure report
config = ReportConfig(
    title="Mean Reversion Strategy Analysis",
    subtitle="2024 Performance Review",
    author="Quantitative Research Team",
    min_sharpe_ratio=1.0,
    max_drawdown_limit=0.20,
    output_formats=["html", "pdf"]
)

# Generate report
generator = StandardReportGenerator(config)
output_files = generator.generate_report(
    backtest_results=backtest_results,
    output_dir="reports/2024/",
    report_name="mean_reversion_q4"
)

print(f"HTML Report: {output_files['html']}")
print(f"PDF Report: {output_files['pdf']}")
```

### Custom Visualization Example

```python
from src.reporting.visualization_types import HeatmapVisualization

# Create parameter sensitivity heatmap
heatmap = HeatmapVisualization(config)

# Prepare sensitivity data
sensitivity_data = {
    "param1_name": "Lookback Period",
    "param1_values": [10, 20, 30, 40, 50],
    "param2_name": "Entry Threshold",
    "param2_values": [1.5, 2.0, 2.5, 3.0],
    "results": sharpe_matrix,  # 2D array of Sharpe ratios
    "metric": "Sharpe Ratio"
}

result = heatmap.create(
    data=sensitivity_data,
    chart_type="parameter_sensitivity",
    save_path="charts/parameter_sensitivity.png"
)
```

### Batch Report Generation

```python
# Generate reports for multiple strategies
strategies = ["momentum", "mean_reversion", "pairs_trading"]

for strategy in strategies:
    results = load_backtest_results(strategy)
    
    config = ReportConfig(
        title=f"{strategy.title()} Strategy Analysis",
        output_formats=["html", "json"]
    )
    
    generator = StandardReportGenerator(config)
    generator.generate_report(
        backtest_results=results,
        output_dir=f"reports/{strategy}/",
        report_name=f"{strategy}_report"
    )
```

## Troubleshooting

### Common Issues

1. **Missing Data Error**
   ```python
   # Ensure all required fields are present
   required_fields = ["equity_curve", "trades", "metrics", "strategy_params"]
   ```

2. **Empty Trades DataFrame**
   ```python
   # Handle strategies with no trades
   if trades.empty:
       trades = pd.DataFrame(columns=["entry_time", "exit_time", "pnl"])
   ```

3. **PDF Generation Issues**
   ```bash
   # Install weasyprint for PDF support
   pip install weasyprint
   ```

4. **Memory Issues with Large Reports**
   ```python
   # Process data in chunks or reduce figure DPI
   config = ReportConfig(figure_dpi=150)
   ```

## Contributing

To add new report sections or visualizations:

1. Extend the appropriate base class
2. Follow the existing naming conventions
3. Ensure consistent formatting using provided methods
4. Add comprehensive docstrings
5. Include examples in documentation

## License

This reporting system is part of the Backtest Suite project and follows the same license terms.