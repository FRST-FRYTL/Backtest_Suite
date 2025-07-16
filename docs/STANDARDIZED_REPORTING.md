# Standardized Reporting System

## Overview

The Backtest Suite now includes a comprehensive standardized reporting system that ensures all strategy reports follow a consistent, professional format. This system was developed based on the successful comprehensive verification report format and provides multiple output formats with rich visualizations.

## Key Features

### 1. **Consistent Structure**
All reports follow the same professional structure:
- Executive Summary
- Performance Analysis
- Risk Analysis
- Trade Analysis
- Market Regime Analysis
- Technical Details
- Recommendations
- Appendices

### 2. **Multiple Output Formats**
- **Markdown**: For documentation and version control
- **HTML**: Interactive dashboard with Plotly charts
- **JSON**: Structured data export for further analysis
- **PDF**: Print-ready reports (optional)

### 3. **Professional Visualizations**
- Equity curves with drawdown shading
- Performance comparison charts
- Risk metric dashboards
- Trade distribution analysis
- Parameter sensitivity heatmaps
- Rolling performance metrics
- Monthly returns heatmaps

### 4. **Flexible Configuration**
```python
from src.reporting.report_config import ReportConfig, ReportTheme

config = ReportConfig(
    title="My Strategy Report",
    include_sections=["executive_summary", "performance_analysis"],
    theme=ReportTheme.PROFESSIONAL,
    output_formats=["html", "markdown"],
    performance_thresholds={
        "min_sharpe_ratio": 1.0,
        "min_annual_return": 0.10
    }
)
```

## Quick Start

### Basic Usage

```python
from src.reporting.standard_report_generator import StandardReportGenerator

# Create generator with default config
generator = StandardReportGenerator()

# Generate report from backtest results
report_paths = generator.generate_report(
    backtest_results, 
    output_dir="reports/my_strategy"
)
```

### Command Line Interface

```bash
# Generate report from backtest results file
python generate_standard_report.py backtest_results.json

# With custom options
python generate_standard_report.py results.json \
  --strategy "My Strategy" \
  --formats html markdown \
  --style professional
```

### Integration with Backtest Engine

```python
from src.backtesting.engine import BacktestEngine

# Run backtest with automatic report generation
engine = BacktestEngine(
    strategy=my_strategy,
    generate_report=True,
    report_config=ReportConfig(theme=ReportTheme.PROFESSIONAL)
)

results = engine.run_backtest(data)
# Report automatically generated in results.report_path
```

## Report Sections

### Executive Summary
- Key performance metrics at a glance
- Performance vs benchmark comparison
- Risk assessment summary
- Clear recommendations

### Performance Analysis
- Total and annualized returns
- Risk-adjusted metrics (Sharpe, Sortino, Calmar)
- Rolling performance analysis
- Comparison with benchmarks

### Risk Analysis
- Maximum drawdown and duration
- Value at Risk (VaR)
- Volatility analysis
- Stress testing results

### Trade Analysis
- Win rate and profit factor
- Average trade statistics
- Trade duration patterns
- Entry/exit analysis

### Market Regime Analysis
- Performance in different market conditions
- Bull/bear market performance
- Volatility regime analysis

### Technical Details
- Strategy configuration
- Backtest parameters
- Data quality assessment
- Computational performance

## Customization

### Adding Custom Sections

```python
class MyCustomSection(ReportSection):
    def generate_content(self, data):
        return {
            "markdown": "# My Custom Analysis\n...",
            "visualizations": [my_custom_chart],
            "metrics": {"custom_metric": 0.95}
        }

# Add to report
generator.add_custom_section(MyCustomSection())
```

### Custom Themes

```python
custom_theme = ReportTheme(
    name="custom",
    primary_color="#1E3A8A",
    font_family="Arial, sans-serif",
    chart_template="plotly_dark"
)

config = ReportConfig(theme=custom_theme)
```

## Best Practices

1. **Always Include Core Sections**: Executive summary, performance, and risk analysis
2. **Use Appropriate Timeframes**: Match report detail to strategy frequency
3. **Set Realistic Thresholds**: Configure performance thresholds based on strategy type
4. **Include Benchmarks**: Always compare against relevant benchmarks
5. **Document Assumptions**: Clearly state all assumptions and limitations

## Examples

See the `examples/` directory for:
- `standard_report_example.py`: Basic usage examples
- `backtest_with_report.py`: Integration with backtesting
- `custom_report_sections.py`: Adding custom sections
- `multi_strategy_report.py`: Comparing multiple strategies

## API Reference

See `src/reporting/README.md` for detailed API documentation.

## Style Guide

Refer to `docs/REPORT_STYLE_GUIDE.md` for:
- Terminology standards
- Metric definitions
- Visual design guidelines
- Writing style best practices

---

The standardized reporting system ensures all backtest reports maintain professional quality and consistency, making it easier to compare strategies and communicate results to stakeholders.