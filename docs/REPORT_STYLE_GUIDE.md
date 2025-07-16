# Backtest Suite Report Style Guide

This document establishes standards for creating consistent, professional reports across all strategies and analyses in the Backtest Suite.

## Table of Contents

1. [Report Structure](#report-structure)
2. [Terminology Standards](#terminology-standards)
3. [Metric Definitions](#metric-definitions)
4. [Visual Design Guidelines](#visual-design-guidelines)
5. [Writing Style](#writing-style)
6. [Best Practices](#best-practices)

## Report Structure

### Standard Report Sections

All reports should follow this hierarchical structure:

1. **Executive Summary**
   - Overall assessment (1-2 sentences)
   - Key metrics dashboard (5-7 metrics)
   - Performance summary paragraph
   - Risk summary paragraph
   - Key findings (3-5 bullet points)
   - Recommendations (3-5 actionable items)

2. **Performance Analysis**
   - Return analysis with distribution
   - Risk-adjusted metrics with interpretations
   - Benchmark comparison (if applicable)
   - Rolling performance analysis
   - Statistical significance tests

3. **Risk Analysis**
   - Drawdown analysis (depth, duration, recovery)
   - Volatility analysis across timeframes
   - Value at Risk (VaR) and Conditional VaR
   - Stress testing scenarios
   - Risk decomposition

4. **Trade Analysis**
   - Trade statistics summary
   - Win/loss analysis
   - Duration and timing analysis
   - Entry/exit pattern analysis
   - Trade clustering

5. **Market Regime Analysis**
   - Current regime identification
   - Historical regime distribution
   - Performance by regime
   - Regime transition analysis

6. **Technical Details**
   - Strategy configuration
   - Execution statistics
   - Data quality metrics
   - Assumptions and limitations

### Section Priority

- **Critical**: Executive Summary, Performance Analysis, Risk Analysis
- **Important**: Trade Analysis, Technical Details
- **Optional**: Market Regime Analysis, Custom Sections

## Terminology Standards

### Performance Metrics

| Term | Definition | Format |
|------|------------|--------|
| Total Return | Cumulative return over entire period | XX.X% |
| Annual Return | Annualized return (geometric) | XX.X% |
| Sharpe Ratio | Risk-adjusted return metric | X.XX |
| Maximum Drawdown | Largest peak-to-trough decline | -XX.X% |
| Win Rate | Percentage of profitable trades | XX.X% |

### Risk Metrics

| Term | Definition | Format |
|------|------------|--------|
| Volatility | Annualized standard deviation | XX.X% |
| Downside Deviation | Standard deviation of negative returns | XX.X% |
| Value at Risk (95%) | 5th percentile of return distribution | -X.X% |
| Conditional VaR | Expected loss beyond VaR | -X.X% |
| Beta | Systematic risk vs benchmark | X.XX |

### Trading Metrics

| Term | Definition | Format |
|------|------------|--------|
| Profit Factor | Gross profits / Gross losses | X.XX |
| Win/Loss Ratio | Average win / Average loss | X.XX |
| Expectancy | Expected profit per trade | $X,XXX |
| Trade Frequency | Average trades per day | X.X |
| Hold Duration | Average position holding time | X.X hours/days |

## Metric Definitions

### Return Calculations

```python
# Total Return
total_return = (final_value / initial_value) - 1

# Annualized Return (CAGR)
years = days / 365.25
annual_return = (final_value / initial_value) ** (1/years) - 1

# Monthly Return
monthly_return = (1 + total_return) ** (1/months) - 1
```

### Risk-Adjusted Metrics

```python
# Sharpe Ratio
sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

# Sortino Ratio
sortino_ratio = (annual_return - risk_free_rate) / downside_deviation

# Calmar Ratio
calmar_ratio = annual_return / abs(max_drawdown)

# Information Ratio
information_ratio = active_return / tracking_error
```

### Drawdown Calculations

```python
# Maximum Drawdown
rolling_max = equity_curve.expanding().max()
drawdown = (equity_curve - rolling_max) / rolling_max
max_drawdown = drawdown.min()

# Average Drawdown
avg_drawdown = drawdown[drawdown < 0].mean()

# Drawdown Duration
# Count consecutive periods below previous peak
```

### Statistical Tests

```python
# T-test for returns
# H0: mean return = 0
# Significance level: 0.05

# Normality test (Jarque-Bera)
# H0: returns are normally distributed
# Significance level: 0.05
```

## Visual Design Guidelines

### Color Palette

**Primary Colors:**
- Primary: `#2E86AB` (Blue) - Main metrics, equity curves
- Secondary: `#A23B72` (Purple) - Benchmark, comparison
- Success: `#27AE60` (Green) - Positive values, profits
- Warning: `#F39C12` (Orange) - Caution indicators
- Danger: `#E74C3C` (Red) - Negative values, losses

**Supporting Colors:**
- Info: `#3498DB` (Light Blue) - Information boxes
- Background: `#FFFFFF` (White)
- Text: `#2C3E50` (Dark Gray)
- Muted: `#95A5A6` (Light Gray)

### Chart Standards

**Equity Curve:**
- Line width: 2px
- Include drawdown shading (red, 10% opacity)
- Mark peak and maximum drawdown points
- Grid: Light gray, subtle

**Distribution Plots:**
- Histogram: Primary color, 70% opacity
- Normal overlay: Secondary color, dashed line
- Include mean and median lines

**Heatmaps:**
- Color scale: RdYlGn for performance, Viridis for neutral
- Always include color bar with units
- Mark optimal points with star marker

### Typography

**Fonts:**
- Headers: System font stack (Arial, Helvetica, sans-serif)
- Body: System font stack
- Monospace: Consolas, Monaco, monospace (for code/numbers)

**Sizes:**
- H1: 2.5rem (Report title)
- H2: 2.0rem (Section titles)
- H3: 1.5rem (Subsections)
- Body: 1.0rem
- Small: 0.875rem

## Writing Style

### Tone and Voice

- **Professional**: Use formal but accessible language
- **Objective**: Present facts without bias
- **Actionable**: Provide specific recommendations
- **Concise**: Be clear and direct

### Metric Presentation

**Good Examples:**
- "The strategy achieved a Sharpe ratio of 1.45, indicating strong risk-adjusted returns."
- "Maximum drawdown of 15.3% occurred during the March 2023 volatility spike."

**Avoid:**
- "The strategy crushed it with amazing returns!"
- "Drawdown was pretty bad in March."

### Interpretations

Always provide context for metrics:

| Sharpe Ratio | Interpretation |
|--------------|----------------|
| > 2.0 | Excellent |
| 1.5 - 2.0 | Very Good |
| 1.0 - 1.5 | Good |
| 0.5 - 1.0 | Adequate |
| < 0.5 | Poor |

| Max Drawdown | Risk Level |
|--------------|------------|
| < 10% | Low Risk |
| 10-20% | Moderate Risk |
| 20-30% | High Risk |
| > 30% | Very High Risk |

### Recommendations Format

Structure recommendations as:
1. **Issue**: What needs attention
2. **Impact**: Why it matters
3. **Action**: Specific step to take

Example:
> **Issue**: Win rate below 50% (currently 45%)  
> **Impact**: Reduces strategy reliability and investor confidence  
> **Action**: Tighten entry criteria by increasing confirmation requirements

## Best Practices

### 1. Data Integrity

- Always validate input data before generating reports
- Handle missing data gracefully with clear notifications
- Check for outliers and data anomalies
- Document any data adjustments or cleaning

### 2. Metric Consistency

- Use the same calculation methods across all reports
- Apply consistent rounding rules:
  - Percentages: 1 decimal place (15.3%)
  - Ratios: 2 decimal places (1.45)
  - Currency: No decimals for large values ($125,000)
  - Small currency: 2 decimals ($15.33)

### 3. Visual Clarity

- Limit charts to essential information
- Use consistent scales when comparing
- Always label axes with units
- Include data source and date range
- Ensure sufficient contrast for readability

### 4. Report Completeness

Every report must include:
- Generation date and time
- Data period covered
- Strategy name and version
- Key assumptions
- Contact information or support link

### 5. Performance Benchmarks

When possible, compare against:
- Relevant market index (S&P 500, etc.)
- Risk-free rate (Treasury bills)
- Category average (if available)
- Buy-and-hold equivalent

### 6. Risk Disclosure

Include standard risk disclaimers:
> "Past performance does not guarantee future results. All trading involves risk of loss."

### 7. Accessibility

- Use high contrast colors
- Provide text alternatives for charts
- Structure content with proper headings
- Export reports in multiple formats (HTML, PDF)

## Report Quality Checklist

Before finalizing any report, verify:

- [ ] All sections follow standard structure
- [ ] Metrics use correct terminology and formatting
- [ ] Calculations follow defined formulas
- [ ] Visualizations adhere to design guidelines
- [ ] Writing is clear and professional
- [ ] Interpretations are provided for all metrics
- [ ] Recommendations are specific and actionable
- [ ] Data period and assumptions are documented
- [ ] Report is free of errors and typos
- [ ] All numbers reconcile correctly

## Version History

- **v1.0** (2024-01): Initial style guide
- **v1.1** (2024-02): Added metric definitions
- **v1.2** (2024-03): Enhanced visual guidelines

---

*This style guide is maintained by the Backtest Suite team. For questions or suggestions, please open an issue in the repository.*