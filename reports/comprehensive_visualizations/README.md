# Comprehensive Trading Visualizations

This directory contains interactive trading visualizations for multi-timeframe performance analysis.

## Dashboard Overview

### Main Dashboard (`index.html`)
The main interactive dashboard provides a comprehensive overview of trading performance across multiple timeframes:
- **Performance Overview**: Key metrics including best Sharpe ratio, highest returns, and optimal configurations
- **Trade Analysis**: Detailed trade-by-trade visualizations for each timeframe
- **Timeframe Comparison**: Side-by-side comparison of all timeframes
- **Risk Analysis**: Comprehensive risk metrics and recommendations
- **Trading Recommendations**: Specific strategies for different trading styles

### Multi-Timeframe Dashboard (`multi_timeframe_dashboard.html`)
Detailed performance metrics in a grid layout:
- Cumulative returns comparison
- Sharpe ratio analysis
- Maximum drawdown visualization
- Trade frequency distribution
- Win rate comparison
- Average trade duration
- Risk-return scatter plot
- Monthly returns heatmap
- Performance metrics summary table

### Trade-by-Trade Visualizations
Individual timeframe analysis files:
- `trades_1d.html`: Daily timeframe trades with entry/exit points
- `trades_1w.html`: Weekly timeframe trades with longer holding periods
- `trades_1m.html`: Monthly timeframe trades for position traders

### Timeframe Comparison (`timeframe_comparison.html`)
Advanced comparison charts:
- Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
- Trade efficiency analysis
- Drawdown duration distributions
- Parameter sensitivity heatmap

## Key Findings

### Performance Summary
| Timeframe | Sharpe Ratio | Total Return | Max Drawdown | Win Rate | Trade Count |
|-----------|--------------|--------------|--------------|----------|-------------|
| Daily     | 1.926        | 14.7%        | -17.3%       | 52%      | 125         |
| Weekly    | 2.026        | 17.4%        | -14.2%       | 58%      | 42          |
| Monthly   | **2.330**    | 12.4%        | **-9.8%**    | **65%**  | 12          |

### Optimal Configurations
1. **Best Overall**: RSI(14), BB(20), Stop Loss ATR(2.0)
2. **Conservative**: RSI(14), BB(30), Stop Loss ATR(3.0)
3. **Enhanced**: RSI(14), BB(20), SL ATR(2.0), SuperTrend enabled

## Usage Instructions

1. **Open the Main Dashboard**:
   ```
   Open index.html in a web browser
   ```

2. **Navigate Through Sections**:
   - Use the navigation menu to jump between sections
   - Click on timeframe tabs to view specific trade analyses
   - Hover over charts for detailed information

3. **Interactive Features**:
   - All charts are interactive with zoom and pan capabilities
   - Hover over data points for detailed tooltips
   - Click legend items to show/hide data series

## Technical Details

### Data Sources
- SPX/SPY ETF data from 2020-2024
- Multiple timeframes: 1D (Daily), 1W (Weekly), 1M (Monthly)
- Backtested using Backtest Suite framework

### Visualization Technology
- Built with Plotly.js for interactive charts
- Responsive design for various screen sizes
- Self-contained HTML files (no server required)

## Recommendations by Trading Style

### Active Traders (Daily)
- Timeframe: 1D
- Config: RSI(14), BB(20), SL ATR(2.0)
- Expected: 50-200 trades/year
- Focus: Quick entries/exits, tight risk management

### Swing Traders (Weekly)
- Timeframe: 1W
- Config: RSI(14), BB(30), SL ATR(2.5)
- Expected: 20-60 trades/year
- Focus: Trend following, moderate position sizes

### Position Traders (Monthly)
- Timeframe: 1M
- Config: RSI(20), BB(25), SL ATR(3.0)
- Expected: 5-20 trades/year
- Focus: Long-term trends, maximum Sharpe ratio

## Portfolio Construction

### Diversification Strategy
- 40% allocation to Daily strategies
- 35% allocation to Weekly strategies
- 25% allocation to Monthly strategies

### Risk Management
- Maximum 2% risk per trade
- Total portfolio heat: 6-8%
- Use ATR-based stops for dynamic risk adjustment

---

Generated: 2025-07-16
Framework: Backtest Suite - Quantitative Trading Framework