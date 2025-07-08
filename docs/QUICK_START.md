# Quick Start Guide

Get up and running with the Backtest Suite in minutes! This guide covers installation, basic usage, and your first backtest.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Example](#quick-example)
- [CLI Usage](#cli-usage)
- [Python API Usage](#python-api-usage)
- [Creating Your First Strategy](#creating-your-first-strategy)
- [Understanding Results](#understanding-results)
- [Next Steps](#next-steps)

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)
- Internet connection (for fetching market data)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/FRST-FRYTL/Backtest_Suite.git
cd Backtest_Suite
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 4. Verify Installation

```bash
# Check CLI is working
backtest --help

# List available indicators
backtest indicators
```

## Quick Example

Let's run a simple RSI mean reversion strategy on Apple stock:

```bash
# 1. Fetch data
backtest fetch -s AAPL -S 2023-01-01 -E 2023-12-31 -o data/AAPL.csv

# 2. Run backtest
backtest run -d data/AAPL.csv -s examples/strategies/rsi_mean_reversion.yaml -o results/ --html

# 3. View results
# Open results/backtest_report.html in your browser
```

## CLI Usage

### Fetching Data

```bash
# Single stock
backtest fetch -s AAPL -S 2023-01-01 -E 2023-12-31 -o data/AAPL.csv

# Multiple stocks
backtest fetch --multiple AAPL,GOOGL,MSFT -S 2023-01-01 -E 2023-12-31 -o data/

# Different interval (5-minute data)
backtest fetch -s AAPL -S 2023-12-01 -E 2023-12-31 -i 5m -o data/AAPL_5m.csv
```

### Running Backtests

```bash
# Basic backtest
backtest run -d data/AAPL.csv -s strategies/my_strategy.yaml -o results/

# With custom capital and commission
backtest run -d data/AAPL.csv -s strategies/my_strategy.yaml -o results/ \
  --initial-capital 50000 --commission 0.002

# Generate HTML report
backtest run -d data/AAPL.csv -s strategies/my_strategy.yaml -o results/ --html

# Export results as JSON
backtest run -d data/AAPL.csv -s strategies/my_strategy.yaml -o results/ --json
```

### Strategy Optimization

```bash
# Grid search optimization
backtest optimize -d data/AAPL.csv -s strategies/rsi.yaml \
  -p params/rsi_params.yaml -o optimization/

# Random search with custom objective
backtest optimize -d data/AAPL.csv -s strategies/rsi.yaml \
  -p params/rsi_params.yaml -o optimization/ \
  --method random --objective total_return
```

### Live Monitoring

```bash
# Start monitoring dashboard
backtest monitor -d data/AAPL.csv -s strategies/my_strategy.yaml --port 8050

# Then open http://localhost:8050 in your browser
```

## Python API Usage

### Basic Backtest

```python
import asyncio
from src.data import StockDataFetcher
from src.indicators import RSI, BollingerBands
from src.strategies import StrategyBuilder
from src.backtesting import BacktestEngine
from src.visualization import ChartBuilder

async def main():
    # 1. Fetch data
    fetcher = StockDataFetcher()
    data = await fetcher.fetch("AAPL", "2023-01-01", "2023-12-31")
    
    # 2. Calculate indicators
    data['rsi'] = RSI(14).calculate(data)
    bb_data = BollingerBands(20).calculate(data)
    data = data.join(bb_data)
    
    # 3. Build strategy
    builder = StrategyBuilder("RSI Mean Reversion")
    builder.add_entry_rule("rsi < 30 and close < bb_lower")
    builder.add_exit_rule("rsi > 70")
    builder.set_risk_management(stop_loss=0.05, take_profit=0.10)
    
    strategy = builder.build()
    
    # 4. Run backtest
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run(data, strategy)
    
    # 5. Display results
    print(f"Total Return: {results['performance']['total_return']:.2f}%")
    print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['performance']['max_drawdown']:.2f}%")
    print(f"Win Rate: {results['performance']['win_rate']:.2f}%")
    
    # 6. Create visualization
    chart = ChartBuilder("RSI Mean Reversion Results")
    chart.add_price_chart(data, results['trades'])
    chart.add_indicator("RSI", data['rsi'], subplot=True)
    chart.add_equity_curve(results['equity_curve'])
    chart.show()

# Run the backtest
asyncio.run(main())
```

### Using Meta Indicators

```python
import asyncio
from src.indicators.meta_indicators import FearGreedIndex, MaxPain
from src.data import StockDataFetcher
from src.strategies import StrategyBuilder

async def meta_strategy():
    # Fetch market data
    fetcher = StockDataFetcher()
    data = await fetcher.fetch("SPY", "2023-01-01", "2023-12-31")
    
    # Get Fear & Greed Index
    fgi = FearGreedIndex()
    fear_greed = await fgi.fetch_historical(limit=365)
    
    # Merge with price data
    data = data.merge(
        fear_greed.set_index('timestamp')[['value']], 
        left_index=True, 
        right_index=True, 
        how='left'
    )
    data['fear_greed'] = data['value'].fillna(method='ffill')
    
    # Build contrarian strategy
    builder = StrategyBuilder("Fear & Greed Contrarian")
    builder.add_entry_rule("fear_greed < 25")  # Extreme fear
    builder.add_exit_rule("fear_greed > 75")   # Extreme greed
    
    strategy = builder.build()
    
    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run(data, strategy)
    
    return results

# Run the meta strategy
results = asyncio.run(meta_strategy())
```

## Creating Your First Strategy

### Strategy YAML Format

Create a file `my_strategy.yaml`:

```yaml
name: "My First Strategy"
version: "1.0"

# Define indicators to use
indicators:
  - name: rsi
    type: RSI
    params:
      period: 14
  
  - name: sma_fast
    type: SMA
    params:
      period: 10
  
  - name: sma_slow
    type: SMA
    params:
      period: 30

# Entry conditions (all must be true)
entry_rules:
  - condition: "rsi < 30"
    logic: AND
  - condition: "sma_fast > sma_slow"
    logic: AND

# Exit conditions (any can trigger exit)
exit_rules:
  - condition: "rsi > 70"
    logic: OR
  - condition: "sma_fast < sma_slow"
    logic: OR

# Risk management
risk_management:
  stop_loss: 0.02      # 2% stop loss
  take_profit: 0.05    # 5% take profit
  position_size: 0.25  # Use 25% of capital per trade
```

### Programmatic Strategy Creation

```python
from src.strategies import StrategyBuilder

# Create strategy programmatically
builder = StrategyBuilder("My Strategy")

# Add multiple entry conditions
builder.add_entry_rule("rsi < 30")
builder.add_entry_rule("volume > volume.rolling(20).mean() * 1.5", logic="AND")
builder.add_entry_rule("close > open", logic="AND")

# Add exit conditions
builder.add_exit_rule("rsi > 70")
builder.add_exit_rule("close < sma_20", logic="OR")

# Set risk parameters
builder.set_risk_management(
    stop_loss=0.03,
    take_profit=0.06,
    trailing_stop=0.02,
    position_size="kelly"  # Use Kelly Criterion
)

# Add filters
builder.add_filter("sma_50 > sma_200")  # Only trade in uptrend

strategy = builder.build()
```

## Understanding Results

### Performance Metrics

After running a backtest, you'll see metrics like:

- **Total Return**: Overall percentage gain/loss
- **Annual Return**: Annualized return
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profit / gross loss
- **Average Trade**: Average return per trade

### Reading the HTML Report

The HTML report includes:

1. **Summary Statistics**: Key performance metrics
2. **Equity Curve**: Portfolio value over time
3. **Drawdown Chart**: Underwater equity chart
4. **Monthly Returns**: Heatmap of returns by month
5. **Trade Analysis**: Detailed trade-by-trade breakdown
6. **Price Chart**: Entry/exit points on price chart

### Analyzing Results

```python
# Access detailed results
print(f"Number of trades: {len(results['trades'])}")
print(f"Best trade: {max(t.pnl_pct for t in results['trades']):.2%}")
print(f"Worst trade: {min(t.pnl_pct for t in results['trades']):.2%}")
print(f"Average trade duration: {results['performance']['avg_trade_duration']}")

# Get trade statistics
winning_trades = [t for t in results['trades'] if t.pnl > 0]
losing_trades = [t for t in results['trades'] if t.pnl < 0]

print(f"Winning trades: {len(winning_trades)}")
print(f"Losing trades: {len(losing_trades)}")
print(f"Largest win: ${max(t.pnl for t in winning_trades):.2f}")
print(f"Largest loss: ${min(t.pnl for t in losing_trades):.2f}")
```

## Next Steps

### 1. Explore More Indicators

Check out the available indicators:

```bash
backtest indicators
```

See the [Indicators Guide](INDICATORS.md) for detailed documentation.

### 2. Learn Strategy Development

Read the [Strategy Development Guide](STRATEGY_DEVELOPMENT.md) to learn:
- Advanced entry/exit rules
- Position sizing techniques
- Market regime filters
- Multi-timeframe strategies

### 3. Optimize Your Strategies

See the [Optimization Guide](OPTIMIZATION_GUIDE.md) for:
- Parameter optimization
- Walk-forward analysis
- Overfitting prevention
- Performance improvement tips

### 4. Use Advanced Features

- **Meta Indicators**: Sentiment analysis, insider trading, options flow
- **Live Monitoring**: Real-time strategy performance tracking
- **Custom Indicators**: Build your own technical indicators
- **Event-Driven Strategies**: React to news and events

### 5. Join the Community

- Report issues on [GitHub](https://github.com/FRST-FRYTL/Backtest_Suite/issues)
- Share your strategies in the examples folder
- Contribute new features via pull requests

## Troubleshooting

If you encounter issues:

1. Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Verify Python version: `python --version` (should be 3.8+)
4. Check the logs in `logs/backtest.log`

Happy backtesting! 🚀