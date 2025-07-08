# Quick Start Guide

This guide will help you get started with the Backtest Suite quickly.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/FRST-FRYTL/Backtest_Suite.git
cd Backtest_Suite
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

## Basic Usage

### 1. Using the CLI

The easiest way to get started is using the command-line interface:

```bash
# Fetch data
backtest fetch -s AAPL -S 2023-01-01 -E 2023-12-31 -o data/AAPL.csv

# Run a backtest with a strategy
backtest run -d data/AAPL.csv -s examples/strategies/rsi_mean_reversion.yaml -o results/

# View available indicators
backtest indicators
```

### 2. Python Script Example

```python
import asyncio
from src.data import StockDataFetcher
from src.indicators import RSI, BollingerBands
from src.strategies import StrategyBuilder
from src.backtesting import BacktestEngine

async def simple_backtest():
    # Fetch data
    fetcher = StockDataFetcher()
    data = await fetcher.fetch("AAPL", "2023-01-01", "2023-12-31")
    
    # Calculate indicators
    rsi = RSI(period=14)
    data['rsi'] = rsi.calculate(data)
    
    # Create strategy
    builder = StrategyBuilder("My Strategy")
    builder.add_entry_rule("rsi < 30")
    builder.add_exit_rule("rsi > 70")
    builder.set_risk_management(stop_loss=0.05)
    
    strategy = builder.build()
    
    # Run backtest
    engine = BacktestEngine()
    results = engine.run(data, strategy)
    
    print(f"Total Return: {results['performance']['total_return']:.2f}%")

asyncio.run(simple_backtest())
```

### 3. Creating a Strategy

Strategies can be defined in YAML:

```yaml
name: My RSI Strategy
description: Simple RSI mean reversion

entry_rules:
  - name: Oversold Entry
    conditions:
      - left: rsi
        operator: <
        right: 30

exit_rules:
  - name: Overbought Exit
    conditions:
      - left: rsi
        operator: >
        right: 70

risk_management:
  stop_loss: 0.05
  take_profit: 0.10
  max_positions: 3

position_sizing:
  method: percent
  size: 0.1
```

Or built programmatically:

```python
builder = StrategyBuilder("My Strategy")

# Add multiple conditions
entry_rule = Rule(operator=LogicalOperator.AND)
entry_rule.add_condition('rsi', '<', 30)
entry_rule.add_condition('close', '<', 'bb_lower')
builder.strategy.entry_rules.append(entry_rule)

# Configure risk management
builder.set_risk_management(
    stop_loss=0.05,
    take_profit=0.10,
    trailing_stop=0.03
)
```

## Available Indicators

### Technical Indicators
- **RSI**: Relative Strength Index
- **Bollinger Bands**: With pattern detection (W-bottom, M-top)
- **VWMA Bands**: Volume-weighted moving average with bands
- **TSV**: Time Segmented Volume
- **VWAP**: Rolling and anchored VWAP with standard deviation bands

### Meta Indicators
- **Fear & Greed Index**: Market sentiment indicator
- **Insider Trading**: Data from OpenInsider
- **Max Pain**: Options-based support/resistance levels

## Strategy Optimization

```bash
# Optimize parameters using grid search
backtest optimize \
    -d data/AAPL.csv \
    -s examples/strategies/rsi_mean_reversion.yaml \
    -p examples/parameter_optimization.yaml \
    -m grid \
    -o optimization_results/
```

## Batch Testing

Test multiple symbols and strategies:

```bash
backtest batch \
    -s AAPL,GOOGL,MSFT,AMZN \
    -S examples/strategies/ \
    -c 100000 \
    -o batch_results/
```

## Visualization

The backtesting engine automatically generates:
- Interactive HTML dashboard
- Equity curves
- Drawdown analysis
- Trade visualization
- Performance metrics

## Next Steps

1. Check out the [examples/](../examples/) directory for more examples
2. Read the [API documentation](API.md) for detailed usage
3. See [INDICATORS.md](INDICATORS.md) for indicator documentation
4. Learn about [strategy optimization](OPTIMIZATION.md)