# Backtest Suite

A comprehensive Python-based backtesting framework for quantitative trading strategies with advanced indicators, optimization tools, and visualization capabilities.

## Features

### 📊 Technical Indicators
- **RSI** - Relative Strength Index with divergence detection
- **Bollinger Bands** - With pattern recognition (W-bottom, M-top)
- **VWMA Bands** - Volume-weighted moving average with bands
- **TSV** - Time Segmented Volume for money flow analysis
- **VWAP** - Rolling and anchored VWAP with standard deviation bands

### 🧠 Meta Indicators
- **Fear & Greed Index** - Market sentiment analysis from Alternative.me
- **Insider Trading** - Real-time scraping from OpenInsider with sentiment scoring
- **Max Pain** - Options-based support/resistance levels from live options chains

### 🚀 Core Features
- **Event-driven backtesting engine** with realistic execution simulation
- **Strategy builder** with rule-based framework and logical operators
- **Portfolio management** with position tracking and risk controls
- **Performance analytics** including Sharpe ratio, drawdown analysis, and more
- **Strategy optimization** using grid search, random search, and differential evolution
- **Walk-forward analysis** for robust parameter selection
- **Interactive visualizations** with Plotly dashboards
- **Live monitoring** with real-time performance tracking
- **CLI interface** for easy command-line usage
- **Comprehensive test suite** with 90%+ code coverage

## Installation

```bash
# Clone the repository
git clone https://github.com/FRST-FRYTL/Backtest_Suite.git
cd Backtest_Suite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### CLI Usage

```bash
# Fetch stock data
backtest fetch -s AAPL -S 2023-01-01 -E 2023-12-31 -o data/AAPL.csv

# Run a backtest
backtest run -d data/AAPL.csv -s examples/strategies/rsi_mean_reversion.yaml -o results/

# Optimize strategy parameters
backtest optimize -d data/AAPL.csv -s examples/strategies/rsi_mean_reversion.yaml -p examples/parameter_optimization.yaml -o optimization_results/
```

### Python Usage

```python
import asyncio
from src.data import StockDataFetcher
from src.indicators import RSI, BollingerBands
from src.strategies import StrategyBuilder
from src.backtesting import BacktestEngine

async def run_backtest():
    # Fetch data
    fetcher = StockDataFetcher()
    data = await fetcher.fetch("AAPL", "2023-01-01", "2023-12-31")
    
    # Calculate indicators
    data['rsi'] = RSI(14).calculate(data)
    bb_data = BollingerBands(20).calculate(data)
    data = data.join(bb_data)
    
    # Build strategy
    builder = StrategyBuilder("RSI Mean Reversion")
    builder.add_entry_rule("rsi < 30 and close < bb_lower")
    builder.add_exit_rule("rsi > 70")
    builder.set_risk_management(stop_loss=0.05, take_profit=0.10)
    
    strategy = builder.build()
    
    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run(data, strategy)
    
    print(f"Total Return: {results['performance']['total_return']:.2f}%")
    print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")

asyncio.run(run_backtest())
```

## Documentation

- [Quick Start Guide](docs/QUICK_START.md) - Get up and running quickly
- [API Reference](docs/API_REFERENCE.md) - Comprehensive API documentation
- [Strategy Development](docs/STRATEGY_DEVELOPMENT.md) - Complete guide to building strategies
- [Indicators Guide](docs/INDICATORS.md) - All available indicators
- [Optimization Guide](docs/OPTIMIZATION_GUIDE.md) - Parameter optimization techniques
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions
- [Monitoring Guide](docs/MONITORING.md) - Live monitoring setup

## Examples

Check out the [examples/](examples/) directory for:
- Complete backtesting workflows
- Strategy YAML configurations
- Parameter optimization setups
- Custom indicator implementations

## Project Structure

```
Backtest_Suite/
├── src/
│   ├── data/           # Data fetching and management
│   ├── indicators/     # Technical and meta indicators
│   │   ├── __init__.py
│   │   ├── technical.py    # RSI, Bollinger Bands, etc.
│   │   └── meta_indicators.py  # Fear & Greed, Insider Trading, Max Pain
│   ├── strategies/     # Strategy framework
│   ├── backtesting/    # Event-driven backtesting engine
│   ├── portfolio/      # Portfolio and position management
│   ├── utils/          # Performance metrics and utilities
│   ├── visualization/  # Plotly charts and dashboards
│   ├── optimization/   # Grid, random, and DE optimization
│   ├── monitoring/     # Live performance monitoring
│   └── cli/           # Command-line interface
├── tests/             # Comprehensive test suite (90%+ coverage)
├── docs/              # Complete documentation
├── examples/          # Example strategies and notebooks
│   ├── strategies/    # YAML strategy configurations
│   ├── notebooks/     # Jupyter notebook tutorials
│   └── scripts/       # Python script examples
├── .github/           # CI/CD workflows
└── data/              # Data storage (gitignored)
```

## Testing

The Backtest Suite includes a comprehensive test suite with high code coverage:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html tests/

# Run specific test module
pytest tests/test_indicators.py -v

# Run only fast tests
pytest -m "not slow"

# Run with parallel execution
pytest -n auto
```

Test categories:
- Unit tests for all modules
- Integration tests for strategies
- Performance benchmarks
- Edge case handling

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Performance

The Backtest Suite is optimized for performance:

- **Vectorized operations** using NumPy and Pandas
- **Async data fetching** for parallel downloads
- **Efficient memory usage** with data type optimization
- **Multi-core support** for optimization tasks
- **Caching** for frequently accessed data

Typical performance benchmarks:
- Backtest 10 years of daily data: < 1 second
- Optimize 1000 parameter combinations: < 1 minute (with parallel processing)
- Real-time monitoring latency: < 100ms

## Acknowledgments

- Built with modern Python libraries: pandas, numpy, asyncio, FastAPI
- Visualization powered by Plotly and matplotlib
- CLI interface using Click and Rich
- Options data from Yahoo Finance
- Fear & Greed Index from Alternative.me
- Insider trading data from OpenInsider
- Testing with pytest and coverage.py
- CI/CD with GitHub Actions