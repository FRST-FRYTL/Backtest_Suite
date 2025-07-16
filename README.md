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
- **Enhanced trade reporting** with entry/exit prices, stop losses, and risk analysis
- **Professional report generation** with HTML dashboards, markdown docs, and JSON exports
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

# Run batch backtests
backtest batch -s SPY,QQQ,AAPL -S strategies/ -c 100000 -o batch_results/

# List available indicators
backtest indicators
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

## Enhanced Trade Reporting

The Backtest Suite now includes comprehensive trade-level reporting with detailed price analysis:

### 📊 Trade Analysis Features
- **Entry/Exit Prices**: Exact execution prices for every trade
- **Stop Loss Analysis**: Stop placement effectiveness and hit rates
- **Risk Per Trade**: Position sizing and risk management analysis
- **Trade Duration**: Precise timing and holding period analysis
- **Slippage Tracking**: Execution quality measurement
- **MAE/MFE Analysis**: Maximum adverse/favorable excursion tracking

### 📈 Professional Reports
- **Interactive HTML Dashboards**: Professional-grade visualizations
- **Detailed Trade Tables**: Complete price and timing information
- **Risk Analysis Charts**: Stop loss effectiveness and risk distribution
- **Performance Metrics**: Comprehensive statistics and benchmarks
- **Multi-Format Export**: HTML, Markdown, and JSON outputs

### 🎯 Configuration Example
```python
from src.reporting.report_config import ReportConfig, TradeReportingConfig

trade_config = TradeReportingConfig(
    enable_detailed_trade_prices=True,
    show_stop_loss_prices=True,
    enable_risk_per_trade_analysis=True,
    include_mae_mfe_analysis=True
)

config = ReportConfig(
    title="My Strategy Report",
    trade_reporting=trade_config,
    output_formats=["html", "markdown", "json"]
)
```

## Documentation

- [Quick Start Guide](docs/QUICK_START.md) - Get up and running quickly
- [API Reference](docs/API_REFERENCE.md) - Comprehensive API documentation
- [Strategy Development](docs/STRATEGY_DEVELOPMENT.md) - Complete guide to building strategies
- [Enhanced Trade Reporting](docs/ENHANCED_TRADE_REPORTING.md) - Detailed trade analysis features
- [Standardized Reporting](docs/STANDARDIZED_REPORTING.md) - Professional report generation
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
- Enhanced trade reporting demonstrations
- Professional report generation examples

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
│   ├── reporting/      # Enhanced trade reporting system
│   │   ├── standard_report_generator.py  # Main report generator
│   │   ├── report_config.py              # Configuration and themes
│   │   ├── report_sections.py            # Modular report sections
│   │   ├── visualization_types.py        # Professional charts
│   │   └── templates/                    # HTML and markdown templates
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
│   ├── scripts/       # Python script examples
│   └── demo_enhanced_trade_reporting.py  # Enhanced reporting demo
├── reports/           # Generated reports and analysis
│   ├── comprehensive_visualizations/     # Interactive dashboards
│   └── enhanced_trade_demo/              # Trade analysis examples
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

## Comprehensive Command Reference

### CLI Commands

#### Data Management
```bash
# Fetch single stock data
backtest fetch -s AAPL -S 2023-01-01 -E 2023-12-31 -o data/AAPL.csv

# Fetch with specific interval (1m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
backtest fetch -s SPY -S 2023-01-01 -E 2023-12-31 -i 1h -o data/SPY_hourly.csv

# Bulk download multiple assets (5 years of data)
python download_data.py

# Export daily data in specific format
python export_daily_data.py
```

#### Backtesting
```bash
# Basic backtest
backtest run -d data/AAPL.csv -s examples/strategies/rsi_mean_reversion.yaml -o results/

# Backtest with custom capital and commission
backtest run -d data/SPY.csv -s strategy.yaml -c 50000 --commission 0.001 -o results/

# Batch backtest multiple symbols and strategies
backtest batch -s SPY,QQQ,AAPL,MSFT -S examples/strategies/ -c 100000 -o batch_results/

# Run without generating charts
backtest run -d data/AAPL.csv -s strategy.yaml --no-chart -o results/
```

#### Strategy Optimization
```bash
# Grid search optimization
backtest optimize -d data/SPY.csv -s strategy.yaml -p params.yaml -m grid -o opt_results/

# Random search optimization
backtest optimize -d data/SPY.csv -s strategy.yaml -p params.yaml -m random -o opt_results/

# Differential evolution optimization
backtest optimize -d data/SPY.csv -s strategy.yaml -p params.yaml -m differential -o opt_results/

# Optimize with custom metric (sharpe, total_return, calmar, sortino)
backtest optimize -d data/SPY.csv -s strategy.yaml -p params.yaml --metric sortino -o opt_results/
```

#### Information Commands
```bash
# List all available indicators
backtest indicators
```

### Development Tools

#### Code Quality
```bash
# Format code with Black
black src/ tests/ examples/

# Lint code with Flake8
flake8 src/ tests/

# Type checking with MyPy
mypy src/

# Sort imports
isort src/ tests/

# Run pre-commit hooks
pre-commit run --all-files
```

#### Testing
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_indicators.py -v

# Run tests in parallel
pytest -n auto

# Run only fast tests
pytest -m "not slow"

# Run integration tests
pytest tests/integration/ -v

# Run performance benchmarks
pytest tests/test_performance_benchmarks.py -v
```

### Data Download Scripts

```bash
# Download 5 years of historical data for multiple assets
python download_data.py

# Alternative data download
python run_data_download.py

# Export data in specific format
python export_daily_data.py
```

### ML and Feature Engineering

```bash
# ML integration examples
python examples/ml_integration_example.py
python examples/ml_optimization_example.py
python examples/ml_backtest_example.py

# Feature engineering
python examples/feature_engineering_example.py
python test_feature_engineering_pipeline.py

# ML model testing
python test_ml_models.py
python test_ml_simple.py

# ML report generation
python test_ml_reports.py
python test_all_ml_reports.py
```

### Strategy Execution

```bash
# Run all strategies
python examples/strategies/execute_all_strategies.py
python examples/strategies/run_all_strategies.py

# Confluence strategy simulations
python run_confluence_simulation.py
python run_confluence_simulation_v2.py
python test_enhanced_confluence.py

# Specific strategy examples
python examples/strategies/rolling_vwap_strategy_example.py
python examples/strategies/monthly_contribution_research.py
python examples/strategies/contribution_timing_strategy.py
```

### Report Generation

```bash
# Generate comprehensive performance report
python generate_comprehensive_report.py

# Generate backtest results
python generate_backtest_results.py

# Generate performance analysis reports
python generate_performance_reports.py

# Generate enhanced trade report with detailed price analysis
python examples/demo_enhanced_trade_reporting.py

# Generate standard report from backtest results
python generate_standard_report.py backtest_results.json

# Generate strategy reports
python examples/reports/generate_report.py
python examples/reports/monthly_contribution_strategy_report.py
python examples/reports/strategy_summary_visual.py
python examples/reports/strategy_dashboard.py

# Test standard reporting system
python examples/test_standard_reporting.py
```

### Visualization and Monitoring

```bash
# Start report preview server (Codespaces/local)
./preview-reports.sh
# Or manually:
cd reports && python -m http.server 8000

# Run monitoring examples
python examples/monitoring_example.py
python examples/backtest_with_monitoring.py

# Phase demos
python examples/phase2_advanced_analytics_demo.py
python examples/phase3_visualization_demo.py
python examples/phase4_portfolio_risk_demo.py
```

### Validation and Testing

```bash
# Monte Carlo simulation
python examples/validation/monte_carlo_simulation.py

# Live paper trading simulation
python examples/validation/live_paper_trading.py

# Stress test scenarios
python examples/backtests/stress_test_scenarios.py

# Comprehensive test suite
python comprehensive_test_suite.py
python run_comprehensive_tests.py

# Test all indicators
python test_all_indicators.py
```

### Example Workflows

#### Complete Backtest Workflow
```bash
# 1. Download data
python download_data.py

# 2. Run backtest
backtest run -d data/SPY_1D_2020-01-01_2024-01-01.csv -s examples/strategies/rsi_mean_reversion.yaml -o results/

# 3. Optimize parameters
backtest optimize -d data/SPY_1D_2020-01-01_2024-01-01.csv -s examples/strategies/rsi_mean_reversion.yaml -p examples/parameter_optimization.yaml -o optimization/

# 4. Generate reports
python generate_comprehensive_report.py

# 5. Preview results
./preview-reports.sh
```

#### ML Integration Workflow
```bash
# 1. Test feature engineering
python test_feature_engineering_pipeline.py

# 2. Run ML backtest
python examples/ml_backtest_example.py

# 3. Generate ML reports
python test_all_ml_reports.py

# 4. Preview ML visualizations
cd reports && python -m http.server 8000
```

#### Development Workflow
```bash
# 1. Format code
black src/

# 2. Run tests
pytest -n auto

# 3. Check types
mypy src/

# 4. Generate coverage
pytest --cov=src --cov-report=html

# 5. Preview coverage report
cd htmlcov && python -m http.server 8001
```

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