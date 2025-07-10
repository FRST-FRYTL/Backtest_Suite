# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Backtest Suite - Quantitative Trading Framework

A comprehensive Python-based backtesting framework for quantitative trading strategies with ML integration, advanced indicators, and visualization capabilities.

### Core Architecture

The codebase follows an event-driven architecture with these key components:

1. **Data Layer** (`src/data/`): Handles data fetching, caching, and preprocessing
   - `StockDataFetcher` for async data downloads
   - Historical data stored in `data/` directory (gitignored)
   - Supports multiple timeframes and data sources

2. **Indicators** (`src/indicators/`): Technical and meta indicators
   - Technical: RSI, Bollinger Bands, VWAP, MACD, ATR, etc.
   - Meta: Fear & Greed Index, Insider Trading, Max Pain
   - All indicators follow a consistent `calculate()` interface

3. **ML Models** (`src/ml/`): Machine learning integration
   - `DirectionPredictor` (XGBoost-based)
   - `VolatilityForecaster` (LSTM with attention)
   - `MarketRegimeDetector` (5 regime states)
   - `EnsembleModel` for model combination
   - Feature engineering pipeline with 60+ features

4. **Backtesting Engine** (`src/backtesting/`): Event-driven simulation
   - Realistic order execution with slippage
   - Position tracking and portfolio management
   - Performance metrics calculation

5. **Visualization** (`src/visualization/`): Plotly-based dashboards
   - HTML report generation
   - Interactive charts and performance metrics
   - Real-time monitoring capabilities

### Common Commands

```bash
# Environment setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run tests
pytest                              # Run all tests
pytest tests/test_indicators.py -v  # Run specific test file
pytest --cov=src --cov-report=html # With coverage report
pytest -n auto                      # Parallel execution

# Code quality
black src/                          # Format code
flake8 src/                        # Lint code
mypy src/                          # Type checking

# CLI usage
backtest fetch -s AAPL -S 2023-01-01 -E 2023-12-31  # Fetch data
backtest run -d data/AAPL.csv -s strategy.yaml      # Run backtest
backtest optimize -d data/AAPL.csv -s strategy.yaml # Optimize params

# Preview HTML reports (in Codespaces)
./preview-reports.sh               # Start preview server
# Or: cd reports && python -m http.server 8000
```

### Development Workflow

1. **Testing**: Always run tests before committing
   - Unit tests for individual components
   - Integration tests in `tests/integration/`
   - Performance benchmarks in `test_performance_benchmarks.py`

2. **Data Management**:
   - Historical data cached in `data/` (not in git)
   - Use `download_historical_data.py` for bulk downloads
   - Data format: OHLCV with timezone-aware timestamps

3. **ML Model Development**:
   - Models in `src/ml/models/`
   - Feature engineering in `src/ml/feature_engineering.py`
   - Report generation in `src/ml/report_generator.py`

4. **Strategy Implementation**:
   - Strategy configs in YAML format
   - Examples in `examples/strategies/`
   - Use `StrategyBuilder` for programmatic creation

### Key Configuration Files

- `config/strategy_config.yaml`: Trading parameters and asset lists
- `pyproject.toml`: Python project configuration
- `.vscode/settings.json`: VS Code configuration for Live Server
- `requirements.txt`: Python dependencies

### Important Patterns

1. **Async Operations**: Data fetching uses asyncio for parallel downloads
2. **Vectorized Calculations**: Indicators use NumPy/Pandas for performance
3. **Event-Driven**: Backtesting engine processes market events sequentially
4. **Memory Efficiency**: Large datasets processed in chunks when needed

## Claude Flow Integration

### Quick Setup (Stdio MCP)
```bash
# Add Claude Flow MCP server
claude mcp add claude-flow npx claude-flow mcp start
```

### Swarm Coordination for Backtest Suite

When working on this codebase with swarms:

1. **For Strategy Development**:
   - Use `hierarchical` topology with specialized agents
   - Spawn: architect, coder (ML), coder (indicators), analyst, tester
   - Coordinate feature engineering and model training tasks

2. **For Performance Testing**:
   - Use `parallel` strategy for multi-asset backtests
   - Batch all data downloads and indicator calculations
   - Generate reports in parallel for different strategies

3. **For ML Integration**:
   - Use researcher agents for hyperparameter search
   - Coordinate model training across different configurations
   - Use memory to track best performing models

### Parallel Execution Pattern

When implementing features or running tests:

```javascript
// CORRECT - Batch operations
[Single Message]:
  Read("src/indicators/technical_indicators.py")
  Read("src/ml/models/xgboost_direction.py")
  Read("config/strategy_config.yaml")
  Bash("pytest tests/test_indicators.py -v")
  Bash("python examples/ml_integration_example.py")
  Write("reports/results.json", results)
```

### Memory Coordination

Store important findings:
```javascript
mcp__claude-flow__memory_usage {
  action: "store",
  key: "backtest/results/strategy_name",
  value: { sharpe: 1.89, return: 35.67, drawdown: -9.87 }
}
```

## Performance Considerations

- Backtest 10 years daily data: < 1 second
- ML model training: Use cached features when possible
- Report generation: Batch HTML creation for efficiency
- Data downloads: Use async fetching for multiple symbols

## Testing Requirements

- Maintain >90% test coverage
- Run tests before any PR
- Add tests for new indicators/strategies
- Use pytest fixtures from `conftest.py`

Remember: **Claude Flow coordinates, Claude Code creates!**