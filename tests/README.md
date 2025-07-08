# Backtest Suite Test Documentation

## Overview

This directory contains a comprehensive test suite for the Backtest Suite project, ensuring code quality, reliability, and performance across all components.

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and test configuration
├── test_portfolio.py              # Portfolio management tests
├── test_backtester.py            # Backtesting engine tests
├── test_data.py                  # Data fetching and caching tests
├── test_performance_benchmarks.py # Performance benchmark tests
├── indicators/
│   └── test_all_indicators.py    # Technical indicator tests
├── integration/
│   └── test_end_to_end.py       # End-to-end integration tests
└── README.md                     # This file
```

## Test Categories

### 1. Unit Tests

#### Portfolio Management (`test_portfolio.py`)
- **Portfolio Initialization**: Tests for portfolio creation with various parameters
- **Order Management**: Order placement, validation, and cancellation
- **Order Execution**: Market, limit, and stop order execution logic
- **Position Management**: Position tracking, averaging, and closing
- **Portfolio Valuation**: P&L calculations and portfolio metrics
- **Position Sizing**: Fixed, risk-based, and Kelly criterion sizing
- **Risk Management**: Stop loss, take profit, and portfolio heat
- **Performance Tracking**: Trade statistics and return calculations

#### Backtesting Engine (`test_backtester.py`)
- **Engine Initialization**: Configuration and parameter validation
- **Event System**: Event queue and processing order
- **Signal Generation**: Strategy signal creation and validation
- **Order Management**: Order generation from signals
- **Risk Management**: Stop loss and take profit execution
- **Performance Metrics**: Metric calculation accuracy
- **Edge Cases**: Data gaps, extreme movements, and error handling

#### Data Module (`test_data.py`)
- **Cache Management**: Storage, retrieval, and expiration
- **Data Fetching**: Stock, options, and fundamental data
- **Data Quality**: Cleaning and validation
- **Performance**: Caching efficiency and concurrent fetching
- **Error Handling**: Network errors and invalid data

### 2. Component Tests

#### Technical Indicators (`indicators/test_all_indicators.py`)
- **RSI**: Calculation accuracy and signal generation
- **Bollinger Bands**: Band calculations and pattern detection
- **VWAP**: Rolling and anchored VWAP
- **TSV**: Time segmented volume analysis
- **VWMA Bands**: Volume-weighted moving average bands
- **Meta Indicators**: Fear/Greed Index, Insider Trading, Max Pain

### 3. Integration Tests (`integration/test_end_to_end.py`)
- **Complete Workflows**: Data fetch → Strategy → Backtest → Results
- **Multi-Symbol Portfolios**: Portfolio backtesting across assets
- **Optimization Workflows**: Parameter optimization and walk-forward analysis
- **Real-Time Simulation**: High-frequency trading simulation
- **Visualization Integration**: Chart and report generation
- **Monitoring Integration**: Live monitoring and alerts

### 4. Performance Tests (`test_performance_benchmarks.py`)
- **Backtesting Performance**: Speed with various data sizes
- **Indicator Performance**: Calculation speed benchmarks
- **Optimization Performance**: Parameter search efficiency
- **Memory Efficiency**: Memory usage and leak detection
- **Concurrency Performance**: Parallel execution benefits

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/test_portfolio.py tests/test_backtester.py tests/test_data.py -v

# Integration tests only
pytest tests/integration/ -v

# Performance tests only
pytest tests/test_performance_benchmarks.py -v

# Indicator tests only
pytest tests/indicators/ -v
```

### Run Tests for Specific Component
```bash
# Portfolio tests
pytest tests/test_portfolio.py -v

# Backtesting engine tests
pytest tests/test_backtester.py::TestBacktestEngine -v

# Specific test method
pytest tests/test_portfolio.py::TestOrderExecution::test_market_buy_execution -v
```

## Test Configuration

### pytest.ini Configuration
The project uses `pyproject.toml` for pytest configuration:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=src --cov-report=html --cov-report=term-missing"
```

### Coverage Configuration
```toml
[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/test_*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

## Test Fixtures

Key fixtures provided in `conftest.py`:

### Data Generation
- `sample_ohlcv_data`: Generate sample OHLCV market data
- `multi_symbol_data`: Generate data for multiple symbols
- `sample_options_data`: Generate options chain data
- `performance_benchmark_data`: Large dataset for performance testing

### Component Fixtures
- `sample_portfolio`: Pre-configured portfolio instance
- `sample_strategy`: Basic trading strategy
- `backtest_engine`: Configured backtest engine
- `cache_manager`: Cache manager with temporary directory

### Mock Fixtures
- `mock_data_fetcher`: Mock data fetcher for testing
- `mock_yfinance_data`: Mock yfinance responses
- `mock_redis_client`: Mock Redis client

### Helper Fixtures
- `test_helpers`: Common assertion helpers
- `performance_monitor`: Performance tracking utility
- `async_mock_factory`: Factory for creating async mocks

## Writing New Tests

### Test Structure Template
```python
class TestNewComponent:
    """Test suite for new component."""
    
    def test_basic_functionality(self):
        """Test basic component functionality."""
        # Arrange
        component = NewComponent()
        
        # Act
        result = component.do_something()
        
        # Assert
        assert result == expected_value
    
    def test_edge_case(self):
        """Test component edge cases."""
        # Test boundary conditions
        pass
    
    def test_error_handling(self):
        """Test component error handling."""
        with pytest.raises(ExpectedError):
            component.do_invalid_operation()
```

### Best Practices
1. **Use descriptive test names** that explain what is being tested
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Test one thing per test** for clarity
4. **Use fixtures** for common setup
5. **Mock external dependencies** (APIs, databases)
6. **Test edge cases** and error conditions
7. **Keep tests fast** - mock slow operations
8. **Use parametrize** for testing multiple inputs

## Coverage Goals

Target coverage metrics:
- **Overall Coverage**: >80%
- **Core Components**: >90% (backtesting engine, portfolio)
- **Critical Paths**: 100% (order execution, risk management)
- **Integration Tests**: Cover all major workflows

## CI/CD Integration

Tests run automatically on:
- Every push to main/develop branches
- All pull requests
- Daily scheduled runs (for integration tests)

GitHub Actions workflow includes:
- Multi-OS testing (Ubuntu, Windows, macOS)
- Multiple Python versions (3.9-3.12)
- Coverage reporting to Codecov
- Performance regression detection
- Security scanning

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -e .  # Install package in development mode
   ```

2. **Async Test Failures**
   ```python
   @pytest.mark.asyncio  # Mark async tests
   async def test_async_function():
       result = await async_function()
   ```

3. **Fixture Not Found**
   - Check fixture is in conftest.py or imported
   - Verify fixture scope matches test needs

4. **Slow Tests**
   - Use mocks for external services
   - Reduce data size in tests
   - Use pytest-timeout for hanging tests

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain or improve coverage
4. Update this documentation
5. Run performance benchmarks for critical paths