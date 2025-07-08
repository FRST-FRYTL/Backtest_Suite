# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Backtest Suite is a comprehensive backtesting tool for stock trading that combines technical indicators with meta indicators to evaluate trading strategies. The project is currently in the planning phase with implementation about to begin.

## Key Technical Decisions

### Architecture
- **Event-driven backtesting engine**: All trades and signals are processed as events for realistic execution simulation
- **Modular design**: Separate modules for data, indicators, strategies, backtesting, and visualization
- **Strategy as configuration**: Trading strategies should be definable via YAML/JSON configuration files
- **Vectorized calculations**: Use pandas/numpy operations for performance

### Data Pipeline
1. Data fetching (yfinance) → Local cache → Preprocessing
2. Indicator calculation → Signal generation → Strategy execution
3. Position management → Performance analysis → Visualization

### Performance Requirements
- Backtesting 1 year of daily data should complete in <1 second
- Memory-efficient processing for large datasets
- Parallel optimization for strategy parameters

## Development Commands

Since the project is in planning phase, these commands will be established as development begins:
- **Dependencies**: Will use `requirements.txt` for Python packages
- **Testing**: Pytest framework planned for unit and integration tests
- **Linting**: Consider using `ruff` or `flake8` for Python code quality
- **Type checking**: Consider using `mypy` for type safety

## Core Components to Implement

### Technical Indicators (Priority: High)
- RSI, VWMA bands, Bollinger bands, TSV
- VWAP (rolling and anchored) with standard deviations
- All indicators should support configurable parameters

### Meta Indicators (Priority: Medium)
- Fear and Greed Index integration (API)
- Insider trading data scraper (openinsider.com)
- Max pain options calculations

### Backtesting Engine (Priority: High)
- Event-driven architecture with realistic order execution
- Support for multiple positions, stop-loss, take-profit
- Commission and slippage modeling

## Code Standards

### File Organization
```
src/
├── data/          # Data fetching and caching
├── indicators/    # Technical and meta indicators
├── strategies/    # Strategy definitions and builder
├── backtesting/   # Core backtesting engine
├── visualization/ # Charts and performance dashboards
└── utils/         # Shared utilities
```

### Indicator Implementation Pattern
- Each indicator should be a separate class
- Support method chaining for easy composition
- Return pandas Series/DataFrame for compatibility
- Include parameter validation

### Strategy Builder Requirements
- Support logical operators (AND, OR, NOT) for signals
- Allow dynamic position sizing
- Include risk management parameters
- Enable easy backtesting of multiple strategies

## External Data Sources

### APIs to Integrate
- **yfinance**: Stock price data
- **Fear and Greed Index**: Market sentiment data
- **Options data**: For max pain calculations

### Web Scraping
- **openinsider.com**: Use BeautifulSoup with proper caching
- Implement retry logic and rate limiting
- Store scraped data locally to minimize requests

## Testing Strategy
- Unit tests for each indicator with known outputs
- Integration tests for full strategy backtests
- Mock external data sources for consistent testing
- Aim for >80% test coverage

## Performance Optimization
- Use vectorized pandas operations over loops
- Implement data caching to avoid repeated API calls
- Consider numba/cython for computationally intensive indicators
- Profile code to identify bottlenecks