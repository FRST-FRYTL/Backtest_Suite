# Troubleshooting Guide

This guide helps you resolve common issues when using the Backtest Suite.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Data Fetching Problems](#data-fetching-problems)
- [Indicator Errors](#indicator-errors)
- [Strategy Execution Issues](#strategy-execution-issues)
- [Performance Problems](#performance-problems)
- [Memory Issues](#memory-issues)
- [Visualization Errors](#visualization-errors)
- [Common Error Messages](#common-error-messages)
- [Debugging Tips](#debugging-tips)
- [Getting Help](#getting-help)

## Installation Issues

### Problem: pip install fails with dependency conflicts

**Solution:**
```bash
# Use a clean virtual environment
python -m venv venv_clean
source venv_clean/bin/activate  # On Windows: venv_clean\Scripts\activate

# Upgrade pip first
pip install --upgrade pip

# Install with specific versions
pip install -r requirements.txt --no-cache-dir
```

### Problem: ModuleNotFoundError after installation

**Solution:**
```bash
# Ensure package is installed in development mode
pip install -e .

# Verify installation
python -c "import src; print(src.__file__)"

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Problem: CLI command 'backtest' not found

**Solution:**
```bash
# Reinstall in development mode
pip uninstall backtest-suite
pip install -e .

# Alternative: use Python module directly
python -m src.cli.main --help
```

## Data Fetching Problems

### Problem: "No data found for symbol"

**Common causes and solutions:**

1. **Invalid symbol**
   ```python
   # Check if symbol exists
   import yfinance as yf
   ticker = yf.Ticker("AAPL")
   print(ticker.info)  # Should return company info
   ```

2. **Date range issues**
   ```python
   # Ensure dates are valid trading days
   from pandas.tseries.offsets import BDay
   start_date = pd.Timestamp('2023-01-01')
   # Adjust to next business day if needed
   if not pd.bdate_range(start_date, start_date):
       start_date += BDay(1)
   ```

3. **Network/API issues**
   ```python
   # Add retry logic
   async def fetch_with_retry(symbol, start, end, max_retries=3):
       for attempt in range(max_retries):
           try:
               data = await fetcher.fetch(symbol, start, end)
               return data
           except Exception as e:
               if attempt == max_retries - 1:
                   raise
               await asyncio.sleep(2 ** attempt)  # Exponential backoff
   ```

### Problem: "Data has gaps or missing values"

**Solution:**
```python
# Check for gaps
def check_data_quality(data):
    # Find gaps
    date_range = pd.bdate_range(data.index[0], data.index[-1])
    missing_dates = date_range.difference(data.index)
    
    if len(missing_dates) > 0:
        print(f"Missing {len(missing_dates)} trading days")
        
    # Check for NaN values
    nan_counts = data.isna().sum()
    if nan_counts.any():
        print("NaN values found:")
        print(nan_counts[nan_counts > 0])
    
    # Forward fill missing data
    data_filled = data.fillna(method='ffill')
    
    return data_filled

# Clean the data
clean_data = check_data_quality(raw_data)
```

### Problem: "Rate limit exceeded"

**Solution:**
```python
# Implement rate limiting
from asyncio import Semaphore
import time

class RateLimitedFetcher:
    def __init__(self, requests_per_minute=60):
        self.semaphore = Semaphore(requests_per_minute)
        self.last_request = 0
        self.min_interval = 60 / requests_per_minute
    
    async def fetch(self, symbol, start, end):
        async with self.semaphore:
            # Ensure minimum time between requests
            elapsed = time.time() - self.last_request
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            
            self.last_request = time.time()
            return await super().fetch(symbol, start, end)
```

## Indicator Errors

### Problem: "ValueError: Length of values does not match length of index"

**Solution:**
```python
# Ensure indicators return same length as input
class SafeIndicator:
    def calculate(self, data):
        result = pd.Series(index=data.index, dtype=float)
        
        # Calculate indicator (example: SMA)
        sma_values = data['close'].rolling(self.period).mean()
        
        # Align with original index
        result.loc[sma_values.index] = sma_values
        
        return result
```

### Problem: "KeyError: 'indicator_name' not in columns"

**Solution:**
```python
# Check if indicators are calculated
def verify_indicators(data, required_indicators):
    missing = [ind for ind in required_indicators if ind not in data.columns]
    
    if missing:
        print(f"Missing indicators: {missing}")
        
        # Calculate missing indicators
        if 'rsi' in missing:
            data['rsi'] = RSI(14).calculate(data)
        if 'bb_upper' in missing:
            bb_data = BollingerBands(20).calculate(data)
            data = data.join(bb_data)
    
    return data
```

### Problem: "RuntimeWarning: divide by zero encountered"

**Solution:**
```python
# Handle division by zero in indicators
def safe_divide(numerator, denominator, fill_value=0):
    """Safe division that handles zero denominator"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(numerator, denominator)
        result[~np.isfinite(result)] = fill_value
    return result

# Example: RSI calculation
def calculate_rsi_safe(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    
    # Safe division
    rs = safe_divide(gain, loss, fill_value=100)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi
```

## Strategy Execution Issues

### Problem: "No trades were executed"

**Debugging steps:**

```python
# 1. Check if entry conditions are ever met
def debug_entry_conditions(data, strategy):
    # Evaluate conditions separately
    conditions = {
        'rsi < 30': (data['rsi'] < 30).sum(),
        'close < bb_lower': (data['close'] < data['bb_lower']).sum(),
        'volume > avg': (data['volume'] > data['volume'].rolling(20).mean()).sum()
    }
    
    print("Individual condition frequencies:")
    for condition, count in conditions.items():
        print(f"  {condition}: {count} times ({count/len(data)*100:.1f}%)")
    
    # Check combined conditions
    all_conditions = (
        (data['rsi'] < 30) & 
        (data['close'] < data['bb_lower']) &
        (data['volume'] > data['volume'].rolling(20).mean())
    )
    print(f"\nAll conditions met: {all_conditions.sum()} times")
    
    if all_conditions.sum() == 0:
        print("\nSuggestion: Relax entry conditions or check indicator calculations")

# 2. Verify strategy logic
def test_strategy_logic(strategy, sample_data):
    # Create test scenarios
    test_cases = [
        {'rsi': 25, 'close': 100, 'bb_lower': 105},  # Should trigger
        {'rsi': 35, 'close': 100, 'bb_lower': 105},  # Should not trigger
    ]
    
    for i, test in enumerate(test_cases):
        sample_data.loc[i] = test
        should_enter = strategy.evaluate_entry(sample_data, i)
        print(f"Test case {i}: {test} -> Entry: {should_enter}")
```

### Problem: "Strategy enters and exits immediately"

**Solution:**
```python
# Add minimum holding period
class ImprovedStrategy(Strategy):
    def __init__(self, min_holding_bars=5, **kwargs):
        super().__init__(**kwargs)
        self.min_holding_bars = min_holding_bars
    
    def evaluate_exit(self, data, index, position):
        # Check minimum holding period
        bars_held = index - position.entry_index
        if bars_held < self.min_holding_bars:
            return False
        
        # Normal exit logic
        return super().evaluate_exit(data, index, position)

# Or add exit condition dampening
def add_exit_dampening(data, exit_signal, lookback=3):
    """Require exit signal to persist for multiple bars"""
    return exit_signal.rolling(lookback).sum() >= lookback
```

### Problem: "Position size is 0"

**Solution:**
```python
# Debug position sizing
def debug_position_sizing(capital, price, strategy_params):
    print(f"Capital: ${capital:,.2f}")
    print(f"Price: ${price:.2f}")
    
    # Fixed percentage
    if isinstance(strategy_params.get('position_size'), float):
        position_value = capital * strategy_params['position_size']
        shares = int(position_value / price)
        print(f"Position size: {strategy_params['position_size']*100}%")
        print(f"Position value: ${position_value:,.2f}")
        print(f"Shares: {shares}")
    
    # Risk-based sizing
    if 'stop_loss' in strategy_params:
        risk_per_trade = capital * 0.01  # 1% risk
        stop_loss_amount = price * strategy_params['stop_loss']
        shares = int(risk_per_trade / stop_loss_amount)
        print(f"Risk per trade: ${risk_per_trade:.2f}")
        print(f"Stop loss amount: ${stop_loss_amount:.2f}")
        print(f"Risk-based shares: {shares}")
    
    # Ensure minimum position size
    min_shares = max(1, int(capital * 0.01 / price))  # At least 1% or 1 share
    return max(shares, min_shares)
```

## Performance Problems

### Problem: "Backtest is running very slowly"

**Solutions:**

1. **Vectorize calculations**
   ```python
   # Slow: Loop-based
   signals = []
   for i in range(len(data)):
       if data['rsi'].iloc[i] < 30:
           signals.append(1)
       else:
           signals.append(0)
   
   # Fast: Vectorized
   signals = (data['rsi'] < 30).astype(int)
   ```

2. **Pre-calculate indicators**
   ```python
   # Calculate all indicators once before backtest
   def prepare_data(data):
       # Add all indicators
       indicators = {
           'rsi': RSI(14),
           'bb': BollingerBands(20),
           'sma_50': SMA(50),
           'sma_200': SMA(200)
       }
       
       for name, indicator in indicators.items():
           result = indicator.calculate(data)
           if isinstance(result, pd.DataFrame):
               data = data.join(result)
           else:
               data[name] = result
       
       return data
   
   # Prepare once
   prepared_data = prepare_data(raw_data)
   ```

3. **Use numba for critical functions**
   ```python
   from numba import jit
   
   @jit(nopython=True)
   def fast_moving_average(prices, period):
       ma = np.empty(len(prices))
       ma[:period-1] = np.nan
       
       for i in range(period-1, len(prices)):
           ma[i] = np.mean(prices[i-period+1:i+1])
       
       return ma
   ```

### Problem: "Optimization takes too long"

**Solutions:**

1. **Reduce parameter space**
   ```python
   # Start with coarse grid
   coarse_grid = {
       'rsi_period': [10, 20, 30],
       'stop_loss': [0.02, 0.04]
   }
   
   # Then refine around best parameters
   best_params = optimize(coarse_grid)
   fine_grid = create_fine_grid(best_params)
   ```

2. **Use parallel processing**
   ```python
   # Ensure n_jobs is set correctly
   import multiprocessing
   
   n_cores = multiprocessing.cpu_count()
   optimizer = GridSearchOptimizer(
       objective='sharpe_ratio',
       n_jobs=n_cores - 1  # Leave one core free
   )
   ```

3. **Cache intermediate results**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def cached_backtest(strategy_params_tuple, data_hash):
       # Convert tuple back to dict
       params = dict(strategy_params_tuple)
       strategy = create_strategy(params)
       return run_backtest(strategy, data)
   ```

## Memory Issues

### Problem: "MemoryError" or system running out of RAM

**Solutions:**

1. **Process data in chunks**
   ```python
   def backtest_in_chunks(data, strategy, chunk_size=50000):
       results = []
       
       for i in range(0, len(data), chunk_size):
           chunk = data.iloc[i:i+chunk_size]
           result = engine.run(chunk, strategy)
           results.append(result)
       
       # Combine results
       return combine_results(results)
   ```

2. **Use memory-efficient data types**
   ```python
   # Reduce memory usage
   def optimize_dtypes(df):
       for col in df.columns:
           col_type = df[col].dtype
           
           if col_type != 'object':
               c_min = df[col].min()
               c_max = df[col].max()
               
               if str(col_type)[:3] == 'int':
                   if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                       df[col] = df[col].astype(np.int8)
                   elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                   elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                       df[col] = df[col].astype(np.int32)
               else:
                   df[col] = df[col].astype(np.float32)
       
       return df
   ```

3. **Clear unused objects**
   ```python
   import gc
   
   # After large operations
   del large_dataframe
   gc.collect()
   ```

## Visualization Errors

### Problem: "Plotly charts not displaying"

**Solutions:**

```python
# 1. For Jupyter notebooks
import plotly.io as pio
pio.renderers.default = 'notebook'

# 2. For scripts
chart.show(renderer='browser')  # Opens in web browser

# 3. For headless servers
chart.write_html('chart.html')  # Save to file
```

### Problem: "Too many data points, chart is slow"

**Solution:**
```python
def downsample_for_plotting(data, max_points=1000):
    """Downsample data for faster plotting"""
    if len(data) <= max_points:
        return data
    
    # Use every nth point
    step = len(data) // max_points
    return data.iloc[::step]

# For OHLC data, use aggregation
def aggregate_ohlc(data, target_bars=1000):
    """Aggregate OHLC data to fewer bars"""
    if len(data) <= target_bars:
        return data
    
    agg_factor = len(data) // target_bars
    
    return data.resample(f'{agg_factor}D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
```

## Common Error Messages

### "AssertionError: Index is not monotonic"

**Solution:**
```python
# Ensure data is sorted by date
data = data.sort_index()

# Remove duplicate indices
data = data[~data.index.duplicated(keep='first')]
```

### "ValueError: cannot reindex from a duplicate axis"

**Solution:**
```python
# Check for duplicate dates
duplicates = data.index[data.index.duplicated()]
print(f"Duplicate dates: {duplicates}")

# Remove duplicates
data = data.loc[~data.index.duplicated(keep='first')]
```

### "KeyError: 'Timestamp not in index'"

**Solution:**
```python
# Ensure date format consistency
data.index = pd.to_datetime(data.index)

# Use date range that exists in data
actual_start = data.index[0]
actual_end = data.index[-1]
print(f"Data available from {actual_start} to {actual_end}")
```

## Debugging Tips

### 1. Enable Detailed Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest_debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### 2. Add Checkpoints

```python
def debug_backtest(data, strategy):
    """Backtest with debug checkpoints"""
    
    logger.info(f"Starting backtest with {len(data)} bars")
    logger.info(f"Strategy parameters: {strategy.params}")
    
    # Add progress tracking
    checkpoint_interval = len(data) // 10
    
    for i in range(len(data)):
        if i % checkpoint_interval == 0:
            logger.info(f"Progress: {i/len(data)*100:.0f}%")
        
        # Your backtest logic here
        
    logger.info("Backtest complete")
```

### 3. Validate Data Pipeline

```python
def validate_data_pipeline(raw_data):
    """Validate each step of data processing"""
    
    print("1. Raw data shape:", raw_data.shape)
    print("   Columns:", raw_data.columns.tolist())
    print("   Date range:", raw_data.index[0], "to", raw_data.index[-1])
    print("   Missing values:", raw_data.isna().sum().sum())
    
    # Add indicators
    data = add_indicators(raw_data)
    print("\n2. After indicators:", data.shape)
    print("   New columns:", set(data.columns) - set(raw_data.columns))
    
    # Check for NaN introduction
    new_nans = data.isna().sum() - raw_data.isna().sum()
    if new_nans.any():
        print("   WARNING: New NaN values introduced:")
        print(new_nans[new_nans > 0])
    
    return data
```

### 4. Test with Minimal Example

```python
def create_minimal_test():
    """Create minimal test case to isolate issues"""
    
    # Simple data
    dates = pd.date_range('2023-01-01', periods=100)
    data = pd.DataFrame({
        'open': 100,
        'high': 101,
        'low': 99,
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': 1000000
    }, index=dates)
    
    # Simple strategy
    builder = StrategyBuilder("Test")
    builder.add_entry_rule("close > close.shift(1)")
    builder.add_exit_rule("close < close.shift(1)")
    strategy = builder.build()
    
    # Run test
    engine = BacktestEngine(initial_capital=10000)
    results = engine.run(data, strategy)
    
    return results
```

## Getting Help

### 1. Check the Documentation

- [API Reference](API_REFERENCE.md) - Detailed API documentation
- [Strategy Development](STRATEGY_DEVELOPMENT.md) - Strategy creation guide
- [Examples](../examples/) - Working code examples

### 2. Enable Debug Mode

```python
# Set environment variable
import os
os.environ['BACKTEST_DEBUG'] = '1'

# Or use debug flag
engine = BacktestEngine(debug=True)
```

### 3. Create Minimal Reproducible Example

When reporting issues:

```python
# minimal_issue.py
import pandas as pd
from src.backtesting import BacktestEngine
from src.strategies import StrategyBuilder

# Minimal data that reproduces the issue
data = pd.DataFrame({
    'open': [100, 101, 102],
    'high': [101, 102, 103],
    'low': [99, 100, 101],
    'close': [100, 101, 102],
    'volume': [1000, 1000, 1000]
}, index=pd.date_range('2023-01-01', periods=3))

# Minimal strategy
builder = StrategyBuilder("Issue Demo")
builder.add_entry_rule("close > 101")
strategy = builder.build()

# Run and show error
engine = BacktestEngine()
results = engine.run(data, strategy)  # Error occurs here
```

### 4. Community Support

- **GitHub Issues**: [Report bugs](https://github.com/FRST-FRYTL/Backtest_Suite/issues)
- **Discussions**: [Ask questions](https://github.com/FRST-FRYTL/Backtest_Suite/discussions)
- **Stack Overflow**: Tag with `backtest-suite`

### 5. System Information

When reporting issues, include:

```python
import sys
import pandas as pd
import numpy as np

print("Python:", sys.version)
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Platform:", sys.platform)

# Package versions
import pkg_resources
installed_packages = pkg_resources.working_set
for package in installed_packages:
    print(f"{package.key}=={package.version}")
```

Remember: Most issues have simple solutions. Check your data, verify your logic, and test with minimal examples to isolate problems.