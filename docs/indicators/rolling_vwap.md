# Rolling VWAP (Volume Weighted Average Price) Implementation

## Overview

Rolling VWAP is a variation of the traditional VWAP indicator that calculates the volume-weighted average price over a rolling window of N periods, rather than resetting at specific time intervals (daily, weekly, etc.).

## Key Features

### 1. Multiple Period Support
- Configurable periods: 5, 10, 20, 50, 100, 200
- Each period provides different market perspectives:
  - **5-10 periods**: Ultra short-term, scalping
  - **20 periods**: Short-term trading
  - **50 periods**: Medium-term trend
  - **100-200 periods**: Long-term market structure

### 2. Standard Deviation Bands
- Configurable bands: 1, 2, and 3 standard deviations
- Volume-weighted standard deviation calculation
- Bands adapt to market volatility

### 3. Additional Calculations
- **Position**: Normalized position within bands (0-1 scale)
- **Distance**: Percentage distance from VWAP
- **Cross Detection**: Boolean flags for crosses above/below

## Usage

### Configuration

Add to `config/strategy_config.yaml`:

```yaml
indicators:
  rolling_vwap:
    periods: [5, 10, 20, 50, 100, 200]
    std_devs: [1, 2, 3]
```

### Implementation

```python
from src.indicators.multi_timeframe_indicators import MultiTimeframeIndicators

# Initialize indicators
indicators = MultiTimeframeIndicators(config)

# Calculate rolling VWAP
df_with_vwap = indicators.calculate_rolling_vwap(
    df,
    periods=[20, 50, 200],
    std_devs=[1, 2, 3]
)
```

### Output Columns

For each period (e.g., 20), the following columns are added:
- `Rolling_VWAP_20`: The rolling VWAP value
- `Rolling_VWAP_20_Upper_1`: Upper 1-sigma band
- `Rolling_VWAP_20_Upper_2`: Upper 2-sigma band
- `Rolling_VWAP_20_Upper_3`: Upper 3-sigma band
- `Rolling_VWAP_20_Lower_1`: Lower 1-sigma band
- `Rolling_VWAP_20_Lower_2`: Lower 2-sigma band
- `Rolling_VWAP_20_Lower_3`: Lower 3-sigma band
- `Rolling_VWAP_20_Position`: Position within 2-sigma bands (0-1)
- `Rolling_VWAP_20_Distance`: % distance from VWAP
- `Rolling_VWAP_20_Cross_Above`: True when price crosses above
- `Rolling_VWAP_20_Cross_Below`: True when price crosses below

## Trading Applications

### 1. Mean Reversion
- Enter long when price touches lower bands
- Enter short when price touches upper bands
- Use tighter bands (1-sigma) for scalping
- Use wider bands (2-3 sigma) for swing trades

### 2. Trend Following
- Long when price above VWAP with upward slope
- Short when price below VWAP with downward slope
- Use multiple periods for confirmation

### 3. Support/Resistance
- VWAP acts as dynamic support in uptrends
- VWAP acts as dynamic resistance in downtrends
- Bands provide additional S/R levels

### 4. Volume Confirmation
- Stronger signals when volume increases at VWAP touches
- Breakouts more reliable with volume expansion

## Calculation Details

### Rolling VWAP Formula
```
Typical Price = (High + Low + Close) / 3
Rolling VWAP = Sum(Typical Price × Volume, N) / Sum(Volume, N)
```

### Volume-Weighted Standard Deviation
```
Variance = Sum((Typical Price - VWAP)² × Volume, N) / Sum(Volume, N)
Std Dev = √Variance
```

### Bands
```
Upper Band = VWAP + (Std Dev × Multiplier)
Lower Band = VWAP - (Std Dev × Multiplier)
```

## Advantages Over Traditional VWAP

1. **No Reset Points**: Continuous calculation without arbitrary resets
2. **Adaptability**: Adjusts to changing market conditions
3. **Multiple Timeframes**: Different periods for different strategies
4. **Consistency**: Same calculation method across all timeframes

## Best Practices

1. **Combine Multiple Periods**: Use short-term for timing, long-term for bias
2. **Volume Filter**: Require minimum volume for valid signals
3. **Trend Alignment**: Confirm with moving averages or other trend indicators
4. **Risk Management**: Use bands for stop-loss placement

## Example Strategy

See `examples/strategies/rolling_vwap_strategy_example.py` for a complete implementation showing:
- Multi-timeframe VWAP analysis
- Signal generation logic
- Position sizing based on VWAP position
- Risk management using bands

## Performance Considerations

- Calculation is O(n) for each period
- Memory usage scales with number of periods
- Pre-calculate for backtesting efficiency
- Use vectorized operations for speed