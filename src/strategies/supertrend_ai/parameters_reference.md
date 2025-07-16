# SuperTrend AI Strategy Parameters Reference

## Input Parameters Summary

### Core Algorithm Parameters
```yaml
core_parameters:
  atr_length:
    default: 10
    min: 1
    description: "ATR calculation period"
    
  factor_range:
    min_mult:
      default: 1
      min: 0
      description: "Minimum ATR multiplier for SuperTrend bands"
    max_mult:
      default: 5
      min: 0
      description: "Maximum ATR multiplier for SuperTrend bands"
    step:
      default: 0.5
      min: 0.1
      step: 0.1
      description: "Increment between factor values"
      
  performance_memory:
    default: 10
    min: 2
    description: "Alpha factor for performance exponential smoothing"
    
  cluster_selection:
    default: "Best"
    options: ["Best", "Average", "Worst"]
    description: "Which performance cluster to use for trading"
```

### Strategy Filters
```yaml
signal_filters:
  signal_strength:
    enabled:
      default: true
      description: "Use signal strength filtering"
    min_strength:
      default: 4
      min: 0
      max: 10
      description: "Minimum signal strength (0-10 scale)"
      
  time_filter:
    enabled:
      default: false
      description: "Restrict trading to specific hours"
    start_hour:
      default: 9
      min: 0
      max: 23
      description: "Trading start hour"
    end_hour:
      default: 16
      min: 0
      max: 23
      description: "Trading end hour"
```

### Risk Management Parameters
```yaml
risk_management:
  stop_loss:
    enabled:
      default: true
      description: "Use stop loss orders"
    type:
      default: "ATR"
      options: ["ATR", "Percentage"]
      description: "Stop loss calculation method"
    atr_multiplier:
      default: 2.0
      min: 0.1
      step: 0.1
      description: "ATR multiplier for stop loss"
    percentage:
      default: 2.0
      min: 0.1
      step: 0.1
      description: "Percentage stop loss"
      
  take_profit:
    enabled:
      default: true
      description: "Use take profit orders"
    type:
      default: "Risk/Reward"
      options: ["Risk/Reward", "ATR", "Percentage"]
      description: "Take profit calculation method"
    risk_reward_ratio:
      default: 2.0
      min: 0.1
      step: 0.1
      description: "Risk/reward ratio for TP"
    atr_multiplier:
      default: 3.0
      min: 0.1
      step: 0.1
      description: "ATR multiplier for take profit"
    percentage:
      default: 4.0
      min: 0.1
      step: 0.1
      description: "Percentage take profit"
```

### Optimization Parameters
```yaml
optimization:
  max_iterations:
    default: 1000
    min: 0
    description: "Maximum K-means iterations"
    
  historical_bars:
    default: 10000
    min: 0
    description: "Number of bars for performance calculation"
```

### Visualization Parameters
```yaml
visualization:
  candle_coloring:
    default: true
    description: "Color candles by performance"
    
  show_signals:
    default: true
    description: "Display entry signals with strength"
    
  dashboard:
    enabled:
      default: true
      description: "Show performance dashboard"
    location:
      default: "Top Right"
      options: ["Top Right", "Bottom Right", "Bottom Left"]
    size:
      default: "Small"
      options: ["Tiny", "Small", "Normal"]
```

## Calculated Values

### Signal Strength Scale
- **Range**: 0-10
- **Calculation**: `int(performance_index * 10)`
- **Usage**: Higher values indicate stronger trend confidence

### Performance Index
- **Formula**: `max(cluster_avg_performance, 0) / ema(abs(price_change), perf_alpha)`
- **Purpose**: Normalizes cluster performance by market volatility

### Dynamic Factor Selection
- **Process**: K-means clustering selects optimal ATR multiplier from performance data
- **Update Frequency**: Every bar (real-time adaptation)