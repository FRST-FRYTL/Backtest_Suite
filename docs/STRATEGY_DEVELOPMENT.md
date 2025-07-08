# Strategy Development Guide

This comprehensive guide covers everything you need to know about developing profitable trading strategies with the Backtest Suite.

## Table of Contents

- [Strategy Fundamentals](#strategy-fundamentals)
- [Entry and Exit Rules](#entry-and-exit-rules)
- [Risk Management](#risk-management)
- [Position Sizing](#position-sizing)
- [Market Regime Filters](#market-regime-filters)
- [Multi-Timeframe Strategies](#multi-timeframe-strategies)
- [Advanced Techniques](#advanced-techniques)
- [Common Strategy Patterns](#common-strategy-patterns)
- [Testing and Validation](#testing-and-validation)
- [Best Practices](#best-practices)

## Strategy Fundamentals

### What Makes a Good Strategy?

A robust trading strategy should have:

1. **Clear Rules**: Unambiguous entry and exit conditions
2. **Risk Management**: Defined stop losses and position sizing
3. **Edge**: Statistical advantage over random trading
4. **Consistency**: Works across different market conditions
5. **Simplicity**: Avoid over-optimization and curve fitting

### Strategy Components

Every strategy in the Backtest Suite consists of:

```yaml
name: "Strategy Name"
version: "1.0"

indicators: []      # Technical indicators used
entry_rules: []     # When to enter positions
exit_rules: []      # When to exit positions
risk_management: {} # Stop loss, take profit, position sizing
filters: []         # Optional trade filters
```

## Entry and Exit Rules

### Basic Rule Structure

Rules use Python-like boolean expressions:

```yaml
entry_rules:
  - condition: "rsi < 30"
    logic: AND
  - condition: "close < bb_lower"
    logic: AND
```

### Available Operators

- **Comparison**: `<`, `>`, `<=`, `>=`, `==`, `!=`
- **Logical**: `and`, `or`, `not`
- **Mathematical**: `+`, `-`, `*`, `/`, `**`
- **Pandas Methods**: `.rolling()`, `.shift()`, `.mean()`, etc.

### Complex Conditions

```yaml
entry_rules:
  # Multiple conditions in one rule
  - condition: "rsi < 30 and volume > volume.rolling(20).mean() * 1.5"
    
  # Using shift for previous values
  - condition: "close > close.shift(1) and close.shift(1) > close.shift(2)"
    
  # Percentage calculations
  - condition: "(close - bb_lower) / bb_lower < 0.02"
```

### Entry Rule Examples

#### Momentum Entry
```yaml
entry_rules:
  - condition: "rsi > 50 and rsi.shift(1) <= 50"  # RSI crossing above 50
  - condition: "close > sma_20"                     # Price above 20 SMA
  - condition: "volume > volume.rolling(10).mean()" # Above average volume
```

#### Mean Reversion Entry
```yaml
entry_rules:
  - condition: "close < bb_lower"                   # Below lower Bollinger Band
  - condition: "rsi < 30"                           # Oversold RSI
  - condition: "close > close.rolling(200).mean()" # Above 200-day MA (uptrend)
```

#### Breakout Entry
```yaml
entry_rules:
  - condition: "close > high.rolling(20).max().shift(1)" # New 20-day high
  - condition: "volume > volume.rolling(20).mean() * 2"   # High volume
  - condition: "atr > atr.rolling(20).mean()"             # Increased volatility
```

### Exit Rule Examples

#### Fixed Target Exit
```yaml
exit_rules:
  - condition: "rsi > 70"                    # Overbought
    logic: OR
  - condition: "close > entry_price * 1.05"  # 5% profit target
    logic: OR
```

#### Trailing Stop Exit
```yaml
exit_rules:
  - condition: "close < highest_price * 0.98"  # 2% trailing stop
    logic: OR
  - condition: "bars_in_trade > 10"            # Time-based exit
    logic: OR
```

#### Technical Exit
```yaml
exit_rules:
  - condition: "close < sma_20"                          # Below moving average
    logic: OR
  - condition: "macd < macd_signal and macd.shift(1) >= macd_signal.shift(1)"  # MACD crossunder
    logic: OR
```

## Risk Management

### Stop Loss Types

#### Fixed Percentage Stop
```yaml
risk_management:
  stop_loss: 0.02  # 2% stop loss
```

#### ATR-Based Stop
```python
builder.set_risk_management(
    stop_loss=lambda data, entry_idx: data['atr'].iloc[entry_idx] * 2
)
```

#### Support-Based Stop
```python
# Set stop below recent low
builder.set_risk_management(
    stop_loss=lambda data, entry_idx: (
        entry_price - data['low'].iloc[max(0, entry_idx-5):entry_idx].min()
    ) / entry_price
)
```

### Take Profit Strategies

#### Fixed Risk-Reward
```yaml
risk_management:
  stop_loss: 0.02    # 2% stop
  take_profit: 0.06  # 6% target (3:1 risk-reward)
```

#### Dynamic Targets
```python
# Take profit at resistance levels
builder.set_risk_management(
    take_profit=lambda data, entry_idx: (
        data['bb_upper'].iloc[entry_idx] - entry_price
    ) / entry_price
)
```

### Trailing Stops

```yaml
risk_management:
  trailing_stop: 0.03        # 3% trailing stop
  trailing_stop_activation: 0.05  # Activate after 5% profit
```

## Position Sizing

### Fixed Position Size
```yaml
risk_management:
  position_size: 0.25  # Use 25% of capital per trade
```

### Risk-Based Position Sizing
```python
# Risk 1% of capital per trade
def calculate_position_size(capital, price, stop_loss_pct):
    risk_amount = capital * 0.01  # 1% risk
    position_value = risk_amount / stop_loss_pct
    shares = int(position_value / price)
    return shares

builder.set_risk_management(
    position_size=calculate_position_size
)
```

### Kelly Criterion
```yaml
risk_management:
  position_size: "kelly"  # Automatic Kelly Criterion sizing
  kelly_fraction: 0.25    # Use 25% of Kelly (conservative)
```

### Volatility-Based Sizing
```python
# Smaller positions in high volatility
def volatility_position_size(data, idx, capital):
    current_vol = data['atr'].iloc[idx] / data['close'].iloc[idx]
    avg_vol = data['atr'].rolling(20).mean().iloc[idx] / data['close'].rolling(20).mean().iloc[idx]
    vol_ratio = avg_vol / current_vol
    base_size = 0.25
    return min(base_size * vol_ratio, 0.5)  # Cap at 50%

builder.set_risk_management(
    position_size=volatility_position_size
)
```

## Market Regime Filters

### Trend Filters

#### Simple Moving Average Filter
```yaml
filters:
  - condition: "sma_50 > sma_200"  # Only trade in uptrends
```

#### ADX Trend Strength
```yaml
filters:
  - condition: "adx > 25"  # Only trade when trend is strong
```

### Volatility Filters

#### VIX Filter
```yaml
filters:
  - condition: "vix < 30"  # Avoid high volatility periods
```

#### ATR Filter
```yaml
filters:
  - condition: "atr < atr.rolling(50).mean() * 1.5"  # Normal volatility
```

### Market Breadth Filters

```python
# Only trade when market internals are positive
filters:
  - condition: "advance_decline_ratio > 1.5"  # More advancers than decliners
  - condition: "percent_above_50ma > 60"      # Majority of stocks above 50 MA
```

### Time-Based Filters

```yaml
filters:
  # Avoid trading first and last 30 minutes
  - condition: "time >= '09:30' and time <= '15:30'"
  
  # Skip Monday opens and Friday closes
  - condition: "not (dayofweek == 0 and hour < 11)"
  - condition: "not (dayofweek == 4 and hour > 14)"
```

## Multi-Timeframe Strategies

### Loading Multiple Timeframes

```python
# Load daily and hourly data
daily_data = await fetcher.fetch("AAPL", start, end, interval="1d")
hourly_data = await fetcher.fetch("AAPL", start, end, interval="1h")

# Calculate indicators on different timeframes
daily_data['rsi_daily'] = RSI(14).calculate(daily_data)
hourly_data['rsi_hourly'] = RSI(14).calculate(hourly_data)

# Merge timeframes
merged_data = pd.merge_asof(
    hourly_data,
    daily_data[['rsi_daily']],
    left_index=True,
    right_index=True,
    direction='backward'
)
```

### Multi-Timeframe Rules

```yaml
# Enter on hourly signals with daily confirmation
entry_rules:
  - condition: "rsi_hourly < 30"        # Hourly oversold
  - condition: "rsi_daily < 50"         # Daily not overbought
  - condition: "close > sma_daily_200"  # Daily uptrend
```

### Timeframe Alignment Strategy

```python
class MultiTimeframeStrategy:
    def __init__(self):
        self.timeframes = ['5m', '1h', '1d']
    
    def check_alignment(self, data):
        # All timeframes must agree
        signals = []
        for tf in self.timeframes:
            signals.append(data[f'signal_{tf}'] == 1)
        
        return all(signals)
```

## Advanced Techniques

### Machine Learning Integration

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Feature engineering
def create_features(data):
    features = pd.DataFrame()
    features['rsi'] = data['rsi']
    features['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    features['price_change'] = data['close'].pct_change(5)
    
    return features

# Train ML model
features = create_features(historical_data)
labels = (historical_data['close'].shift(-5) > historical_data['close']).astype(int)

model = RandomForestClassifier(n_estimators=100)
model.fit(features, labels)

# Use in strategy
def ml_signal(data, idx):
    current_features = create_features(data).iloc[idx:idx+1]
    prediction = model.predict_proba(current_features)[0, 1]
    return prediction > 0.65  # High confidence threshold
```

### Correlation-Based Pairs Trading

```python
# Identify correlated pairs
def find_pairs(symbols, data_dict, threshold=0.8):
    correlations = {}
    for s1 in symbols:
        for s2 in symbols:
            if s1 < s2:
                corr = data_dict[s1]['close'].corr(data_dict[s2]['close'])
                if corr > threshold:
                    correlations[(s1, s2)] = corr
    
    return correlations

# Pairs trading strategy
class PairsStrategy:
    def __init__(self, pair, hedge_ratio):
        self.pair = pair
        self.hedge_ratio = hedge_ratio
    
    def calculate_spread(self, data1, data2):
        return data1['close'] - self.hedge_ratio * data2['close']
    
    def generate_signals(self, spread):
        z_score = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
        
        entry_long = z_score < -2    # Spread too low, buy spread
        entry_short = z_score > 2     # Spread too high, sell spread
        exit_signal = abs(z_score) < 0.5  # Close to mean
        
        return entry_long, entry_short, exit_signal
```

### Options-Based Strategies

```python
# Using max pain as support/resistance
async def options_levels_strategy():
    mp = MaxPain()
    max_pain_data = await mp.calculate("SPY")
    
    # Build strategy using max pain levels
    builder = StrategyBuilder("Options Levels")
    
    # Buy when price touches max pain support
    builder.add_entry_rule(
        f"close <= {max_pain_data['max_pain_price']} * 0.99"
    )
    
    # Sell at resistance
    builder.add_exit_rule(
        f"close >= {max_pain_data['max_pain_price']} * 1.02"
    )
    
    return builder.build()
```

### Sentiment-Based Trading

```python
# Fear & Greed with technical confirmation
async def sentiment_strategy():
    # Get sentiment data
    fgi = FearGreedIndex()
    sentiment = await fgi.fetch_historical()
    
    # Combine with price data
    builder = StrategyBuilder("Sentiment Contrarian")
    
    # Buy extreme fear with technical support
    builder.add_entry_rule("fear_greed < 20")  # Extreme fear
    builder.add_entry_rule("rsi < 40", logic="AND")  # Oversold
    builder.add_entry_rule("close > sma_200", logic="AND")  # Long-term uptrend
    
    # Exit on greed or technical breakdown
    builder.add_exit_rule("fear_greed > 80")  # Extreme greed
    builder.add_exit_rule("close < sma_50", logic="OR")  # Technical breakdown
    
    return builder.build()
```

## Common Strategy Patterns

### 1. Trend Following

```yaml
name: "Simple Trend Following"
indicators:
  - name: sma_fast
    type: SMA
    params: {period: 50}
  - name: sma_slow
    type: SMA
    params: {period: 200}
  - name: atr
    type: ATR
    params: {period: 14}

entry_rules:
  - condition: "sma_fast > sma_slow"
  - condition: "close > sma_fast"

exit_rules:
  - condition: "close < sma_fast"

risk_management:
  stop_loss: "2 * atr"
  position_size: 0.25
```

### 2. Mean Reversion

```yaml
name: "Bollinger Band Mean Reversion"
indicators:
  - name: bb
    type: BollingerBands
    params: {period: 20, std_dev: 2}
  - name: rsi
    type: RSI
    params: {period: 14}

entry_rules:
  - condition: "close < bb_lower"
  - condition: "rsi < 30"

exit_rules:
  - condition: "close > bb_middle"
  - condition: "rsi > 50"

risk_management:
  stop_loss: 0.03
  take_profit: "bb_upper"
```

### 3. Breakout Trading

```yaml
name: "Volatility Breakout"
indicators:
  - name: atr
    type: ATR
    params: {period: 14}
  - name: highest
    type: Highest
    params: {period: 20}

entry_rules:
  - condition: "close > highest.shift(1)"
  - condition: "volume > volume.rolling(20).mean() * 1.5"

exit_rules:
  - condition: "close < entry_price - 2 * atr"

risk_management:
  trailing_stop: 0.02
  position_size: "volatility_adjusted"
```

### 4. Momentum Trading

```yaml
name: "RSI Momentum"
indicators:
  - name: rsi
    type: RSI
    params: {period: 14}
  - name: macd
    type: MACD
    params: {fast: 12, slow: 26, signal: 9}

entry_rules:
  - condition: "rsi > 50 and rsi.shift(1) <= 50"
  - condition: "macd > macd_signal"

exit_rules:
  - condition: "rsi > 70"
  - condition: "macd < macd_signal"

risk_management:
  stop_loss: 0.02
  trailing_stop: 0.03
```

## Testing and Validation

### Walk-Forward Analysis

```python
# Prevent overfitting with walk-forward testing
results = engine.run_walk_forward(
    data=data,
    strategy=strategy,
    in_sample_periods=252,    # 1 year in-sample
    out_sample_periods=63,    # 3 months out-of-sample
    optimization_func=optimize_parameters
)

# Analyze consistency
in_sample_returns = [r['in_sample']['return'] for r in results]
out_sample_returns = [r['out_sample']['return'] for r in results]

consistency_score = np.corrcoef(in_sample_returns, out_sample_returns)[0, 1]
print(f"Consistency Score: {consistency_score:.2f}")
```

### Monte Carlo Simulation

```python
def monte_carlo_test(strategy, data, n_simulations=1000):
    results = []
    
    for i in range(n_simulations):
        # Randomize entry timing
        random_delay = np.random.randint(0, 5, size=len(data))
        modified_data = data.copy()
        
        # Run backtest with randomization
        engine = BacktestEngine()
        result = engine.run(modified_data, strategy)
        results.append(result['performance']['total_return'])
    
    # Analyze distribution
    mean_return = np.mean(results)
    std_return = np.std(results)
    sharpe = mean_return / std_return
    
    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'sharpe': sharpe,
        'win_probability': sum(r > 0 for r in results) / len(results)
    }
```

### Robustness Checks

```python
# Test across different market conditions
def test_market_conditions(strategy, full_data):
    # Define market regimes
    bull_market = full_data['2017':'2019']
    bear_market = full_data['2008':'2009']
    sideways_market = full_data['2015':'2016']
    
    results = {}
    for name, data in [
        ('Bull', bull_market),
        ('Bear', bear_market),
        ('Sideways', sideways_market)
    ]:
        engine = BacktestEngine()
        result = engine.run(data, strategy)
        results[name] = result['performance']
    
    return results
```

## Best Practices

### 1. Start Simple
- Begin with basic strategies
- Add complexity only when justified by results
- Document your reasoning for each rule

### 2. Avoid Overfitting
- Use walk-forward analysis
- Test on out-of-sample data
- Limit the number of parameters
- Prefer robust parameters that work across ranges

### 3. Consider Transaction Costs
- Include realistic commission and slippage
- Account for market impact on large positions
- Test sensitivity to cost assumptions

### 4. Risk Management First
- Always define stop losses
- Size positions appropriately
- Diversify across strategies and assets
- Set maximum drawdown limits

### 5. Continuous Improvement
- Keep a trading journal
- Track real vs. backtest performance
- Regularly review and update strategies
- Learn from both wins and losses

### 6. Code Organization

```python
# Organize strategies in modules
class StrategyBase:
    """Base class for all strategies"""
    
    def __init__(self, params):
        self.params = params
        self.validate_params()
    
    def validate_params(self):
        """Validate strategy parameters"""
        pass
    
    def calculate_indicators(self, data):
        """Calculate required indicators"""
        pass
    
    def generate_signals(self, data):
        """Generate entry/exit signals"""
        pass

# Specific strategy implementation
class RSIMeanReversion(StrategyBase):
    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        params = {
            'rsi_period': rsi_period,
            'oversold': oversold,
            'overbought': overbought
        }
        super().__init__(params)
    
    def calculate_indicators(self, data):
        data['rsi'] = RSI(self.params['rsi_period']).calculate(data)
        return data
```

### 7. Performance Monitoring

```python
# Track strategy metrics over time
class StrategyMonitor:
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name
        self.metrics_history = []
    
    def record_performance(self, date, metrics):
        self.metrics_history.append({
            'date': date,
            'metrics': metrics
        })
    
    def analyze_degradation(self, window=30):
        recent = self.metrics_history[-window:]
        older = self.metrics_history[-2*window:-window]
        
        recent_sharpe = np.mean([m['metrics']['sharpe'] for m in recent])
        older_sharpe = np.mean([m['metrics']['sharpe'] for m in older])
        
        if recent_sharpe < older_sharpe * 0.7:
            print(f"Warning: Strategy {self.strategy_name} showing degradation")
```

## Example: Complete Strategy Development Workflow

```python
import asyncio
from src.data import StockDataFetcher
from src.indicators import RSI, BollingerBands, ATR
from src.strategies import StrategyBuilder
from src.backtesting import BacktestEngine
from src.optimization import GridSearchOptimizer
from src.visualization import ChartBuilder

async def develop_strategy():
    # 1. Data Collection
    fetcher = StockDataFetcher()
    symbols = ['SPY', 'QQQ', 'IWM']  # Test on multiple markets
    
    data_dict = {}
    for symbol in symbols:
        data_dict[symbol] = await fetcher.fetch(symbol, '2020-01-01', '2023-12-31')
    
    # 2. Indicator Calculation
    for symbol, data in data_dict.items():
        data['rsi'] = RSI(14).calculate(data)
        bb_data = BollingerBands(20, 2).calculate(data)
        data = data.join(bb_data)
        data['atr'] = ATR(14).calculate(data)
        data_dict[symbol] = data
    
    # 3. Strategy Design
    builder = StrategyBuilder("Multi-Market Mean Reversion")
    
    # Entry rules
    builder.add_entry_rule("rsi < 30")
    builder.add_entry_rule("close < bb_lower", logic="AND")
    builder.add_entry_rule("volume > volume.rolling(20).mean()", logic="AND")
    
    # Exit rules
    builder.add_exit_rule("rsi > 70")
    builder.add_exit_rule("close > bb_upper", logic="OR")
    
    # Risk management
    builder.set_risk_management(
        stop_loss=0.03,
        take_profit=0.06,
        trailing_stop=0.02,
        position_size=0.25
    )
    
    # Market filter
    builder.add_filter("close > close.rolling(200).mean()")
    
    strategy = builder.build()
    
    # 4. Initial Testing
    engine = BacktestEngine(initial_capital=100000)
    
    results_dict = {}
    for symbol, data in data_dict.items():
        results = engine.run(data, strategy)
        results_dict[symbol] = results
        
        print(f"\n{symbol} Results:")
        print(f"Total Return: {results['performance']['total_return']:.2f}%")
        print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['performance']['max_drawdown']:.2f}%")
        print(f"Win Rate: {results['performance']['win_rate']:.2f}%")
    
    # 5. Parameter Optimization (on best performing market)
    best_symbol = max(results_dict, key=lambda x: results_dict[x]['performance']['sharpe_ratio'])
    print(f"\nOptimizing on {best_symbol}...")
    
    optimizer = GridSearchOptimizer(objective='sharpe_ratio')
    param_grid = {
        'rsi_period': [10, 14, 20],
        'rsi_oversold': [25, 30, 35],
        'bb_period': [15, 20, 25],
        'stop_loss': [0.02, 0.03, 0.04]
    }
    
    opt_results = optimizer.optimize(
        strategy_class=type(strategy),
        parameter_grid=param_grid,
        data=data_dict[best_symbol]
    )
    
    print(f"\nOptimal Parameters: {opt_results['best_params']}")
    print(f"Best Sharpe Ratio: {opt_results['best_score']:.2f}")
    
    # 6. Walk-Forward Validation
    wf_results = engine.run_walk_forward(
        data=data_dict[best_symbol],
        strategy=strategy,
        in_sample_periods=252,
        out_sample_periods=63
    )
    
    # 7. Final Testing on All Markets
    # Apply optimal parameters and test on all markets
    
    # 8. Visualization
    chart = ChartBuilder(f"{best_symbol} Strategy Results")
    chart.add_price_chart(data_dict[best_symbol], results_dict[best_symbol]['trades'])
    chart.add_indicator("RSI", data_dict[best_symbol]['rsi'], subplot=True)
    chart.add_equity_curve(results_dict[best_symbol]['equity_curve'])
    chart.add_drawdown(results_dict[best_symbol]['drawdown_series'])
    chart.save(f"{best_symbol}_strategy_results.html")
    
    return strategy, results_dict

# Run the complete workflow
strategy, results = asyncio.run(develop_strategy())
```

This comprehensive guide provides everything needed to develop robust trading strategies with the Backtest Suite. Remember that successful trading requires continuous learning, disciplined risk management, and realistic expectations.