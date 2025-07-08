# Indicators Documentation

This document provides detailed information about all available indicators in the Backtest Suite.

## Technical Indicators

### RSI (Relative Strength Index)

The RSI is a momentum oscillator that measures the speed and magnitude of price changes.

```python
from src.indicators import RSI

# Create RSI indicator
rsi = RSI(
    period=14,          # Lookback period
    overbought=70,      # Overbought level
    oversold=30         # Oversold level
)

# Calculate RSI
rsi_values = rsi.calculate(data, price_column='close')

# Get trading signals
signals = rsi.get_signals(rsi_values)
# Returns DataFrame with columns:
# - oversold: True when RSI < oversold level
# - overbought: True when RSI > overbought level
# - cross_above_oversold: Crossing above oversold
# - cross_below_overbought: Crossing below overbought
# - cross_above_50: Bullish momentum
# - cross_below_50: Bearish momentum

# Detect divergences
divergences = rsi.divergence(data['close'], rsi_values)
# Returns DataFrame with:
# - bullish_divergence: Price makes lower low, RSI makes higher low
# - bearish_divergence: Price makes higher high, RSI makes lower high
```

### Bollinger Bands

Bollinger Bands consist of a middle band (SMA) and two outer bands (standard deviations).

```python
from src.indicators import BollingerBands

# Create Bollinger Bands
bb = BollingerBands(
    period=20,          # Moving average period
    std_dev=2.0,        # Standard deviations for bands
    ma_type='sma'       # 'sma' or 'ema'
)

# Calculate bands
bb_data = bb.calculate(data, price_column='close')
# Returns DataFrame with:
# - bb_middle: Middle band (moving average)
# - bb_upper: Upper band
# - bb_lower: Lower band
# - bb_width: Band width
# - bb_percent: %B indicator (position within bands)

# Get signals
signals = bb.get_signals(data, bb_data)
# Includes band touches, breakouts, squeezes

# Detect patterns
patterns = bb.detect_patterns(data, bb_data, lookback=20)
# - w_bottom: W-bottom reversal pattern
# - m_top: M-top reversal pattern
# - walking_upper: Trending along upper band
# - walking_lower: Trending along lower band
```

### VWMA Bands (Volume Weighted Moving Average)

VWMA gives more weight to periods with higher volume.

```python
from src.indicators import VWMABands

# Create VWMA Bands
vwma = VWMABands(
    period=20,              # Lookback period
    band_multiplier=2.0,    # Band width multiplier
    price_column='close'    # Price to use
)

# Calculate VWMA and bands
vwma_data = vwma.calculate(data, volume_column='volume')
# Returns DataFrame with:
# - vwma: Volume weighted moving average
# - vwma_upper: Upper band
# - vwma_lower: Lower band
# - vwma_width: Band width

# Volume confirmation signals
volume_signals = vwma.volume_confirmation(data, vwma_data)
# - bullish_volume: Price above VWMA with high volume
# - bearish_volume: Price below VWMA with high volume
# - low_volume_move: Price movement without volume
```

### TSV (Time Segmented Volume)

TSV combines price and volume to assess money flow.

```python
from src.indicators import TSV

# Create TSV indicator
tsv = TSV(
    period=13,          # TSV period
    signal_period=9     # Signal line EMA period
)

# Calculate TSV
tsv_data = tsv.calculate(data)
# Returns DataFrame with:
# - tsv: Time segmented volume
# - tsv_signal: Signal line
# - tsv_histogram: TSV - Signal
# - tsv_raw: Raw TSV values

# Analyze volume patterns
volume_analysis = tsv.volume_analysis(data, tsv_data)
# - strong_buying: High volume with positive TSV
# - strong_selling: High volume with negative TSV
# - weak_move: TSV movement with low volume
# - volume_climax: Extreme volume with TSV reversal
```

### VWAP (Volume Weighted Average Price)

VWAP is the average price weighted by volume, used as a trading benchmark.

```python
from src.indicators import VWAP, AnchoredVWAP

# Rolling VWAP
vwap = VWAP(
    window=20,              # Rolling window (None for session)
    price_type='typical',   # 'typical', 'close', 'hl2', 'ohlc4'
    std_dev_bands=[1, 2, 3] # Standard deviation bands
)

vwap_data = vwap.calculate(
    data,
    volume_column='volume',
    reset_time=time(9, 30)  # For intraday reset
)
# Returns DataFrame with VWAP and bands

# Anchored VWAP from specific date
anchor_date = '2023-01-15'
avwap = AnchoredVWAP(
    anchor_date=anchor_date,
    price_type='typical',
    std_dev_bands=[1, 2]
)

avwap_data = avwap.calculate(data)

# Multiple anchored VWAPs
event_dates = ['2023-01-15', '2023-03-15', '2023-06-15']
multi_avwap = AnchoredVWAP.create_multiple_anchors(
    data, event_dates
)
```

## Meta Indicators

### Fear and Greed Index

Market sentiment indicator ranging from 0 (extreme fear) to 100 (extreme greed).

```python
from src.indicators import FearGreedIndex

# Create Fear & Greed indicator
fg = FearGreedIndex(
    cache_dir='data/cache',
    source='alternative'    # 'alternative' or 'cnn'
)

# Fetch current value
current = await fg.fetch_current()
# Returns:
# - value: Current index value (0-100)
# - classification: 'Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'
# - timestamp: Data timestamp

# Fetch historical data
historical = await fg.fetch_historical(limit=30)  # Last 30 days
# Returns DataFrame with daily values

# Generate signals
signals = fg.get_signals(historical)
# - extreme_fear: Value <= 20
# - extreme_greed: Value >= 80
# - fear_reversal: Exiting extreme fear
# - greed_reversal: Exiting extreme greed

# Correlation analysis with price
correlation = fg.correlation_analysis(
    historical, price_data, window=20
)
```

### Insider Trading

Scrapes and analyzes insider trading data from OpenInsider.

```python
from src.indicators import InsiderTrading

# Create insider trading scraper
insider = InsiderTrading()

# Fetch latest trades
async with insider:
    # All latest trades
    trades = await insider.fetch_latest_trades(limit=100)
    
    # Specific ticker
    aapl_trades = await insider.fetch_latest_trades(
        ticker='AAPL', limit=50
    )
    
    # Cluster buys (multiple insiders buying)
    cluster_buys = await insider.fetch_cluster_buys(days=30)
    
    # Large transactions
    big_buys = await insider.fetch_significant_buys(
        min_value=1000000  # $1M minimum
    )

# Analyze sentiment
sentiment = insider.analyze_sentiment(trades)
# Returns:
# - buy_sell_ratio: Number of buys vs sells
# - bullish_score: 0-100 sentiment score
# - top_buyers: Largest insider purchases
# - top_sellers: Largest insider sales

# Generate signals for specific ticker
signals = insider.get_signals('AAPL', trades, lookback_days=90)
# - cluster_buy: Multiple insiders buying
# - large_buy: Single large purchase
# - insider_accumulation: Sustained buying
```

### Max Pain

Calculates the options strike price with maximum pain for option holders.

```python
from src.indicators import MaxPain

# Create max pain calculator
max_pain = MaxPain()

# Calculate max pain for nearest expiration
result = max_pain.calculate('AAPL')
# Returns:
# - max_pain_price: Strike with maximum pain
# - pain_distribution: Pain at each strike
# - resistance_levels: High call OI strikes
# - support_levels: High put OI strikes
# - put_call_ratio: Put/Call ratio
# - gamma_levels: High gamma exposure strikes

# Calculate for specific expiration
result = max_pain.calculate('AAPL', expiration_date='2023-12-15')

# Multiple expirations analysis
all_expirations = max_pain.calculate_all_expirations(
    'AAPL', max_expirations=4
)

# Generate trading signals
signals = max_pain.get_signals('AAPL', current_price=150, result)
# - max_pain_magnet: Price near max pain
# - extreme_deviation: Far from max pain
# - near_resistance: Close to resistance
# - near_support: Close to support
# - gamma_squeeze_potential: Near high gamma
```

## Combining Indicators

Indicators can be combined for more sophisticated strategies:

```python
# Calculate multiple indicators
data['rsi'] = RSI(14).calculate(data)
bb_data = BollingerBands(20).calculate(data)
vwap_data = VWAP().calculate(data)
tsv_data = TSV().calculate(data)

# Merge data
data = data.join(bb_data).join(vwap_data).join(tsv_data)

# Create complex entry conditions
entry_conditions = (
    (data['rsi'] < 30) &                    # Oversold
    (data['close'] < data['bb_lower']) &    # Below BB
    (data['close'] < data['vwap']) &        # Below VWAP
    (data['tsv'] > data['tsv_signal'])      # Positive momentum
)

# Multiple timeframe analysis
daily_rsi = RSI(14).calculate(daily_data)
hourly_rsi = RSI(14).calculate(hourly_data)

# Combine timeframes
signal = (daily_rsi < 40) & (hourly_rsi < 30)
```

## Performance Considerations

1. **Vectorized Operations**: All indicators use pandas/numpy for fast calculations
2. **Caching**: Meta indicators cache API results to avoid repeated calls
3. **Async Support**: Data fetching and web scraping support async operations
4. **Memory Efficiency**: Large datasets are processed in chunks when needed

## Custom Indicators

To create custom indicators, inherit from the base `Indicator` class:

```python
from src.indicators.base import Indicator

class MyCustomIndicator(Indicator):
    def __init__(self, period=20):
        super().__init__(name="MyCustom")
        self.period = period
        
    def calculate(self, data, **kwargs):
        # Implement calculation logic
        result = data['close'].rolling(self.period).mean()
        return result
```