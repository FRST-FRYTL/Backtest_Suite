# Meta Indicators

Meta indicators provide market-wide sentiment analysis and alternative data sources beyond traditional technical indicators. The Backtest Suite includes three powerful meta indicators:

## Fear and Greed Index

The Fear and Greed Index measures market sentiment on a scale from 0 (extreme fear) to 100 (extreme greed).

### Features

- **Multiple Data Sources**: Supports both Alternative.me (free) and CNN Money APIs
- **Real-time Updates**: Fetch current market sentiment
- **Historical Analysis**: Access historical sentiment data
- **Signal Generation**: Automated trading signals based on sentiment extremes
- **Correlation Analysis**: Analyze relationship between sentiment and price movements

### Usage

```python
from src.indicators.fear_greed import FearGreedIndex

# Initialize with Alternative.me API (free, no key required)
fg_index = FearGreedIndex(source="alternative")

# Fetch current sentiment
current = await fg_index.fetch_current()
print(f"Current Fear & Greed: {current['value']} ({current['classification']})")

# Fetch historical data
historical = await fg_index.fetch_historical(limit=30)  # Last 30 days

# Generate trading signals
signals = fg_index.get_signals(historical)
```

### Trading Signals

- **Extreme Fear (<20)**: Potential buying opportunity (contrarian)
- **Extreme Greed (>80)**: Consider taking profits
- **Fear/Greed Reversals**: Momentum changes from extremes
- **Entering Fear/Greed**: Trend changes in sentiment

### CNN Money Source

The CNN Money source provides additional context:

```python
fg_cnn = FearGreedIndex(source="cnn")
cnn_data = await fg_cnn.fetch_current()
# Includes: current value, 1-week ago, 1-month ago, 1-year ago
```

## Insider Trading Data

Track insider buying and selling patterns to gauge management confidence.

### Features

- **Real-time Scraping**: Live data from OpenInsider.com
- **Ticker Filtering**: Track specific stocks or market-wide activity
- **Cluster Buy Detection**: Identify multiple insiders buying
- **Significant Trade Alerts**: High-value transactions (>$1M)
- **Sentiment Analysis**: Bullish/bearish scoring based on activity

### Usage

```python
from src.indicators.insider import InsiderTrading

async with InsiderTrading() as insider:
    # Fetch latest trades for a specific stock
    trades = await insider.fetch_latest_trades(ticker="AAPL", limit=20)
    
    # Find cluster buys (multiple insiders buying)
    cluster_buys = await insider.fetch_cluster_buys(days=7)
    
    # Get significant trades
    big_trades = await insider.fetch_significant_buys(min_value=1000000)
    
    # Analyze sentiment
    sentiment = insider.analyze_sentiment(trades)
    print(f"Bullish Score: {sentiment['bullish_score']}/100")
```

### Key Metrics

- **Buy/Sell Ratio**: Higher ratios indicate bullish sentiment
- **Value Ratio**: Dollar-weighted buy/sell comparison
- **Bullish Score (0-100)**: Composite sentiment indicator
- **Ownership Change %**: Size of trades relative to holdings

### Trading Signals

- **Cluster Buys**: Multiple insiders buying = strong bullish signal
- **CEO/CFO Purchases**: C-suite buying often precedes positive news
- **10b5-1 Sales**: Pre-planned sales less bearish than discretionary
- **Small Ownership Changes**: Large % changes more significant

## Max Pain Calculator

Options-based support and resistance levels using max pain theory.

### Features

- **Max Pain Calculation**: Price where option holders lose the most
- **Multiple Expirations**: Analyze across different expiry dates
- **Support/Resistance Levels**: Options-based price levels
- **Gamma Exposure**: Identify potential squeeze levels
- **Put/Call Ratios**: Market positioning analysis

### Usage

```python
from src.indicators.max_pain import MaxPain

max_pain_calc = MaxPain()

# Calculate for nearest expiration
result = max_pain_calc.calculate("SPY")
print(f"Max Pain: ${result['max_pain_price']}")
print(f"Current Price: ${result['current_price']}")
print(f"Deviation: {result['price_vs_max_pain']}%")

# Multiple expirations
multi_exp = max_pain_calc.calculate_all_expirations("SPY", max_expirations=4)

# Generate trading signals
signals = max_pain_calc.get_signals("SPY", current_price, result)
```

### Key Concepts

- **Max Pain Price**: Strike price with maximum option holder loss
- **Price Magnetism**: Prices often gravitate toward max pain near expiry
- **Gamma Levels**: High gamma strikes can cause price acceleration
- **Support/Resistance**: High open interest strikes act as price levels

### Trading Signals

- **Max Pain Magnet**: Price within 5% of max pain
- **Extreme Deviation**: Price >10% away from max pain
- **Near Resistance**: Approaching high call OI strikes
- **Near Support**: Approaching high put OI strikes
- **Gamma Squeeze Potential**: Near high gamma strikes

## Combining Meta Indicators

The real power comes from combining multiple meta indicators:

```python
# Example: Combined analysis
async def analyze_opportunity(ticker):
    # Market sentiment
    fg = FearGreedIndex()
    sentiment = await fg.fetch_current()
    
    # Insider activity
    async with InsiderTrading() as insider:
        trades = await insider.fetch_latest_trades(ticker=ticker)
        insider_sentiment = insider.analyze_sentiment(trades)
    
    # Options positioning
    mp = MaxPain()
    max_pain_data = mp.calculate(ticker)
    
    # Combined signal
    if (sentiment['value'] < 30 and  # Fear in market
        insider_sentiment['bullish_score'] > 70 and  # Insiders buying
        max_pain_data['price_vs_max_pain'] < -10):  # Below max pain
        return "STRONG BUY SIGNAL"
```

## Best Practices

1. **Cache Management**: All indicators use caching to reduce API calls
2. **Rate Limiting**: Respect API limits, especially for free services
3. **Error Handling**: Always handle API failures gracefully
4. **Confirmation**: Use multiple indicators for confirmation
5. **Backtesting**: Test meta indicator strategies historically

## Example Strategy

See `examples/meta_indicators_example.py` for a complete demonstration of:
- Fetching and analyzing each indicator
- Combining indicators for stronger signals
- Integration with the backtesting framework
- Real-world usage patterns

## API Requirements

- **Fear & Greed (Alternative.me)**: Free, no API key required
- **Fear & Greed (CNN)**: Free, no API key required
- **Insider Trading**: Web scraping, no API key required
- **Max Pain**: Requires options data access (via yfinance)