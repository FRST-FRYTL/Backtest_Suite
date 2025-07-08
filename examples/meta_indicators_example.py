"""
Example demonstrating the usage of meta indicators in the Backtest Suite.

This example shows how to use:
1. Fear and Greed Index - Market sentiment indicator
2. Insider Trading Data - Track insider buying/selling patterns
3. Max Pain Calculator - Options-based support/resistance levels
"""

import asyncio
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path for imports
import sys
sys.path.append('..')

from src.indicators.fear_greed import FearGreedIndex
from src.indicators.insider import InsiderTrading
from src.indicators.max_pain import MaxPain
from src.data.fetcher import StockDataFetcher


async def fear_greed_example():
    """Demonstrate Fear and Greed Index usage."""
    print("\n" + "="*60)
    print("FEAR AND GREED INDEX EXAMPLE")
    print("="*60)
    
    # Initialize with Alternative.me API (free, no key required)
    fg_index = FearGreedIndex(source="alternative")
    
    # Fetch current Fear & Greed value
    current = await fg_index.fetch_current()
    print(f"\nCurrent Fear & Greed Index:")
    print(f"  Value: {current['value']}")
    print(f"  Classification: {current['classification']}")
    print(f"  Timestamp: {current['timestamp']}")
    
    # Fetch historical data (last 30 days)
    historical = await fg_index.fetch_historical(limit=30)
    print(f"\nHistorical Fear & Greed (last 30 days):")
    print(f"  Average: {historical['value'].mean():.1f}")
    print(f"  Min: {historical['value'].min()} ({historical.loc[historical['value'].idxmin(), 'classification']})")
    print(f"  Max: {historical['value'].max()} ({historical.loc[historical['value'].idxmax(), 'classification']})")
    
    # Generate trading signals
    signals = fg_index.get_signals(historical)
    recent_signals = signals.tail(5)
    print(f"\nRecent Trading Signals:")
    for idx, row in recent_signals.iterrows():
        active_signals = [col for col in signals.columns if row[col]]
        if active_signals:
            print(f"  {idx.date()}: {', '.join(active_signals)}")
    
    # Try CNN Money source (different data format)
    print("\n" + "-"*40)
    print("CNN Money Fear & Greed Index:")
    fg_cnn = FearGreedIndex(source="cnn")
    try:
        cnn_current = await fg_cnn.fetch_current()
        print(f"  Current: {cnn_current['value']:.1f} ({cnn_current['classification']})")
        print(f"  1 Week Ago: {cnn_current['previous_1_week']:.1f}")
        print(f"  1 Month Ago: {cnn_current['previous_1_month']:.1f}")
    except Exception as e:
        print(f"  Error fetching CNN data: {e}")


async def insider_trading_example():
    """Demonstrate Insider Trading data usage."""
    print("\n" + "="*60)
    print("INSIDER TRADING DATA EXAMPLE")
    print("="*60)
    
    async with InsiderTrading() as insider:
        # Fetch latest insider trades for a specific ticker
        ticker = "AAPL"
        print(f"\nFetching insider trades for {ticker}...")
        trades = await insider.fetch_latest_trades(ticker=ticker, limit=10)
        
        if not trades.empty:
            print(f"\nRecent {ticker} Insider Trades:")
            for _, trade in trades.head(5).iterrows():
                print(f"  {trade.get('filing_date', 'N/A')}: {trade.get('insider', 'Unknown')} "
                      f"({trade.get('title', 'N/A')}) - {trade.get('transaction_type', 'N/A')} "
                      f"${trade.get('value', 0):,.0f}")
        
        # Fetch cluster buys (multiple insiders buying)
        print("\n" + "-"*40)
        print("Recent Cluster Buys (Multiple Insiders):")
        cluster_buys = await insider.fetch_cluster_buys(days=7)
        
        if not cluster_buys.empty:
            # Group by ticker to show companies with multiple buyers
            by_ticker = cluster_buys.groupby('ticker').agg({
                'value': ['count', 'sum'],
                'insider': lambda x: list(x)[:3]  # First 3 insiders
            })
            by_ticker.columns = ['trade_count', 'total_value', 'insiders']
            by_ticker = by_ticker.sort_values('trade_count', ascending=False).head(5)
            
            for ticker, row in by_ticker.iterrows():
                print(f"  {ticker}: {row['trade_count']} buys totaling ${row['total_value']:,.0f}")
                print(f"    Buyers: {', '.join(row['insiders'][:2])}...")
        
        # Fetch significant buys (high value)
        print("\n" + "-"*40)
        print("Significant Insider Buys (>$1M):")
        significant = await insider.fetch_significant_buys(min_value=1000000)
        
        if not significant.empty:
            for _, trade in significant.head(5).iterrows():
                print(f"  {trade.get('ticker', 'N/A')}: {trade.get('insider', 'Unknown')} - "
                      f"${trade.get('value', 0):,.0f} on {trade.get('filing_date', 'N/A')}")
        
        # Analyze sentiment for all recent trades
        all_trades = await insider.fetch_latest_trades(limit=100)
        sentiment = insider.analyze_sentiment(all_trades)
        
        print("\n" + "-"*40)
        print("Overall Insider Sentiment Analysis:")
        print(f"  Total Trades: {sentiment['total_trades']}")
        print(f"  Buy/Sell Ratio: {sentiment['buy_sell_ratio']:.2f}")
        print(f"  Bullish Score: {sentiment['bullish_score']:.1f}/100")
        print(f"  Total Buy Value: ${sentiment['buy_value']:,.0f}")
        print(f"  Total Sell Value: ${sentiment['sell_value']:,.0f}")


def max_pain_example():
    """Demonstrate Max Pain calculator usage."""
    print("\n" + "="*60)
    print("MAX PAIN CALCULATOR EXAMPLE")
    print("="*60)
    
    max_pain_calc = MaxPain()
    ticker = "SPY"
    
    # Calculate max pain for nearest expiration
    print(f"\nCalculating Max Pain for {ticker}...")
    max_pain_data = max_pain_calc.calculate(ticker)
    
    if max_pain_data.get('max_pain_price'):
        print(f"\nMax Pain Analysis:")
        print(f"  Max Pain Price: ${max_pain_data['max_pain_price']:.2f}")
        print(f"  Current Price: ${max_pain_data['current_price']:.2f}")
        print(f"  Deviation: {max_pain_data['price_vs_max_pain']:.1f}%")
        print(f"  Put/Call Ratio: {max_pain_data['put_call_ratio']:.2f}")
        
        # Show support and resistance levels
        print(f"\nOptions-Based Levels:")
        print(f"  Resistance: {', '.join(f'${r:.2f}' for r in max_pain_data['resistance_levels'])}")
        print(f"  Support: {', '.join(f'${s:.2f}' for s in max_pain_data['support_levels'])}")
        
        # Show gamma levels
        if max_pain_data.get('gamma_levels'):
            print(f"\nHigh Gamma Strikes (potential squeeze levels):")
            for gamma in max_pain_data['gamma_levels'][:3]:
                print(f"  ${gamma['strike']:.2f} - {gamma['significance']} gamma exposure")
        
        # Generate trading signals
        current_price = max_pain_data['current_price']
        signals = max_pain_calc.get_signals(ticker, current_price, max_pain_data)
        
        print(f"\nTrading Signals:")
        active_signals = [signal for signal, active in signals.items() if active]
        for signal in active_signals:
            print(f"  ✓ {signal.replace('_', ' ').title()}")
    else:
        print(f"  Error: {max_pain_data.get('error', 'Unknown error')}")
    
    # Calculate for multiple expirations
    print("\n" + "-"*40)
    print("Max Pain Across Multiple Expirations:")
    multi_exp = max_pain_calc.calculate_all_expirations(ticker, max_expirations=4)
    
    if not multi_exp.empty:
        for _, row in multi_exp.iterrows():
            print(f"  {row['expiration']}: Max Pain ${row['max_pain']:.2f} "
                  f"(current {row['deviation_pct']:+.1f}% away) - "
                  f"{row['days_to_expiry']} days")


async def combined_analysis_example():
    """Demonstrate combining multiple meta indicators."""
    print("\n" + "="*60)
    print("COMBINED META INDICATORS ANALYSIS")
    print("="*60)
    
    ticker = "TSLA"
    print(f"\nAnalyzing {ticker} with multiple meta indicators...")
    
    # Initialize indicators
    fg_index = FearGreedIndex()
    max_pain_calc = MaxPain()
    
    # Fetch current market sentiment
    fear_greed = await fg_index.fetch_current()
    print(f"\nMarket Sentiment: {fear_greed['value']} ({fear_greed['classification']})")
    
    # Fetch max pain data
    max_pain_data = max_pain_calc.calculate(ticker)
    if max_pain_data.get('max_pain_price'):
        print(f"\n{ticker} Options Analysis:")
        print(f"  Max Pain: ${max_pain_data['max_pain_price']:.2f}")
        print(f"  Current Price: ${max_pain_data['current_price']:.2f}")
        print(f"  Put/Call Ratio: {max_pain_data['put_call_ratio']:.2f}")
    
    # Fetch insider data
    async with InsiderTrading() as insider:
        trades = await insider.fetch_latest_trades(ticker=ticker, limit=20)
        if not trades.empty:
            sentiment = insider.analyze_sentiment(trades)
            print(f"\n{ticker} Insider Activity:")
            print(f"  Recent Trades: {sentiment['total_trades']}")
            print(f"  Buy/Sell Ratio: {sentiment['buy_sell_ratio']:.2f}")
            print(f"  Bullish Score: {sentiment['bullish_score']:.1f}/100")
    
    # Combined signal interpretation
    print(f"\n" + "-"*40)
    print("COMBINED SIGNAL INTERPRETATION:")
    
    # Market sentiment signal
    if fear_greed['value'] < 30:
        print("  ⚠️  Market Sentiment: EXTREME FEAR - Potential buying opportunity")
    elif fear_greed['value'] > 70:
        print("  ⚠️  Market Sentiment: EXTREME GREED - Consider taking profits")
    else:
        print(f"  ✓ Market Sentiment: {fear_greed['classification']} - Normal conditions")
    
    # Options positioning signal
    if max_pain_data.get('max_pain_price'):
        deviation = max_pain_data['price_vs_max_pain']
        if abs(deviation) > 10:
            direction = "above" if deviation > 0 else "below"
            print(f"  ⚠️  Options: Price {abs(deviation):.1f}% {direction} max pain - "
                  f"Potential reversion to ${max_pain_data['max_pain_price']:.2f}")
        else:
            print(f"  ✓ Options: Price near max pain - Stable positioning")
    
    # Insider activity signal
    if not trades.empty and sentiment['bullish_score'] > 70:
        print(f"  ⚠️  Insiders: High bullish score ({sentiment['bullish_score']:.0f}/100) - "
              f"Insiders are buying")
    elif not trades.empty and sentiment['bullish_score'] < 30:
        print(f"  ⚠️  Insiders: Low bullish score ({sentiment['bullish_score']:.0f}/100) - "
              f"Insiders are selling")


async def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("META INDICATORS DEMONSTRATION")
    print("Backtest Suite - Advanced Market Analysis Tools")
    print("="*80)
    
    try:
        # Run Fear & Greed example
        await fear_greed_example()
        
        # Run Insider Trading example
        await insider_trading_example()
        
        # Run Max Pain example (synchronous)
        max_pain_example()
        
        # Run combined analysis
        await combined_analysis_example()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)


if __name__ == "__main__":
    # Run the async examples
    asyncio.run(main())