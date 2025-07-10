#!/usr/bin/env python3
"""
Script to download all historical market data.
Run this to populate the data directory with 5 years of market data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.download_historical_data import MarketDataDownloader
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Download all market data."""
    print("=" * 60)
    print("DOWNLOADING 5 YEARS OF MARKET DATA")
    print("=" * 60)
    print("\nThis will download data for:")
    print("- Assets: SPY, QQQ, AAPL, MSFT, JPM, XLE, GLD, IWM")
    print("- Timeframes: 1H, 4H, 1D, 1W, 1M")
    print("- Period: 2019-2024")
    print("\nData will be cached for future use.")
    print("-" * 60)
    
    downloader = MarketDataDownloader()
    
    # Download all data
    print("\nStarting download...")
    all_data = downloader.download_all_assets(force_refresh=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    
    total_bars = 0
    for symbol, timeframes in all_data.items():
        print(f"\n{symbol}:")
        for tf, df in timeframes.items():
            if not df.empty:
                bars = len(df)
                total_bars += bars
                print(f"  {tf}: {bars:,} bars | {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
                
    print(f"\nTotal data points: {total_bars:,}")
    print(f"Data saved to: data/processed/complete_market_data.pkl")
    
    # Example trading costs
    print("\n" + "=" * 60)
    print("EXAMPLE TRADING COSTS")
    print("=" * 60)
    
    # SPY example
    print("\nSPY (liquid ETF):")
    spy_costs = downloader.get_spreads_and_fees(
        symbol='SPY',
        price=450,
        volume=80_000_000,
        avg_volume=75_000_000,
        volatility=0.012
    )
    for key, value in spy_costs.items():
        if key == 'total_cost_pct':
            print(f"  {key}: {value:.4%}")
        else:
            print(f"  {key}: ${value:.4f}")
            
    # AAPL example
    print("\nAAPL (liquid stock):")
    aapl_costs = downloader.get_spreads_and_fees(
        symbol='AAPL',
        price=180,
        volume=50_000_000,
        avg_volume=55_000_000,
        volatility=0.018
    )
    for key, value in aapl_costs.items():
        if key == 'total_cost_pct':
            print(f"  {key}: {value:.4%}")
        else:
            print(f"  {key}: ${value:.4f}")
            
    # Low liquidity example
    print("\nLow liquidity scenario:")
    low_liq_costs = downloader.get_spreads_and_fees(
        symbol='XLE',
        price=85,
        volume=5_000_000,
        avg_volume=15_000_000,
        volatility=0.025
    )
    for key, value in low_liq_costs.items():
        if key == 'total_cost_pct':
            print(f"  {key}: {value:.4%}")
        else:
            print(f"  {key}: ${value:.4f}")
            
    print("\n✅ Data download complete! You can now run backtests with real market data.")


if __name__ == "__main__":
    main()