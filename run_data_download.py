#!/usr/bin/env python3
"""
Standalone script to download market data.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now import directly
from data.download_historical_data import MarketDataDownloader

if __name__ == "__main__":
    downloader = MarketDataDownloader()
    print("Starting data download...")
    all_data = downloader.download_all_assets(force_refresh=False)
    print("\nDownload complete!")
    
    # Summary
    for symbol in all_data:
        print(f"\n{symbol}:")
        for tf in all_data[symbol]:
            df = all_data[symbol][tf]
            if not df.empty:
                print(f"  {tf}: {len(df)} bars")