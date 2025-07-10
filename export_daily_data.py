#!/usr/bin/env python3
"""Export daily data to CSV files for easier access"""

import pandas as pd
import pickle
import os

# Load the complete market data
with open('data/processed/complete_market_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

# Export daily data for each symbol
for symbol, timeframes in all_data.items():
    if '1D' in timeframes:
        df = timeframes['1D']
        # Save with lowercase column names
        df.columns = df.columns.str.lower()
        # Ensure index is named 'date'
        df.index.name = 'date'
        
        # Save to CSV
        output_path = f'data/{symbol}.csv'
        df.to_csv(output_path)
        print(f"✓ Exported {symbol} daily data: {len(df)} rows to {output_path}")

print("\n✅ Daily data export complete!")