"""Debug data columns."""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import StockDataFetcher


async def main():
    """Check data columns."""
    fetcher = StockDataFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = await fetcher.fetch(
        symbol="SPY",
        start=start_date,
        end=end_date,
        interval="1d"
    )
    
    print("Data columns:", data.columns.tolist())
    print("\nFirst 5 rows:")
    print(data.head())
    

if __name__ == "__main__":
    asyncio.run(main())