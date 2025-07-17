"""
Download and cache historical market data for backtesting.
Supports multiple timeframes and efficient data storage.
"""

import os
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
import pickle
from pathlib import Path
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataDownloader:
    """Download and manage historical market data."""
    
    def __init__(self, config_path: str = "config/strategy_config.yaml"):
        """Initialize the downloader with configuration."""
        self.config = self._load_config(config_path)
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.cache_dir = self.data_dir / "cache"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def download_asset_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1h"
    ) -> pd.DataFrame:
        """
        Download data for a single asset.
        
        Args:
            symbol: Asset symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1m, 5m, 15m, 30m, 60m, 1h, 1d, 1wk, 1mo)
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_file = self.cache_dir / f"{symbol}_{interval}_{start_date}_{end_date}.pkl"
        
        # Check cache first
        if cache_file.exists():
            logger.info(f"Loading {symbol} {interval} data from cache")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
                
        logger.info(f"Downloading {symbol} {interval} data from {start_date} to {end_date}")
        
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            
            # For intraday data, we need to download in chunks (max 60 days per request)
            if interval in ['1h', '60m', '30m', '15m', '5m', '1m']:
                df_list = []
                current_start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                
                while current_start < end:
                    current_end = min(current_start + timedelta(days=59), end)
                    
                    df_chunk = ticker.history(
                        start=current_start.strftime('%Y-%m-%d'),
                        end=current_end.strftime('%Y-%m-%d'),
                        interval=interval
                    )
                    
                    if not df_chunk.empty:
                        df_list.append(df_chunk)
                        
                    current_start = current_end + timedelta(days=1)
                    
                if df_list:
                    df = pd.concat(df_list)
                else:
                    df = pd.DataFrame()
            else:
                # For daily/weekly/monthly data
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval
                )
                
            if df.empty:
                logger.warning(f"No data retrieved for {symbol}")
                return pd.DataFrame()
                
            # Clean the data
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Remove any duplicate indices
            df = df[~df.index.duplicated(keep='first')]
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
                
            logger.info(f"Downloaded {len(df)} rows for {symbol} {interval}")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {str(e)}")
            return pd.DataFrame()
            
    def download_all_assets(self, force_refresh: bool = False) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Download data for all assets and timeframes specified in config.
        
        Args:
            force_refresh: Force re-download even if cache exists
            
        Returns:
            Nested dict: {symbol: {timeframe: DataFrame}}
        """
        if force_refresh:
            logger.info("Force refresh enabled, clearing cache")
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
                
        assets = self.config['assets']
        start_date = self.config['data']['start_date']
        end_date = self.config['data']['end_date']
        
        # Map config timeframes to yfinance intervals
        interval_map = {
            '1H': '1h',
            '4H': '1h',  # Will resample from 1h
            '1D': '1d',
            '1W': '1wk',
            '1M': '1mo'
        }
        
        all_data = {}
        
        for symbol in assets:
            logger.info(f"\nProcessing {symbol}...")
            symbol_data = {}
            
            # Download base data
            for tf_config, yf_interval in interval_map.items():
                if tf_config == '4H':
                    # For 4H, download 1H and resample
                    base_data = self.download_asset_data(
                        symbol, start_date, end_date, '1h'
                    )
                    if not base_data.empty:
                        # Resample to 4H
                        symbol_data['4H'] = base_data.resample('4H').agg({
                            'Open': 'first',
                            'High': 'max',
                            'Low': 'min',
                            'Close': 'last',
                            'Volume': 'sum'
                        }).dropna()
                else:
                    data = self.download_asset_data(
                        symbol, start_date, end_date, yf_interval
                    )
                    if not data.empty:
                        symbol_data[tf_config] = data
                        
            all_data[symbol] = symbol_data
            
        # Save complete dataset
        complete_data_file = self.processed_dir / "complete_market_data.pkl"
        with open(complete_data_file, 'wb') as f:
            pickle.dump(all_data, f)
            
        logger.info(f"\nData download complete. Saved to {complete_data_file}")
        return all_data
    
    def get_spreads_and_fees(self, symbol: str, price: float, volume: float, 
                           avg_volume: float, volatility: float) -> Dict[str, float]:
        """
        Calculate realistic spread and fees for a given trade.
        
        Args:
            symbol: Asset symbol
            price: Current price
            volume: Current volume
            avg_volume: Average volume
            volatility: Current volatility (ATR/price)
            
        Returns:
            Dict with spread, commission, slippage, total_cost
        """
        costs_config = self.config['trading_costs']
        
        # Base spread
        base_spread_pct = costs_config['spread']['base_spread_pct'].get(
            symbol, 0.0002  # Default 2 basis points
        )
        
        # Adjust spread for volatility
        vol_multiplier = 1 + (volatility - 0.01) * costs_config['spread']['volatility_multiplier']
        vol_multiplier = max(0.5, min(3.0, vol_multiplier))  # Cap between 0.5x and 3x
        
        # Adjust spread for volume
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        if volume_ratio < 0.5:
            volume_multiplier = costs_config['spread']['volume_impact']['low_volume_multiplier']
        elif volume_ratio > 2.0:
            volume_multiplier = costs_config['spread']['volume_impact']['high_volume_multiplier']
        else:
            volume_multiplier = 1.0
            
        # Final spread
        spread_pct = base_spread_pct * vol_multiplier * volume_multiplier
        spread_cost = price * spread_pct
        
        # Commission
        commission_pct = costs_config['commission']['percentage']
        commission_fixed = costs_config['commission']['fixed']
        commission = max(
            price * commission_pct + commission_fixed,
            costs_config['commission']['minimum']
        )
        
        # Slippage
        slippage_pct = costs_config['slippage']['base_slippage_pct']
        slippage = price * slippage_pct
        
        return {
            'spread': spread_cost,
            'commission': commission,
            'slippage': slippage,
            'total_cost': spread_cost + commission + slippage,
            'total_cost_pct': (spread_cost + commission + slippage) / price
        }


def load_cached_data(symbol: str, timeframe: str = '1D') -> Optional[pd.DataFrame]:
    """
    Load cached data for a specific symbol and timeframe.
    
    Args:
        symbol: Asset symbol
        timeframe: Timeframe (1H, 4H, 1D, 1W, 1M)
        
    Returns:
        DataFrame with OHLCV data or None if not found
    """
    # Try to load from complete market data pickle
    cache_file = Path("data/processed/complete_market_data.pkl")
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                all_data = pickle.load(f)
                
            if symbol in all_data and timeframe in all_data[symbol]:
                data = all_data[symbol][timeframe]
                if not data.empty:
                    # Ensure column names are lowercase
                    data.columns = [col.lower() for col in data.columns]
                    return data
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
    
    # Try individual cache file
    downloader = MarketDataDownloader()
    individual_cache = downloader.cache_dir / f"{symbol}_{timeframe}_2019-01-01_2024-01-01.pkl"
    
    if individual_cache.exists():
        try:
            with open(individual_cache, 'rb') as f:
                data = pickle.load(f)
                if not data.empty:
                    data.columns = [col.lower() for col in data.columns]
                    return data
        except Exception as e:
            logger.error(f"Error loading individual cache: {e}")
    
    return None




def main():
    """Main function to download all data."""
    downloader = MarketDataDownloader()
    
    # Download all data
    all_data = downloader.download_all_assets(force_refresh=False)
    
    # Print summary
    print("\nData Download Summary:")
    print("-" * 50)
    for symbol, timeframes in all_data.items():
        print(f"\n{symbol}:")
        for tf, df in timeframes.items():
            if not df.empty:
                print(f"  {tf}: {len(df)} bars, from {df.index[0]} to {df.index[-1]}")
                
    # Example: Calculate trading costs
    print("\nExample Trading Costs (SPY at $450):")
    costs = downloader.get_spreads_and_fees(
        symbol='SPY',
        price=450,
        volume=80_000_000,
        avg_volume=75_000_000,
        volatility=0.012  # 1.2% volatility
    )
    for key, value in costs.items():
        if key == 'total_cost_pct':
            print(f"  {key}: {value:.4%}")
        else:
            print(f"  {key}: ${value:.4f}")


if __name__ == "__main__":
    main()