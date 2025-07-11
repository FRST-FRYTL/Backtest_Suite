"""Fear and Greed Index API integration."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List

import pandas as pd
import aiohttp
import requests

try:
    from ..data.cache import DataCache
except ImportError:
    from data.cache import DataCache


class FearGreedIndex:
    """
    Fear and Greed Index integration.
    
    Fetches sentiment data from the Fear and Greed Index API to gauge market sentiment.
    """
    
    # API endpoints
    CNN_API = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    ALTERNATIVE_API = "https://api.alternative.me/fng/"
    
    def __init__(self, cache_dir: str = "data/cache", source: str = "alternative"):
        """
        Initialize Fear and Greed Index fetcher.
        
        Args:
            cache_dir: Directory for caching data
            source: API source ('alternative' or 'cnn')
        """
        self.cache = DataCache(cache_dir)
        self.source = source.lower()
        
        if self.source not in ['alternative', 'cnn']:
            raise ValueError(f"Invalid source: {source}. Must be 'alternative' or 'cnn'")
            
    async def fetch_current(self) -> Dict:
        """
        Fetch current Fear and Greed Index value.
        
        Returns:
            Dictionary with current index data
        """
        if self.source == 'alternative':
            return await self._fetch_alternative_current()
        else:
            return await self._fetch_cnn_current()
            
    async def fetch_historical(
        self,
        limit: Optional[int] = None,
        date_from: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch historical Fear and Greed Index data.
        
        Args:
            limit: Number of days to fetch (alternative API)
            date_from: Start date for historical data
            
        Returns:
            DataFrame with historical index data
        """
        # Check cache first
        if limit:
            cache_key = f"fear_greed_{self.source}_{limit}days"
        elif date_from:
            cache_key = f"fear_greed_{self.source}_{date_from.strftime('%Y%m%d')}"
        else:
            cache_key = f"fear_greed_{self.source}_all"
            
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
            
        # Fetch from API
        if self.source == 'alternative':
            data = await self._fetch_alternative_historical(limit)
        else:
            data = await self._fetch_cnn_historical()
            
        # Filter by date if specified
        if date_from and not limit:
            data = data[data.index >= date_from]
            
        # Cache the data
        self.cache.set(cache_key, data, expire=3600)  # 1 hour cache
        
        return data
        
    async def _fetch_alternative_current(self) -> Dict:
        """Fetch current index from Alternative.me API."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.ALTERNATIVE_API) as response:
                    data = await response.json()
                    
                    if 'data' in data and len(data['data']) > 0:
                        current = data['data'][0]
                        return {
                            'value': int(current['value']),
                            'classification': current['value_classification'],
                            'timestamp': datetime.fromtimestamp(int(current['timestamp'])),
                            'time_until_update': current.get('time_until_update', 0)
                        }
                    else:
                        raise ValueError("No data received from API")
                        
            except Exception as e:
                print(f"Error fetching current Fear & Greed Index: {e}")
                raise
                
    async def _fetch_alternative_historical(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Fetch historical data from Alternative.me API."""
        params = {}
        if limit:
            params['limit'] = limit
            
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.ALTERNATIVE_API, params=params) as response:
                    data = await response.json()
                    
                    if 'data' in data:
                        # Convert to DataFrame
                        df_data = []
                        for item in data['data']:
                            df_data.append({
                                'timestamp': datetime.fromtimestamp(int(item['timestamp'])),
                                'value': int(item['value']),
                                'classification': item['value_classification']
                            })
                            
                        df = pd.DataFrame(df_data)
                        df.set_index('timestamp', inplace=True)
                        df.sort_index(inplace=True)
                        
                        return df
                    else:
                        raise ValueError("No data received from API")
                        
            except Exception as e:
                print(f"Error fetching historical Fear & Greed Index: {e}")
                raise
                
    async def _fetch_cnn_current(self) -> Dict:
        """Fetch current index from CNN API."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.CNN_API) as response:
                    data = await response.json()
                    
                    current = data['fear_and_greed']
                    return {
                        'value': float(current['score']),
                        'classification': current['rating'],
                        'timestamp': datetime.now(),
                        'previous_close': float(current['previous_close']),
                        'previous_1_week': float(current['previous_1_week']),
                        'previous_1_month': float(current['previous_1_month']),
                        'previous_1_year': float(current['previous_1_year'])
                    }
                    
            except Exception as e:
                print(f"Error fetching CNN Fear & Greed Index: {e}")
                raise
                
    async def _fetch_cnn_historical(self) -> pd.DataFrame:
        """Fetch historical data from CNN API."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.CNN_API) as response:
                    data = await response.json()
                    
                    # Extract historical data
                    historical = data['fear_and_greed_historical']['data']
                    
                    df_data = []
                    for item in historical:
                        # CNN uses millisecond timestamps
                        timestamp = datetime.fromtimestamp(item['x'] / 1000)
                        df_data.append({
                            'timestamp': timestamp,
                            'value': float(item['y']),
                            'classification': self._classify_value(float(item['y']))
                        })
                        
                    df = pd.DataFrame(df_data)
                    df.set_index('timestamp', inplace=True)
                    df.sort_index(inplace=True)
                    
                    return df
                    
            except Exception as e:
                print(f"Error fetching CNN historical data: {e}")
                raise
                
    def _classify_value(self, value: float) -> str:
        """Classify Fear & Greed value."""
        if value <= 25:
            return "Extreme Fear"
        elif value <= 44:
            return "Fear"
        elif value <= 55:
            return "Neutral"
        elif value <= 75:
            return "Greed"
        else:
            return "Extreme Greed"
            
    def fetch_sync(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Synchronous version of fetch_historical.
        
        Args:
            limit: Number of days to fetch
            
        Returns:
            DataFrame with historical index data
        """
        return asyncio.run(self.fetch_historical(limit))
        
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Fear & Greed Index.
        
        Args:
            data: DataFrame with Fear & Greed values
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        
        # Extreme levels
        signals['extreme_fear'] = data['value'] <= 20
        signals['extreme_greed'] = data['value'] >= 80
        
        # Level changes
        signals['entering_fear'] = (
            (data['value'] < 45) & 
            (data['value'].shift(1) >= 45)
        )
        
        signals['entering_greed'] = (
            (data['value'] > 55) & 
            (data['value'].shift(1) <= 55)
        )
        
        # Reversals from extremes
        signals['fear_reversal'] = (
            (data['value'] > 25) & 
            (data['value'].shift(1) <= 25)
        )
        
        signals['greed_reversal'] = (
            (data['value'] < 75) & 
            (data['value'].shift(1) >= 75)
        )
        
        # Momentum
        signals['fear_momentum'] = data['value'].diff() < -5
        signals['greed_momentum'] = data['value'].diff() > 5
        
        return signals
        
    def correlation_analysis(
        self,
        fear_greed_data: pd.DataFrame,
        price_data: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Analyze correlation between Fear & Greed Index and price movements.
        
        Args:
            fear_greed_data: Fear & Greed Index data
            price_data: Price data (must have same index)
            window: Rolling correlation window
            
        Returns:
            DataFrame with correlation analysis
        """
        # Align data
        aligned = pd.DataFrame(index=price_data.index)
        aligned['price'] = price_data['close'] if 'close' in price_data.columns else price_data
        
        # Merge with fear & greed data
        aligned = aligned.join(fear_greed_data[['value']], how='left')
        aligned['value'] = aligned['value'].fillna(method='ffill')
        
        # Calculate returns
        aligned['price_return'] = aligned['price'].pct_change()
        aligned['fg_change'] = aligned['value'].diff()
        
        # Rolling correlation
        aligned['correlation'] = aligned['price_return'].rolling(
            window=window
        ).corr(aligned['fg_change'])
        
        # Directional accuracy
        aligned['same_direction'] = (
            (aligned['price_return'] > 0) == (aligned['fg_change'] > 0)
        )
        
        aligned['directional_accuracy'] = aligned['same_direction'].rolling(
            window=window
        ).mean()
        
        return aligned[['correlation', 'directional_accuracy', 'same_direction']]