"""Insider trading data scraper from OpenInsider."""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import aiohttp
from bs4 import BeautifulSoup

try:
    from ..data.cache import DataCache
except ImportError:
    from data.cache import DataCache


class InsiderTrading:
    """
    Scrapes and analyzes insider trading data from OpenInsider.
    """
    
    BASE_URL = "http://openinsider.com"
    
    # Transaction type mappings
    TRANSACTION_TYPES = {
        'P': 'Purchase',
        'S': 'Sale',
        'A': 'Automatic Sale',
        'M': 'Option Exercise',
        'G': 'Gift',
        'D': 'Disposition',
        'F': 'Payment of Tax',
        'I': 'Discretionary',
        'J': 'Other',
    }
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize insider trading scraper.
        
        Args:
            cache_dir: Directory for caching scraped data
        """
        self.cache = DataCache(cache_dir)
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    async def fetch_latest_trades(
        self,
        ticker: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch latest insider trades.
        
        Args:
            ticker: Specific ticker to filter (None for all)
            limit: Maximum number of trades to fetch
            
        Returns:
            DataFrame with insider trading data
        """
        # Check cache
        cache_key = f"insider_latest_{ticker or 'all'}_{limit}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
            
        # Construct URL
        if ticker:
            url = f"{self.BASE_URL}/screener?s={ticker}"
        else:
            url = f"{self.BASE_URL}/latest-insider-trading"
            
        # Fetch and parse
        data = await self._fetch_and_parse(url, limit)
        
        # Cache for 1 hour
        if data is not None and not data.empty:
            self.cache.set(cache_key, data, expire=3600)
            
        return data
        
    async def fetch_cluster_buys(self, days: int = 30) -> pd.DataFrame:
        """
        Fetch cluster buys (multiple insiders buying).
        
        Args:
            days: Number of days to look back
            
        Returns:
            DataFrame with cluster buy data
        """
        cache_key = f"insider_cluster_buys_{days}days"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
            
        url = f"{self.BASE_URL}/cluster-buys"
        data = await self._fetch_and_parse(url, limit=200)
        
        if data is not None and not data.empty:
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            data = data[data['filing_date'] >= cutoff_date]
            self.cache.set(cache_key, data, expire=3600)
            
        return data
        
    async def fetch_significant_buys(
        self,
        min_value: float = 1000000
    ) -> pd.DataFrame:
        """
        Fetch significant insider buys by value.
        
        Args:
            min_value: Minimum transaction value
            
        Returns:
            DataFrame with significant buys
        """
        cache_key = f"insider_significant_{min_value}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
            
        url = f"{self.BASE_URL}/insider-purchases-25k"
        data = await self._fetch_and_parse(url, limit=200)
        
        if data is not None and not data.empty:
            # Filter by value
            data = data[data['value'] >= min_value]
            self.cache.set(cache_key, data, expire=3600)
            
        return data
        
    async def _fetch_and_parse(self, url: str, limit: int) -> pd.DataFrame:
        """
        Fetch and parse insider trading table from URL.
        
        Args:
            url: URL to fetch
            limit: Maximum number of rows
            
        Returns:
            DataFrame with parsed data
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        try:
            async with self.session.get(url) as response:
                html = await response.text()
                
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find the main table
            table = soup.find('table', {'class': 'tinytable'})
            if not table:
                print(f"No table found at {url}")
                return pd.DataFrame()
                
            # Parse table headers
            headers = []
            header_row = table.find('thead').find('tr')
            for th in header_row.find_all('th'):
                headers.append(th.text.strip())
                
            # Parse table rows
            rows = []
            tbody = table.find('tbody')
            for tr in tbody.find_all('tr')[:limit]:
                row = []
                for td in tr.find_all('td'):
                    # Clean text and extract links if present
                    text = td.text.strip()
                    link = td.find('a')
                    if link and link.get('href'):
                        row.append({'text': text, 'link': link['href']})
                    else:
                        row.append(text)
                rows.append(row)
                
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers)
            
            # Process columns
            df = self._process_dataframe(df)
            
            return df
            
        except Exception as e:
            print(f"Error fetching insider data: {e}")
            return pd.DataFrame()
            
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean the scraped DataFrame."""
        if df.empty:
            return df
            
        # Extract ticker from company info
        if 'Company' in df.columns:
            df['ticker'] = df['Company'].apply(self._extract_ticker)
            
        # Parse dates
        date_columns = ['Filing Date', 'Trade Date']
        for col in date_columns:
            if col in df.columns:
                df[col.lower().replace(' ', '_')] = pd.to_datetime(
                    df[col], errors='coerce'
                )
                
        # Parse numeric values
        numeric_columns = ['Shares', 'Price', 'Value', 'Shares Owned']
        for col in numeric_columns:
            if col in df.columns:
                df[col.lower().replace(' ', '_')] = df[col].apply(
                    self._parse_numeric
                )
                
        # Parse transaction type
        if 'Type' in df.columns:
            df['transaction_type'] = df['Type'].map(self.TRANSACTION_TYPES)
            
        # Calculate metrics
        if 'shares' in df.columns and 'shares_owned' in df.columns:
            df['ownership_change_pct'] = (
                df['shares'] / df['shares_owned'] * 100
            )
            
        # Clean column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        return df
        
    def _extract_ticker(self, company_text: str) -> str:
        """Extract ticker symbol from company text."""
        if isinstance(company_text, dict):
            company_text = company_text.get('text', '')
            
        # Look for pattern like "AAPL Apple Inc"
        match = re.search(r'^([A-Z]+)\s', company_text)
        if match:
            return match.group(1)
        return ''
        
    def _parse_numeric(self, value: str) -> float:
        """Parse numeric value from string."""
        if pd.isna(value) or value == '':
            return 0.0
            
        # Remove $, commas, and other characters
        value = str(value).replace('$', '').replace(',', '').replace('+', '')
        
        # Handle K, M, B suffixes
        multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9}
        for suffix, multiplier in multipliers.items():
            if value.endswith(suffix):
                value = value[:-1]
                try:
                    return float(value) * multiplier
                except:
                    return 0.0
                    
        try:
            return float(value)
        except:
            return 0.0
            
    def analyze_sentiment(self, data: pd.DataFrame) -> Dict:
        """
        Analyze insider trading sentiment.
        
        Args:
            data: DataFrame with insider trading data
            
        Returns:
            Dictionary with sentiment analysis
        """
        if data.empty:
            return {
                'total_trades': 0,
                'buy_sell_ratio': 0,
                'bullish_score': 0
            }
            
        # Count buys vs sells
        buys = data[data['transaction_type'].isin(['Purchase', 'Option Exercise'])]
        sells = data[data['transaction_type'].isin(['Sale', 'Automatic Sale'])]
        
        buy_count = len(buys)
        sell_count = len(sells)
        
        # Calculate values
        buy_value = buys['value'].sum() if 'value' in buys.columns else 0
        sell_value = sells['value'].sum() if 'value' in sells.columns else 0
        
        # Buy/sell ratio
        buy_sell_ratio = buy_count / max(sell_count, 1)
        
        # Value-weighted ratio
        value_ratio = buy_value / max(sell_value, 1)
        
        # Bullish score (0-100)
        count_score = min(buy_sell_ratio / 2, 1) * 50
        value_score = min(value_ratio / 2, 1) * 50
        bullish_score = count_score + value_score
        
        return {
            'total_trades': len(data),
            'buy_count': buy_count,
            'sell_count': sell_count,
            'buy_value': buy_value,
            'sell_value': sell_value,
            'buy_sell_ratio': buy_sell_ratio,
            'value_ratio': value_ratio,
            'bullish_score': bullish_score,
            'top_buyers': self._get_top_traders(buys, 'value', 5),
            'top_sellers': self._get_top_traders(sells, 'value', 5)
        }
        
    def _get_top_traders(
        self,
        data: pd.DataFrame,
        sort_column: str,
        limit: int
    ) -> List[Dict]:
        """Get top traders by value or shares."""
        if data.empty or sort_column not in data.columns:
            return []
            
        top = data.nlargest(limit, sort_column)
        
        traders = []
        for _, row in top.iterrows():
            traders.append({
                'ticker': row.get('ticker', ''),
                'insider': row.get('insider', ''),
                'title': row.get('title', ''),
                'value': row.get('value', 0),
                'shares': row.get('shares', 0),
                'date': row.get('filing_date', '')
            })
            
        return traders
        
    def get_signals(
        self,
        ticker: str,
        data: pd.DataFrame,
        lookback_days: int = 90
    ) -> pd.DataFrame:
        """
        Generate trading signals based on insider activity.
        
        Args:
            ticker: Stock ticker
            data: Insider trading data
            lookback_days: Days to look back for patterns
            
        Returns:
            DataFrame with signals
        """
        # Filter by ticker and date
        ticker_data = data[data['ticker'] == ticker.upper()]
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        ticker_data = ticker_data[ticker_data['filing_date'] >= cutoff_date]
        
        if ticker_data.empty:
            return pd.DataFrame()
            
        # Create daily summary
        daily = ticker_data.groupby(ticker_data['filing_date'].dt.date).agg({
            'shares': 'sum',
            'value': 'sum',
            'transaction_type': 'count'
        }).rename(columns={'transaction_type': 'trade_count'})
        
        # Create signals DataFrame
        signals = pd.DataFrame(index=pd.date_range(
            start=cutoff_date,
            end=datetime.now(),
            freq='D'
        ))
        
        # Merge with daily data
        signals = signals.join(daily, how='left')
        signals = signals.fillna(0)
        
        # Calculate signals
        signals['cluster_buy'] = (
            (signals['trade_count'] >= 3) & 
            (signals['shares'] > 0)
        )
        
        signals['large_buy'] = signals['value'] > 1000000
        
        signals['insider_accumulation'] = (
            signals['shares'].rolling(window=20).sum() > 
            signals['shares'].rolling(window=60).sum() * 0.5
        )
        
        return signals