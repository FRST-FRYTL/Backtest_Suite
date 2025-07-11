"""Max Pain options calculations for support/resistance levels."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    from ..data.fetcher import StockDataFetcher
    from ..data.cache import DataCache
except ImportError:
    from data.fetcher import StockDataFetcher
    from data.cache import DataCache


class MaxPain:
    """
    Calculate Max Pain price levels from options data.
    
    Max Pain is the strike price where option holders would experience
    maximum financial loss at expiration.
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize Max Pain calculator.
        
        Args:
            cache_dir: Directory for caching calculations
        """
        self.fetcher = StockDataFetcher(cache_dir)
        self.cache = DataCache(cache_dir)
        
    def calculate(
        self,
        ticker: str,
        expiration_date: Optional[str] = None
    ) -> Dict:
        """
        Calculate max pain for a specific expiration date.
        
        Args:
            ticker: Stock ticker symbol
            expiration_date: Options expiration date (None for nearest)
            
        Returns:
            Dictionary with max pain data
        """
        # Check cache
        cache_key = f"max_pain_{ticker}_{expiration_date or 'nearest'}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
            
        try:
            # Fetch options chain
            calls, puts = self.fetcher.get_options_chain(ticker, expiration_date)
            
            if calls.empty or puts.empty:
                return {
                    'max_pain_price': None,
                    'error': 'No options data available'
                }
                
            # Calculate max pain
            result = self._calculate_max_pain(calls, puts)
            
            # Add metadata
            result['ticker'] = ticker
            result['expiration'] = expiration_date or calls.iloc[0].name
            result['calculation_time'] = datetime.now()
            
            # Cache for 1 hour
            self.cache.set(cache_key, result, expire=3600)
            
            return result
            
        except Exception as e:
            return {
                'max_pain_price': None,
                'error': str(e)
            }
            
    def _calculate_max_pain(
        self,
        calls: pd.DataFrame,
        puts: pd.DataFrame
    ) -> Dict:
        """
        Calculate max pain price from options chain.
        
        Args:
            calls: DataFrame with call options
            puts: DataFrame with put options
            
        Returns:
            Dictionary with max pain calculations
        """
        # Get unique strikes
        strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
        
        # Calculate pain at each strike
        pain_data = []
        
        for strike in strikes:
            call_pain = self._calculate_call_pain(calls, strike)
            put_pain = self._calculate_put_pain(puts, strike)
            total_pain = call_pain + put_pain
            
            pain_data.append({
                'strike': strike,
                'call_pain': call_pain,
                'put_pain': put_pain,
                'total_pain': total_pain
            })
            
        # Create DataFrame
        pain_df = pd.DataFrame(pain_data)
        
        # Find max pain strike
        max_pain_idx = pain_df['total_pain'].idxmin()
        max_pain_strike = pain_df.loc[max_pain_idx, 'strike']
        
        # Calculate additional metrics
        current_price = (calls['lastPrice'].mean() + puts['lastPrice'].mean()) / 2
        
        # Support/resistance levels (strikes with high open interest)
        call_oi = calls.groupby('strike')['openInterest'].sum()
        put_oi = puts.groupby('strike')['openInterest'].sum()
        
        resistance_levels = call_oi.nlargest(3).index.tolist()
        support_levels = put_oi.nlargest(3).index.tolist()
        
        return {
            'max_pain_price': max_pain_strike,
            'total_pain': pain_df.loc[max_pain_idx, 'total_pain'],
            'pain_distribution': pain_df.to_dict('records'),
            'current_price': current_price,
            'price_vs_max_pain': ((current_price - max_pain_strike) / max_pain_strike) * 100,
            'resistance_levels': resistance_levels,
            'support_levels': support_levels,
            'total_call_oi': calls['openInterest'].sum(),
            'total_put_oi': puts['openInterest'].sum(),
            'put_call_ratio': puts['openInterest'].sum() / max(calls['openInterest'].sum(), 1),
            'gamma_levels': self._calculate_gamma_levels(calls, puts)
        }
        
    def _calculate_call_pain(self, calls: pd.DataFrame, strike: float) -> float:
        """Calculate pain for call options at a given strike."""
        pain = 0
        
        for _, option in calls.iterrows():
            if strike > option['strike']:
                # Calls are in the money
                loss = (strike - option['strike']) * option['openInterest'] * 100
                pain += loss
                
        return pain
        
    def _calculate_put_pain(self, puts: pd.DataFrame, strike: float) -> float:
        """Calculate pain for put options at a given strike."""
        pain = 0
        
        for _, option in puts.iterrows():
            if strike < option['strike']:
                # Puts are in the money
                loss = (option['strike'] - strike) * option['openInterest'] * 100
                pain += loss
                
        return pain
        
    def _calculate_gamma_levels(
        self,
        calls: pd.DataFrame,
        puts: pd.DataFrame
    ) -> List[Dict]:
        """Calculate significant gamma levels."""
        # Combine gamma from calls and puts
        gamma_data = []
        
        # Process calls
        for _, option in calls.iterrows():
            if 'gamma' in option and pd.notna(option['gamma']):
                gamma_data.append({
                    'strike': option['strike'],
                    'gamma': option['gamma'] * option['openInterest'],
                    'type': 'call'
                })
                
        # Process puts
        for _, option in puts.iterrows():
            if 'gamma' in option and pd.notna(option['gamma']):
                gamma_data.append({
                    'strike': option['strike'],
                    'gamma': option['gamma'] * option['openInterest'],
                    'type': 'put'
                })
                
        if not gamma_data:
            return []
            
        # Aggregate by strike
        gamma_df = pd.DataFrame(gamma_data)
        gamma_by_strike = gamma_df.groupby('strike')['gamma'].sum().sort_values(ascending=False)
        
        # Get top gamma levels
        top_gamma = []
        for strike, gamma in gamma_by_strike.head(5).items():
            top_gamma.append({
                'strike': strike,
                'gamma_exposure': gamma,
                'significance': 'high' if gamma > gamma_by_strike.mean() * 2 else 'medium'
            })
            
        return top_gamma
        
    def calculate_all_expirations(
        self,
        ticker: str,
        max_expirations: int = 4
    ) -> pd.DataFrame:
        """
        Calculate max pain for multiple expiration dates.
        
        Args:
            ticker: Stock ticker symbol
            max_expirations: Maximum number of expirations to analyze
            
        Returns:
            DataFrame with max pain data for each expiration
        """
        try:
            # Get available expirations
            ticker_obj = self.fetcher.get_info(ticker)
            expirations = ticker_obj.get('expirationDates', [])[:max_expirations]
            
            if not expirations:
                return pd.DataFrame()
                
            results = []
            
            for exp_date in expirations:
                max_pain_data = self.calculate(ticker, exp_date)
                
                if max_pain_data.get('max_pain_price'):
                    results.append({
                        'expiration': exp_date,
                        'max_pain': max_pain_data['max_pain_price'],
                        'current_price': max_pain_data['current_price'],
                        'deviation_pct': max_pain_data['price_vs_max_pain'],
                        'put_call_ratio': max_pain_data['put_call_ratio'],
                        'days_to_expiry': (pd.to_datetime(exp_date) - datetime.now()).days
                    })
                    
            return pd.DataFrame(results)
            
        except Exception as e:
            print(f"Error calculating max pain for {ticker}: {e}")
            return pd.DataFrame()
            
    def get_signals(
        self,
        ticker: str,
        current_price: float,
        max_pain_data: Dict
    ) -> Dict:
        """
        Generate trading signals based on max pain analysis.
        
        Args:
            ticker: Stock ticker
            current_price: Current stock price
            max_pain_data: Max pain calculation results
            
        Returns:
            Dictionary with trading signals
        """
        if not max_pain_data.get('max_pain_price'):
            return {}
            
        max_pain = max_pain_data['max_pain_price']
        deviation = ((current_price - max_pain) / max_pain) * 100
        
        signals = {
            'max_pain_magnet': abs(deviation) < 5,  # Price near max pain
            'above_max_pain': current_price > max_pain,
            'below_max_pain': current_price < max_pain,
            'extreme_deviation': abs(deviation) > 10,
            'bullish_deviation': deviation > 10,  # Price significantly above
            'bearish_deviation': deviation < -10,  # Price significantly below
        }
        
        # Add support/resistance signals
        if 'resistance_levels' in max_pain_data:
            nearest_resistance = min(
                [r for r in max_pain_data['resistance_levels'] if r > current_price],
                default=None
            )
            signals['near_resistance'] = (
                nearest_resistance and 
                (nearest_resistance - current_price) / current_price < 0.02
            )
            
        if 'support_levels' in max_pain_data:
            nearest_support = max(
                [s for s in max_pain_data['support_levels'] if s < current_price],
                default=None
            )
            signals['near_support'] = (
                nearest_support and 
                (current_price - nearest_support) / current_price < 0.02
            )
            
        # Gamma squeeze potential
        if 'gamma_levels' in max_pain_data and max_pain_data['gamma_levels']:
            high_gamma_strikes = [
                g['strike'] for g in max_pain_data['gamma_levels'] 
                if g['significance'] == 'high'
            ]
            signals['gamma_squeeze_potential'] = any(
                abs(current_price - strike) / current_price < 0.01
                for strike in high_gamma_strikes
            )
            
        return signals
        
    def plot_pain_chart(
        self,
        max_pain_data: Dict,
        current_price: float
    ) -> Dict:
        """
        Prepare data for max pain visualization.
        
        Args:
            max_pain_data: Max pain calculation results
            current_price: Current stock price
            
        Returns:
            Dictionary with plotting data
        """
        if 'pain_distribution' not in max_pain_data:
            return {}
            
        pain_df = pd.DataFrame(max_pain_data['pain_distribution'])
        
        return {
            'strikes': pain_df['strike'].tolist(),
            'total_pain': pain_df['total_pain'].tolist(),
            'call_pain': pain_df['call_pain'].tolist(),
            'put_pain': pain_df['put_pain'].tolist(),
            'max_pain_strike': max_pain_data['max_pain_price'],
            'current_price': current_price,
            'resistance_levels': max_pain_data.get('resistance_levels', []),
            'support_levels': max_pain_data.get('support_levels', [])
        }