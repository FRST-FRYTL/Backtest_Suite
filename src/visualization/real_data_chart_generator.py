"""
Generate trading charts using real market data.
Creates interactive visualizations with actual price action and indicators.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pickle
from pathlib import Path

class RealDataChartGenerator:
    """Generate chart data from real market data."""
    
    def __init__(self, data_path: str = "data/processed/complete_market_data.pkl"):
        """Initialize with path to processed data."""
        self.data_path = Path(data_path)
        self.data = self._load_data()
        
    def _load_data(self) -> Dict:
        """Load processed market data."""
        if self.data_path.exists():
            with open(self.data_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
    def prepare_chart_data(
        self,
        symbol: str,
        timeframe: str = '1D',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Prepare data for chart visualization.
        
        Args:
            symbol: Asset symbol
            timeframe: Timeframe to display
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dict with chart data in JSON-serializable format
        """
        if symbol not in self.data:
            raise ValueError(f"Symbol {symbol} not found in data")
            
        if timeframe not in self.data[symbol]:
            raise ValueError(f"Timeframe {timeframe} not available for {symbol}")
            
        df = self.data[symbol][timeframe].copy()
        
        # Filter by date range if specified
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        # Prepare candlestick data
        candlestick_data = []
        for idx, row in df.iterrows():
            candlestick_data.append({
                'time': int(idx.timestamp()),
                'open': round(row['Open'], 2),
                'high': round(row['High'], 2),
                'low': round(row['Low'], 2),
                'close': round(row['Close'], 2)
            })
            
        # Prepare volume data
        volume_data = []
        for idx, row in df.iterrows():
            volume_data.append({
                'time': int(idx.timestamp()),
                'value': int(row['Volume']),
                'color': 'rgba(38, 166, 154, 0.5)' if row['Close'] > row['Open'] else 'rgba(239, 83, 80, 0.5)'
            })
            
        # Prepare indicator data
        indicators = {}
        
        # Bollinger Bands
        if 'BB_Upper_2.2' in df.columns:
            indicators['bb_upper'] = self._prepare_line_data(df, 'BB_Upper_2.2')
            indicators['bb_middle'] = self._prepare_line_data(df, 'BB_Middle')
            indicators['bb_lower'] = self._prepare_line_data(df, 'BB_Lower_2.2')
            
        # SMAs
        for period in [20, 50, 200]:
            col_name = f'SMA_{period}'
            if col_name in df.columns:
                indicators[f'sma_{period}'] = self._prepare_line_data(df, col_name)
                
        # VWAP
        if 'VWAP_daily' in df.columns:
            indicators['vwap'] = self._prepare_line_data(df, 'VWAP_daily')
            
        # Generate trading signals (example based on strategy rules)
        signals = self._generate_trading_signals(df)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'candlestick_data': candlestick_data,
            'volume_data': volume_data,
            'indicators': indicators,
            'buy_signals': signals['buy'],
            'sell_signals': signals['sell'],
            'price_range': {
                'min': float(df['Low'].min()),
                'max': float(df['High'].max())
            },
            'date_range': {
                'start': df.index[0].strftime('%Y-%m-%d'),
                'end': df.index[-1].strftime('%Y-%m-%d')
            }
        }
        
    def _prepare_line_data(self, df: pd.DataFrame, column: str) -> List[Dict]:
        """Prepare line chart data."""
        data = []
        for idx, value in df[column].dropna().items():
            data.append({
                'time': int(idx.timestamp()),
                'value': round(float(value), 2)
            })
        return data
        
    def _generate_trading_signals(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Generate buy/sell signals based on strategy rules."""
        buy_signals = []
        sell_signals = []
        
        # Simple example: Buy when RSI < 30 and price < lower BB
        if 'RSI' in df.columns and 'BB_Lower_2.2' in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row['RSI']) and pd.notna(row['BB_Lower_2.2']):
                    # Buy signal
                    if row['RSI'] < 30 and row['Close'] < row['BB_Lower_2.2']:
                        buy_signals.append({
                            'time': int(idx.timestamp()),
                            'position': 'belowBar',
                            'color': '#26a69a',
                            'shape': 'arrowUp',
                            'text': f'Buy (RSI: {row["RSI"]:.0f})'
                        })
                    
                    # Sell signal (example: RSI > 70)
                    elif row['RSI'] > 70 and row['Close'] > row.get('BB_Upper_2.2', row['Close']):
                        sell_signals.append({
                            'time': int(idx.timestamp()),
                            'position': 'aboveBar',
                            'color': '#ef5350',
                            'shape': 'arrowDown',
                            'text': f'Sell (RSI: {row["RSI"]:.0f})'
                        })
                        
        return {'buy': buy_signals, 'sell': sell_signals}
        
    def generate_multi_timeframe_view(
        self,
        symbol: str,
        timeframes: List[str] = ['1H', '4H', '1D']
    ) -> Dict:
        """Generate data for multi-timeframe analysis."""
        mtf_data = {}
        
        for tf in timeframes:
            if tf in self.data.get(symbol, {}):
                mtf_data[tf] = self.prepare_chart_data(symbol, tf)
                
        return mtf_data
        
    def get_market_overview(self) -> Dict:
        """Get overview of all available market data."""
        overview = {}
        
        for symbol, timeframes in self.data.items():
            overview[symbol] = {}
            for tf, df in timeframes.items():
                if not df.empty:
                    overview[symbol][tf] = {
                        'bars': len(df),
                        'start': df.index[0].strftime('%Y-%m-%d'),
                        'end': df.index[-1].strftime('%Y-%m-%d'),
                        'last_close': float(df['Close'].iloc[-1])
                    }
                    
        return overview


def generate_javascript_data(chart_data: Dict) -> str:
    """Convert chart data to JavaScript variable declaration."""
    js_template = """
// Real market data for {symbol} ({timeframe})
const realMarketData = {{
    candlestick: {candlestick_json},
    volume: {volume_json},
    indicators: {indicators_json},
    buySignals: {buy_signals_json},
    sellSignals: {sell_signals_json},
    priceRange: {price_range_json},
    dateRange: {date_range_json}
}};
"""
    
    return js_template.format(
        symbol=chart_data['symbol'],
        timeframe=chart_data['timeframe'],
        candlestick_json=json.dumps(chart_data['candlestick_data']),
        volume_json=json.dumps(chart_data['volume_data']),
        indicators_json=json.dumps(chart_data['indicators']),
        buy_signals_json=json.dumps(chart_data['buy_signals']),
        sell_signals_json=json.dumps(chart_data['sell_signals']),
        price_range_json=json.dumps(chart_data['price_range']),
        date_range_json=json.dumps(chart_data['date_range'])
    )


if __name__ == "__main__":
    # Test the generator
    try:
        generator = RealDataChartGenerator()
        
        # Get SPY daily data
        spy_data = generator.prepare_chart_data('SPY', '1D', start_date='2023-01-01')
        
        # Generate JavaScript
        js_code = generate_javascript_data(spy_data)
        
        # Save to file
        with open('data/spy_chart_data.js', 'w') as f:
            f.write(js_code)
            
        print(f"Generated chart data for SPY with {len(spy_data['candlestick_data'])} bars")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run download_historical_data.py first!")