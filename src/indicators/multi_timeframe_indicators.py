"""
Multi-timeframe technical indicators with configurable parameters.
Supports SMA, Bollinger Bands, VWAP with multiple standard deviations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import talib
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class MultiTimeframeIndicators:
    """Calculate technical indicators across multiple timeframes."""
    
    def __init__(self, config: dict):
        """Initialize with configuration."""
        self.config = config
        self.indicator_config = config['indicators']
        
    def calculate_sma(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages for multiple periods.
        
        Args:
            df: DataFrame with OHLCV data
            periods: List of periods to calculate
            
        Returns:
            DataFrame with SMA columns added
        """
        for period in periods:
            df[f'SMA_{period}'] = talib.SMA(df['Close'], timeperiod=period)
            
        return df
        
    def calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_devs: List[float] = [1.25, 2.2, 3.2]
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands with multiple standard deviations.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for moving average
            std_devs: List of standard deviations
            
        Returns:
            DataFrame with BB columns added
        """
        # Calculate base moving average
        ma = talib.SMA(df['Close'], timeperiod=period)
        df['BB_Middle'] = ma
        
        # Calculate standard deviation
        std = df['Close'].rolling(window=period).std()
        
        # Calculate bands for each std dev
        for std_dev in std_devs:
            df[f'BB_Upper_{std_dev}'] = ma + (std * std_dev)
            df[f'BB_Lower_{std_dev}'] = ma - (std * std_dev)
            
        # Calculate position relative to bands
        df['BB_Position'] = (df['Close'] - df['BB_Lower_2.2']) / (
            df['BB_Upper_2.2'] - df['BB_Lower_2.2']
        )
        
        # Detect squeeze (low volatility)
        df['BB_Width'] = (df['BB_Upper_2.2'] - df['BB_Lower_2.2']) / df['BB_Middle']
        df['BB_Squeeze'] = df['BB_Width'] < df['BB_Width'].rolling(window=120).quantile(0.2)
        
        return df
        
    def calculate_vwap(
        self,
        df: pd.DataFrame,
        periods: List[str] = ['daily', 'weekly', 'monthly'],
        std_devs: List[float] = [1, 2, 3]
    ) -> pd.DataFrame:
        """
        Calculate VWAP for different periods with standard deviation bands.
        
        Args:
            df: DataFrame with OHLCV data
            periods: List of periods ('daily', 'weekly', 'monthly', 'yearly', '5Y')
            std_devs: List of standard deviations
            
        Returns:
            DataFrame with VWAP columns added
        """
        # Typical price
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        for period in periods:
            # Define resampling rule
            if period == 'daily':
                grouper = pd.Grouper(freq='D')
            elif period == 'weekly':
                grouper = pd.Grouper(freq='W')
            elif period == 'monthly':
                grouper = pd.Grouper(freq='M')
            elif period == 'yearly':
                grouper = pd.Grouper(freq='Y')
            elif period == '5Y':
                # For 5Y, we'll use a rolling window
                window_size = 252 * 5  # Approximate trading days in 5 years
                df[f'VWAP_{period}'] = (
                    (df['Typical_Price'] * df['Volume']).rolling(window=window_size).sum() /
                    df['Volume'].rolling(window=window_size).sum()
                )
                
                # Calculate rolling standard deviation for 5Y
                vwap_col = f'VWAP_{period}'
                price_diff = df['Typical_Price'] - df[vwap_col]
                weighted_var = (price_diff ** 2 * df['Volume']).rolling(window=window_size).sum() / \
                              df['Volume'].rolling(window=window_size).sum()
                vwap_std = np.sqrt(weighted_var)
                
                for std_dev in std_devs:
                    df[f'{vwap_col}_Upper_{std_dev}'] = df[vwap_col] + (vwap_std * std_dev)
                    df[f'{vwap_col}_Lower_{std_dev}'] = df[vwap_col] - (vwap_std * std_dev)
                continue
                
            # Calculate VWAP for each period
            vwap_col = f'VWAP_{period}'
            
            # Group by period and calculate cumulative values
            df['Volume_x_Price'] = df['Typical_Price'] * df['Volume']
            
            # Calculate VWAP within each period
            grouped = df.groupby(grouper)
            df[vwap_col] = grouped['Volume_x_Price'].cumsum() / grouped['Volume'].cumsum()
            
            # Calculate standard deviation bands
            # For each period, calculate the volume-weighted standard deviation
            for group_name, group_data in grouped:
                mask = df.index.isin(group_data.index)
                
                # Volume-weighted variance
                vwap_values = df.loc[mask, vwap_col]
                price_diff = df.loc[mask, 'Typical_Price'] - vwap_values
                weights = df.loc[mask, 'Volume'] / df.loc[mask, 'Volume'].sum()
                weighted_var = (weights * price_diff ** 2).sum()
                vwap_std = np.sqrt(weighted_var)
                
                # Add bands
                for std_dev in std_devs:
                    df.loc[mask, f'{vwap_col}_Upper_{std_dev}'] = vwap_values + (vwap_std * std_dev)
                    df.loc[mask, f'{vwap_col}_Lower_{std_dev}'] = vwap_values - (vwap_std * std_dev)
                    
        # Clean up temporary column
        if 'Volume_x_Price' in df.columns:
            df.drop('Volume_x_Price', axis=1, inplace=True)
            
        return df
        
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate RSI."""
        df['RSI'] = talib.RSI(df['Close'], timeperiod=period)
        
        # Add divergence detection
        df['RSI_Divergence'] = self._detect_divergence(
            df['Close'], df['RSI'], lookback=20
        )
        
        return df
        
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=period)
        df['ATR_Percent'] = df['ATR'] / df['Close'] * 100
        
        # Calculate ATR percentile (for volatility regime detection)
        df['ATR_Percentile'] = df['ATR'].rolling(window=252).rank(pct=True) * 100
        
        return df
        
    def calculate_rolling_vwap(
        self,
        df: pd.DataFrame,
        periods: List[int] = [5, 10, 20, 50, 100, 200],
        std_devs: List[float] = [1, 2, 3]
    ) -> pd.DataFrame:
        """
        Calculate Rolling VWAP for different periods with standard deviation bands.
        Unlike regular VWAP that resets daily/weekly, this calculates rolling VWAP over N periods.
        
        Args:
            df: DataFrame with OHLCV data
            periods: List of rolling window periods
            std_devs: List of standard deviations for bands
            
        Returns:
            DataFrame with Rolling VWAP columns added
        """
        # Calculate typical price
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        
        for period in periods:
            vwap_col = f'Rolling_VWAP_{period}'
            
            # Calculate rolling VWAP
            tp_volume = typical_price * df['Volume']
            rolling_tp_volume = tp_volume.rolling(window=period).sum()
            rolling_volume = df['Volume'].rolling(window=period).sum()
            df[vwap_col] = rolling_tp_volume / rolling_volume
            
            # Calculate volume-weighted standard deviation
            # For each rolling window, compute volume-weighted variance
            vwap_values = df[vwap_col]
            price_diff_squared = (typical_price - vwap_values) ** 2
            
            # Volume-weighted variance using rolling windows
            weighted_variance = (price_diff_squared * df['Volume']).rolling(window=period).sum() / rolling_volume
            vwap_std = np.sqrt(weighted_variance)
            
            # Add standard deviation bands
            for std_dev in std_devs:
                df[f'{vwap_col}_Upper_{std_dev}'] = df[vwap_col].values + (vwap_std.values * std_dev)
                df[f'{vwap_col}_Lower_{std_dev}'] = df[vwap_col].values - (vwap_std.values * std_dev)
            
            # Add position relative to VWAP (0 to 1 scale)
            df[f'{vwap_col}_Position'] = (df['Close'] - df[f'{vwap_col}_Lower_2']) / (
                df[f'{vwap_col}_Upper_2'] - df[f'{vwap_col}_Lower_2']
            )
            
            # Add distance from VWAP as percentage
            df[f'{vwap_col}_Distance'] = ((df['Close'] - df[vwap_col]) / df[vwap_col]) * 100
            
            # Detect price touches/crosses
            df[f'{vwap_col}_Cross_Above'] = (
                (df['Close'].shift(1) <= df[vwap_col].shift(1)) & 
                (df['Close'] > df[vwap_col])
            )
            df[f'{vwap_col}_Cross_Below'] = (
                (df['Close'].shift(1) >= df[vwap_col].shift(1)) & 
                (df['Close'] < df[vwap_col])
            )
            
        return df
        
    def calculate_volume_profile(self, df: pd.DataFrame, bins: int = 50) -> pd.DataFrame:
        """Calculate volume profile indicators."""
        # Volume moving averages
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # Detect volume spikes
        df['Volume_Spike'] = df['Volume_Ratio'] > 2.0
        
        # On-Balance Volume
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        
        return df
        
    def _detect_divergence(
        self,
        price: pd.Series,
        indicator: pd.Series,
        lookback: int = 20
    ) -> pd.Series:
        """
        Detect bullish and bearish divergences.
        
        Returns:
            Series with 1 for bullish divergence, -1 for bearish, 0 for none
        """
        divergence = pd.Series(0, index=price.index)
        
        for i in range(lookback, len(price)):
            # Get recent peaks and troughs
            price_window = price.iloc[i-lookback:i]
            ind_window = indicator.iloc[i-lookback:i]
            
            # Find local minima
            price_min_idx = price_window.idxmin()
            ind_min_idx = ind_window.idxmin()
            
            # Find local maxima
            price_max_idx = price_window.idxmax()
            ind_max_idx = ind_window.idxmax()
            
            # Bullish divergence: price makes lower low, indicator makes higher low
            if (price.iloc[i] < price.loc[price_min_idx] and 
                indicator.iloc[i] > indicator.loc[ind_min_idx]):
                divergence.iloc[i] = 1
                
            # Bearish divergence: price makes higher high, indicator makes lower high
            elif (price.iloc[i] > price.loc[price_max_idx] and 
                  indicator.iloc[i] < indicator.loc[ind_max_idx]):
                divergence.iloc[i] = -1
                
        return divergence
        
    def calculate_all_indicators(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Calculate all indicators for a given timeframe.
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe identifier (1H, 4H, 1D, etc.)
            
        Returns:
            DataFrame with all indicators added
        """
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Skip if not enough data
        if len(df) < 365:
            logger.warning(f"Insufficient data for {timeframe}, need at least 365 bars")
            return df
            
        # Calculate all indicators
        logger.info(f"Calculating indicators for {timeframe}")
        
        # SMAs
        df = self.calculate_sma(df, self.indicator_config['sma']['periods'])
        
        # Bollinger Bands
        df = self.calculate_bollinger_bands(
            df,
            period=self.indicator_config['bollinger_bands']['period'],
            std_devs=self.indicator_config['bollinger_bands']['std_devs']
        )
        
        # VWAP (only for intraday timeframes)
        if timeframe in ['1H', '4H']:
            df = self.calculate_vwap(
                df,
                periods=['daily', 'weekly', 'monthly'],
                std_devs=self.indicator_config['vwap']['std_devs']
            )
        
        # Rolling VWAP - works for all timeframes
        if 'rolling_vwap' in self.indicator_config:
            df = self.calculate_rolling_vwap(
                df,
                periods=self.indicator_config['rolling_vwap']['periods'],
                std_devs=self.indicator_config['rolling_vwap']['std_devs']
            )
        
        # RSI
        df = self.calculate_rsi(df, period=self.indicator_config['rsi']['period'])
        
        # ATR
        df = self.calculate_atr(df, period=self.indicator_config['atr']['period'])
        
        # Volume indicators
        df = self.calculate_volume_profile(df)
        
        return df
        
    def align_timeframes(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Align data across different timeframes for multi-timeframe analysis.
        
        Args:
            data: Dict of {timeframe: DataFrame}
            
        Returns:
            Dict with aligned data
        """
        # Use daily as the base timeframe
        if '1D' not in data:
            logger.error("Daily timeframe required for alignment")
            return data
            
        daily_df = data['1D'].copy()
        
        # Add higher timeframe indicators to daily
        for tf in ['1W', '1M']:
            if tf in data:
                tf_df = data[tf]
                
                # Resample indicators to daily
                for col in tf_df.columns:
                    if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        # Forward fill to propagate weekly/monthly values to daily
                        daily_df[f'{col}_{tf}'] = tf_df[col].reindex(
                            daily_df.index, method='ffill'
                        )
                        
        data['1D_MultiTF'] = daily_df
        return data


def test_indicators():
    """Test indicator calculations with sample data."""
    import yfinance as yf
    
    # Download sample data
    spy = yf.download('SPY', start='2023-01-01', end='2024-01-01', interval='1h')
    
    # Load config
    import yaml
    with open('config/strategy_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Calculate indicators
    calc = MultiTimeframeIndicators(config)
    spy_with_indicators = calc.calculate_all_indicators(spy, '1H')
    
    # Print sample
    print("\nSample indicators for SPY:")
    print(spy_with_indicators[['Close', 'SMA_20', 'SMA_50', 'RSI', 'ATR']].tail())
    
    return spy_with_indicators


if __name__ == "__main__":
    test_indicators()