"""Volume Weighted Average Price (VWAP) indicator implementations."""

from datetime import datetime, time
from typing import Optional, Union, List

import pandas as pd
import numpy as np

from .base import Indicator


class VWAP(Indicator):
    """
    Rolling VWAP (Volume Weighted Average Price) indicator.
    
    VWAP is the average price weighted by volume, commonly used as a trading benchmark.
    """
    
    def __init__(
        self,
        window: Optional[int] = None,
        price_type: str = "typical",
        std_dev_bands: List[float] = [1.0, 2.0, 3.0]
    ):
        """
        Initialize rolling VWAP indicator.
        
        Args:
            window: Rolling window period (None for session-based)
            price_type: Price to use ('typical', 'close', 'hl2', 'hlc3', 'ohlc4')
            std_dev_bands: List of standard deviation multipliers for bands
        """
        super().__init__(name="VWAP")
        self.window = window
        self.price_type = price_type.lower()
        self.std_dev_bands = std_dev_bands
        
    def calculate(
        self,
        data: pd.DataFrame,
        volume_column: str = "volume",
        reset_time: Optional[time] = time(9, 30)
    ) -> pd.DataFrame:
        """
        Calculate rolling VWAP and standard deviation bands.
        
        Args:
            data: DataFrame with OHLCV data
            volume_column: Column name for volume
            reset_time: Time to reset VWAP for intraday (None to disable)
            
        Returns:
            DataFrame with VWAP and bands
        """
        required_cols = self._get_required_columns()
        required_cols.append(volume_column)
        self.validate_data(data, required_cols)
        
        # Calculate price based on type
        price = self._calculate_price(data)
        volume = data[volume_column]
        
        # Calculate VWAP
        if self.window is None and reset_time is not None:
            # Session-based VWAP with daily reset
            vwap_data = self._calculate_session_vwap(price, volume, data.index, reset_time)
        else:
            # Rolling window VWAP
            vwap_data = self._calculate_rolling_vwap(price, volume)
            
        # Add standard deviation bands
        result = self._add_std_bands(vwap_data, price, volume)
        
        return result
        
    def _calculate_price(self, data: pd.DataFrame) -> pd.Series:
        """Calculate price based on price_type."""
        if self.price_type == "typical" or self.price_type == "hlc3":
            return (data['high'] + data['low'] + data['close']) / 3
        elif self.price_type == "close":
            return data['close']
        elif self.price_type == "hl2":
            return (data['high'] + data['low']) / 2
        elif self.price_type == "ohlc4":
            return (data['open'] + data['high'] + data['low'] + data['close']) / 4
        else:
            raise ValueError(f"Invalid price_type: {self.price_type}")
            
    def _get_required_columns(self) -> List[str]:
        """Get required columns based on price_type."""
        if self.price_type in ["typical", "hlc3"]:
            return ['high', 'low', 'close']
        elif self.price_type == "close":
            return ['close']
        elif self.price_type == "hl2":
            return ['high', 'low']
        elif self.price_type == "ohlc4":
            return ['open', 'high', 'low', 'close']
        else:
            return ['close']
            
    def _calculate_rolling_vwap(
        self,
        price: pd.Series,
        volume: pd.Series
    ) -> pd.DataFrame:
        """Calculate rolling window VWAP."""
        window = self.window if self.window is not None else len(price)
        
        # Calculate cumulative price-volume and volume
        pv = price * volume
        
        if window < len(price):
            # Rolling sums
            pv_sum = pv.rolling(window=window, min_periods=1).sum()
            vol_sum = volume.rolling(window=window, min_periods=1).sum()
        else:
            # Cumulative sums
            pv_sum = pv.cumsum()
            vol_sum = volume.cumsum()
            
        # Calculate VWAP
        vwap = pv_sum / vol_sum
        vwap = vwap.fillna(price)  # Handle division by zero
        
        result = pd.DataFrame(index=price.index)
        result['vwap'] = vwap
        result['vwap_volume'] = vol_sum
        
        return result
        
    def _calculate_session_vwap(
        self,
        price: pd.Series,
        volume: pd.Series,
        index: pd.DatetimeIndex,
        reset_time: time
    ) -> pd.DataFrame:
        """Calculate session-based VWAP with daily reset."""
        # Create session groups
        dates = index.date
        times = index.time
        
        # Identify session starts
        session_starts = (dates != pd.Series(dates).shift(1)) | (times == reset_time)
        session_id = session_starts.cumsum()
        
        # Calculate VWAP by session
        pv = price * volume
        
        result = pd.DataFrame(index=index)
        result['session'] = session_id
        result['pv'] = pv
        result['volume'] = volume
        
        # Group by session and calculate cumulative sums
        result['pv_cumsum'] = result.groupby('session')['pv'].cumsum()
        result['vol_cumsum'] = result.groupby('session')['volume'].cumsum()
        
        # Calculate VWAP
        result['vwap'] = result['pv_cumsum'] / result['vol_cumsum']
        result['vwap'] = result['vwap'].fillna(price)
        result['vwap_volume'] = result['vol_cumsum']
        
        return result[['vwap', 'vwap_volume']]
        
    def _add_std_bands(
        self,
        vwap_data: pd.DataFrame,
        price: pd.Series,
        volume: pd.Series
    ) -> pd.DataFrame:
        """Add standard deviation bands to VWAP."""
        result = vwap_data.copy()
        
        # Calculate volume-weighted standard deviation
        vwap = result['vwap']
        
        if self.window is not None:
            # Rolling calculation
            squared_diff = (price - vwap) ** 2
            weighted_squared_diff = squared_diff * volume
            
            wsd_sum = weighted_squared_diff.rolling(
                window=self.window, min_periods=1
            ).sum()
            vol_sum = volume.rolling(window=self.window, min_periods=1).sum()
            
            variance = wsd_sum / vol_sum
            std_dev = np.sqrt(variance)
        else:
            # Cumulative calculation
            squared_diff = (price - vwap) ** 2
            weighted_squared_diff = squared_diff * volume
            
            wsd_cumsum = weighted_squared_diff.cumsum()
            vol_cumsum = volume.cumsum()
            
            variance = wsd_cumsum / vol_cumsum
            std_dev = np.sqrt(variance)
            
        # Add standard deviation bands
        for multiplier in self.std_dev_bands:
            result[f'vwap_upper_{multiplier}'] = vwap + (std_dev * multiplier)
            result[f'vwap_lower_{multiplier}'] = vwap - (std_dev * multiplier)
            
        result['vwap_std'] = std_dev
        
        return result


class AnchoredVWAP(Indicator):
    """
    Anchored VWAP indicator starting from specific dates or events.
    """
    
    def __init__(
        self,
        anchor_date: Union[str, datetime],
        price_type: str = "typical",
        std_dev_bands: List[float] = [1.0, 2.0]
    ):
        """
        Initialize Anchored VWAP indicator.
        
        Args:
            anchor_date: Date to anchor VWAP calculation
            price_type: Price to use ('typical', 'close', 'hl2', 'hlc3', 'ohlc4')
            std_dev_bands: List of standard deviation multipliers for bands
        """
        super().__init__(name="Anchored_VWAP")
        
        if isinstance(anchor_date, str):
            self.anchor_date = pd.to_datetime(anchor_date)
        else:
            self.anchor_date = anchor_date
            
        self.price_type = price_type.lower()
        self.std_dev_bands = std_dev_bands
        
    def calculate(
        self,
        data: pd.DataFrame,
        volume_column: str = "volume"
    ) -> pd.DataFrame:
        """
        Calculate Anchored VWAP from anchor date.
        
        Args:
            data: DataFrame with OHLCV data
            volume_column: Column name for volume
            
        Returns:
            DataFrame with Anchored VWAP and bands
        """
        # Use VWAP base functionality
        vwap = VWAP(window=None, price_type=self.price_type, std_dev_bands=self.std_dev_bands)
        
        # Filter data from anchor date
        if self.anchor_date not in data.index:
            # Find nearest date after anchor
            mask = data.index >= self.anchor_date
            if not mask.any():
                raise ValueError(f"No data available after anchor date {self.anchor_date}")
            anchor_idx = data.index[mask][0]
        else:
            anchor_idx = self.anchor_date
            
        # Calculate VWAP from anchor point
        anchored_data = data.loc[anchor_idx:]
        result = vwap.calculate(anchored_data, volume_column, reset_time=None)
        
        # Rename columns to indicate anchored
        result = result.rename(columns=lambda x: x.replace('vwap', 'avwap'))
        
        # Extend with NaN for dates before anchor
        full_result = pd.DataFrame(index=data.index, columns=result.columns)
        full_result.loc[anchor_idx:] = result
        
        return full_result
        
    @staticmethod
    def create_multiple_anchors(
        data: pd.DataFrame,
        anchor_dates: List[Union[str, datetime]],
        price_type: str = "typical",
        volume_column: str = "volume"
    ) -> pd.DataFrame:
        """
        Create multiple anchored VWAPs from different dates.
        
        Args:
            data: DataFrame with OHLCV data
            anchor_dates: List of dates to anchor VWAP
            price_type: Price to use for calculation
            volume_column: Column name for volume
            
        Returns:
            DataFrame with multiple anchored VWAPs
        """
        result = pd.DataFrame(index=data.index)
        
        for i, anchor_date in enumerate(anchor_dates):
            avwap = AnchoredVWAP(anchor_date, price_type)
            avwap_data = avwap.calculate(data, volume_column)
            
            # Add with unique suffix
            if isinstance(anchor_date, str):
                date_str = anchor_date
            else:
                date_str = anchor_date.strftime('%Y%m%d')
                
            for col in avwap_data.columns:
                new_col = f"{col}_{date_str}"
                result[new_col] = avwap_data[col]
                
        return result
        
    @staticmethod
    def anchor_from_events(
        data: pd.DataFrame,
        events: pd.Series,
        price_type: str = "typical",
        volume_column: str = "volume",
        lookback_days: int = 252
    ) -> pd.DataFrame:
        """
        Create anchored VWAPs from event dates (e.g., earnings, highs/lows).
        
        Args:
            data: DataFrame with OHLCV data
            events: Series with True values on event dates
            price_type: Price to use for calculation
            volume_column: Column name for volume
            lookback_days: Maximum days to look back for VWAP
            
        Returns:
            DataFrame with event-anchored VWAPs
        """
        event_dates = events[events].index.tolist()
        
        if not event_dates:
            return pd.DataFrame(index=data.index)
            
        # Create VWAPs anchored at each event
        result = pd.DataFrame(index=data.index)
        
        for event_date in event_dates:
            # Calculate end date for this VWAP
            end_date = event_date + pd.Timedelta(days=lookback_days)
            if end_date > data.index[-1]:
                end_date = data.index[-1]
                
            # Create anchored VWAP
            avwap = AnchoredVWAP(event_date, price_type)
            event_data = data.loc[event_date:end_date]
            
            if len(event_data) > 0:
                avwap_calc = avwap.calculate(event_data, volume_column)
                
                # Store only the main VWAP line
                col_name = f"avwap_event_{event_date.strftime('%Y%m%d')}"
                result.loc[event_date:end_date, col_name] = avwap_calc['avwap']
                
        return result