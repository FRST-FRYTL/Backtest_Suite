"""
Test Fixture Configuration

Central configuration for all test fixtures to ensure consistency
and avoid pandas compatibility issues.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


class FixtureConfig:
    """Configuration for test fixtures."""
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Date ranges
    DEFAULT_START_DATE = '2023-01-01'
    DEFAULT_END_DATE = '2023-12-31'
    DEFAULT_PERIODS = 252  # One trading year
    
    # Financial parameters
    INITIAL_CAPITAL = 100000
    ANNUAL_RETURN_TARGET = 0.15
    ANNUAL_VOLATILITY_TARGET = 0.20
    MAX_DRAWDOWN_TARGET = -0.15
    SHARPE_RATIO_TARGET = 0.75
    WIN_RATE_TARGET = 0.55
    
    # Trade parameters
    AVG_WIN_PNL = 500
    AVG_LOSS_PNL = -300
    DEFAULT_COMMISSION_RATE = 0.001
    DEFAULT_SLIPPAGE_RATE = 0.0005
    
    # Symbols
    DEFAULT_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    SECTOR_NAMES = ['Technology', 'Finance', 'Healthcare', 'Energy', 'Consumer']
    
    # Visualization parameters
    FIGURE_DPI = 300
    FIGURE_SIZE = (12, 8)
    
    # Window sizes for rolling calculations
    ROLLING_WINDOWS = [20, 60, 126, 252]
    
    # Color scheme
    COLOR_SCHEME = {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e",
        "success": "#2ca02c",
        "warning": "#ff9800",
        "danger": "#d62728",
        "info": "#17a2b8",
        "background": "#ffffff",
        "text": "#333333"
    }
    
    @classmethod
    def get_default_date_range(cls) -> pd.DatetimeIndex:
        """Get default date range for fixtures."""
        return pd.date_range(
            start=cls.DEFAULT_START_DATE,
            end=cls.DEFAULT_END_DATE,
            freq='D'
        )
    
    @classmethod
    def get_intraday_times(cls, date: str, hours: int = 8) -> pd.DatetimeIndex:
        """Get intraday timestamps."""
        start_time = pd.Timestamp(f'{date} 09:30:00')
        return pd.date_range(
            start=start_time,
            periods=hours * 60,
            freq='min'
        )
    
    @classmethod
    def ensure_proper_dtypes(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame has proper dtypes to avoid pandas issues.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with proper dtypes
        """
        # Numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                          'price', 'returns', 'pnl', 'position_size',
                          'entry_price', 'exit_price', 'stop_loss',
                          'take_profit', 'commission', 'slippage']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # String columns
        string_columns = ['symbol', 'side', 'order_type', 'status',
                         'exit_reason', 'strategy', 'trade_type']
        
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Datetime columns
        datetime_columns = ['date', 'timestamp', 'entry_date', 'exit_date',
                           'entry_time', 'exit_time']
        
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Ensure index is datetime if it looks like dates
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                pass
        
        return df
    
    @classmethod
    def validate_ohlcv(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and fix OHLCV data relationships.
        
        Args:
            df: DataFrame with OHLC columns
            
        Returns:
            DataFrame with valid OHLC relationships
        """
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Ensure high >= max(open, close)
            df['high'] = df[['open', 'high', 'close']].max(axis=1)
            
            # Ensure low <= min(open, close)
            df['low'] = df[['open', 'low', 'close']].min(axis=1)
            
            # Ensure volume is positive
            if 'volume' in df.columns:
                df['volume'] = df['volume'].abs()
        
        return df
    
    @classmethod
    def handle_empty_data(cls, df: pd.DataFrame, default_value: Any = 0) -> pd.DataFrame:
        """
        Handle empty DataFrames to avoid pandas issues.
        
        Args:
            df: DataFrame to check
            default_value: Default value for empty cells
            
        Returns:
            DataFrame with at least one row
        """
        if len(df) == 0 and len(df.columns) > 0:
            # Add a single row with default values
            default_row = pd.Series(
                [default_value] * len(df.columns),
                index=df.columns
            )
            df = pd.DataFrame([default_row])
        
        return df


def get_fixture_defaults() -> Dict[str, Any]:
    """Get default parameters for all fixtures."""
    return {
        'seed': FixtureConfig.RANDOM_SEED,
        'periods': FixtureConfig.DEFAULT_PERIODS,
        'start_date': FixtureConfig.DEFAULT_START_DATE,
        'end_date': FixtureConfig.DEFAULT_END_DATE,
        'initial_capital': FixtureConfig.INITIAL_CAPITAL,
        'symbols': FixtureConfig.DEFAULT_SYMBOLS,
        'annual_return': FixtureConfig.ANNUAL_RETURN_TARGET,
        'volatility': FixtureConfig.ANNUAL_VOLATILITY_TARGET,
        'sharpe_ratio': FixtureConfig.SHARPE_RATIO_TARGET,
        'max_drawdown': FixtureConfig.MAX_DRAWDOWN_TARGET,
        'win_rate': FixtureConfig.WIN_RATE_TARGET,
        'commission_rate': FixtureConfig.DEFAULT_COMMISSION_RATE,
        'slippage_rate': FixtureConfig.DEFAULT_SLIPPAGE_RATE
    }


def create_test_dataframe_safe(
    data: Dict[str, List],
    index: pd.Index = None,
    ensure_numeric: List[str] = None
) -> pd.DataFrame:
    """
    Create a DataFrame safely, avoiding pandas compatibility issues.
    
    Args:
        data: Dictionary of column data
        index: Index for the DataFrame
        ensure_numeric: List of columns to ensure are numeric
        
    Returns:
        Safe DataFrame
    """
    # Create DataFrame
    df = pd.DataFrame(data, index=index)
    
    # Ensure proper dtypes
    df = FixtureConfig.ensure_proper_dtypes(df)
    
    # Ensure specific numeric columns if requested
    if ensure_numeric:
        for col in ensure_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Handle empty DataFrames
    df = FixtureConfig.handle_empty_data(df)
    
    return df