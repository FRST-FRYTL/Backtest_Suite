"""
Utility functions for feature engineering
Provides helper functions for feature creation, validation, and processing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
warnings.filterwarnings('ignore')


class FeatureUtils:
    """Utility class for feature engineering operations"""
    
    @staticmethod
    def create_rolling_features(data: pd.Series, windows: List[int], 
                              functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """
        Create rolling window features for a series
        
        Args:
            data: Input series
            windows: List of window sizes
            functions: List of functions to apply
            
        Returns:
            DataFrame with rolling features
        """
        features = pd.DataFrame(index=data.index)
        
        for window in windows:
            for func in functions:
                if func == 'mean':
                    features[f'{data.name}_roll_{func}_{window}'] = data.rolling(window).mean()
                elif func == 'std':
                    features[f'{data.name}_roll_{func}_{window}'] = data.rolling(window).std()
                elif func == 'min':
                    features[f'{data.name}_roll_{func}_{window}'] = data.rolling(window).min()
                elif func == 'max':
                    features[f'{data.name}_roll_{func}_{window}'] = data.rolling(window).max()
                elif func == 'sum':
                    features[f'{data.name}_roll_{func}_{window}'] = data.rolling(window).sum()
                elif func == 'skew':
                    features[f'{data.name}_roll_{func}_{window}'] = data.rolling(window).skew()
                elif func == 'kurt':
                    features[f'{data.name}_roll_{func}_{window}'] = data.rolling(window).kurt()
                elif func == 'quantile_25':
                    features[f'{data.name}_roll_{func}_{window}'] = data.rolling(window).quantile(0.25)
                elif func == 'quantile_75':
                    features[f'{data.name}_roll_{func}_{window}'] = data.rolling(window).quantile(0.75)
                    
        return features
    
    @staticmethod
    def create_expanding_features(data: pd.Series, min_periods: int = 20) -> pd.DataFrame:
        """
        Create expanding window features
        
        Args:
            data: Input series
            min_periods: Minimum periods for expanding window
            
        Returns:
            DataFrame with expanding features
        """
        features = pd.DataFrame(index=data.index)
        
        features[f'{data.name}_exp_mean'] = data.expanding(min_periods=min_periods).mean()
        features[f'{data.name}_exp_std'] = data.expanding(min_periods=min_periods).std()
        features[f'{data.name}_exp_min'] = data.expanding(min_periods=min_periods).min()
        features[f'{data.name}_exp_max'] = data.expanding(min_periods=min_periods).max()
        features[f'{data.name}_exp_median'] = data.expanding(min_periods=min_periods).median()
        
        return features
    
    @staticmethod
    def create_ewm_features(data: pd.Series, spans: List[int]) -> pd.DataFrame:
        """
        Create exponentially weighted moving features
        
        Args:
            data: Input series
            spans: List of span values for EWM
            
        Returns:
            DataFrame with EWM features
        """
        features = pd.DataFrame(index=data.index)
        
        for span in spans:
            features[f'{data.name}_ewm_mean_{span}'] = data.ewm(span=span, adjust=False).mean()
            features[f'{data.name}_ewm_std_{span}'] = data.ewm(span=span, adjust=False).std()
            
        return features
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame, feature_pairs: List[Tuple[str, str]], 
                                  operations: List[str] = ['multiply', 'divide', 'add', 'subtract']) -> pd.DataFrame:
        """
        Create interaction features between pairs of features
        
        Args:
            df: DataFrame with features
            feature_pairs: List of feature pairs to interact
            operations: List of operations to perform
            
        Returns:
            DataFrame with interaction features
        """
        features = pd.DataFrame(index=df.index)
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                if 'multiply' in operations:
                    features[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                if 'divide' in operations and not (df[feat2] == 0).any():
                    features[f'{feat1}_div_{feat2}'] = df[feat1] / df[feat2].replace(0, np.nan)
                if 'add' in operations:
                    features[f'{feat1}_plus_{feat2}'] = df[feat1] + df[feat2]
                if 'subtract' in operations:
                    features[f'{feat1}_minus_{feat2}'] = df[feat1] - df[feat2]
                    
        return features
    
    @staticmethod
    def create_polynomial_features(df: pd.DataFrame, features: List[str], degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features
        
        Args:
            df: DataFrame with features
            features: List of features to create polynomials for
            degree: Polynomial degree
            
        Returns:
            DataFrame with polynomial features
        """
        poly_features = pd.DataFrame(index=df.index)
        
        for feat in features:
            if feat in df.columns:
                for d in range(2, degree + 1):
                    poly_features[f'{feat}_pow_{d}'] = df[feat] ** d
                    
        return poly_features
    
    @staticmethod
    def create_ratio_features(df: pd.DataFrame, numerators: List[str], 
                            denominators: List[str]) -> pd.DataFrame:
        """
        Create ratio features
        
        Args:
            df: DataFrame with features
            numerators: List of numerator features
            denominators: List of denominator features
            
        Returns:
            DataFrame with ratio features
        """
        features = pd.DataFrame(index=df.index)
        
        for num in numerators:
            for den in denominators:
                if num in df.columns and den in df.columns and num != den:
                    # Avoid division by zero
                    features[f'{num}_to_{den}_ratio'] = df[num] / df[den].replace(0, np.nan)
                    
        return features
    
    @staticmethod
    def create_difference_features(df: pd.DataFrame, features: List[str], 
                                 periods: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        Create difference features
        
        Args:
            df: DataFrame with features
            features: List of features to difference
            periods: List of periods for differencing
            
        Returns:
            DataFrame with difference features
        """
        diff_features = pd.DataFrame(index=df.index)
        
        for feat in features:
            if feat in df.columns:
                for period in periods:
                    diff_features[f'{feat}_diff_{period}'] = df[feat].diff(period)
                    diff_features[f'{feat}_pct_change_{period}'] = df[feat].pct_change(period)
                    
        return diff_features
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, features: List[str], 
                          lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features
        
        Args:
            df: DataFrame with features
            features: List of features to lag
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        lag_features = pd.DataFrame(index=df.index)
        
        for feat in features:
            if feat in df.columns:
                for lag in lags:
                    lag_features[f'{feat}_lag_{lag}'] = df[feat].shift(lag)
                    
        return lag_features
    
    @staticmethod
    def create_lead_features(df: pd.DataFrame, features: List[str], 
                           leads: List[int]) -> pd.DataFrame:
        """
        Create lead features (future values)
        
        Args:
            df: DataFrame with features
            features: List of features to lead
            leads: List of lead periods
            
        Returns:
            DataFrame with lead features
        """
        lead_features = pd.DataFrame(index=df.index)
        
        for feat in features:
            if feat in df.columns:
                for lead in leads:
                    lead_features[f'{feat}_lead_{lead}'] = df[feat].shift(-lead)
                    
        return lead_features
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, method: str = 'forward_fill',
                            max_missing_pct: float = 0.2) -> pd.DataFrame:
        """
        Handle missing values in features
        
        Args:
            df: DataFrame with features
            method: Method to handle missing values
            max_missing_pct: Maximum percentage of missing values allowed
            
        Returns:
            DataFrame with handled missing values
        """
        # Remove features with too many missing values
        missing_pct = df.isnull().sum() / len(df)
        cols_to_keep = missing_pct[missing_pct <= max_missing_pct].index
        df = df[cols_to_keep]
        
        # Handle remaining missing values
        if method == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'interpolate':
            df = df.interpolate(method='linear').fillna(method='bfill')
        elif method == 'mean':
            df = df.fillna(df.mean())
        elif method == 'median':
            df = df.fillna(df.median())
        elif method == 'zero':
            df = df.fillna(0)
            
        return df
    
    @staticmethod
    def remove_low_variance_features(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """
        Remove features with low variance
        
        Args:
            df: DataFrame with features
            threshold: Variance threshold
            
        Returns:
            DataFrame without low variance features
        """
        variances = df.var()
        high_variance_features = variances[variances > threshold].index
        
        return df[high_variance_features]
    
    @staticmethod
    def remove_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove highly correlated features
        
        Args:
            df: DataFrame with features
            threshold: Correlation threshold
            
        Returns:
            DataFrame without highly correlated features
        """
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        return df.drop(columns=to_drop)
    
    @staticmethod
    def scale_features(df: pd.DataFrame, method: str = 'standard',
                      feature_range: Tuple[float, float] = (0, 1)) -> Tuple[pd.DataFrame, object]:
        """
        Scale features using specified method
        
        Args:
            df: DataFrame with features
            method: Scaling method ('standard', 'robust', 'minmax')
            feature_range: Range for MinMax scaling
            
        Returns:
            Tuple of (scaled DataFrame, scaler object)
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        else:
            raise ValueError(f"Unknown scaling method: {method}")
            
        scaled_data = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
        
        return scaled_df, scaler
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, method: str = 'zscore',
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in features
        
        Args:
            df: DataFrame with features
            method: Outlier detection method ('zscore', 'iqr')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with boolean values indicating outliers
        """
        outliers = pd.DataFrame(False, index=df.index, columns=df.columns)
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(df, nan_policy='omit'))
            outliers = pd.DataFrame(z_scores > threshold, index=df.index, columns=df.columns)
            
        elif method == 'iqr':
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            outliers = (df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))
            
        return outliers
    
    @staticmethod
    def winsorize_features(df: pd.DataFrame, limits: Tuple[float, float] = (0.01, 0.01)) -> pd.DataFrame:
        """
        Winsorize features to handle outliers
        
        Args:
            df: DataFrame with features
            limits: Lower and upper percentile limits
            
        Returns:
            Winsorized DataFrame
        """
        winsorized_df = df.copy()
        
        for col in df.columns:
            winsorized_df[col] = stats.mstats.winsorize(df[col].dropna(), limits=limits)
            
        return winsorized_df
    
    @staticmethod
    def create_time_series_features(index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Create time-based features from datetime index
        
        Args:
            index: DatetimeIndex
            
        Returns:
            DataFrame with time features
        """
        features = pd.DataFrame(index=index)
        
        # Basic time features
        features['hour'] = index.hour
        features['day'] = index.day
        features['dayofweek'] = index.dayofweek
        features['month'] = index.month
        features['quarter'] = index.quarter
        features['year'] = index.year
        features['dayofyear'] = index.dayofyear
        features['weekofyear'] = index.isocalendar().week
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day'] / 31)
        features['day_cos'] = np.cos(2 * np.pi * features['day'] / 31)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['dayofweek_sin'] = np.sin(2 * np.pi * features['dayofweek'] / 7)
        features['dayofweek_cos'] = np.cos(2 * np.pi * features['dayofweek'] / 7)
        
        # Binary features
        features['is_weekend'] = (features['dayofweek'] >= 5).astype(int)
        features['is_month_start'] = (features['day'] <= 7).astype(int)
        features['is_month_end'] = (features['day'] >= 24).astype(int)
        features['is_quarter_start'] = ((features['month'] - 1) % 3 == 0) & features['is_month_start']
        features['is_quarter_end'] = (features['month'] % 3 == 0) & features['is_month_end']
        
        return features
    
    @staticmethod
    def validate_features(df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate features and return statistics
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'n_features': len(df.columns),
            'n_samples': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'zero_variance_features': df.columns[df.var() == 0].tolist(),
            'duplicate_features': [],
            'constant_features': [],
            'data_types': df.dtypes.to_dict()
        }
        
        # Check for duplicate features
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                if df[col1].equals(df[col2]):
                    validation_results['duplicate_features'].append((col1, col2))
        
        # Check for constant features
        for col in df.columns:
            if df[col].nunique() == 1:
                validation_results['constant_features'].append(col)
                
        return validation_results