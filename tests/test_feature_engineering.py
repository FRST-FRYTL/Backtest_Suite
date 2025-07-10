"""
Test script for feature engineering pipeline
Verifies that feature extraction and selection work correctly
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ml.features import FeatureEngineer, FeatureSelector
from ml.features.feature_utils import FeatureUtils


class TestFeatureEngineering(unittest.TestCase):
    """Test cases for feature engineering"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample OHLCV data
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='H')
        n_points = len(dates)
        
        np.random.seed(42)
        base_price = 100
        returns = np.random.normal(0.0001, 0.01, n_points)
        close_prices = base_price * np.exp(np.cumsum(returns))
        
        self.data = pd.DataFrame({
            'open': close_prices * (1 + np.random.uniform(-0.002, 0.002, n_points)),
            'high': close_prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
            'low': close_prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
            'close': close_prices,
            'volume': np.random.lognormal(10, 1, n_points)
        }, index=dates)
        
        self.engineer = FeatureEngineer()
        self.selector = FeatureSelector()
        
    def test_feature_extraction(self):
        """Test basic feature extraction"""
        features = self.engineer.engineer_features(
            self.data.head(1000),  # Use subset for speed
            lookback_periods=[5, 10, 20]
        )
        
        # Check that features were created
        self.assertGreater(len(features.columns), 100)
        self.assertEqual(len(features), len(self.data.head(1000)))
        
        # Check for specific feature types
        feature_names = features.columns.tolist()
        
        # Price features
        self.assertTrue(any('return_' in name for name in feature_names))
        self.assertTrue(any('log_return_' in name for name in feature_names))
        
        # Technical indicators
        self.assertTrue(any('rsi_' in name for name in feature_names))
        self.assertTrue(any('macd_' in name for name in feature_names))
        self.assertTrue(any('bb_' in name for name in feature_names))
        
        # Volume features
        self.assertTrue(any('volume_' in name for name in feature_names))
        self.assertTrue(any('vwap_' in name for name in feature_names))
        
    def test_feature_scaling(self):
        """Test feature scaling"""
        features = self.engineer.engineer_features(
            self.data.head(500),
            lookback_periods=[5, 10]
        )
        
        # Test standard scaling
        scaled_features = self.engineer.scale_features(features, method='standard')
        
        # Check scaled features have mean ~0 and std ~1
        means = scaled_features.mean()
        stds = scaled_features.std()
        
        self.assertTrue(np.allclose(means, 0, atol=0.1))
        self.assertTrue(np.allclose(stds, 1, atol=0.1))
        
    def test_feature_selection(self):
        """Test feature selection"""
        # Create features
        features = self.engineer.engineer_features(
            self.data.head(1000),
            lookback_periods=[5, 10]
        )
        
        # Create target
        target = self.data['close'].pct_change().shift(-1).head(1000)
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        target = target.fillna(0)
        
        # Select features
        selected = self.selector.select_features(
            features,
            target,
            n_features=50,
            methods=['mutual_info', 'random_forest']
        )
        
        # Check selection results
        self.assertEqual(len(selected.columns), 50)
        self.assertTrue(len(self.selector.selected_features_) == 50)
        self.assertTrue(len(self.selector.feature_importance_) > 0)
        
    def test_feature_utils(self):
        """Test feature utility functions"""
        # Test rolling features
        rolling_features = FeatureUtils.create_rolling_features(
            self.data['close'],
            windows=[5, 10],
            functions=['mean', 'std']
        )
        
        self.assertEqual(len(rolling_features.columns), 4)  # 2 windows * 2 functions
        
        # Test lag features
        lag_features = FeatureUtils.create_lag_features(
            self.data[['close', 'volume']],
            features=['close', 'volume'],
            lags=[1, 5]
        )
        
        self.assertEqual(len(lag_features.columns), 4)  # 2 features * 2 lags
        
        # Test time features
        time_features = FeatureUtils.create_time_series_features(self.data.index)
        
        self.assertIn('hour', time_features.columns)
        self.assertIn('dayofweek', time_features.columns)
        self.assertIn('month', time_features.columns)
        
    def test_missing_value_handling(self):
        """Test missing value handling"""
        # Create data with missing values
        data_with_missing = self.data.copy()
        data_with_missing.iloc[10:20, 0] = np.nan
        data_with_missing.iloc[30:40, 1] = np.nan
        
        # Extract features
        features = self.engineer.engineer_features(
            data_with_missing.head(500),
            lookback_periods=[5, 10]
        )
        
        # Handle missing values
        cleaned_features = FeatureUtils.handle_missing_values(
            features,
            method='forward_fill'
        )
        
        # Check no missing values remain
        self.assertEqual(cleaned_features.isnull().sum().sum(), 0)
        
    def test_outlier_detection(self):
        """Test outlier detection"""
        # Create features
        features = self.engineer.engineer_features(
            self.data.head(500),
            lookback_periods=[5, 10]
        )
        
        # Fill missing values
        features = features.fillna(method='ffill').fillna(0)
        
        # Detect outliers
        outliers = FeatureUtils.detect_outliers(features, method='zscore', threshold=3)
        
        # Check outlier detection results
        self.assertEqual(outliers.shape, features.shape)
        self.assertTrue(outliers.dtype == bool)
        
    def test_multicollinearity_removal(self):
        """Test multicollinearity removal"""
        # Create features
        features = self.engineer.engineer_features(
            self.data.head(500),
            lookback_periods=[5, 10]
        )
        
        # Fill missing values
        features = features.fillna(method='ffill').fillna(0)
        
        # Remove highly correlated features
        reduced_features = FeatureUtils.remove_highly_correlated_features(
            features,
            threshold=0.95
        )
        
        # Check that some features were removed
        self.assertLess(len(reduced_features.columns), len(features.columns))
        
    def test_feature_validation(self):
        """Test feature validation"""
        # Create features
        features = self.engineer.engineer_features(
            self.data.head(500),
            lookback_periods=[5, 10]
        )
        
        # Validate features
        validation_results = FeatureUtils.validate_features(features)
        
        # Check validation results
        self.assertIn('n_features', validation_results)
        self.assertIn('n_samples', validation_results)
        self.assertIn('missing_values', validation_results)
        self.assertIn('zero_variance_features', validation_results)
        
        self.assertEqual(validation_results['n_features'], len(features.columns))
        self.assertEqual(validation_results['n_samples'], len(features))


if __name__ == '__main__':
    unittest.main()