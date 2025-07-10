"""
Data Engineering Agent for ML Pipeline

Handles data collection, preprocessing, validation, and feature engineering
for the machine learning pipeline in the backtesting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import os

from .base_agent import BaseAgent


class DataEngineeringAgent(BaseAgent):
    """
    Agent responsible for data engineering tasks including:
    - Data collection and validation
    - Preprocessing and cleaning
    - Feature engineering
    - Data quality checks
    - Dataset splitting and preparation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("DataEngineeringAgent", config)
        self.data_sources = []
        self.processed_data = None
        self.feature_stats = {}
        self.data_quality_report = {}
        
    def initialize(self) -> bool:
        """Initialize data engineering resources."""
        try:
            self.logger.info("Initializing Data Engineering Agent")
            
            # Validate required configuration
            required_keys = ["data_sources", "preprocessing_config", "validation_rules"]
            if not self.validate_config(required_keys):
                return False
            
            # Initialize data sources
            self.data_sources = self.config.get("data_sources", [])
            
            # Set up preprocessing pipeline
            self.preprocessing_config = self.config.get("preprocessing_config", {})
            
            # Initialize validation rules
            self.validation_rules = self.config.get("validation_rules", {})
            
            self.logger.info("Data Engineering Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute data engineering pipeline.
        
        Args:
            data_path: Optional path to data file
            start_date: Optional start date for data filtering
            end_date: Optional end date for data filtering
            
        Returns:
            Dict containing processed data and statistics
        """
        try:
            # Collect data
            raw_data = self._collect_data(kwargs.get("data_path"))
            
            # Validate data
            validation_results = self._validate_data(raw_data)
            
            # Preprocess data
            processed_data = self._preprocess_data(raw_data)
            
            # Engineer features
            feature_data = self._engineer_features(processed_data)
            
            # Generate quality report
            quality_report = self._generate_quality_report(feature_data)
            
            # Split data
            split_data = self._split_data(
                feature_data,
                kwargs.get("train_ratio", 0.7),
                kwargs.get("val_ratio", 0.15)
            )
            
            self.processed_data = split_data
            self.data_quality_report = quality_report
            
            return {
                "data_shape": {
                    "train": split_data["train"].shape,
                    "val": split_data["val"].shape,
                    "test": split_data["test"].shape
                },
                "feature_count": len(self.feature_stats),
                "quality_report": quality_report,
                "validation_results": validation_results,
                "feature_stats": self.feature_stats
            }
            
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            raise
    
    def _collect_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Collect data from configured sources."""
        self.logger.info("Collecting data from sources")
        
        if data_path and os.path.exists(data_path):
            # Load from file
            if data_path.endswith('.csv'):
                return pd.read_csv(data_path, parse_dates=['date'], index_col='date')
            elif data_path.endswith('.parquet'):
                return pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
        
        # Otherwise, create sample data for demonstration
        self.logger.warning("No data path provided, generating sample data")
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        data = pd.DataFrame({
            'open': np.random.randn(len(dates)).cumsum() + 100,
            'high': np.random.randn(len(dates)).cumsum() + 101,
            'low': np.random.randn(len(dates)).cumsum() + 99,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        return data
    
    def _validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data against configured rules."""
        self.logger.info("Validating data")
        
        validation_results = {
            "passed": True,
            "checks": {}
        }
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        validation_results["checks"]["missing_values"] = {
            "passed": missing_counts.sum() == 0,
            "details": missing_counts.to_dict()
        }
        
        # Check data types
        expected_types = self.validation_rules.get("expected_types", {})
        type_checks = {}
        for col, expected_type in expected_types.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                type_checks[col] = actual_type == expected_type
        
        validation_results["checks"]["data_types"] = {
            "passed": all(type_checks.values()),
            "details": type_checks
        }
        
        # Check value ranges
        range_checks = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            range_checks[col] = {
                "min": data[col].min(),
                "max": data[col].max(),
                "mean": data[col].mean(),
                "std": data[col].std()
            }
        
        validation_results["checks"]["value_ranges"] = range_checks
        
        # Update overall status
        validation_results["passed"] = all(
            check["passed"] for check in validation_results["checks"].values()
            if isinstance(check, dict) and "passed" in check
        )
        
        return validation_results
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw data."""
        self.logger.info("Preprocessing data")
        
        processed = data.copy()
        
        # Handle missing values
        fill_method = self.preprocessing_config.get("fill_method", "forward")
        if fill_method == "forward":
            processed = processed.fillna(method='ffill')
        elif fill_method == "interpolate":
            processed = processed.interpolate()
        
        # Remove outliers
        if self.preprocessing_config.get("remove_outliers", False):
            for col in processed.select_dtypes(include=[np.number]).columns:
                q1 = processed[col].quantile(0.25)
                q3 = processed[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                processed = processed[
                    (processed[col] >= lower_bound) & 
                    (processed[col] <= upper_bound)
                ]
        
        # Normalize if requested
        if self.preprocessing_config.get("normalize", False):
            numeric_cols = processed.select_dtypes(include=[np.number]).columns
            processed[numeric_cols] = (
                processed[numeric_cols] - processed[numeric_cols].mean()
            ) / processed[numeric_cols].std()
        
        return processed
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from preprocessed data."""
        self.logger.info("Engineering features")
        
        features = data.copy()
        
        # Technical indicators
        if 'close' in features.columns:
            # Simple Moving Averages
            for period in [5, 10, 20, 50]:
                features[f'sma_{period}'] = features['close'].rolling(period).mean()
            
            # Exponential Moving Averages
            for period in [12, 26]:
                features[f'ema_{period}'] = features['close'].ewm(span=period).mean()
            
            # RSI
            delta = features['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            features['bb_middle'] = features['close'].rolling(20).mean()
            bb_std = features['close'].rolling(20).std()
            features['bb_upper'] = features['bb_middle'] + 2 * bb_std
            features['bb_lower'] = features['bb_middle'] - 2 * bb_std
            
            # Price changes
            features['returns'] = features['close'].pct_change()
            features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        
        # Volume indicators
        if 'volume' in features.columns:
            features['volume_sma'] = features['volume'].rolling(10).mean()
            features['volume_ratio'] = features['volume'] / features['volume_sma']
        
        # Time-based features
        if isinstance(features.index, pd.DatetimeIndex):
            features['day_of_week'] = features.index.dayofweek
            features['month'] = features.index.month
            features['quarter'] = features.index.quarter
        
        # Calculate feature statistics
        self.feature_stats = {
            col: {
                "mean": features[col].mean(),
                "std": features[col].std(),
                "min": features[col].min(),
                "max": features[col].max()
            }
            for col in features.select_dtypes(include=[np.number]).columns
        }
        
        # Drop rows with NaN values from feature engineering
        features = features.dropna()
        
        return features
    
    def _generate_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        self.logger.info("Generating data quality report")
        
        report = {
            "summary": {
                "total_rows": len(data),
                "total_columns": len(data.columns),
                "date_range": {
                    "start": str(data.index.min()),
                    "end": str(data.index.max())
                } if isinstance(data.index, pd.DatetimeIndex) else None
            },
            "completeness": {
                col: {
                    "missing_count": data[col].isnull().sum(),
                    "missing_percentage": (data[col].isnull().sum() / len(data)) * 100
                }
                for col in data.columns
            },
            "data_types": {
                col: str(dtype) for col, dtype in data.dtypes.items()
            },
            "numeric_summary": {}
        }
        
        # Numeric column statistics
        for col in data.select_dtypes(include=[np.number]).columns:
            report["numeric_summary"][col] = {
                "mean": float(data[col].mean()),
                "std": float(data[col].std()),
                "min": float(data[col].min()),
                "25%": float(data[col].quantile(0.25)),
                "50%": float(data[col].quantile(0.50)),
                "75%": float(data[col].quantile(0.75)),
                "max": float(data[col].max()),
                "skewness": float(data[col].skew()),
                "kurtosis": float(data[col].kurtosis())
            }
        
        return report
    
    def _split_data(self, data: pd.DataFrame, train_ratio: float = 0.7, 
                    val_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        self.logger.info("Splitting data into train/val/test sets")
        
        n = len(data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:train_size + val_size]
        test_data = data.iloc[train_size + val_size:]
        
        self.logger.info(f"Data split - Train: {len(train_data)}, "
                        f"Val: {len(val_data)}, Test: {len(test_data)}")
        
        return {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }
    
    def get_processed_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Return the processed data splits."""
        return self.processed_data
    
    def get_feature_names(self) -> List[str]:
        """Return list of engineered feature names."""
        if self.processed_data and "train" in self.processed_data:
            return list(self.processed_data["train"].columns)
        return []
    
    def export_data(self, output_dir: str, format: str = "parquet"):
        """Export processed data to files."""
        if not self.processed_data:
            raise ValueError("No processed data available to export")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, split_data in self.processed_data.items():
            if format == "parquet":
                split_data.to_parquet(f"{output_dir}/{split_name}.parquet")
            elif format == "csv":
                split_data.to_csv(f"{output_dir}/{split_name}.csv")
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Data exported to {output_dir} in {format} format")