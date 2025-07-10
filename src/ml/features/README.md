# Feature Engineering Module

## Overview

The Feature Engineering module provides a comprehensive pipeline for extracting and selecting features from financial time series data. It can generate 500+ features from OHLCV data across multiple timeframes.

## Key Components

### 1. FeatureEngineer (`feature_engineering.py`)
Main class for feature extraction that creates:

#### Price-Based Features (150+)
- Returns and log returns (multiple periods)
- Price ratios (high/low, close/open)
- Price positions within rolling windows
- Gap features (gap up/down indicators)
- True range and daily range

#### Technical Indicators (200+)
- **Moving Averages**: SMA, EMA across multiple periods
- **Momentum**: RSI, Stochastic, Williams %R
- **Trend**: MACD, ADX, Aroon
- **Volatility**: Bollinger Bands, ATR, Keltner Channels
- **Volume**: OBV, CMF, MFI, VWAP
- **Eastern**: Ichimoku Cloud components

#### Statistical Features (100+)
- Rolling statistics (mean, std, skewness, kurtosis)
- Realized volatility (multiple methods)
- Autocorrelation features
- Distribution moments

#### Market Microstructure Features (30+)
- Bid-ask spread proxy
- Kyle's Lambda (price impact)
- Amihud illiquidity measure
- Roll's effective spread
- Volume clock and trade intensity

#### Time-Based Features (20+)
- Hour, day, month, quarter features
- Cyclical encoding (sin/cos transforms)
- Trading session indicators
- End-of-period flags

#### Interaction Features (50+)
- Indicator crossovers
- Divergences between indicators
- Feature combinations and ratios

#### Lag Features (100+)
- Lagged values of key features
- Multiple lag periods (1, 2, 3, 5, 10, 20)

### 2. FeatureSelector (`feature_selector.py`)
Advanced feature selection with multiple methods:

- **Mutual Information**: Non-linear dependency detection
- **Random Forest Importance**: Tree-based feature ranking
- **Lasso Regularization**: L1-based feature selection
- **Correlation Analysis**: Target correlation ranking
- **Multicollinearity Removal**: Reduces redundant features
- **Recursive Feature Elimination**: Iterative selection
- **Stability Analysis**: Cross-validation based selection

### 3. FeatureUtils (`feature_utils.py`)
Utility functions for:
- Rolling/expanding window features
- Missing value handling
- Outlier detection and treatment
- Feature scaling and normalization
- Feature validation

## Usage Example

```python
from ml.features import FeatureEngineer, FeatureSelector

# Initialize feature engineer
engineer = FeatureEngineer(timeframes=['1H', '4H', 'D'])

# Extract features
features = engineer.engineer_features(
    ohlcv_data,
    lookback_periods=[5, 10, 20, 50, 100]
)

# Scale features
features_scaled = engineer.scale_features(features, method='robust')

# Select top features
selector = FeatureSelector(task_type='regression')
selected_features = selector.select_features(
    features_scaled,
    target,
    n_features=100,
    methods=['mutual_info', 'random_forest', 'lasso']
)

# Get feature importance report
importance_report = selector.get_feature_importance_report()
```

## Configuration

Features are configured via `config/feature_config.yaml`:
- Timeframes for multi-timeframe analysis
- Lookback periods for rolling calculations
- Feature categories to include/exclude
- Selection parameters and thresholds
- Scaling and preprocessing options

## Performance Considerations

- **Memory Usage**: 500+ features can consume significant memory
- **Computation Time**: Full feature extraction takes ~30s per 10k rows
- **Parallel Processing**: Uses n_jobs=-1 for parallel computation
- **Caching**: Features can be cached to disk (parquet format)

## Feature Quality

The pipeline includes several quality checks:
- Removes zero-variance features
- Handles missing values (forward fill, interpolation)
- Detects and removes highly correlated features (>0.95)
- Validates feature distributions
- Tracks feature importance over time

## Integration with ML Models

Selected features are ready for use with:
- XGBoost/LightGBM models
- LSTM/GRU networks
- Ensemble methods
- Linear models (after scaling)

## Best Practices

1. **Start with fewer features**: Use feature selection to identify top 50-100
2. **Monitor feature drift**: Track feature distributions over time
3. **Use robust scaling**: Handles outliers better than standard scaling
4. **Validate on out-of-sample**: Ensure features generalize well
5. **Consider computational cost**: Balance feature richness vs speed

## Future Enhancements

- GPU acceleration for faster computation
- Real-time feature updates
- Alternative data integration (sentiment, news)
- Automated feature engineering (genetic algorithms)
- Feature interaction discovery