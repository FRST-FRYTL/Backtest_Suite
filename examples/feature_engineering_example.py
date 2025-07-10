"""
Example script demonstrating the feature engineering pipeline
Shows how to extract 500+ features and perform feature selection
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ml.features import FeatureEngineer, FeatureSelector
from data.data_loader import DataLoader


def load_feature_config(config_path: str = 'config/feature_config.yaml'):
    """Load feature configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def demonstrate_feature_engineering():
    """Demonstrate the complete feature engineering pipeline"""
    
    print("=== Feature Engineering Pipeline Demo ===\n")
    
    # 1. Load configuration
    config = load_feature_config('../config/feature_config.yaml')
    print("✓ Loaded feature configuration")
    
    # 2. Load sample data
    print("\n2. Loading sample data...")
    data_loader = DataLoader()
    
    # Generate sample data for demonstration
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='H')
    np.random.seed(42)
    
    # Create realistic OHLCV data
    n_points = len(dates)
    base_price = 100
    returns = np.random.normal(0.0001, 0.01, n_points)
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': close_prices * (1 + np.random.uniform(-0.002, 0.002, n_points)),
        'high': close_prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
        'low': close_prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
        'close': close_prices,
        'volume': np.random.lognormal(10, 1, n_points)
    }, index=dates)
    
    print(f"✓ Loaded data: {len(data)} rows")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    
    # 3. Initialize feature engineer
    print("\n3. Initializing feature engineer...")
    engineer = FeatureEngineer(timeframes=config['timeframes'])
    
    # 4. Engineer features
    print("\n4. Engineering features (this may take a while)...")
    features = engineer.engineer_features(
        data, 
        lookback_periods=config['lookback_periods']['all']
    )
    
    print(f"✓ Engineered {len(features.columns)} features")
    print(f"  Feature shape: {features.shape}")
    
    # Display feature categories
    feature_categories = {}
    for col in features.columns:
        category = col.split('_')[0]
        if category not in feature_categories:
            feature_categories[category] = 0
        feature_categories[category] += 1
    
    print("\n  Feature categories:")
    for category, count in sorted(feature_categories.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    - {category}: {count} features")
    
    # 5. Handle missing values and scale features
    print("\n5. Preprocessing features...")
    features_scaled = engineer.scale_features(features, method='robust')
    print(f"✓ Scaled features using robust scaling")
    
    # 6. Create target variable (next period return)
    print("\n6. Creating target variable...")
    target = data['close'].pct_change().shift(-1)  # Next period return
    target = target.loc[features.index]  # Align with features
    
    # Remove last row (no future return)
    features_scaled = features_scaled[:-1]
    target = target[:-1]
    
    print(f"✓ Created target variable: {len(target)} samples")
    
    # 7. Feature selection
    print("\n7. Performing feature selection...")
    selector = FeatureSelector(task_type='regression')
    
    selected_features = selector.select_features(
        features_scaled.iloc[:10000],  # Use subset for faster demo
        target.iloc[:10000],
        n_features=config['feature_selection']['target_features'],
        methods=config['feature_selection']['methods']
    )
    
    print(f"✓ Selected top {len(selected_features.columns)} features")
    
    # 8. Display feature importance
    print("\n8. Feature importance analysis...")
    importance_report = selector.get_feature_importance_report()
    
    print("\nTop 20 most important features:")
    print(importance_report.head(20)[['final_score', 'selected']])
    
    # 9. Correlation analysis
    print("\n9. Correlation analysis of selected features...")
    corr_analysis = selector.get_correlation_analysis(features_scaled)
    
    print(f"  Average absolute correlation: {corr_analysis['average_absolute_correlation']:.3f}")
    print(f"  Maximum correlation: {corr_analysis['max_correlation']:.3f}")
    print(f"  Highly correlated pairs: {len(corr_analysis['high_correlation_pairs'])}")
    
    # 10. Feature statistics
    print("\n10. Feature statistics...")
    feature_stats = engineer.get_feature_stats(selected_features)
    
    print("\nSample feature statistics:")
    print(feature_stats.head(10)[['mean', 'std', 'null_pct']])
    
    # 11. Visualizations
    print("\n11. Creating visualizations...")
    
    # Feature importance plot
    fig1 = selector.plot_feature_importance(top_n=30)
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved feature importance plot")
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = selected_features.iloc[:, :30].corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap (Top 30 Features)')
    plt.tight_layout()
    plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved correlation heatmap")
    
    # Feature distribution
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(selected_features.columns[:6]):
        axes[i].hist(selected_features[feature].dropna(), bins=50, alpha=0.7)
        axes[i].set_title(f'Distribution: {feature}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved feature distribution plots")
    
    # 12. Save results
    print("\n12. Saving results...")
    
    # Save selected features
    selected_features.to_parquet('selected_features.parquet')
    print("✓ Saved selected features to selected_features.parquet")
    
    # Save feature list
    with open('selected_feature_names.txt', 'w') as f:
        for feature in selected_features.columns:
            f.write(f"{feature}\n")
    print("✓ Saved feature names to selected_feature_names.txt")
    
    # Save importance report
    importance_report.to_csv('feature_importance_report.csv')
    print("✓ Saved importance report to feature_importance_report.csv")
    
    print("\n=== Feature Engineering Complete ===")
    print(f"\nSummary:")
    print(f"  - Total features engineered: {len(features.columns)}")
    print(f"  - Features selected: {len(selected_features.columns)}")
    print(f"  - Data points: {len(selected_features)}")
    print(f"  - Feature categories: {len(feature_categories)}")
    
    return selected_features, target, selector


def demonstrate_multi_timeframe_features():
    """Demonstrate multi-timeframe feature engineering"""
    
    print("\n=== Multi-Timeframe Feature Engineering ===\n")
    
    # Load hourly data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='H')
    np.random.seed(42)
    
    n_points = len(dates)
    base_price = 100
    returns = np.random.normal(0.0001, 0.01, n_points)
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    hourly_data = pd.DataFrame({
        'open': close_prices * (1 + np.random.uniform(-0.002, 0.002, n_points)),
        'high': close_prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
        'low': close_prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
        'close': close_prices,
        'volume': np.random.lognormal(10, 1, n_points)
    }, index=dates)
    
    # Resample to different timeframes
    timeframes = {
        '1H': hourly_data,
        '4H': hourly_data.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }),
        'D': hourly_data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
    }
    
    # Engineer features for each timeframe
    all_features = {}
    
    for tf_name, tf_data in timeframes.items():
        print(f"\nProcessing {tf_name} timeframe...")
        engineer = FeatureEngineer()
        features = engineer.engineer_features(tf_data, lookback_periods=[5, 10, 20])
        all_features[tf_name] = features
        print(f"  ✓ Engineered {len(features.columns)} features")
    
    # Align features to hourly timeframe
    print("\nAligning multi-timeframe features...")
    
    aligned_features = all_features['1H'].copy()
    
    # Add higher timeframe features
    for tf_name in ['4H', 'D']:
        tf_features = all_features[tf_name]
        # Resample to hourly and forward fill
        tf_features_hourly = tf_features.resample('H').ffill()
        # Add prefix to column names
        tf_features_hourly.columns = [f'{tf_name}_{col}' for col in tf_features_hourly.columns]
        # Merge with aligned features
        aligned_features = pd.concat([aligned_features, tf_features_hourly], axis=1)
    
    print(f"\n✓ Created multi-timeframe feature set with {len(aligned_features.columns)} features")
    
    return aligned_features


if __name__ == "__main__":
    # Run demonstrations
    selected_features, target, selector = demonstrate_feature_engineering()
    
    # Demonstrate multi-timeframe features
    multi_tf_features = demonstrate_multi_timeframe_features()
    
    print("\n✅ All demonstrations complete!")