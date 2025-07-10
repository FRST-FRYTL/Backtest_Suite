"""
Quick demonstration of the feature engineering pipeline
Shows how to extract 500+ features from OHLCV data
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ml.features import FeatureEngineer, FeatureSelector


def main():
    print("=== Feature Engineering Pipeline Demo ===\n")
    
    # 1. Generate sample data
    print("1. Generating sample OHLCV data...")
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    n_points = len(dates)
    
    np.random.seed(42)
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
    
    print(f"✓ Generated {len(data)} hourly data points")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    
    # 2. Initialize feature engineer
    print("\n2. Initializing feature engineer...")
    engineer = FeatureEngineer(timeframes=['1H', '4H', 'D'])
    
    # 3. Engineer features (using smaller subset for demo)
    print("\n3. Engineering features...")
    print("   This extracts 500+ features including:")
    print("   - Price-based features (returns, ratios, positions)")
    print("   - Technical indicators (RSI, MACD, Bollinger Bands, etc.)")
    print("   - Volume features (VWAP, OBV, CMF, MFI)")
    print("   - Statistical features (rolling stats, volatility)")
    print("   - Market microstructure features")
    print("   - Time-based features")
    print("   - Interaction and lag features")
    
    # Use a subset for faster demo
    data_subset = data.head(2000)
    features = engineer.engineer_features(
        data_subset,
        lookback_periods=[5, 10, 20, 50]
    )
    
    print(f"\n✓ Extracted {len(features.columns)} features!")
    
    # 4. Display feature breakdown
    print("\n4. Feature breakdown by category:")
    categories = {}
    for col in features.columns:
        # Simple categorization based on feature name
        if 'return' in col:
            cat = 'Returns'
        elif 'rsi' in col or 'macd' in col or 'bb_' in col or 'stoch' in col:
            cat = 'Technical Indicators'
        elif 'volume' in col or 'vwap' in col or 'obv' in col:
            cat = 'Volume'
        elif 'roll' in col or 'std' in col or 'mean' in col:
            cat = 'Statistical'
        elif 'hour' in col or 'day' in col or 'month' in col:
            cat = 'Time'
        elif 'lag' in col:
            cat = 'Lag Features'
        else:
            cat = 'Other'
            
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {cat}: {count} features")
    
    # 5. Scale features
    print("\n5. Scaling features...")
    features_scaled = engineer.scale_features(features, method='robust')
    print("✓ Applied robust scaling to handle outliers")
    
    # 6. Feature selection demo
    print("\n6. Feature selection...")
    
    # Create a simple target (next period return)
    target = data_subset['close'].pct_change().shift(-1)
    target = target.loc[features.index].fillna(0)
    
    selector = FeatureSelector(task_type='regression')
    
    # Select top 50 features
    selected_features = selector.select_features(
        features_scaled.iloc[:1000],  # Use subset for speed
        target.iloc[:1000],
        n_features=50,
        methods=['mutual_info', 'correlation']
    )
    
    print(f"✓ Selected top {len(selected_features.columns)} features")
    
    # 7. Display top features
    print("\n7. Top 10 most important features:")
    importance_df = pd.DataFrame({
        'feature': selector.feature_importance_.keys(),
        'importance': selector.feature_importance_.values()
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(10).iterrows():
        print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    # 8. Feature statistics
    print("\n8. Feature statistics summary:")
    stats = engineer.get_feature_stats(selected_features)
    print(f"   - Features with no missing values: {(stats['null_count'] == 0).sum()}")
    print(f"   - Average null percentage: {stats['null_pct'].mean():.2f}%")
    print(f"   - Features with >10% nulls: {(stats['null_pct'] > 10).sum()}")
    
    print("\n✅ Feature engineering pipeline demonstration complete!")
    print(f"\nFinal output:")
    print(f"  - Total features engineered: {len(features.columns)}")
    print(f"  - Features after selection: {len(selected_features.columns)}")
    print(f"  - Ready for ML model training!")
    
    return features, selected_features


if __name__ == "__main__":
    features, selected_features = main()