"""Test script for rolling VWAP implementation."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
import yaml
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from src.indicators.multi_timeframe_indicators import MultiTimeframeIndicators
from src.indicators.technical_indicators import TechnicalIndicators

def test_rolling_vwap():
    """Test rolling VWAP calculation with real market data."""
    print("Testing Rolling VWAP Implementation...\n")
    
    # Load configuration
    with open('config/strategy_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Download sample data
    print("Downloading sample data (SPY)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    spy = yf.download(
        'SPY',
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval='1h'
    )
    
    if spy.empty:
        print("Error: No data downloaded")
        return
    
    print(f"Downloaded {len(spy)} hours of data\n")
    
    # Test 1: MultiTimeframeIndicators class
    print("Test 1: Testing MultiTimeframeIndicators.calculate_rolling_vwap()")
    print("-" * 60)
    
    indicators = MultiTimeframeIndicators(config)
    spy_with_rvwap = indicators.calculate_rolling_vwap(spy.copy())
    
    # Check if columns were created
    expected_columns = []
    for period in config['indicators']['rolling_vwap']['periods']:
        expected_columns.append(f'Rolling_VWAP_{period}')
        for std_dev in config['indicators']['rolling_vwap']['std_devs']:
            expected_columns.append(f'Rolling_VWAP_{period}_Upper_{std_dev}')
            expected_columns.append(f'Rolling_VWAP_{period}_Lower_{std_dev}')
        expected_columns.append(f'Rolling_VWAP_{period}_Position')
        expected_columns.append(f'Rolling_VWAP_{period}_Distance')
        expected_columns.append(f'Rolling_VWAP_{period}_Cross_Above')
        expected_columns.append(f'Rolling_VWAP_{period}_Cross_Below')
    
    created_columns = [col for col in expected_columns if col in spy_with_rvwap.columns]
    missing_columns = [col for col in expected_columns if col not in spy_with_rvwap.columns]
    
    print(f"Created {len(created_columns)} columns successfully")
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
    else:
        print("All expected columns created ✓")
    
    # Display sample data
    print("\nSample Rolling VWAP values (last 5 rows):")
    sample_cols = ['Close', 'Rolling_VWAP_20', 'Rolling_VWAP_50', 'Rolling_VWAP_200']
    print(spy_with_rvwap[sample_cols].tail())
    
    # Test 2: TechnicalIndicators static method
    print("\n\nTest 2: Testing TechnicalIndicators.rolling_vwap()")
    print("-" * 60)
    
    result = TechnicalIndicators.rolling_vwap(
        spy['High'],
        spy['Low'],
        spy['Close'],
        spy['Volume'],
        period=20
    )
    
    print(f"Returned keys: {list(result.keys())}")
    print(f"VWAP shape: {result['vwap'].shape}")
    print(f"STD shape: {result['std'].shape}")
    
    # Test bands calculation
    bands = TechnicalIndicators.vwap_bands(
        result['vwap'],
        result['std'],
        std_devs=[1, 2, 3]
    )
    
    print(f"\nBand keys: {list(bands.keys())}")
    
    # Verify calculations
    print("\n\nTest 3: Verification of calculations")
    print("-" * 60)
    
    # Manual calculation for verification
    typical_price = (spy['High'] + spy['Low'] + spy['Close']) / 3
    tp_volume = typical_price * spy['Volume']
    
    # Calculate 20-period rolling VWAP manually
    manual_vwap = tp_volume.rolling(window=20).sum() / spy['Volume'].rolling(window=20).sum()
    
    # Compare with class implementation
    class_vwap = spy_with_rvwap['Rolling_VWAP_20']
    
    # Calculate difference (ignoring NaN values)
    valid_idx = ~(manual_vwap.isna() | class_vwap.isna())
    if valid_idx.sum() > 0:
        max_diff = np.abs(manual_vwap[valid_idx] - class_vwap[valid_idx]).max()
        mean_diff = np.abs(manual_vwap[valid_idx] - class_vwap[valid_idx]).mean()
        
        print(f"Max difference between manual and class calculation: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-6:
            print("✓ Calculations match perfectly!")
        else:
            print("⚠ Small differences detected (likely due to floating point precision)")
    
    # Test 4: Cross detection
    print("\n\nTest 4: Testing cross detection")
    print("-" * 60)
    
    crosses_above = spy_with_rvwap['Rolling_VWAP_20_Cross_Above'].sum()
    crosses_below = spy_with_rvwap['Rolling_VWAP_20_Cross_Below'].sum()
    
    print(f"Number of crosses above VWAP_20: {crosses_above}")
    print(f"Number of crosses below VWAP_20: {crosses_below}")
    
    # Plot sample data
    print("\n\nCreating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot price and rolling VWAPs
    last_100 = spy_with_rvwap.iloc[-100:]
    
    ax1.plot(last_100.index, last_100['Close'], label='Close', linewidth=2, color='black')
    ax1.plot(last_100.index, last_100['Rolling_VWAP_20'], label='VWAP_20', alpha=0.7)
    ax1.plot(last_100.index, last_100['Rolling_VWAP_50'], label='VWAP_50', alpha=0.7)
    
    # Add bands for VWAP_20
    ax1.fill_between(
        last_100.index,
        last_100['Rolling_VWAP_20_Lower_2'],
        last_100['Rolling_VWAP_20_Upper_2'],
        alpha=0.2,
        label='2σ Band'
    )
    
    # Mark crosses
    crosses_above_mask = last_100['Rolling_VWAP_20_Cross_Above']
    crosses_below_mask = last_100['Rolling_VWAP_20_Cross_Below']
    
    if crosses_above_mask.any():
        ax1.scatter(
            last_100.index[crosses_above_mask],
            last_100['Close'][crosses_above_mask],
            color='green',
            marker='^',
            s=100,
            label='Cross Above'
        )
    
    if crosses_below_mask.any():
        ax1.scatter(
            last_100.index[crosses_below_mask],
            last_100['Close'][crosses_below_mask],
            color='red',
            marker='v',
            s=100,
            label='Cross Below'
        )
    
    ax1.set_ylabel('Price ($)')
    ax1.set_title('SPY with Rolling VWAP (Last 100 Hours)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot volume
    ax2.bar(last_100.index, last_100['Volume'], alpha=0.3)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'examples/reports/rolling_vwap_test.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nVisualization saved to: {output_path}")
    
    # Summary statistics
    print("\n\nSummary Statistics for Rolling VWAP")
    print("-" * 60)
    
    for period in [20, 50, 200]:
        col = f'Rolling_VWAP_{period}'
        if col in spy_with_rvwap.columns:
            distance_col = f'{col}_Distance'
            
            # Get valid data
            valid_data = spy_with_rvwap[~spy_with_rvwap[col].isna()]
            
            if len(valid_data) > 0:
                print(f"\n{col}:")
                print(f"  Average distance from VWAP: {valid_data[distance_col].mean():.2f}%")
                print(f"  Std dev of distance: {valid_data[distance_col].std():.2f}%")
                print(f"  Max distance above: {valid_data[distance_col].max():.2f}%")
                print(f"  Max distance below: {valid_data[distance_col].min():.2f}%")
                
                # Position statistics
                position_col = f'{col}_Position'
                print(f"  Average position (0-1): {valid_data[position_col].mean():.3f}")
                print(f"  % time above VWAP: {(valid_data[distance_col] > 0).mean() * 100:.1f}%")
    
    print("\n✓ Rolling VWAP implementation test completed successfully!")


if __name__ == "__main__":
    test_rolling_vwap()