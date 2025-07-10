#!/usr/bin/env python3
"""
Comprehensive indicator testing script for confluence strategy
Tests all indicators from strategy_config.yaml with real data
"""

import sys
import os
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
import traceback
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from indicators.technical_indicators import TechnicalIndicators
from indicators.bollinger import BollingerBands
from indicators.rsi import RSI
from indicators.vwap import VWAPIndicator
from data.stock_data_fetcher import StockDataFetcher

class IndicatorTester:
    def __init__(self, config_path: str = 'config/strategy_config.yaml'):
        """Initialize with strategy configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.test_results = {}
        self.errors = []
        
    def load_test_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Load historical data for testing"""
        # Check if data exists locally first
        data_path = f'data/{symbol}.csv'
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path, index_col='date', parse_dates=True)
            print(f"✓ Loaded {symbol} data from cache: {len(df)} rows")
            return df
        else:
            print(f"⚠️  No cached data for {symbol}, please run download_data.py first")
            return None
    
    def test_sma_indicators(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Test Simple Moving Average indicators"""
        results = {}
        periods = self.config['indicators']['sma']['periods']
        
        print(f"\n📊 Testing SMA indicators for {symbol}...")
        
        for period in periods:
            try:
                # Calculate SMA
                sma_values = df['close'].rolling(window=period).mean()
                
                # Validate
                non_nan_count = sma_values.notna().sum()
                if non_nan_count < len(df) - period:
                    raise ValueError(f"SMA{period} has insufficient values")
                
                results[f'SMA{period}'] = {
                    'success': True,
                    'non_nan_values': non_nan_count,
                    'last_value': float(sma_values.iloc[-1]),
                    'current_signal': 'bullish' if df['close'].iloc[-1] > sma_values.iloc[-1] else 'bearish'
                }
                print(f"  ✓ SMA{period}: {results[f'SMA{period}']['last_value']:.2f}")
                
            except Exception as e:
                results[f'SMA{period}'] = {'success': False, 'error': str(e)}
                self.errors.append(f"SMA{period} failed: {str(e)}")
                print(f"  ✗ SMA{period}: {str(e)}")
        
        return results
    
    def test_bollinger_bands(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Test Bollinger Bands indicators"""
        results = {}
        periods = self.config['indicators']['bollinger_bands']['periods']
        std_devs = self.config['indicators']['bollinger_bands']['std_devs']
        
        print(f"\n📊 Testing Bollinger Bands for {symbol}...")
        
        for period in periods:
            for std_dev in std_devs:
                try:
                    bb = BollingerBands(period=period, std_dev=std_dev)
                    bb_data = bb.calculate(df)
                    
                    # Validate
                    if bb_data['upper'].isna().all():
                        raise ValueError(f"BB{period}_{std_dev} all NaN")
                    
                    last_close = df['close'].iloc[-1]
                    last_upper = bb_data['upper'].iloc[-1]
                    last_lower = bb_data['lower'].iloc[-1]
                    
                    # Position relative to bands
                    position = 'neutral'
                    if last_close > last_upper:
                        position = 'above_upper'
                    elif last_close < last_lower:
                        position = 'below_lower'
                    
                    results[f'BB{period}_{std_dev}'] = {
                        'success': True,
                        'upper': float(last_upper),
                        'lower': float(last_lower),
                        'middle': float(bb_data['middle'].iloc[-1]),
                        'position': position,
                        'bandwidth': float((last_upper - last_lower) / bb_data['middle'].iloc[-1])
                    }
                    print(f"  ✓ BB{period}_{std_dev}: Position={position}, BW={results[f'BB{period}_{std_dev}']['bandwidth']:.3f}")
                    
                except Exception as e:
                    results[f'BB{period}_{std_dev}'] = {'success': False, 'error': str(e)}
                    self.errors.append(f"BB{period}_{std_dev} failed: {str(e)}")
                    print(f"  ✗ BB{period}_{std_dev}: {str(e)}")
        
        return results
    
    def test_rsi(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Test RSI indicator"""
        results = {}
        period = self.config['indicators']['rsi']['period']
        oversold = self.config['indicators']['rsi']['oversold']
        overbought = self.config['indicators']['rsi']['overbought']
        
        print(f"\n📊 Testing RSI for {symbol}...")
        
        try:
            rsi_indicator = RSI(period=period)
            rsi_data = rsi_indicator.calculate(df)
            
            last_rsi = rsi_data['rsi'].iloc[-1]
            
            # Determine signal
            signal = 'neutral'
            if last_rsi < oversold:
                signal = 'oversold'
            elif last_rsi > overbought:
                signal = 'overbought'
            
            results['RSI'] = {
                'success': True,
                'value': float(last_rsi),
                'signal': signal,
                'oversold_threshold': oversold,
                'overbought_threshold': overbought
            }
            print(f"  ✓ RSI{period}: {last_rsi:.2f} ({signal})")
            
        except Exception as e:
            results['RSI'] = {'success': False, 'error': str(e)}
            self.errors.append(f"RSI failed: {str(e)}")
            print(f"  ✗ RSI: {str(e)}")
        
        return results
    
    def test_vwap(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Test VWAP indicators"""
        results = {}
        
        print(f"\n📊 Testing VWAP for {symbol}...")
        
        try:
            # Simple daily VWAP
            vwap = VWAPIndicator()
            vwap_data = vwap.calculate(df)
            
            if 'vwap' in vwap_data:
                last_vwap = vwap_data['vwap'].iloc[-1]
                last_close = df['close'].iloc[-1]
                
                results['VWAP'] = {
                    'success': True,
                    'value': float(last_vwap),
                    'price_position': 'above' if last_close > last_vwap else 'below',
                    'deviation_pct': float((last_close - last_vwap) / last_vwap * 100)
                }
                print(f"  ✓ VWAP: {last_vwap:.2f} (Price {results['VWAP']['price_position']}, {results['VWAP']['deviation_pct']:.2f}%)")
            else:
                raise ValueError("VWAP calculation returned no data")
                
        except Exception as e:
            results['VWAP'] = {'success': False, 'error': str(e)}
            self.errors.append(f"VWAP failed: {str(e)}")
            print(f"  ✗ VWAP: {str(e)}")
        
        return results
    
    def test_atr(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Test ATR indicator"""
        results = {}
        period = self.config['indicators']['atr']['period']
        
        print(f"\n📊 Testing ATR for {symbol}...")
        
        try:
            # Calculate ATR manually
            high = df['high']
            low = df['low']
            close = df['close']
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR
            atr = tr.rolling(window=period).mean()
            
            last_atr = atr.iloc[-1]
            last_close = close.iloc[-1]
            atr_pct = (last_atr / last_close) * 100
            
            results['ATR'] = {
                'success': True,
                'value': float(last_atr),
                'percentage': float(atr_pct),
                'volatility': 'high' if atr_pct > 2 else 'normal' if atr_pct > 1 else 'low'
            }
            print(f"  ✓ ATR{period}: {last_atr:.2f} ({atr_pct:.2f}% - {results['ATR']['volatility']} volatility)")
            
        except Exception as e:
            results['ATR'] = {'success': False, 'error': str(e)}
            self.errors.append(f"ATR failed: {str(e)}")
            print(f"  ✗ ATR: {str(e)}")
        
        return results
    
    def run_all_tests(self):
        """Run tests on all configured assets"""
        print("="*60)
        print("🔬 COMPREHENSIVE INDICATOR TESTING")
        print("="*60)
        
        assets = self.config['assets']
        
        for symbol in assets:
            print(f"\n\n{'='*40}")
            print(f"Testing {symbol}")
            print(f"{'='*40}")
            
            # Load data
            df = self.load_test_data(symbol)
            if df is None:
                self.errors.append(f"Failed to load data for {symbol}")
                continue
            
            # Run all indicator tests
            symbol_results = {
                'data_info': {
                    'rows': len(df),
                    'start_date': str(df.index[0]),
                    'end_date': str(df.index[-1]),
                    'current_price': float(df['close'].iloc[-1])
                },
                'indicators': {}
            }
            
            # Test each indicator type
            symbol_results['indicators']['sma'] = self.test_sma_indicators(df, symbol)
            symbol_results['indicators']['bollinger'] = self.test_bollinger_bands(df, symbol)
            symbol_results['indicators']['rsi'] = self.test_rsi(df, symbol)
            symbol_results['indicators']['vwap'] = self.test_vwap(df, symbol)
            symbol_results['indicators']['atr'] = self.test_atr(df, symbol)
            
            self.test_results[symbol] = symbol_results
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
    
    def print_summary(self):
        """Print test summary"""
        print("\n\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        total_tests = 0
        successful_tests = 0
        
        for symbol, results in self.test_results.items():
            if 'indicators' not in results:
                continue
                
            for indicator_type, indicators in results['indicators'].items():
                for name, result in indicators.items():
                    total_tests += 1
                    if result.get('success', False):
                        successful_tests += 1
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success Rate: {(successful_tests/total_tests*100):.1f}%")
        
        if self.errors:
            print(f"\n⚠️  Errors ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
    
    def save_results(self):
        """Save test results to file"""
        import json
        
        output_path = 'reports/confluence_simulation/indicator_test_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    tester = IndicatorTester()
    tester.run_all_tests()