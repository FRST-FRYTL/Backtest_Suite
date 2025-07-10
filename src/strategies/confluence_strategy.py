"""
Multi-Indicator Confluence Strategy
Implements a sophisticated trading strategy based on confluence of multiple technical indicators
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import os

class ConfluenceStrategy:
    """
    Multi-indicator confluence strategy with configurable scoring system
    """
    
    def __init__(self, config_path: str = 'config/strategy_config.yaml'):
        """Initialize strategy with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Confluence weights
        self.weights = {
            'trend': 0.30,      # SMA alignments
            'momentum': 0.25,   # RSI signals
            'volatility': 0.25, # Bollinger Bands position
            'volume': 0.20      # VWAP relationships
        }
        
        # Signal thresholds
        self.entry_threshold = self.config['strategy']['entry']['min_confluence_score']
        self.positions = {}  # Track open positions
        self.signals = []    # Store all signals
        
    def calculate_trend_score(self, data: pd.DataFrame, idx: int) -> Tuple[float, Dict]:
        """Calculate trend score based on SMA alignment"""
        sma_periods = self.config['indicators']['sma']['periods']
        current_price = data['close'].iloc[idx]
        
        # Calculate SMAs
        sma_values = {}
        for period in sma_periods:
            if idx >= period:
                sma_values[f'SMA{period}'] = data['close'].iloc[idx-period:idx].mean()
            else:
                sma_values[f'SMA{period}'] = np.nan
        
        # Check alignment
        bullish_count = 0
        bearish_count = 0
        total_comparisons = 0
        
        # Price vs SMAs
        for period in sma_periods:
            if not np.isnan(sma_values[f'SMA{period}']):
                if current_price > sma_values[f'SMA{period}']:
                    bullish_count += 1
                else:
                    bearish_count += 1
                total_comparisons += 1
        
        # SMA alignment (shorter above longer)
        for i in range(len(sma_periods)-1):
            shorter = sma_values[f'SMA{sma_periods[i]}']
            longer = sma_values[f'SMA{sma_periods[i+1]}']
            if not np.isnan(shorter) and not np.isnan(longer):
                if shorter > longer:
                    bullish_count += 1
                else:
                    bearish_count += 1
                total_comparisons += 1
        
        # Calculate score
        if total_comparisons > 0:
            bullish_score = bullish_count / total_comparisons
            bearish_score = bearish_count / total_comparisons
            score = bullish_score  # 0 to 1, where 1 is perfect bullish alignment
        else:
            score = 0.5  # Neutral
        
        details = {
            'sma_values': sma_values,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'alignment_score': score
        }
        
        return score, details
    
    def calculate_momentum_score(self, data: pd.DataFrame, idx: int) -> Tuple[float, Dict]:
        """Calculate momentum score based on RSI"""
        period = self.config['indicators']['rsi']['period']
        oversold = self.config['indicators']['rsi']['oversold']
        overbought = self.config['indicators']['rsi']['overbought']
        
        if idx < period:
            return 0.5, {'rsi': np.nan, 'signal': 'insufficient_data'}
        
        # Calculate RSI
        close_prices = data['close'].iloc[idx-period:idx+1].values
        gains = []
        losses = []
        
        for i in range(1, len(close_prices)):
            change = close_prices[i] - close_prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Score calculation
        if rsi < oversold:
            score = 1.0  # Strong buy signal
            signal = 'oversold'
        elif rsi < 50:
            score = 0.5 + (50 - rsi) / (50 - oversold) * 0.3  # 0.5 to 0.8
            signal = 'bullish'
        elif rsi < overbought:
            score = 0.5 - (rsi - 50) / (overbought - 50) * 0.3  # 0.2 to 0.5
            signal = 'bearish'
        else:
            score = 0.0  # Strong sell signal
            signal = 'overbought'
        
        details = {
            'rsi': rsi,
            'signal': signal,
            'oversold_threshold': oversold,
            'overbought_threshold': overbought
        }
        
        return score, details
    
    def calculate_volatility_score(self, data: pd.DataFrame, idx: int) -> Tuple[float, Dict]:
        """Calculate volatility score based on Bollinger Bands position"""
        # Use 20-period BB with 2 std dev as primary
        period = 20
        std_dev = 2.0
        
        if idx < period:
            return 0.5, {'position': 'insufficient_data'}
        
        # Calculate Bollinger Bands
        close_prices = data['close'].iloc[idx-period+1:idx+1]
        sma = close_prices.mean()
        std = close_prices.std()
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        current_price = data['close'].iloc[idx]
        
        # Calculate position within bands
        band_width = upper_band - lower_band
        if band_width > 0:
            position_in_band = (current_price - lower_band) / band_width
        else:
            position_in_band = 0.5
        
        # Score calculation
        if position_in_band < 0.2:
            score = 0.9  # Near lower band - buy signal
            signal = 'near_lower'
        elif position_in_band < 0.4:
            score = 0.7
            signal = 'lower_half'
        elif position_in_band < 0.6:
            score = 0.5
            signal = 'middle'
        elif position_in_band < 0.8:
            score = 0.3
            signal = 'upper_half'
        else:
            score = 0.1  # Near upper band - sell signal
            signal = 'near_upper'
        
        # Adjust for squeeze (low volatility)
        bandwidth_pct = (band_width / sma) * 100
        if bandwidth_pct < 2:  # Tight bands
            score = score * 1.2  # Boost score for potential breakout
            signal += '_squeeze'
        
        details = {
            'upper_band': upper_band,
            'lower_band': lower_band,
            'middle_band': sma,
            'position_in_band': position_in_band,
            'bandwidth_pct': bandwidth_pct,
            'signal': signal
        }
        
        return min(score, 1.0), details
    
    def calculate_volume_score(self, data: pd.DataFrame, idx: int) -> Tuple[float, Dict]:
        """Calculate volume score based on VWAP relationship"""
        if idx < 20:  # Need minimum data
            return 0.5, {'signal': 'insufficient_data'}
        
        # Calculate simple VWAP for the day
        typical_price = (data['high'].iloc[idx-20:idx+1] + 
                        data['low'].iloc[idx-20:idx+1] + 
                        data['close'].iloc[idx-20:idx+1]) / 3
        volume = data['volume'].iloc[idx-20:idx+1]
        
        vwap = (typical_price * volume).sum() / volume.sum()
        current_price = data['close'].iloc[idx]
        current_volume = data['volume'].iloc[idx]
        avg_volume = data['volume'].iloc[idx-20:idx].mean()
        
        # Price vs VWAP
        price_vs_vwap = (current_price - vwap) / vwap
        
        # Volume analysis
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Score calculation
        if current_price < vwap:
            if volume_ratio > 1.5:
                score = 0.9  # Below VWAP with high volume - strong buy
                signal = 'below_vwap_high_volume'
            else:
                score = 0.7  # Below VWAP with normal volume
                signal = 'below_vwap'
        else:
            if volume_ratio < 0.7:
                score = 0.3  # Above VWAP with low volume - weak
                signal = 'above_vwap_low_volume'
            else:
                score = 0.5  # Above VWAP with normal volume
                signal = 'above_vwap'
        
        details = {
            'vwap': vwap,
            'price_vs_vwap_pct': price_vs_vwap * 100,
            'volume_ratio': volume_ratio,
            'signal': signal
        }
        
        return score, details
    
    def calculate_confluence_score(self, data: pd.DataFrame, idx: int) -> Tuple[float, Dict]:
        """Calculate overall confluence score combining all indicators"""
        # Get individual scores
        trend_score, trend_details = self.calculate_trend_score(data, idx)
        momentum_score, momentum_details = self.calculate_momentum_score(data, idx)
        volatility_score, volatility_details = self.calculate_volatility_score(data, idx)
        volume_score, volume_details = self.calculate_volume_score(data, idx)
        
        # Calculate weighted confluence score
        confluence_score = (
            trend_score * self.weights['trend'] +
            momentum_score * self.weights['momentum'] +
            volatility_score * self.weights['volatility'] +
            volume_score * self.weights['volume']
        )
        
        # Compile details
        details = {
            'confluence_score': confluence_score,
            'component_scores': {
                'trend': trend_score,
                'momentum': momentum_score,
                'volatility': volatility_score,
                'volume': volume_score
            },
            'component_details': {
                'trend': trend_details,
                'momentum': momentum_details,
                'volatility': volatility_details,
                'volume': volume_details
            },
            'timestamp': data.index[idx],
            'price': data['close'].iloc[idx]
        }
        
        return confluence_score, details
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate trading signals based on confluence scores"""
        signals = []
        
        # Calculate minimum required data points
        min_period = max(self.config['indicators']['sma']['periods'] + [20])
        
        for idx in range(min_period, len(data)):
            # Skip if we have a recent position
            if self._check_reentry_delay(symbol, data.index[idx]):
                continue
            
            # Calculate confluence score
            score, details = self.calculate_confluence_score(data, idx)
            
            # Generate signal if threshold met
            if score >= self.entry_threshold:
                signal = {
                    'timestamp': data.index[idx],
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': data['close'].iloc[idx],
                    'confluence_score': score,
                    'details': details
                }
                signals.append(signal)
                
                # Update position tracking
                self.positions[symbol] = {
                    'entry_date': data.index[idx],
                    'entry_price': data['close'].iloc[idx],
                    'confluence_score': score
                }
        
        # Convert to DataFrame
        if signals:
            signals_df = pd.DataFrame(signals)
            signals_df.set_index('timestamp', inplace=True)
        else:
            signals_df = pd.DataFrame()
        
        return signals_df
    
    def _check_reentry_delay(self, symbol: str, current_date: pd.Timestamp) -> bool:
        """Check if we're within the reentry delay period"""
        if symbol not in self.positions:
            return False
        
        last_entry = self.positions[symbol]['entry_date']
        delay_days = self.config['strategy']['entry']['reentry_delay_days']
        
        if isinstance(last_entry, str):
            last_entry = pd.to_datetime(last_entry)
        
        return (current_date - last_entry).days < delay_days
    
    def calculate_position_size(self, capital: float, score: float) -> float:
        """Calculate position size using Kelly criterion"""
        kelly_fraction = self.config['strategy']['position_sizing']['kelly_fraction']
        max_position_pct = self.config['strategy']['position_sizing']['max_position_pct']
        min_position_size = self.config['strategy']['position_sizing']['min_position_size']
        
        # Adjust Kelly fraction based on confluence score
        adjusted_kelly = kelly_fraction * (score - 0.5) * 2  # Scale from 0 to kelly_fraction
        
        # Calculate position size
        position_size = capital * adjusted_kelly
        
        # Apply constraints
        position_size = min(position_size, capital * max_position_pct)
        position_size = max(position_size, min_position_size)
        
        return position_size