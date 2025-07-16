"""
SuperTrend AI Strategy Implementation

Advanced trading strategy that combines SuperTrend indicator with K-means clustering
for dynamic parameter optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from sklearn.cluster import KMeans
from collections import deque

from ..indicators.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


@dataclass
class SuperTrendConfig:
    """Configuration for SuperTrend AI strategy"""
    atr_length: int = 10
    min_factor: float = 1.0
    max_factor: float = 5.0
    factor_step: float = 0.5
    performance_alpha: float = 10.0
    cluster_selection: str = 'Best'  # 'Best', 'Average', 'Worst'
    use_signal_strength: bool = True
    min_signal_strength: int = 4
    use_time_filter: bool = False
    start_hour: int = 9
    end_hour: int = 16
    max_iterations: int = 1000
    max_data_points: int = 10000
    
    # Risk management
    use_stop_loss: bool = True
    stop_loss_type: str = 'ATR'  # 'ATR' or 'Percentage'
    stop_loss_atr_mult: float = 2.0
    stop_loss_percent: float = 2.0
    use_take_profit: bool = True
    take_profit_type: str = 'Risk/Reward'  # 'Risk/Reward', 'ATR', 'Percentage'
    risk_reward_ratio: float = 2.0
    take_profit_atr_mult: float = 3.0
    take_profit_percent: float = 4.0


class SuperTrendInstance:
    """Single SuperTrend calculation instance"""
    
    def __init__(self, factor: float):
        self.factor = factor
        self.upper = None
        self.lower = None
        self.trend = 0
        self.performance = 0.0
        self.output = None
        
    def update(self, hl2: float, atr: float, close: float, close_prev: float, 
               perf_alpha: float) -> float:
        """Update SuperTrend calculation"""
        # Calculate new bands
        up = hl2 + atr * self.factor
        dn = hl2 - atr * self.factor
        
        # Update trend
        if close > self.upper:
            self.trend = 1
        elif close < self.lower:
            self.trend = 0
        
        # Update bands based on previous close
        if self.upper is not None and close_prev < self.upper:
            self.upper = min(up, self.upper)
        else:
            self.upper = up
            
        if self.lower is not None and close_prev > self.lower:
            self.lower = max(dn, self.lower)
        else:
            self.lower = dn
        
        # Calculate performance
        if self.output is not None:
            price_change = close - close_prev
            position_direction = np.sign(close_prev - self.output)
            diff = price_change * position_direction
            self.performance += 2 / (perf_alpha + 1) * (diff - self.performance)
        
        # Set output
        self.output = self.lower if self.trend == 1 else self.upper
        
        return self.output


class SuperTrendAIStrategy:
    """SuperTrend AI Strategy with K-means clustering"""
    
    def __init__(self, config: Optional[SuperTrendConfig] = None):
        self.config = config or SuperTrendConfig()
        self.supertrends = []
        self.factors = []
        self.target_factor = None
        self.performance_index = None
        self.performance_ama = None
        self.current_position = 0
        self.signals = []
        
        # Initialize SuperTrend instances
        self._initialize_supertrends()
        
    def _initialize_supertrends(self):
        """Initialize multiple SuperTrend instances with different factors"""
        self.factors = []
        self.supertrends = []
        
        num_steps = int((self.config.max_factor - self.config.min_factor) / self.config.factor_step) + 1
        
        for i in range(num_steps):
            factor = self.config.min_factor + i * self.config.factor_step
            self.factors.append(factor)
            self.supertrends.append(SuperTrendInstance(factor))
            
        logger.info(f"Initialized {len(self.factors)} SuperTrend instances with factors: {self.factors}")
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate SuperTrend AI signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals and additional columns
        """
        # Validate input
        required_cols = ['High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        # Calculate ATR
        atr = TechnicalIndicators.atr(
            data['High'], 
            data['Low'], 
            data['Close'], 
            self.config.atr_length
        )
        
        # Initialize result dataframe
        result = data.copy()
        result['signal'] = 0
        result['signal_strength'] = 0
        result['supertrend'] = np.nan
        result['trend_direction'] = 0
        result['selected_factor'] = np.nan
        result['performance_index'] = 0.0
        
        # Storage for clustering
        performance_history = deque(maxlen=self.config.max_data_points)
        
        # Process each bar
        for i in range(self.config.atr_length, len(data)):
            if pd.isna(atr.iloc[i]):
                continue
                
            hl2 = (data['High'].iloc[i] + data['Low'].iloc[i]) / 2
            close = data['Close'].iloc[i]
            close_prev = data['Close'].iloc[i-1] if i > 0 else close
            current_atr = atr.iloc[i]
            
            # Update all SuperTrend instances
            performances = []
            for st in self.supertrends:
                st.update(hl2, current_atr, close, close_prev, self.config.performance_alpha)
                performances.append(st.performance)
            
            # Store performance data for clustering
            performance_history.append({
                'performances': performances.copy(),
                'factors': self.factors.copy()
            })
            
            # Perform clustering if enough data
            if len(performance_history) >= 100 and i % 50 == 0:  # Cluster every 50 bars
                self.target_factor = self._perform_clustering(performance_history)
            
            # Use selected factor or default
            if self.target_factor is None:
                self.target_factor = self.factors[len(self.factors) // 2]  # Middle factor
            
            # Calculate performance index
            if i > self.config.atr_length + 1:
                den = abs(close - close_prev)
                if den > 0:
                    # Find performance of selected cluster
                    factor_idx = self._find_closest_factor_index(self.target_factor)
                    cluster_perf = max(performances[factor_idx], 0)
                    self.performance_index = cluster_perf / den
                else:
                    self.performance_index = 0
            
            # Calculate main SuperTrend with selected factor
            factor_idx = self._find_closest_factor_index(self.target_factor)
            selected_st = self.supertrends[factor_idx]
            
            result.loc[result.index[i], 'supertrend'] = selected_st.output
            result.loc[result.index[i], 'trend_direction'] = selected_st.trend
            result.loc[result.index[i], 'selected_factor'] = self.target_factor
            result.loc[result.index[i], 'performance_index'] = self.performance_index
            
            # Calculate signal strength (0-10 scale)
            signal_strength = min(int(self.performance_index * 10), 10) if self.performance_index else 0
            result.loc[result.index[i], 'signal_strength'] = signal_strength
            
            # Generate trading signals
            if i > 0:
                prev_trend = result['trend_direction'].iloc[i-1]
                curr_trend = selected_st.trend
                
                # Check time filter
                time_ok = True
                if self.config.use_time_filter and hasattr(result.index[i], 'hour'):
                    hour = result.index[i].hour
                    time_ok = self.config.start_hour <= hour <= self.config.end_hour
                
                # Check signal strength filter
                strength_ok = True
                if self.config.use_signal_strength:
                    strength_ok = signal_strength >= self.config.min_signal_strength
                
                # Generate signals
                if time_ok and strength_ok:
                    if curr_trend > prev_trend:  # Bullish crossover
                        result.loc[result.index[i], 'signal'] = 1
                    elif curr_trend < prev_trend:  # Bearish crossover
                        result.loc[result.index[i], 'signal'] = -1
        
        # Add stop loss and take profit levels
        result = self._add_risk_management_levels(result)
        
        return result
    
    def _find_closest_factor_index(self, target_factor: float) -> int:
        """Find index of closest factor to target"""
        min_diff = float('inf')
        closest_idx = 0
        
        for i, factor in enumerate(self.factors):
            diff = abs(factor - target_factor)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
                
        return closest_idx
    
    def _perform_clustering(self, performance_history: deque) -> float:
        """Perform K-means clustering and return selected factor"""
        # Extract performance data
        all_performances = []
        all_factors = []
        
        for entry in performance_history:
            perfs = entry['performances']
            factors = entry['factors']
            
            # Add each factor's performance
            for i, perf in enumerate(perfs):
                all_performances.append(perf)
                all_factors.append(factors[i])
        
        if len(all_performances) < 3:
            return self.target_factor or self.factors[len(self.factors) // 2]
        
        # Convert to numpy arrays
        performances = np.array(all_performances).reshape(-1, 1)
        factors = np.array(all_factors)
        
        try:
            # Initialize K-means with 3 clusters
            kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
            cluster_labels = kmeans.fit_predict(performances)
            
            # Calculate average performance per cluster
            cluster_performances = []
            cluster_factors = []
            
            for i in range(3):
                mask = cluster_labels == i
                if np.any(mask):
                    avg_perf = np.mean(performances[mask])
                    avg_factor = np.mean(factors[mask])
                    cluster_performances.append(avg_perf)
                    cluster_factors.append(avg_factor)
                else:
                    cluster_performances.append(0)
                    cluster_factors.append(self.factors[len(self.factors) // 2])
            
            # Sort clusters by performance
            sorted_indices = np.argsort(cluster_performances)[::-1]  # Descending order
            
            # Select cluster based on configuration
            if self.config.cluster_selection == 'Best':
                selected_idx = sorted_indices[0]
            elif self.config.cluster_selection == 'Average':
                selected_idx = sorted_indices[1]
            else:  # Worst
                selected_idx = sorted_indices[2]
            
            selected_factor = cluster_factors[selected_idx]
            
            logger.debug(f"Clustering complete. Selected factor: {selected_factor:.2f} "
                        f"from {self.config.cluster_selection} cluster")
            
            return selected_factor
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return self.target_factor or self.factors[len(self.factors) // 2]
    
    def _add_risk_management_levels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add stop loss and take profit levels to the dataframe"""
        data['stop_loss'] = np.nan
        data['take_profit'] = np.nan
        
        if not self.config.use_stop_loss and not self.config.use_take_profit:
            return data
        
        # Track position entry
        position = 0
        entry_price = None
        entry_idx = None
        
        for i in range(len(data)):
            signal = data['signal'].iloc[i]
            
            # New position entry
            if signal != 0 and position == 0:
                position = signal
                entry_price = data['Close'].iloc[i]
                entry_idx = i
            
            # Position exit
            elif position != 0 and signal == -position:
                position = 0
                entry_price = None
                entry_idx = None
            
            # Calculate levels for active position
            if position != 0 and entry_price is not None:
                # Calculate ATR at entry
                if 'atr' not in data.columns:
                    atr_series = TechnicalIndicators.atr(
                        data['High'], 
                        data['Low'], 
                        data['Close'], 
                        self.config.atr_length
                    )
                else:
                    atr_series = data['atr']
                
                entry_atr = atr_series.iloc[entry_idx] if entry_idx < len(atr_series) else 0
                
                # Stop loss
                if self.config.use_stop_loss:
                    if self.config.stop_loss_type == 'ATR':
                        sl_distance = entry_atr * self.config.stop_loss_atr_mult
                    else:  # Percentage
                        sl_distance = entry_price * self.config.stop_loss_percent / 100
                    
                    if position > 0:  # Long
                        data.loc[data.index[i], 'stop_loss'] = entry_price - sl_distance
                    else:  # Short
                        data.loc[data.index[i], 'stop_loss'] = entry_price + sl_distance
                
                # Take profit
                if self.config.use_take_profit:
                    if self.config.take_profit_type == 'Risk/Reward':
                        # Need stop loss distance for R:R calculation
                        if self.config.use_stop_loss:
                            tp_distance = sl_distance * self.config.risk_reward_ratio
                        else:
                            tp_distance = entry_price * 0.02 * self.config.risk_reward_ratio  # Default 2%
                    elif self.config.take_profit_type == 'ATR':
                        tp_distance = entry_atr * self.config.take_profit_atr_mult
                    else:  # Percentage
                        tp_distance = entry_price * self.config.take_profit_percent / 100
                    
                    if position > 0:  # Long
                        data.loc[data.index[i], 'take_profit'] = entry_price + tp_distance
                    else:  # Short
                        data.loc[data.index[i], 'take_profit'] = entry_price - tp_distance
        
        return data
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get strategy performance metrics"""
        metrics = {
            'num_supertrends': len(self.supertrends),
            'current_factor': self.target_factor or 0,
            'performance_index': self.performance_index or 0,
            'factors_min': self.config.min_factor,
            'factors_max': self.config.max_factor,
            'factors_step': self.config.factor_step
        }
        
        # Add performance by factor
        for i, (factor, st) in enumerate(zip(self.factors, self.supertrends)):
            metrics[f'perf_factor_{factor:.1f}'] = st.performance
            
        return metrics