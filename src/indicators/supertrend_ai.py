"""
SuperTrend AI indicator implementation with K-means clustering optimization.

This module implements the SuperTrend indicator with AI-enhanced parameter optimization
using K-means clustering to identify optimal ATR multiplier and period values.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


@dataclass
class SuperTrendResult:
    """Container for SuperTrend calculation results."""
    trend: pd.Series  # 1 for uptrend, -1 for downtrend
    upper_band: pd.Series
    lower_band: pd.Series
    support_resistance: pd.Series  # Current support/resistance level
    signal: pd.Series  # Buy/sell signals
    atr_values: pd.Series
    optimal_params: Dict[str, float]
    cluster_info: Optional[Dict[str, any]] = None


class SuperTrendAI:
    """
    AI-enhanced SuperTrend indicator with K-means clustering optimization.
    
    Features:
    - Dynamic ATR period and multiplier optimization
    - K-means clustering for market condition identification
    - Adaptive parameter adjustment based on volatility regime
    - Support/resistance level tracking
    """
    
    def __init__(
        self,
        atr_periods: List[int] = None,
        multipliers: List[float] = None,
        n_clusters: int = 5,
        lookback_window: int = 252,
        adaptive: bool = True,
        volatility_adjustment: bool = True
    ):
        """
        Initialize SuperTrend AI indicator.
        
        Args:
            atr_periods: List of ATR periods to test (default: [7, 10, 14, 20])
            multipliers: List of multipliers to test (default: [1.5, 2.0, 2.5, 3.0])
            n_clusters: Number of clusters for K-means
            lookback_window: Lookback period for optimization
            adaptive: Whether to use adaptive parameters
            volatility_adjustment: Whether to adjust for volatility regime
        """
        self.atr_periods = atr_periods or [7, 10, 14, 20]
        self.multipliers = multipliers or [1.5, 2.0, 2.5, 3.0]
        self.n_clusters = n_clusters
        self.lookback_window = lookback_window
        self.adaptive = adaptive
        self.volatility_adjustment = volatility_adjustment
        
        self.kmeans = None
        self.scaler = StandardScaler()
        self.cluster_params = {}
        
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_supertrend(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series,
        period: int,
        multiplier: float
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate basic SuperTrend indicator.
        
        Returns:
            Tuple of (trend, upper_band, lower_band)
        """
        # Calculate ATR
        atr = self.calculate_atr(high, low, close, period)
        
        # Calculate basic bands
        hl_avg = (high + low) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        # Initialize trend
        trend = pd.Series(index=close.index, dtype=float)
        trend.iloc[0] = 1
        
        # Calculate trend
        for i in range(1, len(close)):
            if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                trend.iloc[i] = trend.iloc[i-1]
                continue
                
            # Uptrend conditions
            if close.iloc[i] <= upper_band.iloc[i]:
                if trend.iloc[i-1] == 1:
                    if lower_band.iloc[i] > lower_band.iloc[i-1]:
                        lower_band.iloc[i] = lower_band.iloc[i]
                    else:
                        lower_band.iloc[i] = lower_band.iloc[i-1]
            
            # Downtrend conditions
            if close.iloc[i] >= lower_band.iloc[i]:
                if trend.iloc[i-1] == -1:
                    if upper_band.iloc[i] < upper_band.iloc[i-1]:
                        upper_band.iloc[i] = upper_band.iloc[i]
                    else:
                        upper_band.iloc[i] = upper_band.iloc[i-1]
            
            # Determine trend
            if close.iloc[i] > upper_band.iloc[i]:
                trend.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1]
                
        return trend, upper_band, lower_band
    
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for K-means clustering."""
        features = []
        
        # Price-based features
        returns = data['close'].pct_change()
        features.append(returns.rolling(20).mean())  # 20-day return
        features.append(returns.rolling(20).std())   # 20-day volatility
        
        # Volume features
        if 'volume' in data.columns:
            volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
            features.append(volume_ratio)
        
        # Price position
        price_position = (data['close'] - data['low'].rolling(20).min()) / \
                        (data['high'].rolling(20).max() - data['low'].rolling(20).min())
        features.append(price_position)
        
        # ATR ratio
        atr = self.calculate_atr(data['high'], data['low'], data['close'], 14)
        atr_ratio = atr / data['close']
        features.append(atr_ratio)
        
        # Combine features
        feature_matrix = pd.concat(features, axis=1).dropna()
        
        return feature_matrix.values
    
    def optimize_parameters(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Optimize SuperTrend parameters using K-means clustering.
        
        Returns:
            Dictionary with optimal period and multiplier
        """
        # Extract features
        features = self.extract_features(data)
        
        if len(features) < self.n_clusters:
            logger.warning("Insufficient data for clustering, using defaults")
            return {"period": 10, "multiplier": 2.0}
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Perform K-means clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = self.kmeans.fit_predict(features_scaled)
        
        # Test parameter combinations for each cluster
        cluster_performance = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = clusters == cluster_id
            if not any(cluster_mask):
                continue
                
            best_score = -np.inf
            best_params = {"period": 10, "multiplier": 2.0}
            
            # Get cluster data
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) < 50:  # Need minimum data
                continue
                
            for period in self.atr_periods:
                for multiplier in self.multipliers:
                    # Calculate SuperTrend for this parameter set
                    trend, _, _ = self.calculate_supertrend(
                        data['high'], data['low'], data['close'],
                        period, multiplier
                    )
                    
                    # Calculate performance metric (profit factor)
                    returns = data['close'].pct_change()
                    strategy_returns = returns * trend.shift(1)
                    
                    # Focus on cluster-specific performance
                    cluster_returns = strategy_returns.iloc[cluster_indices]
                    
                    if len(cluster_returns) > 0:
                        wins = cluster_returns[cluster_returns > 0].sum()
                        losses = abs(cluster_returns[cluster_returns < 0].sum())
                        
                        if losses > 0:
                            profit_factor = wins / losses
                        else:
                            profit_factor = wins
                        
                        if profit_factor > best_score:
                            best_score = profit_factor
                            best_params = {"period": period, "multiplier": multiplier}
            
            cluster_performance[cluster_id] = {
                "params": best_params,
                "score": best_score,
                "size": sum(cluster_mask)
            }
        
        # Store cluster parameters
        self.cluster_params = cluster_performance
        
        # Return parameters for current market condition
        current_features = features_scaled[-1].reshape(1, -1)
        current_cluster = self.kmeans.predict(current_features)[0]
        
        if current_cluster in cluster_performance:
            return cluster_performance[current_cluster]["params"]
        else:
            return {"period": 10, "multiplier": 2.0}
    
    def calculate(self, data: pd.DataFrame) -> SuperTrendResult:
        """
        Calculate AI-enhanced SuperTrend indicator.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            SuperTrendResult with trends, bands, and signals
        """
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        # Get optimal parameters
        if self.adaptive and len(data) >= self.lookback_window:
            optimal_params = self.optimize_parameters(data.tail(self.lookback_window))
        else:
            optimal_params = {"period": 10, "multiplier": 2.0}
        
        # Calculate SuperTrend with optimal parameters
        trend, upper_band, lower_band = self.calculate_supertrend(
            data['high'], data['low'], data['close'],
            int(optimal_params['period']),
            optimal_params['multiplier']
        )
        
        # Adjust for volatility if enabled
        if self.volatility_adjustment:
            current_vol = data['close'].pct_change().rolling(20).std().iloc[-1]
            hist_vol = data['close'].pct_change().rolling(252).std().iloc[-1]
            
            if not pd.isna(current_vol) and not pd.isna(hist_vol) and hist_vol > 0:
                vol_ratio = current_vol / hist_vol
                vol_adjustment = np.clip(vol_ratio, 0.5, 2.0)
                
                # Widen bands in high volatility
                upper_band = upper_band * (1 + (vol_adjustment - 1) * 0.5)
                lower_band = lower_band * (1 - (vol_adjustment - 1) * 0.5)
        
        # Calculate support/resistance levels
        support_resistance = pd.Series(index=data.index, dtype=float)
        for i in range(len(trend)):
            if trend.iloc[i] == 1:
                support_resistance.iloc[i] = lower_band.iloc[i]
            else:
                support_resistance.iloc[i] = upper_band.iloc[i]
        
        # Generate signals
        signal = pd.Series(0, index=data.index)
        signal[(trend == 1) & (trend.shift(1) == -1)] = 1   # Buy signal
        signal[(trend == -1) & (trend.shift(1) == 1)] = -1  # Sell signal
        
        # Calculate ATR for reference
        atr_values = self.calculate_atr(
            data['high'], data['low'], data['close'], 
            int(optimal_params['period'])
        )
        
        # Prepare cluster info if available
        cluster_info = None
        if self.kmeans is not None and self.cluster_params:
            cluster_info = {
                "n_clusters": self.n_clusters,
                "cluster_params": self.cluster_params,
                "current_cluster": self.kmeans.predict(
                    self.scaler.transform(self.extract_features(data)[-1:])
                )[0] if len(self.extract_features(data)) > 0 else None
            }
        
        return SuperTrendResult(
            trend=trend,
            upper_band=upper_band,
            lower_band=lower_band,
            support_resistance=support_resistance,
            signal=signal,
            atr_values=atr_values,
            optimal_params=optimal_params,
            cluster_info=cluster_info
        )
    
    def get_signal_strength(self, result: SuperTrendResult, current_price: float) -> float:
        """
        Calculate signal strength based on price distance from bands.
        
        Returns:
            Signal strength between 0 and 1
        """
        if result.trend.iloc[-1] == 1:  # Uptrend
            # Distance from lower band (support)
            distance = (current_price - result.lower_band.iloc[-1]) / result.atr_values.iloc[-1]
        else:  # Downtrend
            # Distance from upper band (resistance)
            distance = (result.upper_band.iloc[-1] - current_price) / result.atr_values.iloc[-1]
        
        # Normalize to 0-1 range
        strength = 1 / (1 + np.exp(-distance))
        
        return strength