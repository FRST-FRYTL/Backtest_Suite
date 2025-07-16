"""
K-means clustering optimizer for parameter optimization and market regime identification.

This module provides utilities for using K-means clustering to optimize trading
strategy parameters based on market conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ClusterAnalysis:
    """Container for cluster analysis results."""
    n_clusters: int
    cluster_centers: np.ndarray
    cluster_labels: np.ndarray
    silhouette_score: float
    calinski_harabasz_score: float
    cluster_sizes: Dict[int, int]
    cluster_characteristics: Dict[int, Dict[str, float]]
    feature_importance: Dict[str, float]
    optimal_params: Dict[int, Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class KMeansOptimizer:
    """
    K-means based optimizer for trading strategy parameters.
    
    Features:
    - Automatic optimal cluster number selection
    - Feature engineering for market conditions
    - Parameter optimization per cluster
    - Cluster stability analysis
    - Real-time cluster assignment
    """
    
    def __init__(
        self,
        min_clusters: int = 3,
        max_clusters: int = 10,
        feature_cols: Optional[List[str]] = None,
        lookback_periods: List[int] = None,
        stability_threshold: float = 0.7,
        min_cluster_size: int = 50,
        random_state: int = 42
    ):
        """
        Initialize K-means optimizer.
        
        Args:
            min_clusters: Minimum number of clusters to test
            max_clusters: Maximum number of clusters to test
            feature_cols: Columns to use for clustering
            lookback_periods: Lookback periods for feature calculation
            stability_threshold: Minimum stability score for clusters
            min_cluster_size: Minimum samples per cluster
            random_state: Random seed for reproducibility
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.feature_cols = feature_cols
        self.lookback_periods = lookback_periods or [5, 10, 20, 50]
        self.stability_threshold = stability_threshold
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.analysis_result = None
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for clustering from OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        for period in self.lookback_periods:
            # Returns
            features[f'return_{period}'] = data['close'].pct_change(period)
            
            # Volatility
            features[f'volatility_{period}'] = data['close'].pct_change().rolling(period).std()
            
            # Price position
            features[f'price_position_{period}'] = (
                (data['close'] - data['low'].rolling(period).min()) /
                (data['high'].rolling(period).max() - data['low'].rolling(period).min())
            )
            
            # High-low range
            features[f'hl_range_{period}'] = (
                (data['high'] - data['low']) / data['close']
            ).rolling(period).mean()
        
        # Volume features if available
        if 'volume' in data.columns:
            for period in self.lookback_periods:
                # Volume ratio
                features[f'volume_ratio_{period}'] = (
                    data['volume'] / data['volume'].rolling(period).mean()
                )
                
                # Price-volume correlation
                features[f'pv_corr_{period}'] = (
                    data['close'].pct_change().rolling(period).corr(
                        data['volume'].pct_change()
                    )
                )
        
        # Technical features
        # RSI
        for period in [14, 28]:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Moving average ratios
        for short, long in [(10, 50), (20, 100)]:
            ma_short = data['close'].rolling(short).mean()
            ma_long = data['close'].rolling(long).mean()
            features[f'ma_ratio_{short}_{long}'] = ma_short / ma_long
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        return features
    
    def select_optimal_clusters(self, features: np.ndarray) -> int:
        """
        Select optimal number of clusters using elbow method and silhouette score.
        
        Args:
            features: Scaled feature matrix
            
        Returns:
            Optimal number of clusters
        """
        scores = []
        silhouette_scores = []
        
        for n_clusters in range(self.min_clusters, self.max_clusters + 1):
            # Skip if we don't have enough samples
            if len(features) < n_clusters * self.min_cluster_size:
                continue
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(features)
            
            # Calculate metrics
            inertia = kmeans.inertia_
            if n_clusters > 1:
                silhouette = silhouette_score(features, labels)
            else:
                silhouette = 0
                
            scores.append(inertia)
            silhouette_scores.append(silhouette)
        
        # Find elbow point
        if len(scores) < 2:
            return self.min_clusters
            
        # Calculate rate of change
        derivatives = np.diff(scores)
        second_derivatives = np.diff(derivatives)
        
        # Find elbow (maximum second derivative)
        if len(second_derivatives) > 0:
            elbow_idx = np.argmax(second_derivatives) + 1
            optimal_clusters = self.min_clusters + elbow_idx
        else:
            # Fallback to best silhouette score
            optimal_clusters = self.min_clusters + np.argmax(silhouette_scores)
        
        # Ensure we're within bounds
        optimal_clusters = max(self.min_clusters, min(optimal_clusters, self.max_clusters))
        
        logger.info(f"Selected {optimal_clusters} clusters based on analysis")
        
        return optimal_clusters
    
    def analyze_clusters(
        self, 
        data: pd.DataFrame,
        features: pd.DataFrame,
        labels: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """
        Analyze characteristics of each cluster.
        
        Args:
            data: Original OHLCV data
            features: Feature DataFrame
            labels: Cluster labels
            
        Returns:
            Dictionary of cluster characteristics
        """
        characteristics = {}
        
        for cluster_id in range(self.kmeans_model.n_clusters):
            cluster_mask = labels == cluster_id
            if not any(cluster_mask):
                continue
                
            # Get cluster data
            cluster_data = data[cluster_mask]
            cluster_features = features[cluster_mask]
            
            # Calculate characteristics
            chars = {
                'size': int(cluster_mask.sum()),
                'avg_return': float(cluster_data['close'].pct_change().mean()),
                'avg_volatility': float(cluster_data['close'].pct_change().std()),
                'trend_strength': float(
                    (cluster_data['close'].iloc[-1] - cluster_data['close'].iloc[0]) /
                    cluster_data['close'].iloc[0] if len(cluster_data) > 0 else 0
                ),
            }
            
            # Add average feature values
            for feature in self.feature_names[:5]:  # Top 5 features
                if feature in cluster_features.columns:
                    chars[f'avg_{feature}'] = float(cluster_features[feature].mean())
            
            characteristics[cluster_id] = chars
            
        return characteristics
    
    def optimize_parameters(
        self,
        data: pd.DataFrame,
        parameter_grid: Dict[str, List[Any]],
        objective_function: Callable,
        n_clusters: Optional[int] = None
    ) -> ClusterAnalysis:
        """
        Optimize parameters for each cluster.
        
        Args:
            data: OHLCV data
            parameter_grid: Dictionary of parameters to optimize
            objective_function: Function to maximize (takes data, params, returns score)
            n_clusters: Number of clusters (None for auto-selection)
            
        Returns:
            ClusterAnalysis with optimized parameters per cluster
        """
        # Engineer features
        features_df = self.engineer_features(data)
        
        # Remove NaN values
        valid_mask = ~features_df.isna().any(axis=1)
        features_df = features_df[valid_mask]
        data_clean = data[valid_mask]
        
        if len(features_df) < self.min_cluster_size * self.min_clusters:
            raise ValueError(f"Insufficient data for clustering: {len(features_df)} samples")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_df)
        
        # Select optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = self.select_optimal_clusters(features_scaled)
        
        # Perform clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = self.kmeans_model.fit_predict(features_scaled)
        
        # Calculate cluster metrics
        if n_clusters > 1:
            silhouette = silhouette_score(features_scaled, labels)
            calinski = calinski_harabasz_score(features_scaled, labels)
        else:
            silhouette = 0
            calinski = 0
        
        # Analyze clusters
        cluster_chars = self.analyze_clusters(data_clean, features_df, labels)
        
        # Calculate cluster sizes
        cluster_sizes = {i: int((labels == i).sum()) for i in range(n_clusters)}
        
        # Optimize parameters for each cluster
        optimal_params = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            if cluster_sizes[cluster_id] < self.min_cluster_size:
                logger.warning(f"Cluster {cluster_id} too small ({cluster_sizes[cluster_id]} samples)")
                continue
                
            cluster_data = data_clean[cluster_mask]
            
            best_score = -np.inf
            best_params = {}
            
            # Grid search over parameter combinations
            param_names = list(parameter_grid.keys())
            param_values = list(parameter_grid.values())
            
            # Generate all combinations
            from itertools import product
            for param_combo in product(*param_values):
                params = dict(zip(param_names, param_combo))
                
                try:
                    score = objective_function(cluster_data, params)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                except Exception as e:
                    logger.warning(f"Error evaluating params {params}: {e}")
                    continue
            
            optimal_params[cluster_id] = {
                'params': best_params,
                'score': best_score,
                'size': cluster_sizes[cluster_id]
            }
        
        # Calculate feature importance
        feature_importance = self.calculate_feature_importance(features_scaled, labels)
        
        # Create analysis result
        self.analysis_result = ClusterAnalysis(
            n_clusters=n_clusters,
            cluster_centers=self.kmeans_model.cluster_centers_,
            cluster_labels=labels,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski,
            cluster_sizes=cluster_sizes,
            cluster_characteristics=cluster_chars,
            feature_importance=feature_importance,
            optimal_params=optimal_params,
            metadata={
                'n_samples': len(labels),
                'n_features': features_scaled.shape[1],
                'feature_names': self.feature_names,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        return self.analysis_result
    
    def calculate_feature_importance(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate feature importance for clustering.
        
        Uses variance ratio between clusters vs within clusters.
        """
        importance = {}
        
        for i, feature_name in enumerate(self.feature_names):
            # Calculate between-cluster variance
            cluster_means = []
            for cluster_id in range(self.kmeans_model.n_clusters):
                cluster_mask = labels == cluster_id
                if any(cluster_mask):
                    cluster_means.append(features[cluster_mask, i].mean())
            
            between_var = np.var(cluster_means) if len(cluster_means) > 1 else 0
            
            # Calculate within-cluster variance
            within_vars = []
            for cluster_id in range(self.kmeans_model.n_clusters):
                cluster_mask = labels == cluster_id
                if any(cluster_mask):
                    within_vars.append(np.var(features[cluster_mask, i]))
            
            within_var = np.mean(within_vars) if within_vars else 1
            
            # Importance is ratio of between to within variance
            importance[feature_name] = float(between_var / (within_var + 1e-8))
        
        # Normalize to sum to 1
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v/total_importance for k, v in importance.items()}
        
        return importance
    
    def predict_cluster(self, data: pd.DataFrame) -> Tuple[int, float]:
        """
        Predict cluster assignment for new data.
        
        Args:
            data: New OHLCV data
            
        Returns:
            Tuple of (cluster_id, confidence)
        """
        if self.kmeans_model is None:
            raise ValueError("Model not fitted. Call optimize_parameters first.")
        
        # Engineer features
        features = self.engineer_features(data)
        
        # Get last valid row
        valid_mask = ~features.isna().any(axis=1)
        if not any(valid_mask):
            raise ValueError("No valid features could be extracted")
        
        last_features = features[valid_mask].iloc[-1].values.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(last_features)
        
        # Predict cluster
        cluster_id = self.kmeans_model.predict(features_scaled)[0]
        
        # Calculate confidence (inverse of distance to centroid)
        distances = np.linalg.norm(
            features_scaled - self.kmeans_model.cluster_centers_[cluster_id],
            axis=1
        )
        confidence = 1 / (1 + distances[0])
        
        return int(cluster_id), float(confidence)
    
    def get_cluster_stability(self, data: pd.DataFrame, window: int = 20) -> float:
        """
        Calculate cluster assignment stability over time.
        
        Args:
            data: OHLCV data
            window: Rolling window for stability calculation
            
        Returns:
            Stability score (0-1)
        """
        if self.kmeans_model is None:
            raise ValueError("Model not fitted. Call optimize_parameters first.")
        
        # Engineer features for entire dataset
        features = self.engineer_features(data)
        valid_mask = ~features.isna().any(axis=1)
        features_clean = features[valid_mask]
        
        if len(features_clean) < window:
            return 0.0
        
        # Scale features
        features_scaled = self.scaler.transform(features_clean)
        
        # Predict clusters for all data points
        clusters = self.kmeans_model.predict(features_scaled)
        
        # Calculate stability as fraction of time in same cluster
        stability_scores = []
        for i in range(window, len(clusters)):
            window_clusters = clusters[i-window:i]
            mode_cluster = pd.Series(window_clusters).mode()[0]
            stability = (window_clusters == mode_cluster).mean()
            stability_scores.append(stability)
        
        return float(np.mean(stability_scores)) if stability_scores else 0.0