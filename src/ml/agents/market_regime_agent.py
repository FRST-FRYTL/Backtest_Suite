"""
Market Regime Agent for ML Pipeline

Identifies and analyzes market regimes to improve model performance
across different market conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_white
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent


class MarketRegimeAgent(BaseAgent):
    """
    Agent responsible for market regime analysis including:
    - Regime detection and classification
    - Volatility regime analysis
    - Trend identification
    - Market microstructure analysis
    - Regime transition probability modeling
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MarketRegimeAgent", config)
        self.regime_labels = None
        self.regime_model = None
        self.regime_features = None
        self.transition_matrix = None
        self.regime_statistics = {}
        self.current_regime = None
        
    def initialize(self) -> bool:
        """Initialize market regime analysis resources."""
        try:
            self.logger.info("Initializing Market Regime Agent")
            
            # Validate required configuration
            required_keys = ["regime_method", "n_regimes", "features_config"]
            if not self.validate_config(required_keys):
                return False
            
            # Initialize regime detection settings
            self.regime_method = self.config.get("regime_method", "hmm")
            self.n_regimes = self.config.get("n_regimes", 4)
            self.features_config = self.config.get("features_config", {
                "returns": True,
                "volatility": True,
                "volume": True,
                "microstructure": True
            })
            
            # Initialize analysis windows
            self.lookback_windows = self.config.get("lookback_windows", [5, 20, 60])
            
            # Initialize regime names
            self.regime_names = self.config.get("regime_names", [
                "Bull Market", "Bear Market", "High Volatility", "Low Volatility"
            ])
            
            self.logger.info("Market Regime Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False
    
    def execute(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Execute market regime analysis.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            Dict containing regime analysis results
        """
        try:
            # Engineer regime features
            regime_features = self._engineer_regime_features(data)
            
            # Detect regimes
            regimes = self._detect_regimes(regime_features)
            
            # Analyze regime characteristics
            regime_stats = self._analyze_regime_characteristics(
                data, regimes, regime_features
            )
            
            # Model regime transitions
            transition_analysis = self._analyze_regime_transitions(regimes)
            
            # Identify current regime
            current_regime = self._identify_current_regime(
                regime_features, regimes
            )
            
            # Analyze regime stability
            stability_analysis = self._analyze_regime_stability(
                regimes, regime_features
            )
            
            # Generate regime forecasts
            regime_forecast = self._forecast_regime_transitions(
                regimes, transition_analysis
            )
            
            # Create visualizations
            viz_results = self._generate_regime_visualizations(
                data, regimes, regime_features
            )
            
            # Store results
            self.regime_labels = regimes
            self.regime_statistics = regime_stats
            self.current_regime = current_regime
            
            return {
                "regime_labels": regimes,
                "n_regimes_detected": len(np.unique(regimes)),
                "regime_statistics": regime_stats,
                "transition_analysis": transition_analysis,
                "current_regime": current_regime,
                "stability_analysis": stability_analysis,
                "regime_forecast": regime_forecast,
                "visualizations": viz_results,
                "regime_features": regime_features.columns.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            raise
    
    def _engineer_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for regime detection."""
        self.logger.info("Engineering regime features")
        
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        if self.features_config.get("returns", True):
            # Returns at different scales
            for window in self.lookback_windows:
                features[f'returns_{window}d'] = data['close'].pct_change(window)
                features[f'returns_{window}d_ma'] = (
                    data['close'].pct_change().rolling(window).mean()
                )
            
            # Momentum indicators
            features['momentum_short'] = data['close'].pct_change(5).rolling(20).mean()
            features['momentum_long'] = data['close'].pct_change(20).rolling(60).mean()
        
        # Volatility features
        if self.features_config.get("volatility", True):
            returns = data['close'].pct_change()
            
            for window in self.lookback_windows:
                # Realized volatility
                features[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
                
                # GARCH-like features
                features[f'squared_returns_{window}d'] = (returns ** 2).rolling(window).mean()
                
                # High-low volatility
                features[f'hl_volatility_{window}d'] = (
                    (np.log(data['high'] / data['low'])).rolling(window).mean()
                )
            
            # Volatility ratios
            features['vol_ratio_short_long'] = (
                features['volatility_5d'] / features['volatility_60d']
            )
        
        # Volume features
        if self.features_config.get("volume", True) and 'volume' in data.columns:
            for window in self.lookback_windows:
                features[f'volume_ma_{window}d'] = data['volume'].rolling(window).mean()
                features[f'volume_ratio_{window}d'] = (
                    data['volume'] / data['volume'].rolling(window).mean()
                )
            
            # Volume-price correlation
            features['volume_price_corr'] = (
                data['close'].pct_change().rolling(20).corr(data['volume'].pct_change())
            )
        
        # Market microstructure features
        if self.features_config.get("microstructure", True):
            # Bid-ask spread proxy (using high-low)
            features['spread_proxy'] = (data['high'] - data['low']) / data['close']
            
            # Amihud illiquidity
            if 'volume' in data.columns:
                features['illiquidity'] = (
                    abs(data['close'].pct_change()) / (data['volume'] * data['close'])
                ).rolling(20).mean()
            
            # Price efficiency (variance ratio test)
            features['variance_ratio'] = self._calculate_variance_ratio(data['close'])
        
        # Technical indicators for regime
        features['rsi'] = self._calculate_rsi(data['close'])
        features['trend_strength'] = self._calculate_trend_strength(data['close'])
        
        # Drop NaN values
        features = features.dropna()
        self.regime_features = features
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_trend_strength(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate trend strength using linear regression slope."""
        def calculate_slope(x):
            if len(x) < 2:
                return 0
            indices = np.arange(len(x))
            slope, _ = np.polyfit(indices, x, 1)
            return slope / x.mean() if x.mean() != 0 else 0
        
        return prices.rolling(window).apply(calculate_slope)
    
    def _calculate_variance_ratio(self, prices: pd.Series, lag: int = 5) -> pd.Series:
        """Calculate variance ratio for market efficiency test."""
        returns = prices.pct_change().dropna()
        
        def variance_ratio(returns_window):
            if len(returns_window) < lag * 2:
                return 1.0
            
            # Calculate variances
            var_1 = returns_window.var()
            var_lag = returns_window.rolling(lag).sum().dropna().var() / lag
            
            return var_lag / var_1 if var_1 > 0 else 1.0
        
        return returns.rolling(60).apply(variance_ratio)
    
    def _detect_regimes(self, features: pd.DataFrame) -> np.ndarray:
        """Detect market regimes using configured method."""
        self.logger.info(f"Detecting regimes using {self.regime_method}")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        if self.regime_method == "hmm":
            regimes = self._detect_regimes_hmm(features_scaled)
        elif self.regime_method == "kmeans":
            regimes = self._detect_regimes_kmeans(features_scaled)
        elif self.regime_method == "gmm":
            regimes = self._detect_regimes_gmm(features_scaled)
        elif self.regime_method == "dbscan":
            regimes = self._detect_regimes_dbscan(features_scaled)
        else:
            regimes = self._detect_regimes_hierarchical(features_scaled)
        
        return regimes
    
    def _detect_regimes_hmm(self, features: np.ndarray) -> np.ndarray:
        """Detect regimes using Hidden Markov Model."""
        # Use PCA to reduce dimensionality if needed
        if features.shape[1] > 10:
            pca = PCA(n_components=10)
            features = pca.fit_transform(features)
        
        # Fit HMM
        model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        model.fit(features)
        regimes = model.predict(features)
        
        self.regime_model = model
        self.transition_matrix = model.transmat_
        
        return regimes
    
    def _detect_regimes_kmeans(self, features: np.ndarray) -> np.ndarray:
        """Detect regimes using K-Means clustering."""
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        regimes = kmeans.fit_predict(features)
        self.regime_model = kmeans
        return regimes
    
    def _detect_regimes_gmm(self, features: np.ndarray) -> np.ndarray:
        """Detect regimes using Gaussian Mixture Model."""
        gmm = GaussianMixture(n_components=self.n_regimes, random_state=42)
        regimes = gmm.fit_predict(features)
        self.regime_model = gmm
        return regimes
    
    def _detect_regimes_dbscan(self, features: np.ndarray) -> np.ndarray:
        """Detect regimes using DBSCAN clustering."""
        dbscan = DBSCAN(eps=0.5, min_samples=50)
        regimes = dbscan.fit_predict(features)
        
        # Map outliers to nearest cluster
        if -1 in regimes:
            # Find nearest non-outlier point for each outlier
            outlier_indices = np.where(regimes == -1)[0]
            non_outlier_indices = np.where(regimes != -1)[0]
            
            for idx in outlier_indices:
                distances = np.sum((features[non_outlier_indices] - features[idx]) ** 2, axis=1)
                nearest_idx = non_outlier_indices[np.argmin(distances)]
                regimes[idx] = regimes[nearest_idx]
        
        self.regime_model = dbscan
        return regimes
    
    def _detect_regimes_hierarchical(self, features: np.ndarray) -> np.ndarray:
        """Detect regimes using hierarchical clustering."""
        hierarchical = AgglomerativeClustering(n_clusters=self.n_regimes)
        regimes = hierarchical.fit_predict(features)
        self.regime_model = hierarchical
        return regimes
    
    def _analyze_regime_characteristics(self, data: pd.DataFrame, 
                                      regimes: np.ndarray,
                                      features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of each regime."""
        self.logger.info("Analyzing regime characteristics")
        
        regime_stats = {}
        
        # Align data with regime labels
        regime_data = data.loc[features.index].copy()
        regime_data['regime'] = regimes
        
        for regime in np.unique(regimes):
            regime_mask = regimes == regime
            regime_subset = regime_data[regime_data['regime'] == regime]
            
            if len(regime_subset) == 0:
                continue
            
            # Calculate returns statistics
            returns = regime_subset['close'].pct_change().dropna()
            
            stats = {
                "count": len(regime_subset),
                "percentage": len(regime_subset) / len(regime_data) * 100,
                "returns": {
                    "mean_daily": returns.mean(),
                    "mean_annual": returns.mean() * 252,
                    "volatility_daily": returns.std(),
                    "volatility_annual": returns.std() * np.sqrt(252),
                    "sharpe_ratio": (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                    "skewness": returns.skew(),
                    "kurtosis": returns.kurtosis(),
                    "max_drawdown": self._calculate_max_drawdown(regime_subset['close'])
                },
                "duration": {
                    "mean_days": self._calculate_regime_durations(regimes == regime).mean(),
                    "max_days": self._calculate_regime_durations(regimes == regime).max(),
                    "min_days": self._calculate_regime_durations(regimes == regime).min()
                }
            }
            
            # Add feature averages
            feature_means = features[regime_mask].mean()
            stats["feature_averages"] = feature_means.to_dict()
            
            # Determine regime name based on characteristics
            regime_name = self._determine_regime_name(stats, regime)
            regime_stats[regime_name] = stats
        
        return regime_stats
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_regime_durations(self, regime_mask: np.ndarray) -> np.ndarray:
        """Calculate durations of regime periods."""
        # Find regime change points
        changes = np.diff(np.concatenate([[0], regime_mask.astype(int), [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        durations = ends - starts
        return durations if len(durations) > 0 else np.array([0])
    
    def _determine_regime_name(self, stats: Dict[str, Any], regime: int) -> str:
        """Determine regime name based on characteristics."""
        if regime < len(self.regime_names):
            return self.regime_names[regime]
        
        # Auto-determine based on statistics
        mean_return = stats["returns"]["mean_annual"]
        volatility = stats["returns"]["volatility_annual"]
        
        if mean_return > 0.1 and volatility < 0.2:
            return "Bull Market"
        elif mean_return < -0.1 and volatility > 0.25:
            return "Bear Market"
        elif volatility > 0.3:
            return "High Volatility"
        else:
            return f"Regime {regime}"
    
    def _analyze_regime_transitions(self, regimes: np.ndarray) -> Dict[str, Any]:
        """Analyze regime transition patterns."""
        self.logger.info("Analyzing regime transitions")
        
        # Calculate transition matrix
        n_regimes = len(np.unique(regimes))
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regimes) - 1):
            transition_matrix[regimes[i], regimes[i + 1]] += 1
        
        # Normalize to get probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(
            transition_matrix, row_sums,
            where=row_sums != 0,
            out=np.zeros_like(transition_matrix)
        )
        
        # Calculate steady-state distribution
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        steady_state_idx = np.argmax(np.abs(eigenvalues))
        steady_state = np.real(eigenvectors[:, steady_state_idx])
        steady_state = steady_state / steady_state.sum()
        
        # Calculate expected regime durations
        expected_durations = 1 / (1 - np.diag(transition_matrix))
        expected_durations[np.isinf(expected_durations)] = 0
        
        return {
            "transition_matrix": transition_matrix.tolist(),
            "steady_state_distribution": steady_state.tolist(),
            "expected_durations": expected_durations.tolist(),
            "persistence": np.diag(transition_matrix).tolist()
        }
    
    def _identify_current_regime(self, features: pd.DataFrame, 
                               regimes: np.ndarray) -> Dict[str, Any]:
        """Identify current market regime."""
        current_regime_idx = regimes[-1]
        
        # Calculate confidence using regime model
        confidence = 1.0
        if hasattr(self.regime_model, 'predict_proba'):
            probabilities = self.regime_model.predict_proba(
                features.iloc[-1:].values
            )[0]
            confidence = probabilities[current_regime_idx]
        
        # Determine regime name
        regime_name = self._get_regime_name(current_regime_idx)
        
        # Calculate time in current regime
        time_in_regime = 1
        for i in range(len(regimes) - 2, -1, -1):
            if regimes[i] == current_regime_idx:
                time_in_regime += 1
            else:
                break
        
        return {
            "regime": regime_name,
            "regime_index": int(current_regime_idx),
            "confidence": float(confidence),
            "days_in_regime": time_in_regime,
            "features": features.iloc[-1].to_dict()
        }
    
    def _get_regime_name(self, regime_idx: int) -> str:
        """Get regime name from index."""
        if regime_idx < len(self.regime_names):
            return self.regime_names[regime_idx]
        return f"Regime {regime_idx}"
    
    def _analyze_regime_stability(self, regimes: np.ndarray, 
                                features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze regime stability and transition risk."""
        self.logger.info("Analyzing regime stability")
        
        # Calculate regime switching frequency
        switches = np.diff(regimes) != 0
        switching_frequency = switches.sum() / len(regimes)
        
        # Calculate average time between switches
        switch_indices = np.where(switches)[0]
        if len(switch_indices) > 1:
            avg_time_between_switches = np.mean(np.diff(switch_indices))
        else:
            avg_time_between_switches = len(regimes)
        
        # Analyze feature stability in current regime
        current_regime = regimes[-1]
        regime_mask = regimes == current_regime
        
        if regime_mask.sum() > 20:
            recent_features = features[regime_mask].tail(20)
            feature_stability = {
                col: {
                    "std": recent_features[col].std(),
                    "trend": self._calculate_trend_strength(recent_features[col]).iloc[-1]
                }
                for col in recent_features.columns[:5]  # Top 5 features
            }
        else:
            feature_stability = {}
        
        return {
            "switching_frequency": float(switching_frequency),
            "avg_days_between_switches": float(avg_time_between_switches),
            "current_regime_stability": self._assess_stability(
                regimes, current_regime
            ),
            "feature_stability": feature_stability
        }
    
    def _assess_stability(self, regimes: np.ndarray, current_regime: int) -> str:
        """Assess stability of current regime."""
        # Look at recent regime history
        recent_regimes = regimes[-60:]  # Last 60 periods
        regime_changes = np.sum(np.diff(recent_regimes) != 0)
        
        if regime_changes == 0:
            return "Very Stable"
        elif regime_changes <= 2:
            return "Stable"
        elif regime_changes <= 5:
            return "Moderately Stable"
        else:
            return "Unstable"
    
    def _forecast_regime_transitions(self, regimes: np.ndarray,
                                   transition_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast potential regime transitions."""
        self.logger.info("Forecasting regime transitions")
        
        current_regime = regimes[-1]
        transition_probs = transition_analysis["transition_matrix"][current_regime]
        
        # Calculate expected time to transition
        persistence = transition_probs[current_regime]
        expected_duration = 1 / (1 - persistence) if persistence < 1 else float('inf')
        
        # Find most likely next regime
        next_regime_probs = transition_probs.copy()
        next_regime_probs[current_regime] = 0  # Exclude staying in same regime
        
        if next_regime_probs.sum() > 0:
            next_regime_probs = next_regime_probs / next_regime_probs.sum()
            most_likely_next = np.argmax(next_regime_probs)
            transition_probability = next_regime_probs[most_likely_next]
        else:
            most_likely_next = current_regime
            transition_probability = 0
        
        return {
            "expected_days_to_transition": float(expected_duration),
            "most_likely_next_regime": self._get_regime_name(most_likely_next),
            "transition_probability": float(transition_probability),
            "regime_probabilities": {
                self._get_regime_name(i): float(prob)
                for i, prob in enumerate(transition_probs)
            }
        }
    
    def _generate_regime_visualizations(self, data: pd.DataFrame,
                                      regimes: np.ndarray,
                                      features: pd.DataFrame) -> Dict[str, str]:
        """Generate regime analysis visualizations."""
        self.logger.info("Generating regime visualizations")
        
        viz_paths = {}
        
        try:
            # Regime timeline plot
            plt.figure(figsize=(14, 8))
            
            # Plot price with regime coloring
            regime_data = data.loc[features.index].copy()
            regime_colors = plt.cm.tab10(regimes)
            
            plt.subplot(3, 1, 1)
            plt.scatter(regime_data.index, regime_data['close'], 
                       c=regime_colors, s=1, alpha=0.6)
            plt.plot(regime_data.index, regime_data['close'], 
                    color='black', linewidth=0.5, alpha=0.3)
            plt.title('Market Regimes Over Time')
            plt.ylabel('Price')
            
            # Plot returns distribution by regime
            plt.subplot(3, 1, 2)
            returns = regime_data['close'].pct_change()
            for regime in np.unique(regimes):
                regime_returns = returns[regimes == regime]
                plt.hist(regime_returns, bins=50, alpha=0.5, 
                        label=self._get_regime_name(regime))
            plt.xlabel('Returns')
            plt.ylabel('Frequency')
            plt.legend()
            plt.title('Return Distributions by Regime')
            
            # Plot regime indicator
            plt.subplot(3, 1, 3)
            plt.plot(regime_data.index, regimes, drawstyle='steps-post')
            plt.ylabel('Regime')
            plt.xlabel('Date')
            plt.title('Regime Indicator')
            
            plt.tight_layout()
            plt.savefig('/tmp/regime_timeline.png')
            viz_paths["regime_timeline"] = '/tmp/regime_timeline.png'
            plt.close()
            
            # Transition matrix heatmap
            if self.transition_matrix is not None:
                plt.figure(figsize=(8, 6))
                sns.heatmap(self.transition_matrix, annot=True, fmt='.2f',
                           cmap='Blues', square=True,
                           xticklabels=[self._get_regime_name(i) for i in range(len(self.transition_matrix))],
                           yticklabels=[self._get_regime_name(i) for i in range(len(self.transition_matrix))])
                plt.title('Regime Transition Probability Matrix')
                plt.xlabel('To Regime')
                plt.ylabel('From Regime')
                plt.tight_layout()
                plt.savefig('/tmp/transition_matrix.png')
                viz_paths["transition_matrix"] = '/tmp/transition_matrix.png'
                plt.close()
            
            # Feature importance by regime
            if len(features.columns) > 0:
                plt.figure(figsize=(10, 6))
                feature_importance = pd.DataFrame()
                
                for regime in np.unique(regimes):
                    regime_mask = regimes == regime
                    regime_features = features[regime_mask]
                    feature_importance[self._get_regime_name(regime)] = regime_features.mean()
                
                # Normalize and plot top features
                feature_importance = feature_importance.div(
                    feature_importance.max(axis=1), axis=0
                )
                top_features = feature_importance.std(axis=1).nlargest(10).index
                
                feature_importance.loc[top_features].plot(kind='bar')
                plt.title('Normalized Feature Values by Regime (Top 10)')
                plt.xlabel('Features')
                plt.ylabel('Normalized Value')
                plt.legend(title='Regime')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig('/tmp/regime_features.png')
                viz_paths["regime_features"] = '/tmp/regime_features.png'
                plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to generate some visualizations: {str(e)}")
        
        return viz_paths
    
    def get_regime_recommendations(self, current_regime: str) -> Dict[str, Any]:
        """Get trading recommendations based on current regime."""
        recommendations = {
            "Bull Market": {
                "strategy": "Trend Following",
                "risk_level": "Moderate to High",
                "suggestions": [
                    "Increase long exposure",
                    "Use momentum strategies",
                    "Consider growth assets"
                ]
            },
            "Bear Market": {
                "strategy": "Risk Management",
                "risk_level": "Low",
                "suggestions": [
                    "Reduce exposure",
                    "Focus on capital preservation",
                    "Consider defensive assets"
                ]
            },
            "High Volatility": {
                "strategy": "Volatility Trading",
                "risk_level": "Low to Moderate",
                "suggestions": [
                    "Reduce position sizes",
                    "Use volatility strategies",
                    "Increase hedging"
                ]
            },
            "Low Volatility": {
                "strategy": "Carry Strategies",
                "risk_level": "Moderate",
                "suggestions": [
                    "Consider mean reversion",
                    "Increase leverage carefully",
                    "Focus on income generation"
                ]
            }
        }
        
        return recommendations.get(current_regime, {
            "strategy": "Adaptive",
            "risk_level": "Moderate",
            "suggestions": ["Monitor regime changes closely"]
        })