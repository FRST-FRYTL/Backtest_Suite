"""
Market Regime Detection using Hidden Markov Models and clustering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Literal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime types."""
    STRONG_BULL = "Strong Bull"
    BULL = "Bull"
    SIDEWAYS = "Sideways"
    BEAR = "Bear"
    STRONG_BEAR = "Strong Bear"
    
    @classmethod
    def from_int(cls, value: int) -> 'MarketRegime':
        """Convert integer to MarketRegime."""
        mapping = {
            0: cls.STRONG_BEAR,
            1: cls.BEAR,
            2: cls.SIDEWAYS,
            3: cls.BULL,
            4: cls.STRONG_BULL
        }
        return mapping.get(value, cls.SIDEWAYS)

@dataclass
class RegimeDetection:
    """Container for regime detection results."""
    current_regime: MarketRegime
    regime_probabilities: Dict[MarketRegime, float]
    transition_matrix: np.ndarray
    regime_history: pd.Series
    confidence: float

class MarketRegimeDetector:
    """
    Market regime detection using Hidden Markov Models and clustering.
    
    Identifies 5 regimes: Strong Bull, Bull, Sideways, Bear, Strong Bear.
    """
    
    def __init__(self,
                 method: Literal['hmm', 'clustering', 'ensemble'] = 'hmm',
                 n_regimes: int = 5,
                 lookback_period: int = 252,
                 hmm_n_iter: int = 100,
                 random_state: int = 42):
        """
        Initialize MarketRegimeDetector.
        
        Args:
            method: Detection method ('hmm', 'clustering', or 'ensemble')
            n_regimes: Number of market regimes
            lookback_period: Period for calculating features
            hmm_n_iter: Number of HMM iterations
            random_state: Random seed
        """
        self.method = method
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.hmm_n_iter = hmm_n_iter
        self.random_state = random_state
        
        self.hmm_model = None
        self.clustering_model = None
        self.scaler = StandardScaler()
        self.regime_mapping = {}
        self.transition_matrix = None
        self.regime_history = None
        
    def create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for regime detection.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime features
        """
        features = pd.DataFrame(index=data.index)
        
        # Returns at different scales
        for period in [1, 5, 20, 60]:
            features[f'returns_{period}'] = data['close'].pct_change(period)
        
        # Trend indicators
        for period in [20, 50, 200]:
            sma = data['close'].rolling(period).mean()
            features[f'trend_{period}'] = (data['close'] - sma) / sma
            features[f'sma_slope_{period}'] = sma.pct_change(20)
        
        # Volatility measures
        returns = data['close'].pct_change()
        for period in [20, 60]:
            features[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252)
            features[f'volatility_ratio_{period}'] = (
                features[f'volatility_{period}'] / 
                features[f'volatility_{period}'].rolling(252).mean()
            )
        
        # Market breadth (if volume available)
        features['volume_trend'] = data['volume'].rolling(20).mean().pct_change(20)
        features['volume_volatility'] = data['volume'].pct_change().rolling(20).std()
        
        # Momentum indicators
        features['rsi'] = self._calculate_rsi(data['close'], 14)
        features['macd'] = self._calculate_macd(data['close'])
        
        # Market microstructure
        features['high_low_spread'] = (data['high'] - data['low']) / data['close']
        features['close_location'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Volatility regime indicators
        features['realized_vol'] = returns.rolling(20).std() * np.sqrt(252)
        features['vol_of_vol'] = features['realized_vol'].rolling(60).std()
        
        # Correlation with market (if available)
        # This would require market index data
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD histogram."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd - signal
    
    def _fit_hmm(self, features: np.ndarray) -> hmm.GaussianHMM:
        """
        Fit Hidden Markov Model.
        
        Args:
            features: Feature array
            
        Returns:
            Fitted HMM model
        """
        model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.hmm_n_iter,
            random_state=self.random_state
        )
        
        model.fit(features)
        return model
    
    def _fit_clustering(self, features: np.ndarray) -> GaussianMixture:
        """
        Fit Gaussian Mixture Model for clustering.
        
        Args:
            features: Feature array
            
        Returns:
            Fitted GMM model
        """
        model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            n_init=10,
            random_state=self.random_state
        )
        
        model.fit(features)
        return model
    
    def _map_regimes_to_market_states(self, 
                                     features: pd.DataFrame,
                                     labels: np.ndarray) -> Dict[int, MarketRegime]:
        """
        Map numerical regime labels to market states.
        
        Args:
            features: Feature DataFrame
            labels: Regime labels
            
        Returns:
            Mapping from label to MarketRegime
        """
        # Calculate average returns for each regime
        regime_returns = {}
        for regime in range(self.n_regimes):
            mask = labels == regime
            if mask.any():
                avg_return = features.loc[mask, 'returns_20'].mean()
                avg_volatility = features.loc[mask, 'volatility_20'].mean()
                regime_returns[regime] = (avg_return, avg_volatility)
        
        # Sort regimes by average returns
        sorted_regimes = sorted(regime_returns.items(), key=lambda x: x[1][0])
        
        # Map to market states
        mapping = {}
        for i, (regime, _) in enumerate(sorted_regimes):
            mapping[regime] = MarketRegime.from_int(i)
        
        return mapping
    
    def _calculate_transition_matrix(self, regime_sequence: np.ndarray) -> np.ndarray:
        """
        Calculate regime transition probability matrix.
        
        Args:
            regime_sequence: Sequence of regime labels
            
        Returns:
            Transition probability matrix
        """
        n_regimes = self.n_regimes
        trans_matrix = np.zeros((n_regimes, n_regimes))
        
        # Count transitions
        for i in range(len(regime_sequence) - 1):
            from_regime = regime_sequence[i]
            to_regime = regime_sequence[i + 1]
            trans_matrix[from_regime, to_regime] += 1
        
        # Normalize to probabilities
        for i in range(n_regimes):
            row_sum = trans_matrix[i].sum()
            if row_sum > 0:
                trans_matrix[i] /= row_sum
            else:
                # If no transitions from this state, assume equal probability
                trans_matrix[i] = 1 / n_regimes
        
        return trans_matrix
    
    def fit(self, data: pd.DataFrame) -> 'MarketRegimeDetector':
        """
        Fit the regime detection model.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Self for chaining
        """
        # Create features
        features_df = self.create_regime_features(data)
        
        # Remove NaN values
        valid_mask = ~features_df.isna().any(axis=1)
        features_df = features_df[valid_mask]
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_df)
        
        # Fit model based on method
        if self.method == 'hmm':
            self.hmm_model = self._fit_hmm(features_scaled)
            labels = self.hmm_model.predict(features_scaled)
            
        elif self.method == 'clustering':
            self.clustering_model = self._fit_clustering(features_scaled)
            labels = self.clustering_model.predict(features_scaled)
            
        elif self.method == 'ensemble':
            # Fit both models and combine predictions
            self.hmm_model = self._fit_hmm(features_scaled)
            self.clustering_model = self._fit_clustering(features_scaled)
            
            hmm_labels = self.hmm_model.predict(features_scaled)
            gmm_labels = self.clustering_model.predict(features_scaled)
            
            # Combine predictions (simple voting)
            labels = np.round((hmm_labels + gmm_labels) / 2).astype(int)
            labels = np.clip(labels, 0, self.n_regimes - 1)
        
        # Map regimes to market states
        self.regime_mapping = self._map_regimes_to_market_states(features_df, labels)
        
        # Calculate transition matrix
        self.transition_matrix = self._calculate_transition_matrix(labels)
        
        # Store regime history
        self.regime_history = pd.Series(
            [self.regime_mapping[label] for label in labels],
            index=features_df.index
        )
        
        return self
    
    def predict(self, data: pd.DataFrame) -> RegimeDetection:
        """
        Predict current market regime.
        
        Args:
            data: DataFrame with recent OHLCV data
            
        Returns:
            RegimeDetection object
        """
        if self.hmm_model is None and self.clustering_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create features
        features_df = self.create_regime_features(data)
        
        # Get last valid row
        valid_mask = ~features_df.isna().any(axis=1)
        last_features = features_df[valid_mask].iloc[-1:]
        
        if last_features.empty:
            raise ValueError("No valid features found")
        
        # Scale features
        features_scaled = self.scaler.transform(last_features)
        
        # Get predictions and probabilities
        if self.method == 'hmm':
            label = self.hmm_model.predict(features_scaled)[0]
            probs = self.hmm_model.predict_proba(features_scaled)[0]
            
        elif self.method == 'clustering':
            label = self.clustering_model.predict(features_scaled)[0]
            probs = self.clustering_model.predict_proba(features_scaled)[0]
            
        elif self.method == 'ensemble':
            # Combine predictions from both models
            hmm_label = self.hmm_model.predict(features_scaled)[0]
            hmm_probs = self.hmm_model.predict_proba(features_scaled)[0]
            
            gmm_label = self.clustering_model.predict(features_scaled)[0]
            gmm_probs = self.clustering_model.predict_proba(features_scaled)[0]
            
            # Average probabilities
            probs = (hmm_probs + gmm_probs) / 2
            label = np.argmax(probs)
        
        # Map to market regime
        current_regime = self.regime_mapping[label]
        
        # Create probability distribution
        regime_probabilities = {}
        for i, prob in enumerate(probs):
            regime = self.regime_mapping.get(i, MarketRegime.SIDEWAYS)
            regime_probabilities[regime] = float(prob)
        
        # Calculate confidence (max probability)
        confidence = float(np.max(probs))
        
        return RegimeDetection(
            current_regime=current_regime,
            regime_probabilities=regime_probabilities,
            transition_matrix=self.transition_matrix,
            regime_history=self.regime_history,
            confidence=confidence
        )
    
    def get_regime_statistics(self) -> Dict[MarketRegime, Dict[str, float]]:
        """
        Get statistics for each regime.
        
        Returns:
            Dictionary of regime statistics
        """
        if self.regime_history is None:
            return {}
        
        stats = {}
        
        for regime in MarketRegime:
            regime_mask = self.regime_history == regime
            
            if not regime_mask.any():
                continue
            
            # Calculate duration statistics
            regime_changes = regime_mask.astype(int).diff() != 0
            regime_starts = regime_changes & regime_mask
            regime_lengths = []
            
            current_length = 0
            for is_regime in regime_mask:
                if is_regime:
                    current_length += 1
                elif current_length > 0:
                    regime_lengths.append(current_length)
                    current_length = 0
            
            if current_length > 0:
                regime_lengths.append(current_length)
            
            stats[regime] = {
                'frequency': float(regime_mask.sum() / len(self.regime_history)),
                'avg_duration': float(np.mean(regime_lengths)) if regime_lengths else 0,
                'max_duration': float(np.max(regime_lengths)) if regime_lengths else 0,
                'occurrences': len(regime_lengths)
            }
        
        return stats
    
    def plot_regime_history(self, prices: pd.Series = None) -> None:
        """
        Plot regime history with optional price overlay.
        
        Args:
            prices: Optional price series to overlay
        """
        if self.regime_history is None:
            logger.warning("No regime history available")
            return
        
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Define colors for each regime
        regime_colors = {
            MarketRegime.STRONG_BULL: 'darkgreen',
            MarketRegime.BULL: 'green',
            MarketRegime.SIDEWAYS: 'gray',
            MarketRegime.BEAR: 'red',
            MarketRegime.STRONG_BEAR: 'darkred'
        }
        
        # Plot regimes
        for regime, color in regime_colors.items():
            mask = self.regime_history == regime
            ax1.fill_between(
                self.regime_history.index,
                0, 1,
                where=mask,
                alpha=0.3,
                color=color,
                label=regime.value
            )
        
        ax1.set_ylabel('Market Regime')
        ax1.set_ylim(0, 1)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot prices if provided
        if prices is not None and not prices.empty:
            ax2.plot(prices.index, prices.values, 'b-', linewidth=1)
            ax2.set_ylabel('Price')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            
            # Add regime coloring to price plot
            for regime, color in regime_colors.items():
                mask = self.regime_history == regime
                if mask.any():
                    for idx in self.regime_history[mask].index:
                        if idx in prices.index:
                            ax2.axvspan(
                                idx, idx + pd.Timedelta(days=1),
                                alpha=0.1, color=color
                            )
        
        plt.suptitle('Market Regime History')
        plt.tight_layout()
        plt.show()