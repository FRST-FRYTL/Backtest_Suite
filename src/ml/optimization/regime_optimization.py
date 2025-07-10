"""
Market Regime Optimization (Loop 3)

Optimizes regime detection, classification, and regime-specific strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import optuna
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from hmmlearn import hmm
import talib
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class RegimeOptimization:
    """
    Optimizes market regime detection and regime-specific trading strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize regime optimization
        
        Args:
            config: Regime optimization configuration
        """
        self.config = config
        self.regime_cache = {}
        
    def get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Get hyperparameters from Optuna trial for regime optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of regime detection parameters
        """
        # Choose regime detection method
        detection_method = trial.suggest_categorical(
            'regime_detection_method',
            ['hmm', 'clustering', 'volatility', 'trend', 'ensemble']
        )
        
        params = {
            'regime_detection': {
                'method': detection_method,
                'n_regimes': trial.suggest_int('n_regimes', 2, 5),
                'lookback_period': trial.suggest_int('lookback_period', 20, 100),
                'update_frequency': trial.suggest_categorical('update_frequency', ['daily', 'weekly', 'monthly']),
                'min_regime_length': trial.suggest_int('min_regime_length', 5, 20),
            }
        }
        
        # Method-specific parameters
        if detection_method == 'hmm':
            params['regime_detection']['hmm'] = {
                'covariance_type': trial.suggest_categorical('hmm_covariance_type', ['full', 'diag', 'spherical']),
                'n_iter': trial.suggest_int('hmm_n_iter', 50, 200),
                'features': trial.suggest_categorical(
                    'hmm_features',
                    ['returns', 'returns_volatility', 'price_volume', 'technical', 'all']
                ),
            }
            
        elif detection_method == 'clustering':
            clustering_algo = trial.suggest_categorical(
                'clustering_algorithm',
                ['kmeans', 'gaussian_mixture', 'dbscan', 'hierarchical']
            )
            params['regime_detection']['clustering'] = {
                'algorithm': clustering_algo,
                'features': trial.suggest_categorical(
                    'clustering_features',
                    ['volatility', 'trend', 'volume', 'technical', 'all']
                ),
                'scaling': trial.suggest_categorical('clustering_scaling', [True, False]),
            }
            
            if clustering_algo == 'dbscan':
                params['regime_detection']['clustering']['eps'] = trial.suggest_float('dbscan_eps', 0.1, 2.0)
                params['regime_detection']['clustering']['min_samples'] = trial.suggest_int('dbscan_min_samples', 5, 20)
                
        elif detection_method == 'volatility':
            params['regime_detection']['volatility'] = {
                'volatility_measure': trial.suggest_categorical(
                    'volatility_measure',
                    ['std', 'garch', 'realized', 'range']
                ),
                'threshold_method': trial.suggest_categorical(
                    'volatility_threshold',
                    ['percentile', 'std', 'adaptive']
                ),
                'low_vol_threshold': trial.suggest_float('low_vol_threshold', 0.1, 0.4),
                'high_vol_threshold': trial.suggest_float('high_vol_threshold', 0.6, 0.9),
            }
            
        elif detection_method == 'trend':
            params['regime_detection']['trend'] = {
                'trend_measure': trial.suggest_categorical(
                    'trend_measure',
                    ['sma_cross', 'adx', 'linear_regression', 'price_position']
                ),
                'trend_strength_threshold': trial.suggest_float('trend_strength_threshold', 0.3, 0.7),
                'consolidation_threshold': trial.suggest_float('consolidation_threshold', 0.1, 0.3),
            }
            
        elif detection_method == 'ensemble':
            params['regime_detection']['ensemble'] = {
                'methods': trial.suggest_categorical(
                    'ensemble_methods',
                    [
                        ['hmm', 'volatility'],
                        ['clustering', 'trend'],
                        ['volatility', 'trend'],
                        ['hmm', 'clustering', 'volatility'],
                        ['all']
                    ]
                ),
                'voting': trial.suggest_categorical('ensemble_voting', ['majority', 'weighted', 'stacked']),
            }
        
        # Regime-specific strategy parameters
        params['regime_strategies'] = {}
        for i in range(params['regime_detection']['n_regimes']):
            params['regime_strategies'][f'regime_{i}'] = {
                'strategy_type': trial.suggest_categorical(
                    f'regime_{i}_strategy',
                    ['trend_following', 'mean_reversion', 'breakout', 'momentum', 'neutral']
                ),
                'position_size': trial.suggest_float(f'regime_{i}_position_size', 0.1, 1.0),
                'stop_loss': trial.suggest_float(f'regime_{i}_stop_loss', 0.01, 0.05),
                'take_profit': trial.suggest_float(f'regime_{i}_take_profit', 0.02, 0.10),
                'entry_threshold': trial.suggest_float(f'regime_{i}_entry_threshold', 0.5, 0.9),
            }
        
        # Regime transition handling
        params['regime_transitions'] = {
            'transition_delay': trial.suggest_int('transition_delay', 0, 5),
            'confirmation_period': trial.suggest_int('confirmation_period', 1, 5),
            'position_adjustment': trial.suggest_categorical(
                'position_adjustment',
                ['immediate', 'gradual', 'wait_for_exit']
            ),
        }
        
        return params
    
    def evaluate(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """
        Evaluate regime detection and regime-specific strategies
        
        Args:
            data: Market data
            params: Complete parameter set including regime params
            
        Returns:
            Performance metric (higher is better)
        """
        try:
            # Detect regimes
            regimes = self._detect_regimes(data, params.get('regime_detection', {}))
            
            if regimes is None or len(np.unique(regimes)) < 2:
                return -np.inf
            
            # Evaluate regime quality
            regime_quality = self._evaluate_regime_quality(data, regimes, params.get('regime_detection', {}))
            
            # Evaluate regime-specific strategies
            strategy_performance = self._evaluate_regime_strategies(
                data, regimes, 
                params.get('regime_strategies', {}),
                params.get('regime_transitions', {})
            )
            
            # Combine scores
            total_score = 0.4 * regime_quality + 0.6 * strategy_performance
            
            return total_score
            
        except Exception as e:
            logger.error(f"Error in regime evaluation: {str(e)}")
            return -np.inf
    
    def _detect_regimes(self, data: pd.DataFrame, detection_params: Dict[str, Any]) -> Optional[np.ndarray]:
        """Detect market regimes based on parameters"""
        method = detection_params.get('method', 'volatility')
        
        if method == 'hmm':
            return self._detect_hmm_regimes(data, detection_params)
        elif method == 'clustering':
            return self._detect_clustering_regimes(data, detection_params)
        elif method == 'volatility':
            return self._detect_volatility_regimes(data, detection_params)
        elif method == 'trend':
            return self._detect_trend_regimes(data, detection_params)
        elif method == 'ensemble':
            return self._detect_ensemble_regimes(data, detection_params)
        else:
            return None
    
    def _detect_hmm_regimes(self, data: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        """Detect regimes using Hidden Markov Model"""
        hmm_params = params.get('hmm', {})
        n_regimes = params.get('n_regimes', 3)
        
        # Prepare features
        features = self._prepare_hmm_features(data, hmm_params.get('features', 'returns'))
        
        if features.shape[0] < 100:
            return np.zeros(len(data))
        
        # Train HMM
        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type=hmm_params.get('covariance_type', 'diag'),
            n_iter=hmm_params.get('n_iter', 100),
            random_state=42
        )
        
        try:
            model.fit(features)
            regimes = model.predict(features)
            
            # Extend regimes to match data length
            full_regimes = np.zeros(len(data))
            full_regimes[-len(regimes):] = regimes
            
            return full_regimes
            
        except Exception as e:
            logger.error(f"HMM fitting error: {str(e)}")
            return np.zeros(len(data))
    
    def _prepare_hmm_features(self, data: pd.DataFrame, feature_type: str) -> np.ndarray:
        """Prepare features for HMM"""
        features = []
        
        if feature_type in ['returns', 'all']:
            returns = data['close'].pct_change().fillna(0)
            features.append(returns.values.reshape(-1, 1))
        
        if feature_type in ['returns_volatility', 'all']:
            returns = data['close'].pct_change().fillna(0)
            volatility = returns.rolling(20).std().fillna(0)
            features.append(volatility.values.reshape(-1, 1))
        
        if feature_type in ['price_volume', 'all'] and 'volume' in data.columns:
            volume_ratio = (data['volume'] / data['volume'].rolling(20).mean()).fillna(1)
            features.append(volume_ratio.values.reshape(-1, 1))
            
            price_range = ((data['high'] - data['low']) / data['close']).fillna(0)
            features.append(price_range.values.reshape(-1, 1))
        
        if feature_type in ['technical', 'all']:
            # RSI
            rsi = talib.RSI(data['close'].values, timeperiod=14)
            features.append(rsi.reshape(-1, 1))
            
            # ADX
            adx = talib.ADX(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
            features.append(adx.reshape(-1, 1))
        
        if not features:
            # Default to returns
            returns = data['close'].pct_change().fillna(0)
            features.append(returns.values.reshape(-1, 1))
        
        # Combine features and remove NaN rows
        feature_matrix = np.hstack(features)
        mask = ~np.any(np.isnan(feature_matrix), axis=1)
        
        return feature_matrix[mask]
    
    def _detect_clustering_regimes(self, data: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        """Detect regimes using clustering algorithms"""
        clustering_params = params.get('clustering', {})
        n_regimes = params.get('n_regimes', 3)
        algorithm = clustering_params.get('algorithm', 'kmeans')
        
        # Prepare features
        features = self._prepare_clustering_features(data, clustering_params.get('features', 'all'))
        
        if features.shape[0] < 100:
            return np.zeros(len(data))
        
        # Scale features if requested
        if clustering_params.get('scaling', True):
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        
        try:
            if algorithm == 'kmeans':
                model = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
                regimes = model.fit_predict(features)
                
            elif algorithm == 'gaussian_mixture':
                model = GaussianMixture(n_components=n_regimes, random_state=42)
                regimes = model.fit_predict(features)
                
            elif algorithm == 'dbscan':
                eps = clustering_params.get('eps', 0.5)
                min_samples = clustering_params.get('min_samples', 10)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                regimes = model.fit_predict(features)
                
                # Map DBSCAN labels to regime numbers
                unique_labels = np.unique(regimes)
                regime_map = {label: i % n_regimes for i, label in enumerate(unique_labels)}
                regimes = np.array([regime_map[label] for label in regimes])
                
            elif algorithm == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n_regimes)
                regimes = model.fit_predict(features)
                
            else:
                return np.zeros(len(data))
            
            # Extend regimes to match data length
            full_regimes = np.zeros(len(data))
            full_regimes[-len(regimes):] = regimes
            
            return full_regimes
            
        except Exception as e:
            logger.error(f"Clustering error: {str(e)}")
            return np.zeros(len(data))
    
    def _prepare_clustering_features(self, data: pd.DataFrame, feature_type: str) -> np.ndarray:
        """Prepare features for clustering"""
        features = []
        
        if feature_type in ['volatility', 'all']:
            returns = data['close'].pct_change()
            # Rolling volatility
            for window in [5, 10, 20]:
                vol = returns.rolling(window).std()
                features.append(vol.values.reshape(-1, 1))
            
            # Realized volatility
            high_low_vol = np.log(data['high'] / data['low'])
            features.append(high_low_vol.values.reshape(-1, 1))
        
        if feature_type in ['trend', 'all']:
            # Moving average ratios
            for period in [10, 20, 50]:
                ma = data['close'].rolling(period).mean()
                ma_ratio = data['close'] / ma
                features.append(ma_ratio.values.reshape(-1, 1))
            
            # Price momentum
            for period in [5, 10, 20]:
                momentum = data['close'] / data['close'].shift(period) - 1
                features.append(momentum.values.reshape(-1, 1))
        
        if feature_type in ['volume', 'all'] and 'volume' in data.columns:
            # Volume indicators
            volume_ma = data['volume'].rolling(20).mean()
            volume_ratio = data['volume'] / volume_ma
            features.append(volume_ratio.values.reshape(-1, 1))
            
            # On-balance volume change
            obv = talib.OBV(data['close'].values, data['volume'].values)
            obv_change = pd.Series(obv).pct_change()
            features.append(obv_change.values.reshape(-1, 1))
        
        if feature_type in ['technical', 'all']:
            # Technical indicators
            rsi = talib.RSI(data['close'].values, timeperiod=14)
            features.append(rsi.reshape(-1, 1))
            
            # ADX for trend strength
            adx = talib.ADX(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
            features.append(adx.reshape(-1, 1))
            
            # ATR for volatility
            atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)
            features.append(atr.reshape(-1, 1))
        
        if not features:
            # Default features
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std()
            features.append(returns.values.reshape(-1, 1))
            features.append(volatility.values.reshape(-1, 1))
        
        # Combine features and handle NaN
        feature_matrix = np.hstack(features)
        
        # Forward fill NaN values
        df_features = pd.DataFrame(feature_matrix)
        df_features = df_features.fillna(method='ffill').fillna(0)
        
        return df_features.values
    
    def _detect_volatility_regimes(self, data: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        """Detect regimes based on volatility levels"""
        vol_params = params.get('volatility', {})
        n_regimes = params.get('n_regimes', 3)
        lookback = params.get('lookback_period', 20)
        
        # Calculate volatility
        returns = data['close'].pct_change()
        
        volatility_measure = vol_params.get('volatility_measure', 'std')
        if volatility_measure == 'std':
            volatility = returns.rolling(lookback).std()
        elif volatility_measure == 'realized':
            log_returns = np.log(data['close'] / data['close'].shift(1))
            volatility = np.sqrt((log_returns ** 2).rolling(lookback).sum())
        elif volatility_measure == 'range':
            volatility = (data['high'] - data['low']) / data['close']
            volatility = volatility.rolling(lookback).mean()
        else:
            volatility = returns.rolling(lookback).std()
        
        # Determine thresholds
        threshold_method = vol_params.get('threshold_method', 'percentile')
        
        if threshold_method == 'percentile':
            low_threshold = vol_params.get('low_vol_threshold', 0.33)
            high_threshold = vol_params.get('high_vol_threshold', 0.67)
            
            low_val = volatility.quantile(low_threshold)
            high_val = volatility.quantile(high_threshold)
            
        elif threshold_method == 'std':
            mean_vol = volatility.mean()
            std_vol = volatility.std()
            
            low_val = mean_vol - std_vol
            high_val = mean_vol + std_vol
            
        else:  # adaptive
            # Use rolling quantiles
            window = 252  # 1 year
            low_threshold = vol_params.get('low_vol_threshold', 0.33)
            high_threshold = vol_params.get('high_vol_threshold', 0.67)
            
            low_val = volatility.rolling(window).quantile(low_threshold)
            high_val = volatility.rolling(window).quantile(high_threshold)
        
        # Assign regimes
        regimes = np.zeros(len(data))
        
        if n_regimes == 2:
            regimes[volatility >= volatility.median()] = 1
        elif n_regimes == 3:
            regimes[volatility <= low_val] = 0  # Low volatility
            regimes[(volatility > low_val) & (volatility < high_val)] = 1  # Normal
            regimes[volatility >= high_val] = 2  # High volatility
        else:
            # Use quantiles for more regimes
            for i in range(n_regimes):
                lower = volatility.quantile(i / n_regimes)
                upper = volatility.quantile((i + 1) / n_regimes)
                mask = (volatility >= lower) & (volatility < upper)
                regimes[mask] = i
        
        return regimes
    
    def _detect_trend_regimes(self, data: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        """Detect regimes based on trend characteristics"""
        trend_params = params.get('trend', {})
        n_regimes = params.get('n_regimes', 3)
        lookback = params.get('lookback_period', 20)
        
        trend_measure = trend_params.get('trend_measure', 'sma_cross')
        
        if trend_measure == 'sma_cross':
            # Use multiple SMA crossovers
            sma_short = data['close'].rolling(lookback).mean()
            sma_long = data['close'].rolling(lookback * 3).mean()
            
            regimes = np.zeros(len(data))
            regimes[data['close'] > sma_long] = 1  # Uptrend
            regimes[data['close'] < sma_long] = 0  # Downtrend
            
            if n_regimes >= 3:
                # Add consolidation regime
                threshold = trend_params.get('consolidation_threshold', 0.02)
                distance = np.abs(data['close'] - sma_long) / sma_long
                regimes[distance < threshold] = 2  # Consolidation
                
        elif trend_measure == 'adx':
            # Use ADX for trend strength
            adx = talib.ADX(data['high'].values, data['low'].values, 
                           data['close'].values, timeperiod=lookback)
            plus_di = talib.PLUS_DI(data['high'].values, data['low'].values,
                                   data['close'].values, timeperiod=lookback)
            minus_di = talib.MINUS_DI(data['high'].values, data['low'].values,
                                     data['close'].values, timeperiod=lookback)
            
            threshold = trend_params.get('trend_strength_threshold', 25)
            
            regimes = np.zeros(len(data))
            # Strong uptrend
            regimes[(adx > threshold) & (plus_di > minus_di)] = 0
            # Strong downtrend
            regimes[(adx > threshold) & (plus_di < minus_di)] = 1
            # No trend/consolidation
            regimes[adx <= threshold] = 2
            
        elif trend_measure == 'linear_regression':
            # Use rolling linear regression slope
            slopes = []
            for i in range(lookback, len(data)):
                y = data['close'].iloc[i-lookback:i].values
                x = np.arange(lookback)
                slope, _ = np.polyfit(x, y, 1)
                slopes.append(slope)
            
            slopes = np.array(slopes)
            regimes = np.zeros(len(data))
            
            # Normalize slopes
            price_scale = data['close'].rolling(lookback).mean()
            normalized_slopes = slopes / price_scale.iloc[lookback:].values
            
            if n_regimes == 2:
                regimes[lookback:][normalized_slopes > 0] = 1
            else:
                threshold = trend_params.get('trend_strength_threshold', 0.001)
                regimes[lookback:][normalized_slopes > threshold] = 0  # Uptrend
                regimes[lookback:][normalized_slopes < -threshold] = 1  # Downtrend
                regimes[lookback:][np.abs(normalized_slopes) <= threshold] = 2  # Sideways
                
        else:  # price_position
            # Position relative to recent high/low
            high = data['high'].rolling(lookback).max()
            low = data['low'].rolling(lookback).min()
            position = (data['close'] - low) / (high - low)
            
            if n_regimes == 2:
                regimes[position > 0.5] = 1
            else:
                regimes[position > 0.7] = 0  # Near highs (uptrend)
                regimes[position < 0.3] = 1  # Near lows (downtrend)
                regimes[(position >= 0.3) & (position <= 0.7)] = 2  # Middle (consolidation)
        
        return regimes
    
    def _detect_ensemble_regimes(self, data: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        """Detect regimes using ensemble of methods"""
        ensemble_params = params.get('ensemble', {})
        methods = ensemble_params.get('methods', ['volatility', 'trend'])
        
        if methods == ['all']:
            methods = ['hmm', 'clustering', 'volatility', 'trend']
        
        all_regimes = []
        
        for method in methods:
            if method == 'hmm':
                regimes = self._detect_hmm_regimes(data, params)
            elif method == 'clustering':
                regimes = self._detect_clustering_regimes(data, params)
            elif method == 'volatility':
                regimes = self._detect_volatility_regimes(data, params)
            elif method == 'trend':
                regimes = self._detect_trend_regimes(data, params)
            else:
                continue
            
            all_regimes.append(regimes)
        
        if not all_regimes:
            return np.zeros(len(data))
        
        # Combine regimes
        voting = ensemble_params.get('voting', 'majority')
        
        if voting == 'majority':
            # Simple majority voting
            from scipy.stats import mode
            ensemble_regimes = mode(np.array(all_regimes), axis=0)[0].flatten()
            
        elif voting == 'weighted':
            # Weight by method performance (simplified: equal weights)
            weights = np.ones(len(all_regimes)) / len(all_regimes)
            ensemble_regimes = np.zeros(len(data))
            
            for i in range(params.get('n_regimes', 3)):
                regime_votes = np.zeros(len(data))
                for j, regimes in enumerate(all_regimes):
                    regime_votes += weights[j] * (regimes == i)
                ensemble_regimes[regime_votes == regime_votes.max()] = i
                
        else:  # stacked
            # Use all regime predictions as features for meta-classifier
            # Simplified: use most frequent combination
            regime_combinations = np.column_stack(all_regimes)
            unique_combinations, inverse = np.unique(regime_combinations, axis=0, return_inverse=True)
            ensemble_regimes = inverse % params.get('n_regimes', 3)
        
        return ensemble_regimes
    
    def _evaluate_regime_quality(self, data: pd.DataFrame, regimes: np.ndarray, 
                               detection_params: Dict[str, Any]) -> float:
        """Evaluate quality of detected regimes"""
        scores = []
        
        # 1. Regime stability (regimes should persist)
        min_length = detection_params.get('min_regime_length', 5)
        regime_lengths = []
        current_regime = regimes[0]
        current_length = 1
        
        for regime in regimes[1:]:
            if regime == current_regime:
                current_length += 1
            else:
                regime_lengths.append(current_length)
                current_regime = regime
                current_length = 1
        regime_lengths.append(current_length)
        
        stability_score = sum(l >= min_length for l in regime_lengths) / len(regime_lengths)
        scores.append(stability_score)
        
        # 2. Regime distinction (regimes should have different characteristics)
        returns = data['close'].pct_change()
        regime_stats = []
        
        for regime in range(int(regimes.max()) + 1):
            mask = regimes == regime
            if mask.sum() > 10:
                regime_returns = returns[mask]
                regime_stats.append({
                    'mean': regime_returns.mean(),
                    'std': regime_returns.std(),
                    'skew': regime_returns.skew(),
                    'count': mask.sum()
                })
        
        if len(regime_stats) > 1:
            # Calculate distinction score based on statistical differences
            distinction_scores = []
            for i in range(len(regime_stats)):
                for j in range(i + 1, len(regime_stats)):
                    # Normalized difference in means
                    mean_diff = abs(regime_stats[i]['mean'] - regime_stats[j]['mean'])
                    std_avg = (regime_stats[i]['std'] + regime_stats[j]['std']) / 2
                    if std_avg > 0:
                        distinction_scores.append(mean_diff / std_avg)
                    
                    # Difference in volatility
                    vol_diff = abs(regime_stats[i]['std'] - regime_stats[j]['std'])
                    vol_avg = (regime_stats[i]['std'] + regime_stats[j]['std']) / 2
                    if vol_avg > 0:
                        distinction_scores.append(vol_diff / vol_avg)
            
            distinction_score = np.mean(distinction_scores) if distinction_scores else 0
            scores.append(min(distinction_score, 1.0))
        else:
            scores.append(0)
        
        # 3. Regime balance (avoid having one dominant regime)
        regime_counts = [stats['count'] for stats in regime_stats]
        if regime_counts:
            total_count = sum(regime_counts)
            expected_count = total_count / len(regime_counts)
            imbalance = sum(abs(count - expected_count) for count in regime_counts) / total_count
            balance_score = 1 - imbalance
            scores.append(balance_score)
        else:
            scores.append(0)
        
        # 4. Regime predictability (transitions should follow patterns)
        transition_matrix = np.zeros((int(regimes.max()) + 1, int(regimes.max()) + 1))
        for i in range(len(regimes) - 1):
            transition_matrix[int(regimes[i]), int(regimes[i + 1])] += 1
        
        # Normalize rows
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / (row_sums[:, np.newaxis] + 1e-8)
        
        # Predictability is high when transitions are concentrated
        predictability_scores = []
        for row in transition_matrix:
            if row.sum() > 0:
                # Entropy of transition probabilities
                entropy = -np.sum(row * np.log(row + 1e-8))
                max_entropy = np.log(len(row))
                predictability = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
                predictability_scores.append(predictability)
        
        if predictability_scores:
            scores.append(np.mean(predictability_scores))
        else:
            scores.append(0)
        
        return np.mean(scores)
    
    def _evaluate_regime_strategies(self, data: pd.DataFrame, regimes: np.ndarray,
                                  strategy_params: Dict[str, Any],
                                  transition_params: Dict[str, Any]) -> float:
        """Evaluate performance of regime-specific strategies"""
        returns = data['close'].pct_change()
        strategy_returns = pd.Series(0, index=data.index)
        positions = pd.Series(0, index=data.index)
        
        # Apply regime-specific strategies
        for regime in range(int(regimes.max()) + 1):
            regime_mask = regimes == regime
            regime_strategy = strategy_params.get(f'regime_{regime}', {})
            
            strategy_type = regime_strategy.get('strategy_type', 'neutral')
            position_size = regime_strategy.get('position_size', 0.5)
            
            if strategy_type == 'trend_following':
                # Simple trend following
                sma_short = data['close'].rolling(10).mean()
                sma_long = data['close'].rolling(30).mean()
                signal = (sma_short > sma_long).astype(float) * 2 - 1
                positions[regime_mask] = signal[regime_mask] * position_size
                
            elif strategy_type == 'mean_reversion':
                # Mean reversion using RSI
                rsi = talib.RSI(data['close'].values, timeperiod=14)
                signal = np.zeros_like(rsi)
                signal[rsi < 30] = 1  # Oversold - buy
                signal[rsi > 70] = -1  # Overbought - sell
                positions[regime_mask] = signal[regime_mask] * position_size
                
            elif strategy_type == 'breakout':
                # Channel breakout
                high_20 = data['high'].rolling(20).max()
                low_20 = data['low'].rolling(20).min()
                signal = np.zeros(len(data))
                signal[data['close'] > high_20.shift(1)] = 1
                signal[data['close'] < low_20.shift(1)] = -1
                positions[regime_mask] = signal[regime_mask] * position_size
                
            elif strategy_type == 'momentum':
                # Simple momentum
                momentum = data['close'] / data['close'].shift(10) - 1
                signal = (momentum > 0).astype(float) * 2 - 1
                positions[regime_mask] = signal[regime_mask] * position_size
                
            else:  # neutral
                positions[regime_mask] = 0
        
        # Handle regime transitions
        transition_delay = transition_params.get('transition_delay', 1)
        if transition_delay > 0:
            # Delay position changes at regime transitions
            regime_changes = np.diff(regimes) != 0
            change_indices = np.where(regime_changes)[0] + 1
            
            for idx in change_indices:
                if idx + transition_delay < len(positions):
                    # Keep previous position during transition
                    positions.iloc[idx:idx + transition_delay] = positions.iloc[idx - 1]
        
        # Calculate strategy returns
        strategy_returns = positions.shift(1) * returns
        strategy_returns = strategy_returns.fillna(0)
        
        # Apply stop loss and take profit
        cumulative_returns = (1 + strategy_returns).cumprod()
        drawdown = 1 - cumulative_returns / cumulative_returns.cummax()
        
        # Calculate performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        
        if len(strategy_returns) > 0:
            sharpe_ratio = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252)
            max_drawdown = drawdown.max()
            
            # Win rate
            winning_days = (strategy_returns > 0).sum()
            total_days = (strategy_returns != 0).sum()
            win_rate = winning_days / total_days if total_days > 0 else 0
            
            # Profit factor
            gross_profits = strategy_returns[strategy_returns > 0].sum()
            gross_losses = abs(strategy_returns[strategy_returns < 0].sum())
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else 2.0
            
            # Combine metrics
            score = (
                0.3 * (sharpe_ratio / 2) +  # Normalized Sharpe
                0.2 * (1 - max_drawdown) +  # Drawdown penalty
                0.2 * win_rate +
                0.2 * min(profit_factor / 2, 1) +  # Normalized profit factor
                0.1 * (1 / (1 + abs(total_return)))  # Smoothness bonus
            )
            
            return max(0, min(score, 1))
        else:
            return 0
    
    def get_regime_labels(self, data: pd.DataFrame, best_params: Dict[str, Any]) -> pd.Series:
        """
        Get regime labels using optimized parameters
        
        Args:
            data: Market data
            best_params: Optimized parameters
            
        Returns:
            Series of regime labels
        """
        regimes = self._detect_regimes(data, best_params.get('regime_detection', {}))
        return pd.Series(regimes, index=data.index, name='regime')