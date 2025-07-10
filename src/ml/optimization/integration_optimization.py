"""
Integration and Ensemble Optimization (Loop 5)

Optimizes the integration of all components and ensemble methods for final strategy.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import optuna
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
import joblib
import logging

logger = logging.getLogger(__name__)


class IntegrationOptimization:
    """
    Optimizes integration of all ML components and ensemble strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize integration optimization
        
        Args:
            config: Integration optimization configuration
        """
        self.config = config
        self.ensemble_cache = {}
        
    def get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Get hyperparameters from Optuna trial for integration optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of integration parameters
        """
        params = {
            'integration': {
                # Ensemble method
                'ensemble_method': trial.suggest_categorical(
                    'ensemble_method',
                    ['voting', 'stacking', 'blending', 'dynamic', 'hierarchical']
                ),
                
                # Model combination
                'model_combination': {
                    'use_base_models': trial.suggest_categorical('use_base_models', [True, False]),
                    'use_regime_models': trial.suggest_categorical('use_regime_models', [True, False]),
                    'use_meta_features': trial.suggest_categorical('use_meta_features', [True, False]),
                    'model_weights': trial.suggest_categorical(
                        'model_weights',
                        ['equal', 'performance', 'confidence', 'adaptive']
                    ),
                },
                
                # Signal generation
                'signal_generation': {
                    'signal_aggregation': trial.suggest_categorical(
                        'signal_aggregation',
                        ['average', 'weighted_average', 'majority_vote', 'confidence_weighted']
                    ),
                    'signal_threshold': trial.suggest_float('signal_threshold', 0.3, 0.7),
                    'signal_smoothing': trial.suggest_categorical('signal_smoothing', [True, False]),
                    'smoothing_window': trial.suggest_int('smoothing_window', 2, 10) 
                        if trial.params.get('signal_smoothing', False) else None,
                },
                
                # Multi-timeframe integration
                'timeframe_integration': {
                    'use_multiple_timeframes': trial.suggest_categorical('use_multiple_timeframes', [True, False]),
                    'timeframes': trial.suggest_categorical(
                        'timeframes',
                        [['1D'], ['1D', '1H'], ['1D', '4H', '1H'], ['1D', '1W']]
                    ) if trial.params.get('use_multiple_timeframes', False) else None,
                    'timeframe_weights': trial.suggest_categorical(
                        'timeframe_weights',
                        ['equal', 'exponential', 'adaptive']
                    ) if trial.params.get('use_multiple_timeframes', False) else None,
                },
                
                # Confidence estimation
                'confidence_estimation': {
                    'method': trial.suggest_categorical(
                        'confidence_method',
                        ['probability', 'agreement', 'historical', 'combined']
                    ),
                    'min_confidence': trial.suggest_float('min_confidence', 0.5, 0.8),
                    'confidence_scaling': trial.suggest_categorical('confidence_scaling', [True, False]),
                },
                
                # Strategy selection
                'strategy_selection': {
                    'selection_method': trial.suggest_categorical(
                        'selection_method',
                        ['static', 'dynamic', 'regime_based', 'performance_based']
                    ),
                    'rebalance_frequency': trial.suggest_categorical(
                        'rebalance_frequency',
                        ['daily', 'weekly', 'monthly', 'adaptive']
                    ),
                    'min_strategies': trial.suggest_int('min_strategies', 1, 3),
                    'max_strategies': trial.suggest_int('max_strategies', 3, 10),
                },
                
                # Execution optimization
                'execution': {
                    'order_type': trial.suggest_categorical(
                        'order_type',
                        ['market', 'limit', 'adaptive']
                    ),
                    'slippage_model': trial.suggest_categorical(
                        'slippage_model',
                        ['fixed', 'linear', 'square_root', 'adaptive']
                    ),
                    'execution_delay': trial.suggest_int('execution_delay', 0, 5),
                    'partial_fills': trial.suggest_categorical('partial_fills', [True, False]),
                },
                
                # Performance tracking
                'performance_tracking': {
                    'track_attribution': trial.suggest_categorical('track_attribution', [True, False]),
                    'update_frequency': trial.suggest_categorical(
                        'update_frequency',
                        ['trade', 'daily', 'weekly']
                    ),
                    'decay_factor': trial.suggest_float('decay_factor', 0.9, 0.99),
                }
            }
        }
        
        return params
    
    def evaluate(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """
        Evaluate integrated strategy performance
        
        Args:
            data: Market data
            params: Complete parameter set from all optimization loops
            
        Returns:
            Overall performance metric (higher is better)
        """
        try:
            # Generate ensemble predictions
            predictions = self._generate_ensemble_predictions(data, params)
            
            if predictions is None or len(predictions) < 100:
                return -np.inf
            
            # Generate trading signals
            signals = self._generate_integrated_signals(predictions, params.get('integration', {}))
            
            # Apply confidence filtering
            signals = self._apply_confidence_filter(signals, predictions, params.get('integration', {}))
            
            # Apply multi-timeframe integration if enabled
            if params.get('integration', {}).get('timeframe_integration', {}).get('use_multiple_timeframes', False):
                signals = self._integrate_timeframes(data, signals, params.get('integration', {}))
            
            # Apply execution optimization
            executed_signals = self._optimize_execution(data, signals, params.get('integration', {}))
            
            # Calculate final performance
            performance = self._calculate_integrated_performance(data, executed_signals, params)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error in integration evaluation: {str(e)}")
            return -np.inf
    
    def _generate_ensemble_predictions(self, data: pd.DataFrame, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Generate predictions from ensemble of models"""
        predictions = pd.DataFrame(index=data.index)
        
        # Simulate predictions from different models (in practice, would use trained models)
        # Base model predictions
        if params.get('integration', {}).get('model_combination', {}).get('use_base_models', True):
            # Trend following model
            sma_short = data['close'].rolling(10).mean()
            sma_long = data['close'].rolling(30).mean()
            predictions['trend_model'] = (sma_short > sma_long).astype(float)
            predictions['trend_confidence'] = abs(sma_short - sma_long) / sma_long
            
            # Mean reversion model
            rsi = self._calculate_rsi(data['close'], 14)
            predictions['mean_reversion_model'] = np.where(
                rsi < 30, 1.0,
                np.where(rsi > 70, 0.0, 0.5)
            )
            predictions['mean_reversion_confidence'] = abs(rsi - 50) / 50
            
            # Momentum model
            momentum = data['close'] / data['close'].shift(20) - 1
            predictions['momentum_model'] = (momentum > 0).astype(float)
            predictions['momentum_confidence'] = abs(momentum).clip(0, 1)
        
        # Regime-based predictions
        if params.get('integration', {}).get('model_combination', {}).get('use_regime_models', True):
            # Simulate regime predictions
            volatility = data['close'].pct_change().rolling(20).std()
            vol_regime = pd.cut(volatility, bins=3, labels=[0, 1, 2])
            
            # Different predictions for different regimes
            predictions['regime_model'] = np.where(
                vol_regime == 0, predictions.get('trend_model', 0.5),
                np.where(vol_regime == 2, predictions.get('mean_reversion_model', 0.5), 0.5)
            )
            predictions['regime_confidence'] = 0.7  # Fixed confidence for simplicity
        
        # Meta features
        if params.get('integration', {}).get('model_combination', {}).get('use_meta_features', True):
            # Agreement between models
            base_predictions = predictions[[col for col in predictions.columns if '_model' in col and 'confidence' not in col]]
            if not base_predictions.empty:
                predictions['model_agreement'] = base_predictions.std(axis=1)
                predictions['model_consensus'] = base_predictions.mean(axis=1)
        
        return predictions.dropna()
    
    def _generate_integrated_signals(self, predictions: pd.DataFrame, 
                                   integration_params: Dict[str, Any]) -> pd.Series:
        """Generate trading signals from ensemble predictions"""
        signal_params = integration_params.get('signal_generation', {})
        aggregation = signal_params.get('signal_aggregation', 'average')
        
        # Get model predictions and confidences
        model_cols = [col for col in predictions.columns if '_model' in col and 'confidence' not in col]
        confidence_cols = [col for col in predictions.columns if 'confidence' in col]
        
        if not model_cols:
            return pd.Series(0, index=predictions.index)
        
        if aggregation == 'average':
            signals = predictions[model_cols].mean(axis=1)
            
        elif aggregation == 'weighted_average':
            if confidence_cols:
                weights = predictions[confidence_cols].values
                weighted_sum = (predictions[model_cols].values * weights).sum(axis=1)
                weight_sum = weights.sum(axis=1)
                signals = pd.Series(weighted_sum / (weight_sum + 1e-6), index=predictions.index)
            else:
                signals = predictions[model_cols].mean(axis=1)
                
        elif aggregation == 'majority_vote':
            threshold = signal_params.get('signal_threshold', 0.5)
            votes = (predictions[model_cols] > threshold).sum(axis=1)
            signals = (votes > len(model_cols) / 2).astype(float)
            
        else:  # confidence_weighted
            if 'model_consensus' in predictions:
                signals = predictions['model_consensus']
            else:
                signals = predictions[model_cols].mean(axis=1)
        
        # Apply signal smoothing
        if signal_params.get('signal_smoothing', False):
            window = signal_params.get('smoothing_window', 3)
            signals = signals.rolling(window).mean()
        
        # Convert to trading signals (-1, 0, 1)
        threshold = signal_params.get('signal_threshold', 0.5)
        trading_signals = pd.Series(0, index=signals.index)
        trading_signals[signals > threshold] = 1
        trading_signals[signals < (1 - threshold)] = -1
        
        return trading_signals
    
    def _apply_confidence_filter(self, signals: pd.Series, predictions: pd.DataFrame,
                               integration_params: Dict[str, Any]) -> pd.Series:
        """Filter signals based on confidence estimation"""
        confidence_params = integration_params.get('confidence_estimation', {})
        method = confidence_params.get('method', 'probability')
        min_confidence = confidence_params.get('min_confidence', 0.6)
        
        # Calculate confidence scores
        if method == 'probability':
            # Use model prediction probabilities
            model_cols = [col for col in predictions.columns if '_model' in col and 'confidence' not in col]
            if model_cols:
                # Distance from 0.5 indicates confidence
                confidence = abs(predictions[model_cols].mean(axis=1) - 0.5) * 2
            else:
                confidence = pd.Series(1, index=signals.index)
                
        elif method == 'agreement':
            # Model agreement as confidence
            if 'model_agreement' in predictions:
                # Low std = high agreement = high confidence
                confidence = 1 - predictions['model_agreement']
            else:
                confidence = pd.Series(1, index=signals.index)
                
        elif method == 'historical':
            # Use historical accuracy as confidence
            # Simplified: use rolling win rate
            returns = predictions.index.to_series().diff()  # Placeholder for returns
            signal_returns = signals.shift(1) * returns
            win_rate = signal_returns.rolling(50).apply(lambda x: (x > 0).sum() / len(x))
            confidence = win_rate
            
        else:  # combined
            # Combine multiple confidence measures
            confidences = []
            
            model_cols = [col for col in predictions.columns if '_model' in col and 'confidence' not in col]
            if model_cols:
                prob_confidence = abs(predictions[model_cols].mean(axis=1) - 0.5) * 2
                confidences.append(prob_confidence)
            
            if 'model_agreement' in predictions:
                agreement_confidence = 1 - predictions['model_agreement']
                confidences.append(agreement_confidence)
            
            confidence_cols = [col for col in predictions.columns if 'confidence' in col]
            if confidence_cols:
                avg_confidence = predictions[confidence_cols].mean(axis=1)
                confidences.append(avg_confidence)
            
            if confidences:
                confidence = pd.concat(confidences, axis=1).mean(axis=1)
            else:
                confidence = pd.Series(1, index=signals.index)
        
        # Apply confidence filter
        filtered_signals = signals.copy()
        filtered_signals[confidence < min_confidence] = 0
        
        # Apply confidence scaling if enabled
        if confidence_params.get('confidence_scaling', False):
            # Scale position size by confidence
            filtered_signals = filtered_signals * confidence.clip(0, 1)
        
        return filtered_signals
    
    def _integrate_timeframes(self, data: pd.DataFrame, signals: pd.Series,
                            integration_params: Dict[str, Any]) -> pd.Series:
        """Integrate signals across multiple timeframes"""
        timeframe_params = integration_params.get('timeframe_integration', {})
        timeframes = timeframe_params.get('timeframes', ['1D'])
        weights_method = timeframe_params.get('timeframe_weights', 'equal')
        
        if len(timeframes) == 1:
            return signals
        
        # Simulate signals from different timeframes
        timeframe_signals = {}
        
        for tf in timeframes:
            if tf == '1D':
                timeframe_signals[tf] = signals
            elif tf == '1H':
                # Simulate hourly signals (more frequent)
                hourly_signals = signals.copy()
                noise = np.random.normal(0, 0.1, len(hourly_signals))
                hourly_signals = hourly_signals + noise
                timeframe_signals[tf] = hourly_signals.clip(-1, 1)
            elif tf == '4H':
                # Simulate 4-hour signals
                four_hour_signals = signals.rolling(4).mean()
                timeframe_signals[tf] = four_hour_signals
            elif tf == '1W':
                # Simulate weekly signals (less frequent)
                weekly_signals = signals.rolling(5).mean()
                timeframe_signals[tf] = weekly_signals
        
        # Combine timeframe signals
        if weights_method == 'equal':
            weights = {tf: 1/len(timeframes) for tf in timeframes}
        elif weights_method == 'exponential':
            # Give more weight to longer timeframes
            weights = {}
            tf_order = ['1H', '4H', '1D', '1W']
            for i, tf in enumerate(timeframes):
                idx = tf_order.index(tf)
                weights[tf] = 2 ** idx
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
        else:  # adaptive
            # Weight by signal consistency
            weights = {}
            for tf in timeframes:
                # Measure signal stability
                stability = 1 - timeframe_signals[tf].diff().abs().mean()
                weights[tf] = stability
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
        
        # Weighted combination
        integrated = pd.Series(0, index=signals.index)
        for tf, weight in weights.items():
            integrated += timeframe_signals[tf] * weight
        
        return integrated
    
    def _optimize_execution(self, data: pd.DataFrame, signals: pd.Series,
                          integration_params: Dict[str, Any]) -> pd.Series:
        """Optimize signal execution"""
        execution_params = integration_params.get('execution', {})
        order_type = execution_params.get('order_type', 'market')
        slippage_model = execution_params.get('slippage_model', 'fixed')
        delay = execution_params.get('execution_delay', 0)
        
        # Apply execution delay
        if delay > 0:
            executed_signals = signals.shift(delay)
        else:
            executed_signals = signals.copy()
        
        # Apply slippage adjustment
        if slippage_model != 'adaptive':  # Simplified slippage
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std()
            
            if slippage_model == 'fixed':
                slippage = 0.0001  # 1 basis point
            elif slippage_model == 'linear':
                slippage = 0.0001 * abs(executed_signals)
            else:  # square_root
                slippage = 0.0001 * np.sqrt(abs(executed_signals))
            
            # Reduce signal magnitude by slippage
            executed_signals = executed_signals * (1 - slippage)
        
        # Handle partial fills
        if execution_params.get('partial_fills', False):
            # Simulate partial fills for large orders
            fill_probability = 1 - abs(executed_signals) * 0.2  # Larger orders less likely to fill
            fill_probability = fill_probability.clip(0.5, 1)
            
            random_fills = pd.Series(np.random.uniform(0, 1, len(executed_signals)), index=executed_signals.index)
            executed_signals = executed_signals * (random_fills < fill_probability)
        
        return executed_signals
    
    def _calculate_integrated_performance(self, data: pd.DataFrame, signals: pd.Series,
                                        params: Dict[str, Any]) -> float:
        """Calculate comprehensive performance of integrated strategy"""
        returns = data['close'].pct_change()
        strategy_returns = signals.shift(1) * returns
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) < 100:
            return -np.inf
        
        # Basic performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / (volatility + 1e-6)
        
        # Drawdown analysis
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (running_max - cumulative) / running_max
        max_drawdown = drawdown.max()
        
        # Risk metrics
        var_95 = strategy_returns.quantile(0.05)
        cvar_95 = strategy_returns[strategy_returns <= var_95].mean()
        
        # Trading metrics
        n_trades = (signals.diff() != 0).sum()
        if n_trades > 0:
            trades_per_year = n_trades * 252 / len(signals)
            
            # Win rate
            trade_returns = []
            position = 0
            entry_price = 0
            
            for i in range(len(signals)):
                if signals.iloc[i] != position and signals.iloc[i] != 0:
                    if position != 0:
                        # Close previous position
                        exit_price = data['close'].iloc[i]
                        if position > 0:
                            trade_return = (exit_price - entry_price) / entry_price
                        else:
                            trade_return = (entry_price - exit_price) / entry_price
                        trade_returns.append(trade_return)
                    
                    # Open new position
                    position = signals.iloc[i]
                    entry_price = data['close'].iloc[i]
            
            if trade_returns:
                win_rate = sum(r > 0 for r in trade_returns) / len(trade_returns)
                avg_win = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
                avg_loss = abs(np.mean([r for r in trade_returns if r < 0])) if any(r < 0 for r in trade_returns) else 1
                profit_factor = (avg_win * win_rate) / (avg_loss * (1 - win_rate)) if avg_loss > 0 else 2
            else:
                win_rate = 0.5
                profit_factor = 1
        else:
            trades_per_year = 0
            win_rate = 0.5
            profit_factor = 1
        
        # Attribution analysis (if enabled)
        attribution_score = 1.0
        if params.get('integration', {}).get('performance_tracking', {}).get('track_attribution', False):
            # Simplified: check if different components contribute positively
            model_contributions = []
            
            # Trend component
            trend_signals = (data['close'].rolling(20).mean() > data['close'].rolling(50).mean()).astype(float) * 2 - 1
            trend_returns = trend_signals.shift(1) * returns
            model_contributions.append(trend_returns.mean())
            
            # Mean reversion component
            rsi = self._calculate_rsi(data['close'], 14)
            mr_signals = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
            mr_returns = pd.Series(mr_signals, index=data.index).shift(1) * returns
            model_contributions.append(mr_returns.mean())
            
            # Attribution score based on positive contributions
            positive_contributions = sum(c > 0 for c in model_contributions)
            attribution_score = 0.8 + 0.2 * (positive_contributions / len(model_contributions))
        
        # Calculate final score
        score_components = {
            'sharpe': max(0, min(sharpe_ratio / 2, 1)),  # Normalize to [0, 1]
            'returns': max(0, min(annual_return, 1)),  # Cap at 100% annual return
            'drawdown': 1 - min(max_drawdown * 2, 1),  # Penalize drawdowns > 50%
            'win_rate': win_rate,
            'profit_factor': min(profit_factor / 2, 1),  # Normalize
            'risk_adjusted': max(0, min(-annual_return / cvar_95 if cvar_95 < 0 else 1, 1)),
            'attribution': attribution_score,
            'trade_frequency': min(trades_per_year / 100, 1)  # Normalize
        }
        
        # Weighted combination
        weights = {
            'sharpe': 0.25,
            'returns': 0.15,
            'drawdown': 0.20,
            'win_rate': 0.10,
            'profit_factor': 0.10,
            'risk_adjusted': 0.10,
            'attribution': 0.05,
            'trade_frequency': 0.05
        }
        
        final_score = sum(score_components[k] * weights[k] for k in weights)
        
        # Apply penalties
        if max_drawdown > 0.5:
            final_score *= 0.5
        if volatility > 0.5:
            final_score *= 0.8
        if trades_per_year < 10:
            final_score *= 0.9  # Penalize too few trades
        
        return final_score
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def build_final_strategy(self, data: pd.DataFrame, best_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build final integrated strategy using optimized parameters
        
        Args:
            data: Market data
            best_params: Optimized parameters from all loops
            
        Returns:
            Dictionary containing strategy components and configuration
        """
        strategy = {
            'features': best_params.get('feature_engineering', {}),
            'models': best_params.get('model_architecture', {}),
            'regimes': best_params.get('regime_detection', {}),
            'risk': best_params.get('risk_management', {}),
            'integration': best_params.get('integration', {}),
            'performance_metrics': {},
            'execution_params': {}
        }
        
        # Add performance tracking
        predictions = self._generate_ensemble_predictions(data, best_params)
        signals = self._generate_integrated_signals(predictions, best_params.get('integration', {}))
        
        returns = data['close'].pct_change()
        strategy_returns = signals.shift(1) * returns
        
        strategy['performance_metrics'] = {
            'sharpe_ratio': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252),
            'total_return': (1 + strategy_returns).prod() - 1,
            'max_drawdown': self._calculate_max_drawdown(strategy_returns),
            'n_trades': (signals.diff() != 0).sum()
        }
        
        return strategy
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (running_max - cumulative) / running_max
        return drawdown.max()