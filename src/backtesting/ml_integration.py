"""
ML Integration module for the backtesting engine.

This module provides the bridge between ML models and the backtesting engine,
handling signal generation, position sizing, and performance tracking.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .events import SignalEvent, EventType
from .engine import BacktestEngine
from ..strategies.ml_strategy import MLStrategy, MLSignal
from ..ml.features.feature_engineering import FeatureEngineer


@dataclass
class MLBacktestConfig:
    """Configuration for ML-enhanced backtesting."""
    use_walk_forward: bool = True
    walk_forward_window: int = 252  # 1 year
    retrain_frequency: int = 63  # Quarterly
    validation_split: float = 0.2
    min_training_samples: int = 500
    feature_selection: bool = True
    feature_importance_threshold: float = 0.01
    ensemble_voting: str = 'soft'  # 'soft' or 'hard'
    risk_parity: bool = True
    max_correlation: float = 0.95


class MLBacktestEngine(BacktestEngine):
    """
    Extended backtesting engine with ML integration.
    
    Adds ML-specific functionality to the base BacktestEngine:
    - ML model training and prediction
    - Feature engineering pipeline
    - Walk-forward analysis
    - Model performance tracking
    - Dynamic position sizing based on ML confidence
    """
    
    def __init__(
        self,
        ml_config: Optional[MLBacktestConfig] = None,
        **kwargs
    ):
        """
        Initialize ML-enhanced backtesting engine.
        
        Args:
            ml_config: ML-specific configuration
            **kwargs: Arguments for base BacktestEngine
        """
        super().__init__(**kwargs)
        
        self.ml_config = ml_config or MLBacktestConfig()
        self.feature_engineer = FeatureEngineer()
        
        # ML tracking
        self.ml_predictions = []
        self.feature_importance_history = []
        self.model_performance_history = []
        self.regime_history = []
        
        # Walk-forward tracking
        self.training_windows = []
        self.out_of_sample_periods = []
        
    def run_ml_backtest(
        self,
        data: pd.DataFrame,
        ml_strategy: MLStrategy,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        progress_bar: bool = True
    ) -> Dict[str, Any]:
        """
        Run ML-enhanced backtest with walk-forward analysis.
        
        Args:
            data: Market data
            ml_strategy: ML trading strategy
            start_date: Backtest start date
            end_date: Backtest end date
            progress_bar: Show progress bar
            
        Returns:
            Enhanced backtest results with ML metrics
        """
        # Prepare data with features
        print("Preparing features...")
        featured_data = self._prepare_ml_data(data)
        
        if self.ml_config.use_walk_forward:
            results = self._run_walk_forward_backtest(
                featured_data,
                ml_strategy,
                start_date,
                end_date,
                progress_bar
            )
        else:
            # Train once and run standard backtest
            print("Training ML models...")
            self._train_ml_strategy(ml_strategy, featured_data)
            
            results = self.run(
                featured_data,
                ml_strategy,
                start_date,
                end_date,
                progress_bar
            )
        
        # Add ML-specific metrics
        results['ml_metrics'] = self._calculate_ml_metrics()
        results['feature_importance'] = self._aggregate_feature_importance()
        results['regime_analysis'] = self._analyze_regimes()
        results['prediction_analysis'] = self._analyze_predictions()
        
        return results
    
    def _prepare_ml_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with ML features."""
        # Use feature engineer to create comprehensive feature set
        features_config = {
            'price_features': True,
            'volume_features': True,
            'technical_indicators': True,
            'statistical_features': True,
            'market_microstructure': True,
            'regime_features': True
        }
        
        featured_data = self.feature_engineer.engineer_features(
            data,
            config=features_config
        )
        
        # Feature selection if enabled
        if self.ml_config.feature_selection:
            selected_features = self.feature_engineer.select_features(
                featured_data,
                target='returns',
                method='mutual_info',
                threshold=self.ml_config.feature_importance_threshold
            )
            
            # Keep only selected features plus OHLCV
            base_columns = ['open', 'high', 'low', 'close', 'volume']
            keep_columns = base_columns + selected_features
            featured_data = featured_data[keep_columns]
        
        # Handle high correlation features
        if self.ml_config.max_correlation < 1.0:
            featured_data = self._remove_correlated_features(
                featured_data,
                threshold=self.ml_config.max_correlation
            )
        
        return featured_data
    
    def _run_walk_forward_backtest(
        self,
        data: pd.DataFrame,
        ml_strategy: MLStrategy,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        progress_bar: bool
    ) -> Dict[str, Any]:
        """Run walk-forward ML backtest."""
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Calculate walk-forward windows
        total_periods = len(data)
        window_size = self.ml_config.walk_forward_window
        step_size = self.ml_config.retrain_frequency
        
        results_list = []
        
        # Initial training period
        if total_periods < self.ml_config.min_training_samples:
            raise ValueError(f"Insufficient data for walk-forward analysis. Need at least {self.ml_config.min_training_samples} samples.")
        
        current_pos = self.ml_config.min_training_samples
        
        while current_pos < total_periods:
            # Define training window
            train_start_idx = max(0, current_pos - window_size)
            train_end_idx = current_pos
            
            # Define out-of-sample test window
            test_start_idx = current_pos
            test_end_idx = min(current_pos + step_size, total_periods)
            
            # Get data slices
            train_data = data.iloc[train_start_idx:train_end_idx]
            test_data = data.iloc[test_start_idx:test_end_idx]
            
            # Train ML models
            print(f"\nTraining on data from {train_data.index[0]} to {train_data.index[-1]}")
            self._train_ml_strategy(ml_strategy, train_data)
            
            # Store training window info
            self.training_windows.append({
                'start': train_data.index[0],
                'end': train_data.index[-1],
                'samples': len(train_data)
            })
            
            # Run backtest on out-of-sample period
            print(f"Testing on data from {test_data.index[0]} to {test_data.index[-1]}")
            
            # Run standard backtest on this window
            window_results = super().run(
                test_data,
                ml_strategy,
                progress_bar=progress_bar
            )
            
            results_list.append(window_results)
            
            # Store out-of-sample period info
            self.out_of_sample_periods.append({
                'start': test_data.index[0],
                'end': test_data.index[-1],
                'samples': len(test_data),
                'performance': window_results['performance']
            })
            
            # Move to next window
            current_pos += step_size
        
        # Aggregate results from all windows
        aggregated_results = self._aggregate_walk_forward_results(results_list)
        
        return aggregated_results
    
    def _train_ml_strategy(self, ml_strategy: MLStrategy, train_data: pd.DataFrame):
        """Train ML models in the strategy."""
        # Prepare features
        featured_data = ml_strategy.prepare_features(train_data)
        
        # Train models
        if ml_strategy.use_ensemble:
            ml_strategy.model.fit(featured_data, validate=True, n_splits=3)
            
            # Store feature importance
            if hasattr(ml_strategy.model, 'get_feature_importance'):
                importance = ml_strategy.model.get_feature_importance()
                self.feature_importance_history.append({
                    'timestamp': featured_data.index[-1],
                    'importance': importance
                })
        else:
            # Train individual models
            ml_strategy.direction_model.fit(featured_data, validate=True, n_splits=3)
            ml_strategy.volatility_model.fit(featured_data, validate=True, n_splits=3)
            ml_strategy.regime_model.fit(featured_data)
        
        # Update training tracking
        ml_strategy.last_training_date = train_data.index[-1]
        ml_strategy.bars_since_training = 0
    
    def _handle_market_event(self, event):
        """Extended market event handler with ML signal generation."""
        # Call parent handler
        super()._handle_market_event(event)
        
        # Generate ML signals if strategy is MLStrategy
        if isinstance(self.strategy, MLStrategy):
            # Get recent data for ML prediction
            current_idx = self.market_data_index - 1
            if current_idx < self.strategy.feature_lookback:
                return
            
            # Get data slice
            lookback_start = max(0, current_idx - self.strategy.feature_lookback)
            data_slice = self.data.iloc[lookback_start:current_idx + 1]
            
            # Get current positions
            current_positions = self.portfolio.get_open_positions()
            
            # Generate ML signal
            ml_signal = self.strategy.generate_signal(data_slice, current_positions)
            
            if ml_signal:
                # Store prediction for analysis
                self.ml_predictions.append({
                    'timestamp': ml_signal.timestamp,
                    'symbol': ml_signal.symbol,
                    'direction': ml_signal.direction,
                    'confidence': ml_signal.confidence,
                    'probability': ml_signal.probability,
                    'regime': ml_signal.market_regime.value,
                    'volatility_forecast': ml_signal.volatility_forecast,
                    'risk_score': ml_signal.risk_score
                })
                
                # Store regime
                self.regime_history.append({
                    'timestamp': ml_signal.timestamp,
                    'regime': ml_signal.market_regime,
                    'confidence': ml_signal.confidence
                })
                
                # Calculate position size
                position_size = self.strategy.calculate_position_size(
                    ml_signal,
                    self.portfolio.current_capital,
                    event.close
                )
                
                # Generate trading signal event
                signal_event = SignalEvent(
                    timestamp=ml_signal.timestamp,
                    symbol=ml_signal.symbol,
                    signal_type='ML_LONG' if ml_signal.direction == 1 else 'ML_SHORT',
                    strength=ml_signal.confidence,
                    metadata={
                        'ml_signal': ml_signal,
                        'position_size': position_size
                    }
                )
                
                self.events_queue.put(signal_event)
                self.generated_signals += 1
    
    def _aggregate_walk_forward_results(self, results_list: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from walk-forward windows."""
        # Combine equity curves
        equity_curves = []
        for result in results_list:
            equity_curves.append(result['equity_curve'])
        
        combined_equity = pd.concat(equity_curves)
        
        # Aggregate trades
        all_trades = []
        for result in results_list:
            all_trades.extend(result.get('trades', []))
        
        # Calculate overall performance metrics
        total_return = (combined_equity.iloc[-1] / combined_equity.iloc[0] - 1) * 100
        
        # Calculate other metrics
        returns = combined_equity.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Maximum drawdown
        rolling_max = combined_equity.expanding().max()
        drawdown = (combined_equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'equity_curve': combined_equity,
            'trades': all_trades,
            'performance': {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': len(all_trades),
                'walk_forward_windows': len(results_list)
            },
            'walk_forward_details': {
                'training_windows': self.training_windows,
                'out_of_sample_periods': self.out_of_sample_periods
            }
        }
    
    def _calculate_ml_metrics(self) -> Dict[str, Any]:
        """Calculate ML-specific performance metrics."""
        if not self.ml_predictions:
            return {}
        
        predictions_df = pd.DataFrame(self.ml_predictions)
        
        # Prediction accuracy (would need actual outcomes for real accuracy)
        # For now, we'll analyze prediction distribution and confidence
        
        return {
            'total_predictions': len(predictions_df),
            'avg_confidence': predictions_df['confidence'].mean(),
            'avg_probability': predictions_df['probability'].mean(),
            'direction_distribution': predictions_df['direction'].value_counts().to_dict(),
            'regime_distribution': predictions_df['regime'].value_counts().to_dict(),
            'avg_risk_score': predictions_df['risk_score'].mean(),
            'confidence_by_regime': predictions_df.groupby('regime')['confidence'].mean().to_dict(),
            'volatility_forecast_stats': {
                'mean': predictions_df['volatility_forecast'].mean(),
                'std': predictions_df['volatility_forecast'].std(),
                'min': predictions_df['volatility_forecast'].min(),
                'max': predictions_df['volatility_forecast'].max()
            }
        }
    
    def _aggregate_feature_importance(self) -> Dict[str, float]:
        """Aggregate feature importance across training windows."""
        if not self.feature_importance_history:
            return {}
        
        # Combine all feature importances
        all_features = {}
        for record in self.feature_importance_history:
            for feature, importance in record['importance'].items():
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(importance)
        
        # Average importance
        avg_importance = {
            feature: np.mean(scores)
            for feature, scores in all_features.items()
        }
        
        # Sort by importance
        sorted_importance = dict(
            sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_importance
    
    def _analyze_regimes(self) -> Dict[str, Any]:
        """Analyze market regime patterns."""
        if not self.regime_history:
            return {}
        
        regime_df = pd.DataFrame(self.regime_history)
        
        # Calculate regime statistics
        regime_counts = regime_df['regime'].value_counts()
        regime_durations = {}
        
        # Calculate average duration for each regime
        current_regime = None
        regime_start = None
        durations = {regime: [] for regime in regime_df['regime'].unique()}
        
        for idx, row in regime_df.iterrows():
            if row['regime'] != current_regime:
                if current_regime is not None and regime_start is not None:
                    duration = (row['timestamp'] - regime_start).days
                    durations[current_regime].append(duration)
                
                current_regime = row['regime']
                regime_start = row['timestamp']
        
        # Calculate average durations
        for regime, dur_list in durations.items():
            if dur_list:
                regime_durations[regime.value] = {
                    'avg_duration_days': np.mean(dur_list),
                    'max_duration_days': max(dur_list),
                    'min_duration_days': min(dur_list)
                }
        
        return {
            'regime_counts': regime_counts.to_dict(),
            'regime_durations': regime_durations,
            'regime_transitions': self._calculate_regime_transitions(regime_df),
            'regime_confidence_stats': regime_df.groupby('regime')['confidence'].describe().to_dict()
        }
    
    def _calculate_regime_transitions(self, regime_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Calculate regime transition matrix."""
        transitions = {}
        
        for i in range(1, len(regime_df)):
            from_regime = regime_df.iloc[i-1]['regime']
            to_regime = regime_df.iloc[i]['regime']
            
            if from_regime not in transitions:
                transitions[from_regime] = {}
            
            if to_regime not in transitions[from_regime]:
                transitions[from_regime][to_regime] = 0
            
            transitions[from_regime][to_regime] += 1
        
        # Convert to readable format
        transition_matrix = {}
        for from_regime, to_dict in transitions.items():
            transition_matrix[from_regime.value] = {
                to_regime.value: count
                for to_regime, count in to_dict.items()
            }
        
        return transition_matrix
    
    def _analyze_predictions(self) -> Dict[str, Any]:
        """Analyze ML prediction patterns."""
        if not self.ml_predictions:
            return {}
        
        predictions_df = pd.DataFrame(self.ml_predictions)
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        predictions_df.set_index('timestamp', inplace=True)
        
        # Analyze prediction patterns by time
        hourly_patterns = predictions_df.groupby(predictions_df.index.hour).agg({
            'confidence': 'mean',
            'probability': 'mean',
            'risk_score': 'mean'
        }).to_dict()
        
        # Analyze by day of week
        daily_patterns = predictions_df.groupby(predictions_df.index.dayofweek).agg({
            'confidence': 'mean',
            'probability': 'mean',
            'risk_score': 'mean'
        }).to_dict()
        
        # Confidence distribution
        confidence_bins = pd.cut(predictions_df['confidence'], bins=5)
        confidence_distribution = confidence_bins.value_counts().to_dict()
        
        return {
            'hourly_patterns': hourly_patterns,
            'daily_patterns': daily_patterns,
            'confidence_distribution': confidence_distribution,
            'high_confidence_ratio': (predictions_df['confidence'] > 0.8).mean(),
            'prediction_frequency': len(predictions_df) / len(self.data) if len(self.data) > 0 else 0
        }
    
    def _remove_correlated_features(
        self,
        data: pd.DataFrame,
        threshold: float = 0.95
    ) -> pd.DataFrame:
        """Remove highly correlated features."""
        # Calculate correlation matrix
        corr_matrix = data.corr().abs()
        
        # Find features to remove
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [
            column for column in upper_tri.columns
            if any(upper_tri[column] > threshold)
        ]
        
        # Keep OHLCV columns
        base_columns = ['open', 'high', 'low', 'close', 'volume']
        to_drop = [col for col in to_drop if col not in base_columns]
        
        return data.drop(columns=to_drop)