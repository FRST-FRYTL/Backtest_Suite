"""
ML-enhanced trading strategy that integrates machine learning models with the backtesting engine.

This module provides a strategy class that uses ML predictions for trading decisions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np

from .builder import Strategy, PositionSizing, RiskManagement
from .rules import Rule, Condition
from ..ml.models import (
    DirectionPredictor,
    VolatilityForecaster,
    MarketRegimeDetector,
    EnsembleModel,
    MarketRegime
)
from ..indicators.technical_indicators import SMA, RSI, BollingerBands, ATR


@dataclass
class MLSignal:
    """Container for ML-based trading signals."""
    timestamp: datetime
    symbol: str
    direction: int  # 1 for long, -1 for short, 0 for neutral
    confidence: float
    probability: float
    volatility_forecast: float
    market_regime: MarketRegime
    risk_score: float
    position_size_adjustment: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLStrategy(Strategy):
    """
    Machine Learning enhanced trading strategy.
    
    This strategy uses ML models to generate trading signals based on:
    - Direction prediction (XGBoost)
    - Volatility forecasting (LSTM)
    - Market regime detection
    - Ensemble predictions
    """
    
    def __init__(
        self,
        name: str = "ML Enhanced Strategy",
        use_ensemble: bool = True,
        direction_threshold: float = 0.65,
        confidence_threshold: float = 0.7,
        regime_filter: bool = True,
        allowed_regimes: Optional[List[MarketRegime]] = None,
        volatility_scaling: bool = True,
        max_volatility_multiplier: float = 2.0,
        risk_per_trade: float = 0.02,
        feature_lookback: int = 50,
        retrain_frequency: int = 252,  # Retrain yearly
        **kwargs
    ):
        """
        Initialize ML Strategy.
        
        Args:
            name: Strategy name
            use_ensemble: Whether to use ensemble model or individual models
            direction_threshold: Minimum probability for direction prediction
            confidence_threshold: Minimum confidence for trading
            regime_filter: Whether to filter trades by market regime
            allowed_regimes: List of regimes to trade in (None = all)
            volatility_scaling: Whether to scale position by volatility
            max_volatility_multiplier: Maximum volatility adjustment
            risk_per_trade: Risk per trade as fraction of capital
            feature_lookback: Lookback period for feature calculation
            retrain_frequency: How often to retrain models (in bars)
            **kwargs: Additional arguments for parent Strategy
        """
        super().__init__(name=name, **kwargs)
        
        self.use_ensemble = use_ensemble
        self.direction_threshold = direction_threshold
        self.confidence_threshold = confidence_threshold
        self.regime_filter = regime_filter
        self.allowed_regimes = allowed_regimes or [
            MarketRegime.BULLISH,
            MarketRegime.NEUTRAL
        ]
        self.volatility_scaling = volatility_scaling
        self.max_volatility_multiplier = max_volatility_multiplier
        self.risk_per_trade = risk_per_trade
        self.feature_lookback = feature_lookback
        self.retrain_frequency = retrain_frequency
        
        # Initialize ML models
        self._initialize_models()
        
        # Training tracking
        self.last_training_date = None
        self.bars_since_training = 0
        self.model_performance = {}
        
        # Signal history for analysis
        self.signal_history = []
        
    def _initialize_models(self):
        """Initialize ML models."""
        if self.use_ensemble:
            self.model = EnsembleModel(
                ensemble_method='dynamic',
                direction_weight=0.4,
                volatility_weight=0.3,
                regime_weight=0.3,
                use_meta_learner=True,
                risk_adjustment=True
            )
        else:
            self.direction_model = DirectionPredictor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05
            )
            self.volatility_model = VolatilityForecaster(
                sequence_length=40,
                lstm_units=[64, 32],
                epochs=50,
                use_attention=True
            )
            self.regime_model = MarketRegimeDetector(
                method='ensemble',
                n_regimes=5,
                lookback_period=252
            )
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML models.
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            DataFrame with calculated features
        """
        features = data.copy()
        
        # Price-based features
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        features['high_low_ratio'] = features['high'] / features['low']
        features['close_open_ratio'] = features['close'] / features['open']
        
        # Volume features
        features['volume_sma'] = features['volume'].rolling(20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_sma']
        
        # Technical indicators
        # SMA features
        for period in [10, 20, 50, 200]:
            sma = SMA(period=period)
            features[f'sma_{period}'] = sma.calculate(features)['sma']
            features[f'price_to_sma_{period}'] = features['close'] / features[f'sma_{period}']
        
        # RSI
        rsi = RSI(period=14)
        features['rsi'] = rsi.calculate(features)['rsi']
        
        # Bollinger Bands
        bb = BollingerBands(period=20, std_dev=2)
        bb_data = bb.calculate(features)
        features['bb_upper'] = bb_data['upper']
        features['bb_lower'] = bb_data['lower']
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['close']
        features['bb_position'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # ATR for volatility
        atr = ATR(period=14)
        features['atr'] = atr.calculate(features)['atr']
        features['atr_percent'] = features['atr'] / features['close']
        
        # Market microstructure
        features['spread'] = (features['high'] - features['low']) / features['close']
        features['overnight_gap'] = (features['open'] - features['close'].shift(1)) / features['close'].shift(1)
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volume_lag_{lag}'] = features['volume'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'return_std_{window}'] = features['returns'].rolling(window).std()
            features[f'return_mean_{window}'] = features['returns'].rolling(window).mean()
            features[f'volume_std_{window}'] = features['volume'].rolling(window).std()
        
        return features.dropna()
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        current_positions: Dict[str, Any]
    ) -> Optional[MLSignal]:
        """
        Generate ML-based trading signal.
        
        Args:
            data: Historical data with features
            current_positions: Current open positions
            
        Returns:
            MLSignal if trade should be made, None otherwise
        """
        # Check if we need to retrain
        if self._should_retrain(data):
            self._retrain_models(data)
        
        # Prepare features
        features = self.prepare_features(data)
        if len(features) < self.feature_lookback:
            return None
        
        # Get current data
        current_time = features.index[-1]
        symbol = data.attrs.get('symbol', 'UNKNOWN')
        
        # Make predictions
        if self.use_ensemble:
            prediction = self.model.predict(features)
            
            direction = prediction.direction
            probability = prediction.direction_probability
            confidence = prediction.ensemble_confidence
            volatility_forecast = prediction.atr_forecast
            market_regime = prediction.market_regime
            risk_score = prediction.risk_score
            
        else:
            # Use individual models
            dir_pred = self.direction_model.predict(features)
            vol_pred = self.volatility_model.predict(features)
            regime_pred = self.regime_model.predict(features)
            
            direction = dir_pred.direction
            probability = dir_pred.probability
            confidence = dir_pred.confidence
            volatility_forecast = vol_pred.atr_prediction
            market_regime = regime_pred.current_regime
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(
                volatility_forecast,
                features['atr'].iloc[-1],
                regime_pred.confidence
            )
        
        # Apply filters
        if not self._should_trade(
            probability,
            confidence,
            market_regime,
            current_positions
        ):
            return None
        
        # Calculate position size adjustment
        position_size_adjustment = self._calculate_position_adjustment(
            volatility_forecast,
            features['atr'].iloc[-1],
            risk_score
        )
        
        # Create signal
        signal = MLSignal(
            timestamp=current_time,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            probability=probability,
            volatility_forecast=volatility_forecast,
            market_regime=market_regime,
            risk_score=risk_score,
            position_size_adjustment=position_size_adjustment,
            metadata={
                'features': features.iloc[-1].to_dict(),
                'model_type': 'ensemble' if self.use_ensemble else 'individual'
            }
        )
        
        # Store signal for analysis
        self.signal_history.append(signal)
        
        return signal
    
    def _should_retrain(self, data: pd.DataFrame) -> bool:
        """Check if models should be retrained."""
        if self.last_training_date is None:
            return True
        
        self.bars_since_training += 1
        return self.bars_since_training >= self.retrain_frequency
    
    def _retrain_models(self, data: pd.DataFrame):
        """Retrain ML models with recent data."""
        features = self.prepare_features(data)
        
        # Split data for training
        train_size = int(len(features) * 0.8)
        train_data = features.iloc[:train_size]
        
        if self.use_ensemble:
            self.model.fit(train_data, validate=True, n_splits=3)
        else:
            self.direction_model.fit(train_data, validate=True, n_splits=3)
            self.volatility_model.fit(train_data, validate=True, n_splits=3)
            self.regime_model.fit(train_data)
        
        self.last_training_date = data.index[-1]
        self.bars_since_training = 0
    
    def _should_trade(
        self,
        probability: float,
        confidence: float,
        market_regime: MarketRegime,
        current_positions: Dict[str, Any]
    ) -> bool:
        """Apply filters to determine if trade should be made."""
        # Check probability threshold
        if probability < self.direction_threshold:
            return False
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return False
        
        # Check market regime
        if self.regime_filter and market_regime not in self.allowed_regimes:
            return False
        
        # Check position limits
        if len(current_positions) >= self.risk_management.max_positions:
            return False
        
        return True
    
    def _calculate_risk_score(
        self,
        volatility_forecast: float,
        current_atr: float,
        regime_confidence: float
    ) -> float:
        """Calculate overall risk score."""
        # Volatility risk (higher volatility = higher risk)
        vol_risk = min(volatility_forecast / (current_atr + 1e-6), 2.0)
        
        # Regime uncertainty risk
        regime_risk = 1.0 - regime_confidence
        
        # Combined risk score
        risk_score = (vol_risk * 0.6 + regime_risk * 0.4)
        
        return np.clip(risk_score, 0.0, 1.0)
    
    def _calculate_position_adjustment(
        self,
        volatility_forecast: float,
        current_atr: float,
        risk_score: float
    ) -> float:
        """Calculate position size adjustment based on volatility and risk."""
        if not self.volatility_scaling:
            return 1.0
        
        # Inverse volatility scaling
        vol_ratio = current_atr / (volatility_forecast + 1e-6)
        vol_adjustment = np.clip(vol_ratio, 0.5, self.max_volatility_multiplier)
        
        # Risk-based adjustment
        risk_adjustment = 1.0 - (risk_score * 0.5)  # Max 50% reduction
        
        return vol_adjustment * risk_adjustment
    
    def calculate_position_size(
        self,
        signal: MLSignal,
        capital: float,
        price: float
    ) -> int:
        """
        Calculate position size based on ML signal and risk management.
        
        Args:
            signal: ML signal with predictions
            capital: Available capital
            price: Current asset price
            
        Returns:
            Number of shares to trade
        """
        # Base position size from Kelly Criterion or fixed risk
        if hasattr(self, 'model_performance') and self.model_performance:
            # Use Kelly Criterion with ML performance
            win_rate = self.model_performance.get('win_rate', 0.5)
            avg_win = self.model_performance.get('avg_win', 1.0)
            avg_loss = self.model_performance.get('avg_loss', 1.0)
            
            position_size = self.position_sizing.calculate_size(
                capital=capital,
                price=price,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss
            )
        else:
            # Use risk-based sizing
            risk_amount = capital * self.risk_per_trade
            stop_distance = price * signal.volatility_forecast
            position_size = int(risk_amount / stop_distance)
        
        # Apply ML-based adjustment
        adjusted_size = int(position_size * signal.position_size_adjustment)
        
        # Apply maximum position limit
        max_shares = int((capital * self.position_sizing.max_position) / price)
        
        return min(adjusted_size, max_shares)
    
    def update_performance(self, trade_results: Dict[str, Any]):
        """Update model performance metrics based on trade results."""
        if 'return' in trade_results:
            if 'returns' not in self.model_performance:
                self.model_performance['returns'] = []
            
            self.model_performance['returns'].append(trade_results['return'])
            
            # Update performance metrics
            returns = self.model_performance['returns']
            self.model_performance['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)
            self.model_performance['avg_win'] = np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
            self.model_performance['avg_loss'] = abs(np.mean([r for r in returns if r <= 0])) if any(r <= 0 for r in returns) else 0
            self.model_performance['sharpe_ratio'] = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-6)
    
    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of ML signals generated."""
        if not self.signal_history:
            return {}
        
        signals_df = pd.DataFrame([
            {
                'timestamp': s.timestamp,
                'direction': s.direction,
                'confidence': s.confidence,
                'probability': s.probability,
                'regime': s.market_regime.value,
                'risk_score': s.risk_score
            }
            for s in self.signal_history
        ])
        
        return {
            'total_signals': len(signals_df),
            'long_signals': (signals_df['direction'] == 1).sum(),
            'short_signals': (signals_df['direction'] == -1).sum(),
            'avg_confidence': signals_df['confidence'].mean(),
            'avg_probability': signals_df['probability'].mean(),
            'regime_distribution': signals_df['regime'].value_counts().to_dict(),
            'avg_risk_score': signals_df['risk_score'].mean()
        }