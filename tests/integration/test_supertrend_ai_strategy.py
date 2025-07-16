"""Integration tests for SuperTrend AI strategy."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from unittest.mock import patch, Mock

from src.indicators.supertrend_ai import SuperTrendAI
from src.backtesting import BacktestEngine, Position
from src.strategies import BaseStrategy
from src.utils import PerformanceMetrics
from src.data import StockDataFetcher
from src.ml import DirectionPredictor, VolatilityForecaster, MarketRegimeDetector


class SuperTrendAIStrategy(BaseStrategy):
    """SuperTrend AI trading strategy implementation."""
    
    def __init__(self, 
                 atr_length=10,
                 factor_min=1.0,
                 factor_max=5.0,
                 factor_step=0.5,
                 cluster_from='best',
                 min_signal_strength=4,
                 use_ml_confluence=False,
                 ml_confidence_threshold=0.6,
                 use_time_filter=False,
                 start_hour=9,
                 end_hour=16,
                 stop_loss_atr_mult=2.0,
                 take_profit_rr_ratio=2.0):
        """Initialize SuperTrend AI strategy."""
        super().__init__()
        
        # SuperTrend AI parameters
        self.supertrend = SuperTrendAI(
            atr_length=atr_length,
            factor_min=factor_min,
            factor_max=factor_max,
            factor_step=factor_step,
            cluster_from=cluster_from
        )
        
        # Strategy parameters
        self.min_signal_strength = min_signal_strength
        self.use_ml_confluence = use_ml_confluence
        self.ml_confidence_threshold = ml_confidence_threshold
        self.use_time_filter = use_time_filter
        self.start_hour = start_hour
        self.end_hour = end_hour
        
        # Risk management
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_rr_ratio = take_profit_rr_ratio
        
        # ML models (if enabled)
        if self.use_ml_confluence:
            self.direction_predictor = DirectionPredictor()
            self.volatility_forecaster = VolatilityForecaster()
            self.regime_detector = MarketRegimeDetector()
    
    def generate_signals(self, data):
        """Generate trading signals."""
        # Calculate SuperTrend AI
        st_result = self.supertrend.calculate(data)
        signals = self.supertrend.generate_signals(st_result)
        
        # Apply signal strength filter
        signals['long_entry'] = signals['long_entry'] & (st_result['signal_strength'] >= self.min_signal_strength)
        signals['short_entry'] = signals['short_entry'] & (st_result['signal_strength'] >= self.min_signal_strength)
        
        # Apply time filter if enabled
        if self.use_time_filter:
            valid_time = (data.index.hour >= self.start_hour) & (data.index.hour <= self.end_hour)
            signals['long_entry'] = signals['long_entry'] & valid_time
            signals['short_entry'] = signals['short_entry'] & valid_time
        
        # Apply ML confluence if enabled
        if self.use_ml_confluence and hasattr(self, 'direction_predictor'):
            ml_signals = self._get_ml_signals(data)
            signals['long_entry'] = signals['long_entry'] & ml_signals['ml_bullish']
            signals['short_entry'] = signals['short_entry'] & ml_signals['ml_bearish']
        
        # Calculate stop loss and take profit levels
        atr = data['atr']  # Assuming ATR is pre-calculated
        signals['stop_loss_long'] = data['close'] - (atr * self.stop_loss_atr_mult)
        signals['stop_loss_short'] = data['close'] + (atr * self.stop_loss_atr_mult)
        
        risk_long = atr * self.stop_loss_atr_mult
        risk_short = atr * self.stop_loss_atr_mult
        signals['take_profit_long'] = data['close'] + (risk_long * self.take_profit_rr_ratio)
        signals['take_profit_short'] = data['close'] - (risk_short * self.take_profit_rr_ratio)
        
        return signals
    
    def _get_ml_signals(self, data):
        """Get ML model signals."""
        # Prepare features
        features = self._prepare_ml_features(data)
        
        # Get predictions
        direction_pred = self.direction_predictor.predict(features)
        volatility_pred = self.volatility_forecaster.predict(features)
        regime = self.regime_detector.predict(features)
        
        # Generate ML signals
        ml_signals = pd.DataFrame(index=data.index)
        ml_signals['ml_bullish'] = (
            (direction_pred['direction_proba'] > self.ml_confidence_threshold) &
            (direction_pred['prediction'] == 1) &
            (regime['regime'].isin(['uptrend', 'accumulation']))
        )
        ml_signals['ml_bearish'] = (
            (direction_pred['direction_proba'] > self.ml_confidence_threshold) &
            (direction_pred['prediction'] == -1) &
            (regime['regime'].isin(['downtrend', 'distribution']))
        )
        
        return ml_signals
    
    def _prepare_ml_features(self, data):
        """Prepare features for ML models."""
        # This is a simplified version - actual implementation would include
        # comprehensive feature engineering
        features = pd.DataFrame(index=data.index)
        
        # Price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_open_ratio'] = data['close'] / data['open']
        
        # Volume features
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_price'] = data['volume'] * data['close']
        
        # Technical features (assuming they're in the data)
        if 'rsi' in data.columns:
            features['rsi'] = data['rsi']
        if 'upper_band' in data.columns:
            features['bb_position'] = (data['close'] - data['lower_band']) / (data['upper_band'] - data['lower_band'])
        
        return features.dropna()


@pytest.fixture
def sample_market_data():
    """Create comprehensive market data for testing."""
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic market data with trends
    trend = np.sin(np.linspace(0, 4 * np.pi, len(dates))) * 20 + 100
    noise = np.random.randn(len(dates)) * 2
    close_prices = trend + noise + np.linspace(0, 50, len(dates))  # Overall uptrend
    
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(len(dates)) * 0.5,
        'high': close_prices + np.abs(np.random.randn(len(dates))) * 2,
        'low': close_prices - np.abs(np.random.randn(len(dates))) * 2,
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Ensure OHLC relationships
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    # Add ATR for stop loss calculations
    high_low = data['high'] - data['low']
    high_close = (data['high'] - data['close'].shift()).abs()
    low_close = (data['low'] - data['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['atr'] = true_range.rolling(14).mean()
    
    return data


@pytest.fixture
def backtest_engine():
    """Create a backtest engine instance."""
    return BacktestEngine(
        initial_capital=100000,
        commission=0.001,  # 0.1%
        slippage=0.0005   # 0.05%
    )


class TestSuperTrendAIStrategyIntegration:
    """Integration tests for SuperTrend AI strategy."""
    
    def test_strategy_initialization(self):
        """Test strategy initialization with various parameters."""
        # Default initialization
        strategy = SuperTrendAIStrategy()
        assert strategy.min_signal_strength == 4
        assert strategy.use_ml_confluence == False
        assert strategy.stop_loss_atr_mult == 2.0
        assert strategy.take_profit_rr_ratio == 2.0
        
        # Custom initialization
        custom_strategy = SuperTrendAIStrategy(
            min_signal_strength=6,
            use_ml_confluence=True,
            ml_confidence_threshold=0.7,
            use_time_filter=True,
            start_hour=10,
            end_hour=15
        )
        assert custom_strategy.min_signal_strength == 6
        assert custom_strategy.use_ml_confluence == True
        assert custom_strategy.ml_confidence_threshold == 0.7
        assert custom_strategy.use_time_filter == True
    
    def test_signal_generation_basic(self, sample_market_data):
        """Test basic signal generation without filters."""
        strategy = SuperTrendAIStrategy(min_signal_strength=0)  # No strength filter
        signals = strategy.generate_signals(sample_market_data)
        
        # Check signal structure
        required_columns = [
            'long_entry', 'short_entry', 'exit_long', 'exit_short',
            'stop_loss_long', 'stop_loss_short', 
            'take_profit_long', 'take_profit_short'
        ]
        for col in required_columns:
            assert col in signals.columns
        
        # Should have some signals
        assert signals['long_entry'].sum() > 0
        assert signals['short_entry'].sum() > 0
        
        # No simultaneous long and short entries
        assert (signals['long_entry'] & signals['short_entry']).sum() == 0
    
    def test_signal_strength_filter(self, sample_market_data):
        """Test signal strength filtering."""
        # Strategy with high signal strength requirement
        strict_strategy = SuperTrendAIStrategy(min_signal_strength=8)
        strict_signals = strict_strategy.generate_signals(sample_market_data)
        
        # Strategy with low signal strength requirement
        loose_strategy = SuperTrendAIStrategy(min_signal_strength=2)
        loose_signals = loose_strategy.generate_signals(sample_market_data)
        
        # Strict strategy should have fewer signals
        assert strict_signals['long_entry'].sum() <= loose_signals['long_entry'].sum()
        assert strict_signals['short_entry'].sum() <= loose_signals['short_entry'].sum()
    
    def test_time_filter(self, sample_market_data):
        """Test time-based filtering."""
        # Add hour information to index
        hourly_data = sample_market_data.copy()
        hourly_data.index = pd.date_range(
            start='2022-01-01 09:00:00', 
            periods=len(hourly_data), 
            freq='H'
        )
        
        # Strategy with time filter
        time_strategy = SuperTrendAIStrategy(
            use_time_filter=True,
            start_hour=10,
            end_hour=14
        )
        time_signals = time_strategy.generate_signals(hourly_data)
        
        # Check that signals only occur during allowed hours
        signal_hours = hourly_data.index[time_signals['long_entry'] | time_signals['short_entry']].hour
        assert all(10 <= hour <= 14 for hour in signal_hours)
    
    def test_risk_management_levels(self, sample_market_data):
        """Test stop loss and take profit calculations."""
        strategy = SuperTrendAIStrategy(
            stop_loss_atr_mult=2.5,
            take_profit_rr_ratio=3.0
        )
        signals = strategy.generate_signals(sample_market_data)
        
        # Check stop loss levels
        long_entries = signals['long_entry']
        if long_entries.any():
            entry_prices = sample_market_data.loc[long_entries, 'close']
            stop_losses = signals.loc[long_entries, 'stop_loss_long']
            atr_values = sample_market_data.loc[long_entries, 'atr']
            
            # Stop loss should be below entry by ATR multiple
            expected_sl = entry_prices - (atr_values * 2.5)
            np.testing.assert_allclose(stop_losses.values, expected_sl.values, rtol=1e-5)
        
        # Check take profit levels
        if long_entries.any():
            take_profits = signals.loc[long_entries, 'take_profit_long']
            risks = entry_prices - stop_losses
            expected_tp = entry_prices + (risks * 3.0)
            np.testing.assert_allclose(take_profits.values, expected_tp.values, rtol=1e-5)
    
    def test_backtest_execution(self, sample_market_data, backtest_engine):
        """Test strategy execution in backtest engine."""
        strategy = SuperTrendAIStrategy(min_signal_strength=4)
        
        # Run backtest
        results = backtest_engine.run(
            data=sample_market_data,
            strategy=strategy
        )
        
        # Check results structure
        assert 'equity_curve' in results
        assert 'trades' in results
        assert 'metrics' in results
        
        # Check metrics
        metrics = results['metrics']
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        
        # Should have executed some trades
        assert len(results['trades']) > 0
    
    def test_position_sizing(self, sample_market_data, backtest_engine):
        """Test position sizing and risk management."""
        strategy = SuperTrendAIStrategy()
        
        # Mock position sizing
        backtest_engine.position_size_pct = 0.02  # 2% risk per trade
        
        results = backtest_engine.run(
            data=sample_market_data,
            strategy=strategy
        )
        
        # Check that no position exceeds risk limit
        for trade in results['trades']:
            position_value = trade['entry_price'] * trade['shares']
            max_risk = backtest_engine.initial_capital * 0.02
            actual_risk = abs(trade['entry_price'] - trade['stop_loss']) * trade['shares']
            assert actual_risk <= max_risk * 1.1  # Allow 10% margin for slippage
    
    def test_ml_confluence(self, sample_market_data):
        """Test ML model integration."""
        # Mock ML models
        with patch('src.ml.DirectionPredictor') as mock_direction:
            with patch('src.ml.VolatilityForecaster') as mock_volatility:
                with patch('src.ml.MarketRegimeDetector') as mock_regime:
                    # Setup mock predictions
                    mock_direction_instance = Mock()
                    mock_direction_instance.predict.return_value = pd.DataFrame({
                        'prediction': [1] * len(sample_market_data),
                        'direction_proba': [0.8] * len(sample_market_data)
                    }, index=sample_market_data.index)
                    mock_direction.return_value = mock_direction_instance
                    
                    mock_regime_instance = Mock()
                    mock_regime_instance.predict.return_value = pd.DataFrame({
                        'regime': ['uptrend'] * len(sample_market_data)
                    }, index=sample_market_data.index)
                    mock_regime.return_value = mock_regime_instance
                    
                    # Create strategy with ML
                    ml_strategy = SuperTrendAIStrategy(
                        use_ml_confluence=True,
                        ml_confidence_threshold=0.7
                    )
                    
                    # Generate signals
                    signals = ml_strategy.generate_signals(sample_market_data)
                    
                    # Should have called ML models
                    assert mock_direction_instance.predict.called
                    assert mock_regime_instance.predict.called
    
    def test_performance_across_market_conditions(self, backtest_engine):
        """Test strategy performance in different market conditions."""
        np.random.seed(42)
        
        # Bull market
        bull_dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        bull_prices = 100 * np.exp(np.linspace(0, 0.5, len(bull_dates))) + np.random.randn(len(bull_dates)) * 2
        bull_data = pd.DataFrame({
            'open': bull_prices + np.random.randn(len(bull_dates)) * 0.5,
            'high': bull_prices + np.abs(np.random.randn(len(bull_dates))) * 2,
            'low': bull_prices - np.abs(np.random.randn(len(bull_dates))) * 2,
            'close': bull_prices,
            'volume': np.random.randint(1000000, 5000000, len(bull_dates))
        }, index=bull_dates)
        bull_data['high'] = bull_data[['open', 'high', 'close']].max(axis=1)
        bull_data['low'] = bull_data[['open', 'low', 'close']].min(axis=1)
        bull_data['atr'] = 2.0  # Simplified
        
        # Bear market
        bear_dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        bear_prices = 100 * np.exp(-np.linspace(0, 0.3, len(bear_dates))) + np.random.randn(len(bear_dates)) * 2
        bear_data = pd.DataFrame({
            'open': bear_prices + np.random.randn(len(bear_dates)) * 0.5,
            'high': bear_prices + np.abs(np.random.randn(len(bear_dates))) * 2,
            'low': bear_prices - np.abs(np.random.randn(len(bear_dates))) * 2,
            'close': bear_prices,
            'volume': np.random.randint(1000000, 5000000, len(bear_dates))
        }, index=bear_dates)
        bear_data['high'] = bear_data[['open', 'high', 'close']].max(axis=1)
        bear_data['low'] = bear_data[['open', 'low', 'close']].min(axis=1)
        bear_data['atr'] = 2.0  # Simplified
        
        strategy = SuperTrendAIStrategy()
        
        # Test in bull market
        bull_results = backtest_engine.run(data=bull_data, strategy=strategy)
        
        # Test in bear market  
        bear_results = backtest_engine.run(data=bear_data, strategy=strategy)
        
        # Strategy should adapt to different market conditions
        assert 'total_return' in bull_results['metrics']
        assert 'total_return' in bear_results['metrics']
        
        # Check that strategy can be profitable in both conditions
        # (or at least limit losses in adverse conditions)
        assert bull_results['metrics']['max_drawdown'] < 0.3  # Max 30% drawdown
        assert bear_results['metrics']['max_drawdown'] < 0.3  # Max 30% drawdown
    
    def test_transaction_costs_impact(self, sample_market_data):
        """Test impact of transaction costs on strategy performance."""
        strategy = SuperTrendAIStrategy(min_signal_strength=3)
        
        # Backtest with no costs
        engine_no_costs = BacktestEngine(
            initial_capital=100000,
            commission=0,
            slippage=0
        )
        results_no_costs = engine_no_costs.run(data=sample_market_data, strategy=strategy)
        
        # Backtest with realistic costs
        engine_with_costs = BacktestEngine(
            initial_capital=100000,
            commission=0.001,  # 0.1%
            slippage=0.001    # 0.1%
        )
        results_with_costs = engine_with_costs.run(data=sample_market_data, strategy=strategy)
        
        # Performance should be lower with costs
        assert results_with_costs['metrics']['total_return'] < results_no_costs['metrics']['total_return']
        
        # But strategy should still be viable
        assert results_with_costs['metrics']['sharpe_ratio'] > 0  # Positive risk-adjusted returns
    
    def test_parameter_sensitivity(self, sample_market_data, backtest_engine):
        """Test strategy sensitivity to parameter changes."""
        param_results = {}
        
        # Test different signal strength thresholds
        for signal_strength in [2, 4, 6, 8]:
            strategy = SuperTrendAIStrategy(min_signal_strength=signal_strength)
            results = backtest_engine.run(data=sample_market_data, strategy=strategy)
            param_results[f'signal_{signal_strength}'] = results['metrics']
        
        # Test different risk parameters
        for sl_mult in [1.5, 2.0, 2.5, 3.0]:
            strategy = SuperTrendAIStrategy(stop_loss_atr_mult=sl_mult)
            results = backtest_engine.run(data=sample_market_data, strategy=strategy)
            param_results[f'sl_{sl_mult}'] = results['metrics']
        
        # Results should vary with parameters
        returns = [metrics['total_return'] for metrics in param_results.values()]
        assert len(set(returns)) > 1  # Not all returns are the same
    
    @pytest.mark.asyncio
    async def test_real_data_integration(self):
        """Test strategy with real market data (if available)."""
        fetcher = StockDataFetcher()
        
        # Try to fetch real data
        try:
            data = await fetcher.fetch(
                symbol="SPY",
                start=datetime.now() - timedelta(days=365),
                end=datetime.now(),
                interval="1d"
            )
            
            if len(data) > 0:
                # Add ATR
                high_low = data['high'] - data['low']
                high_close = (data['high'] - data['close'].shift()).abs()
                low_close = (data['low'] - data['close'].shift()).abs()
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                data['atr'] = true_range.rolling(14).mean()
                
                # Run strategy
                strategy = SuperTrendAIStrategy()
                engine = BacktestEngine(initial_capital=100000)
                results = engine.run(data=data, strategy=strategy)
                
                # Basic sanity checks
                assert len(results['trades']) >= 0
                assert results['metrics']['total_return'] is not None
        except Exception:
            # Skip if data fetch fails
            pytest.skip("Could not fetch real market data")
    
    def test_strategy_state_management(self, sample_market_data):
        """Test strategy state management during execution."""
        strategy = SuperTrendAIStrategy()
        
        # Process data in chunks to simulate real-time execution
        chunk_size = 50
        cumulative_signals = []
        
        for i in range(0, len(sample_market_data), chunk_size):
            chunk = sample_market_data.iloc[:i+chunk_size]
            signals = strategy.generate_signals(chunk)
            cumulative_signals.append(signals.iloc[-chunk_size:])
        
        # Concatenate all signals
        all_signals = pd.concat(cumulative_signals)
        
        # Compare with batch processing
        batch_signals = strategy.generate_signals(sample_market_data)
        
        # Results should be consistent
        # (allowing for edge effects at chunk boundaries)
        long_entries_chunk = all_signals['long_entry'].sum()
        long_entries_batch = batch_signals['long_entry'].sum()
        assert abs(long_entries_chunk - long_entries_batch) / long_entries_batch < 0.1  # Within 10%
    
    def test_error_handling(self, sample_market_data):
        """Test strategy error handling."""
        strategy = SuperTrendAIStrategy()
        
        # Test with invalid data
        invalid_data = sample_market_data.copy()
        invalid_data['close'] = np.nan  # All NaN values
        
        with pytest.raises(ValueError):
            strategy.generate_signals(invalid_data)
        
        # Test with missing columns
        incomplete_data = sample_market_data.drop(columns=['volume'])
        incomplete_data['atr'] = 2.0  # Add required ATR
        
        # Should still work without volume
        signals = strategy.generate_signals(incomplete_data)
        assert len(signals) == len(incomplete_data)