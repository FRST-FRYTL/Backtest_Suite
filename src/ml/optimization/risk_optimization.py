"""
Risk Management Optimization (Loop 4)

Optimizes risk management parameters including position sizing, stop losses, and portfolio allocation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import optuna
from scipy import stats
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class RiskOptimization:
    """
    Optimizes risk management parameters for trading strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk optimization
        
        Args:
            config: Risk optimization configuration
        """
        self.config = config
        self.risk_cache = {}
        
    def get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Get hyperparameters from Optuna trial for risk optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of risk management parameters
        """
        params = {
            'risk_management': {
                # Position sizing
                'position_sizing': {
                    'method': trial.suggest_categorical(
                        'position_sizing_method',
                        ['fixed', 'kelly', 'volatility_adjusted', 'risk_parity', 'dynamic']
                    ),
                    'base_position_size': trial.suggest_float('base_position_size', 0.1, 1.0),
                    'max_position_size': trial.suggest_float('max_position_size', 0.5, 1.0),
                    'min_position_size': trial.suggest_float('min_position_size', 0.01, 0.2),
                    'kelly_fraction': trial.suggest_float('kelly_fraction', 0.1, 0.5),
                    'volatility_lookback': trial.suggest_int('volatility_lookback', 10, 50),
                    'volatility_target': trial.suggest_float('volatility_target', 0.1, 0.3),
                },
                
                # Stop loss and take profit
                'stop_loss': {
                    'enabled': trial.suggest_categorical('stop_loss_enabled', [True, False]),
                    'method': trial.suggest_categorical(
                        'stop_loss_method',
                        ['fixed', 'atr', 'volatility', 'trailing', 'dynamic']
                    ) if trial.params.get('stop_loss_enabled', False) else None,
                    'fixed_stop': trial.suggest_float('fixed_stop', 0.01, 0.05) 
                        if trial.params.get('stop_loss_enabled', False) and trial.params.get('stop_loss_method', '') == 'fixed' else None,
                    'atr_multiplier': trial.suggest_float('atr_multiplier', 1.0, 3.0)
                        if trial.params.get('stop_loss_enabled', False) and trial.params.get('stop_loss_method', '') == 'atr' else None,
                    'trailing_distance': trial.suggest_float('trailing_distance', 0.01, 0.05)
                        if trial.params.get('stop_loss_enabled', False) and trial.params.get('stop_loss_method', '') == 'trailing' else None,
                },
                
                'take_profit': {
                    'enabled': trial.suggest_categorical('take_profit_enabled', [True, False]),
                    'method': trial.suggest_categorical(
                        'take_profit_method',
                        ['fixed', 'atr', 'risk_reward', 'dynamic']
                    ) if trial.params.get('take_profit_enabled', False) else None,
                    'fixed_target': trial.suggest_float('fixed_target', 0.02, 0.10)
                        if trial.params.get('take_profit_enabled', False) and trial.params.get('take_profit_method', '') == 'fixed' else None,
                    'risk_reward_ratio': trial.suggest_float('risk_reward_ratio', 1.5, 4.0)
                        if trial.params.get('take_profit_enabled', False) and trial.params.get('take_profit_method', '') == 'risk_reward' else None,
                },
                
                # Portfolio risk controls
                'portfolio_risk': {
                    'max_portfolio_risk': trial.suggest_float('max_portfolio_risk', 0.02, 0.10),
                    'max_correlation': trial.suggest_float('max_correlation', 0.5, 0.95),
                    'max_sector_exposure': trial.suggest_float('max_sector_exposure', 0.3, 0.7),
                    'max_drawdown_limit': trial.suggest_float('max_drawdown_limit', 0.10, 0.30),
                    'var_confidence': trial.suggest_float('var_confidence', 0.90, 0.99),
                    'cvar_multiplier': trial.suggest_float('cvar_multiplier', 1.0, 2.0),
                },
                
                # Dynamic risk adjustment
                'dynamic_adjustment': {
                    'enabled': trial.suggest_categorical('dynamic_risk_enabled', [True, False]),
                    'adjustment_method': trial.suggest_categorical(
                        'adjustment_method',
                        ['drawdown_based', 'volatility_based', 'performance_based', 'regime_based']
                    ) if trial.params.get('dynamic_risk_enabled', False) else None,
                    'adjustment_factor': trial.suggest_float('adjustment_factor', 0.5, 2.0)
                        if trial.params.get('dynamic_risk_enabled', False) else None,
                    'adjustment_speed': trial.suggest_categorical('adjustment_speed', ['fast', 'medium', 'slow'])
                        if trial.params.get('dynamic_risk_enabled', False) else None,
                },
                
                # Risk metrics weights
                'risk_weights': {
                    'sharpe_weight': trial.suggest_float('sharpe_weight', 0.1, 0.5),
                    'sortino_weight': trial.suggest_float('sortino_weight', 0.1, 0.5),
                    'calmar_weight': trial.suggest_float('calmar_weight', 0.1, 0.5),
                    'max_drawdown_weight': trial.suggest_float('max_drawdown_weight', 0.1, 0.5),
                }
            }
        }
        
        # Normalize weights
        weights = params['risk_management']['risk_weights']
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight
        
        return params
    
    def evaluate(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """
        Evaluate risk management configuration
        
        Args:
            data: Market data with signals
            params: Complete parameter set including risk params
            
        Returns:
            Risk-adjusted performance metric (higher is better)
        """
        try:
            # Generate trading signals (simplified - would come from previous loops)
            signals = self._generate_signals(data, params)
            
            # Apply position sizing
            positions = self._apply_position_sizing(data, signals, params.get('risk_management', {}))
            
            # Apply stop loss and take profit
            positions = self._apply_stops(data, positions, params.get('risk_management', {}))
            
            # Apply portfolio risk controls
            positions = self._apply_portfolio_controls(data, positions, params.get('risk_management', {}))
            
            # Apply dynamic adjustments
            if params.get('risk_management', {}).get('dynamic_adjustment', {}).get('enabled', False):
                positions = self._apply_dynamic_adjustments(data, positions, params.get('risk_management', {}))
            
            # Calculate risk-adjusted performance
            score = self._calculate_risk_adjusted_score(data, positions, params.get('risk_management', {}))
            
            return score
            
        except Exception as e:
            logger.error(f"Error in risk evaluation: {str(e)}")
            return -np.inf
    
    def _generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Generate simple trading signals for evaluation"""
        # This would normally come from optimized model predictions
        # Using simple momentum signal for demonstration
        returns = data['close'].pct_change()
        momentum = returns.rolling(20).mean()
        signals = (momentum > 0).astype(float) * 2 - 1
        
        # Add some randomness to test risk management
        noise = np.random.normal(0, 0.1, len(signals))
        signals = signals + noise
        signals = np.clip(signals, -1, 1)
        
        return pd.Series(signals, index=data.index)
    
    def _apply_position_sizing(self, data: pd.DataFrame, signals: pd.Series, 
                              risk_params: Dict[str, Any]) -> pd.Series:
        """Apply position sizing based on risk parameters"""
        sizing_params = risk_params.get('position_sizing', {})
        method = sizing_params.get('method', 'fixed')
        
        base_size = sizing_params.get('base_position_size', 0.5)
        max_size = sizing_params.get('max_position_size', 1.0)
        min_size = sizing_params.get('min_position_size', 0.1)
        
        if method == 'fixed':
            positions = signals * base_size
            
        elif method == 'kelly':
            # Simplified Kelly criterion
            kelly_fraction = sizing_params.get('kelly_fraction', 0.25)
            
            # Estimate win probability and win/loss ratio
            returns = data['close'].pct_change()
            signal_returns = signals.shift(1) * returns
            
            wins = signal_returns[signal_returns > 0]
            losses = signal_returns[signal_returns < 0]
            
            if len(wins) > 0 and len(losses) > 0:
                win_prob = len(wins) / len(signal_returns[signal_returns != 0])
                avg_win = wins.mean()
                avg_loss = abs(losses.mean())
                
                # Kelly formula: f = p - q/b
                # where p = win probability, q = loss probability, b = win/loss ratio
                if avg_loss > 0:
                    kelly_size = win_prob - (1 - win_prob) / (avg_win / avg_loss)
                    kelly_size = max(0, kelly_size) * kelly_fraction
                else:
                    kelly_size = base_size
            else:
                kelly_size = base_size
            
            positions = signals * kelly_size
            
        elif method == 'volatility_adjusted':
            # Size inversely proportional to volatility
            lookback = sizing_params.get('volatility_lookback', 20)
            target_vol = sizing_params.get('volatility_target', 0.15)
            
            returns = data['close'].pct_change()
            volatility = returns.rolling(lookback).std() * np.sqrt(252)
            
            vol_scalar = target_vol / (volatility + 1e-6)
            vol_scalar = vol_scalar.clip(0.5, 2.0)  # Limit scaling factor
            
            positions = signals * base_size * vol_scalar
            
        elif method == 'risk_parity':
            # Equal risk contribution (simplified for single asset)
            lookback = sizing_params.get('volatility_lookback', 20)
            
            returns = data['close'].pct_change()
            volatility = returns.rolling(lookback).std()
            
            # Inverse volatility weighting
            inv_vol = 1 / (volatility + 1e-6)
            normalized_inv_vol = inv_vol / inv_vol.rolling(lookback).mean()
            
            positions = signals * base_size * normalized_inv_vol
            
        else:  # dynamic
            # Combine multiple sizing methods
            # Start with volatility adjustment
            lookback = sizing_params.get('volatility_lookback', 20)
            returns = data['close'].pct_change()
            volatility = returns.rolling(lookback).std() * np.sqrt(252)
            
            vol_scalar = 0.15 / (volatility + 1e-6)
            vol_scalar = vol_scalar.clip(0.5, 2.0)
            
            # Add performance adjustment
            rolling_returns = (1 + returns).rolling(lookback).apply(lambda x: x.prod()) - 1
            perf_scalar = 1 + rolling_returns.clip(-0.5, 0.5)
            
            positions = signals * base_size * vol_scalar * perf_scalar
        
        # Apply size limits
        positions = positions.clip(-max_size, max_size)
        positions[abs(positions) < min_size] = 0
        
        return positions
    
    def _apply_stops(self, data: pd.DataFrame, positions: pd.Series, 
                    risk_params: Dict[str, Any]) -> pd.Series:
        """Apply stop loss and take profit logic"""
        stop_params = risk_params.get('stop_loss', {})
        profit_params = risk_params.get('take_profit', {})
        
        adjusted_positions = positions.copy()
        
        # Track entry prices and position changes
        position_changes = positions.diff().fillna(positions)
        entry_prices = pd.Series(index=data.index, dtype=float)
        
        current_position = 0
        current_entry_price = 0
        highest_price = 0
        
        for i in range(len(data)):
            price = data['close'].iloc[i]
            
            # Check for position change
            if position_changes.iloc[i] != 0:
                current_position = positions.iloc[i]
                current_entry_price = price
                entry_prices.iloc[i] = price
                highest_price = price
            
            # Apply stop loss
            if stop_params.get('enabled', False) and current_position != 0:
                stop_hit = False
                
                if stop_params.get('method') == 'fixed':
                    stop_distance = stop_params.get('fixed_stop', 0.02)
                    if current_position > 0:  # Long position
                        stop_price = current_entry_price * (1 - stop_distance)
                        stop_hit = price <= stop_price
                    else:  # Short position
                        stop_price = current_entry_price * (1 + stop_distance)
                        stop_hit = price >= stop_price
                        
                elif stop_params.get('method') == 'trailing':
                    trailing_distance = stop_params.get('trailing_distance', 0.02)
                    if current_position > 0:  # Long position
                        highest_price = max(highest_price, price)
                        stop_price = highest_price * (1 - trailing_distance)
                        stop_hit = price <= stop_price
                    else:  # Short position
                        highest_price = min(highest_price, price)
                        stop_price = highest_price * (1 + trailing_distance)
                        stop_hit = price >= stop_price
                
                if stop_hit:
                    adjusted_positions.iloc[i:] = 0
                    current_position = 0
            
            # Apply take profit
            if profit_params.get('enabled', False) and current_position != 0:
                profit_hit = False
                
                if profit_params.get('method') == 'fixed':
                    target_distance = profit_params.get('fixed_target', 0.05)
                    if current_position > 0:  # Long position
                        target_price = current_entry_price * (1 + target_distance)
                        profit_hit = price >= target_price
                    else:  # Short position
                        target_price = current_entry_price * (1 - target_distance)
                        profit_hit = price <= target_price
                
                if profit_hit:
                    adjusted_positions.iloc[i:] = 0
                    current_position = 0
        
        return adjusted_positions
    
    def _apply_portfolio_controls(self, data: pd.DataFrame, positions: pd.Series,
                                 risk_params: Dict[str, Any]) -> pd.Series:
        """Apply portfolio-level risk controls"""
        portfolio_params = risk_params.get('portfolio_risk', {})
        
        # Calculate rolling metrics
        returns = data['close'].pct_change()
        position_returns = positions.shift(1) * returns
        
        # Maximum drawdown control
        max_dd_limit = portfolio_params.get('max_drawdown_limit', 0.20)
        cumulative_returns = (1 + position_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = 1 - cumulative_returns / running_max
        
        # Reduce positions when approaching drawdown limit
        dd_scalar = 1 - (drawdown / max_dd_limit).clip(0, 1)
        adjusted_positions = positions * dd_scalar
        
        # VaR and CVaR controls
        var_confidence = portfolio_params.get('var_confidence', 0.95)
        cvar_multiplier = portfolio_params.get('cvar_multiplier', 1.5)
        
        # Calculate rolling VaR
        lookback = 252  # 1 year
        rolling_var = position_returns.rolling(lookback).quantile(1 - var_confidence)
        
        # Reduce positions if recent returns approach VaR limit
        recent_returns = position_returns.rolling(20).mean()
        var_scalar = 1 - (recent_returns / (rolling_var * cvar_multiplier)).clip(0, 1)
        adjusted_positions = adjusted_positions * var_scalar
        
        return adjusted_positions
    
    def _apply_dynamic_adjustments(self, data: pd.DataFrame, positions: pd.Series,
                                  risk_params: Dict[str, Any]) -> pd.Series:
        """Apply dynamic risk adjustments"""
        dynamic_params = risk_params.get('dynamic_adjustment', {})
        method = dynamic_params.get('adjustment_method', 'drawdown_based')
        factor = dynamic_params.get('adjustment_factor', 1.0)
        speed = dynamic_params.get('adjustment_speed', 'medium')
        
        # Set adjustment window based on speed
        window_map = {'fast': 10, 'medium': 20, 'slow': 50}
        window = window_map.get(speed, 20)
        
        returns = data['close'].pct_change()
        position_returns = positions.shift(1) * returns
        
        if method == 'drawdown_based':
            # Reduce risk during drawdowns
            cumulative_returns = (1 + position_returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = 1 - cumulative_returns / running_max
            
            # Smoothed adjustment based on drawdown
            dd_adjustment = 1 - (drawdown * factor).clip(0, 0.5)
            dd_adjustment = dd_adjustment.rolling(window).mean()
            
            adjusted_positions = positions * dd_adjustment
            
        elif method == 'volatility_based':
            # Adjust based on volatility regime
            current_vol = returns.rolling(window).std()
            median_vol = current_vol.rolling(252).median()
            
            vol_ratio = current_vol / median_vol
            vol_adjustment = 1 / (1 + (vol_ratio - 1) * factor)
            vol_adjustment = vol_adjustment.clip(0.5, 1.5)
            
            adjusted_positions = positions * vol_adjustment
            
        elif method == 'performance_based':
            # Increase risk after good performance, decrease after poor
            rolling_returns = position_returns.rolling(window).mean()
            rolling_sharpe = rolling_returns / (position_returns.rolling(window).std() + 1e-6)
            
            # Normalize Sharpe to adjustment factor
            perf_adjustment = 1 + (rolling_sharpe / 2).clip(-0.5, 0.5) * factor
            
            adjusted_positions = positions * perf_adjustment
            
        else:  # regime_based
            # Would use regime detection from previous loop
            # Simplified: use volatility regimes
            vol = returns.rolling(20).std()
            vol_percentile = vol.rolling(252).rank(pct=True)
            
            regime_adjustment = np.where(
                vol_percentile < 0.33, 1.2,  # Low vol: increase risk
                np.where(vol_percentile > 0.67, 0.8, 1.0)  # High vol: decrease risk
            )
            regime_adjustment = pd.Series(regime_adjustment, index=data.index)
            regime_adjustment = regime_adjustment.rolling(window).mean()
            
            adjusted_positions = positions * regime_adjustment * factor
        
        return adjusted_positions
    
    def _calculate_risk_adjusted_score(self, data: pd.DataFrame, positions: pd.Series,
                                      risk_params: Dict[str, Any]) -> float:
        """Calculate comprehensive risk-adjusted performance score"""
        returns = data['close'].pct_change()
        position_returns = positions.shift(1) * returns
        position_returns = position_returns.dropna()
        
        if len(position_returns) < 50:
            return -np.inf
        
        # Get risk metric weights
        weights = risk_params.get('risk_weights', {
            'sharpe_weight': 0.25,
            'sortino_weight': 0.25,
            'calmar_weight': 0.25,
            'max_drawdown_weight': 0.25
        })
        
        scores = {}
        
        # 1. Sharpe Ratio
        mean_return = position_returns.mean() * 252
        std_return = position_returns.std() * np.sqrt(252)
        sharpe = mean_return / (std_return + 1e-6)
        scores['sharpe'] = max(0, min(sharpe / 2, 1))  # Normalize to [0, 1]
        
        # 2. Sortino Ratio
        downside_returns = position_returns[position_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = mean_return / (downside_std + 1e-6)
        scores['sortino'] = max(0, min(sortino / 3, 1))  # Normalize to [0, 1]
        
        # 3. Calmar Ratio
        cumulative_returns = (1 + position_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (running_max - cumulative_returns) / running_max
        max_drawdown = drawdown.max()
        
        calmar = mean_return / (max_drawdown + 1e-6)
        scores['calmar'] = max(0, min(calmar / 3, 1))  # Normalize to [0, 1]
        
        # 4. Maximum Drawdown (inverse)
        scores['max_drawdown'] = 1 - min(max_drawdown * 2, 1)  # Penalize large drawdowns
        
        # 5. Additional risk metrics
        # Win rate
        winning_days = (position_returns > 0).sum()
        total_days = (position_returns != 0).sum()
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        # Profit factor
        gross_profits = position_returns[position_returns > 0].sum()
        gross_losses = abs(position_returns[position_returns < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else 2.0
        
        # Risk-adjusted returns
        var_95 = position_returns.quantile(0.05)
        cvar_95 = position_returns[position_returns <= var_95].mean()
        risk_adjusted_return = mean_return / abs(cvar_95) if cvar_95 < 0 else 2.0
        
        # Combine all metrics
        weighted_score = (
            weights.get('sharpe_weight', 0.25) * scores['sharpe'] +
            weights.get('sortino_weight', 0.25) * scores['sortino'] +
            weights.get('calmar_weight', 0.25) * scores['calmar'] +
            weights.get('max_drawdown_weight', 0.25) * scores['max_drawdown']
        )
        
        # Bonus for good secondary metrics
        bonus = (
            0.1 * win_rate +
            0.1 * min(profit_factor / 2, 1) +
            0.1 * min(risk_adjusted_return / 2, 1)
        )
        
        total_score = weighted_score + bonus
        
        # Penalty for excessive risk
        if max_drawdown > 0.3:
            total_score *= 0.5
        if std_return > 0.5:
            total_score *= 0.8
        
        return min(total_score, 1.0)
    
    def optimize_portfolio_weights(self, returns_df: pd.DataFrame, 
                                  risk_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Optimize portfolio weights for multiple assets
        
        Args:
            returns_df: DataFrame of asset returns
            risk_params: Risk parameters
            
        Returns:
            Dictionary of optimal weights
        """
        n_assets = len(returns_df.columns)
        
        # Calculate covariance matrix
        cov_matrix = returns_df.cov() * 252  # Annualized
        mean_returns = returns_df.mean() * 252
        
        # Define objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = portfolio_return / portfolio_vol
            return -sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Add correlation constraint
        max_corr = risk_params.get('portfolio_risk', {}).get('max_correlation', 0.8)
        corr_matrix = returns_df.corr()
        
        def correlation_constraint(weights):
            weighted_corr = 0
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    weighted_corr += weights[i] * weights[j] * corr_matrix.iloc[i, j]
            return max_corr - weighted_corr
        
        constraints.append({'type': 'ineq', 'fun': correlation_constraint})
        
        # Bounds
        bounds = tuple((0, 0.4) for _ in range(n_assets))  # Max 40% per asset
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
        else:
            # Fall back to equal weighting
            weights = np.array([1/n_assets] * n_assets)
        
        return dict(zip(returns_df.columns, weights))