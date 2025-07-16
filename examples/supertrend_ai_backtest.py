"""
SuperTrend AI Strategy Backtest Script

This script demonstrates how to backtest the SuperTrend AI strategy with various
market conditions and parameter configurations.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from src.data import StockDataFetcher
from src.indicators import RSI, BollingerBands, VWAP, VWMABands
from src.indicators.technical_indicators import MACD, ATR, StochasticOscillator
from src.indicators.supertrend_ai import SuperTrendAI
from src.backtesting import BacktestEngine
from src.strategies import BaseStrategy
from src.utils import PerformanceMetrics
from src.visualization import Dashboard, ChartGenerator
from src.ml import DirectionPredictor, VolatilityForecaster, MarketRegimeDetector


class SuperTrendAIStrategy(BaseStrategy):
    """SuperTrend AI trading strategy implementation."""
    
    def __init__(self, config=None):
        """Initialize strategy with configuration."""
        super().__init__()
        
        # Default configuration
        default_config = {
            # SuperTrend AI parameters
            'atr_length': 10,
            'factor_min': 1.0,
            'factor_max': 5.0,
            'factor_step': 0.5,
            'perf_alpha': 10.0,
            'cluster_from': 'best',
            'max_iter': 1000,
            'max_data': 10000,
            
            # Strategy parameters
            'min_signal_strength': 4,
            'use_signal_strength_filter': True,
            'use_time_filter': False,
            'start_hour': 9,
            'end_hour': 16,
            
            # Risk Management
            'use_stop_loss': True,
            'stop_loss_type': 'ATR',  # 'ATR' or 'Percentage'
            'stop_loss_atr_mult': 2.0,
            'stop_loss_perc': 2.0,
            'use_take_profit': True,
            'take_profit_type': 'Risk/Reward',  # 'Risk/Reward', 'ATR', or 'Percentage'
            'risk_reward_ratio': 2.0,
            'take_profit_atr_mult': 3.0,
            'take_profit_perc': 4.0,
            
            # ML Confluence
            'use_ml_confluence': False,
            'ml_confidence_threshold': 0.6,
            'ml_features': ['price', 'volume', 'technical'],
            
            # Position Sizing
            'position_size_method': 'fixed',  # 'fixed', 'volatility', 'kelly'
            'position_size_pct': 0.1,  # 10% of capital per trade
            'max_positions': 1,
            'risk_per_trade': 0.02  # 2% risk per trade
        }
        
        # Update with provided config
        if config:
            default_config.update(config)
        self.config = default_config
        
        # Initialize SuperTrend AI indicator
        self.supertrend = SuperTrendAI(
            atr_length=self.config['atr_length'],
            factor_min=self.config['factor_min'],
            factor_max=self.config['factor_max'],
            factor_step=self.config['factor_step'],
            perf_alpha=self.config['perf_alpha'],
            cluster_from=self.config['cluster_from'],
            max_iter=self.config['max_iter'],
            max_data=self.config['max_data']
        )
        
        # Initialize ML models if enabled
        if self.config['use_ml_confluence']:
            self.direction_predictor = DirectionPredictor()
            self.volatility_forecaster = VolatilityForecaster()
            self.regime_detector = MarketRegimeDetector()
            
        # State tracking
        self.positions = {}
        self.pending_orders = {}
        
    def prepare_data(self, data):
        """Prepare data with required indicators."""
        # Calculate ATR for risk management
        atr_indicator = ATR(period=self.config['atr_length'])
        data['atr'] = atr_indicator.calculate(data)
        
        # Calculate SuperTrend AI
        st_result = self.supertrend.calculate(data)
        
        # Merge results
        for col in st_result.columns:
            data[f'st_{col}'] = st_result[col]
            
        # Add other technical indicators for ML features if needed
        if self.config['use_ml_confluence']:
            # RSI
            rsi = RSI(period=14)
            data['rsi'] = rsi.calculate(data)
            
            # Bollinger Bands
            bb = BollingerBands(period=20)
            bb_result = bb.calculate(data)
            data = data.join(bb_result)
            
            # MACD
            macd = MACD()
            macd_result = macd.calculate(data)
            data = data.join(macd_result)
            
            # Volume indicators
            data['volume_sma'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
            
        return data
        
    def generate_signals(self, data):
        """Generate trading signals based on SuperTrend AI."""
        signals = pd.DataFrame(index=data.index)
        
        # Get SuperTrend signals
        st_signals = self.supertrend.generate_signals(data[[col for col in data.columns if col.startswith('st_')]])
        signals = signals.join(st_signals)
        
        # Apply signal strength filter
        if self.config['use_signal_strength_filter']:
            min_strength = self.config['min_signal_strength']
            signals['long_entry'] = signals['long_entry'] & (data['st_signal_strength'] >= min_strength)
            signals['short_entry'] = signals['short_entry'] & (data['st_signal_strength'] >= min_strength)
        
        # Apply time filter
        if self.config['use_time_filter'] and hasattr(data.index, 'hour'):
            valid_hours = (data.index.hour >= self.config['start_hour']) & \
                         (data.index.hour <= self.config['end_hour'])
            signals['long_entry'] = signals['long_entry'] & valid_hours
            signals['short_entry'] = signals['short_entry'] & valid_hours
        
        # Apply ML confluence if enabled
        if self.config['use_ml_confluence'] and hasattr(self, 'direction_predictor'):
            ml_signals = self._get_ml_signals(data)
            signals['long_entry'] = signals['long_entry'] & ml_signals['ml_long']
            signals['short_entry'] = signals['short_entry'] & ml_signals['ml_short']
        
        # Calculate stop loss and take profit levels
        self._calculate_risk_levels(data, signals)
        
        return signals
    
    def _get_ml_signals(self, data):
        """Generate ML-based signals."""
        # Prepare features
        features = self._prepare_ml_features(data)
        
        # Get predictions
        direction_pred = self.direction_predictor.predict(features)
        regime = self.regime_detector.predict(features)
        
        # Create ML signals
        ml_signals = pd.DataFrame(index=data.index)
        ml_signals['ml_long'] = (
            (direction_pred['direction_proba'] > self.config['ml_confidence_threshold']) &
            (direction_pred['prediction'] == 1) &
            regime['regime'].isin(['uptrend', 'accumulation'])
        )
        ml_signals['ml_short'] = (
            (direction_pred['direction_proba'] > self.config['ml_confidence_threshold']) &
            (direction_pred['prediction'] == -1) &
            regime['regime'].isin(['downtrend', 'distribution'])
        )
        
        return ml_signals
    
    def _prepare_ml_features(self, data):
        """Prepare features for ML models."""
        features = pd.DataFrame(index=data.index)
        
        if 'price' in self.config['ml_features']:
            # Price-based features
            features['returns'] = data['close'].pct_change()
            features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            features['high_low_spread'] = (data['high'] - data['low']) / data['close']
            features['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
            
        if 'volume' in self.config['ml_features']:
            # Volume-based features
            features['volume_ratio'] = data.get('volume_ratio', 1.0)
            features['price_volume'] = data['close'] * data['volume']
            
        if 'technical' in self.config['ml_features']:
            # Technical indicator features
            if 'rsi' in data.columns:
                features['rsi'] = data['rsi']
                features['rsi_signal'] = (data['rsi'] < 30).astype(int) - (data['rsi'] > 70).astype(int)
                
            if 'macd' in data.columns:
                features['macd'] = data['macd']
                features['macd_signal_diff'] = data.get('macd_signal_diff', 0)
                
            if 'upper_band' in data.columns:
                features['bb_position'] = (data['close'] - data['middle_band']) / (data['upper_band'] - data['middle_band'])
                
        return features.fillna(0)
    
    def _calculate_risk_levels(self, data, signals):
        """Calculate stop loss and take profit levels."""
        # Stop Loss calculation
        if self.config['use_stop_loss']:
            if self.config['stop_loss_type'] == 'ATR':
                sl_distance = data['atr'] * self.config['stop_loss_atr_mult']
            else:  # Percentage
                sl_distance = data['close'] * (self.config['stop_loss_perc'] / 100)
                
            signals['stop_loss_long'] = data['close'] - sl_distance
            signals['stop_loss_short'] = data['close'] + sl_distance
        else:
            signals['stop_loss_long'] = 0
            signals['stop_loss_short'] = float('inf')
        
        # Take Profit calculation
        if self.config['use_take_profit']:
            if self.config['take_profit_type'] == 'Risk/Reward':
                # Based on stop loss distance
                risk_long = data['close'] - signals['stop_loss_long']
                risk_short = signals['stop_loss_short'] - data['close']
                
                signals['take_profit_long'] = data['close'] + (risk_long * self.config['risk_reward_ratio'])
                signals['take_profit_short'] = data['close'] - (risk_short * self.config['risk_reward_ratio'])
                
            elif self.config['take_profit_type'] == 'ATR':
                tp_distance = data['atr'] * self.config['take_profit_atr_mult']
                signals['take_profit_long'] = data['close'] + tp_distance
                signals['take_profit_short'] = data['close'] - tp_distance
                
            else:  # Percentage
                tp_distance = data['close'] * (self.config['take_profit_perc'] / 100)
                signals['take_profit_long'] = data['close'] + tp_distance
                signals['take_profit_short'] = data['close'] - tp_distance
        else:
            signals['take_profit_long'] = float('inf')
            signals['take_profit_short'] = 0
            
        return signals
    
    def calculate_position_size(self, capital, price, stop_loss):
        """Calculate position size based on risk management rules."""
        if self.config['position_size_method'] == 'fixed':
            # Fixed percentage of capital
            position_value = capital * self.config['position_size_pct']
            shares = int(position_value / price)
            
        elif self.config['position_size_method'] == 'volatility':
            # Volatility-based sizing
            risk_amount = capital * self.config['risk_per_trade']
            risk_per_share = abs(price - stop_loss)
            shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
            
        else:  # Kelly or other methods
            # Simplified Kelly criterion (would need win rate and avg win/loss)
            position_value = capital * min(self.config['position_size_pct'], 0.25)
            shares = int(position_value / price)
            
        return max(shares, 0)


async def fetch_and_prepare_data(symbol, start_date, end_date):
    """Fetch and prepare data for backtesting."""
    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    
    fetcher = StockDataFetcher()
    data = await fetcher.fetch(
        symbol=symbol,
        start=start_date,
        end=end_date,
        interval="1d"
    )
    
    print(f"Fetched {len(data)} bars of data")
    return data


def run_backtest(data, strategy_config, initial_capital=100000):
    """Run backtest with given data and configuration."""
    print("\nRunning backtest...")
    
    # Create strategy
    strategy = SuperTrendAIStrategy(config=strategy_config)
    
    # Prepare data
    prepared_data = strategy.prepare_data(data.copy())
    
    # Create backtest engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=0.001,  # 0.1%
        slippage=0.0005   # 0.05%
    )
    
    # Run backtest
    results = engine.run(
        data=prepared_data,
        strategy=strategy
    )
    
    return results


def analyze_results(results, strategy_name="SuperTrend AI"):
    """Analyze and display backtest results."""
    print(f"\n{strategy_name} Backtest Results:")
    print("=" * 50)
    
    metrics = results['metrics']
    
    # Display key metrics
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annual Return: {metrics.get('annual_return', 0):.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
    print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    
    # Trade statistics
    trades = results['trades']
    if len(trades) > 0:
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in trades) else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in trades) else 0
        
        print(f"\nAverage Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Best Trade: ${max(t['pnl'] for t in trades):.2f}")
        print(f"Worst Trade: ${min(t['pnl'] for t in trades):.2f}")
    
    return metrics


def run_parameter_optimization(data, base_config, param_grid):
    """Run parameter optimization over a grid."""
    print("\nRunning parameter optimization...")
    
    results = []
    
    for params in param_grid:
        # Update config
        config = base_config.copy()
        config.update(params)
        
        # Run backtest
        try:
            backtest_results = run_backtest(data, config)
            metrics = backtest_results['metrics']
            
            # Store results
            result = {
                'params': params,
                'sharpe_ratio': metrics['sharpe_ratio'],
                'total_return': metrics['total_return'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics.get('win_rate', 0),
                'total_trades': metrics.get('total_trades', 0)
            }
            results.append(result)
            
            print(f"  Tested {params}: Sharpe={result['sharpe_ratio']:.2f}, Return={result['total_return']:.2%}")
            
        except Exception as e:
            print(f"  Error with {params}: {str(e)}")
    
    # Find best parameters
    best_result = max(results, key=lambda x: x['sharpe_ratio'])
    print(f"\nBest parameters: {best_result['params']}")
    print(f"Best Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
    
    return results


def generate_report(results, output_dir="reports"):
    """Generate comprehensive backtest report."""
    print("\nGenerating report...")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create dashboard
    dashboard = Dashboard()
    
    # Add visualizations
    equity_curve = results['equity_curve']
    trades = results['trades']
    
    # Generate charts
    chart_gen = ChartGenerator()
    
    # Equity curve chart
    equity_fig = chart_gen.plot_equity_curve(equity_curve)
    
    # Monthly returns heatmap
    if 'monthly_returns' in results:
        returns_fig = chart_gen.plot_monthly_returns(results['monthly_returns'])
    
    # Trade analysis
    if len(trades) > 0:
        trade_fig = chart_gen.plot_trade_analysis(trades)
    
    # Save report
    report_path = os.path.join(output_dir, "supertrend_ai_backtest_report.html")
    dashboard.save(report_path)
    
    print(f"Report saved to: {report_path}")
    
    # Also save metrics as JSON
    metrics_path = os.path.join(output_dir, "supertrend_ai_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    
    return report_path


async def main():
    """Main execution function."""
    # Configuration
    symbols = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years of data
    
    # Base strategy configuration
    base_config = {
        'atr_length': 10,
        'factor_min': 1.0,
        'factor_max': 5.0,
        'factor_step': 0.5,
        'cluster_from': 'best',
        'min_signal_strength': 4,
        'use_signal_strength_filter': True,
        'stop_loss_atr_mult': 2.0,
        'risk_reward_ratio': 2.0,
        'position_size_pct': 0.1
    }
    
    # Test different market conditions
    market_configs = {
        'Conservative': {
            'min_signal_strength': 7,
            'stop_loss_atr_mult': 1.5,
            'risk_reward_ratio': 3.0,
            'position_size_pct': 0.05
        },
        'Moderate': base_config,
        'Aggressive': {
            'min_signal_strength': 3,
            'stop_loss_atr_mult': 2.5,
            'risk_reward_ratio': 1.5,
            'position_size_pct': 0.15
        }
    }
    
    all_results = {}
    
    # Test on multiple symbols
    for symbol in symbols[:1]:  # Test on first symbol for demo
        print(f"\n{'='*60}")
        print(f"Testing {symbol}")
        print('='*60)
        
        # Fetch data
        data = await fetch_and_prepare_data(symbol, start_date, end_date)
        
        # Test different configurations
        for config_name, config in market_configs.items():
            print(f"\n--- {config_name} Configuration ---")
            results = run_backtest(data, config)
            metrics = analyze_results(results, f"{symbol} - {config_name}")
            
            all_results[f"{symbol}_{config_name}"] = {
                'symbol': symbol,
                'config_name': config_name,
                'config': config,
                'metrics': metrics,
                'results': results
            }
        
        # Parameter optimization for best configuration
        if symbol == symbols[0]:  # Only optimize on first symbol
            param_grid = [
                {'min_signal_strength': 3, 'stop_loss_atr_mult': 1.5},
                {'min_signal_strength': 4, 'stop_loss_atr_mult': 2.0},
                {'min_signal_strength': 5, 'stop_loss_atr_mult': 2.5},
                {'min_signal_strength': 6, 'stop_loss_atr_mult': 2.0},
                {'min_signal_strength': 7, 'stop_loss_atr_mult': 1.5},
            ]
            
            optimization_results = run_parameter_optimization(data, base_config, param_grid)
    
    # Generate final report
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    # Find best overall configuration
    best_config = max(all_results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'])
    print(f"\nBest Configuration: {best_config[0]}")
    print(f"Sharpe Ratio: {best_config[1]['metrics']['sharpe_ratio']:.2f}")
    print(f"Total Return: {best_config[1]['metrics']['total_return']:.2%}")
    
    # Generate report for best configuration
    generate_report(best_config[1]['results'])
    
    # Save all results
    results_path = "reports/supertrend_ai_all_results.json"
    with open(results_path, 'w') as f:
        # Convert results to serializable format
        serializable_results = {}
        for key, value in all_results.items():
            serializable_results[key] = {
                'symbol': value['symbol'],
                'config_name': value['config_name'],
                'config': value['config'],
                'metrics': value['metrics']
            }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nAll results saved to: {results_path}")
    print("\nBacktest completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())