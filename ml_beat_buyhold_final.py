#!/usr/bin/env python3
"""
ML-Powered Strategy to Beat Buy-and-Hold
Leverages ensemble models, regime detection, and advanced risk management
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from src.ml.models.ensemble import EnsembleModel
from src.ml.models.regime_detection import MarketRegimeDetector
from src.ml.feature_engineering import FeatureEngineer
from src.indicators.technical_indicators import RSI, BollingerBands, MACD, ATR, VWAP
from src.backtesting.portfolio import Portfolio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

class MLPoweredStrategy:
    """ML-powered strategy using ensemble predictions and regime detection"""
    
    def __init__(self):
        self.ensemble_model = EnsembleModel()
        self.regime_detector = MarketRegimeDetector()
        self.feature_engineer = FeatureEngineer()
        self.trades = []
        self.regime_history = []
        self.prediction_history = []
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        df = data.copy()
        
        # Add technical indicators
        rsi = RSI(period=14)
        bb = BollingerBands(period=20)
        macd = MACD()
        atr = ATR(period=14)
        vwap = VWAP()
        
        df['RSI'] = rsi.calculate(df)
        bb_data = bb.calculate(df)
        df['BB_Upper'] = bb_data['upper']
        df['BB_Middle'] = bb_data['middle']
        df['BB_Lower'] = bb_data['lower']
        
        macd_data = macd.calculate(df)
        df['MACD'] = macd_data['macd']
        df['MACD_Signal'] = macd_data['signal']
        df['MACD_Histogram'] = macd_data['histogram']
        
        df['ATR'] = atr.calculate(df)
        df['VWAP'] = vwap.calculate(df)
        
        # Create all features using feature engineer
        df = self.feature_engineer.create_features(df)
        
        # Additional features for ML
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Market microstructure
        df['High_Low_Spread'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Spread'] = (df['Close'] - df['Open']) / df['Open']
        
        return df.dropna()
    
    def train_models(self, train_data: pd.DataFrame):
        """Train ML models on historical data"""
        logger.info("Training ML models...")
        
        # Prepare features
        df = self.prepare_features(train_data)
        
        if len(df) < 500:
            logger.warning("Insufficient data for training")
            return False
            
        # Create target variables
        df['Target_Return'] = df['Returns'].shift(-1)
        df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)
        
        # Train ensemble model
        feature_cols = [col for col in df.columns if col not in ['Target_Return', 'Target_Direction', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        X = df[feature_cols].iloc[:-1]
        y = df['Target_Direction'].iloc[:-1]
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train
        self.ensemble_model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.ensemble_model.score(X_train, y_train)
        test_score = self.ensemble_model.score(X_test, y_test)
        
        logger.info(f"Model trained - Train accuracy: {train_score:.2f}, Test accuracy: {test_score:.2f}")
        
        return True
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate ML-based trading signals"""
        df = self.prepare_features(data)
        signals = pd.Series(0.0, index=df.index)
        
        # Train models on first 80% of data
        train_size = int(len(df) * 0.8)
        if train_size < 500:
            logger.warning("Insufficient data for ML strategy")
            return signals
            
        train_data = data.iloc[:train_size]
        self.train_models(train_data)
        
        # Generate signals for remaining data
        feature_cols = [col for col in df.columns if col not in ['Target_Return', 'Target_Direction', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Log_Returns']]
        
        for i in range(train_size, len(df)):
            try:
                # Get current features
                current_features = df[feature_cols].iloc[i:i+1]
                
                # Detect regime
                regime_data = df.iloc[:i+1]
                regime = self.regime_detector.detect_regime(regime_data)
                self.regime_history.append(regime)
                
                # Get ensemble predictions
                direction_prob = self.ensemble_model.predict_proba(current_features)[0]
                bull_prob = direction_prob[1] if len(direction_prob) > 1 else 0.5
                
                # Store prediction
                self.prediction_history.append({
                    'date': df.index[i],
                    'bull_prob': bull_prob,
                    'regime': regime
                })
                
                # Generate signal based on regime and predictions
                signal_strength = 0
                
                # Base signal from ML prediction
                if bull_prob > 0.65:
                    signal_strength = (bull_prob - 0.5) * 2  # Scale to 0-1
                elif bull_prob < 0.35:
                    signal_strength = -(0.5 - bull_prob) * 2  # Scale to -1-0
                
                # Regime adjustments
                if regime == 'BULL_TRENDING':
                    # Strong trend following
                    if signal_strength > 0:
                        signal_strength *= 1.5
                    else:
                        signal_strength *= 0.5  # Reduce shorts in bull market
                        
                elif regime == 'BEAR_TRENDING':
                    # Be more cautious
                    if signal_strength > 0:
                        signal_strength *= 0.5  # Reduce longs in bear market
                    else:
                        signal_strength *= 1.5  # Strong exits
                        
                elif regime == 'HIGH_VOLATILITY':
                    # Reduce position size
                    signal_strength *= 0.7
                    
                elif regime == 'RANGING':
                    # Mean reversion
                    rsi = df['RSI'].iloc[i]
                    if rsi < 30 and bull_prob > 0.5:
                        signal_strength = 0.8
                    elif rsi > 70 and bull_prob < 0.5:
                        signal_strength = -0.8
                
                # Risk management
                current_volatility = df['ATR'].iloc[i] / df['Close'].iloc[i]
                if current_volatility > 0.03:  # High volatility
                    signal_strength *= 0.6
                
                # Apply signal
                if abs(signal_strength) > 0.3:
                    signals.iloc[i] = np.clip(signal_strength, -1, 1)
                    
                    if signal_strength > 0:
                        self.trades.append({
                            'type': 'buy',
                            'date': df.index[i],
                            'price': df['Close'].iloc[i],
                            'ml_prob': bull_prob,
                            'regime': regime
                        })
                    else:
                        self.trades.append({
                            'type': 'sell',
                            'date': df.index[i],
                            'price': df['Close'].iloc[i],
                            'ml_prob': bull_prob,
                            'regime': regime
                        })
                        
            except Exception as e:
                logger.debug(f"Error generating signal at {i}: {e}")
                continue
                
        return signals

class AdaptiveRiskStrategy:
    """Adaptive strategy with dynamic risk management"""
    
    def __init__(self):
        self.trades = []
        self.risk_params = {
            'max_position': 1.0,
            'stop_loss': 0.02,
            'take_profit': 0.05,
            'trailing_stop': 0.015
        }
        
    def calculate_position_size(self, confidence: float, volatility: float, capital: float) -> float:
        """Calculate position size using modified Kelly criterion"""
        # Estimated win rate and payoff from backtesting
        win_rate = 0.55 + confidence * 0.1  # 55-65% win rate
        avg_win = 0.02
        avg_loss = 0.015
        
        # Kelly formula
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Conservative approach: use 25% of Kelly
        position_pct = min(kelly * 0.25, 0.25)
        
        # Adjust for volatility
        if volatility > 0.02:
            position_pct *= 0.7
            
        return position_pct
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals with adaptive risk management"""
        # Prepare indicators
        df = data.copy()
        
        # Calculate indicators
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['ATR'] = self.calculate_atr(df)
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
        
        signals = pd.Series(0.0, index=df.index)
        position = 0
        entry_price = None
        highest_price = None
        
        for i in range(50, len(df)):
            # Current market conditions
            price = df['Close'].iloc[i]
            volatility = df['ATR'].iloc[i] / price
            
            # Exit conditions (always check first)
            if position > 0 and entry_price:
                # Stop loss
                if price < entry_price * (1 - self.risk_params['stop_loss']):
                    signals.iloc[i] = -1
                    position = 0
                    entry_price = None
                    continue
                    
                # Take profit
                if price > entry_price * (1 + self.risk_params['take_profit']):
                    signals.iloc[i] = -1
                    position = 0
                    entry_price = None
                    continue
                    
                # Trailing stop
                if highest_price and price < highest_price * (1 - self.risk_params['trailing_stop']):
                    signals.iloc[i] = -1
                    position = 0
                    entry_price = None
                    continue
                    
                # Update highest price
                if price > highest_price:
                    highest_price = price
            
            # Entry conditions
            if position == 0:
                # Multi-factor entry signal
                bull_signals = 0
                confidence = 0
                
                # Trend following
                if df['SMA_20'].iloc[i] > df['SMA_50'].iloc[i]:
                    bull_signals += 1
                    confidence += 0.2
                    
                # Momentum
                if df['Close'].iloc[i] > df['Close'].iloc[i-5]:
                    bull_signals += 1
                    confidence += 0.1
                    
                # RSI not overbought
                if 30 < df['RSI'].iloc[i] < 65:
                    bull_signals += 1
                    confidence += 0.15
                    
                # Price above BB middle
                if df['Close'].iloc[i] > df['BB_Middle'].iloc[i]:
                    bull_signals += 1
                    confidence += 0.15
                    
                # Volume confirmation
                if df['Volume'].iloc[i] > df['Volume'].iloc[i-20:i].mean():
                    confidence += 0.1
                
                # Generate entry signal
                if bull_signals >= 3 and confidence > 0.5:
                    position_size = self.calculate_position_size(confidence, volatility, 1.0)
                    signals.iloc[i] = position_size
                    position = position_size
                    entry_price = price
                    highest_price = price
                    
                    self.trades.append({
                        'type': 'buy',
                        'date': df.index[i],
                        'price': price,
                        'confidence': confidence,
                        'position_size': position_size
                    })
        
        return signals
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """Calculate Bollinger Bands"""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

def backtest_strategy(data: pd.DataFrame, signals: pd.Series, initial_capital: float = 100000) -> dict:
    """Run backtest with realistic execution"""
    position = 0
    cash = initial_capital
    portfolio_values = []
    trades = []
    
    for i in range(len(data)):
        price = data['Close'].iloc[i]
        
        if i < len(signals):
            signal = signals.iloc[i]
            
            if signal > 0 and cash > 0:
                # Buy
                invest_amount = cash * abs(signal)
                shares = invest_amount / (price * 1.001)  # Include commission
                position += shares
                cash -= invest_amount
                trades.append(('buy', price, shares))
                
            elif signal < 0 and position > 0:
                # Sell
                sell_shares = position * abs(signal)
                proceeds = sell_shares * price * 0.999  # Include commission
                position -= sell_shares
                cash += proceeds
                trades.append(('sell', price, sell_shares))
        
        # Portfolio value
        total_value = cash + position * price
        portfolio_values.append(total_value)
    
    # Calculate metrics
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    total_return = (portfolio_values[-1] - initial_capital) / initial_capital * 100
    
    # Sharpe ratio
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-6)
    
    # Max drawdown
    cumulative = pd.Series(portfolio_values)
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Win rate
    positive_returns = (returns > 0).sum()
    total_days = len(returns)
    win_rate = positive_returns / total_days * 100 if total_days > 0 else 0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': len(trades),
        'final_value': portfolio_values[-1]
    }

def calculate_buyhold_benchmark(data: pd.DataFrame) -> dict:
    """Calculate buy-and-hold metrics"""
    initial = data['Close'].iloc[0]
    final = data['Close'].iloc[-1]
    
    total_return = (final - initial) / initial * 100
    
    returns = data['Close'].pct_change().dropna()
    sharpe = np.sqrt(252) * returns.mean() / returns.std()
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown
    }

def load_data(symbol: str, timeframe: str = '1D') -> pd.DataFrame:
    """Load data from CSV"""
    filename = DATA_DIR / f"{symbol}_{timeframe}_2020-01-01_2024-01-01.csv"
    if filename.exists():
        df = pd.read_csv(filename, index_col='Date', parse_dates=True)
        df.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        return df
    return None

def main():
    """Main execution"""
    logger.info("="*60)
    logger.info("ML-POWERED STRATEGY TO BEAT BUY-AND-HOLD")
    logger.info("="*60)
    
    symbols = ['AAPL', 'MSFT', 'SPY', 'QQQ', 'TSLA', 'GLD']
    
    all_results = []
    
    # Test both strategies
    strategies = {
        'ML Ensemble': MLPoweredStrategy(),
        'Adaptive Risk': AdaptiveRiskStrategy()
    }
    
    for symbol in symbols:
        logger.info(f"\nTesting {symbol}...")
        
        # Load data
        data = load_data(symbol)
        if data is None or len(data) < 500:
            logger.warning(f"Insufficient data for {symbol}")
            continue
        
        # Calculate benchmark
        benchmark = calculate_buyhold_benchmark(data)
        logger.info(f"Buy-and-Hold: Return={benchmark['total_return']:.2f}%, Sharpe={benchmark['sharpe_ratio']:.2f}")
        
        # Test each strategy
        for strategy_name, strategy in strategies.items():
            logger.info(f"\nTesting {strategy_name}...")
            
            try:
                # Generate signals
                signals = strategy.generate_signals(data)
                
                # Backtest
                results = backtest_strategy(data, signals)
                
                # Compare
                beats_buyhold = results['sharpe_ratio'] > benchmark['sharpe_ratio']
                
                logger.info(f"{strategy_name}: Return={results['total_return']:.2f}%, Sharpe={results['sharpe_ratio']:.2f}")
                logger.info(f"Beats Buy-and-Hold: {'✅ YES' if beats_buyhold else '❌ NO'}")
                logger.info(f"Max Drawdown: {results['max_drawdown']:.2f}%, Win Rate: {results['win_rate']:.1f}%")
                
                all_results.append({
                    'symbol': symbol,
                    'strategy': strategy_name,
                    'strategy_return': results['total_return'],
                    'benchmark_return': benchmark['total_return'],
                    'strategy_sharpe': results['sharpe_ratio'],
                    'benchmark_sharpe': benchmark['sharpe_ratio'],
                    'beats_benchmark': beats_buyhold,
                    'max_drawdown': results['max_drawdown'],
                    'win_rate': results['win_rate'],
                    'num_trades': results['num_trades']
                })
                
            except Exception as e:
                logger.error(f"Error testing {strategy_name} on {symbol}: {e}")
                continue
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*60)
    
    df_results = pd.DataFrame(all_results)
    
    if len(df_results) > 0:
        wins = df_results['beats_benchmark'].sum()
        total = len(df_results)
        success_rate = wins / total * 100
        
        logger.info(f"\nOverall Success Rate: {wins}/{total} ({success_rate:.1f}%)")
        
        # Best performing combinations
        best_results = df_results[df_results['beats_benchmark']]
        if len(best_results) > 0:
            logger.info("\nWinning Strategies:")
            for _, row in best_results.iterrows():
                logger.info(f"- {row['symbol']} with {row['strategy']}: Sharpe {row['strategy_sharpe']:.2f} vs {row['benchmark_sharpe']:.2f}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        # Create summary report
        html = f"""
        <html>
        <head>
            <title>ML Strategy vs Buy-and-Hold Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; }}
                .success {{ color: #27ae60; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                .winner {{ background-color: #d5f4e6; }}
            </style>
        </head>
        <body>
            <h1>ML-Powered Trading Strategy Results</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p class="success">Success Rate: {success_rate:.1f}% ({wins}/{total} tests)</p>
                <p>Test Period: 2020-01-01 to 2024-01-01</p>
                <p>Strategies Tested: ML Ensemble (XGBoost + LSTM + Regime Detection), Adaptive Risk Management</p>
            </div>
            
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Strategy</th>
                    <th>Strategy Return</th>
                    <th>B&H Return</th>
                    <th>Strategy Sharpe</th>
                    <th>B&H Sharpe</th>
                    <th>Max DD</th>
                    <th>Win Rate</th>
                    <th>Trades</th>
                    <th>Result</th>
                </tr>
        """
        
        for _, row in df_results.iterrows():
            row_class = 'winner' if row['beats_benchmark'] else ''
            result_text = '✅' if row['beats_benchmark'] else '❌'
            
            html += f"""
                <tr class="{row_class}">
                    <td>{row['symbol']}</td>
                    <td>{row['strategy']}</td>
                    <td>{row['strategy_return']:.2f}%</td>
                    <td>{row['benchmark_return']:.2f}%</td>
                    <td><strong>{row['strategy_sharpe']:.3f}</strong></td>
                    <td>{row['benchmark_sharpe']:.3f}</td>
                    <td>{row['max_drawdown']:.2f}%</td>
                    <td>{row['win_rate']:.1f}%</td>
                    <td>{row['num_trades']}</td>
                    <td>{result_text}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <div class="summary">
                <h3>Strategy Features</h3>
                <ul>
                    <li><strong>ML Ensemble:</strong> Combines XGBoost direction prediction with LSTM volatility forecasting</li>
                    <li><strong>Regime Detection:</strong> Adapts strategy based on market conditions</li>
                    <li><strong>Risk Management:</strong> Dynamic position sizing with stop-loss and trailing stops</li>
                    <li><strong>Feature Engineering:</strong> 60+ technical and market microstructure features</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(report_dir / f"ml_beat_buyhold_{timestamp}.html", 'w') as f:
            f.write(html)
            
        with open(report_dir / f"ml_beat_buyhold_{timestamp}.json", 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'success_rate': float(success_rate),
                'results': df_results.to_dict('records')
            }, f, indent=2)
            
        logger.info(f"\nReports saved to {report_dir}/")
        
        if success_rate >= 70:
            logger.info("\n🎉 SUCCESS! ML strategy consistently beats buy-and-hold!")
        elif success_rate >= 50:
            logger.info("\n📈 Partial success - strategy beats buy-and-hold in majority of cases")
        else:
            logger.info("\n🔧 Further optimization needed")

if __name__ == "__main__":
    main()