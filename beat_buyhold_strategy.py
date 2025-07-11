#!/usr/bin/env python3
"""
Beat Buy-and-Hold Strategy - Iterative Optimization
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
import json
import logging
from pathlib import Path
import yfinance as yf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BuyAndHoldBenchmark:
    """Calculate buy-and-hold returns for comparison"""
    
    @staticmethod
    def calculate_returns(data: pd.DataFrame) -> dict:
        """Calculate buy-and-hold performance metrics"""
        initial_price = data['Close'].iloc[0]
        final_price = data['Close'].iloc[-1]
        total_return = (final_price - initial_price) / initial_price
        
        # Daily returns
        daily_returns = data['Close'].pct_change().dropna()
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        
        # Max drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate (days with positive returns)
        win_rate = (daily_returns > 0).sum() / len(daily_returns)
        
        return {
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown * 100,
            'volatility': daily_returns.std() * np.sqrt(252) * 100,
            'win_rate': win_rate * 100
        }

class EnhancedMomentumStrategy:
    """Advanced momentum strategy with multiple signals"""
    
    def __init__(self, lookback_short=20, lookback_long=50, rsi_period=14):
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.rsi_period = rsi_period
        self.trades = []
        
    def prepare_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        df = data.copy()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # ATR for volatility
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(14).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        
        # Price patterns
        df['Higher_High'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
        
        # Momentum
        df['ROC'] = df['Close'].pct_change(10) * 100
        df['MOM'] = df['Close'] - df['Close'].shift(20)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals"""
        df = self.prepare_indicators(data)
        signals = pd.Series(0, index=df.index)
        
        for i in range(200, len(df)):
            # Score-based system
            buy_score = 0
            sell_score = 0
            
            # Trend following
            if df['SMA_20'].iloc[i] > df['SMA_50'].iloc[i] > df['SMA_200'].iloc[i]:
                buy_score += 2
            elif df['SMA_20'].iloc[i] < df['SMA_50'].iloc[i] < df['SMA_200'].iloc[i]:
                sell_score += 2
                
            # MACD
            if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i] and df['MACD_Histogram'].iloc[i] > 0:
                buy_score += 1.5
            elif df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i] and df['MACD_Histogram'].iloc[i] < 0:
                sell_score += 1.5
                
            # RSI
            if df['RSI'].iloc[i] < 30:
                buy_score += 2
            elif df['RSI'].iloc[i] > 70:
                sell_score += 2
            elif 40 < df['RSI'].iloc[i] < 60:
                # Neutral zone - trend confirmation
                if df['RSI'].iloc[i] > df['RSI'].iloc[i-1]:
                    buy_score += 0.5
                else:
                    sell_score += 0.5
                    
            # Bollinger Bands
            if df['BB_Position'].iloc[i] < 0.2:
                buy_score += 1.5
            elif df['BB_Position'].iloc[i] > 0.8:
                sell_score += 1.5
                
            # Volume confirmation
            if df['Volume_Ratio'].iloc[i] > 1.5:
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    buy_score += 1
                else:
                    sell_score += 1
                    
            # Momentum
            if df['ROC'].iloc[i] > 5:
                buy_score += 1
            elif df['ROC'].iloc[i] < -5:
                sell_score += 1
                
            # Price action
            if df['Higher_High'].iloc[i]:
                buy_score += 0.5
            if df['Lower_Low'].iloc[i]:
                sell_score += 0.5
                
            # Generate signal
            if buy_score > 6:
                signals.iloc[i] = min(buy_score / 10, 1)  # Position sizing
                self.trades.append({'type': 'buy', 'date': df.index[i], 'price': df['Close'].iloc[i]})
            elif sell_score > 6:
                signals.iloc[i] = -min(sell_score / 10, 1)
                self.trades.append({'type': 'sell', 'date': df.index[i], 'price': df['Close'].iloc[i]})
                
        return signals

class MLEnhancedStrategy:
    """Machine Learning enhanced strategy"""
    
    def __init__(self):
        self.trades = []
        self.feature_cols = []
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create ML features"""
        df = data.copy()
        
        # Price-based features
        for period in [5, 10, 20, 50]:
            df[f'Return_{period}'] = df['Close'].pct_change(period)
            df[f'High_Low_Ratio_{period}'] = df['High'].rolling(period).max() / df['Low'].rolling(period).min()
            
        # Technical indicators from previous strategy
        momentum = EnhancedMomentumStrategy()
        df = momentum.prepare_indicators(df)
        
        # Additional ML features
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        
        # Volatility features
        df['Volatility_20'] = df['Close'].pct_change().rolling(20).std()
        df['Volatility_Ratio'] = df['Volatility_20'] / df['Volatility_20'].rolling(50).mean()
        
        # Market microstructure
        df['High_Low_Spread'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Spread'] = (df['Close'] - df['Open']) / df['Open']
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate ML-based signals"""
        df = self.create_features(data)
        signals = pd.Series(0, index=df.index)
        
        # Simple rule-based ML (without external dependencies)
        for i in range(200, len(df)):
            # Feature extraction
            features = {
                'momentum': df['ROC'].iloc[i],
                'rsi': df['RSI'].iloc[i],
                'macd_hist': df['MACD_Histogram'].iloc[i],
                'bb_position': df['BB_Position'].iloc[i],
                'volume_ratio': df['Volume_Ratio'].iloc[i],
                'volatility_ratio': df['Volatility_Ratio'].iloc[i],
                'trend_strength': (df['SMA_20'].iloc[i] - df['SMA_50'].iloc[i]) / df['SMA_50'].iloc[i] * 100
            }
            
            # Decision tree-like logic
            signal_strength = 0
            
            # Strong momentum with low volatility
            if features['momentum'] > 5 and features['volatility_ratio'] < 1.2:
                signal_strength += 0.4
                
            # Oversold with positive MACD
            if features['rsi'] < 30 and features['macd_hist'] > 0:
                signal_strength += 0.5
                
            # Trend following with volume
            if features['trend_strength'] > 2 and features['volume_ratio'] > 1.3:
                signal_strength += 0.3
                
            # Mean reversion at extremes
            if features['bb_position'] < 0.1:
                signal_strength += 0.4
            elif features['bb_position'] > 0.9:
                signal_strength -= 0.4
                
            # Overbought conditions
            if features['rsi'] > 70 and features['momentum'] < -5:
                signal_strength -= 0.5
                
            # Apply signal
            if abs(signal_strength) > 0.5:
                signals.iloc[i] = np.clip(signal_strength, -1, 1)
                
        return signals

class StrategyOptimizer:
    """Main optimizer to beat buy-and-hold"""
    
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.results = []
        
    def fetch_data(self, symbol):
        """Fetch stock data"""
        try:
            data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
            return data
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
            
    def backtest_strategy(self, data, strategy, initial_capital=100000):
        """Run backtest for a strategy"""
        signals = strategy.generate_signals(data)
        
        # Portfolio simulation
        position = 0
        cash = initial_capital
        portfolio_value = []
        
        for i in range(len(data)):
            price = data['Close'].iloc[i]
            
            if i < len(signals):
                signal = signals.iloc[i]
                
                if signal > 0 and cash > 0:
                    # Buy
                    shares_to_buy = (cash * abs(signal)) / price
                    position += shares_to_buy
                    cash -= shares_to_buy * price * 1.001  # Commission
                    
                elif signal < 0 and position > 0:
                    # Sell
                    shares_to_sell = position * abs(signal)
                    position -= shares_to_sell
                    cash += shares_to_sell * price * 0.999  # Commission
                    
            # Track portfolio value
            total_value = cash + position * price
            portfolio_value.append(total_value)
            
        # Calculate returns
        portfolio_returns = pd.Series(portfolio_value).pct_change().dropna()
        
        # Performance metrics
        total_return = (portfolio_value[-1] - initial_capital) / initial_capital * 100
        sharpe = np.sqrt(252) * portfolio_returns.mean() / (portfolio_returns.std() + 1e-6)
        
        # Max drawdown
        cumulative = pd.Series(portfolio_value)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'trades': len(strategy.trades)
        }
    
    def optimize(self):
        """Run optimization across all symbols"""
        logger.info("Starting strategy optimization to beat buy-and-hold...")
        
        all_results = []
        
        for symbol in self.symbols:
            logger.info(f"\nTesting {symbol}...")
            
            # Fetch data
            data = self.fetch_data(symbol)
            if data is None or len(data) < 200:
                continue
                
            # Calculate buy-and-hold benchmark
            benchmark = BuyAndHoldBenchmark.calculate_returns(data)
            logger.info(f"Buy-and-Hold - Return: {benchmark['total_return']:.2f}%, Sharpe: {benchmark['sharpe_ratio']:.2f}")
            
            # Test strategies
            strategies = {
                'Enhanced Momentum': EnhancedMomentumStrategy(),
                'ML Enhanced': MLEnhancedStrategy()
            }
            
            for name, strategy in strategies.items():
                results = self.backtest_strategy(data, strategy)
                results['symbol'] = symbol
                results['strategy'] = name
                results['beats_buyhold'] = results['sharpe_ratio'] > benchmark['sharpe_ratio']
                
                logger.info(f"{name} - Return: {results['total_return']:.2f}%, Sharpe: {results['sharpe_ratio']:.2f}, "
                          f"Beats B&H: {'YES' if results['beats_buyhold'] else 'NO'}")
                
                all_results.append(results)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("="*60)
        
        df_results = pd.DataFrame(all_results)
        
        # Best performing strategies
        best_by_sharpe = df_results.loc[df_results.groupby('symbol')['sharpe_ratio'].idxmax()]
        
        wins = 0
        total = 0
        
        for _, row in best_by_sharpe.iterrows():
            if row['beats_buyhold']:
                wins += 1
                logger.info(f"✓ {row['symbol']}: {row['strategy']} beats buy-and-hold (Sharpe: {row['sharpe_ratio']:.2f})")
            else:
                logger.info(f"✗ {row['symbol']}: Failed to beat buy-and-hold")
            total += 1
            
        success_rate = wins / total * 100
        logger.info(f"\nSuccess Rate: {success_rate:.1f}% ({wins}/{total} symbols)")
        
        # Save results
        self.save_results(df_results)
        
        return df_results
    
    def save_results(self, results_df):
        """Save optimization results"""
        # Create reports directory
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Save to JSON
        results_file = reports_dir / f"beat_buyhold_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_df.to_json(results_file, orient='records', indent=2)
        
        # Create HTML report
        html = f"""
        <html>
        <head>
            <title>Beat Buy-and-Hold Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .winner {{ background-color: #d4edda; }}
                .loser {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <h1>Strategy Optimization Results</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Performance Summary</h2>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Strategy</th>
                    <th>Total Return (%)</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown (%)</th>
                    <th>Trades</th>
                    <th>Beats Buy-Hold</th>
                </tr>
        """
        
        for _, row in results_df.iterrows():
            row_class = 'winner' if row['beats_buyhold'] else 'loser'
            html += f"""
                <tr class="{row_class}">
                    <td>{row['symbol']}</td>
                    <td>{row['strategy']}</td>
                    <td>{row['total_return']:.2f}</td>
                    <td>{row['sharpe_ratio']:.2f}</td>
                    <td>{row['max_drawdown']:.2f}</td>
                    <td>{row['trades']}</td>
                    <td>{'YES' if row['beats_buyhold'] else 'NO'}</td>
                </tr>
            """
            
        html += """
            </table>
        </body>
        </html>
        """
        
        html_file = reports_dir / f"beat_buyhold_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_file, 'w') as f:
            f.write(html)
            
        logger.info(f"\nResults saved to {results_file}")
        logger.info(f"HTML report saved to {html_file}")

def main():
    """Main execution"""
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'QQQ']
    
    # Date range
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    # Run optimizer
    optimizer = StrategyOptimizer(symbols, start_date, end_date)
    results = optimizer.optimize()
    
    logger.info("\nOptimization complete!")

if __name__ == "__main__":
    main()