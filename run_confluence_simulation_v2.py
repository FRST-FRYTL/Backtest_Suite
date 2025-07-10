#!/usr/bin/env python3
"""
Run Confluence Strategy Simulation with 3 Iterations
Iteration 1: Baseline implementation
Iteration 2: Optimize for maximum profit
Iteration 3: Optimize for risk-adjusted returns
"""

import pandas as pd
import numpy as np
import yaml
from datetime import datetime
import json
import os
from typing import Dict, List, Any, Tuple

class ConfluenceStrategy:
    """Multi-indicator confluence strategy with configurable scoring system"""
    
    def __init__(self, config: Dict, iteration_params: Dict = None):
        self.config = config
        
        # Default confluence weights
        self.weights = {
            'trend': 0.30,
            'momentum': 0.25,
            'volatility': 0.25,
            'volume': 0.20
        }
        
        # Update with iteration parameters
        if iteration_params and 'weights' in iteration_params:
            self.weights.update(iteration_params['weights'])
        
        self.entry_threshold = iteration_params.get('entry_threshold', 0.75) if iteration_params else 0.75
        self.positions = {}
        self.signals = []
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all technical indicators"""
        indicators = {}
        
        # SMAs
        for period in [20, 50, 100, 200]:
            indicators[f'SMA{period}'] = data['close'].rolling(window=period).mean()
        
        # Bollinger Bands (20, 2)
        sma20 = indicators['SMA20']
        std20 = data['close'].rolling(window=20).std()
        indicators['BB20_upper'] = sma20 + (2 * std20)
        indicators['BB20_lower'] = sma20 - (2 * std20)
        indicators['BB20_middle'] = sma20
        
        # RSI
        indicators['RSI'] = self.calculate_rsi(data['close'], 14)
        
        # VWAP (simple daily)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        indicators['VWAP'] = (typical_price * data['volume']).rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
        
        # ATR
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators['ATR'] = true_range.rolling(window=14).mean()
        
        return indicators
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=period, min_periods=1).mean()
        avg_losses = losses.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_confluence_score(self, data: pd.DataFrame, indicators: Dict, idx: int) -> Tuple[float, Dict]:
        """Calculate overall confluence score"""
        if idx < 200:  # Need minimum data
            return 0, {'insufficient_data': True}
        
        # Trend score (SMA alignment)
        trend_score = 0
        price = data['close'].iloc[idx]
        sma_scores = []
        for period in [20, 50, 100, 200]:
            if not pd.isna(indicators[f'SMA{period}'].iloc[idx]):
                if price > indicators[f'SMA{period}'].iloc[idx]:
                    sma_scores.append(1)
                else:
                    sma_scores.append(0)
        trend_score = np.mean(sma_scores) if sma_scores else 0.5
        
        # Momentum score (RSI)
        rsi = indicators['RSI'].iloc[idx]
        if rsi < 30:
            momentum_score = 1.0
        elif rsi < 50:
            momentum_score = 0.5 + (50 - rsi) / 40 * 0.5
        elif rsi < 70:
            momentum_score = 0.5 - (rsi - 50) / 40 * 0.3
        else:
            momentum_score = 0.2
        
        # Volatility score (Bollinger Bands)
        bb_upper = indicators['BB20_upper'].iloc[idx]
        bb_lower = indicators['BB20_lower'].iloc[idx]
        bb_width = bb_upper - bb_lower
        if bb_width > 0:
            position_in_band = (price - bb_lower) / bb_width
            if position_in_band < 0.3:
                volatility_score = 0.8
            elif position_in_band < 0.7:
                volatility_score = 0.5
            else:
                volatility_score = 0.2
        else:
            volatility_score = 0.5
        
        # Volume score (VWAP)
        vwap = indicators['VWAP'].iloc[idx]
        if not pd.isna(vwap):
            if price < vwap:
                volume_score = 0.7
            else:
                volume_score = 0.3
        else:
            volume_score = 0.5
        
        # Calculate weighted score
        confluence_score = (
            trend_score * self.weights['trend'] +
            momentum_score * self.weights['momentum'] +
            volatility_score * self.weights['volatility'] +
            volume_score * self.weights['volume']
        )
        
        details = {
            'scores': {
                'trend': trend_score,
                'momentum': momentum_score,
                'volatility': volatility_score,
                'volume': volume_score
            },
            'confluence': confluence_score,
            'price': price,
            'rsi': rsi,
            'bb_position': position_in_band if bb_width > 0 else 0.5
        }
        
        return confluence_score, details


class ConfluenceSimulation:
    """Main simulation runner"""
    
    def __init__(self):
        with open('config/strategy_config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results = {}
    
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load historical data"""
        path = f'data/{symbol}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            print(f"✓ Loaded {symbol}: {len(df)} rows")
            return df
        return None
    
    def simulate_iteration(self, iteration: int, focus: str, params: Dict) -> Dict:
        """Run one iteration of the simulation"""
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration}: {focus.upper()}")
        print(f"{'='*50}")
        
        # Update configuration display
        if params:
            print("\nIteration Parameters:")
            for key, value in params.items():
                print(f"  {key}: {value}")
        
        iteration_results = {}
        test_assets = ['SPY', 'QQQ', 'AAPL', 'GLD', 'TLT']
        
        for symbol in test_assets:
            print(f"\n📊 Processing {symbol}...")
            
            # Load data
            data = self.load_data(symbol)
            if data is None:
                continue
            
            # Initialize strategy
            strategy = ConfluenceStrategy(self.config, params)
            
            # Calculate indicators
            indicators = strategy.calculate_indicators(data)
            
            # Generate signals and calculate scores
            signals = []
            confluence_scores = []
            
            for idx in range(200, len(data)):
                score, details = strategy.calculate_confluence_score(data, indicators, idx)
                confluence_scores.append({
                    'date': data.index[idx],
                    'score': score,
                    'details': details
                })
                
                # Generate buy signal if threshold met
                if score >= strategy.entry_threshold:
                    # Check reentry delay
                    if not signals or (data.index[idx] - signals[-1]['date']).days >= 5:
                        signals.append({
                            'date': data.index[idx],
                            'symbol': symbol,
                            'action': 'BUY',
                            'price': data['close'].iloc[idx],
                            'confluence_score': score,
                            'details': details
                        })
            
            # Simulate trading
            initial_capital = 10000
            cash = initial_capital
            positions = []
            trades = []
            portfolio_values = []
            
            position_size_pct = params.get('position_size_pct', 0.2)
            stop_loss_pct = params.get('stop_loss_pct', 0.04)
            profit_target_pct = params.get('profit_target_pct', 0.15)
            
            for idx, row in data.iterrows():
                # Check for buy signals
                for signal in signals:
                    if signal['date'] == idx and len(positions) == 0:  # Only one position at a time
                        # Buy
                        position_size = cash * position_size_pct
                        shares = position_size / signal['price']
                        cash -= position_size
                        
                        positions.append({
                            'entry_date': idx,
                            'entry_price': signal['price'],
                            'shares': shares,
                            'confluence_score': signal['confluence_score']
                        })
                        
                        trades.append({
                            'date': idx,
                            'action': 'BUY',
                            'price': signal['price'],
                            'shares': shares,
                            'confluence_score': signal['confluence_score']
                        })
                
                # Check exits for open positions
                positions_to_close = []
                for i, pos in enumerate(positions):
                    current_price = row['close']
                    pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                    days_held = (idx - pos['entry_date']).days
                    
                    # Exit conditions
                    if (pnl_pct >= profit_target_pct or 
                        pnl_pct <= -stop_loss_pct or 
                        days_held >= 30):
                        
                        # Sell
                        proceeds = pos['shares'] * current_price
                        cash += proceeds
                        
                        trades.append({
                            'date': idx,
                            'action': 'SELL',
                            'price': current_price,
                            'shares': pos['shares'],
                            'pnl_pct': pnl_pct,
                            'days_held': days_held
                        })
                        
                        positions_to_close.append(i)
                
                # Remove closed positions
                for i in sorted(positions_to_close, reverse=True):
                    positions.pop(i)
                
                # Calculate portfolio value
                positions_value = sum(pos['shares'] * row['close'] for pos in positions)
                total_value = cash + positions_value
                portfolio_values.append({
                    'date': idx,
                    'value': total_value,
                    'cash': cash,
                    'positions': len(positions)
                })
            
            # Calculate metrics
            portfolio_df = pd.DataFrame(portfolio_values).set_index('date')
            returns = portfolio_df['value'].pct_change().fillna(0)
            
            # Performance metrics
            total_return = (portfolio_df['value'].iloc[-1] / initial_capital - 1) * 100
            annual_return = ((portfolio_df['value'].iloc[-1] / initial_capital) ** (252 / len(returns)) - 1) * 100
            
            # Sharpe ratio
            sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            # Win rate
            winning_trades = [t for t in trades if t.get('action') == 'SELL' and t.get('pnl_pct', 0) > 0]
            total_sells = [t for t in trades if t.get('action') == 'SELL']
            win_rate = len(winning_trades) / len(total_sells) * 100 if total_sells else 0
            
            results = {
                'symbol': symbol,
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': len([t for t in trades if t['action'] == 'BUY']),
                'signals': len(signals),
                'avg_confluence': np.mean([s['confluence_score'] for s in signals]) if signals else 0
            }
            
            iteration_results[symbol] = results
            
            # Print results
            print(f"  Return: {total_return:.1f}% | Sharpe: {sharpe:.2f} | DD: {max_drawdown:.1f}% | Trades: {results['total_trades']}")
        
        # Store results
        self.results[f'iteration_{iteration}'] = {
            'focus': focus,
            'params': params,
            'results': iteration_results
        }
        
        return iteration_results
    
    def run_all_iterations(self):
        """Run all 3 iterations"""
        print("="*60)
        print("🚀 CONFLUENCE STRATEGY SIMULATION")
        print("="*60)
        
        # Define iterations
        iterations = [
            {
                'num': 1,
                'focus': 'baseline',
                'params': {}
            },
            {
                'num': 2,
                'focus': 'profit',
                'params': {
                    'weights': {
                        'trend': 0.35,
                        'momentum': 0.30,
                        'volatility': 0.20,
                        'volume': 0.15
                    },
                    'entry_threshold': 0.65,
                    'position_size_pct': 0.25,
                    'profit_target_pct': 0.20,
                    'stop_loss_pct': 0.05
                }
            },
            {
                'num': 3,
                'focus': 'risk',
                'params': {
                    'weights': {
                        'trend': 0.25,
                        'momentum': 0.20,
                        'volatility': 0.35,
                        'volume': 0.20
                    },
                    'entry_threshold': 0.80,
                    'position_size_pct': 0.15,
                    'profit_target_pct': 0.12,
                    'stop_loss_pct': 0.03
                }
            }
        ]
        
        # Run iterations
        for iteration in iterations:
            results = self.simulate_iteration(
                iteration['num'],
                iteration['focus'],
                iteration['params']
            )
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate comparison summary"""
        print("\n" + "="*60)
        print("📊 SIMULATION SUMMARY")
        print("="*60)
        
        # Create comparison table
        comparison_data = []
        
        for iter_key, iter_data in self.results.items():
            iter_num = iter_key.split('_')[1]
            focus = iter_data['focus']
            
            avg_return = []
            avg_sharpe = []
            avg_drawdown = []
            total_trades = 0
            
            for symbol, results in iter_data['results'].items():
                avg_return.append(results['annual_return'])
                avg_sharpe.append(results['sharpe_ratio'])
                avg_drawdown.append(results['max_drawdown'])
                total_trades += results['total_trades']
            
            comparison_data.append({
                'Iteration': f"{iter_num} ({focus})",
                'Avg Return': f"{np.mean(avg_return):.1f}%",
                'Avg Sharpe': f"{np.mean(avg_sharpe):.2f}",
                'Avg Drawdown': f"{np.mean(avg_drawdown):.1f}%",
                'Total Trades': total_trades
            })
        
        # Print table
        df = pd.DataFrame(comparison_data)
        print("\n" + df.to_string(index=False))
        
        # Save detailed results
        os.makedirs('reports/confluence_simulation', exist_ok=True)
        
        with open('reports/confluence_simulation/simulation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print("\n✅ Results saved to: reports/confluence_simulation/simulation_results.json")
        
        # Generate HTML reports
        self.generate_html_reports()
    
    def generate_html_reports(self):
        """Generate HTML reports for each iteration"""
        for iter_key, iter_data in self.results.items():
            iter_num = iter_key.split('_')[1]
            focus = iter_data['focus']
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Confluence Strategy - Iteration {iter_num} ({focus})</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #0a0e1a;
            color: #e0e0e0;
            margin: 20px;
        }}
        h1 {{
            color: #4a90e2;
            text-align: center;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: #1a1f2e;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4a90e2;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #2a2f3e;
        }}
        th {{
            background: #1a1f2e;
            color: #4a90e2;
        }}
    </style>
</head>
<body>
    <h1>Confluence Strategy Report - Iteration {iter_num}</h1>
    <h2>Focus: {focus.title()}</h2>
    
    <div class="metrics-grid">
"""
            
            # Add summary metrics
            avg_metrics = {
                'return': [],
                'sharpe': [],
                'drawdown': [],
                'winrate': []
            }
            
            for symbol, results in iter_data['results'].items():
                avg_metrics['return'].append(results['annual_return'])
                avg_metrics['sharpe'].append(results['sharpe_ratio'])
                avg_metrics['drawdown'].append(results['max_drawdown'])
                avg_metrics['winrate'].append(results['win_rate'])
            
            html_content += f"""
        <div class="metric-card">
            <h3>Avg Annual Return</h3>
            <div class="metric-value">{np.mean(avg_metrics['return']):.1f}%</div>
        </div>
        <div class="metric-card">
            <h3>Avg Sharpe Ratio</h3>
            <div class="metric-value">{np.mean(avg_metrics['sharpe']):.2f}</div>
        </div>
        <div class="metric-card">
            <h3>Avg Max Drawdown</h3>
            <div class="metric-value">{np.mean(avg_metrics['drawdown']):.1f}%</div>
        </div>
        <div class="metric-card">
            <h3>Avg Win Rate</h3>
            <div class="metric-value">{np.mean(avg_metrics['winrate']):.1f}%</div>
        </div>
    </div>
    
    <h2>Asset Performance</h2>
    <table>
        <tr>
            <th>Symbol</th>
            <th>Annual Return</th>
            <th>Sharpe Ratio</th>
            <th>Max Drawdown</th>
            <th>Win Rate</th>
            <th>Total Trades</th>
        </tr>
"""
            
            for symbol, results in iter_data['results'].items():
                html_content += f"""
        <tr>
            <td>{symbol}</td>
            <td>{results['annual_return']:.1f}%</td>
            <td>{results['sharpe_ratio']:.2f}</td>
            <td>{results['max_drawdown']:.1f}%</td>
            <td>{results['win_rate']:.1f}%</td>
            <td>{results['total_trades']}</td>
        </tr>
"""
            
            html_content += """
    </table>
</body>
</html>
"""
            
            # Save report
            filename = f'reports/confluence_simulation/iteration_{iter_num}_{focus}_report.html'
            with open(filename, 'w') as f:
                f.write(html_content)
            
            print(f"📄 Report saved: {filename}")


def main():
    simulation = ConfluenceSimulation()
    simulation.run_all_iterations()


if __name__ == "__main__":
    main()