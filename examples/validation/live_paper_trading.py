"""
Live Paper Trading Simulation for Strategy Validation

This module sets up a paper trading environment to validate the contribution
timing strategy with real-time market data simulation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
import queue
import logging
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade execution"""
    timestamp: datetime
    action: str  # 'BUY' or 'SELL'
    symbol: str
    quantity: float
    price: float
    contribution_amount: float
    timing_multiplier: float
    reason: str
    

@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio at a point in time"""
    timestamp: datetime
    total_value: float
    cash: float
    positions: Dict[str, float]
    total_contributions: float
    total_return: float
    drawdown: float
    

class PaperTradingEngine:
    """Paper trading engine for strategy validation"""
    
    def __init__(self, initial_capital: float = 10000,
                 monthly_contribution: float = 1000,
                 symbols: List[str] = None):
        self.initial_capital = initial_capital
        self.monthly_contribution = monthly_contribution
        self.symbols = symbols or ['SPY', 'VTI', 'QQQ']
        
        # Portfolio state
        self.cash = initial_capital
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.total_contributions = initial_capital
        
        # Trading history
        self.trades: List[Trade] = []
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.peak_value = initial_capital
        
        # Market data cache
        self.market_data_cache = {}
        self.last_prices = {}
        
        # Threading for real-time simulation
        self.running = False
        self.update_thread = None
        self.contribution_thread = None
        
    def fetch_market_data(self, symbol: str, period: str = '1d') -> pd.DataFrame:
        """Fetch latest market data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if not data.empty:
                # Calculate technical indicators
                data['MA200'] = data['Close'].rolling(200).mean()
                data['RSI'] = self.calculate_rsi(data['Close'])
                data['Returns'] = data['Close'].pct_change()
                data['Volatility'] = data['Returns'].rolling(20).std() * np.sqrt(252)
                
                self.market_data_cache[symbol] = data
                self.last_prices[symbol] = data['Close'].iloc[-1]
                
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def evaluate_contribution_timing(self) -> Tuple[str, float, str]:
        """Evaluate market conditions for contribution timing"""
        
        # Default to equal-weight SPY
        selected_symbol = 'SPY'
        timing_multiplier = 1.0
        reasons = []
        
        # Analyze each symbol
        symbol_scores = {}
        
        for symbol in self.symbols:
            score = 0
            data = self.market_data_cache.get(symbol, pd.DataFrame())
            
            if data.empty:
                continue
                
            latest = data.iloc[-1]
            
            # Check if below 200-day MA (bearish, good for buying)
            if pd.notna(latest.get('MA200', None)):
                if latest['Close'] < latest['MA200']:
                    score += 2
                    if latest['Close'] < latest['MA200'] * 0.95:
                        score += 1  # Deep discount
            
            # Check RSI
            if pd.notna(latest.get('RSI', None)):
                if latest['RSI'] < 30:
                    score += 3  # Oversold
                elif latest['RSI'] < 40:
                    score += 1
            
            # Check volatility
            if pd.notna(latest.get('Volatility', None)):
                if latest['Volatility'] > 0.25:  # High volatility
                    score += 1
            
            # Recent drawdown
            recent_high = data['Close'].rolling(20).max().iloc[-1]
            drawdown = (latest['Close'] - recent_high) / recent_high
            if drawdown < -0.10:
                score += 2
            
            symbol_scores[symbol] = score
        
        # Select best opportunity
        if symbol_scores:
            selected_symbol = max(symbol_scores, key=symbol_scores.get)
            best_score = symbol_scores[selected_symbol]
            
            # Determine timing multiplier based on score
            if best_score >= 5:
                timing_multiplier = 2.0
                reasons.append("Strong oversold conditions")
            elif best_score >= 3:
                timing_multiplier = 1.5
                reasons.append("Moderate buying opportunity")
            elif best_score >= 1:
                timing_multiplier = 1.2
                reasons.append("Slight market weakness")
            else:
                timing_multiplier = 1.0
                reasons.append("Normal market conditions")
        
        return selected_symbol, timing_multiplier, "; ".join(reasons)
    
    def execute_contribution(self):
        """Execute monthly contribution with timing strategy"""
        
        # Fetch latest market data
        for symbol in self.symbols:
            self.fetch_market_data(symbol)
        
        # Evaluate timing
        symbol, multiplier, reason = self.evaluate_contribution_timing()
        
        # Calculate contribution amount
        contribution = self.monthly_contribution * multiplier
        
        # Execute trade
        if symbol in self.last_prices:
            price = self.last_prices[symbol]
            shares = contribution / price
            
            # Update portfolio
            self.cash += contribution
            self.cash -= contribution  # Buy immediately
            self.positions[symbol] += shares
            self.total_contributions += contribution
            
            # Record trade
            trade = Trade(
                timestamp=datetime.now(),
                action='BUY',
                symbol=symbol,
                quantity=shares,
                price=price,
                contribution_amount=contribution,
                timing_multiplier=multiplier,
                reason=reason
            )
            self.trades.append(trade)
            
            logger.info(f"Executed contribution: {symbol} x{shares:.4f} @ ${price:.2f} "
                       f"(${contribution:.2f}, {multiplier}x timing)")
    
    def calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        total_value = self.cash
        
        for symbol, shares in self.positions.items():
            if symbol in self.last_prices and shares > 0:
                total_value += shares * self.last_prices[symbol]
        
        return total_value
    
    def update_portfolio_snapshot(self):
        """Create and store portfolio snapshot"""
        current_value = self.calculate_portfolio_value()
        
        # Update peak value for drawdown calculation
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        drawdown = (current_value - self.peak_value) / self.peak_value
        
        total_return = (current_value - self.total_contributions) / self.total_contributions
        
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_value=current_value,
            cash=self.cash,
            positions=self.positions.copy(),
            total_contributions=self.total_contributions,
            total_return=total_return,
            drawdown=drawdown
        )
        
        self.portfolio_history.append(snapshot)
    
    def compare_with_backtest(self, backtest_results: Dict) -> Dict:
        """Compare paper trading results with backtest expectations"""
        
        if not self.portfolio_history:
            return {}
        
        latest_snapshot = self.portfolio_history[-1]
        
        comparison = {
            'portfolio_value': {
                'paper_trading': latest_snapshot.total_value,
                'backtest_expected': backtest_results.get('expected_value', 0),
                'difference_pct': 0
            },
            'total_return': {
                'paper_trading': latest_snapshot.total_return,
                'backtest_expected': backtest_results.get('expected_return', 0),
                'difference_pct': 0
            },
            'drawdown': {
                'paper_trading': latest_snapshot.drawdown,
                'backtest_expected': backtest_results.get('expected_drawdown', 0),
                'difference_pct': 0
            },
            'contribution_timing': {
                'paper_trading_avg': np.mean([t.timing_multiplier for t in self.trades]) if self.trades else 1.0,
                'backtest_expected': backtest_results.get('expected_timing', 1.0),
                'effectiveness': 0
            }
        }
        
        # Calculate differences
        for metric in ['portfolio_value', 'total_return', 'drawdown']:
            if comparison[metric]['backtest_expected'] != 0:
                diff = (comparison[metric]['paper_trading'] - comparison[metric]['backtest_expected'])
                comparison[metric]['difference_pct'] = diff / abs(comparison[metric]['backtest_expected']) * 100
        
        # Calculate timing effectiveness
        comparison['contribution_timing']['effectiveness'] = (
            comparison['contribution_timing']['paper_trading_avg'] / 
            comparison['contribution_timing']['backtest_expected']
        )
        
        return comparison
    
    def monitor_strategy_decay(self) -> Dict:
        """Monitor for signs of strategy decay"""
        
        if len(self.portfolio_history) < 10:
            return {'status': 'insufficient_data'}
        
        # Analyze recent performance
        recent_returns = []
        for i in range(1, min(11, len(self.portfolio_history))):
            prev = self.portfolio_history[-i-1].total_value
            curr = self.portfolio_history[-i].total_value
            ret = (curr - prev) / prev
            recent_returns.append(ret)
        
        # Calculate rolling statistics
        decay_indicators = {
            'recent_avg_return': np.mean(recent_returns),
            'recent_volatility': np.std(recent_returns),
            'recent_sharpe': np.mean(recent_returns) / (np.std(recent_returns) + 1e-6),
            'timing_effectiveness': np.mean([t.timing_multiplier for t in self.trades[-10:]]) if len(self.trades) >= 10 else 1.0,
            'consecutive_losses': 0,
            'max_recent_drawdown': min(s.drawdown for s in self.portfolio_history[-10:])
        }
        
        # Count consecutive losses
        for ret in recent_returns:
            if ret < 0:
                decay_indicators['consecutive_losses'] += 1
            else:
                break
        
        # Determine decay status
        warnings = []
        if decay_indicators['recent_sharpe'] < 0.5:
            warnings.append("Low risk-adjusted returns")
        if decay_indicators['consecutive_losses'] >= 3:
            warnings.append("Multiple consecutive losses")
        if decay_indicators['max_recent_drawdown'] < -0.15:
            warnings.append("Significant recent drawdown")
        if decay_indicators['timing_effectiveness'] < 1.1:
            warnings.append("Timing strategy not adding value")
        
        decay_indicators['warnings'] = warnings
        decay_indicators['status'] = 'healthy' if not warnings else 'warning'
        
        return decay_indicators
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        
        if not self.portfolio_history:
            return "No trading history available"
        
        latest = self.portfolio_history[-1]
        
        # Calculate statistics
        total_trades = len(self.trades)
        avg_timing_multiplier = np.mean([t.timing_multiplier for t in self.trades]) if self.trades else 1.0
        enhanced_contributions = sum(1 for t in self.trades if t.timing_multiplier > 1.0)
        
        # Time-based metrics
        start_date = self.portfolio_history[0].timestamp
        end_date = latest.timestamp
        days_active = (end_date - start_date).days
        
        report = f"""
PAPER TRADING PERFORMANCE REPORT
================================
Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Trading Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
Days Active: {days_active}

PORTFOLIO SUMMARY
=================
Initial Capital: ${self.initial_capital:,.2f}
Total Contributions: ${latest.total_contributions:,.2f}
Current Value: ${latest.total_value:,.2f}
Total Return: {latest.total_return:.2%}
Current Drawdown: {latest.drawdown:.2%}

POSITION BREAKDOWN
==================
Cash: ${latest.cash:,.2f} ({latest.cash/latest.total_value:.1%})
"""
        
        for symbol, shares in latest.positions.items():
            if shares > 0 and symbol in self.last_prices:
                value = shares * self.last_prices[symbol]
                weight = value / latest.total_value
                report += f"{symbol}: {shares:.4f} shares @ ${self.last_prices[symbol]:.2f} = ${value:,.2f} ({weight:.1%})\n"
        
        report += f"""

TRADING ACTIVITY
================
Total Trades: {total_trades}
Average Timing Multiplier: {avg_timing_multiplier:.2f}x
Enhanced Contributions: {enhanced_contributions} ({enhanced_contributions/total_trades*100:.1%} of trades)

RECENT TRADES (Last 5)
----------------------
"""
        
        for trade in self.trades[-5:]:
            report += f"{trade.timestamp.strftime('%Y-%m-%d')}: {trade.action} {trade.quantity:.4f} "
            report += f"{trade.symbol} @ ${trade.price:.2f} (${trade.contribution_amount:.2f}, "
            report += f"{trade.timing_multiplier}x) - {trade.reason}\n"
        
        # Add decay monitoring
        decay_status = self.monitor_strategy_decay()
        if decay_status['status'] != 'insufficient_data':
            report += f"""

STRATEGY HEALTH MONITORING
==========================
Status: {decay_status['status'].upper()}
Recent Sharpe Ratio: {decay_status.get('recent_sharpe', 0):.2f}
Timing Effectiveness: {decay_status.get('timing_effectiveness', 1):.2f}x
Recent Max Drawdown: {decay_status.get('max_recent_drawdown', 0):.2%}
"""
            
            if decay_status.get('warnings'):
                report += "\nWarnings:\n"
                for warning in decay_status['warnings']:
                    report += f"  - {warning}\n"
        
        return report
    
    def export_results(self, filepath: str):
        """Export trading results to JSON"""
        
        results = {
            'summary': {
                'initial_capital': self.initial_capital,
                'total_contributions': self.total_contributions,
                'final_value': self.calculate_portfolio_value(),
                'total_return': (self.calculate_portfolio_value() - self.total_contributions) / self.total_contributions,
                'number_of_trades': len(self.trades),
                'avg_timing_multiplier': np.mean([t.timing_multiplier for t in self.trades]) if self.trades else 1.0
            },
            'trades': [asdict(trade) for trade in self.trades],
            'portfolio_history': [
                {
                    'timestamp': snapshot.timestamp.isoformat(),
                    'total_value': snapshot.total_value,
                    'total_return': snapshot.total_return,
                    'drawdown': snapshot.drawdown
                }
                for snapshot in self.portfolio_history
            ],
            'current_positions': self.positions,
            'decay_monitoring': self.monitor_strategy_decay()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def start_live_simulation(self, update_interval: int = 3600, 
                            contribution_day: int = 1):
        """Start live paper trading simulation"""
        
        self.running = True
        
        # Portfolio update thread
        def update_loop():
            while self.running:
                try:
                    # Update market data
                    for symbol in self.symbols:
                        self.fetch_market_data(symbol)
                    
                    # Update portfolio snapshot
                    self.update_portfolio_snapshot()
                    
                    # Log status
                    logger.info(f"Portfolio value: ${self.calculate_portfolio_value():,.2f}")
                    
                    # Sleep until next update
                    time.sleep(update_interval)
                    
                except Exception as e:
                    logger.error(f"Error in update loop: {e}")
        
        # Monthly contribution thread
        def contribution_loop():
            while self.running:
                try:
                    # Check if it's contribution day
                    if datetime.now().day == contribution_day:
                        self.execute_contribution()
                        # Wait until next month
                        time.sleep(86400 * 25)  # Sleep for 25 days
                    else:
                        # Check again tomorrow
                        time.sleep(86400)
                        
                except Exception as e:
                    logger.error(f"Error in contribution loop: {e}")
        
        # Start threads
        self.update_thread = threading.Thread(target=update_loop)
        self.contribution_thread = threading.Thread(target=contribution_loop)
        
        self.update_thread.start()
        self.contribution_thread.start()
        
        logger.info("Paper trading simulation started")
    
    def stop_simulation(self):
        """Stop the live simulation"""
        self.running = False
        
        if self.update_thread:
            self.update_thread.join()
        if self.contribution_thread:
            self.contribution_thread.join()
        
        logger.info("Paper trading simulation stopped")


def run_paper_trading_validation():
    """Run paper trading validation with sample backtest comparison"""
    
    # Initialize paper trading engine
    engine = PaperTradingEngine(
        initial_capital=10000,
        monthly_contribution=1000,
        symbols=['SPY', 'QQQ', 'IWM']  # S&P 500, NASDAQ, Russell 2000
    )
    
    # Sample backtest results for comparison
    backtest_results = {
        'expected_value': 15000,
        'expected_return': 0.12,
        'expected_drawdown': -0.08,
        'expected_timing': 1.3
    }
    
    # Run simulation for demonstration (normally would run continuously)
    print("Starting paper trading simulation...")
    
    # Simulate some months of trading
    for month in range(6):
        print(f"\nMonth {month + 1}:")
        
        # Fetch market data
        for symbol in engine.symbols:
            engine.fetch_market_data(symbol, period='1mo')
        
        # Execute monthly contribution
        engine.execute_contribution()
        
        # Update portfolio
        engine.update_portfolio_snapshot()
        
        # Show current status
        current_value = engine.calculate_portfolio_value()
        total_return = (current_value - engine.total_contributions) / engine.total_contributions
        print(f"Portfolio Value: ${current_value:,.2f}")
        print(f"Total Return: {total_return:.2%}")
    
    # Generate reports
    print("\nGenerating performance report...")
    report = engine.generate_performance_report()
    
    # Save report
    report_path = Path('/workspaces/Backtest_Suite/examples/reports')
    report_path.mkdir(exist_ok=True)
    
    with open(report_path / 'paper_trading_report.txt', 'w') as f:
        f.write(report)
    
    # Export results
    engine.export_results(str(report_path / 'paper_trading_results.json'))
    
    # Compare with backtest
    comparison = engine.compare_with_backtest(backtest_results)
    
    print("\nPaper Trading vs Backtest Comparison:")
    print(json.dumps(comparison, indent=2))
    
    # Check for strategy decay
    decay_status = engine.monitor_strategy_decay()
    print(f"\nStrategy Health: {decay_status.get('status', 'unknown').upper()}")
    
    return engine


if __name__ == "__main__":
    engine = run_paper_trading_validation()