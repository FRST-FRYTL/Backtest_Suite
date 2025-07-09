"""
Contribution Timing Strategy Implementation

This module implements the optimized contribution timing strategy for 
dollar-cost averaging with market-based enhancements.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContributionTimingStrategy:
    """
    Enhanced dollar-cost averaging strategy that adjusts contribution amounts
    based on market conditions to improve long-term returns.
    """
    
    def __init__(self, 
                 base_contribution: float = 1000,
                 max_multiplier: float = 2.0,
                 ma_period: int = 200,
                 rsi_period: int = 14,
                 rsi_oversold: int = 30,
                 ma_discount_threshold: float = 0.05,
                 vix_high_threshold: float = 25):
        """
        Initialize strategy parameters.
        
        Args:
            base_contribution: Base monthly contribution amount
            max_multiplier: Maximum contribution multiplier (cap)
            ma_period: Moving average period for trend detection
            rsi_period: RSI calculation period
            rsi_oversold: RSI threshold for oversold condition
            ma_discount_threshold: Percentage below MA to trigger bonus
            vix_high_threshold: VIX level to indicate high volatility
        """
        self.base_contribution = base_contribution
        self.max_multiplier = max_multiplier
        self.ma_period = ma_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.ma_discount_threshold = ma_discount_threshold
        self.vix_high_threshold = vix_high_threshold
        
        # Track strategy performance
        self.contribution_history = []
        self.signal_history = []
    
    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI indicator"""
        if len(prices) < self.rsi_period + 1:
            return 50.0  # Neutral if insufficient data
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        if loss.iloc[-1] == 0:
            return 100.0
        
        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_signals(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate timing signals based on market conditions.
        
        Args:
            market_data: DataFrame with columns ['Close', 'Volume'] minimum
            
        Returns:
            Dictionary of signals and their values
        """
        signals = {
            'ma_signal': 0,
            'rsi_signal': 0,
            'volatility_signal': 0,
            'composite_multiplier': 1.0
        }
        
        if len(market_data) < self.ma_period:
            logger.warning("Insufficient data for full signal calculation")
            return signals
        
        # Current price and indicators
        current_price = market_data['Close'].iloc[-1]
        
        # Moving Average Signal
        ma = market_data['Close'].rolling(self.ma_period).mean().iloc[-1]
        price_to_ma_ratio = current_price / ma
        
        if price_to_ma_ratio < (1 - self.ma_discount_threshold):
            # Price is below MA by threshold
            discount = 1 - price_to_ma_ratio
            if discount > 0.10:  # More than 10% below
                signals['ma_signal'] = 1.0  # 100% bonus
            elif discount > 0.05:  # 5-10% below
                signals['ma_signal'] = 0.5  # 50% bonus
            else:
                signals['ma_signal'] = 0.3  # 30% bonus
        
        # RSI Signal
        rsi = self.calculate_rsi(market_data['Close'])
        
        if rsi < 25:  # Extremely oversold
            signals['rsi_signal'] = 0.5
        elif rsi < self.rsi_oversold:  # Oversold
            signals['rsi_signal'] = 0.3
        elif rsi < 40:  # Moderately oversold
            signals['rsi_signal'] = 0.1
        
        # Volatility Signal (using price volatility as VIX proxy)
        if len(market_data) >= 30:
            returns = market_data['Close'].pct_change()
            volatility = returns.rolling(30).std().iloc[-1] * np.sqrt(252) * 100
            
            if volatility > 35:  # Very high volatility
                signals['volatility_signal'] = 0.4
            elif volatility > self.vix_high_threshold:  # High volatility
                signals['volatility_signal'] = 0.2
        
        # Calculate composite multiplier
        total_bonus = (signals['ma_signal'] + 
                      signals['rsi_signal'] + 
                      signals['volatility_signal'])
        
        signals['composite_multiplier'] = min(1.0 + total_bonus, self.max_multiplier)
        
        return signals
    
    def get_contribution_amount(self, market_data: pd.DataFrame) -> Tuple[float, Dict]:
        """
        Calculate the contribution amount for the current period.
        
        Args:
            market_data: Current market data
            
        Returns:
            Tuple of (contribution_amount, signals_dict)
        """
        signals = self.calculate_signals(market_data)
        contribution = self.base_contribution * signals['composite_multiplier']
        
        # Log decision
        logger.info(f"Contribution decision: ${contribution:.2f} "
                   f"({signals['composite_multiplier']:.2f}x multiplier)")
        logger.info(f"Signals - MA: {signals['ma_signal']:.2f}, "
                   f"RSI: {signals['rsi_signal']:.2f}, "
                   f"Vol: {signals['volatility_signal']:.2f}")
        
        # Store in history
        self.contribution_history.append({
            'timestamp': datetime.now(),
            'amount': contribution,
            'multiplier': signals['composite_multiplier'],
            'base_amount': self.base_contribution
        })
        
        self.signal_history.append({
            'timestamp': datetime.now(),
            **signals
        })
        
        return contribution, signals
    
    def get_signal_summary(self, market_data: pd.DataFrame) -> str:
        """
        Get a human-readable summary of current market signals.
        
        Args:
            market_data: Current market data
            
        Returns:
            String summary of signals and recommendation
        """
        signals = self.calculate_signals(market_data)
        current_price = market_data['Close'].iloc[-1]
        ma = market_data['Close'].rolling(self.ma_period).mean().iloc[-1]
        rsi = self.calculate_rsi(market_data['Close'])
        
        summary = f"""
CONTRIBUTION TIMING ANALYSIS
===========================
Current Price: ${current_price:.2f}
200-day MA: ${ma:.2f} ({((current_price/ma - 1) * 100):.1f}% vs MA)
RSI(14): {rsi:.1f}

SIGNALS ACTIVE:
"""
        
        if signals['ma_signal'] > 0:
            summary += f"✅ Price below MA200 - Bonus: {signals['ma_signal']*100:.0f}%\n"
        
        if signals['rsi_signal'] > 0:
            summary += f"✅ RSI oversold - Bonus: {signals['rsi_signal']*100:.0f}%\n"
        
        if signals['volatility_signal'] > 0:
            summary += f"✅ High volatility - Bonus: {signals['volatility_signal']*100:.0f}%\n"
        
        if signals['composite_multiplier'] == 1.0:
            summary += "❌ No timing signals active\n"
        
        summary += f"""
RECOMMENDATION:
==============
Contribution Multiplier: {signals['composite_multiplier']:.2f}x
Recommended Contribution: ${self.base_contribution * signals['composite_multiplier']:,.2f}
(Base amount: ${self.base_contribution:,.2f})

Strategy: {"ENHANCED CONTRIBUTION - Market conditions favorable" 
           if signals['composite_multiplier'] > 1.0 
           else "STANDARD CONTRIBUTION - Normal market conditions"}
"""
        
        return summary
    
    def backtest_contribution_schedule(self, 
                                     market_data: pd.DataFrame,
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Backtest the contribution strategy over historical data.
        
        Args:
            market_data: Historical market data
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            DataFrame with contribution schedule and performance
        """
        if start_date:
            market_data = market_data[market_data.index >= start_date]
        if end_date:
            market_data = market_data[market_data.index <= end_date]
        
        # Generate monthly contribution dates
        contribution_dates = pd.date_range(
            start=market_data.index[0],
            end=market_data.index[-1],
            freq='MS'  # Month start
        )
        
        results = []
        
        for date in contribution_dates:
            # Get data up to contribution date
            historical_data = market_data[market_data.index <= date]
            
            if len(historical_data) >= self.ma_period:
                contribution, signals = self.get_contribution_amount(historical_data)
                
                results.append({
                    'date': date,
                    'price': historical_data['Close'].iloc[-1],
                    'contribution': contribution,
                    'multiplier': signals['composite_multiplier'],
                    'ma_signal': signals['ma_signal'],
                    'rsi_signal': signals['rsi_signal'],
                    'volatility_signal': signals['volatility_signal']
                })
        
        return pd.DataFrame(results)
    
    def calculate_strategy_metrics(self, backtest_results: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics for the strategy.
        
        Args:
            backtest_results: Results from backtest_contribution_schedule
            
        Returns:
            Dictionary of performance metrics
        """
        if backtest_results.empty:
            return {}
        
        total_contributions = backtest_results['contribution'].sum()
        base_contributions = len(backtest_results) * self.base_contribution
        
        # Calculate shares purchased
        backtest_results['shares'] = (backtest_results['contribution'] / 
                                     backtest_results['price'])
        total_shares = backtest_results['shares'].sum()
        
        # Final value (using last price)
        final_price = backtest_results['price'].iloc[-1]
        final_value = total_shares * final_price
        
        # Calculate metrics
        metrics = {
            'total_contributions': total_contributions,
            'base_contributions': base_contributions,
            'contribution_enhancement': total_contributions / base_contributions,
            'total_shares': total_shares,
            'final_value': final_value,
            'total_return': (final_value - total_contributions) / total_contributions,
            'timing_effectiveness': (backtest_results['multiplier'] > 1.0).sum() / len(backtest_results),
            'avg_multiplier': backtest_results['multiplier'].mean(),
            'max_multiplier_used': backtest_results['multiplier'].max(),
            'enhanced_contributions': (backtest_results['multiplier'] > 1.0).sum()
        }
        
        return metrics


def example_usage():
    """Example of how to use the contribution timing strategy"""
    
    # Initialize strategy
    strategy = ContributionTimingStrategy(
        base_contribution=1000,
        max_multiplier=2.0,
        rsi_oversold=30,
        ma_discount_threshold=0.05
    )
    
    # Load market data (example with SPY)
    import yfinance as yf
    spy = yf.download('SPY', start='2020-01-01', end='2024-01-01')
    
    # Get current contribution recommendation
    current_contribution, signals = strategy.get_contribution_amount(spy)
    print(f"Current month contribution: ${current_contribution:,.2f}")
    
    # Get signal summary
    summary = strategy.get_signal_summary(spy)
    print(summary)
    
    # Run backtest
    backtest_results = strategy.backtest_contribution_schedule(spy)
    
    # Calculate metrics
    metrics = strategy.calculate_strategy_metrics(backtest_results)
    
    print("\nBacktest Results:")
    print(f"Total Contributions: ${metrics['total_contributions']:,.2f}")
    print(f"Enhancement Factor: {metrics['contribution_enhancement']:.2f}x")
    print(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Timing Effectiveness: {metrics['timing_effectiveness']:.1%}")
    
    return strategy, backtest_results, metrics


if __name__ == "__main__":
    strategy, results, metrics = example_usage()