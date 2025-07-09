"""
Market Research Analysis for Monthly Contribution Strategy
Analyzes optimal strategies for a $10,000 initial account with $500 monthly contributions
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.fetcher import StockDataFetcher
from src.indicators.fear_greed import FearGreedIndex
from src.indicators.insider import InsiderTrading
from src.indicators.max_pain import MaxPain
from src.utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown


class MarketResearchAnalyzer:
    """Comprehensive market research for retail investment strategies."""
    
    def __init__(self):
        self.fetcher = StockDataFetcher()
        self.fear_greed = FearGreedIndex()
        self.max_pain = MaxPain()
        self.initial_capital = 10000
        self.monthly_contribution = 500
        
        # Major indices to analyze
        self.indices = {
            'SPY': 'S&P 500 ETF',
            'QQQ': 'NASDAQ 100 ETF', 
            'IWM': 'Russell 2000 ETF',
            'DIA': 'Dow Jones ETF',
            'VTI': 'Total Market ETF',
            'EFA': 'International Developed Markets',
            'EEM': 'Emerging Markets ETF',
            'AGG': 'Bond ETF',
            'GLD': 'Gold ETF',
            'VNQ': 'Real Estate ETF'
        }
        
        # Analysis period
        self.start_date = datetime.now() - timedelta(days=365*10)  # 10 years
        self.end_date = datetime.now()
        
    async def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all indices."""
        print("Fetching historical data for all indices...")
        data = {}
        
        async with self.fetcher:
            # Fetch index data
            results = await self.fetcher.fetch_multiple(
                list(self.indices.keys()),
                self.start_date,
                self.end_date
            )
            
            for symbol, df in results.items():
                if not df.empty:
                    data[symbol] = df
                    print(f"✓ Fetched {symbol}: {len(df)} days of data")
                else:
                    print(f"✗ Failed to fetch {symbol}")
                    
        return data
    
    def analyze_historical_performance(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Analyze historical performance of each index."""
        print("\nAnalyzing historical performance...")
        
        results = []
        
        for symbol, df in data.items():
            if df.empty:
                continue
                
            # Calculate returns
            df['daily_return'] = df['Close'].pct_change()
            df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
            
            # Annual metrics
            annual_return = df['daily_return'].mean() * 252
            annual_volatility = df['daily_return'].std() * np.sqrt(252)
            sharpe_ratio = calculate_sharpe_ratio(df['daily_return'].values)
            max_drawdown = calculate_max_drawdown(df['Close'].values)
            
            # Market regime analysis
            bull_days = (df['daily_return'] > 0).sum()
            bear_days = (df['daily_return'] < 0).sum()
            
            # Calculate rolling volatility regimes
            df['volatility_20d'] = df['daily_return'].rolling(20).std() * np.sqrt(252)
            low_vol_days = (df['volatility_20d'] < df['volatility_20d'].quantile(0.33)).sum()
            high_vol_days = (df['volatility_20d'] > df['volatility_20d'].quantile(0.67)).sum()
            
            results.append({
                'Symbol': symbol,
                'Name': self.indices[symbol],
                'Annual_Return': annual_return,
                'Annual_Volatility': annual_volatility,
                'Sharpe_Ratio': sharpe_ratio,
                'Max_Drawdown': max_drawdown,
                'Bull_Days_Pct': bull_days / len(df) * 100,
                'Bear_Days_Pct': bear_days / len(df) * 100,
                'Low_Vol_Days_Pct': low_vol_days / len(df) * 100,
                'High_Vol_Days_Pct': high_vol_days / len(df) * 100,
                'Best_Day': df['daily_return'].max(),
                'Worst_Day': df['daily_return'].min(),
                'Total_Return': df['cumulative_return'].iloc[-1]
            })
            
        return pd.DataFrame(results).sort_values('Sharpe_Ratio', ascending=False)
    
    def analyze_correlations(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Analyze correlations between different asset classes."""
        print("\nAnalyzing asset correlations...")
        
        # Create returns matrix
        returns_data = {}
        
        for symbol, df in data.items():
            if not df.empty:
                returns_data[symbol] = df['Close'].pct_change()
                
        returns_df = pd.DataFrame(returns_data).dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Find best diversification pairs (lowest correlations)
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                symbol1 = correlation_matrix.columns[i]
                symbol2 = correlation_matrix.columns[j]
                corr = correlation_matrix.iloc[i, j]
                correlations.append({
                    'Asset1': symbol1,
                    'Asset2': symbol2,
                    'Correlation': corr
                })
                
        corr_df = pd.DataFrame(correlations).sort_values('Correlation')
        
        return correlation_matrix, corr_df
    
    def simulate_monthly_contributions(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Simulate portfolio growth with monthly contributions."""
        print("\nSimulating monthly contribution strategies...")
        
        results = []
        
        for symbol, df in data.items():
            if df.empty:
                continue
                
            # Resample to monthly
            monthly = df.resample('M').last()
            
            # Simulate DCA strategy
            portfolio_value = self.initial_capital
            shares = 0
            contributions = []
            values = []
            
            for i, (date, row) in enumerate(monthly.iterrows()):
                # Buy shares with initial capital or monthly contribution
                if i == 0:
                    investment = self.initial_capital
                else:
                    investment = self.monthly_contribution
                    
                price = row['Close']
                new_shares = investment / price
                shares += new_shares
                
                # Track portfolio value
                portfolio_value = shares * price
                contributions.append(self.initial_capital + i * self.monthly_contribution)
                values.append(portfolio_value)
                
            # Calculate metrics
            total_invested = contributions[-1]
            final_value = values[-1]
            total_return = (final_value - total_invested) / total_invested
            
            # Calculate volatility of portfolio value
            portfolio_returns = pd.Series(values).pct_change().dropna()
            portfolio_volatility = portfolio_returns.std() * np.sqrt(12)
            
            results.append({
                'Symbol': symbol,
                'Name': self.indices[symbol],
                'Total_Invested': total_invested,
                'Final_Value': final_value,
                'Total_Return': total_return,
                'Annualized_Return': (final_value / total_invested) ** (1 / (len(monthly) / 12)) - 1,
                'Portfolio_Volatility': portfolio_volatility,
                'Best_Month': portfolio_returns.max(),
                'Worst_Month': portfolio_returns.min()
            })
            
        return pd.DataFrame(results).sort_values('Annualized_Return', ascending=False)
    
    def analyze_trading_frequency(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Analyze optimal trading frequencies considering transaction costs."""
        print("\nAnalyzing trading frequency impact...")
        
        # Assume $5 commission per trade
        commission = 5
        
        results = []
        
        for symbol, df in data.items():
            if df.empty:
                continue
                
            # Test different rebalancing frequencies
            frequencies = {
                'Daily': 1,
                'Weekly': 5,
                'Bi-weekly': 10,
                'Monthly': 21,
                'Quarterly': 63
            }
            
            for freq_name, freq_days in frequencies.items():
                # Simulate trading at this frequency
                trades = len(df) // freq_days
                total_commission = trades * commission * 2  # Buy and sell
                
                # Calculate returns at this frequency
                resampled = df['Close'].iloc[::freq_days]
                returns = resampled.pct_change().dropna()
                
                # Adjust for transaction costs
                avg_position_size = self.initial_capital + (self.monthly_contribution * 60)  # Avg over 5 years
                commission_drag = total_commission / avg_position_size
                
                annual_return = returns.mean() * (252 / freq_days)
                adjusted_return = annual_return - commission_drag
                
                results.append({
                    'Symbol': symbol,
                    'Frequency': freq_name,
                    'Trades_Per_Year': 252 / freq_days,
                    'Annual_Commission': (252 / freq_days) * commission * 2,
                    'Gross_Return': annual_return,
                    'Commission_Drag': commission_drag,
                    'Net_Return': adjusted_return
                })
                
        return pd.DataFrame(results)
    
    async def analyze_meta_indicators(self) -> Dict:
        """Analyze effectiveness of meta indicators."""
        print("\nAnalyzing meta indicators effectiveness...")
        
        # Fetch Fear & Greed historical data
        fear_greed_data = await self.fear_greed.fetch_historical(limit=365*2)  # 2 years
        
        # Fetch SPY data for comparison
        spy_data = self.fetcher.fetch_sync('SPY', 
                                          datetime.now() - timedelta(days=365*2),
                                          datetime.now())
        
        # Analyze correlation
        correlation_analysis = self.fear_greed.correlation_analysis(
            fear_greed_data, 
            spy_data,
            window=20
        )
        
        # Generate signals
        fg_signals = self.fear_greed.get_signals(fear_greed_data)
        
        # Backtest meta indicator signals
        results = {
            'fear_greed': {
                'correlation': correlation_analysis['correlation'].mean(),
                'directional_accuracy': correlation_analysis['directional_accuracy'].mean(),
                'extreme_fear_returns': self._calculate_signal_returns(
                    fg_signals['extreme_fear'], spy_data
                ),
                'extreme_greed_returns': self._calculate_signal_returns(
                    fg_signals['extreme_greed'], spy_data
                )
            }
        }
        
        # Analyze max pain effectiveness
        max_pain_data = self.max_pain.calculate('SPY')
        if max_pain_data.get('max_pain_price'):
            current_price = spy_data['Close'].iloc[-1]
            max_pain_signals = self.max_pain.get_signals('SPY', current_price, max_pain_data)
            results['max_pain'] = max_pain_signals
            
        return results
    
    def _calculate_signal_returns(self, signals: pd.Series, price_data: pd.DataFrame) -> Dict:
        """Calculate returns following specific signals."""
        if signals.empty or price_data.empty:
            return {}
            
        # Align data
        aligned = price_data.copy()
        aligned['signal'] = signals
        aligned = aligned.dropna()
        
        # Calculate forward returns
        aligned['return_1d'] = aligned['Close'].shift(-1) / aligned['Close'] - 1
        aligned['return_5d'] = aligned['Close'].shift(-5) / aligned['Close'] - 1
        aligned['return_20d'] = aligned['Close'].shift(-20) / aligned['Close'] - 1
        
        # Get returns when signal is true
        signal_returns = aligned[aligned['signal'] == True]
        
        if signal_returns.empty:
            return {}
            
        return {
            'count': len(signal_returns),
            'avg_return_1d': signal_returns['return_1d'].mean(),
            'avg_return_5d': signal_returns['return_5d'].mean(),
            'avg_return_20d': signal_returns['return_20d'].mean(),
            'win_rate_1d': (signal_returns['return_1d'] > 0).mean(),
            'win_rate_5d': (signal_returns['return_5d'] > 0).mean(),
            'win_rate_20d': (signal_returns['return_20d'] > 0).mean()
        }
    
    def generate_recommendations(self, 
                               performance_df: pd.DataFrame,
                               correlation_matrix: pd.DataFrame,
                               dca_results: pd.DataFrame,
                               frequency_analysis: pd.DataFrame,
                               meta_indicators: Dict) -> Dict:
        """Generate investment recommendations based on analysis."""
        print("\nGenerating recommendations...")
        
        recommendations = {
            'best_single_asset': performance_df.iloc[0].to_dict(),
            'best_risk_adjusted': performance_df.nlargest(3, 'Sharpe_Ratio')[['Symbol', 'Name', 'Sharpe_Ratio']].to_dict('records'),
            'best_for_dca': dca_results.iloc[0].to_dict(),
            'optimal_frequency': self._determine_optimal_frequency(frequency_analysis),
            'diversification_pairs': self._find_best_pairs(correlation_matrix),
            'meta_indicator_usage': self._analyze_meta_effectiveness(meta_indicators),
            'position_sizing': self._calculate_position_sizing(),
            'risk_management': self._generate_risk_rules()
        }
        
        return recommendations
    
    def _determine_optimal_frequency(self, freq_df: pd.DataFrame) -> Dict:
        """Determine optimal trading frequency."""
        # Group by frequency and average across all symbols
        freq_summary = freq_df.groupby('Frequency')['Net_Return'].mean().sort_values(ascending=False)
        
        return {
            'recommended': freq_summary.index[0],
            'net_return': freq_summary.iloc[0],
            'rationale': "Balances returns with transaction costs"
        }
    
    def _find_best_pairs(self, corr_matrix: pd.DataFrame) -> List[Dict]:
        """Find best asset pairs for diversification."""
        pairs = []
        
        # Find low correlation pairs
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr = corr_matrix.iloc[i, j]
                if corr < 0.5:  # Low correlation threshold
                    pairs.append({
                        'asset1': corr_matrix.index[i],
                        'asset2': corr_matrix.columns[j],
                        'correlation': corr,
                        'allocation': '60/40'  # Simple allocation
                    })
                    
        return sorted(pairs, key=lambda x: x['correlation'])[:3]
    
    def _analyze_meta_effectiveness(self, meta_data: Dict) -> Dict:
        """Analyze meta indicator effectiveness."""
        recommendations = {}
        
        if 'fear_greed' in meta_data:
            fg = meta_data['fear_greed']
            if fg.get('extreme_fear_returns', {}).get('win_rate_20d', 0) > 0.6:
                recommendations['fear_greed'] = {
                    'use': True,
                    'strategy': 'Buy on extreme fear (< 20)',
                    'expected_win_rate': fg['extreme_fear_returns']['win_rate_20d']
                }
            else:
                recommendations['fear_greed'] = {
                    'use': False,
                    'reason': 'Insufficient predictive power'
                }
                
        return recommendations
    
    def _calculate_position_sizing(self) -> Dict:
        """Calculate optimal position sizing with monthly contributions."""
        return {
            'initial_allocation': {
                'stocks': 0.8,
                'bonds': 0.15,
                'alternatives': 0.05
            },
            'monthly_contribution_split': {
                'primary_holding': 0.7,
                'secondary_holding': 0.2,
                'opportunistic': 0.1
            },
            'rebalancing': 'Quarterly or when allocation drifts > 10%',
            'max_position_size': 0.3,  # No single position > 30%
            'min_position_size': 0.05  # No position < 5%
        }
    
    def _generate_risk_rules(self) -> Dict:
        """Generate risk management rules."""
        return {
            'stop_loss': None,  # Not recommended for long-term DCA
            'portfolio_stop': -25,  # Consider strategy review if down 25%
            'volatility_adjustment': 'Reduce equity allocation by 10% when VIX > 30',
            'correlation_monitoring': 'Review allocations if correlations increase > 0.7',
            'max_leverage': 1.0,  # No leverage for retail accounts
            'emergency_fund': '6 months expenses before investing'
        }
    
    async def run_analysis(self):
        """Run complete market research analysis."""
        print("Starting comprehensive market research analysis...")
        print(f"Initial Capital: ${self.initial_capital:,}")
        print(f"Monthly Contribution: ${self.monthly_contribution:,}")
        print("="*60)
        
        # Fetch all data
        market_data = await self.fetch_all_data()
        
        # Run analyses
        performance_analysis = self.analyze_historical_performance(market_data)
        correlation_matrix, correlation_pairs = self.analyze_correlations(market_data)
        dca_simulation = self.simulate_monthly_contributions(market_data)
        frequency_analysis = self.analyze_trading_frequency(market_data)
        meta_indicators = await self.analyze_meta_indicators()
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            performance_analysis,
            correlation_matrix,
            dca_simulation,
            frequency_analysis,
            meta_indicators
        )
        
        # Display results
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        
        print("\n📊 Historical Performance Analysis:")
        print(performance_analysis.to_string())
        
        print("\n💰 DCA Simulation Results:")
        print(dca_simulation[['Symbol', 'Name', 'Total_Invested', 'Final_Value', 'Annualized_Return']].to_string())
        
        print("\n🎯 Recommendations:")
        print(f"\nBest Single Asset: {recommendations['best_single_asset']['Symbol']} ({recommendations['best_single_asset']['Name']})")
        print(f"- Annual Return: {recommendations['best_single_asset']['Annual_Return']:.2%}")
        print(f"- Sharpe Ratio: {recommendations['best_single_asset']['Sharpe_Ratio']:.2f}")
        
        print(f"\nOptimal Trading Frequency: {recommendations['optimal_frequency']['recommended']}")
        print(f"- Expected Net Return: {recommendations['optimal_frequency']['net_return']:.2%}")
        
        print("\nDiversification Recommendations:")
        for pair in recommendations['diversification_pairs']:
            print(f"- {pair['asset1']}/{pair['asset2']} (correlation: {pair['correlation']:.2f})")
            
        # Save detailed report
        self._save_report(
            performance_analysis,
            correlation_matrix,
            dca_simulation,
            frequency_analysis,
            meta_indicators,
            recommendations
        )
        
        return recommendations
    
    def _save_report(self, *args):
        """Save analysis results to file."""
        # This will be used by the agent to create the markdown report
        pass


async def main():
    """Run the market research analysis."""
    analyzer = MarketResearchAnalyzer()
    recommendations = await analyzer.run_analysis()
    
    print("\n✅ Analysis complete! See detailed recommendations above.")
    print("📄 Full report will be saved to docs/strategies/MARKET_RESEARCH.md")
    
    return recommendations


if __name__ == "__main__":
    # Run the analysis
    asyncio.run(main())