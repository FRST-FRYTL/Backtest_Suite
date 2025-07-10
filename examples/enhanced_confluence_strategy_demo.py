"""
Enhanced Confluence Strategy Demo

This script demonstrates the enhanced confluence strategy with multi-timeframe
analysis, proper baselines, and comprehensive reporting.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

# Add src to path for imports
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from data.multi_timeframe_data_manager import MultiTimeframeDataManager, Timeframe
from strategies.enhanced_confluence_engine import EnhancedConfluenceEngine
from analysis.baseline_comparisons import BaselineComparison
from analysis.enhanced_trade_tracker import EnhancedTradeTracker, ExitReason
from visualization.enhanced_interactive_charts import EnhancedInteractiveCharts
from indicators.technical_indicators import TechnicalIndicators

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedConfluenceStrategyDemo:
    """
    Comprehensive demo of the enhanced confluence strategy.
    """
    
    def __init__(self):
        """Initialize the demo components."""
        self.data_manager = MultiTimeframeDataManager()
        self.confluence_engine = EnhancedConfluenceEngine(self.data_manager)
        self.baseline_comparison = BaselineComparison()
        self.trade_tracker = EnhancedTradeTracker()
        self.charts = EnhancedInteractiveCharts()
        self.indicators_calc = TechnicalIndicators()
        
        # Demo configuration
        self.symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'GLD']
        self.start_date = '2020-01-01'
        self.end_date = '2024-01-01'
        self.initial_capital = 10000
        self.position_size = 0.20  # 20% per position
        
    async def run_complete_demo(self):
        """Run the complete enhanced confluence strategy demo."""
        logger.info("🚀 Starting Enhanced Confluence Strategy Demo")
        
        try:
            # Step 1: Load multi-timeframe data
            logger.info("📊 Loading multi-timeframe data...")
            data_by_symbol = await self._load_data()
            
            # Step 2: Create baselines
            logger.info("📈 Creating baseline comparisons...")
            baselines = await self._create_baselines()
            
            # Step 3: Run confluence analysis
            logger.info("🎯 Running confluence analysis...")
            confluence_results = await self._run_confluence_analysis(data_by_symbol)
            
            # Step 4: Simulate trading
            logger.info("💰 Simulating enhanced trading strategy...")
            strategy_results = await self._simulate_trading(confluence_results, data_by_symbol)
            
            # Step 5: Generate comprehensive analysis
            logger.info("📊 Generating comprehensive analysis...")
            analysis_results = await self._generate_analysis(strategy_results, baselines)
            
            # Step 6: Create visualizations
            logger.info("🎨 Creating interactive visualizations...")
            await self._create_visualizations(analysis_results, confluence_results, data_by_symbol)
            
            # Step 7: Generate final report
            logger.info("📋 Generating final report...")
            await self._generate_final_report(analysis_results)
            
            logger.info("✅ Enhanced Confluence Strategy Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
    
    async def _load_data(self):
        """Load multi-timeframe data for all symbols."""
        timeframes = [Timeframe.DAY_1, Timeframe.WEEK_1, Timeframe.MONTH_1]
        
        data_by_symbol = await self.data_manager.load_multi_timeframe_data(
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            timeframes=timeframes
        )
        
        # Validate data quality
        for symbol, data_by_tf in data_by_symbol.items():
            validation = self.data_manager.validate_data_quality(data_by_tf)
            logger.info(f"Data validation for {symbol}:")
            for tf, results in validation.items():
                logger.info(f"  {tf.value}: {results['data_points']} points, "
                           f"{results['missing_percentage']:.1f}% missing")
        
        return data_by_symbol
    
    async def _create_baselines(self):
        """Create comprehensive baseline comparisons."""
        baselines = {}
        
        # SPY Buy-and-Hold
        baselines['spy_buy_hold'] = self.baseline_comparison.create_buy_hold_baseline(
            symbol='SPY',
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            monthly_contribution=500
        )
        
        # Equal Weight Portfolio
        baselines['equal_weight'] = self.baseline_comparison.create_equal_weight_portfolio(
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            monthly_contribution=500
        )
        
        # 60/40 Portfolio
        baselines['60_40'] = self.baseline_comparison.create_60_40_portfolio(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            monthly_contribution=500
        )
        
        # Log baseline results
        for name, baseline in baselines.items():
            logger.info(f"{name}: {baseline.total_return:.2f}% total return, "
                       f"Sharpe: {baseline.sharpe_ratio:.2f}")
        
        return baselines
    
    async def _run_confluence_analysis(self, data_by_symbol):
        """Run confluence analysis for all symbols."""
        confluence_results = {}
        
        for symbol in self.symbols:
            try:
                logger.info(f"Analyzing confluence for {symbol}...")
                
                # Calculate confluence scores
                confluence_df = self.confluence_engine.calculate_confluence_scores(symbol)
                
                if not confluence_df.empty:
                    confluence_results[symbol] = confluence_df
                    
                    # Log summary
                    summary = self.confluence_engine.get_confluence_summary(confluence_df)
                    logger.info(f"{symbol} confluence summary:")
                    logger.info(f"  Avg score: {summary.get('avg_confluence_score', 0):.3f}")
                    logger.info(f"  Strong signals: {summary.get('strong_signals', 0)}")
                    logger.info(f"  Above threshold: {summary.get('above_threshold', 0)}")
                else:
                    logger.warning(f"No confluence data generated for {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                continue
        
        return confluence_results
    
    async def _simulate_trading(self, confluence_results, data_by_symbol):
        """Simulate trading based on confluence signals."""
        portfolio_value = self.initial_capital
        monthly_contribution = 500
        positions = {}  # symbol -> shares
        
        # Combine all dates for simulation
        all_dates = set()
        for symbol, confluence_df in confluence_results.items():
            all_dates.update(confluence_df.index)
        
        simulation_dates = sorted(all_dates)
        portfolio_history = []
        
        for date in simulation_dates:
            # Monthly contribution (simplified)
            if date.day <= 5:  # First week of month
                portfolio_value += monthly_contribution
            
            # Check for signals
            for symbol, confluence_df in confluence_results.items():
                if date not in confluence_df.index:
                    continue
                
                row = confluence_df.loc[date]
                confluence_score = row['confluence_score']
                close_price = row['close_price']
                
                # Entry signal
                if (confluence_score >= 0.65 and 
                    symbol not in positions and 
                    row['signal_strength'] in ['moderate', 'strong']):
                    
                    # Calculate position size
                    position_value = portfolio_value * self.position_size
                    shares = position_value / close_price
                    
                    # Record trade entry
                    trade_id = self.trade_tracker.record_trade_entry(
                        symbol=symbol,
                        price=close_price,
                        shares=shares,
                        confluence_score=confluence_score,
                        timeframe_scores={
                            '1D': row.get('timeframe_1D', 0),
                            '1W': row.get('timeframe_1W', 0),
                            '1M': row.get('timeframe_1M', 0)
                        },
                        indicators={
                            'close': close_price,
                            'confluence': confluence_score
                        },
                        signal_components={
                            'trend': row.get('trend_score', 0),
                            'momentum': row.get('momentum_score', 0),
                            'volume': row.get('volume_score', 0),
                            'volatility': row.get('volatility_score', 0)
                        },
                        stop_loss=close_price * 0.95,  # 5% stop loss
                        take_profit=close_price * 1.15,  # 15% take profit
                        position_size_pct=self.position_size
                    )
                    
                    positions[symbol] = {
                        'shares': shares,
                        'entry_price': close_price,
                        'trade_id': trade_id,
                        'entry_date': date
                    }
                    
                    portfolio_value -= position_value
                    logger.info(f"🟢 ENTRY: {symbol} @ {close_price:.2f} "
                               f"(confluence: {confluence_score:.3f})")
                
                # Exit signal (simplified)
                elif (symbol in positions and 
                      (confluence_score < 0.45 or  # Confluence deteriorated
                       close_price <= positions[symbol]['entry_price'] * 0.95 or  # Stop loss
                       close_price >= positions[symbol]['entry_price'] * 1.15 or  # Take profit
                       (date - positions[symbol]['entry_date']).days > 30)):  # Time exit
                    
                    position = positions[symbol]
                    exit_value = position['shares'] * close_price
                    
                    # Determine exit reason
                    if close_price <= position['entry_price'] * 0.95:
                        exit_reason = ExitReason.STOP_LOSS
                    elif close_price >= position['entry_price'] * 1.15:
                        exit_reason = ExitReason.TAKE_PROFIT
                    elif (date - position['entry_date']).days > 30:
                        exit_reason = ExitReason.TIME_EXIT
                    else:
                        exit_reason = ExitReason.CONFLUENCE_EXIT
                    
                    # Record trade exit
                    self.trade_tracker.record_trade_exit(
                        trade_id=position['trade_id'],
                        price=close_price,
                        exit_reason=exit_reason,
                        exit_trigger=f"{exit_reason.value}_trigger",
                        exit_confluence=confluence_score,
                        market_return=0.0  # Simplified
                    )
                    
                    portfolio_value += exit_value
                    return_pct = (close_price / position['entry_price'] - 1) * 100
                    
                    logger.info(f"🔴 EXIT: {symbol} @ {close_price:.2f} "
                               f"({return_pct:+.2f}%, reason: {exit_reason.value})")
                    
                    del positions[symbol]
            
            # Calculate current portfolio value
            current_value = portfolio_value
            for symbol, position in positions.items():
                if symbol in confluence_results and date in confluence_results[symbol].index:
                    current_price = confluence_results[symbol].loc[date, 'close_price']
                    current_value += position['shares'] * current_price
            
            portfolio_history.append({
                'date': date,
                'portfolio_value': current_value,
                'cash': portfolio_value,
                'positions': len(positions)
            })
        
        # Calculate strategy performance
        portfolio_df = pd.DataFrame(portfolio_history).set_index('date')
        
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        returns = portfolio_df['portfolio_value'].pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        max_drawdown = self._calculate_max_drawdown(portfolio_df['portfolio_value'])
        
        strategy_results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': returns.std() * np.sqrt(252) * 100,
            'portfolio_history': portfolio_df,
            'trade_summary': self.trade_tracker.get_trade_summary_statistics()
        }
        
        logger.info(f"Strategy Results:")
        logger.info(f"  Total Return: {total_return:.2f}%")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2f}%")
        logger.info(f"  Total Trades: {strategy_results['trade_summary'].get('total_trades', 0)}")
        
        return strategy_results
    
    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown."""
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve / rolling_max - 1) * 100
        return drawdown.min()
    
    async def _generate_analysis(self, strategy_results, baselines):
        """Generate comprehensive analysis comparing strategy to baselines."""
        
        # Convert strategy results to BaselineResults format for comparison
        from analysis.baseline_comparisons import BaselineResults
        
        strategy_baseline = BaselineResults(
            strategy_name="Enhanced Confluence Strategy",
            total_return=strategy_results['total_return'],
            annual_return=strategy_results['total_return'],  # Simplified
            volatility=strategy_results['volatility'],
            sharpe_ratio=strategy_results['sharpe_ratio'],
            max_drawdown=strategy_results['max_drawdown'],
            calmar_ratio=abs(strategy_results['total_return'] / strategy_results['max_drawdown']) if strategy_results['max_drawdown'] != 0 else 0,
            sortino_ratio=strategy_results['sharpe_ratio'],  # Simplified
            var_95=0,  # Would need daily returns
            cvar_95=0,  # Would need daily returns
            total_trades=strategy_results['trade_summary'].get('total_trades', 0),
            total_contributions=self.initial_capital,
            dividend_income=0,
            transaction_costs=0,
            equity_curve=strategy_results['portfolio_history']['portfolio_value'],
            monthly_returns=pd.Series(dtype=float),
            drawdown_series=pd.Series(dtype=float)
        )
        
        # Compare with baselines
        baseline_list = list(baselines.values())
        comparisons = self.baseline_comparison.compare_strategies(strategy_baseline, baseline_list)
        
        analysis_results = {
            'strategy_results': strategy_baseline,
            'baseline_results': baseline_list,
            'comparisons': comparisons,
            'trade_analysis': self.trade_tracker.generate_trade_analysis_report()
        }
        
        # Log key comparisons
        logger.info("📊 Strategy vs Baseline Comparisons:")
        for baseline_name, comparison in comparisons.items():
            alpha = comparison['alpha_total_return']
            logger.info(f"  vs {baseline_name}: {alpha:+.2f}% alpha")
        
        return analysis_results
    
    async def _create_visualizations(self, analysis_results, confluence_results, data_by_symbol):
        """Create comprehensive interactive visualizations."""
        
        # Create master trading view for SPY
        if 'SPY' in confluence_results and 'SPY' in data_by_symbol:
            spy_data = data_by_symbol['SPY']
            spy_confluence = confluence_results['SPY']
            spy_trades = [t for t in self.trade_tracker.completed_trades if t.entry.symbol == 'SPY']
            
            # Calculate indicators for visualization
            spy_indicators = {}
            for tf, data in spy_data.items():
                spy_indicators[tf] = {
                    'rsi': self.indicators_calc.rsi(data['close']),
                    'vwap': self.indicators_calc.vwap(data['high'], data['low'], data['close'], data['volume']),
                    'sma_20': self.indicators_calc.sma(data['close'], 20),
                    'sma_50': self.indicators_calc.sma(data['close'], 50)
                }
            
            master_chart = self.charts.create_master_trading_view(
                data_by_timeframe=spy_data,
                confluence_scores=spy_confluence,
                trades=spy_trades,
                indicators=spy_indicators
            )
            self.charts.save_chart(master_chart, 'enhanced_confluence_master_view.html')
        
        # Create confluence heatmap
        all_trades = self.trade_tracker.completed_trades
        if confluence_results:
            first_confluence = list(confluence_results.values())[0]
            heatmap_chart = self.charts.create_confluence_heatmap(first_confluence, all_trades)
            self.charts.save_chart(heatmap_chart, 'confluence_performance_heatmap.html')
        
        # Create timeframe participation radar
        radar_chart = self.charts.create_timeframe_participation_radar(all_trades)
        self.charts.save_chart(radar_chart, 'timeframe_participation_radar.html')
        
        # Create trade performance dashboard
        dashboard = self.charts.create_trade_performance_dashboard(self.trade_tracker)
        self.charts.save_chart(dashboard, 'trade_performance_dashboard.html')
        
        # Create baseline comparison chart
        baseline_chart = self.charts.create_baseline_comparison_chart(
            strategy_results=analysis_results['strategy_results'].__dict__,
            baseline_results=analysis_results['baseline_results']
        )
        self.charts.save_chart(baseline_chart, 'baseline_comparison_chart.html')
        
        logger.info("📊 Created 5 interactive visualization files")
    
    async def _generate_final_report(self, analysis_results):
        """Generate comprehensive final report."""
        
        # Export detailed trade records
        trade_file = self.trade_tracker.export_trade_details()
        logger.info(f"📁 Exported trade details to: {trade_file}")
        
        # Create summary report
        summary = {
            'demo_completed': datetime.now().isoformat(),
            'strategy_performance': {
                'total_return': analysis_results['strategy_results'].total_return,
                'sharpe_ratio': analysis_results['strategy_results'].sharpe_ratio,
                'max_drawdown': analysis_results['strategy_results'].max_drawdown,
                'total_trades': analysis_results['strategy_results'].total_trades
            },
            'key_improvements': [
                f"Multi-timeframe analysis with proper VWAP implementation",
                f"Comprehensive baseline comparisons",
                f"Detailed trade tracking with confluence breakdown",
                f"Interactive visualizations",
                f"Statistical validation framework"
            ],
            'performance_vs_baselines': {},
            'files_generated': [
                'enhanced_confluence_master_view.html',
                'confluence_performance_heatmap.html',
                'timeframe_participation_radar.html',
                'trade_performance_dashboard.html',
                'baseline_comparison_chart.html',
                trade_file
            ]
        }
        
        # Add baseline comparisons
        for baseline_name, comparison in analysis_results['comparisons'].items():
            summary['performance_vs_baselines'][baseline_name] = {
                'alpha_total_return': comparison['alpha_total_return'],
                'sharpe_ratio_improvement': comparison['sharpe_ratio_diff'],
                'information_ratio': comparison['information_ratio']
            }
        
        # Save summary report
        import json
        summary_file = Path("reports") / "enhanced_confluence_demo_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📋 Generated final summary report: {summary_file}")
        
        # Print key results
        print("\n" + "="*80)
        print("🎯 ENHANCED CONFLUENCE STRATEGY DEMO RESULTS")
        print("="*80)
        print(f"📈 Total Return: {summary['strategy_performance']['total_return']:.2f}%")
        print(f"📊 Sharpe Ratio: {summary['strategy_performance']['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {summary['strategy_performance']['max_drawdown']:.2f}%")
        print(f"🔄 Total Trades: {summary['strategy_performance']['total_trades']}")
        print(f"\n📁 Files Generated: {len(summary['files_generated'])}")
        for file in summary['files_generated']:
            print(f"   • {file}")
        print("\n🚀 Demo completed successfully!")
        print("="*80)

async def main():
    """Run the enhanced confluence strategy demo."""
    demo = EnhancedConfluenceStrategyDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    asyncio.run(main())