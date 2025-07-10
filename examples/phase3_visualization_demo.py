"""
Phase 3 Visualization and Reporting Demo

This script demonstrates all visualization and reporting components
created in Phase 3 of the enhanced confluence strategy.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import visualization modules
from visualization.multi_timeframe_chart import MultiTimeframeMasterChart
from visualization.confluence_charts import ConfluenceCharts
from visualization.trade_explorer import InteractiveTradeExplorer
from visualization.timeframe_charts import TimeframeCharts
from visualization.executive_summary import ExecutiveSummaryDashboard
from visualization.performance_report import PerformanceAnalysisReport
from visualization.benchmark_comparison import BenchmarkComparison
from visualization.export_utils import ExportManager

# Import data structures
from data.multi_timeframe_data_manager import Timeframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase3VisualizationDemo:
    """
    Demonstrates Phase 3 visualization and reporting capabilities.
    """
    
    def __init__(self):
        """Initialize demo components."""
        self.master_chart = MultiTimeframeMasterChart()
        self.confluence_charts = ConfluenceCharts()
        self.trade_explorer = InteractiveTradeExplorer()
        self.timeframe_charts = TimeframeCharts()
        self.executive_summary = ExecutiveSummaryDashboard()
        self.performance_report = PerformanceAnalysisReport()
        self.benchmark_comparison = BenchmarkComparison()
        self.export_manager = ExportManager('reports/phase3_demo')
        
        # Demo data parameters
        self.start_date = '2023-01-01'
        self.end_date = '2024-01-01'
        self.symbol = 'DEMO'
        
    def run_complete_demo(self):
        """Run complete Phase 3 visualization demo."""
        logger.info("🚀 Starting Phase 3 Visualization Demo")
        
        try:
            # Generate demo data
            logger.info("📊 Generating demo data...")
            demo_data = self._generate_demo_data()
            
            # 1. Multi-Timeframe Master Chart
            logger.info("\n📈 Creating Multi-Timeframe Master Chart...")
            master_fig = self.master_chart.create_master_chart(
                demo_data['data_by_timeframe'],
                demo_data['indicators_by_timeframe'],
                demo_data['trades'],
                self.symbol
            )
            self.master_chart.save_chart(master_fig, 'master_chart.html')
            
            # 2. Confluence Analysis Charts
            logger.info("\n🎯 Creating Confluence Analysis Charts...")
            confluence_figs = self._create_confluence_charts(demo_data)
            
            # 3. Interactive Trade Explorer
            logger.info("\n🔍 Creating Interactive Trade Explorer...")
            trade_figs = self._create_trade_explorer(demo_data['trades'])
            
            # 4. Timeframe Participation Visualizations
            logger.info("\n⏰ Creating Timeframe Participation Charts...")
            timeframe_figs = self._create_timeframe_charts(demo_data['trades'])
            
            # 5. Executive Summary Dashboard
            logger.info("\n📊 Creating Executive Summary Dashboard...")
            executive_html = self._create_executive_summary(demo_data)
            
            # 6. Performance Analysis Reports
            logger.info("\n📈 Creating Performance Analysis Reports...")
            performance_figs = self._create_performance_reports(demo_data)
            
            # 7. Benchmark Comparison
            logger.info("\n🎯 Creating Benchmark Comparison...")
            benchmark_figs = self._create_benchmark_comparison(demo_data)
            
            # 8. Export Data
            logger.info("\n💾 Exporting data in multiple formats...")
            export_results = self._export_all_data(demo_data, executive_html)
            
            # Generate summary report
            self._generate_summary_report(export_results)
            
            logger.info("\n✅ Phase 3 Visualization Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_demo_data(self):
        """Generate synthetic demo data."""
        # Create date range
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        # Generate multi-timeframe data
        data_by_timeframe = {}
        indicators_by_timeframe = {}
        
        # Base price series
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * (1 + returns).cumprod()
        
        # Generate data for each timeframe
        for timeframe in [Timeframe.HOUR_1, Timeframe.HOUR_4, Timeframe.DAY_1, 
                         Timeframe.WEEK_1, Timeframe.MONTH_1]:
            # Resample based on timeframe
            if timeframe == Timeframe.HOUR_1:
                tf_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='h')
                tf_prices = np.interp(
                    np.linspace(0, len(prices)-1, len(tf_dates)),
                    np.arange(len(prices)),
                    prices
                )
            else:
                freq_map = {
                    Timeframe.HOUR_4: '4h',
                    Timeframe.DAY_1: 'D',
                    Timeframe.WEEK_1: 'W',
                    Timeframe.MONTH_1: 'ME'
                }
                tf_dates = pd.date_range(
                    start=self.start_date, 
                    end=self.end_date, 
                    freq=freq_map[timeframe]
                )
                # Properly resample prices
                step = max(1, len(dates) // len(tf_dates))
                tf_prices = prices[::step][:len(tf_dates)]
                if len(tf_prices) < len(tf_dates):
                    # Interpolate if needed
                    tf_prices = np.interp(
                        np.linspace(0, len(prices)-1, len(tf_dates)),
                        np.arange(len(prices)),
                        prices
                    )
            
            # Create OHLCV data
            ohlcv = pd.DataFrame(index=tf_dates)
            ohlcv['open'] = tf_prices * (1 + np.random.uniform(-0.001, 0.001, len(tf_prices)))
            ohlcv['high'] = tf_prices * (1 + np.random.uniform(0, 0.01, len(tf_prices)))
            ohlcv['low'] = tf_prices * (1 - np.random.uniform(0, 0.01, len(tf_prices)))
            ohlcv['close'] = tf_prices
            ohlcv['volume'] = np.random.lognormal(17, 0.5, len(tf_prices))
            
            data_by_timeframe[timeframe] = ohlcv
            
            # Generate indicators
            indicators = {}
            indicators['sma_20'] = ohlcv['close'].rolling(20).mean()
            indicators['sma_50'] = ohlcv['close'].rolling(50).mean()
            indicators['sma_200'] = ohlcv['close'].rolling(200).mean()
            
            # Bollinger Bands
            sma = ohlcv['close'].rolling(20).mean()
            std = ohlcv['close'].rolling(20).std()
            indicators['bb_upper'] = sma + 2 * std
            indicators['bb_lower'] = sma - 2 * std
            
            indicators_by_timeframe[timeframe] = indicators
        
        # Generate trades
        trades = self._generate_demo_trades(dates, prices)
        
        # Generate confluence history
        confluence_history = self._generate_confluence_history(dates)
        
        # Generate equity curve
        equity_curve = pd.Series(
            10000 * (1 + returns).cumprod(),
            index=dates
        )
        
        # Generate strategy metrics
        strategy_metrics = {
            'total_return': 0.35,
            'annual_return': 0.35,
            'sharpe_ratio': 1.8,
            'max_drawdown': -0.08,
            'volatility': 0.19,
            'win_rate': 0.62,
            'total_trades': len(trades),
            'profit_factor': 1.9,
            'calmar_ratio': 4.375,
            'sortino_ratio': 2.5,
            'var_95': -0.025,
            'cvar_95': -0.035,
            'beta': 0.8,
            'correlation': 0.65
        }
        
        # Generate benchmark results
        benchmark_results = self._generate_benchmark_results(dates, returns)
        
        return {
            'data_by_timeframe': data_by_timeframe,
            'indicators_by_timeframe': indicators_by_timeframe,
            'trades': trades,
            'confluence_history': confluence_history,
            'equity_curve': equity_curve,
            'strategy_metrics': strategy_metrics,
            'benchmark_results': benchmark_results
        }
    
    def _generate_demo_trades(self, dates, prices):
        """Generate synthetic trades."""
        trades = []
        
        # Generate 50 trades
        for i in range(50):
            entry_idx = i * (len(dates) // 50)
            exit_idx = min(entry_idx + np.random.randint(5, 30), len(dates) - 1)
            
            entry_price = prices[entry_idx]
            exit_price = prices[exit_idx]
            
            # Add some randomness
            entry_price *= (1 + np.random.uniform(-0.002, 0.002))
            exit_price *= (1 + np.random.uniform(-0.01, 0.01))
            
            trade_return = (exit_price - entry_price) / entry_price
            
            trade = {
                'trade_id': f'T{i+1:03d}',
                'symbol': self.symbol,
                'entry_time': dates[entry_idx],
                'exit_time': dates[exit_idx],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': 0.1 + np.random.uniform(0, 0.1),
                'return': trade_return,
                'pnl': trade_return * 1000,
                'confluence_score': 0.6 + np.random.uniform(0, 0.3),
                'hold_days': (dates[exit_idx] - dates[entry_idx]).days,
                'exit_reason': np.random.choice(['take_profit', 'stop_loss', 'signal']),
                'timeframe_scores': {
                    '1H': np.random.random(),
                    '4H': np.random.random(),
                    '1D': np.random.random(),
                    '1W': np.random.random(),
                    '1M': np.random.random()
                },
                'component_scores': {
                    'trend': np.random.random(),
                    'momentum': np.random.random(),
                    'volume': np.random.random(),
                    'volatility': np.random.random()
                },
                'max_profit': abs(trade_return) * 1.5,
                'max_loss': -abs(trade_return) * 0.5
            }
            
            trades.append(trade)
        
        return trades
    
    def _generate_confluence_history(self, dates):
        """Generate confluence score history."""
        # Create DataFrame with scores for each timeframe
        confluence_df = pd.DataFrame(index=dates[::4])  # Sample every 4 days
        
        for tf in ['1H', '4H', '1D', '1W', '1M']:
            # Generate correlated random walk
            scores = [0.5]
            for _ in range(len(confluence_df) - 1):
                change = np.random.normal(0, 0.05)
                new_score = max(0, min(1, scores[-1] + change))
                scores.append(new_score)
            
            confluence_df[tf] = scores
        
        return confluence_df
    
    def _generate_benchmark_results(self, dates, base_returns):
        """Generate benchmark results."""
        benchmarks = {}
        
        # Buy & Hold
        buy_hold_returns = base_returns * 0.8  # Slightly worse than strategy
        benchmarks['buy_hold'] = {
            'total_return': 0.28,
            'annual_return': 0.28,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.12,
            'volatility': 0.23,
            'equity_curve': pd.Series(
                10000 * (1 + buy_hold_returns).cumprod(),
                index=dates
            )
        }
        
        # SPY
        spy_returns = base_returns * 0.7
        benchmarks['spy'] = {
            'total_return': 0.22,
            'annual_return': 0.22,
            'sharpe_ratio': 0.95,
            'max_drawdown': -0.15,
            'volatility': 0.25,
            'equity_curve': pd.Series(
                10000 * (1 + spy_returns).cumprod(),
                index=dates
            )
        }
        
        # 60/40 Portfolio
        balanced_returns = base_returns * 0.6
        benchmarks['60_40'] = {
            'total_return': 0.18,
            'annual_return': 0.18,
            'sharpe_ratio': 1.1,
            'max_drawdown': -0.08,
            'volatility': 0.16,
            'equity_curve': pd.Series(
                10000 * (1 + balanced_returns).cumprod(),
                index=dates
            )
        }
        
        return benchmarks
    
    def _create_confluence_charts(self, demo_data):
        """Create all confluence analysis charts."""
        charts = {}
        
        # Confluence heatmap
        charts['heatmap'] = self.confluence_charts.create_confluence_heatmap(
            demo_data['confluence_history'],
            {'1H': 0.05, '4H': 0.15, '1D': 0.20, '1W': 0.25, '1M': 0.35}
        )
        
        # Component breakdown
        component_scores = {
            'trend': demo_data['confluence_history']['1D'] * 0.4,
            'momentum': demo_data['confluence_history']['1D'] * 0.3,
            'volume': demo_data['confluence_history']['1D'] * 0.2,
            'volatility': demo_data['confluence_history']['1D'] * 0.1
        }
        charts['components'] = self.confluence_charts.create_component_breakdown(
            component_scores,
            {'trend': 0.4, 'momentum': 0.3, 'volume': 0.2, 'volatility': 0.1}
        )
        
        # Confluence distribution
        all_scores = pd.Series([t['confluence_score'] for t in demo_data['trades']])
        trade_returns = pd.Series([t['return'] for t in demo_data['trades']])
        charts['distribution'] = self.confluence_charts.create_confluence_distribution(
            all_scores, trade_returns
        )
        
        # Signal quality timeline
        charts['signal_quality'] = self.confluence_charts.create_signal_quality_timeline(
            demo_data['trades']
        )
        
        # Save all charts
        self.confluence_charts.save_all_charts(charts)
        
        return charts
    
    def _create_trade_explorer(self, trades):
        """Create trade explorer visualizations."""
        figs = {}
        
        # Interactive trade table
        figs['trade_table'] = self.trade_explorer.create_trade_table(trades)
        
        # Trade scatter matrix
        figs['scatter_matrix'] = self.trade_explorer.create_trade_scatter_matrix(trades)
        
        # Trade performance summary
        figs['performance_summary'] = self.trade_explorer.create_trade_performance_summary(trades)
        
        # Save all figures
        for name, fig in figs.items():
            self.trade_explorer.save_explorer(fig, f"trade_explorer_{name}.html")
        
        return figs
    
    def _create_timeframe_charts(self, trades):
        """Create timeframe participation charts."""
        charts = {}
        
        # Participation chart
        charts['participation'] = self.timeframe_charts.create_timeframe_participation_chart(trades)
        
        # Importance pie chart
        charts['importance'] = self.timeframe_charts.create_timeframe_importance_pie(trades)
        
        # Participation timeline
        charts['timeline'] = self.timeframe_charts.create_participation_timeline(trades)
        
        # Performance analysis
        charts['performance'] = self.timeframe_charts.create_timeframe_performance_analysis(trades)
        
        # Correlation matrix
        charts['correlation'] = self.timeframe_charts.create_timeframe_correlation_matrix(trades)
        
        # Save all charts
        self.timeframe_charts.save_all_charts(charts)
        
        return charts
    
    def _create_executive_summary(self, demo_data):
        """Create executive summary dashboard."""
        html_content = self.executive_summary.create_dashboard(
            demo_data['strategy_metrics'],
            demo_data['benchmark_results']['buy_hold'],
            demo_data['trades'],
            demo_data['equity_curve']
        )
        
        # Save dashboard
        filepath = self.executive_summary.save_dashboard(html_content)
        
        return html_content
    
    def _create_performance_reports(self, demo_data):
        """Create performance analysis reports."""
        figs = self.performance_report.create_performance_report(
            demo_data['trades'],
            demo_data['equity_curve'],
            demo_data['strategy_metrics']
        )
        
        # Save all figures
        self.performance_report.save_all_figures(figs)
        
        return figs
    
    def _create_benchmark_comparison(self, demo_data):
        """Create benchmark comparison visualizations."""
        figs = self.benchmark_comparison.create_benchmark_comparison(
            {**demo_data['strategy_metrics'], 'equity_curve': demo_data['equity_curve']},
            demo_data['benchmark_results']
        )
        
        # Save all comparisons
        self.benchmark_comparison.save_all_comparisons(figs)
        
        return figs
    
    def _export_all_data(self, demo_data, executive_html):
        """Export data in all formats."""
        # Prepare benchmark comparison without equity curves for JSON export
        benchmark_comparison = {}
        benchmark_comparison['strategy'] = demo_data['strategy_metrics']
        
        for name, results in demo_data['benchmark_results'].items():
            # Exclude equity_curve from benchmark results
            benchmark_comparison[name] = {
                k: v for k, v in results.items() 
                if k != 'equity_curve'
            }
        
        export_results = self.export_manager.export_all(
            trades=demo_data['trades'],
            metrics=demo_data['strategy_metrics'],
            confluence_history=demo_data['confluence_history'],
            benchmark_comparison=benchmark_comparison,
            html_report=executive_html
        )
        
        return export_results
    
    def _generate_summary_report(self, export_results):
        """Generate final summary report."""
        summary = f"""
================================================================================
🎯 PHASE 3 VISUALIZATION DEMO - SUMMARY
================================================================================

📊 Visualizations Created:
  ✅ Multi-Timeframe Master Chart
  ✅ Confluence Heatmaps and Analysis
  ✅ Interactive Trade Explorer
  ✅ Timeframe Participation Charts
  ✅ Executive Summary Dashboard
  ✅ Performance Analysis Reports
  ✅ Benchmark Comparison System

💾 Data Exported:
"""
        
        for format_type, filepath in export_results.items():
            summary += f"  ✅ {format_type}: {filepath}\n"
        
        summary += """
📈 Key Results Demonstrated:
  • Total Return: 35.0%
  • Sharpe Ratio: 1.8
  • Win Rate: 62%
  • Max Drawdown: -8%
  • Alpha vs Buy & Hold: +7%

🎯 All Phase 3 components successfully demonstrated!
================================================================================
"""
        
        print(summary)
        
        # Save summary
        summary_path = Path('reports/phase3_demo/demo_summary.txt')
        summary_path.parent.mkdir(exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write(summary)

def main():
    """Run the Phase 3 visualization demo."""
    demo = Phase3VisualizationDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()