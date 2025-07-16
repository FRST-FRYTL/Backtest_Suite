"""
Enhanced Trade Reporting Example

This example demonstrates how to use the enhanced trade reporting features
including detailed price analysis, stop loss analysis, and risk per trade metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.reporting.standard_report_generator import StandardReportGenerator
from src.reporting.report_config import ReportConfig, TradeReportingConfig
from src.reporting.visualizations import ReportVisualizations
from src.reporting.enhanced_json_export import create_enhanced_json_export
from src.reporting.markdown_template import generate_markdown_report, create_enhanced_trade_summary


def create_sample_data():
    """Create comprehensive sample data for enhanced trade reporting"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Create sample price data
    initial_price = 150.0
    returns = np.random.normal(0.0002, 0.015, len(dates))
    prices = pd.Series(initial_price * np.exp(returns.cumsum()), index=dates)
    
    price_data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.random.uniform(0.001, 0.02, len(dates))),
        'low': prices * (1 - np.random.uniform(0.001, 0.02, len(dates))),
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    # Create comprehensive trades data
    n_trades = 100
    trade_dates = np.random.choice(dates[:-5], n_trades, replace=False)
    
    trades_data = []
    for i, trade_date in enumerate(trade_dates):
        side = np.random.choice(['long', 'short'])
        entry_price = prices.loc[trade_date] * np.random.uniform(0.999, 1.001)
        
        # Duration in hours
        duration = np.random.uniform(2, 48)
        exit_date = trade_date + timedelta(hours=duration)
        
        # Price movement
        price_change = np.random.normal(0.001, 0.02) * (duration / 24)
        exit_price = entry_price * (1 + price_change)
        
        # Stop loss and take profit levels
        if side == 'long':
            stop_loss = entry_price * np.random.uniform(0.96, 0.98)
            take_profit = entry_price * np.random.uniform(1.02, 1.06)
        else:
            stop_loss = entry_price * np.random.uniform(1.02, 1.04)
            take_profit = entry_price * np.random.uniform(0.94, 0.98)
        
        # Trade size
        size = np.random.randint(50, 500)
        
        # Calculate P&L
        if side == 'long':
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size
        
        # Determine exit reason
        exit_reasons = ['target', 'stop', 'time']
        weights = [0.4, 0.3, 0.3]  # 40% targets, 30% stops, 30% time
        exit_reason = np.random.choice(exit_reasons, p=weights)
        
        # Adjust exit price based on exit reason
        if exit_reason == 'target':
            exit_price = take_profit * np.random.uniform(0.995, 1.005)
            pnl = (take_profit - entry_price) * size if side == 'long' else (entry_price - take_profit) * size
        elif exit_reason == 'stop':
            exit_price = stop_loss * np.random.uniform(0.995, 1.005)
            pnl = (stop_loss - entry_price) * size if side == 'long' else (entry_price - stop_loss) * size
        
        # Commission and slippage
        commission = np.random.uniform(3, 8)
        slippage = np.random.uniform(0.01, 0.1)
        
        # Adjust P&L for costs
        pnl -= commission + slippage
        
        trade_data = {
            'trade_id': i + 1,
            'entry_time': trade_date,
            'exit_time': exit_date,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'size': size,
            'pnl': pnl,
            'duration': duration,
            'exit_reason': exit_reason,
            'commission': commission,
            'slippage': slippage
        }
        
        trades_data.append(trade_data)
    
    trades_df = pd.DataFrame(trades_data)
    
    # Create equity curve
    trades_df_sorted = trades_df.sort_values('entry_time')
    cumulative_pnl = trades_df_sorted['pnl'].cumsum()
    initial_capital = 100000
    equity_curve = pd.Series(initial_capital + cumulative_pnl.values, 
                           index=trades_df_sorted['entry_time'])
    
    # Resample to daily
    daily_equity = equity_curve.resample('D').last().fillna(method='ffill')
    daily_returns = daily_equity.pct_change().dropna()
    
    # Calculate comprehensive metrics
    total_return = (daily_equity.iloc[-1] / daily_equity.iloc[0] - 1)
    annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility != 0 else 0
    
    # Drawdown calculation
    rolling_max = daily_equity.expanding().max()
    drawdown = (daily_equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Trade metrics
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]
    
    win_rate = len(winning_trades) / len(trades_df)
    
    gross_profit = winning_trades['pnl'].sum()
    gross_loss = abs(losing_trades['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # Enhanced trade metrics
    avg_entry_price = trades_df['entry_price'].mean()
    avg_exit_price = trades_df['exit_price'].mean()
    
    # Stop loss analysis
    stop_hits = trades_df[trades_df['exit_reason'] == 'stop']
    stop_loss_hit_rate = len(stop_hits) / len(trades_df)
    
    # Risk analysis
    trades_with_stops = trades_df.dropna(subset=['stop_loss'])
    if not trades_with_stops.empty:
        risk_per_trade = abs(trades_with_stops['stop_loss'] - trades_with_stops['entry_price']) / trades_with_stops['entry_price']
        avg_risk_per_trade = risk_per_trade.mean()
    else:
        avg_risk_per_trade = 0
    
    # Risk/reward ratio
    if len(winning_trades) > 0 and len(losing_trades) > 0:
        avg_win = winning_trades['pnl'].mean()
        avg_loss = abs(losing_trades['pnl'].mean())
        risk_reward_ratio = avg_win / avg_loss if avg_loss != 0 else float('inf')
    else:
        risk_reward_ratio = 0
    
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': annual_return / (daily_returns[daily_returns < 0].std() * np.sqrt(252)),
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': len(trades_df),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
        'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
        'avg_trade_pnl': trades_df['pnl'].mean(),
        'avg_entry_price': avg_entry_price,
        'avg_exit_price': avg_exit_price,
        'stop_loss_hit_rate': stop_loss_hit_rate,
        'avg_risk_per_trade': avg_risk_per_trade,
        'risk_reward_ratio': risk_reward_ratio,
        'avg_trade_duration': trades_df['duration'].mean(),
        'total_commission': trades_df['commission'].sum(),
        'total_slippage': trades_df['slippage'].sum()
    }
    
    # Strategy parameters
    strategy_params = {
        'name': 'Enhanced Trade Analysis Demo Strategy',
        'version': '1.0.0',
        'description': 'Demonstration strategy for enhanced trade reporting',
        'parameters': {
            'lookback_period': {'value': 20, 'type': 'int', 'description': 'Lookback period for signals'},
            'stop_loss_pct': {'value': 0.02, 'type': 'float', 'description': 'Stop loss percentage'},
            'take_profit_pct': {'value': 0.04, 'type': 'float', 'description': 'Take profit percentage'},
            'position_size': {'value': 200, 'type': 'int', 'description': 'Average position size'},
            'risk_per_trade': {'value': 0.01, 'type': 'float', 'description': 'Risk per trade as % of capital'}
        }
    }
    
    return {
        'equity_curve': daily_equity,
        'trades': trades_df,
        'metrics': metrics,
        'returns': daily_returns,
        'strategy_params': strategy_params,
        'price_data': price_data
    }


def demonstrate_enhanced_visualizations(trades_df, price_data, output_dir):
    """Demonstrate enhanced visualization capabilities"""
    
    print("Creating enhanced visualizations...")
    
    # Initialize visualizations
    viz = ReportVisualizations()
    
    # Create trade price chart
    print("  - Creating trade price chart...")
    trade_price_chart = viz.create_trade_price_chart(trades_df, price_data)
    trade_price_chart.write_html(os.path.join(output_dir, "trade_price_chart.html"))
    
    # Create stop loss analysis chart
    print("  - Creating stop loss analysis chart...")
    stop_loss_chart = viz.create_stop_loss_analysis(trades_df)
    stop_loss_chart.write_html(os.path.join(output_dir, "stop_loss_analysis.html"))
    
    # Create trade risk chart
    print("  - Creating trade risk chart...")
    trade_risk_chart = viz.create_trade_risk_chart(trades_df)
    trade_risk_chart.write_html(os.path.join(output_dir, "trade_risk_analysis.html"))
    
    # Create additional standard charts
    print("  - Creating standard charts...")
    
    # Cumulative returns chart
    returns = trades_df.set_index('entry_time')['pnl'].cumsum()
    cumulative_chart = viz.cumulative_returns(returns.pct_change().dropna())
    cumulative_chart.write_html(os.path.join(output_dir, "cumulative_returns.html"))
    
    # Trade distribution chart
    trade_dist_chart = viz.trade_distribution(trades_df)
    trade_dist_chart.write_html(os.path.join(output_dir, "trade_distribution.html"))
    
    print(f"  - All visualizations saved to {output_dir}")
    
    return {
        'trade_price_chart': trade_price_chart,
        'stop_loss_chart': stop_loss_chart,
        'trade_risk_chart': trade_risk_chart,
        'cumulative_chart': cumulative_chart,
        'trade_dist_chart': trade_dist_chart
    }


def demonstrate_enhanced_json_export(backtest_results, output_dir):
    """Demonstrate enhanced JSON export functionality"""
    
    print("Creating enhanced JSON export...")
    
    # Configure export
    export_config = {
        'include_detailed_trades': True,
        'include_price_analysis': True,
        'include_stop_loss_analysis': True,
        'include_risk_analysis': True,
        'max_trades_export': 500,
        'decimal_places': 4
    }
    
    # Create mock report data structure
    report_data = {
        'metadata': {
            'report_name': 'Enhanced Trade Analysis Demo',
            'generated_at': datetime.now().isoformat(),
            'generator_version': '1.0.0'
        },
        'sections': {
            'tradeanalysis': {
                'trade_statistics': {
                    'summary': {
                        'total_trades': len(backtest_results['trades']),
                        'winning_trades': len(backtest_results['trades'][backtest_results['trades']['pnl'] > 0]),
                        'losing_trades': len(backtest_results['trades'][backtest_results['trades']['pnl'] <= 0]),
                        'win_rate': (backtest_results['trades']['pnl'] > 0).mean(),
                        'profit_factor': backtest_results['metrics']['profit_factor']
                    }
                },
                'detailed_trades': backtest_results['trades'].to_dict('records')
            }
        },
        'raw_backtest_results': backtest_results
    }
    
    # Create enhanced export
    json_path = create_enhanced_json_export(
        report_data=report_data,
        output_path=os.path.join(output_dir, "enhanced_export.json"),
        config=export_config
    )
    
    print(f"  - Enhanced JSON export saved to {json_path}")
    
    return json_path


def demonstrate_markdown_generation(backtest_results, output_dir):
    """Demonstrate enhanced markdown report generation"""
    
    print("Creating enhanced markdown report...")
    
    # Create enhanced trade summary
    trade_summary = create_enhanced_trade_summary(backtest_results['trades'])
    
    # Create mock section data for markdown generation
    section_data = {
        'trade_statistics': {
            'summary': trade_summary
        },
        'win_loss_analysis': {
            'win_rate': f"{backtest_results['metrics']['win_rate']*100:.1f}%",
            'winning_trades': {
                'count': backtest_results['metrics']['winning_trades'],
                'avg_win': f"${backtest_results['metrics']['avg_win']:.2f}" if backtest_results['metrics']['avg_win'] != 0 else 'N/A'
            },
            'losing_trades': {
                'count': backtest_results['metrics']['losing_trades'],
                'avg_loss': f"${backtest_results['metrics']['avg_loss']:.2f}" if backtest_results['metrics']['avg_loss'] != 0 else 'N/A'
            }
        },
        'price_analysis': {
            'avg_entry_price': f"${backtest_results['metrics']['avg_entry_price']:.2f}",
            'avg_exit_price': f"${backtest_results['metrics']['avg_exit_price']:.2f}",
            'price_improvement': f"{((backtest_results['metrics']['avg_exit_price'] - backtest_results['metrics']['avg_entry_price']) / backtest_results['metrics']['avg_entry_price'] * 100):.2f}%"
        },
        'stop_loss_analysis': {
            'stop_loss_hit_rate': f"{backtest_results['metrics']['stop_loss_hit_rate']*100:.1f}%",
            'avg_risk_per_trade': f"{backtest_results['metrics']['avg_risk_per_trade']*100:.2f}%",
            'stop_effectiveness': 'Good' if backtest_results['metrics']['stop_loss_hit_rate'] < 0.4 else 'Fair'
        },
        'risk_analysis': {
            'avg_risk_per_trade': f"{backtest_results['metrics']['avg_risk_per_trade']*100:.2f}%",
            'risk_reward_ratio': f"1:{backtest_results['metrics']['risk_reward_ratio']:.1f}",
            'max_risk_per_trade': f"{backtest_results['trades']['stop_loss'].max():.2f}%" if not backtest_results['trades']['stop_loss'].empty else 'N/A'
        },
        'detailed_trades': backtest_results['trades'].head(20).to_dict('records')  # First 20 trades
    }
    
    # Generate markdown
    from src.reporting.markdown_template import generate_trade_analysis_md
    markdown_content = generate_trade_analysis_md(section_data)
    
    # Save to file
    markdown_path = os.path.join(output_dir, "enhanced_trade_analysis.md")
    with open(markdown_path, 'w') as f:
        f.write(markdown_content)
    
    print(f"  - Enhanced markdown report saved to {markdown_path}")
    
    return markdown_path


def demonstrate_full_report_generation(backtest_results, output_dir):
    """Demonstrate full enhanced report generation"""
    
    print("Generating comprehensive enhanced trade report...")
    
    # Configure enhanced trade reporting
    trade_config = TradeReportingConfig(
        enable_detailed_trade_prices=True,
        price_display_format="absolute",
        show_entry_exit_prices=True,
        show_stop_loss_prices=True,
        show_take_profit_prices=True,
        enable_stop_loss_analysis=True,
        enable_risk_per_trade_analysis=True,
        max_trades_in_detailed_table=100,
        include_trade_timing_analysis=True,
        show_trade_price_charts=True
    )
    
    # Create main report configuration
    config = ReportConfig()
    config.trade_reporting = trade_config
    config.style.chart_height = 600
    config.style.chart_width = 1000
    
    # Generate report using StandardReportGenerator
    # Note: This would require integration with the actual report generator
    # For demonstration, we'll show the configuration
    
    print("  - Trade reporting configuration:")
    print(f"    * Detailed trade prices: {trade_config.enable_detailed_trade_prices}")
    print(f"    * Stop loss analysis: {trade_config.enable_stop_loss_analysis}")
    print(f"    * Risk per trade analysis: {trade_config.enable_risk_per_trade_analysis}")
    print(f"    * Price display format: {trade_config.price_display_format}")
    print(f"    * Max trades in table: {trade_config.max_trades_in_detailed_table}")
    
    # Show key metrics that would be included
    print("\n  - Key enhanced metrics:")
    print(f"    * Total trades: {backtest_results['metrics']['total_trades']}")
    print(f"    * Win rate: {backtest_results['metrics']['win_rate']:.1%}")
    print(f"    * Profit factor: {backtest_results['metrics']['profit_factor']:.2f}")
    print(f"    * Avg entry price: ${backtest_results['metrics']['avg_entry_price']:.2f}")
    print(f"    * Avg exit price: ${backtest_results['metrics']['avg_exit_price']:.2f}")
    print(f"    * Stop loss hit rate: {backtest_results['metrics']['stop_loss_hit_rate']:.1%}")
    print(f"    * Avg risk per trade: {backtest_results['metrics']['avg_risk_per_trade']:.2%}")
    print(f"    * Risk/reward ratio: 1:{backtest_results['metrics']['risk_reward_ratio']:.1f}")
    
    return config


def main():
    """Main demonstration function"""
    
    print("=" * 60)
    print("ENHANCED TRADE REPORTING DEMONSTRATION")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("output/enhanced_trade_reporting_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create sample data
    print("\n1. Creating comprehensive sample data...")
    backtest_results = create_sample_data()
    
    print(f"   - Created {len(backtest_results['trades'])} trades")
    print(f"   - Date range: {backtest_results['trades']['entry_time'].min()} to {backtest_results['trades']['entry_time'].max()}")
    print(f"   - Price data points: {len(backtest_results['price_data'])}")
    
    # Step 2: Demonstrate enhanced visualizations
    print("\n2. Demonstrating enhanced visualizations...")
    visualizations = demonstrate_enhanced_visualizations(
        backtest_results['trades'], 
        backtest_results['price_data'], 
        output_dir
    )
    
    # Step 3: Demonstrate enhanced JSON export
    print("\n3. Demonstrating enhanced JSON export...")
    json_export_path = demonstrate_enhanced_json_export(backtest_results, output_dir)
    
    # Step 4: Demonstrate markdown generation
    print("\n4. Demonstrating enhanced markdown generation...")
    markdown_path = demonstrate_markdown_generation(backtest_results, output_dir)
    
    # Step 5: Demonstrate full report configuration
    print("\n5. Demonstrating full report generation configuration...")
    config = demonstrate_full_report_generation(backtest_results, output_dir)
    
    # Step 6: Save sample data for reference
    print("\n6. Saving sample data for reference...")
    
    # Save trades data
    trades_path = output_dir / "sample_trades.csv"
    backtest_results['trades'].to_csv(trades_path, index=False)
    
    # Save price data
    price_path = output_dir / "sample_price_data.csv"
    backtest_results['price_data'].to_csv(price_path)
    
    # Save metrics
    metrics_path = output_dir / "sample_metrics.json"
    import json
    with open(metrics_path, 'w') as f:
        json.dump(backtest_results['metrics'], f, indent=2, default=str)
    
    print(f"   - Sample data saved to {output_dir}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - trade_price_chart.html: Interactive trade price visualization")
    print("  - stop_loss_analysis.html: Stop loss effectiveness analysis")
    print("  - trade_risk_analysis.html: Risk per trade analysis")
    print("  - cumulative_returns.html: Cumulative returns chart")
    print("  - trade_distribution.html: Trade P&L distribution")
    print("  - enhanced_export.json: Comprehensive JSON export")
    print("  - enhanced_trade_analysis.md: Markdown report")
    print("  - sample_trades.csv: Sample trade data")
    print("  - sample_price_data.csv: Sample price data")
    print("  - sample_metrics.json: Sample metrics")
    
    print("\nKey features demonstrated:")
    print("  ✓ Detailed trade price tracking")
    print("  ✓ Stop loss analysis and visualization")
    print("  ✓ Risk per trade analysis")
    print("  ✓ Enhanced JSON export with comprehensive data")
    print("  ✓ Markdown report generation with trade details")
    print("  ✓ Interactive visualizations with Plotly")
    print("  ✓ Configuration system for customization")
    
    print(f"\nTo view the interactive charts, open the HTML files in {output_dir}")
    print("To see the enhanced data export, examine the enhanced_export.json file")
    print("To review the markdown report, open enhanced_trade_analysis.md")


if __name__ == "__main__":
    main()