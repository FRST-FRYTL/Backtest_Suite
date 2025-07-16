#!/usr/bin/env python3
"""
Demo of Enhanced Trade Reporting with SuperTrend AI data
Shows entry/exit prices, stop losses, and comprehensive trade analysis
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from src.reporting.standard_report_generator import StandardReportGenerator
from src.reporting.report_config import ReportConfig, TradeReportingConfig, ReportTheme

def create_sample_supertrend_trades():
    """Create sample SuperTrend AI trades with detailed price information"""
    
    # Sample SPY price data for context
    base_price = 420.0
    trades = []
    
    # Sample trades with realistic SuperTrend AI behavior
    trade_data = [
        {
            "trade_id": "ST_001",
            "entry_time": "2024-01-15 09:30:00",
            "exit_time": "2024-01-22 15:45:00",
            "entry_price": 418.50,
            "exit_price": 432.25,
            "stop_loss": 405.20,  # ATR-based stop (2.0x ATR)
            "take_profit": 445.15,  # 2:1 risk/reward
            "side": "long",
            "size": 100,
            "pnl": 1375.0,
            "commission": 2.0,
            "slippage": 0.05,
            "exit_reason": "take_profit",
            "supertrend_signal": 8.5,  # Signal strength 1-10
            "market_regime": "trending_up",
            "atr_at_entry": 6.65,
            "volatility_regime": "medium"
        },
        {
            "trade_id": "ST_002", 
            "entry_time": "2024-01-25 10:15:00",
            "exit_time": "2024-01-26 14:30:00",
            "entry_price": 435.80,
            "exit_price": 429.15,
            "stop_loss": 422.30,
            "take_profit": 462.80,
            "side": "long",
            "size": 100,
            "pnl": -665.0,
            "commission": 2.0,
            "slippage": 0.03,
            "exit_reason": "stop_loss",
            "supertrend_signal": 6.2,
            "market_regime": "choppy",
            "atr_at_entry": 8.25,
            "volatility_regime": "high"
        },
        {
            "trade_id": "ST_003",
            "entry_time": "2024-02-02 09:45:00",
            "exit_time": "2024-02-14 13:20:00",
            "entry_price": 441.20,
            "exit_price": 456.85,
            "stop_loss": 426.75,
            "take_profit": 470.10,
            "side": "long",
            "size": 100,
            "pnl": 1565.0,
            "commission": 2.0,
            "slippage": 0.04,
            "exit_reason": "trailing_stop",
            "supertrend_signal": 9.1,
            "market_regime": "trending_up",
            "atr_at_entry": 7.22,
            "volatility_regime": "low"
        },
        {
            "trade_id": "ST_004",
            "entry_time": "2024-02-20 11:00:00",
            "exit_time": "2024-02-21 10:30:00",
            "entry_price": 445.60,
            "exit_price": 442.90,
            "stop_loss": 431.40,
            "take_profit": 473.80,
            "side": "long",
            "size": 100,
            "pnl": -270.0,
            "commission": 2.0,
            "slippage": 0.02,
            "exit_reason": "signal_reversal",
            "supertrend_signal": 7.8,
            "market_regime": "sideways",
            "atr_at_entry": 7.10,
            "volatility_regime": "medium"
        },
        {
            "trade_id": "ST_005",
            "entry_time": "2024-03-01 09:30:00",
            "exit_time": "2024-03-15 16:00:00",
            "entry_price": 448.30,
            "exit_price": 467.45,
            "stop_loss": 433.85,
            "take_profit": 477.20,
            "side": "long",
            "size": 100,
            "pnl": 1915.0,
            "commission": 2.0,
            "slippage": 0.06,
            "exit_reason": "take_profit",
            "supertrend_signal": 8.9,
            "market_regime": "trending_up",
            "atr_at_entry": 7.22,
            "volatility_regime": "medium"
        }
    ]
    
    # Calculate additional metrics for each trade
    for trade in trade_data:
        # Calculate trade duration
        entry_dt = datetime.strptime(trade["entry_time"], "%Y-%m-%d %H:%M:%S")
        exit_dt = datetime.strptime(trade["exit_time"], "%Y-%m-%d %H:%M:%S")
        trade["duration"] = (exit_dt - entry_dt).total_seconds() / 3600  # hours
        
        # Calculate risk metrics
        if trade["side"] == "long":
            trade["risk_amount"] = (trade["entry_price"] - trade["stop_loss"]) * trade["size"]
            trade["potential_reward"] = (trade["take_profit"] - trade["entry_price"]) * trade["size"]
        else:
            trade["risk_amount"] = (trade["stop_loss"] - trade["entry_price"]) * trade["size"]
            trade["potential_reward"] = (trade["entry_price"] - trade["take_profit"]) * trade["size"]
        
        trade["risk_reward_ratio"] = trade["potential_reward"] / trade["risk_amount"] if trade["risk_amount"] > 0 else 0
        trade["actual_risk_reward"] = trade["pnl"] / trade["risk_amount"] if trade["risk_amount"] > 0 else 0
        
        # Calculate percentage returns
        trade["pnl_percent"] = (trade["pnl"] / (trade["entry_price"] * trade["size"])) * 100
        trade["risk_percent"] = (trade["risk_amount"] / (trade["entry_price"] * trade["size"])) * 100
        
        # Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE)
        # Simulated based on typical SuperTrend behavior
        if trade["exit_reason"] == "stop_loss":
            trade["mae"] = trade["risk_amount"]
            trade["mfe"] = abs(trade["risk_amount"]) * 0.3  # Typically went slightly positive first
        else:
            trade["mae"] = abs(trade["risk_amount"]) * 0.4  # Went against initially
            trade["mfe"] = abs(trade["pnl"]) * 1.2  # Went more positive than final exit
    
    return trade_data

def create_enhanced_report():
    """Create enhanced trade report with detailed price analysis"""
    
    # Generate sample trade data
    trades = create_sample_supertrend_trades()
    
    # Create comprehensive backtest results
    backtest_results = {
        "strategy_name": "SuperTrend AI Enhanced",
        "symbol": "SPY",
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "initial_capital": 100000,
        "final_capital": 103820,
        "timeframe": "1H",
        "trades": trades,
        "metrics": {
            "total_return": 0.0382,
            "annualized_return": 0.156,
            "sharpe_ratio": 1.85,
            "sortino_ratio": 2.44,
            "calmar_ratio": 1.12,
            "max_drawdown": -0.0845,
            "win_rate": 0.60,
            "profit_factor": 1.89,
            "total_trades": len(trades),
            "winning_trades": sum(1 for t in trades if t["pnl"] > 0),
            "losing_trades": sum(1 for t in trades if t["pnl"] <= 0),
            "average_win": sum(t["pnl"] for t in trades if t["pnl"] > 0) / sum(1 for t in trades if t["pnl"] > 0),
            "average_loss": sum(t["pnl"] for t in trades if t["pnl"] <= 0) / sum(1 for t in trades if t["pnl"] <= 0),
            "best_trade": max(t["pnl"] for t in trades),
            "worst_trade": min(t["pnl"] for t in trades),
            "average_trade_duration": sum(t["duration"] for t in trades) / len(trades)
        },
        "strategy_parameters": {
            "atr_period": 14,
            "multiplier_range": "1.0-4.0",
            "signal_threshold": 6.0,
            "stop_loss_atr": 2.0,
            "take_profit_rr": 2.0,
            "use_ml_clustering": True,
            "volatility_filter": True
        }
    }
    
    # Configure enhanced reporting
    trade_config = TradeReportingConfig(
        enable_detailed_trade_prices=True,
        price_display_format="absolute",
        show_entry_exit_prices=True,
        show_stop_loss_prices=True,
        show_take_profit_prices=True,
        enable_stop_loss_analysis=True,
        enable_risk_per_trade_analysis=True,
        max_trades_in_detailed_table=50,
        include_trade_timing_analysis=True,
        show_trade_price_charts=True,
        include_mae_mfe_analysis=True
    )
    
    config = ReportConfig(
        title="SuperTrend AI Enhanced Trade Report",
        author="Backtest Suite",
        company="Trading Analytics Lab",
        include_sections=[
            "executive_summary",
            "performance_analysis",
            "risk_analysis", 
            "trade_analysis",
            "technical_details"
        ],
        theme=ReportTheme.PROFESSIONAL,
        output_formats=["html", "markdown", "json"],
        trade_reporting=trade_config,
        performance_thresholds={
            "min_sharpe_ratio": 1.0,
            "min_annual_return": 0.10,
            "max_drawdown": -0.20,
            "min_win_rate": 0.50,
            "min_profit_factor": 1.5
        }
    )
    
    # Generate enhanced report
    generator = StandardReportGenerator(config)
    output_dir = "reports/enhanced_trade_demo"
    
    print("🔄 Generating Enhanced Trade Report...")
    print(f"📊 Processing {len(trades)} trades with detailed price analysis...")
    
    report_paths = generator.generate_report(backtest_results, output_dir)
    
    print("\n✅ Enhanced Trade Report Generated Successfully!")
    print("\n📁 Output Files:")
    for format_type, path in report_paths.items():
        print(f"  {format_type.upper()}: {path}")
    
    # Display key trade insights
    print("\n📈 Trade Analysis Summary:")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Win Rate: {backtest_results['metrics']['win_rate']:.1%}")
    print(f"  Profit Factor: {backtest_results['metrics']['profit_factor']:.2f}")
    print(f"  Average Risk per Trade: ${sum(t['risk_amount'] for t in trades) / len(trades):.2f}")
    print(f"  Average Risk/Reward: {sum(t['risk_reward_ratio'] for t in trades) / len(trades):.2f}")
    
    # Stop loss analysis
    stop_losses = [t for t in trades if t["exit_reason"] == "stop_loss"]
    print(f"  Trades Stopped Out: {len(stop_losses)} ({len(stop_losses)/len(trades):.1%})")
    
    # Top performing trades
    top_trades = sorted(trades, key=lambda x: x["pnl"], reverse=True)[:2]
    print("\n🏆 Top 2 Trades:")
    for i, trade in enumerate(top_trades, 1):
        print(f"  {i}. {trade['trade_id']}: ${trade['pnl']:.2f} "
              f"({trade['entry_price']:.2f} → {trade['exit_price']:.2f})")
    
    print(f"\n🎯 Full interactive report available at: {report_paths.get('html', 'N/A')}")
    
    return report_paths

if __name__ == "__main__":
    report_paths = create_enhanced_report()
    print("\n🚀 Enhanced Trade Reporting Demo Complete!")
    print("The report now includes detailed entry/exit prices, stop losses, and comprehensive trade analysis.")