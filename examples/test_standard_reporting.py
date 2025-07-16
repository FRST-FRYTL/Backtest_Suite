#!/usr/bin/env python3
"""
Test the new standardized reporting system with SuperTrend AI results
"""

import json
import os
from datetime import datetime
from src.reporting.standard_report_generator import StandardReportGenerator
from src.reporting.report_config import ReportConfig, ReportTheme

def test_standard_reporting():
    """Test the standardized reporting system"""
    
    # Create sample backtest results (simplified version of SuperTrend AI results)
    backtest_results = {
        "strategy_name": "SuperTrend AI Strategy",
        "start_date": "2023-07-17",
        "end_date": "2025-07-14",
        "initial_capital": 100000,
        "final_capital": 140192,
        "metrics": {
            "total_return": 0.40192,
            "annualized_return": 0.185,
            "sharpe_ratio": 1.976,
            "sortino_ratio": 2.856,
            "calmar_ratio": 1.360,
            "max_drawdown": -0.136,
            "win_rate": 0.62,
            "profit_factor": 2.15,
            "total_trades": 48,
            "winning_trades": 30,
            "losing_trades": 18,
            "average_win": 0.0312,
            "average_loss": -0.0145,
            "best_trade": 0.0687,
            "worst_trade": -0.0394,
            "average_trade_duration": "8.5 days",
            "volatility": 0.0937
        },
        "timeframe_analysis": {
            "Monthly": {"return": 0.185, "sharpe": 1.976, "trades": 12},
            "Weekly": {"return": 0.174, "sharpe": 1.735, "trades": 48},
            "Daily": {"return": 0.158, "sharpe": 1.450, "trades": 180}
        },
        "parameter_sensitivity": {
            "atr_period": {
                "10": {"return": 0.145, "sharpe": 1.523},
                "14": {"return": 0.185, "sharpe": 1.976},
                "20": {"return": 0.167, "sharpe": 1.712}
            },
            "factor_range": {
                "1.0-3.0": {"return": 0.162, "sharpe": 1.654},
                "1.0-4.0": {"return": 0.185, "sharpe": 1.976},
                "1.5-4.5": {"return": 0.171, "sharpe": 1.823}
            }
        },
        "equity_curve": [100000, 101200, 102500, 101800, 103400, 105200, 104300, 106800, 108400, 110200, 109500, 112000, 114500, 113200, 116000, 118500, 117800, 120500, 123000, 122100, 125000, 127800, 126900, 130000, 132500, 131400, 134500, 137000, 136200, 139500, 140192],
        "trades": [
            {"entry_date": "2023-08-01", "exit_date": "2023-08-15", "pnl": 0.0215, "type": "long"},
            {"entry_date": "2023-09-05", "exit_date": "2023-09-12", "pnl": -0.0087, "type": "long"},
            {"entry_date": "2023-10-20", "exit_date": "2023-11-02", "pnl": 0.0342, "type": "long"}
        ]
    }
    
    # Configure the report
    config = ReportConfig(
        title="SuperTrend AI Strategy - Standard Report Demo",
        author="Backtest Suite",
        company="Trading Research Lab",
        include_sections=[
            "executive_summary",
            "performance_analysis", 
            "risk_analysis",
            "trade_analysis",
            "technical_details"
        ],
        theme=ReportTheme.PROFESSIONAL,
        output_formats=["html", "markdown", "json"],
        performance_thresholds={
            "min_sharpe_ratio": 1.0,
            "min_annual_return": 0.10,
            "max_drawdown": -0.20
        }
    )
    
    # Generate the report
    generator = StandardReportGenerator(config)
    output_dir = "reports/standard_report_demo"
    
    print("Generating standardized report...")
    report_paths = generator.generate_report(backtest_results, output_dir)
    
    print("\nReport generated successfully!")
    print("\nOutput files:")
    for format_type, path in report_paths.items():
        print(f"  {format_type}: {path}")
    
    # Display report summary
    print("\nReport Summary:")
    print(f"  Strategy: {backtest_results['strategy_name']}")
    print(f"  Annual Return: {backtest_results['metrics']['annualized_return']:.1%}")
    print(f"  Sharpe Ratio: {backtest_results['metrics']['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {backtest_results['metrics']['max_drawdown']:.1%}")
    print(f"  Total Trades: {backtest_results['metrics']['total_trades']}")
    
    print("\nThe standardized reporting system is now integrated and ready for use!")
    print("All future reports will follow this professional format.")

if __name__ == "__main__":
    test_standard_reporting()