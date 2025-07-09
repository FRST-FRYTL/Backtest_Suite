"""
Quick script to generate the monthly contribution strategy report.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from monthly_contribution_strategy_report import MonthlyContributionReport


async def generate_quick_report():
    """Generate a quick 2-year report for demonstration."""
    print("Generating Monthly Contribution Strategy Report...")
    print("-" * 60)
    
    # Create report generator with $10k initial and $500/month
    report = MonthlyContributionReport(
        initial_capital=10000,
        monthly_contribution=500
    )
    
    # Run 2-year backtest for faster demo
    print("\nRunning 2-year backtest on SPY...")
    try:
        results, metrics = await report.run_backtest(symbol="SPY", years=2)
        
        # Generate and save report
        print("\nGenerating visualizations and analysis...")
        output_dir = report.save_report()
        
        print("\n" + "="*60)
        print("REPORT GENERATION COMPLETE!")
        print("="*60)
        print(f"\nReport files saved to: {os.path.abspath(output_dir)}")
        print("\nOpen the following files in your browser:")
        print("1. main_dashboard.html - Interactive dashboard")
        print("2. performance_analysis.html - Detailed performance charts")
        print("3. strategy_analysis.html - Strategy signal analysis")
        print("4. executive_summary.md - Text summary")
        
    except Exception as e:
        print(f"\nError generating report: {str(e)}")
        print("Make sure you have all dependencies installed:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    asyncio.run(generate_quick_report())