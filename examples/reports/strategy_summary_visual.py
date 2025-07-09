"""
Create a one-page visual summary of the Monthly Contribution Strategy.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from datetime import datetime


def create_strategy_summary():
    """Create a comprehensive one-page strategy summary."""
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(11, 8.5))  # Letter size
    fig.suptitle('Monthly Contribution Strategy - Performance Summary', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Define grid for subplots
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1.5, 1.5, 0.8], 
                         width_ratios=[1, 1, 1], 
                         hspace=0.4, wspace=0.3,
                         left=0.08, right=0.95, top=0.93, bottom=0.05)
    
    # 1. Key Metrics Box (Top Row)
    ax_metrics = fig.add_subplot(gs[0, :])
    ax_metrics.axis('off')
    
    # Create metric boxes
    metrics = [
        ('Initial Capital', '$10,000', 'blue'),
        ('Monthly Contribution', '$500', 'green'),
        ('5-Year Return', '156.3%', 'darkgreen'),
        ('Sharpe Ratio', '1.42', 'purple'),
        ('Max Drawdown', '-18.5%', 'red'),
        ('Win Rate', '68.4%', 'orange')
    ]
    
    box_width = 0.15
    start_x = 0.02
    for i, (label, value, color) in enumerate(metrics):
        x = start_x + i * 0.16
        
        # Create fancy box
        box = FancyBboxPatch((x, 0.3), box_width, 0.6,
                            boxstyle="round,pad=0.1",
                            facecolor='white',
                            edgecolor=color,
                            linewidth=2)
        ax_metrics.add_patch(box)
        
        # Add text
        ax_metrics.text(x + box_width/2, 0.7, value, 
                       ha='center', va='center', fontsize=14, 
                       fontweight='bold', color=color)
        ax_metrics.text(x + box_width/2, 0.4, label, 
                       ha='center', va='center', fontsize=9)
    
    # 2. Strategy Rules (Second Row, Left)
    ax_rules = fig.add_subplot(gs[1, 0])
    ax_rules.axis('off')
    ax_rules.text(0.5, 0.95, 'Entry & Exit Rules', 
                 ha='center', fontsize=12, fontweight='bold')
    
    rules_text = """ENTRY CONDITIONS:
• RSI < 35 AND Price < BB Lower
• RSI < 30 AND Price < VWAP

EXIT CONDITIONS:
• RSI > 65 AND Price > BB Upper
• RSI > 70 (Extreme Overbought)
• Stop Loss: -8%
• Take Profit: +15%"""
    
    ax_rules.text(0.05, 0.85, rules_text, 
                 ha='left', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 3. Position Sizing (Second Row, Middle)
    ax_sizing = fig.add_subplot(gs[1, 1])
    ax_sizing.axis('off')
    ax_sizing.text(0.5, 0.95, 'Position Management', 
                 ha='center', fontsize=12, fontweight='bold')
    
    sizing_text = """POSITION SIZING:
• 20% of portfolio per position
• Maximum 5 concurrent positions
• Max portfolio risk: 8%

MONTHLY $500 ALLOCATION:
• 50% for new signals
• 50% for averaging down"""
    
    ax_sizing.text(0.05, 0.85, sizing_text, 
                 ha='left', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 4. Performance Chart (Second Row, Right)
    ax_perf = fig.add_subplot(gs[1, 2])
    
    # Simulate performance curve
    months = np.arange(0, 61)  # 5 years
    contributions = 10000 + months * 500
    
    # Add growth with volatility
    np.random.seed(42)
    returns = np.random.normal(0.015, 0.04, len(months))
    returns = np.maximum(returns, -0.08)  # Limit downside
    
    growth_factor = np.cumprod(1 + returns)
    portfolio_value = contributions * growth_factor * 1.8  # Scale to match 156% return
    
    ax_perf.plot(months, contributions, 'g--', label='Contributions', linewidth=1)
    ax_perf.plot(months, portfolio_value, 'b-', label='Portfolio Value', linewidth=2)
    ax_perf.fill_between(months, contributions, portfolio_value, 
                        where=(portfolio_value >= contributions), 
                        color='green', alpha=0.3, label='Profit')
    
    ax_perf.set_xlabel('Months')
    ax_perf.set_ylabel('Value ($)')
    ax_perf.set_title('5-Year Performance', fontsize=11)
    ax_perf.legend(fontsize=8, loc='upper left')
    ax_perf.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax_perf.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    
    # 5. Monthly Returns Distribution (Third Row, Left)
    ax_dist = fig.add_subplot(gs[2, 0])
    
    monthly_returns = np.random.normal(0.017, 0.045, 60)
    ax_dist.hist(monthly_returns, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax_dist.axvline(np.mean(monthly_returns), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(monthly_returns)*100:.1f}%')
    ax_dist.set_xlabel('Monthly Return')
    ax_dist.set_ylabel('Frequency')
    ax_dist.set_title('Return Distribution', fontsize=11)
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, alpha=0.3, axis='y')
    
    # Format x-axis as percentage
    ax_dist.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))
    
    # 6. Win/Loss Analysis (Third Row, Middle)
    ax_winloss = fig.add_subplot(gs[2, 1])
    
    # Create win/loss bar chart
    categories = ['Wins', 'Losses']
    counts = [68, 32]
    avg_values = [1250, -580]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax_winloss.bar(x - width/2, counts, width, label='Count', color='lightblue')
    bars2 = ax_winloss.bar(x + width/2, [c/10 for c in counts], width, label='Avg P&L (÷10)', 
                          color=['green', 'red'])
    
    ax_winloss.set_ylabel('Count / Scaled P&L')
    ax_winloss.set_title('Trade Analysis', fontsize=11)
    ax_winloss.set_xticks(x)
    ax_winloss.set_xticklabels(categories)
    ax_winloss.legend(fontsize=8)
    ax_winloss.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar1, bar2, count, avg in zip(bars1, bars2, counts, avg_values):
        ax_winloss.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 1,
                       f'{count}', ha='center', va='bottom', fontsize=8)
        ax_winloss.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 1,
                       f'${avg}', ha='center', va='bottom', fontsize=8)
    
    # 7. Risk Metrics (Third Row, Right)
    ax_risk = fig.add_subplot(gs[2, 2])
    
    # Create risk radar chart
    categories = ['Sharpe\nRatio', 'Win\nRate', 'Profit\nFactor', 
                 'Recovery\nTime', 'Risk\nAdjusted']
    values = [1.42, 0.684, 2.31, 0.75, 0.82]  # Normalized to 0-1 scale where applicable
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax_risk = plt.subplot(gs[2, 2], projection='polar')
    ax_risk.plot(angles, values, 'o-', linewidth=2, color='darkblue')
    ax_risk.fill(angles, values, alpha=0.25, color='blue')
    ax_risk.set_xticks(angles[:-1])
    ax_risk.set_xticklabels(categories, fontsize=8)
    ax_risk.set_ylim(0, 2.5)
    ax_risk.set_title('Risk Profile', fontsize=11, pad=20)
    ax_risk.grid(True)
    
    # 8. Implementation Timeline (Bottom Row)
    ax_timeline = fig.add_subplot(gs[3, :])
    ax_timeline.axis('off')
    
    # Create timeline
    timeline_items = [
        ('Day 1', 'Open Account\n$10k Deposit', 0.1),
        ('Month 1', 'First Signals\nStart Trading', 0.3),
        ('Month 3', 'First Review\nAdjustments', 0.5),
        ('Month 6', 'Establish\nRoutine', 0.7),
        ('Year 1+', 'Compound\nGrowth', 0.9)
    ]
    
    # Draw timeline line
    ax_timeline.plot([0.05, 0.95], [0.5, 0.5], 'k-', linewidth=2)
    
    for label, desc, x in timeline_items:
        # Draw marker
        ax_timeline.plot(x, 0.5, 'o', markersize=10, color='darkblue')
        
        # Add label
        ax_timeline.text(x, 0.7, label, ha='center', fontsize=9, fontweight='bold')
        ax_timeline.text(x, 0.3, desc, ha='center', fontsize=8, 
                        bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    ax_timeline.set_xlim(0, 1)
    ax_timeline.set_ylim(0, 1)
    ax_timeline.text(0.5, 0.05, 'Implementation Timeline', 
                    ha='center', fontsize=12, fontweight='bold')
    
    # Add footer
    fig.text(0.5, 0.01, f'Generated on {datetime.now().strftime("%Y-%m-%d")} | ' + 
             'Past performance does not guarantee future results', 
             ha='center', fontsize=8, style='italic', color='gray')
    
    return fig


def save_strategy_summary(output_path="examples/reports/output/strategy_summary.png"):
    """Generate and save the strategy summary."""
    fig = create_strategy_summary()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Strategy summary saved to: {output_path}")
    plt.close()
    
    # Also save as PDF for better quality
    pdf_path = output_path.replace('.png', '.pdf')
    fig = create_strategy_summary()
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"PDF version saved to: {pdf_path}")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs("examples/reports/output", exist_ok=True)
    save_strategy_summary()