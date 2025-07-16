"""
Markdown Template Generator for Enhanced Trade Reporting

This module generates professional markdown reports with enhanced trade analysis,
including detailed trade prices, stop loss analysis, and risk metrics.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime


def generate_markdown_report(report_data: Dict[str, Any], config: Any) -> str:
    """Generate complete markdown report with enhanced trade analysis"""
    
    # Extract data
    metadata = report_data.get("metadata", {})
    sections = report_data.get("sections", {})
    summary = report_data.get("backtest_summary", {})
    
    # Generate markdown content
    markdown = f"""# {config.title}

**{config.subtitle if hasattr(config, 'subtitle') and config.subtitle else 'Backtest Analysis Report'}**

Generated on: {metadata.get('generated_at', datetime.now().isoformat())}

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Performance Analysis](#performance-analysis)
3. [Risk Analysis](#risk-analysis)
4. [Trade Analysis](#trade-analysis)
5. [Market Regime Analysis](#market-regime-analysis)
6. [Technical Details](#technical-details)

---

{generate_executive_summary_md(sections.get('executivesummary', {}))}

{generate_performance_analysis_md(sections.get('performanceanalysis', {}))}

{generate_risk_analysis_md(sections.get('riskanalysis', {}))}

{generate_trade_analysis_md(sections.get('tradeanalysis', {}))}

{generate_market_regime_analysis_md(sections.get('marketregimeanalysis', {}))}

{generate_technical_details_md(sections.get('technicaldetails', {}))}

---

## Report Generation

This report was generated using the Backtest Suite's standardized reporting system.

- **Generator Version**: {metadata.get('generator_version', 'Unknown')}
- **Generated At**: {metadata.get('generated_at', 'Unknown')}
- **Report Type**: Enhanced Trade Analysis Report
"""
    
    return markdown


def generate_executive_summary_md(section_data: Dict[str, Any]) -> str:
    """Generate executive summary markdown"""
    if not section_data:
        return ""
    
    key_metrics = section_data.get('key_metrics', {})
    assessment = section_data.get('strategy_assessment', 'N/A')
    findings = section_data.get('key_findings', [])
    recommendations = section_data.get('recommendations', [])
    
    # Format key metrics table
    metrics_table = "| Metric | Value |\n|--------|-------|\n"
    for metric, value in key_metrics.items():
        metrics_table += f"| {metric} | {value} |\n"
    
    # Format findings
    findings_md = ""
    if findings:
        findings_md = "### Key Findings\n\n"
        for finding in findings:
            findings_md += f"- {finding}\n"
        findings_md += "\n"
    
    # Format recommendations
    recommendations_md = ""
    if recommendations:
        recommendations_md = "### Recommendations\n\n"
        for rec in recommendations:
            recommendations_md += f"- {rec}\n"
        recommendations_md += "\n"
    
    return f"""## Executive Summary

### Overall Assessment: {assessment}

### Key Performance Metrics

{metrics_table}

### Performance Summary

{section_data.get('performance_summary', 'No performance summary available.')}

### Risk Summary

{section_data.get('risk_summary', 'No risk summary available.')}

{findings_md}{recommendations_md}"""


def generate_performance_analysis_md(section_data: Dict[str, Any]) -> str:
    """Generate performance analysis markdown"""
    if not section_data:
        return ""
    
    return_analysis = section_data.get('return_analysis', {})
    risk_adjusted = section_data.get('risk_adjusted_metrics', {})
    benchmark = section_data.get('benchmark_comparison', {})
    
    # Return analysis table
    return_stats = return_analysis.get('summary_statistics', {})
    returns_table = "| Metric | Value |\n|--------|-------|\n"
    for metric, value in return_stats.items():
        returns_table += f"| {metric.replace('_', ' ').title()} | {value} |\n"
    
    # Risk-adjusted metrics
    risk_adj_md = ""
    if risk_adjusted:
        risk_adj_md = "### Risk-Adjusted Performance\n\n"
        for metric, data in risk_adjusted.items():
            if isinstance(data, dict):
                risk_adj_md += f"**{metric.replace('_', ' ').title()}**: {data.get('value', 'N/A')}\n"
                if 'interpretation' in data:
                    risk_adj_md += f"- {data['interpretation']}\n"
                risk_adj_md += "\n"
    
    # Benchmark comparison
    benchmark_md = ""
    if benchmark:
        benchmark_md = "### Benchmark Comparison\n\n| Metric | Value |\n|--------|-------|\n"
        for metric, value in benchmark.items():
            benchmark_md += f"| {metric.replace('_', ' ').title()} | {value} |\n"
        benchmark_md += "\n"
    
    return f"""## Performance Analysis

### Return Analysis

{returns_table}

{risk_adj_md}{benchmark_md}"""


def generate_risk_analysis_md(section_data: Dict[str, Any]) -> str:
    """Generate risk analysis markdown"""
    if not section_data:
        return ""
    
    drawdown = section_data.get('drawdown_analysis', {})
    volatility = section_data.get('volatility_analysis', {})
    var_analysis = section_data.get('var_analysis', {})
    
    # Drawdown analysis
    drawdown_md = ""
    if drawdown:
        dd_summary = drawdown.get('maximum_drawdown', {})
        drawdown_md = f"""### Drawdown Analysis

**Maximum Drawdown**: {dd_summary.get('value', 'N/A')} (Date: {dd_summary.get('date', 'N/A')})

**Average Drawdown**: {drawdown.get('average_drawdown', 'N/A')}

"""
        
        # Top drawdowns table
        top_drawdowns = drawdown.get('top_drawdowns', [])
        if top_drawdowns:
            drawdown_md += "#### Top 5 Drawdowns\n\n"
            drawdown_md += "| Rank | Depth | Duration | Period |\n|------|-------|----------|--------|\n"
            for dd in top_drawdowns[:5]:
                drawdown_md += f"| {dd.get('rank', 'N/A')} | {dd.get('depth', 'N/A')} | {dd.get('duration', 'N/A')} | {dd.get('period', 'N/A')} |\n"
            drawdown_md += "\n"
    
    # Volatility analysis
    volatility_md = ""
    if volatility:
        vol_current = volatility.get('current_volatility', {})
        volatility_md = f"""### Volatility Analysis

**Current Volatility**: {vol_current.get('annualized', 'N/A')}

"""
    
    # VaR analysis
    var_md = ""
    if var_analysis:
        var_md = "### Value at Risk (VaR)\n\n"
        for confidence, metrics in var_analysis.items():
            if isinstance(metrics, dict):
                var_md += f"**{confidence}**: {metrics.get('historical_var', 'N/A')}\n"
                if 'interpretation' in metrics:
                    var_md += f"- {metrics['interpretation']}\n"
                var_md += "\n"
    
    return f"""## Risk Analysis

{drawdown_md}{volatility_md}{var_md}"""


def generate_trade_analysis_md(section_data: Dict[str, Any]) -> str:
    """Generate enhanced trade analysis markdown"""
    if not section_data:
        return ""
    
    if 'message' in section_data:
        return f"""## Trade Analysis

{section_data['message']}
"""
    
    statistics = section_data.get('trade_statistics', {})
    win_loss = section_data.get('win_loss_analysis', {})
    duration = section_data.get('trade_duration_analysis', {})
    price_analysis = section_data.get('price_analysis', {})
    stop_loss_analysis = section_data.get('stop_loss_analysis', {})
    risk_analysis = section_data.get('risk_analysis', {})
    detailed_trades = section_data.get('detailed_trades', [])
    
    # Trade statistics
    trade_stats_md = ""
    if statistics and 'summary' in statistics:
        trade_stats_md = "### Trade Statistics\n\n"
        trade_stats_md += "| Metric | Value |\n|--------|-------|\n"
        for metric, value in statistics['summary'].items():
            trade_stats_md += f"| {metric.replace('_', ' ').title()} | {value} |\n"
        trade_stats_md += "\n"
    
    # Win/Loss analysis
    win_loss_md = ""
    if win_loss:
        win_data = win_loss.get('winning_trades', {})
        loss_data = win_loss.get('losing_trades', {})
        
        win_loss_md = f"""### Win/Loss Analysis

**Win Rate**: {win_loss.get('win_rate', 'N/A')}

**Profit Factor**: {statistics.get('profitability', {}).get('profit_factor', 'N/A')}

**Win/Loss Ratio**: {win_loss.get('ratios', {}).get('win_loss_ratio', 'N/A')}

#### Detailed Comparison

| Metric | Winners | Losers |
|--------|---------|--------|
| Count | {win_data.get('count', 0)} | {loss_data.get('count', 0)} |
| Average P&L | {win_data.get('avg_win', 'N/A')} | {loss_data.get('avg_loss', 'N/A')} |
| Median P&L | {win_data.get('median_win', 'N/A')} | {loss_data.get('median_loss', 'N/A')} |
| Largest | {win_data.get('largest_win', 'N/A')} | {loss_data.get('largest_loss', 'N/A')} |
| Avg Duration | {win_data.get('avg_duration', 'N/A')} | {loss_data.get('avg_duration', 'N/A')} |

"""
    
    # Price analysis
    price_analysis_md = ""
    if price_analysis:
        price_analysis_md = "### Price Analysis\n\n"
        price_analysis_md += "| Metric | Value |\n|--------|-------|\n"
        for metric, value in price_analysis.items():
            if metric != 'detailed_trades':
                price_analysis_md += f"| {metric.replace('_', ' ').title()} | {value} |\n"
        price_analysis_md += "\n"
    
    # Stop loss analysis
    stop_loss_md = ""
    if stop_loss_analysis:
        stop_loss_md = "### Stop Loss Analysis\n\n"
        stop_loss_md += "| Metric | Value |\n|--------|-------|\n"
        for metric, value in stop_loss_analysis.items():
            if isinstance(value, (int, float, str)):
                stop_loss_md += f"| {metric.replace('_', ' ').title()} | {value} |\n"
        stop_loss_md += "\n"
    
    # Risk analysis
    risk_analysis_md = ""
    if risk_analysis:
        risk_analysis_md = "### Risk per Trade Analysis\n\n"
        risk_analysis_md += "| Metric | Value |\n|--------|-------|\n"
        for metric, value in risk_analysis.items():
            if isinstance(value, (int, float, str)):
                risk_analysis_md += f"| {metric.replace('_', ' ').title()} | {value} |\n"
        risk_analysis_md += "\n"
    
    # Detailed trades table
    detailed_trades_md = ""
    if detailed_trades:
        detailed_trades_md = "### Detailed Trades\n\n"
        detailed_trades_md += "| Trade ID | Entry Time | Exit Time | Side | Entry Price | Exit Price | Stop Loss | Take Profit | P&L | Duration |\n"
        detailed_trades_md += "|----------|------------|-----------|------|-------------|------------|-----------|-------------|-----|----------|\n"
        
        # Show first 50 trades in markdown to avoid overwhelming the document
        for trade in detailed_trades[:50]:
            detailed_trades_md += f"| {trade.get('trade_id', 'N/A')} | {trade.get('entry_time', 'N/A')} | {trade.get('exit_time', 'N/A')} | {trade.get('side', 'N/A')} | {trade.get('entry_price', 'N/A')} | {trade.get('exit_price', 'N/A')} | {trade.get('stop_loss', 'N/A')} | {trade.get('take_profit', 'N/A')} | {trade.get('pnl', 'N/A')} | {trade.get('duration', 'N/A')} |\n"
        
        if len(detailed_trades) > 50:
            detailed_trades_md += f"\n*Showing first 50 trades out of {len(detailed_trades)} total trades.*\n"
        detailed_trades_md += "\n"
    
    # Duration analysis
    duration_md = ""
    if duration and 'message' not in duration:
        duration_md = "### Trade Duration Analysis\n\n"
        
        # Overall statistics
        overall_stats = duration.get('overall_statistics', {})
        if overall_stats:
            duration_md += "#### Overall Statistics\n\n"
            duration_md += "| Metric | Value |\n|--------|-------|\n"
            for metric, value in overall_stats.items():
                duration_md += f"| {metric.replace('_', ' ').title()} | {value} |\n"
            duration_md += "\n"
        
        # Duration buckets
        buckets = duration.get('duration_buckets', {})
        if buckets:
            duration_md += "#### Performance by Duration Range\n\n"
            duration_md += "| Duration Range | Count | Win Rate | Avg P&L |\n|----------------|-------|----------|--------|\n"
            for bucket, data in buckets.items():
                if isinstance(data, dict):
                    duration_md += f"| {bucket} | {data.get('count', 0)} | {data.get('win_rate', 'N/A')} | {data.get('avg_pnl', 'N/A')} |\n"
            duration_md += "\n"
    
    return f"""## Trade Analysis

{trade_stats_md}{win_loss_md}{price_analysis_md}{stop_loss_md}{risk_analysis_md}{detailed_trades_md}{duration_md}"""


def generate_market_regime_analysis_md(section_data: Dict[str, Any]) -> str:
    """Generate market regime analysis markdown"""
    if not section_data:
        return ""
    
    identification = section_data.get('regime_identification', {})
    performance = section_data.get('regime_performance', {})
    
    # Current regime
    current_regime = identification.get('current_regime', 'Unknown')
    
    # Regime distribution
    regime_dist = identification.get('regime_distribution', {})
    dist_table = ""
    if regime_dist:
        dist_table = "### Regime Distribution\n\n"
        dist_table += "| Regime | Percentage | Periods |\n|--------|------------|--------|\n"
        for regime, data in regime_dist.items():
            if isinstance(data, dict):
                dist_table += f"| {regime} | {data.get('percentage', 'N/A')} | {data.get('periods', 0)} |\n"
        dist_table += "\n"
    
    # Performance by regime
    perf_table = ""
    if performance:
        perf_table = "### Performance by Regime\n\n"
        perf_table += "| Regime | Annual Return | Sharpe Ratio | Win Rate |\n|--------|---------------|--------------|----------|\n"
        for regime, metrics in performance.items():
            if isinstance(metrics, dict):
                perf_table += f"| {regime.replace('_', ' ').title()} | {metrics.get('returns', 'N/A')} | {metrics.get('sharpe', 'N/A')} | {metrics.get('win_rate', 'N/A')} |\n"
        perf_table += "\n"
    
    return f"""## Market Regime Analysis

**Current Market Regime**: {current_regime}

{dist_table}{perf_table}"""


def generate_technical_details_md(section_data: Dict[str, Any]) -> str:
    """Generate technical details markdown"""
    if not section_data:
        return ""
    
    config = section_data.get('strategy_configuration', {})
    execution = section_data.get('execution_statistics', {})
    data_quality = section_data.get('data_quality', {})
    assumptions = section_data.get('backtest_assumptions', {})
    notes = section_data.get('implementation_notes', [])
    
    # Strategy configuration
    config_md = ""
    if config:
        params = config.get('parameters', {})
        if params:
            config_md = "### Strategy Configuration\n\n"
            config_md += "| Parameter | Value | Type | Description |\n|-----------|-------|------|-------------|\n"
            for param, info in params.items():
                if isinstance(info, dict):
                    config_md += f"| {param} | {info.get('value', '')} | {info.get('type', '')} | {info.get('description', '')} |\n"
            config_md += "\n"
    
    # Execution statistics
    exec_md = ""
    if execution:
        exec_md = "### Execution Statistics\n\n"
        order_exec = execution.get('order_execution', {})
        if order_exec:
            exec_md += "| Metric | Value |\n|--------|-------|\n"
            for metric, value in order_exec.items():
                exec_md += f"| {metric.replace('_', ' ').title()} | {value} |\n"
            exec_md += "\n"
    
    # Data quality
    data_quality_md = ""
    if data_quality:
        data_quality_md = "### Data Quality\n\n"
        
        coverage = data_quality.get('data_coverage', {})
        if coverage:
            data_quality_md += "#### Data Coverage\n\n"
            data_quality_md += f"- **Period**: {coverage.get('start_date', 'N/A')} to {coverage.get('end_date', 'N/A')}\n"
            data_quality_md += f"- **Total Bars**: {coverage.get('total_bars', 'N/A')}\n"
            data_quality_md += f"- **Missing Data**: {coverage.get('missing_data', 'N/A')}\n\n"
        
        integrity = data_quality.get('data_integrity', {})
        if integrity:
            data_quality_md += "#### Data Integrity\n\n"
            data_quality_md += f"- **Outliers**: {integrity.get('outliers_detected', 0)}\n"
            data_quality_md += f"- **Data Gaps**: {integrity.get('data_gaps', 0)}\n"
            data_quality_md += f"- **Suspicious Values**: {integrity.get('suspicious_values', 0)}\n\n"
    
    # Assumptions
    assumptions_md = ""
    if assumptions:
        assumptions_md = "### Backtest Assumptions\n\n"
        for category, items in assumptions.items():
            if isinstance(items, dict):
                assumptions_md += f"#### {category.replace('_', ' ').title()}\n\n"
                for key, value in items.items():
                    assumptions_md += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                assumptions_md += "\n"
    
    # Implementation notes
    notes_md = ""
    if notes:
        notes_md = "### Implementation Notes\n\n"
        for note in notes:
            notes_md += f"- {note}\n"
        notes_md += "\n"
    
    return f"""## Technical Details

{config_md}{exec_md}{data_quality_md}{assumptions_md}{notes_md}"""


def format_trade_table_row(trade: Dict[str, Any], price_format: str = "absolute") -> List[str]:
    """Format a single trade row for markdown table"""
    if price_format == "percentage":
        # Format prices as percentage changes
        entry_price = trade.get('entry_price', 'N/A')
        exit_price = trade.get('exit_price', 'N/A')
        
        if entry_price != 'N/A' and exit_price != 'N/A':
            try:
                pct_change = ((float(exit_price) - float(entry_price)) / float(entry_price)) * 100
                exit_price = f"{pct_change:.2f}%"
            except (ValueError, ZeroDivisionError):
                pass
    
    return [
        str(trade.get('trade_id', 'N/A')),
        str(trade.get('entry_time', 'N/A')),
        str(trade.get('exit_time', 'N/A')),
        str(trade.get('side', 'N/A')),
        str(trade.get('entry_price', 'N/A')),
        str(trade.get('exit_price', 'N/A')),
        str(trade.get('stop_loss', 'N/A')),
        str(trade.get('take_profit', 'N/A')),
        str(trade.get('pnl', 'N/A')),
        str(trade.get('duration', 'N/A'))
    ]


def create_enhanced_trade_summary(trades: pd.DataFrame) -> Dict[str, Any]:
    """Create enhanced trade summary with price analysis"""
    if trades.empty:
        return {'message': 'No trades available for analysis'}
    
    summary = {}
    
    # Basic trade statistics
    summary['total_trades'] = len(trades)
    summary['winning_trades'] = len(trades[trades['pnl'] > 0]) if 'pnl' in trades.columns else 0
    summary['losing_trades'] = len(trades[trades['pnl'] <= 0]) if 'pnl' in trades.columns else 0
    summary['win_rate'] = (summary['winning_trades'] / summary['total_trades']) * 100 if summary['total_trades'] > 0 else 0
    
    # Price analysis
    if 'entry_price' in trades.columns:
        summary['avg_entry_price'] = trades['entry_price'].mean()
        summary['min_entry_price'] = trades['entry_price'].min()
        summary['max_entry_price'] = trades['entry_price'].max()
    
    if 'exit_price' in trades.columns:
        summary['avg_exit_price'] = trades['exit_price'].mean()
        summary['min_exit_price'] = trades['exit_price'].min()
        summary['max_exit_price'] = trades['exit_price'].max()
    
    # Stop loss analysis
    if 'stop_loss' in trades.columns:
        stop_trades = trades.dropna(subset=['stop_loss'])
        if not stop_trades.empty:
            summary['trades_with_stop_loss'] = len(stop_trades)
            summary['stop_loss_usage_rate'] = (len(stop_trades) / len(trades)) * 100
            
            # Calculate stop loss hit rate
            if 'exit_reason' in trades.columns:
                stop_hits = trades[trades['exit_reason'].str.contains('stop', case=False, na=False)]
                summary['stop_loss_hit_rate'] = (len(stop_hits) / len(stop_trades)) * 100
    
    # Risk analysis
    if 'entry_price' in trades.columns and 'stop_loss' in trades.columns:
        stop_trades = trades.dropna(subset=['entry_price', 'stop_loss'])
        if not stop_trades.empty:
            stop_trades['risk_pct'] = abs(stop_trades['stop_loss'] - stop_trades['entry_price']) / stop_trades['entry_price'] * 100
            summary['avg_risk_per_trade'] = stop_trades['risk_pct'].mean()
            summary['max_risk_per_trade'] = stop_trades['risk_pct'].max()
            summary['min_risk_per_trade'] = stop_trades['risk_pct'].min()
    
    return summary