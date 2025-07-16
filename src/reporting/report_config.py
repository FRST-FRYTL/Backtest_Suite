"""
Configuration system for standardized reports.

This module defines the structure, sections, and customization options
for generating consistent reports across different strategies and backtests.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class ReportSection(Enum):
    """Standard report sections."""
    EXECUTIVE_SUMMARY = "executive_summary"
    METHODOLOGY = "methodology"
    PERFORMANCE_METRICS = "performance_metrics"
    RISK_ANALYSIS = "risk_analysis"
    TRADE_ANALYSIS = "trade_analysis"
    PARAMETER_OPTIMIZATION = "parameter_optimization"
    TIMEFRAME_ANALYSIS = "timeframe_analysis"
    RECOMMENDATIONS = "recommendations"
    TECHNICAL_DETAILS = "technical_details"
    APPENDIX = "appendix"


@dataclass
class SectionConfig:
    """Configuration for a report section."""
    title: str
    enabled: bool = True
    order: int = 0
    subsections: List[str] = field(default_factory=list)
    custom_content: Optional[str] = None
    visualizations: List[str] = field(default_factory=list)


@dataclass
class MetricThresholds:
    """Thresholds for categorizing metric values."""
    excellent: float
    good: float
    acceptable: float
    poor: float
    
    def categorize(self, value: float, higher_is_better: bool = True) -> str:
        """Categorize a metric value based on thresholds."""
        if higher_is_better:
            if value >= self.excellent:
                return "Excellent"
            elif value >= self.good:
                return "Good"
            elif value >= self.acceptable:
                return "Acceptable"
            else:
                return "Poor"
        else:
            if value <= self.excellent:
                return "Excellent"
            elif value <= self.good:
                return "Good"
            elif value <= self.acceptable:
                return "Acceptable"
            else:
                return "Poor"


@dataclass
class TradeReportingConfig:
    """Configuration for trade-specific reporting features."""
    enable_detailed_trade_prices: bool = True
    price_display_format: str = "absolute"  # "absolute" or "percentage"
    show_entry_exit_prices: bool = True
    show_stop_loss_prices: bool = True
    show_take_profit_prices: bool = True
    enable_stop_loss_analysis: bool = True
    enable_risk_per_trade_analysis: bool = True
    max_trades_in_detailed_table: int = 100
    include_trade_timing_analysis: bool = True
    show_trade_price_charts: bool = True


@dataclass
class ReportStyle:
    """Styling configuration for reports."""
    # Color scheme
    primary_color: str = "#2E86AB"
    secondary_color: str = "#A23B72"
    success_color: str = "#27AE60"
    warning_color: str = "#F39C12"
    danger_color: str = "#E74C3C"
    
    # Chart styling
    chart_template: str = "plotly_white"
    chart_height: int = 500
    chart_width: int = 800
    
    # Table styling
    table_style: str = "striped"
    highlight_best: bool = True
    
    # Font settings
    font_family: str = "Arial, sans-serif"
    font_size: int = 12
    
    # Logo and branding
    logo_path: Optional[str] = None
    company_name: Optional[str] = None


class ReportConfig:
    """Main configuration class for report generation."""
    
    def __init__(self):
        self.sections = self._initialize_sections()
        self.metrics = self._initialize_metrics()
        self.thresholds = self._initialize_thresholds()
        self.style = ReportStyle()
        self.trade_reporting = TradeReportingConfig()
        
    def _initialize_sections(self) -> Dict[ReportSection, SectionConfig]:
        """Initialize default section configurations."""
        return {
            ReportSection.EXECUTIVE_SUMMARY: SectionConfig(
                title="Executive Summary",
                order=1,
                subsections=["Key Achievements", "Performance Highlights", "Critical Findings"],
                visualizations=["performance_summary_chart"]
            ),
            ReportSection.METHODOLOGY: SectionConfig(
                title="Methodology",
                order=2,
                subsections=["Data Collection", "Strategy Description", "Evaluation Metrics"],
                visualizations=[]
            ),
            ReportSection.PERFORMANCE_METRICS: SectionConfig(
                title="Performance Metrics",
                order=3,
                subsections=["Returns Analysis", "Risk-Adjusted Metrics", "Comparison to Benchmark"],
                visualizations=["cumulative_returns", "monthly_returns_heatmap", "performance_table"]
            ),
            ReportSection.RISK_ANALYSIS: SectionConfig(
                title="Risk Analysis",
                order=4,
                subsections=["Drawdown Analysis", "Volatility Metrics", "Risk Metrics"],
                visualizations=["drawdown_chart", "rolling_volatility", "risk_metrics_radar"]
            ),
            ReportSection.TRADE_ANALYSIS: SectionConfig(
                title="Trade Analysis",
                order=5,
                subsections=["Trade Statistics", "Win/Loss Distribution", "Trade Duration", "Trade Prices", "Stop Loss Analysis", "Risk per Trade"],
                visualizations=["trade_distribution", "win_loss_chart", "trade_duration_histogram", "trade_price_chart", "stop_loss_analysis", "trade_risk_chart"]
            ),
            ReportSection.PARAMETER_OPTIMIZATION: SectionConfig(
                title="Parameter Optimization",
                order=6,
                subsections=["Parameter Grid", "Optimization Results", "Sensitivity Analysis"],
                visualizations=["parameter_heatmap", "optimization_surface", "sensitivity_charts"]
            ),
            ReportSection.TIMEFRAME_ANALYSIS: SectionConfig(
                title="Timeframe Analysis",
                order=7,
                subsections=["Performance by Timeframe", "Optimal Parameters", "Recommendations"],
                visualizations=["timeframe_comparison", "parameter_stability"]
            ),
            ReportSection.RECOMMENDATIONS: SectionConfig(
                title="Recommendations",
                order=8,
                subsections=["Trading Guidelines", "Risk Management", "Implementation Notes"],
                visualizations=[]
            ),
            ReportSection.TECHNICAL_DETAILS: SectionConfig(
                title="Technical Details",
                order=9,
                subsections=["Backtest Configuration", "Data Quality", "Limitations"],
                visualizations=[]
            ),
            ReportSection.APPENDIX: SectionConfig(
                title="Appendix",
                order=10,
                subsections=["Glossary", "References", "Additional Charts"],
                visualizations=[]
            )
        }
    
    def _initialize_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Initialize metric definitions and properties."""
        return {
            # Return metrics
            "total_return": {
                "name": "Total Return",
                "format": "percentage",
                "higher_is_better": True,
                "description": "Total return over the backtest period"
            },
            "annual_return": {
                "name": "Annual Return",
                "format": "percentage",
                "higher_is_better": True,
                "description": "Annualized return"
            },
            "monthly_return": {
                "name": "Average Monthly Return",
                "format": "percentage",
                "higher_is_better": True,
                "description": "Average monthly return"
            },
            
            # Risk metrics
            "sharpe_ratio": {
                "name": "Sharpe Ratio",
                "format": "decimal",
                "higher_is_better": True,
                "description": "Risk-adjusted return metric"
            },
            "sortino_ratio": {
                "name": "Sortino Ratio",
                "format": "decimal",
                "higher_is_better": True,
                "description": "Downside risk-adjusted return"
            },
            "max_drawdown": {
                "name": "Maximum Drawdown",
                "format": "percentage",
                "higher_is_better": False,
                "description": "Largest peak-to-trough decline"
            },
            "volatility": {
                "name": "Annual Volatility",
                "format": "percentage",
                "higher_is_better": False,
                "description": "Standard deviation of returns"
            },
            
            # Trade metrics
            "total_trades": {
                "name": "Total Trades",
                "format": "integer",
                "higher_is_better": None,
                "description": "Total number of trades"
            },
            "win_rate": {
                "name": "Win Rate",
                "format": "percentage",
                "higher_is_better": True,
                "description": "Percentage of profitable trades"
            },
            "profit_factor": {
                "name": "Profit Factor",
                "format": "decimal",
                "higher_is_better": True,
                "description": "Gross profit / Gross loss"
            },
            "avg_win": {
                "name": "Average Win",
                "format": "currency",
                "higher_is_better": True,
                "description": "Average profit per winning trade"
            },
            "avg_loss": {
                "name": "Average Loss",
                "format": "currency",
                "higher_is_better": False,
                "description": "Average loss per losing trade"
            },
            "win_loss_ratio": {
                "name": "Win/Loss Ratio",
                "format": "decimal",
                "higher_is_better": True,
                "description": "Average win / Average loss"
            },
            
            # Trade price metrics
            "avg_entry_price": {
                "name": "Average Entry Price",
                "format": "currency",
                "higher_is_better": None,
                "description": "Average entry price across all trades"
            },
            "avg_exit_price": {
                "name": "Average Exit Price",
                "format": "currency",
                "higher_is_better": None,
                "description": "Average exit price across all trades"
            },
            "stop_loss_hit_rate": {
                "name": "Stop Loss Hit Rate",
                "format": "percentage",
                "higher_is_better": False,
                "description": "Percentage of trades that hit stop loss"
            },
            "avg_risk_per_trade": {
                "name": "Average Risk per Trade",
                "format": "percentage",
                "higher_is_better": False,
                "description": "Average risk per trade based on stop distance"
            },
            "risk_reward_ratio": {
                "name": "Risk/Reward Ratio",
                "format": "decimal",
                "higher_is_better": False,
                "description": "Average risk to reward ratio"
            }
        }
    
    def _initialize_thresholds(self) -> Dict[str, MetricThresholds]:
        """Initialize default thresholds for metrics."""
        return {
            "sharpe_ratio": MetricThresholds(
                excellent=2.0,
                good=1.5,
                acceptable=1.0,
                poor=0.5
            ),
            "sortino_ratio": MetricThresholds(
                excellent=2.5,
                good=2.0,
                acceptable=1.5,
                poor=1.0
            ),
            "max_drawdown": MetricThresholds(
                excellent=0.10,  # 10%
                good=0.15,       # 15%
                acceptable=0.25,  # 25%
                poor=0.35        # 35%
            ),
            "win_rate": MetricThresholds(
                excellent=0.60,  # 60%
                good=0.55,       # 55%
                acceptable=0.50,  # 50%
                poor=0.45        # 45%
            ),
            "profit_factor": MetricThresholds(
                excellent=2.0,
                good=1.5,
                acceptable=1.2,
                poor=1.0
            ),
            "annual_return": MetricThresholds(
                excellent=0.20,  # 20%
                good=0.15,       # 15%
                acceptable=0.10,  # 10%
                poor=0.05        # 5%
            )
        }
    
    def get_section_order(self) -> List[ReportSection]:
        """Get sections in order."""
        return sorted(
            [s for s, config in self.sections.items() if config.enabled],
            key=lambda s: self.sections[s].order
        )
    
    def enable_section(self, section: ReportSection) -> None:
        """Enable a report section."""
        if section in self.sections:
            self.sections[section].enabled = True
    
    def disable_section(self, section: ReportSection) -> None:
        """Disable a report section."""
        if section in self.sections:
            self.sections[section].enabled = False
    
    def set_section_order(self, section: ReportSection, order: int) -> None:
        """Set the order for a section."""
        if section in self.sections:
            self.sections[section].order = order
    
    def add_custom_metric(self, key: str, name: str, format: str = "decimal",
                         higher_is_better: bool = True, description: str = "") -> None:
        """Add a custom metric definition."""
        self.metrics[key] = {
            "name": name,
            "format": format,
            "higher_is_better": higher_is_better,
            "description": description
        }
    
    def set_threshold(self, metric: str, excellent: float, good: float,
                     acceptable: float, poor: float) -> None:
        """Set custom thresholds for a metric."""
        self.thresholds[metric] = MetricThresholds(
            excellent=excellent,
            good=good,
            acceptable=acceptable,
            poor=poor
        )
    
    def customize_style(self, **kwargs) -> None:
        """Customize report styling."""
        for key, value in kwargs.items():
            if hasattr(self.style, key):
                setattr(self.style, key, value)