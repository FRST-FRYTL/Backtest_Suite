"""
Standardized Reporting System for Backtest Suite

This module provides a consistent, professional reporting framework
for all backtest results and analysis outputs.
"""

from .standard_report_generator import StandardReportGenerator
from .report_sections import (
    ExecutiveSummary,
    PerformanceAnalysis,
    RiskAnalysis,
    TradeAnalysis,
    MarketRegimeAnalysis,
    TechnicalDetails
)
from .visualization_types import (
    EquityCurveChart,
    DrawdownChart,
    ReturnsDistribution,
    TradeScatterPlot,
    RollingMetricsChart,
    HeatmapVisualization
)

__all__ = [
    'StandardReportGenerator',
    'ExecutiveSummary',
    'PerformanceAnalysis',
    'RiskAnalysis',
    'TradeAnalysis',
    'MarketRegimeAnalysis',
    'TechnicalDetails',
    'EquityCurveChart',
    'DrawdownChart',
    'ReturnsDistribution',
    'TradeScatterPlot',
    'RollingMetricsChart',
    'HeatmapVisualization'
]