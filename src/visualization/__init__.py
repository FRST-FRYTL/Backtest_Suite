"""Visualization module for backtesting results."""

from .charts import ChartGenerator
from .dashboard import Dashboard

try:
    from .enhanced_interactive_charts import EnhancedInteractiveCharts
    ENHANCED_CHARTS_AVAILABLE = True
except ImportError:
    ENHANCED_CHARTS_AVAILABLE = False

__all__ = ["ChartGenerator", "Dashboard"]

if ENHANCED_CHARTS_AVAILABLE:
    __all__.append("EnhancedInteractiveCharts")