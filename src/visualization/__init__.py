"""Visualization module for backtesting results."""

from .charts import ChartGenerator
from .dashboard import Dashboard
from .comprehensive_trading_dashboard import ComprehensiveTradingDashboard

# Create alias for backward compatibility
DashboardBuilder = ComprehensiveTradingDashboard

try:
    from .enhanced_interactive_charts import EnhancedInteractiveCharts
    ENHANCED_CHARTS_AVAILABLE = True
except ImportError:
    ENHANCED_CHARTS_AVAILABLE = False

__all__ = ["ChartGenerator", "Dashboard", "ComprehensiveTradingDashboard", "DashboardBuilder"]

if ENHANCED_CHARTS_AVAILABLE:
    __all__.append("EnhancedInteractiveCharts")