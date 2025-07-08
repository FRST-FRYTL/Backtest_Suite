"""
Real-time monitoring and alerts system for Backtest Suite.

This module provides:
- Real-time performance tracking
- Alert system for trading signals
- Position monitoring dashboard
- Risk exposure tracking
- System health monitoring
"""

from .alerts import AlertEngine, Alert, AlertType, AlertPriority
from .dashboard import MonitoringDashboard, DashboardConfig
from .collectors import (
    MetricCollector,
    PerformanceCollector,
    PositionCollector,
    SystemHealthCollector
)
from .config import MonitoringConfig

__all__ = [
    # Alerts
    'AlertEngine',
    'Alert',
    'AlertType',
    'AlertPriority',
    
    # Dashboard
    'MonitoringDashboard',
    'DashboardConfig',
    
    # Collectors
    'MetricCollector',
    'PerformanceCollector',
    'PositionCollector',
    'SystemHealthCollector',
    
    # Config
    'MonitoringConfig',
]