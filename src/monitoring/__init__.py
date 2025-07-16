"""
Real-time monitoring and alerts system for Backtest Suite.

This module provides:
- Real-time performance tracking
- Alert system for trading signals
- Position monitoring dashboard
- Risk exposure tracking
- System health monitoring
"""

from .alerts import AlertEngine, Alert, AlertType, AlertPriority, create_drawdown_alert, create_position_size_alert, create_signal_alert
from .dashboard import MonitoringDashboard, DashboardConfig
from .collectors import (
    MetricCollector,
    PerformanceCollector,
    PositionCollector,
    SystemHealthCollector,
    MetricPoint,
    MetricBuffer,
    MetricAggregator
)
from .config import MonitoringConfig, AlertConfig, CollectorConfig

__all__ = [
    # Alerts
    'AlertEngine',
    'Alert',
    'AlertType',
    'AlertPriority',
    'create_drawdown_alert',
    'create_position_size_alert',
    'create_signal_alert',
    
    # Dashboard
    'MonitoringDashboard',
    'DashboardConfig',
    
    # Collectors
    'MetricCollector',
    'PerformanceCollector',
    'PositionCollector',
    'SystemHealthCollector',
    'MetricPoint',
    'MetricBuffer',
    'MetricAggregator',
    
    # Config
    'MonitoringConfig',
    'AlertConfig',
    'CollectorConfig',
]