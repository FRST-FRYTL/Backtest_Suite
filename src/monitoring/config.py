"""
Configuration for the monitoring system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


@dataclass
class AlertConfig:
    """Configuration for alert system."""
    enabled: bool = True
    email_enabled: bool = False
    webhook_enabled: bool = False
    email_recipients: List[str] = field(default_factory=list)
    webhook_urls: List[str] = field(default_factory=list)
    
    # Alert thresholds
    drawdown_threshold: float = 0.10  # 10% drawdown
    position_size_threshold: float = 0.25  # 25% of portfolio
    win_rate_threshold: float = 0.30  # Alert if win rate drops below 30%
    
    # Rate limiting
    max_alerts_per_minute: int = 10
    alert_cooldown_seconds: int = 60


@dataclass
class CollectorConfig:
    """Configuration for metric collectors."""
    collection_interval_seconds: float = 1.0
    buffer_size: int = 1000
    persist_to_disk: bool = True
    persistence_path: Path = field(default_factory=lambda: Path("data/metrics"))
    
    # Metric-specific settings
    performance_metrics: List[str] = field(default_factory=lambda: [
        "total_return", "sharpe_ratio", "win_rate", "profit_factor",
        "max_drawdown", "current_drawdown", "volatility"
    ])
    position_metrics: List[str] = field(default_factory=lambda: [
        "open_positions", "position_sizes", "exposure", "pnl"
    ])
    system_metrics: List[str] = field(default_factory=lambda: [
        "cpu_usage", "memory_usage", "latency", "tick_rate"
    ])


@dataclass 
class DashboardConfig:
    """Configuration for monitoring dashboard."""
    enabled: bool = True
    host: str = "localhost"
    port: int = 8050
    update_interval_ms: int = 1000
    
    # WebSocket settings
    websocket_enabled: bool = True
    websocket_port: int = 8051
    
    # UI settings
    theme: str = "dark"
    show_charts: bool = True
    max_chart_points: int = 500
    
    # Layout configuration
    layout: Dict[str, Any] = field(default_factory=lambda: {
        "performance": {"row": 0, "col": 0, "width": 6, "height": 4},
        "positions": {"row": 0, "col": 6, "width": 6, "height": 4},
        "alerts": {"row": 4, "col": 0, "width": 4, "height": 3},
        "system": {"row": 4, "col": 4, "width": 4, "height": 3},
        "chart": {"row": 4, "col": 8, "width": 4, "height": 3}
    })


@dataclass
class MonitoringConfig:
    """Main monitoring system configuration."""
    enabled: bool = True
    log_level: str = "INFO"
    
    # Sub-configurations
    alerts: AlertConfig = field(default_factory=AlertConfig)
    collectors: CollectorConfig = field(default_factory=CollectorConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    
    # Storage settings
    metric_retention_days: int = 30
    alert_history_days: int = 7
    
    @classmethod
    def from_file(cls, filepath: Path) -> 'MonitoringConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Parse nested configurations
        if 'alerts' in data:
            data['alerts'] = AlertConfig(**data['alerts'])
        if 'collectors' in data:
            data['collectors'] = CollectorConfig(**data['collectors'])
        if 'dashboard' in data:
            data['dashboard'] = DashboardConfig(**data['dashboard'])
            
        return cls(**data)
    
    def to_file(self, filepath: Path):
        """Save configuration to JSON file."""
        data = {
            'enabled': self.enabled,
            'log_level': self.log_level,
            'metric_retention_days': self.metric_retention_days,
            'alert_history_days': self.alert_history_days,
            'alerts': {
                'enabled': self.alerts.enabled,
                'email_enabled': self.alerts.email_enabled,
                'webhook_enabled': self.alerts.webhook_enabled,
                'email_recipients': self.alerts.email_recipients,
                'webhook_urls': self.alerts.webhook_urls,
                'drawdown_threshold': self.alerts.drawdown_threshold,
                'position_size_threshold': self.alerts.position_size_threshold,
                'win_rate_threshold': self.alerts.win_rate_threshold,
                'max_alerts_per_minute': self.alerts.max_alerts_per_minute,
                'alert_cooldown_seconds': self.alerts.alert_cooldown_seconds
            },
            'collectors': {
                'collection_interval_seconds': self.collectors.collection_interval_seconds,
                'buffer_size': self.collectors.buffer_size,
                'persist_to_disk': self.collectors.persist_to_disk,
                'persistence_path': str(self.collectors.persistence_path),
                'performance_metrics': self.collectors.performance_metrics,
                'position_metrics': self.collectors.position_metrics,
                'system_metrics': self.collectors.system_metrics
            },
            'dashboard': {
                'enabled': self.dashboard.enabled,
                'host': self.dashboard.host,
                'port': self.dashboard.port,
                'update_interval_ms': self.dashboard.update_interval_ms,
                'websocket_enabled': self.dashboard.websocket_enabled,
                'websocket_port': self.dashboard.websocket_port,
                'theme': self.dashboard.theme,
                'show_charts': self.dashboard.show_charts,
                'max_chart_points': self.dashboard.max_chart_points,
                'layout': self.dashboard.layout
            }
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Alert validation
        if self.alerts.enabled:
            if self.alerts.email_enabled and not self.alerts.email_recipients:
                errors.append("Email alerts enabled but no recipients configured")
            if self.alerts.webhook_enabled and not self.alerts.webhook_urls:
                errors.append("Webhook alerts enabled but no URLs configured")
            if self.alerts.drawdown_threshold <= 0 or self.alerts.drawdown_threshold >= 1:
                errors.append("Drawdown threshold must be between 0 and 1")
                
        # Collector validation
        if self.collectors.collection_interval_seconds <= 0:
            errors.append("Collection interval must be positive")
        if self.collectors.buffer_size <= 0:
            errors.append("Buffer size must be positive")
            
        # Dashboard validation
        if self.dashboard.enabled:
            if self.dashboard.port <= 0 or self.dashboard.port > 65535:
                errors.append("Dashboard port must be between 1 and 65535")
            if self.dashboard.websocket_port <= 0 or self.dashboard.websocket_port > 65535:
                errors.append("WebSocket port must be between 1 and 65535")
            if self.dashboard.port == self.dashboard.websocket_port:
                errors.append("Dashboard and WebSocket ports must be different")
                
        return errors


# Default configuration instance
default_config = MonitoringConfig()