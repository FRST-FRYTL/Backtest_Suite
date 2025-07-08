"""
Alert engine and notification system for monitoring.
"""

import asyncio
import json
import logging
import smtplib
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Deque

import aiohttp
from jinja2 import Template

from .config import AlertConfig


class AlertType(Enum):
    """Types of alerts."""
    DRAWDOWN = "drawdown"
    POSITION_SIZE = "position_size"
    WIN_RATE = "win_rate"
    SYSTEM_ERROR = "system_error"
    SIGNAL = "signal"
    RISK = "risk"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Alert:
    """Represents a single alert."""
    id: str
    type: AlertType
    priority: AlertPriority
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'type': self.type.value,
            'priority': self.priority.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'acknowledged': self.acknowledged
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create alert from dictionary."""
        return cls(
            id=data['id'],
            type=AlertType(data['type']),
            priority=AlertPriority(data['priority']),
            title=data['title'],
            message=data['message'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            data=data.get('data', {}),
            acknowledged=data.get('acknowledged', False)
        )


class AlertEngine:
    """Main alert engine for managing and dispatching alerts."""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Alert storage
        self.alerts: Deque[Alert] = deque(maxlen=1000)
        self.alert_history: List[Alert] = []
        
        # Rate limiting
        self.alert_timestamps: Deque[float] = deque()
        self.cooldown_alerts: Dict[str, float] = {}  # alert_key -> last_sent_time
        
        # Handlers
        self.handlers: Dict[AlertType, List[Callable]] = {
            alert_type: [] for alert_type in AlertType
        }
        
        # Templates for notifications
        self.email_template = Template("""
        <html>
        <body>
            <h2>{{ alert.title }}</h2>
            <p><strong>Type:</strong> {{ alert.type.value }}</p>
            <p><strong>Priority:</strong> {{ alert.priority.name }}</p>
            <p><strong>Time:</strong> {{ alert.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}</p>
            <hr>
            <p>{{ alert.message }}</p>
            {% if alert.data %}
            <h3>Additional Data:</h3>
            <pre>{{ alert.data | tojson(indent=2) }}</pre>
            {% endif %}
        </body>
        </html>
        """)
        
        # Async event loop for notifications
        self._notification_task: Optional[asyncio.Task] = None
        self._notification_queue: asyncio.Queue = asyncio.Queue()
        
    async def start(self):
        """Start the alert engine."""
        if self.config.enabled:
            self._notification_task = asyncio.create_task(self._notification_worker())
            self.logger.info("Alert engine started")
    
    async def stop(self):
        """Stop the alert engine."""
        if self._notification_task:
            self._notification_task.cancel()
            try:
                await self._notification_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Alert engine stopped")
    
    def register_handler(self, alert_type: AlertType, handler: Callable):
        """Register a custom handler for specific alert type."""
        self.handlers[alert_type].append(handler)
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send an alert through the system."""
        # Check rate limiting
        if not self._check_rate_limit():
            self.logger.warning(f"Rate limit exceeded, dropping alert: {alert.title}")
            return False
        
        # Check cooldown
        alert_key = f"{alert.type.value}:{alert.title}"
        if alert_key in self.cooldown_alerts:
            last_sent = self.cooldown_alerts[alert_key]
            if time.time() - last_sent < self.config.alert_cooldown_seconds:
                self.logger.debug(f"Alert in cooldown: {alert.title}")
                return False
        
        # Store alert
        self.alerts.append(alert)
        self.alert_history.append(alert)
        
        # Update cooldown
        self.cooldown_alerts[alert_key] = time.time()
        
        # Execute handlers
        for handler in self.handlers[alert.type]:
            try:
                await handler(alert) if asyncio.iscoroutinefunction(handler) else handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
        
        # Queue for notifications
        await self._notification_queue.put(alert)
        
        self.logger.info(f"Alert sent: [{alert.priority.name}] {alert.title}")
        return True
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        current_time = time.time()
        cutoff_time = current_time - 60  # 1 minute window
        
        # Remove old timestamps
        while self.alert_timestamps and self.alert_timestamps[0] < cutoff_time:
            self.alert_timestamps.popleft()
        
        # Check limit
        if len(self.alert_timestamps) >= self.config.max_alerts_per_minute:
            return False
        
        self.alert_timestamps.append(current_time)
        return True
    
    async def _notification_worker(self):
        """Worker to process notification queue."""
        while True:
            try:
                alert = await self._notification_queue.get()
                
                # Send notifications based on configuration
                tasks = []
                
                if self.config.email_enabled:
                    tasks.append(self._send_email_notification(alert))
                
                if self.config.webhook_enabled:
                    tasks.append(self._send_webhook_notification(alert))
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in notification worker: {e}")
    
    async def _send_email_notification(self, alert: Alert):
        """Send email notification."""
        try:
            # This is a placeholder - in production, you'd configure SMTP settings
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.priority.name}] {alert.title}"
            msg['From'] = "backtest-suite@example.com"
            msg['To'] = ", ".join(self.config.email_recipients)
            
            # Create HTML content
            html_content = self.email_template.render(alert=alert)
            msg.attach(MIMEText(html_content, 'html'))
            
            # In production, you would send via SMTP here
            self.logger.info(f"Email notification queued for: {alert.title}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
    
    async def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification."""
        async with aiohttp.ClientSession() as session:
            for webhook_url in self.config.webhook_urls:
                try:
                    payload = {
                        'alert': alert.to_dict(),
                        'source': 'backtest-suite'
                    }
                    
                    async with session.post(
                        webhook_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status != 200:
                            self.logger.warning(
                                f"Webhook returned status {response.status} for {webhook_url}"
                            )
                            
                except Exception as e:
                    self.logger.error(f"Failed to send webhook to {webhook_url}: {e}")
    
    def get_recent_alerts(self, limit: int = 50) -> List[Alert]:
        """Get recent alerts."""
        return list(self.alerts)[-limit:]
    
    def get_alerts_by_type(self, alert_type: AlertType) -> List[Alert]:
        """Get alerts filtered by type."""
        return [alert for alert in self.alerts if alert.type == alert_type]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def save_history(self, filepath: Path):
        """Save alert history to file."""
        data = [alert.to_dict() for alert in self.alert_history]
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_history(self, filepath: Path):
        """Load alert history from file."""
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.alert_history = [Alert.from_dict(item) for item in data]


# Helper functions for common alerts
def create_drawdown_alert(current_drawdown: float, max_drawdown: float) -> Alert:
    """Create a drawdown alert."""
    return Alert(
        id=f"dd_{int(time.time() * 1000)}",
        type=AlertType.DRAWDOWN,
        priority=AlertPriority.HIGH if current_drawdown > max_drawdown * 0.8 else AlertPriority.MEDIUM,
        title="Drawdown Alert",
        message=f"Current drawdown ({current_drawdown:.2%}) is approaching maximum ({max_drawdown:.2%})",
        data={
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown
        }
    )


def create_position_size_alert(symbol: str, position_size: float, threshold: float) -> Alert:
    """Create a position size alert."""
    return Alert(
        id=f"ps_{symbol}_{int(time.time() * 1000)}",
        type=AlertType.POSITION_SIZE,
        priority=AlertPriority.HIGH,
        title="Position Size Alert",
        message=f"Position in {symbol} ({position_size:.2%}) exceeds threshold ({threshold:.2%})",
        data={
            'symbol': symbol,
            'position_size': position_size,
            'threshold': threshold
        }
    )


def create_signal_alert(symbol: str, signal_type: str, strength: float) -> Alert:
    """Create a trading signal alert."""
    return Alert(
        id=f"sig_{symbol}_{int(time.time() * 1000)}",
        type=AlertType.SIGNAL,
        priority=AlertPriority.MEDIUM if strength < 0.8 else AlertPriority.HIGH,
        title=f"Trading Signal: {signal_type}",
        message=f"{signal_type} signal detected for {symbol} with strength {strength:.2f}",
        data={
            'symbol': symbol,
            'signal_type': signal_type,
            'strength': strength
        }
    )