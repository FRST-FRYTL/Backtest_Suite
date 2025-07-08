"""
Tests for the monitoring system.
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import pytest

from src.monitoring import (
    MonitoringConfig,
    AlertConfig,
    CollectorConfig,
    DashboardConfig,
    Alert,
    AlertType,
    AlertPriority,
    AlertEngine,
    MetricPoint,
    MetricBuffer,
    MetricCollector,
    PerformanceCollector,
    PositionCollector,
    SystemHealthCollector,
    MetricAggregator,
    create_drawdown_alert,
    create_position_size_alert,
    create_signal_alert
)
from src.backtesting import Portfolio, Position


class TestMonitoringConfig:
    """Tests for monitoring configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MonitoringConfig()
        
        assert config.enabled is True
        assert config.log_level == "INFO"
        assert config.metric_retention_days == 30
        assert config.alert_history_days == 7
        
        # Check nested configs
        assert isinstance(config.alerts, AlertConfig)
        assert isinstance(config.collectors, CollectorConfig)
        assert isinstance(config.dashboard, DashboardConfig)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = MonitoringConfig()
        
        # Valid config should have no errors
        errors = config.validate()
        assert len(errors) == 0
        
        # Test invalid alert config
        config.alerts.email_enabled = True
        config.alerts.email_recipients = []
        errors = config.validate()
        assert len(errors) > 0
        assert "Email alerts enabled but no recipients" in errors[0]
        
        # Test invalid thresholds
        config.alerts.drawdown_threshold = 1.5
        errors = config.validate()
        assert any("Drawdown threshold" in e for e in errors)
    
    def test_config_serialization(self, tmp_path):
        """Test saving and loading configuration."""
        config = MonitoringConfig()
        config.alerts.email_recipients = ["test@example.com"]
        config.dashboard.port = 8080
        
        # Save config
        config_file = tmp_path / "test_config.json"
        config.to_file(config_file)
        
        # Load config
        loaded_config = MonitoringConfig.from_file(config_file)
        
        assert loaded_config.alerts.email_recipients == ["test@example.com"]
        assert loaded_config.dashboard.port == 8080


class TestAlerts:
    """Tests for alert system."""
    
    def test_alert_creation(self):
        """Test alert creation and serialization."""
        alert = Alert(
            id="test_123",
            type=AlertType.DRAWDOWN,
            priority=AlertPriority.HIGH,
            title="Test Alert",
            message="This is a test alert",
            data={"value": 0.15}
        )
        
        # Test conversion
        alert_dict = alert.to_dict()
        assert alert_dict["id"] == "test_123"
        assert alert_dict["type"] == "drawdown"
        assert alert_dict["priority"] == 3
        
        # Test from dict
        new_alert = Alert.from_dict(alert_dict)
        assert new_alert.id == alert.id
        assert new_alert.type == alert.type
        assert new_alert.priority == alert.priority
    
    @pytest.mark.asyncio
    async def test_alert_engine_basic(self):
        """Test basic alert engine functionality."""
        config = AlertConfig()
        engine = AlertEngine(config)
        
        await engine.start()
        
        # Send alert
        alert = create_drawdown_alert(0.12, 0.20)
        sent = await engine.send_alert(alert)
        assert sent is True
        
        # Check alert storage
        recent_alerts = engine.get_recent_alerts(limit=10)
        assert len(recent_alerts) == 1
        assert recent_alerts[0].type == AlertType.DRAWDOWN
        
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_alert_rate_limiting(self):
        """Test alert rate limiting."""
        config = AlertConfig(max_alerts_per_minute=3)
        engine = AlertEngine(config)
        
        await engine.start()
        
        # Send alerts up to limit
        for i in range(3):
            alert = Alert(
                id=f"test_{i}",
                type=AlertType.SYSTEM_ERROR,
                priority=AlertPriority.LOW,
                title=f"Test {i}",
                message="Test"
            )
            sent = await engine.send_alert(alert)
            assert sent is True
        
        # Next alert should be rejected
        alert = Alert(
            id="test_overflow",
            type=AlertType.SYSTEM_ERROR,
            priority=AlertPriority.LOW,
            title="Overflow",
            message="Should be rejected"
        )
        sent = await engine.send_alert(alert)
        assert sent is False
        
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_alert_cooldown(self):
        """Test alert cooldown functionality."""
        config = AlertConfig(alert_cooldown_seconds=2)
        engine = AlertEngine(config)
        
        await engine.start()
        
        # Send first alert
        alert1 = create_signal_alert("AAPL", "BUY", 0.9)
        sent = await engine.send_alert(alert1)
        assert sent is True
        
        # Same alert should be in cooldown
        alert2 = create_signal_alert("AAPL", "BUY", 0.95)
        sent = await engine.send_alert(alert2)
        assert sent is False
        
        # Wait for cooldown
        await asyncio.sleep(2.1)
        
        # Now it should work
        alert3 = create_signal_alert("AAPL", "BUY", 0.85)
        sent = await engine.send_alert(alert3)
        assert sent is True
        
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_alert_handlers(self):
        """Test custom alert handlers."""
        config = AlertConfig()
        engine = AlertEngine(config)
        
        # Track handler calls
        handler_calls = []
        
        async def custom_handler(alert: Alert):
            handler_calls.append(alert)
        
        engine.register_handler(AlertType.POSITION_SIZE, custom_handler)
        
        await engine.start()
        
        # Send position size alert
        alert = create_position_size_alert("TSLA", 0.30, 0.25)
        await engine.send_alert(alert)
        
        # Handler should be called
        assert len(handler_calls) == 1
        assert handler_calls[0].type == AlertType.POSITION_SIZE
        
        await engine.stop()


class TestCollectors:
    """Tests for metric collectors."""
    
    @pytest.mark.asyncio
    async def test_metric_buffer(self):
        """Test metric buffer functionality."""
        buffer = MetricBuffer(max_size=100)
        
        # Add metrics
        metric1 = MetricPoint(time.time(), "test_metric", 42.0)
        metric2 = MetricPoint(time.time(), "test_metric", 43.0)
        
        await buffer.add(metric1)
        await buffer.add(metric2)
        
        # Get all metrics
        metrics = await buffer.get_all()
        assert len(metrics) == 2
        
        # Buffer should be empty after get_all
        metrics = await buffer.get_all()
        assert len(metrics) == 0
        
        # Test get_latest
        await buffer.add(metric1)
        await buffer.add(metric2)
        
        latest = await buffer.get_latest("test_metric")
        assert latest is not None
        assert latest.value == 43.0
    
    @pytest.mark.asyncio
    async def test_performance_collector(self):
        """Test performance metric collection."""
        portfolio = Portfolio(initial_capital=100000)
        config = CollectorConfig(collection_interval_seconds=0.1)
        collector = PerformanceCollector(config, portfolio)
        
        # Add some portfolio activity
        portfolio.open_position("AAPL", 100, 150.0)
        portfolio.update_equity(105000)  # 5% gain
        portfolio.record_trade(1000, True)
        portfolio.record_trade(-500, False)
        
        # Collect metrics
        metrics = await collector.collect()
        
        # Check metrics
        metric_names = [m.name for m in metrics]
        assert "total_return" in metric_names
        assert "equity" in metric_names
        assert "win_rate" in metric_names
        assert "open_positions" in metric_names
        
        # Check values
        total_return_metric = next(m for m in metrics if m.name == "total_return")
        assert total_return_metric.value == 0.05  # 5% return
    
    @pytest.mark.asyncio
    async def test_position_collector(self):
        """Test position metric collection."""
        portfolio = Portfolio(initial_capital=100000)
        config = CollectorConfig()
        collector = PositionCollector(config, portfolio)
        
        # Add positions
        portfolio.open_position("AAPL", 100, 150.0)
        portfolio.open_position("GOOGL", 50, 2800.0)
        portfolio.open_position("AAPL", 50, 155.0)  # Additional AAPL position
        
        # Collect metrics
        metrics = await collector.collect()
        
        # Check metrics
        exposure_metrics = [m for m in metrics if m.name == "total_exposure"]
        assert len(exposure_metrics) == 1
        
        # Check position concentration
        aapl_concentration = [m for m in metrics 
                             if m.name == "position_concentration" 
                             and m.tags.get("symbol") == "AAPL"]
        assert len(aapl_concentration) == 1
    
    @pytest.mark.asyncio
    async def test_system_health_collector(self):
        """Test system health metric collection."""
        config = CollectorConfig()
        collector = SystemHealthCollector(config)
        
        # Collect metrics
        metrics = await collector.collect()
        
        # Check metrics
        metric_names = [m.name for m in metrics]
        assert "cpu_usage" in metric_names
        assert "memory_usage" in metric_names
        assert "thread_count" in metric_names
        
        # Check values are reasonable
        cpu_metric = next(m for m in metrics if m.name == "cpu_usage")
        assert 0 <= cpu_metric.value <= 100
    
    @pytest.mark.asyncio
    async def test_metric_aggregator(self):
        """Test metric aggregator functionality."""
        portfolio = Portfolio(initial_capital=100000)
        config = CollectorConfig(collection_interval_seconds=0.1)
        
        aggregator = MetricAggregator()
        
        # Add collectors
        perf_collector = PerformanceCollector(config, portfolio)
        system_collector = SystemHealthCollector(config)
        
        aggregator.add_collector(perf_collector)
        aggregator.add_collector(system_collector)
        
        # Start all collectors
        await aggregator.start_all()
        
        # Wait for collection
        await asyncio.sleep(0.2)
        
        # Get latest metrics
        metrics = await aggregator.get_latest_metrics()
        assert len(metrics) > 0
        
        # Stop all collectors
        await aggregator.stop_all()


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_create_drawdown_alert(self):
        """Test drawdown alert creation."""
        alert = create_drawdown_alert(0.15, 0.20)
        
        assert alert.type == AlertType.DRAWDOWN
        assert alert.priority == AlertPriority.MEDIUM
        assert "15.00%" in alert.message
        assert alert.data["current_drawdown"] == 0.15
        
        # Test high priority
        alert = create_drawdown_alert(0.18, 0.20)
        assert alert.priority == AlertPriority.HIGH
    
    def test_create_position_size_alert(self):
        """Test position size alert creation."""
        alert = create_position_size_alert("TSLA", 0.30, 0.25)
        
        assert alert.type == AlertType.POSITION_SIZE
        assert alert.priority == AlertPriority.HIGH
        assert "TSLA" in alert.message
        assert "30.00%" in alert.message
        assert alert.data["symbol"] == "TSLA"
    
    def test_create_signal_alert(self):
        """Test signal alert creation."""
        alert = create_signal_alert("AAPL", "BUY", 0.75)
        
        assert alert.type == AlertType.SIGNAL
        assert alert.priority == AlertPriority.MEDIUM
        assert "AAPL" in alert.message
        assert alert.data["signal_type"] == "BUY"
        
        # Test high priority
        alert = create_signal_alert("AAPL", "SELL", 0.90)
        assert alert.priority == AlertPriority.HIGH


@pytest.mark.integration
class TestMonitoringIntegration:
    """Integration tests for the complete monitoring system."""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_system(self, tmp_path):
        """Test the complete monitoring system integration."""
        # Create config
        config = MonitoringConfig()
        config.collectors.persistence_path = tmp_path / "metrics"
        
        # Initialize components
        portfolio = Portfolio(initial_capital=100000)
        alert_engine = AlertEngine(config.alerts)
        
        aggregator = MetricAggregator()
        aggregator.add_collector(PerformanceCollector(config.collectors, portfolio))
        aggregator.add_collector(PositionCollector(config.collectors, portfolio))
        aggregator.add_collector(SystemHealthCollector(config.collectors))
        
        # Start system
        await alert_engine.start()
        await aggregator.start_all()
        
        # Simulate portfolio activity
        portfolio.open_position("AAPL", 100, 150.0)
        portfolio.record_trade(500, True)
        
        # Wait for metrics
        await asyncio.sleep(1.5)
        
        # Check metrics collected
        metrics = await aggregator.get_latest_metrics()
        assert len(metrics) > 0
        
        # Send alert
        alert = create_drawdown_alert(0.08, 0.10)
        sent = await alert_engine.send_alert(alert)
        assert sent is True
        
        # Stop system
        await aggregator.stop_all()
        await alert_engine.stop()
        
        # Check persistence
        metric_files = list(config.collectors.persistence_path.glob("*.json"))
        assert len(metric_files) > 0