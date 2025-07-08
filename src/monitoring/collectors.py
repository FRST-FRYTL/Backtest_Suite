"""
Performance metric collectors for monitoring system.
"""

import asyncio
import json
import logging
import psutil
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from ..backtesting import Portfolio, Position
from ..utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown
from .config import CollectorConfig


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'name': self.name,
            'value': self.value,
            'tags': self.tags
        }


class MetricBuffer:
    """Thread-safe buffer for metrics."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: Deque[MetricPoint] = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
    
    async def add(self, metric: MetricPoint):
        """Add metric to buffer."""
        async with self._lock:
            self.buffer.append(metric)
    
    async def get_all(self) -> List[MetricPoint]:
        """Get all metrics and clear buffer."""
        async with self._lock:
            metrics = list(self.buffer)
            self.buffer.clear()
            return metrics
    
    async def get_latest(self, name: str) -> Optional[MetricPoint]:
        """Get latest metric by name."""
        async with self._lock:
            for metric in reversed(self.buffer):
                if metric.name == name:
                    return metric
            return None


class MetricCollector(ABC):
    """Base class for metric collectors."""
    
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.buffer = MetricBuffer(config.buffer_size)
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start collecting metrics."""
        self._running = True
        self._task = asyncio.create_task(self._collect_loop())
        self.logger.info(f"{self.__class__.__name__} started")
    
    async def stop(self):
        """Stop collecting metrics."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        # Save remaining metrics
        if self.config.persist_to_disk:
            await self._persist_metrics()
        
        self.logger.info(f"{self.__class__.__name__} stopped")
    
    async def _collect_loop(self):
        """Main collection loop."""
        while self._running:
            try:
                # Collect metrics
                metrics = await self.collect()
                
                # Add to buffer
                for metric in metrics:
                    await self.buffer.add(metric)
                
                # Persist if needed
                if self.config.persist_to_disk and len(self.buffer.buffer) >= self.config.buffer_size * 0.8:
                    await self._persist_metrics()
                
                # Wait for next collection
                await asyncio.sleep(self.config.collection_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
    
    @abstractmethod
    async def collect(self) -> List[MetricPoint]:
        """Collect metrics - must be implemented by subclasses."""
        pass
    
    async def _persist_metrics(self):
        """Persist metrics to disk."""
        metrics = await self.buffer.get_all()
        if not metrics:
            return
        
        # Create persistence path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.config.persistence_path / f"{self.__class__.__name__}_{timestamp}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        data = [metric.to_dict() for metric in metrics]
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        self.logger.debug(f"Persisted {len(metrics)} metrics to {filepath}")


class PerformanceCollector(MetricCollector):
    """Collector for trading performance metrics."""
    
    def __init__(self, config: CollectorConfig, portfolio: Portfolio):
        super().__init__(config)
        self.portfolio = portfolio
        self.initial_capital = portfolio.initial_capital
        self.returns_history: Deque[float] = deque(maxlen=252)  # 1 year of daily returns
    
    async def collect(self) -> List[MetricPoint]:
        """Collect performance metrics."""
        timestamp = time.time()
        metrics = []
        
        # Total return
        total_value = self.portfolio.total_value
        total_return = (total_value - self.initial_capital) / self.initial_capital
        metrics.append(MetricPoint(timestamp, "total_return", total_return))
        
        # Current equity
        metrics.append(MetricPoint(timestamp, "equity", total_value))
        
        # Calculate returns for Sharpe ratio
        if len(self.returns_history) > 0:
            current_return = (total_value - self.returns_history[-1]) / self.returns_history[-1]
            self.returns_history.append(current_return)
            
            if len(self.returns_history) > 20:  # Need some history
                returns = np.array(list(self.returns_history))
                sharpe = calculate_sharpe_ratio(returns)
                metrics.append(MetricPoint(timestamp, "sharpe_ratio", sharpe))
        else:
            self.returns_history.append(total_value)
        
        # Win rate
        if self.portfolio.total_trades > 0:
            win_rate = self.portfolio.winning_trades / self.portfolio.total_trades
            metrics.append(MetricPoint(timestamp, "win_rate", win_rate))
        
        # Profit factor
        if self.portfolio.gross_loss != 0:
            profit_factor = abs(self.portfolio.gross_profit / self.portfolio.gross_loss)
            metrics.append(MetricPoint(timestamp, "profit_factor", profit_factor))
        
        # Drawdown
        equity_curve = self.portfolio.equity_curve
        if len(equity_curve) > 1:
            values = [point[1] for point in equity_curve]
            current_dd, max_dd = self._calculate_drawdown(values)
            metrics.append(MetricPoint(timestamp, "current_drawdown", current_dd))
            metrics.append(MetricPoint(timestamp, "max_drawdown", max_dd))
        
        # Number of open positions
        open_positions = len([p for p in self.portfolio.positions.values() if not p.is_closed])
        metrics.append(MetricPoint(timestamp, "open_positions", float(open_positions)))
        
        return metrics
    
    def _calculate_drawdown(self, values: List[float]) -> Tuple[float, float]:
        """Calculate current and maximum drawdown."""
        if not values:
            return 0.0, 0.0
        
        peak = values[0]
        max_dd = 0.0
        current_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        # Current drawdown is from most recent peak
        current_value = values[-1]
        recent_peak = max(values[-20:]) if len(values) > 20 else max(values)
        current_dd = (recent_peak - current_value) / recent_peak if recent_peak > 0 else 0
        
        return current_dd, max_dd


class PositionCollector(MetricCollector):
    """Collector for position-related metrics."""
    
    def __init__(self, config: CollectorConfig, portfolio: Portfolio):
        super().__init__(config)
        self.portfolio = portfolio
    
    async def collect(self) -> List[MetricPoint]:
        """Collect position metrics."""
        timestamp = time.time()
        metrics = []
        
        total_value = self.portfolio.total_value
        open_positions = [p for p in self.portfolio.positions.values() if not p.is_closed]
        
        # Position count by symbol
        position_counts: Dict[str, int] = {}
        position_values: Dict[str, float] = {}
        position_pnls: Dict[str, float] = {}
        
        for position in open_positions:
            symbol = position.symbol
            position_counts[symbol] = position_counts.get(symbol, 0) + 1
            position_values[symbol] = position_values.get(symbol, 0) + position.market_value
            position_pnls[symbol] = position_pnls.get(symbol, 0) + position.unrealized_pnl
        
        # Total exposure
        total_exposure = sum(position_values.values())
        exposure_pct = total_exposure / total_value if total_value > 0 else 0
        metrics.append(MetricPoint(timestamp, "total_exposure", exposure_pct))
        
        # Position concentration
        for symbol, value in position_values.items():
            concentration = value / total_value if total_value > 0 else 0
            metrics.append(
                MetricPoint(timestamp, "position_concentration", concentration, {"symbol": symbol})
            )
        
        # Unrealized P&L
        total_unrealized_pnl = sum(position_pnls.values())
        metrics.append(MetricPoint(timestamp, "unrealized_pnl", total_unrealized_pnl))
        
        # Per-symbol metrics
        for symbol in position_counts:
            metrics.append(
                MetricPoint(timestamp, "position_count", float(position_counts[symbol]), 
                           {"symbol": symbol})
            )
            metrics.append(
                MetricPoint(timestamp, "position_value", position_values[symbol], 
                           {"symbol": symbol})
            )
            metrics.append(
                MetricPoint(timestamp, "position_pnl", position_pnls[symbol], 
                           {"symbol": symbol})
            )
        
        # Long vs short exposure
        long_exposure = sum(v for p, v in position_values.items() 
                           if any(pos.quantity > 0 for pos in open_positions if pos.symbol == p))
        short_exposure = sum(v for p, v in position_values.items()
                            if any(pos.quantity < 0 for pos in open_positions if pos.symbol == p))
        
        metrics.append(MetricPoint(timestamp, "long_exposure", long_exposure / total_value if total_value > 0 else 0))
        metrics.append(MetricPoint(timestamp, "short_exposure", short_exposure / total_value if total_value > 0 else 0))
        
        return metrics


class SystemHealthCollector(MetricCollector):
    """Collector for system health metrics."""
    
    def __init__(self, config: CollectorConfig):
        super().__init__(config)
        self.process = psutil.Process()
        self.last_tick_time = time.time()
        self.tick_count = 0
    
    async def collect(self) -> List[MetricPoint]:
        """Collect system health metrics."""
        timestamp = time.time()
        metrics = []
        
        # CPU usage
        cpu_percent = self.process.cpu_percent(interval=0.1)
        metrics.append(MetricPoint(timestamp, "cpu_usage", cpu_percent))
        
        # Memory usage
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        metrics.append(MetricPoint(timestamp, "memory_usage", memory_percent))
        metrics.append(MetricPoint(timestamp, "memory_rss", float(memory_info.rss)))
        
        # Tick rate (processing speed)
        current_time = time.time()
        time_diff = current_time - self.last_tick_time
        if time_diff > 0:
            tick_rate = self.tick_count / time_diff
            metrics.append(MetricPoint(timestamp, "tick_rate", tick_rate))
            
            # Reset counters every 10 seconds
            if time_diff > 10:
                self.last_tick_time = current_time
                self.tick_count = 0
        
        self.tick_count += 1
        
        # Thread count
        num_threads = self.process.num_threads()
        metrics.append(MetricPoint(timestamp, "thread_count", float(num_threads)))
        
        # Open file descriptors (Linux/Mac)
        try:
            num_fds = self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
            metrics.append(MetricPoint(timestamp, "open_fds", float(num_fds)))
        except Exception:
            pass
        
        # Network connections
        try:
            connections = self.process.connections()
            metrics.append(MetricPoint(timestamp, "network_connections", float(len(connections))))
        except Exception:
            pass
        
        return metrics
    
    def record_tick(self):
        """Record a processing tick for rate calculation."""
        self.tick_count += 1


class MetricAggregator:
    """Aggregates metrics from multiple collectors."""
    
    def __init__(self):
        self.collectors: List[MetricCollector] = []
        self.logger = logging.getLogger(__name__)
    
    def add_collector(self, collector: MetricCollector):
        """Add a collector to the aggregator."""
        self.collectors.append(collector)
    
    async def start_all(self):
        """Start all collectors."""
        for collector in self.collectors:
            await collector.start()
    
    async def stop_all(self):
        """Stop all collectors."""
        for collector in self.collectors:
            await collector.stop()
    
    async def get_latest_metrics(self) -> Dict[str, MetricPoint]:
        """Get latest metrics from all collectors."""
        all_metrics = {}
        
        for collector in self.collectors:
            metrics = await collector.buffer.get_all()
            for metric in metrics:
                key = f"{metric.name}:{','.join(f'{k}={v}' for k, v in metric.tags.items())}"
                all_metrics[key] = metric
        
        return all_metrics
    
    async def get_metric_history(self, metric_name: str, duration_seconds: int = 3600) -> List[MetricPoint]:
        """Get metric history for specified duration."""
        cutoff_time = time.time() - duration_seconds
        history = []
        
        for collector in self.collectors:
            # This would need to be implemented with persistent storage
            # For now, we return what's in the buffer
            metrics = await collector.buffer.get_all()
            history.extend([m for m in metrics if m.name == metric_name and m.timestamp > cutoff_time])
        
        return sorted(history, key=lambda m: m.timestamp)