"""
Real-time monitoring dashboard for backtesting.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import dash
    import plotly.graph_objs as go
    from dash import dcc, html, Input, Output, State
    from dash.exceptions import PreventUpdate
    from flask import Flask
    from flask_cors import CORS
    import websockets
    import websockets.server
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    dash = None
    go = None
    dcc = html = Input = Output = State = None
    PreventUpdate = None
    Flask = None
    CORS = None
    websockets = None
    
import pandas as pd

from .alerts import Alert, AlertEngine, AlertType
from .collectors import MetricAggregator, MetricPoint
from .config import DashboardConfig


class MonitoringDashboard:
    """Real-time monitoring dashboard using Dash and WebSockets."""
    
    def __init__(self, config: DashboardConfig, alert_engine: AlertEngine, 
                 metric_aggregator: MetricAggregator):
        self.config = config
        self.alert_engine = alert_engine
        self.metric_aggregator = metric_aggregator
        self.logger = logging.getLogger(__name__)
        
        if not DASH_AVAILABLE:
            self.logger.warning("Dash dependencies not available. Dashboard will be disabled.")
            return
        
        # Initialize Flask server
        self.server = Flask(__name__)
        CORS(self.server)
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            server=self.server,
            url_base_pathname='/',
            suppress_callback_exceptions=True
        )
        
        # WebSocket clients
        self.ws_clients: List[websockets.WebSocketServerProtocol] = []
        self.ws_server = None
        
        # Data storage
        self.metrics_history: Dict[str, List[MetricPoint]] = {}
        self.max_history_points = config.max_chart_points
        
        # Setup layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Backtest Suite Monitoring Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.Hr()
            ]),
            
            # Main container
            html.Div([
                # Performance metrics row
                html.Div([
                    # Total Return Card
                    html.Div([
                        html.H3("Total Return", style={'textAlign': 'center'}),
                        html.H2(id='total-return-value', children='0.00%',
                               style={'textAlign': 'center', 'color': '#27ae60'})
                    ], className='metric-card', style={'width': '24%', 'display': 'inline-block'}),
                    
                    # Sharpe Ratio Card
                    html.Div([
                        html.H3("Sharpe Ratio", style={'textAlign': 'center'}),
                        html.H2(id='sharpe-ratio-value', children='0.00',
                               style={'textAlign': 'center', 'color': '#3498db'})
                    ], className='metric-card', style={'width': '24%', 'display': 'inline-block'}),
                    
                    # Win Rate Card
                    html.Div([
                        html.H3("Win Rate", style={'textAlign': 'center'}),
                        html.H2(id='win-rate-value', children='0.00%',
                               style={'textAlign': 'center', 'color': '#9b59b6'})
                    ], className='metric-card', style={'width': '24%', 'display': 'inline-block'}),
                    
                    # Max Drawdown Card
                    html.Div([
                        html.H3("Max Drawdown", style={'textAlign': 'center'}),
                        html.H2(id='max-drawdown-value', children='0.00%',
                               style={'textAlign': 'center', 'color': '#e74c3c'})
                    ], className='metric-card', style={'width': '24%', 'display': 'inline-block'}),
                ], style={'marginBottom': '20px'}),
                
                # Charts row
                html.Div([
                    # Equity curve chart
                    html.Div([
                        dcc.Graph(id='equity-curve-chart')
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    # Position exposure chart
                    html.Div([
                        dcc.Graph(id='position-exposure-chart')
                    ], style={'width': '50%', 'display': 'inline-block'}),
                ]),
                
                # Alerts and positions row
                html.Div([
                    # Recent alerts
                    html.Div([
                        html.H3("Recent Alerts"),
                        html.Div(id='alerts-container', style={'height': '300px', 'overflowY': 'scroll'})
                    ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                    
                    # Open positions
                    html.Div([
                        html.H3("Open Positions"),
                        html.Div(id='positions-container', style={'height': '300px', 'overflowY': 'scroll'})
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ]),
                
                # System health row
                html.Div([
                    html.H3("System Health"),
                    html.Div([
                        # CPU usage
                        html.Div([
                            html.P("CPU Usage"),
                            html.Progress(id='cpu-usage-bar', max='100', value='0'),
                            html.Span(id='cpu-usage-text', children='0%')
                        ], style={'width': '24%', 'display': 'inline-block'}),
                        
                        # Memory usage
                        html.Div([
                            html.P("Memory Usage"),
                            html.Progress(id='memory-usage-bar', max='100', value='0'),
                            html.Span(id='memory-usage-text', children='0%')
                        ], style={'width': '24%', 'display': 'inline-block'}),
                        
                        # Tick rate
                        html.Div([
                            html.P("Tick Rate"),
                            html.Span(id='tick-rate-text', children='0 ticks/s')
                        ], style={'width': '24%', 'display': 'inline-block'}),
                        
                        # Connection status
                        html.Div([
                            html.P("WebSocket Status"),
                            html.Span(id='ws-status', children='Disconnected',
                                     style={'color': '#e74c3c'})
                        ], style={'width': '24%', 'display': 'inline-block'}),
                    ])
                ], style={'marginTop': '20px'}),
            ], style={'padding': '20px'}),
            
            # Update interval
            dcc.Interval(
                id='interval-component',
                interval=self.config.update_interval_ms,
                n_intervals=0
            ),
            
            # Hidden div to store metrics data
            html.Div(id='metrics-store', style={'display': 'none'})
        ])
    
    def _setup_callbacks(self):
        """Setup Dash callbacks for real-time updates."""
        
        @self.app.callback(
            [Output('metrics-store', 'children'),
             Output('total-return-value', 'children'),
             Output('total-return-value', 'style'),
             Output('sharpe-ratio-value', 'children'),
             Output('win-rate-value', 'children'),
             Output('max-drawdown-value', 'children'),
             Output('cpu-usage-bar', 'value'),
             Output('cpu-usage-text', 'children'),
             Output('memory-usage-bar', 'value'),
             Output('memory-usage-text', 'children'),
             Output('tick-rate-text', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(n):
            """Update metrics display."""
            # Get latest metrics
            metrics = asyncio.run(self.metric_aggregator.get_latest_metrics())
            
            # Extract values
            total_return = self._get_metric_value(metrics, 'total_return', 0.0)
            sharpe_ratio = self._get_metric_value(metrics, 'sharpe_ratio', 0.0)
            win_rate = self._get_metric_value(metrics, 'win_rate', 0.0)
            max_drawdown = self._get_metric_value(metrics, 'max_drawdown', 0.0)
            cpu_usage = self._get_metric_value(metrics, 'cpu_usage', 0.0)
            memory_usage = self._get_metric_value(metrics, 'memory_usage', 0.0)
            tick_rate = self._get_metric_value(metrics, 'tick_rate', 0.0)
            
            # Format values
            return_color = '#27ae60' if total_return >= 0 else '#e74c3c'
            
            return (
                json.dumps({k: v.to_dict() for k, v in metrics.items()}),
                f"{total_return*100:.2f}%",
                {'textAlign': 'center', 'color': return_color},
                f"{sharpe_ratio:.2f}",
                f"{win_rate*100:.2f}%",
                f"{max_drawdown*100:.2f}%",
                str(int(cpu_usage)),
                f"{cpu_usage:.1f}%",
                str(int(memory_usage)),
                f"{memory_usage:.1f}%",
                f"{tick_rate:.1f} ticks/s"
            )
        
        @self.app.callback(
            Output('equity-curve-chart', 'figure'),
            [Input('metrics-store', 'children')]
        )
        def update_equity_curve(metrics_json):
            """Update equity curve chart."""
            if not metrics_json:
                raise PreventUpdate
            
            # Parse metrics
            metrics = json.loads(metrics_json)
            
            # Get equity history
            equity_points = [
                MetricPoint(**m) for k, m in metrics.items() 
                if 'equity' in k
            ]
            
            if not equity_points:
                raise PreventUpdate
            
            # Sort by timestamp
            equity_points.sort(key=lambda p: p.timestamp)
            
            # Create figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[datetime.fromtimestamp(p.timestamp) for p in equity_points],
                y=[p.value for p in equity_points],
                mode='lines',
                name='Equity',
                line=dict(color='#3498db', width=2)
            ))
            
            fig.update_layout(
                title='Equity Curve',
                xaxis_title='Time',
                yaxis_title='Value',
                template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white',
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output('position-exposure-chart', 'figure'),
            [Input('metrics-store', 'children')]
        )
        def update_position_exposure(metrics_json):
            """Update position exposure chart."""
            if not metrics_json:
                raise PreventUpdate
            
            # Parse metrics
            metrics = json.loads(metrics_json)
            
            # Get position data
            position_data = {}
            for k, m in metrics.items():
                if 'position_concentration' in k:
                    metric = MetricPoint(**m)
                    symbol = metric.tags.get('symbol', 'Unknown')
                    position_data[symbol] = metric.value
            
            if not position_data:
                raise PreventUpdate
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(position_data.keys()),
                values=[abs(v) for v in position_data.values()],
                hole=0.3
            )])
            
            fig.update_layout(
                title='Position Exposure',
                template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white',
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output('alerts-container', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_alerts(n):
            """Update alerts display."""
            alerts = self.alert_engine.get_recent_alerts(limit=10)
            
            if not alerts:
                return html.P("No recent alerts", style={'color': '#95a5a6'})
            
            alert_elements = []
            for alert in reversed(alerts):  # Show newest first
                color = {
                    1: '#95a5a6',  # LOW
                    2: '#3498db',  # MEDIUM  
                    3: '#f39c12',  # HIGH
                    4: '#e74c3c'   # CRITICAL
                }.get(alert.priority.value, '#95a5a6')
                
                alert_elements.append(html.Div([
                    html.Span(f"[{alert.priority.name}] ", style={'color': color, 'fontWeight': 'bold'}),
                    html.Span(alert.title),
                    html.Br(),
                    html.Small(alert.message, style={'color': '#7f8c8d'}),
                    html.Hr(style={'margin': '5px 0'})
                ], style={'marginBottom': '10px'}))
            
            return alert_elements
        
        @self.app.callback(
            Output('positions-container', 'children'),
            [Input('metrics-store', 'children')]
        )
        def update_positions(metrics_json):
            """Update positions display."""
            if not metrics_json:
                raise PreventUpdate
            
            # Parse metrics
            metrics = json.loads(metrics_json)
            
            # Get position data
            positions = []
            symbols = set()
            
            for k, m in metrics.items():
                if 'position_count' in k:
                    metric = MetricPoint(**m)
                    symbol = metric.tags.get('symbol')
                    if symbol:
                        symbols.add(symbol)
            
            for symbol in symbols:
                count = self._get_metric_value_by_tag(metrics, 'position_count', 'symbol', symbol, 0)
                value = self._get_metric_value_by_tag(metrics, 'position_value', 'symbol', symbol, 0.0)
                pnl = self._get_metric_value_by_tag(metrics, 'position_pnl', 'symbol', symbol, 0.0)
                
                if count > 0:
                    positions.append({
                        'symbol': symbol,
                        'count': int(count),
                        'value': value,
                        'pnl': pnl
                    })
            
            if not positions:
                return html.P("No open positions", style={'color': '#95a5a6'})
            
            # Create table
            return html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Symbol"),
                        html.Th("Count"),
                        html.Th("Value"),
                        html.Th("P&L")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(pos['symbol']),
                        html.Td(str(pos['count'])),
                        html.Td(f"${pos['value']:,.2f}"),
                        html.Td(
                            f"${pos['pnl']:,.2f}",
                            style={'color': '#27ae60' if pos['pnl'] >= 0 else '#e74c3c'}
                        )
                    ]) for pos in sorted(positions, key=lambda p: abs(p['value']), reverse=True)
                ])
            ], style={'width': '100%'})
    
    def _get_metric_value(self, metrics: Dict[str, Any], name: str, default: float) -> float:
        """Get metric value by name."""
        for k, v in metrics.items():
            if name in k and ':' not in k:  # No tags
                return v.value
        return default
    
    def _get_metric_value_by_tag(self, metrics: Dict[str, Any], name: str, 
                                tag_name: str, tag_value: str, default: float) -> float:
        """Get metric value by name and tag."""
        for k, v in metrics.items():
            if name in k and f"{tag_name}={tag_value}" in k:
                return v.value
        return default
    
    async def start(self):
        """Start the dashboard."""
        if not self.config.enabled:
            self.logger.info("Dashboard is disabled")
            return
            
        if not DASH_AVAILABLE:
            self.logger.warning("Dashboard dependencies not available. Skipping dashboard startup.")
            return
        
        # Start WebSocket server if enabled
        if self.config.websocket_enabled:
            self.ws_server = await websockets.serve(
                self._handle_websocket,
                self.config.host,
                self.config.websocket_port
            )
            self.logger.info(f"WebSocket server started on ws://{self.config.host}:{self.config.websocket_port}")
        
        # Run Dash app in separate thread
        import threading
        dash_thread = threading.Thread(
            target=self.app.run_server,
            kwargs={
                'host': self.config.host,
                'port': self.config.port,
                'debug': False,
                'use_reloader': False
            }
        )
        dash_thread.daemon = True
        dash_thread.start()
        
        self.logger.info(f"Dashboard started on http://{self.config.host}:{self.config.port}")
    
    async def stop(self):
        """Stop the dashboard."""
        if self.ws_server:
            self.ws_server.close()
            await self.ws_server.wait_closed()
        
        self.logger.info("Dashboard stopped")
    
    async def _handle_websocket(self, websocket, path):
        """Handle WebSocket connections."""
        self.ws_clients.append(websocket)
        self.logger.info(f"WebSocket client connected: {websocket.remote_address}")
        
        try:
            # Send initial data
            metrics = await self.metric_aggregator.get_latest_metrics()
            await websocket.send(json.dumps({
                'type': 'metrics',
                'data': {k: v.to_dict() for k, v in metrics.items()}
            }))
            
            # Keep connection alive
            await websocket.wait_closed()
            
        finally:
            self.ws_clients.remove(websocket)
            self.logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
    
    async def broadcast_metrics(self, metrics: Dict[str, MetricPoint]):
        """Broadcast metrics to all WebSocket clients."""
        if not self.ws_clients:
            return
        
        message = json.dumps({
            'type': 'metrics',
            'data': {k: v.to_dict() for k, v in metrics.items()}
        })
        
        # Send to all connected clients
        disconnected = []
        for client in self.ws_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.ws_clients.remove(client)
    
    async def broadcast_alert(self, alert: Alert):
        """Broadcast alert to all WebSocket clients."""
        if not self.ws_clients:
            return
        
        message = json.dumps({
            'type': 'alert',
            'data': alert.to_dict()
        })
        
        # Send to all connected clients
        disconnected = []
        for client in self.ws_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.ws_clients.remove(client)