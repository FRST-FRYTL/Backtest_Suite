"""Interactive dashboard for backtesting results."""

import os
from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly import offline


class Dashboard:
    """Create interactive HTML dashboard for backtest results."""
    
    def __init__(self):
        """Initialize dashboard."""
        self.figures = []
        
    def create_dashboard(
        self,
        results: Dict,
        output_path: str = "backtest_dashboard.html",
        title: str = "Backtest Results Dashboard"
    ) -> str:
        """
        Create complete dashboard from backtest results.
        
        Args:
            results: Backtest results dictionary
            output_path: Path to save HTML file
            title: Dashboard title
            
        Returns:
            Path to created HTML file
        """
        # Clear previous figures
        self.figures = []
        
        # Extract data
        equity_curve = results.get('equity_curve', pd.DataFrame())
        trades = results.get('trades', pd.DataFrame())
        performance = results.get('performance', {})
        
        # Create figures
        if not equity_curve.empty:
            self.figures.append(self._create_equity_chart(equity_curve))
            self.figures.append(self._create_drawdown_chart(equity_curve))
            
        if not trades.empty:
            self.figures.append(self._create_trades_chart(trades))
            self.figures.append(self._create_trade_analysis(trades))
            
        if performance:
            self.figures.append(self._create_metrics_table(performance))
            self.figures.append(self._create_metrics_gauges(performance))
            
        # Generate HTML
        html = self._generate_html(title)
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(html)
            
        return output_path
        
    def _create_equity_chart(self, equity_curve: pd.DataFrame) -> go.Figure:
        """Create equity curve chart."""
        fig = go.Figure()
        
        # Portfolio value
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve['total_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        # Cash
        if 'cash' in equity_curve.columns:
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve['cash'],
                mode='lines',
                name='Cash',
                line=dict(color='green', width=1, dash='dash')
            ))
            
        # Positions value
        if 'positions_value' in equity_curve.columns:
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve['positions_value'],
                mode='lines',
                name='Positions Value',
                line=dict(color='orange', width=1, dash='dash')
            ))
            
        fig.update_layout(
            title='Portfolio Equity Curve',
            xaxis_title='Date',
            yaxis_title='Value ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
        
    def _create_drawdown_chart(self, equity_curve: pd.DataFrame) -> go.Figure:
        """Create drawdown chart."""
        # Calculate drawdown
        running_max = equity_curve['total_value'].expanding().max()
        drawdown = (equity_curve['total_value'] - running_max) / running_max * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red', width=1)
        ))
        
        # Mark maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        
        fig.add_trace(go.Scatter(
            x=[max_dd_idx],
            y=[max_dd_value],
            mode='markers+text',
            name='Max Drawdown',
            marker=dict(size=10, color='darkred'),
            text=[f'{max_dd_value:.2f}%'],
            textposition='top center'
        ))
        
        fig.update_layout(
            title='Portfolio Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_white'
        )
        
        return fig
        
    def _create_trades_chart(self, trades: pd.DataFrame) -> go.Figure:
        """Create trades timeline chart."""
        fig = go.Figure()
        
        # Filter for actual trades
        trades_only = trades[trades['type'].isin(['OPEN', 'CLOSE'])]
        
        if trades_only.empty:
            return fig
            
        # Group by symbol
        for symbol in trades_only['symbol'].unique():
            symbol_trades = trades_only[trades_only['symbol'] == symbol]
            
            # Buy trades
            buys = symbol_trades[symbol_trades['quantity'] > 0]
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys['timestamp'],
                    y=buys['price'],
                    mode='markers',
                    name=f'{symbol} Buy',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color='green'
                    )
                ))
                
            # Sell trades
            sells = symbol_trades[symbol_trades['quantity'] < 0]
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells['timestamp'],
                    y=sells['price'],
                    mode='markers',
                    name=f'{symbol} Sell',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color='red'
                    )
                ))
                
        fig.update_layout(
            title='Trade Timeline',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
        
    def _create_trade_analysis(self, trades: pd.DataFrame) -> go.Figure:
        """Create trade analysis charts."""
        # Filter for closed positions
        closed_trades = trades[
            (trades['type'] == 'CLOSE') & 
            (trades['position_pnl'].notna())
        ]
        
        if closed_trades.empty:
            return go.Figure()
            
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'P&L Distribution',
                'P&L by Trade',
                'Win/Loss Ratio',
                'Trade Duration'
            )
        )
        
        # P&L Distribution
        fig.add_trace(
            go.Histogram(
                x=closed_trades['position_pnl'],
                nbinsx=20,
                marker_color='blue',
                name='P&L'
            ),
            row=1, col=1
        )
        
        # P&L by trade (waterfall)
        cumulative_pnl = closed_trades['position_pnl'].cumsum()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(closed_trades))),
                y=cumulative_pnl,
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='green', width=2)
            ),
            row=1, col=2
        )
        
        # Win/Loss pie chart
        wins = (closed_trades['position_pnl'] > 0).sum()
        losses = (closed_trades['position_pnl'] <= 0).sum()
        
        fig.add_trace(
            go.Pie(
                labels=['Wins', 'Losses'],
                values=[wins, losses],
                marker_colors=['green', 'red'],
                name='Win/Loss'
            ),
            row=2, col=1
        )
        
        # Trade duration histogram (if available)
        # This is a placeholder - actual duration would need to be calculated
        
        fig.update_layout(
            title='Trade Analysis',
            showlegend=False,
            template='plotly_white',
            height=800
        )
        
        return fig
        
    def _create_metrics_table(self, performance: Dict) -> go.Figure:
        """Create performance metrics table."""
        # Prepare data for table
        metrics_data = []
        
        # Group metrics
        groups = {
            'Returns': ['total_return', 'total_pnl', 'unrealized_pnl', 'realized_pnl'],
            'Risk': ['max_drawdown', 'total_commission'],
            'Trade Stats': ['total_trades', 'winning_trades', 'losing_trades', 
                          'avg_win', 'avg_loss'],
            'Ratios': ['sharpe_ratio']
        }
        
        for group, metrics in groups.items():
            for metric in metrics:
                if metric in performance:
                    value = performance[metric]
                    # Format value
                    if isinstance(value, float):
                        if 'return' in metric or 'pnl' in metric:
                            formatted = f"${value:,.2f}"
                        elif 'ratio' in metric:
                            formatted = f"{value:.2f}"
                        else:
                            formatted = f"{value:,.0f}"
                    else:
                        formatted = str(value)
                        
                    metrics_data.append({
                        'Group': group,
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': formatted
                    })
                    
        df = pd.DataFrame(metrics_data)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df.columns),
                fill_color='lightblue',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[df[col] for col in df.columns],
                fill_color=[['white', 'lightgray'] * (len(df) // 2 + 1)],
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title='Performance Metrics',
            template='plotly_white',
            height=600
        )
        
        return fig
        
    def _create_metrics_gauges(self, performance: Dict) -> go.Figure:
        """Create gauge charts for key metrics."""
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Win Rate', 'Sharpe Ratio', 'Return'),
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
        )
        
        # Win rate gauge
        win_rate = 0
        if performance.get('total_trades', 0) > 0:
            win_rate = performance.get('winning_trades', 0) / performance['total_trades'] * 100
            
        fig.add_trace(
            go.Indicator(
                mode='gauge+number',
                value=win_rate,
                title={'text': 'Win Rate %'},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': 'green' if win_rate > 50 else 'red'},
                    'threshold': {
                        'line': {'color': 'black', 'width': 2},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ),
            row=1, col=1
        )
        
        # Sharpe ratio gauge
        sharpe = performance.get('sharpe_ratio', 0)
        fig.add_trace(
            go.Indicator(
                mode='gauge+number',
                value=sharpe,
                title={'text': 'Sharpe Ratio'},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [-2, 3]},
                    'bar': {'color': 'green' if sharpe > 1 else 'orange' if sharpe > 0 else 'red'},
                    'threshold': {
                        'line': {'color': 'black', 'width': 2},
                        'thickness': 0.75,
                        'value': 1
                    }
                }
            ),
            row=1, col=2
        )
        
        # Total return gauge
        total_return = performance.get('total_return', 0) / performance.get('initial_capital', 1) * 100
        fig.add_trace(
            go.Indicator(
                mode='gauge+number',
                value=total_return,
                title={'text': 'Total Return %'},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [-50, 100]},
                    'bar': {'color': 'green' if total_return > 0 else 'red'},
                    'threshold': {
                        'line': {'color': 'black', 'width': 2},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            title='Key Performance Indicators',
            template='plotly_white',
            height=300
        )
        
        return fig
        
    def _generate_html(self, title: str) -> str:
        """Generate HTML with all figures."""
        # Create HTML content
        html_parts = [
            f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background-color: #f5f5f5;
                    }}
                    h1 {{
                        color: #333;
                        text-align: center;
                    }}
                    .chart-container {{
                        background-color: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        margin: 20px 0;
                        padding: 20px;
                    }}
                </style>
            </head>
            <body>
                <h1>{title}</h1>
            """
        ]
        
        # Add each figure
        for i, fig in enumerate(self.figures):
            div_id = f'chart_{i}'
            html_parts.append(f'<div class="chart-container"><div id="{div_id}"></div></div>')
            html_parts.append(offline.plot(
                fig,
                output_type='div',
                include_plotlyjs=False,
                div_id=div_id
            ))
            
        html_parts.append("""
            </body>
            </html>
        """)
        
        return '\n'.join(html_parts)