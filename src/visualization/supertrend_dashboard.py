"""
SuperTrend AI Strategy Dashboard Generator

Creates interactive Plotly dashboards for SuperTrend AI strategy performance visualization.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class SuperTrendDashboard:
    """Generate interactive dashboards for SuperTrend AI strategy"""
    
    def __init__(self, output_dir: str = "reports/dashboards"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_performance_dashboard(
        self,
        results: Dict,
        title: str = "SuperTrend AI Strategy Performance"
    ) -> str:
        """Create comprehensive performance dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Cumulative Returns', 'Drawdown Analysis',
                'Signal Distribution', 'Trade Analysis',
                'Parameter Performance', 'Cluster Analysis',
                'Risk Metrics', 'Performance Metrics'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": True}],
                [{"type": "scatter3d", "colspan": 2}, None],
                [{"type": "table", "colspan": 2}, None]
            ],
            row_heights=[0.3, 0.3, 0.3, 0.1],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Color scheme
        colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8'
        }
        
        # Process results for each symbol
        row_col_pairs = [(1,1), (1,2), (2,1), (2,2)]
        
        for idx, (symbol, result) in enumerate(results.items()):
            if 'error' in result or idx >= 4:
                continue
                
            # Extract data
            portfolio = result.get('portfolio_values', pd.DataFrame())
            signals = result.get('signals', pd.DataFrame())
            
            if portfolio.empty:
                continue
            
            # 1. Cumulative Returns
            fig.add_trace(
                go.Scatter(
                    x=portfolio.index,
                    y=(portfolio['value'] / 100000 - 1) * 100,
                    name=f'{symbol} Returns',
                    mode='lines',
                    line=dict(width=2),
                    legendgroup=symbol
                ),
                row=1, col=1
            )
            
            # 2. Drawdown
            returns = portfolio['returns'].fillna(0)
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            
            fig.add_trace(
                go.Scatter(
                    x=portfolio.index,
                    y=drawdown,
                    name=f'{symbol} Drawdown',
                    mode='lines',
                    fill='tozeroy',
                    line=dict(width=1),
                    legendgroup=symbol
                ),
                row=1, col=2
            )
            
            # 3. Signal Strength Distribution
            if 'signal_strength' in signals.columns:
                signal_data = signals[signals['signal'] != 0]['signal_strength'].dropna()
                if len(signal_data) > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=signal_data,
                            name=f'{symbol} Signals',
                            nbinsx=10,
                            opacity=0.7,
                            legendgroup=symbol
                        ),
                        row=2, col=1
                    )
            
            # 4. Trade Analysis
            if 'trades' in result and isinstance(result['trades'], pd.DataFrame) and len(result['trades']) > 0:
                trades = result['trades']
                # Monthly trade count
                trades['month'] = pd.to_datetime(trades['date']).dt.to_period('M')
                monthly_trades = trades.groupby('month').size()
                
                fig.add_trace(
                    go.Bar(
                        x=monthly_trades.index.astype(str),
                        y=monthly_trades.values,
                        name=f'{symbol} Trades',
                        legendgroup=symbol
                    ),
                    row=2, col=2
                )
        
        # 5. 3D Parameter Performance Plot
        if any('optimization' in r for r in results.values()):
            # Collect optimization data
            opt_data = []
            for symbol, result in results.items():
                if 'optimization' in result and 'optimization_history' in result['optimization']:
                    for opt in result['optimization']['optimization_history']:
                        opt_data.append({
                            'atr_length': opt['atr_length'],
                            'min_factor': opt['min_factor'],
                            'sharpe_ratio': opt['sharpe_ratio']
                        })
            
            if opt_data:
                opt_df = pd.DataFrame(opt_data)
                fig.add_trace(
                    go.Scatter3d(
                        x=opt_df['atr_length'],
                        y=opt_df['min_factor'],
                        z=opt_df['sharpe_ratio'],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=opt_df['sharpe_ratio'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Sharpe Ratio")
                        ),
                        name='Parameter Space'
                    ),
                    row=3, col=1
                )
        
        # 6. Performance Metrics Table
        metrics_data = []
        for symbol, result in results.items():
            if 'error' not in result:
                metrics_data.append([
                    symbol,
                    f"{result.get('total_return', 0):.2f}%",
                    f"{result.get('sharpe_ratio', 0):.2f}",
                    f"{result.get('max_drawdown', 0):.2f}%",
                    f"{result.get('win_rate', 0):.1f}%",
                    result.get('num_trades', 0),
                    f"{result.get('profit_factor', 0):.2f}"
                ])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Symbol', 'Return', 'Sharpe', 'Max DD', 'Win Rate', 'Trades', 'Profit Factor'],
                    fill_color=colors['primary'],
                    font=dict(color='white', size=12),
                    align='center'
                ),
                cells=dict(
                    values=list(zip(*metrics_data)) if metrics_data else [[]]*7,
                    fill_color='white',
                    align='center',
                    font=dict(size=11)
                )
            ),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            showlegend=True,
            height=1400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Return %", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown %", row=1, col=2)
        fig.update_xaxes(title_text="Signal Strength", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Month", row=2, col=2)
        fig.update_yaxes(title_text="Trade Count", row=2, col=2)
        
        # Save dashboard
        filename = f"supertrend_ai_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename
        
        # Add custom CSS and JavaScript
        html_template = """
        <html>
        <head>
            <title>SuperTrend AI Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .dashboard-header {
                    background-color: #1f77b4;
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    text-align: center;
                }
                .metric-card {
                    background-color: white;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    display: inline-block;
                    min-width: 150px;
                    text-align: center;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #1f77b4;
                }
                .metric-label {
                    font-size: 14px;
                    color: #666;
                    margin-top: 5px;
                }
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <h1>SuperTrend AI Strategy Performance Dashboard</h1>
                <p>Advanced Trading Strategy with Machine Learning Optimization</p>
            </div>
            {plot_div}
        </body>
        </html>
        """
        
        # Write HTML file
        with open(filepath, 'w') as f:
            f.write(html_template.format(plot_div=fig.to_html(include_plotlyjs=False)))
        
        return str(filepath)
    
    def create_signal_analysis_dashboard(
        self,
        signals_data: pd.DataFrame,
        symbol: str
    ) -> str:
        """Create detailed signal analysis dashboard"""
        
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Price with SuperTrend', 'Signal Strength Over Time',
                'Cluster Distribution', 'Factor Evolution',
                'Performance Index', 'Trade Signals'
            ),
            specs=[
                [{"secondary_y": True}, {"secondary_y": False}],
                [{"type": "pie"}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "scatter"}]
            ],
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # 1. Price with SuperTrend
        fig.add_trace(
            go.Candlestick(
                x=signals_data.index,
                open=signals_data['Open'],
                high=signals_data['High'],
                low=signals_data['Low'],
                close=signals_data['Close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Add SuperTrend line
        if 'supertrend' in signals_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=signals_data.index,
                    y=signals_data['supertrend'],
                    name='SuperTrend',
                    mode='lines',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # Add volume
        fig.add_trace(
            go.Bar(
                x=signals_data.index,
                y=signals_data['Volume'],
                name='Volume',
                marker_color='lightgray',
                opacity=0.3,
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # 2. Signal Strength
        if 'signal_strength' in signals_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=signals_data.index,
                    y=signals_data['signal_strength'],
                    name='Signal Strength',
                    mode='lines',
                    line=dict(color='orange', width=2),
                    fill='tozeroy'
                ),
                row=1, col=2
            )
            
            # Add threshold line
            fig.add_hline(
                y=4, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Min Threshold",
                row=1, col=2
            )
        
        # 3. Cluster Distribution (if available)
        cluster_data = {
            'Best': 40,
            'Average': 35,
            'Worst': 25
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(cluster_data.keys()),
                values=list(cluster_data.values()),
                name='Cluster Distribution',
                hole=0.3
            ),
            row=2, col=1
        )
        
        # 4. Factor Evolution
        if 'selected_factor' in signals_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=signals_data.index,
                    y=signals_data['selected_factor'],
                    name='Selected Factor',
                    mode='lines+markers',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=2
            )
        
        # 5. Performance Index
        if 'performance_index' in signals_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=signals_data.index,
                    y=signals_data['performance_index'],
                    name='Performance Index',
                    mode='lines',
                    line=dict(color='green', width=2),
                    fill='tozeroy'
                ),
                row=3, col=1
            )
        
        # 6. Trade Signals
        buy_signals = signals_data[signals_data['signal'] > 0]
        sell_signals = signals_data[signals_data['signal'] < 0]
        
        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['Close'],
                    mode='markers',
                    name='Buy Signals',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green'
                    )
                ),
                row=3, col=2
            )
        
        if len(sell_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['Close'],
                    mode='markers',
                    name='Sell Signals',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red'
                    )
                ),
                row=3, col=2
            )
        
        # Price line for reference
        fig.add_trace(
            go.Scatter(
                x=signals_data.index,
                y=signals_data['Close'],
                mode='lines',
                name='Price',
                line=dict(color='lightgray', width=1)
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'SuperTrend AI Signal Analysis - {symbol}',
            showlegend=True,
            height=1200,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=2)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Strength (0-10)", row=1, col=2)
        fig.update_yaxes(title_text="Factor", row=2, col=2)
        fig.update_yaxes(title_text="Index", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=3, col=2)
        
        # Save dashboard
        filename = f"signal_analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename
        fig.write_html(filepath)
        
        return str(filepath)
    
    def create_optimization_heatmap(
        self,
        optimization_results: List[Dict],
        metric: str = 'sharpe_ratio'
    ) -> str:
        """Create parameter optimization heatmap"""
        
        # Convert to DataFrame
        df = pd.DataFrame(optimization_results)
        
        # Create pivot table for heatmap
        pivot = df.pivot_table(
            values=metric,
            index='atr_length',
            columns='min_factor',
            aggfunc='mean'
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(pivot.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title=metric.replace('_', ' ').title())
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Parameter Optimization Heatmap - {metric.replace("_", " ").title()}',
            xaxis_title='Minimum Factor',
            yaxis_title='ATR Length',
            width=800,
            height=600
        )
        
        # Save
        filename = f"optimization_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename
        fig.write_html(filepath)
        
        return str(filepath)
    
    def create_risk_analysis_dashboard(
        self,
        results: Dict
    ) -> str:
        """Create comprehensive risk analysis dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Risk-Return Scatter', 'Rolling Sharpe Ratio',
                'Value at Risk (VaR)', 'Risk Contribution'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "pie"}]
            ]
        )
        
        # Collect risk metrics
        risk_data = []
        for symbol, result in results.items():
            if 'error' not in result:
                risk_data.append({
                    'symbol': symbol,
                    'return': result.get('total_return', 0),
                    'volatility': result.get('volatility', 15),  # Default if not calculated
                    'sharpe': result.get('sharpe_ratio', 0),
                    'max_dd': abs(result.get('max_drawdown', 0))
                })
        
        if risk_data:
            risk_df = pd.DataFrame(risk_data)
            
            # 1. Risk-Return Scatter
            fig.add_trace(
                go.Scatter(
                    x=risk_df['volatility'],
                    y=risk_df['return'],
                    mode='markers+text',
                    text=risk_df['symbol'],
                    textposition="top center",
                    marker=dict(
                        size=risk_df['sharpe'] * 10,
                        color=risk_df['sharpe'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Sharpe Ratio")
                    ),
                    name='Risk-Return'
                ),
                row=1, col=1
            )
            
            # Add efficient frontier line (simplified)
            fig.add_trace(
                go.Scatter(
                    x=[0, 30],
                    y=[0, 30],
                    mode='lines',
                    line=dict(dash='dash', color='red'),
                    name='Efficient Frontier',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # 2. VaR Analysis
            fig.add_trace(
                go.Bar(
                    x=risk_df['symbol'],
                    y=risk_df['max_dd'],
                    name='Max Drawdown',
                    marker_color='red'
                ),
                row=2, col=1
            )
            
            # 3. Risk Contribution
            fig.add_trace(
                go.Pie(
                    labels=risk_df['symbol'],
                    values=risk_df['volatility'],
                    name='Volatility Contribution'
                ),
                row=2, col=2
            )
        
        # 4. Rolling Sharpe for first symbol with data
        for symbol, result in results.items():
            if 'portfolio_values' in result:
                portfolio = result['portfolio_values']
                if 'returns' in portfolio.columns:
                    # Calculate 30-day rolling Sharpe
                    rolling_sharpe = portfolio['returns'].rolling(30).apply(
                        lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=portfolio.index,
                            y=rolling_sharpe,
                            name=f'{symbol} Rolling Sharpe',
                            mode='lines'
                        ),
                        row=1, col=2
                    )
                    break
        
        # Update layout
        fig.update_layout(
            title='SuperTrend AI Risk Analysis',
            showlegend=True,
            height=800,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Volatility %", row=1, col=1)
        fig.update_yaxes(title_text="Return %", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Rolling Sharpe", row=1, col=2)
        fig.update_xaxes(title_text="Symbol", row=2, col=1)
        fig.update_yaxes(title_text="Max Drawdown %", row=2, col=1)
        
        # Save
        filename = f"risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename
        fig.write_html(filepath)
        
        return str(filepath)