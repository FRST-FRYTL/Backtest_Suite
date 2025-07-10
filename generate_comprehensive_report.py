"""
Generate comprehensive reports for all Backtest Suite features
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from src.data.download_historical_data import load_cached_data


class ComprehensiveReportGenerator:
    """Generate comprehensive visual reports for Backtest Suite"""
    
    def __init__(self):
        self.report_dir = Path("reports")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_all_reports(self):
        """Generate all report types"""
        print("=" * 80)
        print("GENERATING COMPREHENSIVE REPORTS")
        print("=" * 80)
        print(f"Started: {datetime.now()}\n")
        
        # 1. Data Quality Report
        self.generate_data_quality_report()
        
        # 2. Market Overview Report
        self.generate_market_overview_report()
        
        # 3. Indicator Analysis Report
        self.generate_indicator_report()
        
        # 4. Performance Dashboard
        self.generate_performance_dashboard()
        
        print("\n" + "=" * 80)
        print("REPORT GENERATION COMPLETE")
        print("=" * 80)
        print(f"All reports saved to: {self.report_dir}")
        
    def generate_data_quality_report(self):
        """Generate data quality visualization report"""
        print("\n📊 Generating Data Quality Report...")
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Data Availability by Asset',
                'Data Points by Timeframe',
                'Date Coverage',
                'Data Quality Scores'
            ),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'indicator'}]]
        )
        
        # Analyze data for each asset
        assets = ['SPY', 'QQQ', 'AAPL', 'MSFT']
        timeframes = ['1D', '1W', '1M']
        colors = px.colors.qualitative.Set3
        
        # Data availability
        availability_data = []
        quality_scores = []
        
        for i, asset in enumerate(assets):
            asset_data = []
            for tf in timeframes:
                data = load_cached_data(asset, tf)
                if data is not None:
                    asset_data.append(len(data))
                else:
                    asset_data.append(0)
            
            # Add to availability chart
            fig.add_trace(
                go.Bar(name=asset, x=timeframes, y=asset_data, marker_color=colors[i]),
                row=1, col=1
            )
            
            # Calculate quality score
            daily_data = load_cached_data(asset, '1D')
            if daily_data is not None:
                missing = daily_data.isnull().sum().sum()
                total = daily_data.size
                quality_score = (1 - missing/total) * 100
                quality_scores.append(quality_score)
            else:
                quality_scores.append(0)
        
        # Data points by timeframe
        for i, tf in enumerate(timeframes):
            tf_counts = []
            for asset in assets:
                data = load_cached_data(asset, tf)
                tf_counts.append(len(data) if data is not None else 0)
            
            fig.add_trace(
                go.Bar(name=tf, x=assets, y=tf_counts, marker_color=colors[i+4]),
                row=1, col=2
            )
        
        # Date coverage timeline
        for i, asset in enumerate(assets):
            data = load_cached_data(asset, '1D')
            if data is not None and len(data) > 0:
                dates = pd.date_range(data.index[0], data.index[-1], periods=10)
                y_values = [i] * len(dates)
                
                fig.add_trace(
                    go.Scatter(
                        x=dates, y=y_values,
                        mode='lines+markers',
                        name=asset,
                        line=dict(color=colors[i], width=4),
                        marker=dict(size=8)
                    ),
                    row=2, col=1
                )
        
        # Quality score gauge
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=avg_quality,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Average Data Quality"},
                delta={'reference': 95},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95
                    }
                }
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Data Quality Report",
            showlegend=True,
            height=800,
            template="plotly_white"
        )
        
        # Save report
        output_path = self.report_dir / 'data_quality' / f'data_quality_report_{self.timestamp}.html'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        
        print(f"✅ Data Quality Report saved: {output_path}")
        
    def generate_market_overview_report(self):
        """Generate market overview report"""
        print("\n📈 Generating Market Overview Report...")
        
        # Load SPY data as market proxy
        spy_data = load_cached_data('SPY', '1D')
        
        if spy_data is None:
            print("❌ No SPY data available for market overview")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'SPY Price History',
                'Market Returns Distribution',
                'Volume Analysis',
                'Volatility Over Time',
                'Correlation Matrix',
                'Market Regimes'
            ),
            specs=[[{'secondary_y': True}, {'type': 'histogram'}],
                   [{'secondary_y': True}, {'type': 'scatter'}],
                   [{'type': 'heatmap'}, {'type': 'bar'}]],
            row_heights=[0.35, 0.35, 0.3]
        )
        
        # 1. Price history with volume
        fig.add_trace(
            go.Candlestick(
                x=spy_data.index,
                open=spy_data['open'],
                high=spy_data['high'],
                low=spy_data['low'],
                close=spy_data['close'],
                name='SPY Price'
            ),
            row=1, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Bar(
                x=spy_data.index,
                y=spy_data['volume'],
                name='Volume',
                marker_color='lightblue',
                opacity=0.3
            ),
            row=1, col=1, secondary_y=True
        )
        
        # 2. Returns distribution
        returns = spy_data['close'].pct_change().dropna()
        
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name='Daily Returns',
                marker_color='green',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Add normal distribution overlay
        mean_return = returns.mean()
        std_return = returns.std()
        x_norm = np.linspace(returns.min(), returns.max(), 100)
        y_norm = ((1 / (std_return * np.sqrt(2 * np.pi))) * 
                  np.exp(-0.5 * ((x_norm - mean_return) / std_return) ** 2))
        
        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm * len(returns) * (returns.max() - returns.min()) / 50,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
        
        # 3. Volume analysis
        volume_ma = spy_data['volume'].rolling(20).mean()
        
        fig.add_trace(
            go.Scatter(
                x=spy_data.index,
                y=spy_data['volume'],
                mode='lines',
                name='Daily Volume',
                line=dict(color='blue', width=1),
                opacity=0.5
            ),
            row=2, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=spy_data.index,
                y=volume_ma,
                mode='lines',
                name='20-day MA Volume',
                line=dict(color='red', width=2)
            ),
            row=2, col=1, secondary_y=False
        )
        
        # 4. Volatility (rolling std)
        volatility = returns.rolling(30).std() * np.sqrt(252)
        
        fig.add_trace(
            go.Scatter(
                x=volatility.index,
                y=volatility,
                mode='lines',
                name='30-day Volatility',
                line=dict(color='purple', width=2),
                fill='tozeroy',
                fillcolor='rgba(128, 0, 128, 0.1)'
            ),
            row=2, col=2
        )
        
        # 5. Correlation matrix (load multiple assets)
        assets = ['SPY', 'QQQ', 'AAPL', 'MSFT']
        returns_data = pd.DataFrame()
        
        for asset in assets:
            data = load_cached_data(asset, '1D')
            if data is not None:
                returns_data[asset] = data['close'].pct_change()
        
        if not returns_data.empty:
            corr_matrix = returns_data.corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ),
                row=3, col=1
            )
        
        # 6. Market regimes (simple volatility-based)
        vol_percentile = volatility.rank(pct=True)
        regimes = pd.cut(vol_percentile, bins=[0, 0.33, 0.67, 1.0], 
                        labels=['Low Vol', 'Normal', 'High Vol'])
        regime_counts = regimes.value_counts()
        
        fig.add_trace(
            go.Bar(
                x=regime_counts.index,
                y=regime_counts.values,
                marker_color=['green', 'yellow', 'red'],
                name='Regime Days'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Market Overview Dashboard",
            height=1200,
            showlegend=True,
            template="plotly_white"
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
        
        # Save report
        output_path = self.report_dir / 'visualizations' / f'market_overview_{self.timestamp}.html'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        
        print(f"✅ Market Overview Report saved: {output_path}")
        
    def generate_indicator_report(self):
        """Generate technical indicator analysis report"""
        print("\n📊 Generating Indicator Analysis Report...")
        
        # Load SPY data
        spy_data = load_cached_data('SPY', '1D')
        
        if spy_data is None:
            print("❌ No data available for indicator analysis")
            return
        
        # Calculate indicators
        spy_data['SMA_20'] = spy_data['close'].rolling(20).mean()
        spy_data['SMA_50'] = spy_data['close'].rolling(50).mean()
        spy_data['RSI'] = self._calculate_rsi(spy_data['close'])
        bb_data = self._calculate_bollinger_bands(spy_data['close'])
        spy_data = pd.concat([spy_data, bb_data], axis=1)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Price with Moving Averages',
                'RSI Indicator',
                'Bollinger Bands'
            ),
            row_heights=[0.4, 0.3, 0.3],
            vertical_spacing=0.05
        )
        
        # 1. Price with SMAs
        fig.add_trace(
            go.Scatter(
                x=spy_data.index,
                y=spy_data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='black', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=spy_data.index,
                y=spy_data['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=spy_data.index,
                y=spy_data['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
        
        # 2. RSI
        fig.add_trace(
            go.Scatter(
                x=spy_data.index,
                y=spy_data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # 3. Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=spy_data.index,
                y=spy_data['close'],
                mode='lines',
                name='Price',
                line=dict(color='black', width=1)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=spy_data.index,
                y=spy_data['BB_upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=spy_data.index,
                y=spy_data['BB_lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.1)'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title="Technical Indicator Analysis",
            height=1000,
            showlegend=True,
            template="plotly_white"
        )
        
        # Save report
        output_path = self.report_dir / 'indicators' / f'indicator_analysis_{self.timestamp}.html'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        
        print(f"✅ Indicator Analysis Report saved: {output_path}")
        
    def generate_performance_dashboard(self):
        """Generate performance dashboard"""
        print("\n📈 Generating Performance Dashboard...")
        
        # Create a sample performance dashboard
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Simulate portfolio performance
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, len(dates))
        equity = 100000 * (1 + returns).cumprod()
        
        # Calculate metrics
        total_return = (equity[-1] / equity[0] - 1) * 100
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        max_dd = self._calculate_max_drawdown(equity)
        
        # Create dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Portfolio Value',
                'Daily Returns',
                'Drawdown',
                'Monthly Returns',
                'Risk Metrics',
                'Performance Summary'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'indicator'}, {'type': 'table'}]],
            row_heights=[0.6, 0.4]
        )
        
        # 1. Equity curve
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=equity,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 0, 255, 0.1)'
            ),
            row=1, col=1
        )
        
        # 2. Returns distribution
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=30,
                name='Daily Returns',
                marker_color='green',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # 3. Drawdown
        equity_series = pd.Series(equity)
        dd = (equity_series - equity_series.expanding().max()) / equity_series.expanding().max() * 100
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=dd,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)'
            ),
            row=1, col=3
        )
        
        # 4. Monthly returns
        monthly_returns = pd.Series(returns, index=dates).resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        fig.add_trace(
            go.Bar(
                x=monthly_returns.index,
                y=monthly_returns.values * 100,
                name='Monthly Returns',
                marker_color=['green' if r > 0 else 'red' for r in monthly_returns.values]
            ),
            row=2, col=1
        )
        
        # 5. Risk gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=sharpe_ratio,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sharpe Ratio"},
                gauge={
                    'axis': {'range': [0, 3]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgray"},
                        {'range': [1, 2], 'color': "gray"},
                        {'range': [2, 3], 'color': "lightgreen"}
                    ]
                }
            ),
            row=2, col=2
        )
        
        # 6. Summary table
        summary_data = [
            ['Total Return', f'{total_return:.2f}%'],
            ['Sharpe Ratio', f'{sharpe_ratio:.2f}'],
            ['Max Drawdown', f'{max_dd:.2f}%'],
            ['Win Rate', f'{(returns > 0).mean() * 100:.1f}%'],
            ['Best Day', f'{returns.max() * 100:.2f}%'],
            ['Worst Day', f'{returns.min() * 100:.2f}%']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=list(zip(*summary_data)),
                          fill_color='lavender',
                          align='left')
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title="Performance Dashboard",
            height=800,
            showlegend=False,
            template="plotly_white"
        )
        
        # Save report
        output_path = self.report_dir / 'performance' / f'performance_dashboard_{self.timestamp}.html'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        
        # Create latest symlink
        latest_path = self.report_dir / 'visualizations' / 'performance_dashboard.html'
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(f'../../performance/performance_dashboard_{self.timestamp}.html')
        except:
            # If symlink fails, copy the file
            import shutil
            shutil.copy(output_path, latest_path)
        
        print(f"✅ Performance Dashboard saved: {output_path}")
        print(f"📊 Latest dashboard: {latest_path}")
        
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        bb_data = pd.DataFrame()
        bb_data['BB_middle'] = sma
        bb_data['BB_upper'] = sma + (std_dev * std)
        bb_data['BB_lower'] = sma - (std_dev * std)
        
        return bb_data
    
    def _calculate_max_drawdown(self, equity):
        """Calculate maximum drawdown"""
        # Convert to pandas Series if numpy array
        if isinstance(equity, np.ndarray):
            equity = pd.Series(equity)
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        return drawdown.min() * 100


if __name__ == "__main__":
    generator = ComprehensiveReportGenerator()
    generator.generate_all_reports()