# Multi-Indicator Confluence Strategy Simulation Plan

## Executive Summary

This document outlines a simplified plan for implementing a multi-indicator confluence strategy using the existing configuration in `/config/`. The focus is on model simulation and backtesting without advanced machine learning, leveraging the comprehensive indicator setup already defined.

### Key Objectives
- Use existing configuration from `strategy_config.yaml` as baseline
- Implement confluence scoring with configured indicators
- Focus on simulation and backtesting performance
- Use realistic trading costs from configuration
- Generate comprehensive performance reports using enhanced visualization style

## 1. Configuration Review

### 1.1 Current Assets (from config)
```yaml
assets:
  - SPY   # S&P 500 ETF
  - QQQ   # NASDAQ 100 ETF
  - AAPL  # Apple Inc.
  - MSFT  # Microsoft Corp.
  - JPM   # JPMorgan Chase
  - XLE   # Energy Select Sector SPDR
  - GLD   # SPDR Gold Trust
  - IWM   # iShares Russell 2000 ETF
  - TLT   # iShares 20+ Year Treasury Bond ETF
```

### 1.2 Data Settings
- **Period**: 2019-01-01 to 2025-07-10 (6.5 years)
- **Timeframes**: 1H, 4H, 1D, 1W, 1M
- **Data Available**: All assets have complete data

### 1.3 Configured Indicators
From `strategy_config.yaml`:
- **SMA**: [20, 50, 100, 200, 365] periods
- **Bollinger Bands**: [20, 50, 100, 200, 365] periods with [1.25, 2.2, 3.2] std devs
- **VWAP**: Daily, Weekly, Monthly, Yearly, 5Y with [1, 2, 3] std devs
- **Rolling VWAP**: Same periods as VWAP
- **RSI**: 14-period (oversold: 30, overbought: 70)
- **ATR**: 14-period

## 2. Trading Costs (From Configuration)

### 2.1 Commission Structure
```yaml
commission:
  fixed: 0.0
  percentage: 0.0005  # 5 basis points
  minimum: 0.0
```

### 2.2 Spread Model
```yaml
base_spread_pct:
  SPY: 0.0001   # 1 basis point
  QQQ: 0.0001   # 1 basis point
  AAPL: 0.0002  # 2 basis points
  MSFT: 0.0002  # 2 basis points
  JPM: 0.0003   # 3 basis points
  XLE: 0.0003   # 3 basis points
  GLD: 0.0002   # 2 basis points
  IWM: 0.0003   # 3 basis points
  TLT: 0.0002   # 2 basis points
```

### 2.3 Slippage & Market Impact
- Base slippage: 0.0001 (1 basis point)
- Size impact: 0.00001 per $10k traded
- Linear impact: 0.0001
- Square-root impact: 0.00001

## 3. Strategy Implementation

### 3.1 Confluence Scoring System

**Simple Weighted Scoring (No ML):**
```python
# Weight distribution based on indicator categories
weights = {
    'trend': 0.30,      # SMA alignments
    'momentum': 0.25,   # RSI signals
    'volatility': 0.25, # Bollinger Bands position
    'volume': 0.20      # VWAP relationships
}

# Score calculation
confluence_score = (
    trend_score * weights['trend'] +
    momentum_score * weights['momentum'] +
    volatility_score * weights['volatility'] +
    volume_score * weights['volume']
)
```

### 3.2 Indicator Signals

**Trend Signals (SMA):**
- Bullish: Price > SMA20 > SMA50 > SMA100 > SMA200
- Bearish: Price < SMA20 < SMA50 < SMA100 < SMA200
- Score: Percentage of aligned SMAs

**Momentum Signals (RSI):**
- Strong Buy: RSI < 30 (oversold)
- Buy: RSI 30-50 and rising
- Neutral: RSI 50-70
- Sell: RSI > 70 (overbought)

**Volatility Signals (Bollinger Bands):**
- Buy: Price near lower band (within 1.25 std)
- Sell: Price near upper band (within 1.25 std)
- Squeeze detection: Bands narrowing

**Volume Signals (VWAP):**
- Buy: Price < VWAP with increasing volume
- Sell: Price > VWAP with decreasing volume
- Multi-timeframe VWAP alignment

### 3.3 Entry/Exit Rules (From Config)

**Entry Conditions:**
- Min confluence score: 0.75 (75%)
- Reentry delay: 5 days
- Position sizing: Kelly fraction 0.3
- Max position: 20% of portfolio

**Exit Conditions:**
- Profit target: 15%
- Time stop: 30 days
- Stop loss: Dynamic ATR (2x multiplier)
- Min stop: 0.8%, Max stop: 4%

## 4. Portfolio Management

### 4.1 Position Sizing (From Config)
```yaml
position_sizing:
  method: "kelly"
  kelly_fraction: 0.3
  max_position_pct: 0.2
  min_position_size: 1000
```

### 4.2 Risk Management
```yaml
risk_management:
  max_positions: 10
  correlation_limit: 0.7
  sector_limit: 0.3
```

### 4.3 Rebalancing
- Frequency: Quarterly (from config)
- Method: Target weights based on confluence scores

## 5. Enhanced Visualization Reporting

### 5.1 Report Template Structure
Based on `enhanced_strategy_visualization_report.html`, all reports will include:

**Core Components:**
1. **Header Section**
   - Strategy name and version
   - AI optimization status
   - Live indicator animation

2. **Enhanced Metrics Dashboard**
   - Expected annual return with improvement indicators
   - Sharpe ratio with comparison
   - Max drawdown reduction
   - Win rate improvements
   - Confluence requirements
   - Stop loss type and range

3. **Master Trading View**
   - TradingView Lite integration
   - Candlestick chart with black price action
   - All indicators overlay:
     - Bollinger Bands (blue)
     - VWAP (orange)
     - Max Pain levels (pink)
     - Buy/Sell signals with confluence scores
   - Volume histogram
   - Interactive tooltips

4. **Multi-Indicator Analysis Panels**
   - RSI panel with oversold/overbought zones
   - Volume analysis panel
   - Confluence score timeline
   - Real-time indicator values

5. **Real-Time Confluence Meter**
   - Visual progress bar (0-100%)
   - Individual indicator scores
   - Color-coded thresholds
   - Animated updates

6. **Trade Entry Examples**
   - High-confluence trade cards
   - Detailed signal breakdown
   - Entry/exit prices
   - Result tracking

7. **Performance Analytics**
   - Monthly performance heatmap
   - Drawdown visualization
   - Stop loss comparison (fixed vs dynamic)
   - Win/loss distribution

8. **Detailed Trade List**
   - Sortable/filterable table
   - Export to CSV functionality
   - Signal details for each trade
   - Performance metrics per trade

### 5.2 Visual Design Standards

**Color Scheme:**
```css
background: #0a0e1a;        /* Dark background */
primary: #4a90e2;           /* Blue for primary elements */
success: #44ff44;           /* Green for positive */
danger: #ff4444;            /* Red for negative */
warning: #ffaa00;           /* Orange for neutral */
text: #e0e0e0;              /* Light text */
```

**Chart Configuration:**
```javascript
// TradingView Lite settings
layout: {
    backgroundColor: '#0a0e1a',
    textColor: '#e0e0e0',
}
// Black candlesticks for price action
candlestick: {
    upColor: '#000000',
    downColor: '#000000',
}
```

### 5.3 Interactive Features

1. **Dynamic Updates**
   - Confluence meter animation
   - Live indicator values
   - Hover effects on all elements

2. **Export Capabilities**
   - Trade list to CSV
   - Charts as PNG
   - Full report as PDF

3. **Responsive Design**
   - Mobile-friendly layout
   - Auto-resizing charts
   - Touch-enabled interactions

### 5.4 Report Generation Workflow

**File: `src/visualization/enhanced_report_generator.py`**

```python
class EnhancedReportGenerator:
    def __init__(self):
        self.template_path = 'templates/enhanced_report_template.html'
        self.output_dir = 'reports/confluence_simulation/'
    
    def generate_report(self, backtest_results, strategy_config):
        # Load template
        # Inject data
        # Generate charts
        # Create interactive elements
        # Save HTML report
        pass
```

## 6. Implementation Steps

### 6.1 Phase 1: Indicator Testing & Debugging
**Files to create/test:**
- `test_all_indicators.py` - Verify each indicator works with real data
- `debug_indicators.py` - Fix any issues found

**Actions:**
1. Test SMA calculations on all periods
2. Verify Bollinger Bands with multiple periods
3. Test VWAP and Rolling VWAP calculations
4. Verify RSI and ATR calculations
5. Create visualization of all indicators

### 6.2 Phase 2: Confluence Strategy Implementation
**File: `src/strategies/confluence_strategy.py`**

```python
class ConfluenceStrategy:
    def __init__(self, config_path='config/strategy_config.yaml'):
        self.config = load_config(config_path)
        self.indicators = {}
        self.confluence_scores = {}
    
    def calculate_indicators(self, data):
        # Calculate all configured indicators
        pass
    
    def calculate_confluence_score(self, symbol, timestamp):
        # Combine indicator signals
        pass
    
    def generate_signals(self):
        # Entry/exit signals based on confluence
        pass
```

### 6.3 Phase 3: Enhanced Report Generator
**File: `src/visualization/enhanced_report_generator.py`**

Key features to implement:
- TradingView Lite chart integration
- Real-time confluence meter
- Trade entry example cards
- Performance heatmap
- Interactive trade list with export
- Dynamic stop loss visualization

### 6.4 Phase 4: Backtesting Setup
**File: `run_confluence_simulation.py`**

```python
# Configuration
config = {
    'initial_capital': 10000,
    'monthly_contribution': 500,
    'start_date': '2019-01-01',
    'end_date': '2025-07-10'
}

# Run backtest on all assets
for asset in ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']:
    backtest_asset(asset, config)

# Generate enhanced report
report_generator = EnhancedReportGenerator()
report_generator.generate_report(results, config)
```

## 7. Testing Protocol

### 7.1 Unit Tests
- Test each indicator individually
- Verify confluence score calculation
- Test entry/exit signal generation
- Validate position sizing

### 7.2 Integration Tests
- Test full strategy on single asset
- Verify trading cost calculations
- Test portfolio constraints
- Validate risk management rules

### 7.3 Simulation Tests
- Run on historical data (2019-2023)
- Out-of-sample test (2024-2025)
- Stress test with high volatility periods
- Compare to buy-and-hold benchmark

### 7.4 Report Validation
- Verify all charts render correctly
- Test interactive features
- Validate export functionality
- Check mobile responsiveness

## 8. Expected Outputs

### 8.1 Enhanced Performance Reports
**Location: `reports/confluence_simulation/`**

1. **Main Report** (`enhanced_strategy_report.html`)
   - Full interactive dashboard
   - All visualizations and metrics
   - Trade list with export
   - Based on enhanced template style

2. **Asset-Specific Reports** (`{asset}_report.html`)
   - Individual asset performance
   - Detailed trade analysis
   - Indicator effectiveness

3. **Comparison Report** (`benchmark_comparison.html`)
   - Strategy vs Buy-and-hold
   - Risk-adjusted metrics
   - Drawdown analysis

### 8.2 Data Outputs
- `backtest_results.json` - All trade data
- `performance_metrics.csv` - Key metrics
- `indicator_signals.parquet` - All indicator values
- `confluence_scores.csv` - Score history

## 9. Optimization Approach (Simple Grid Search)

### 9.1 Parameters to Optimize (From Config)
```yaml
parameter_ranges:
  confluence_threshold: [0.65, 0.85]
  atr_multiplier: [1.5, 3.0]
  kelly_fraction: [0.2, 0.4]
```

### 9.2 Walk-Forward Analysis
- In-sample: 12 months
- Out-of-sample: 3 months
- Step: 1 month

### 9.3 Objective Function
- Primary: Sharpe ratio
- Secondary: Calmar ratio
- Constraint: Max drawdown < 20%

## 10. Implementation Timeline

### Week 1: Foundation
- [ ] Fix test infrastructure
- [ ] Test all indicators with real data
- [ ] Create debugging scripts
- [ ] Set up report template structure

### Week 2: Strategy Development
- [ ] Implement confluence scoring
- [ ] Create signal generation logic
- [ ] Add position sizing
- [ ] Implement risk management

### Week 3: Report Generator
- [ ] Create enhanced report generator
- [ ] Implement TradingView Lite charts
- [ ] Add interactive elements
- [ ] Test report generation

### Week 4: Backtesting & Analysis
- [ ] Run comprehensive backtests
- [ ] Optimize parameters
- [ ] Generate final reports
- [ ] Document results

## 11. Success Metrics

### Performance Targets
- Sharpe Ratio: > 1.2
- Annual Return: > 12%
- Max Drawdown: < 15%
- Win Rate: > 50%

### Report Quality
- All charts render correctly
- Interactive features work
- Mobile responsive
- Export functionality operational

### Operational Targets
- All indicators functioning correctly
- Backtests complete on all assets
- Enhanced reports generated successfully
- Code well-documented

## 12. Report Examples Structure

Based on the enhanced template, each report will contain:

```html
<!-- Header with live indicator -->
<header>
    <h1>Strategy Name v2.0</h1>
    <p>AI-Optimized Multi-Indicator Trading System</p>
    <span class="live-indicator"></span>
</header>

<!-- Metrics Dashboard -->
<div class="metrics-grid">
    <!-- 6 key metrics with improvement indicators -->
</div>

<!-- Master Chart -->
<div id="master-chart" class="main-chart">
    <!-- TradingView Lite implementation -->
</div>

<!-- Indicator Panels -->
<div class="indicator-panel">
    <!-- RSI, Volume, Confluence charts -->
</div>

<!-- Real-time Confluence Meter -->
<div class="confluence-meter">
    <!-- Animated progress bar -->
</div>

<!-- Trade Examples -->
<div class="trade-entry-card">
    <!-- High-confluence examples -->
</div>

<!-- Performance Analytics -->
<div class="performance-heatmap">
    <!-- Monthly returns heatmap -->
</div>

<!-- Trade List -->
<table class="asset-table">
    <!-- Detailed trade history -->
</table>
```

This plan focuses on practical implementation using the existing configuration without complex ML models, while incorporating the enhanced visualization style for professional-grade reporting.