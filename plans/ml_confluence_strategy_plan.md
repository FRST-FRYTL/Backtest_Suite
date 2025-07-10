# ML-Enhanced Multi-Indicator Confluence Strategy Plan

## Executive Summary

This document outlines a comprehensive plan for developing an advanced trading strategy that combines multiple technical and meta indicators with machine learning models to optimize portfolio performance across multiple assets (SPY, QQQ, IWM, TLT, GLD).

### Key Objectives
- Integrate 15+ technical indicators with confluence scoring
- Incorporate meta indicators (Fear & Greed, Insider Trading, Max Pain)
- Use ML ensemble models for prediction and optimization
- Implement realistic trading costs based on Interactive Brokers
- Optimize for risk-adjusted returns (Sharpe ratio > 1.5)
- Generate institutional-grade reporting

## 1. Infrastructure Preparation

### 1.1 Fix Testing Framework
**Priority: Critical**
**Files to modify:**
- `tests/conftest.py` - Fix imports (DataFetcher → StockDataFetcher, CacheManager → DataCache)

**Actions:**
```python
# Current (broken)
from src.data import DataFetcher, CacheManager

# Fixed
from src.data import StockDataFetcher, DataCache
```

### 1.2 Verify Data Availability
**Current Status:**
- ✅ SPY: 1,638 daily bars (2019-01-02 to 2025-07-09)
- ✅ QQQ: Complete data through 2025-07-09
- ✅ IWM: Complete data through 2025-07-09
- ✅ TLT: Complete data through 2025-07-09
- ✅ GLD: Complete data through 2025-07-09

## 2. Real-World Trading Costs Configuration

### 2.1 Commission Structure (Interactive Brokers Pro)
```yaml
commission:
  per_share: 0.005  # $0.005 per share
  minimum: 1.00     # $1.00 minimum per order
  maximum_pct: 0.005  # 0.5% of trade value maximum
```

### 2.2 Bid-Ask Spreads (2025 Market Data)
```yaml
spread:
  base_spread_pct:
    SPY: 0.00003   # 0.3 basis points (extremely liquid)
    QQQ: 0.00003   # 0.3 basis points
    IWM: 0.00005   # 0.5 basis points
    TLT: 0.00023   # 2.3 basis points (less liquid)
    GLD: 0.00005   # 0.5 basis points
```

### 2.3 Slippage Model
```yaml
slippage:
  base_slippage_pct: 0.00005  # 0.5 basis points
  size_impact: 0.00001        # Per $10k traded
  urgency_multiplier: 1.5     # For market orders
```

## 3. Technical Indicators Integration

### 3.1 Trend Following Indicators
- **SMA**: 20, 50, 100, 200 periods
- **EMA**: 12, 26, 50 periods
- **MACD**: (12, 26, 9) with signal line
- **ADX**: 14-period for trend strength

### 3.2 Momentum Indicators
- **RSI**: 14 and 21 periods
- **Stochastic**: (14, 3, 3) %K and %D
- **Rate of Change (ROC)**: 10 and 20 periods

### 3.3 Volatility Indicators
- **Bollinger Bands**: 20-period, [1.5, 2.0, 2.5] std devs
- **ATR**: 14-period for volatility measurement
- **Keltner Channels**: 20-period, 2x ATR

### 3.4 Volume Indicators
- **OBV**: On-Balance Volume
- **Volume SMA**: 20-period average
- **VWAP**: Daily, Weekly, Monthly
- **Rolling VWAP**: [5, 10, 20, 50] periods

### 3.5 Market Structure
- **Support/Resistance**: Automated detection
- **Pivot Points**: Daily and weekly
- **Market Profile**: Volume at price levels

## 4. Meta Indicators Integration

### 4.1 Fear & Greed Index
- **Source**: Alternative.me API
- **Interpretation**:
  - < 20: Extreme Fear (Bullish contrarian)
  - 20-40: Fear
  - 40-60: Neutral
  - 60-80: Greed
  - > 80: Extreme Greed (Bearish contrarian)

### 4.2 Insider Trading
- **Source**: OpenInsider scraping
- **Metrics**:
  - Net insider buying/selling ratio
  - Transaction size relative to market cap
  - Cluster detection (multiple insiders)

### 4.3 Max Pain
- **Calculation**: Options open interest analysis
- **Usage**:
  - Support/resistance levels
  - Expiration week positioning
  - Gamma exposure analysis

## 5. Confluence Scoring System

### 5.1 Signal Generation
```python
confluence_score = weighted_sum([
    trend_score * 0.25,      # SMA, EMA, MACD alignment
    momentum_score * 0.20,   # RSI, Stochastic, ROC
    volatility_score * 0.15, # BB, ATR position
    volume_score * 0.15,     # OBV, VWAP confirmation
    sentiment_score * 0.15,  # Fear & Greed, Insider
    ml_score * 0.10         # ML model predictions
])
```

### 5.2 Multi-Timeframe Confirmation
- **Primary**: Daily timeframe
- **Confirmation**: 4H and Weekly
- **Entry**: 1H for timing

### 5.3 Signal Thresholds
- **Strong Buy**: Confluence > 80
- **Buy**: Confluence > 65
- **Neutral**: 35-65
- **Sell**: Confluence < 35
- **Strong Sell**: Confluence < 20

## 6. Machine Learning Architecture

### 6.1 Feature Engineering (100+ features)
**Price Features:**
- Returns: [1, 2, 5, 10, 20] days
- Log returns
- Price/SMA ratios: [20, 50, 200]
- High-low range percentages
- Close position in daily range

**Technical Features:**
- All indicator values (normalized)
- Indicator crossovers (binary)
- Divergence detection
- Distance from key levels (%)

**Market Microstructure:**
- Bid-ask spread changes
- Volume profiles
- Intraday volatility
- Opening gaps

**Cross-Asset Features:**
- Correlations (rolling 20-day)
- Relative strength
- Beta to SPY
- Sector momentum

### 6.2 Model Ensemble
**XGBoost Direction Predictor:**
- Target: 5-day forward return direction
- Features: 100+ technical and fundamental
- Validation: Walk-forward with 3-month windows

**LSTM Volatility Forecaster:**
- Architecture: 2-layer LSTM with attention
- Sequence length: 60 days
- Output: 5-day ATR forecast with confidence

**Market Regime Detector:**
- 5 regimes: Bull, Bear, Sideways, High Vol, Low Vol
- Hidden Markov Model
- Transition probabilities

**Ensemble Weighting:**
- Recent accuracy tracking (20-day)
- Confidence-weighted averaging
- Regime-specific adjustments

## 7. Portfolio Optimization Strategy

### 7.1 Asset Allocation Framework
**Optimization Methods:**
1. **Mean-Variance Optimization**
   - Maximize Sharpe ratio
   - Monthly rebalancing
   - Transaction cost aware

2. **Risk Parity**
   - Equal risk contribution
   - Daily volatility estimates
   - Leverage constraints

3. **Black-Litterman**
   - Market equilibrium baseline
   - ML views integration
   - Confidence scaling

### 7.2 Constraints
```python
constraints = {
    'min_weight': 0.10,      # Minimum 10% when active
    'max_weight': 0.40,      # Maximum 40% per asset
    'max_correlation': 0.70,  # Between any two assets
    'min_assets': 3,         # Diversification requirement
    'max_turnover': 0.50     # Monthly turnover limit
}
```

### 7.3 Dynamic Adjustments
- **Volatility Scaling**: Reduce exposure in high volatility
- **Regime Adaptation**: Conservative in bear markets
- **Momentum Tilt**: Overweight trending assets

## 8. Risk Management Framework

### 8.1 Position Sizing
**Kelly Criterion (Modified):**
```python
position_size = kelly_fraction * (expected_return / variance) * capital
kelly_fraction = 0.25  # Conservative 25% of full Kelly
```

**Volatility-Based:**
```python
position_size = (risk_per_trade * capital) / (ATR * ATR_multiplier)
risk_per_trade = 0.02  # 2% maximum
ATR_multiplier = 2.0   # Stop distance
```

### 8.2 Stop Loss Strategy
**Dynamic ATR-Based:**
- Initial: 2.0 × ATR
- Trailing: 1.5 × ATR after 5% profit
- Maximum: 4% from entry

**Support/Resistance:**
- Place below nearest support
- Minimum 1% buffer
- Adjust for volatility

### 8.3 Portfolio Risk Limits
- **Maximum Drawdown**: 15% (circuit breaker)
- **Daily VaR (95%)**: 2% of portfolio
- **Correlation Limit**: 0.7 between positions
- **Sector Exposure**: 60% maximum
- **Beta to Market**: 0.5 to 1.5 range

## 9. Backtesting Methodology

### 9.1 Data Configuration
- **Period**: 2019-01-01 to 2025-07-10 (6.5 years)
- **Initial Capital**: $10,000
- **Monthly Contributions**: $500
- **Total Investment**: ~$49,000

### 9.2 Walk-Forward Analysis
```
Training Window: 12 months
Testing Window: 3 months
Step Size: 1 month
Total Folds: 20+
```

### 9.3 Performance Metrics
**Primary Metrics:**
- Sharpe Ratio (target > 1.5)
- Maximum Drawdown (< 15%)
- Calmar Ratio (> 1.0)
- Win Rate (> 55%)

**Secondary Metrics:**
- Profit Factor
- Average Win/Loss Ratio
- Recovery Time
- Tail Ratio
- Omega Ratio

### 9.4 Robustness Testing
- Monte Carlo simulation (1000 runs)
- Parameter sensitivity analysis
- Regime performance breakdown
- Transaction cost sensitivity

## 10. Reporting Suite

### 10.1 Performance Dashboard
**Interactive Plotly Dashboard:**
- Cumulative returns vs benchmarks
- Rolling performance metrics
- Drawdown visualization
- Monthly returns heatmap
- Asset allocation timeline

### 10.2 ML Analytics Report
**Model Performance:**
- Feature importance rankings
- Prediction accuracy over time
- Confidence calibration plots
- Regime detection accuracy
- Out-of-sample validation

### 10.3 Risk Analysis Report
**Risk Metrics:**
- VaR and CVaR evolution
- Correlation matrices (dynamic)
- Beta exposure tracking
- Stress test scenarios
- Worst-case analysis

### 10.4 Trade Analysis
**Detailed Metrics:**
- Entry/exit efficiency
- Slippage analysis
- Win/loss distribution
- Holding period analysis
- Best/worst trades

## 11. Implementation Timeline

### Phase 1: Foundation (Week 1)
- [ ] Fix testing infrastructure
- [ ] Test all indicators with real data
- [ ] Update trading cost configuration
- [ ] Create indicator debugging scripts

### Phase 2: Strategy Development (Week 2)
- [ ] Implement confluence scoring system
- [ ] Integrate all technical indicators
- [ ] Add meta indicators
- [ ] Create multi-timeframe analysis

### Phase 3: ML Integration (Week 3)
- [ ] Build feature engineering pipeline
- [ ] Train individual ML models
- [ ] Create ensemble framework
- [ ] Validate predictions

### Phase 4: Portfolio & Risk (Week 4)
- [ ] Implement portfolio optimization
- [ ] Add risk management rules
- [ ] Create position sizing logic
- [ ] Test stop-loss mechanisms

### Phase 5: Testing & Reporting (Week 5)
- [ ] Run comprehensive backtests
- [ ] Generate performance reports
- [ ] Conduct robustness testing
- [ ] Create final documentation

## 12. Success Criteria

### Performance Targets
- **Sharpe Ratio**: > 1.5
- **Annual Return**: > 15%
- **Maximum Drawdown**: < 15%
- **Win Rate**: > 55%
- **Recovery Time**: < 3 months

### Risk Targets
- **Daily VaR (95%)**: < 2%
- **Beta to SPY**: 0.5-1.0
- **Correlation between positions**: < 0.7
- **Tail Risk Ratio**: > 1.0

### Operational Targets
- **Execution latency**: < 100ms
- **Rebalancing frequency**: Monthly
- **Computation time**: < 5 minutes
- **Report generation**: < 1 minute

## 13. Future Enhancements

### Short Term (3 months)
- Add options strategies for hedging
- Integrate real-time data feeds
- Add more alternative data sources
- Implement paper trading mode

### Medium Term (6 months)
- Deep learning models (Transformers)
- Reinforcement learning for optimization
- Multi-strategy portfolio
- API for external access

### Long Term (12 months)
- Fully automated execution
- Cloud-based deployment
- Mobile app interface
- Institutional features

## Appendix A: Technical Requirements

### Computing Resources
- **CPU**: 8+ cores recommended
- **RAM**: 16GB minimum, 32GB preferred
- **Storage**: 100GB for historical data
- **GPU**: Optional for deep learning

### Software Dependencies
- Python 3.9+
- pandas, numpy, scikit-learn
- XGBoost, LightGBM
- TensorFlow/PyTorch
- Plotly, matplotlib
- yfinance for data

### Development Tools
- VS Code with Python extensions
- Jupyter notebooks for exploration
- Git for version control
- Docker for deployment

## Appendix B: Risk Disclaimer

This strategy involves significant risks including but not limited to:
- Market risk and potential losses
- Model risk from ML predictions
- Execution risk from slippage
- Technology risk from system failures
- Past performance does not guarantee future results

Always conduct thorough testing and consider consulting with financial professionals before deploying real capital.