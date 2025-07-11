# ML Real-World Backtesting Reports

## Report Structure

This directory contains comprehensive reports from ML-enhanced backtesting with real market data.

### Directory Structure

```
ml_real_world/
├── index.html                     # Master Dashboard (Entry Point)
├── executive_summary.html         # High-level performance overview
├── ml_models/                     # Machine Learning Model Reports
│   ├── direction_predictor_report.html
│   ├── volatility_forecaster_report.html
│   ├── regime_detector_report.html
│   └── ensemble_performance.html
├── backtesting/                   # Backtesting Analysis
│   ├── strategy_comparison.html
│   ├── asset_performance.html
│   ├── trade_analysis.html
│   └── risk_metrics.html
├── feature_analysis/              # Feature Engineering Reports
│   ├── importance_scores.html
│   ├── correlation_matrix.html
│   └── engineering_pipeline.html
├── performance/                   # Performance Metrics
│   ├── returns_analysis.html
│   ├── sharpe_sortino.html
│   ├── drawdown_analysis.html
│   └── rolling_metrics.html
├── market_analysis/               # Market Analysis
│   ├── regime_transitions.html
│   ├── volatility_patterns.html
│   └── correlation_dynamics.html
└── data/                         # Raw Data Export
    └── raw_results.json
```

### Report Components

#### 1. Master Dashboard (index.html)
- Overview of all metrics
- Navigation to all sub-reports
- Key performance indicators
- Quick access to all report sections

#### 2. Executive Summary
- Strategy overview
- Key findings
- Performance highlights
- Risk assessment
- Recommendations

#### 3. ML Model Reports
- **Direction Predictor**: XGBoost/LightGBM/CatBoost ensemble performance
- **Volatility Forecaster**: LSTM prediction accuracy and patterns
- **Regime Detector**: Market state classification results
- **Ensemble Performance**: Combined model effectiveness

#### 4. Backtesting Analysis
- **Strategy Comparison**: ML vs traditional strategies
- **Asset Performance**: Individual asset returns
- **Trade Analysis**: Entry/exit patterns, win/loss distribution
- **Risk Metrics**: VaR, CVaR, maximum drawdown analysis

#### 5. Feature Analysis
- **Feature Importance**: Top predictive features
- **Correlation Matrix**: Feature relationships
- **Engineering Pipeline**: Feature creation process

#### 6. Performance Metrics
- **Returns Analysis**: Daily, monthly, yearly returns
- **Risk-Adjusted Returns**: Sharpe, Sortino, Calmar ratios
- **Drawdown Analysis**: Depth, duration, recovery
- **Rolling Metrics**: Time-varying performance

#### 7. Market Analysis
- **Regime Transitions**: Market state changes over time
- **Volatility Patterns**: Clustering and forecasting
- **Correlation Dynamics**: Asset correlation evolution

### Usage

1. **Run the ML Backtest**:
   ```bash
   python ml_real_world_backtest.py
   ```

2. **Fix ML Integration Issues** (if needed):
   ```bash
   python fix_ml_integration.py
   python test_ml_integration.py
   ```

3. **View Reports**:
   - Open `index.html` in a web browser
   - Navigate through the comprehensive report structure
   - All reports are interconnected with navigation

### Report Features

- **Interactive Charts**: Plotly-based visualizations
- **Responsive Design**: Works on all devices
- **Drill-Down Capability**: Click through for detailed analysis
- **Export Options**: Save as PDF or download raw data
- **Real-Time Updates**: Can be refreshed with new data

### Customization

The report template is designed to be extensible:
- Add new report sections by creating HTML files in appropriate directories
- Update `index.html` to include links to new reports
- Maintain consistent styling using the provided CSS framework
- Use the same navigation structure for consistency

### Data Sources

All reports are generated from:
- Real market data from Yahoo Finance
- ML model predictions on actual test data
- Backtest results with realistic execution simulation
- Comprehensive performance metrics