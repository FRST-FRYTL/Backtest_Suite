# SPX Multi-Timeframe Data Summary Report

**Generated:** 2025-07-15 13:37:36
**Symbol:** SPY (SPY ETF as S&P 500 proxy)

## Data Overview

### 1min - High-frequency trading data

- **Rows:** 4,696
- **Date Range:** 2025-07-08 to 2025-07-14
- **File:** `1min/SPY_1min_latest.csv`
- **Data Quality:**
  - High/Low Consistency: ✅
  - OHLC Consistency: ✅
  - Volume Valid: ✅
- **Date Gaps:** 927 missing (31.46%)

### 5min - Intraday momentum tracking

- **Rows:** 7,143
- **Date Range:** 2025-05-19 to 2025-07-14
- **File:** `5min/SPY_5min_latest.csv`
- **Data Quality:**
  - High/Low Consistency: ✅
  - OHLC Consistency: ✅
  - Volume Valid: ✅
- **Date Gaps:** 1731 missing (35.73%)

### 15min - Short-term trend analysis

- **Rows:** 2,382
- **Date Range:** 2025-05-19 to 2025-07-14
- **File:** `15min/SPY_15min_latest.csv`
- **Data Quality:**
  - High/Low Consistency: ✅
  - OHLC Consistency: ✅
  - Volume Valid: ✅
- **Date Gaps:** 589 missing (35.63%)

### 30min - Intraday swing trading

- **Rows:** 1,211
- **Date Range:** 2025-05-19 to 2025-07-14
- **File:** `30min/SPY_30min_latest.csv`
- **Data Quality:**
  - High/Low Consistency: ❌
  - OHLC Consistency: ❌
  - Volume Valid: ✅
- **Date Gaps:** 285 missing (33.33%)

### 1H - Daily trend confirmation

- **Rows:** 2,035
- **Date Range:** 2025-01-16 to 2025-07-14
- **File:** `1H/SPY_1H_latest.csv`
- **Data Quality:**
  - High/Low Consistency: ✅
  - OHLC Consistency: ✅
  - Volume Valid: ✅
- **Date Gaps:** 3127 missing (72.54%)

### 4H - Multi-day swing trading

- **Rows:** 2,188
- **Date Range:** 2024-07-15 to 2025-07-14
- **File:** `4H/SPY_4H_latest.csv`
- **Data Quality:**
  - High/Low Consistency: ❌
  - OHLC Consistency: ❌
  - Volume Valid: ✅
- **Date Gaps:** 0 missing (0.00%)

### 1D - Position trading and trend following

- **Rows:** 500
- **Date Range:** 2023-07-17 to 2025-07-14
- **File:** `1D/SPY_1D_latest.csv`
- **Data Quality:**
  - High/Low Consistency: ✅
  - OHLC Consistency: ✅
  - Volume Valid: ✅
- **Date Gaps:** 21 missing (4.03%)

## Usage Instructions

```python
import pandas as pd

# Load a specific timeframe
df_daily = pd.read_csv('data/SPX/1D/SPY_1D_latest.csv', index_col=0, parse_dates=True)

# Load multiple timeframes
timeframes = ['1min', '5min', '15min', '30min', '1H', '4H', '1D']
data = {}
for tf in timeframes:
    data[tf] = pd.read_csv(f'data/SPX/{tf}/SPY_{tf}_latest.csv', index_col=0, parse_dates=True)
```
