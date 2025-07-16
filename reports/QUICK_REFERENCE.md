# 🚀 SuperTrend AI SPX - Quick Reference Card

## 📊 Optimal Settings (Copy & Paste Ready)

### 🎯 Best Overall Configuration
```
ATR Length: 14
Factor Range Min: 1.0
Factor Range Max: 4.0
Performance Memory: 10
Cluster: Best
Min Signal Strength: 4
Stop Loss Type: ATR
Stop Loss ATR: 2.0
Take Profit Type: Risk/Reward
Risk/Reward Ratio: 2.5
```

## ⚡ Performance by Timeframe

| Timeframe | Sharpe | Return | Drawdown | Settings Adjustment |
|-----------|--------|--------|----------|-------------------|
| **Monthly** | 1.98 | 18.5% | -13.6% | Use defaults ✅ |
| **Weekly** | 1.74 | 17.4% | -15.2% | Signal Strength: 5 |
| **Daily** | 1.45 | 15.8% | -16.8% | Use defaults ✅ |
| **4-Hour** | 1.22 | 14.2% | -18.4% | ATR: 12, Strength: 3 |

## 🎮 Quick Commands

### TradingView Setup
```
1. Pine Editor → New
2. Paste supertrend_ai_optimized.pine
3. Add to Chart → Done! ✅
```

### Python Backtest
```python
# Quick test
python examples/supertrend_ai_demo.py

# Full analysis
python examples/run_multi_timeframe_analysis.py
```

### Fetch SPX Data
```python
python src/data/spx_multi_timeframe_fetcher.py
```

## 📈 Trading Rules

### Entry Conditions
✅ SuperTrend direction change  
✅ Signal strength ≥ 4  
✅ Volume > 1.2x average  
✅ Within trading hours (optional)  

### Exit Conditions
❌ SuperTrend reversal  
❌ Stop loss hit (2.0x ATR)  
✅ Take profit hit (2.5:1 R:R)  

## 🛡️ Risk Management

### Position Sizing
- **Conservative**: 50% of capital
- **Moderate**: 75% of capital
- **Aggressive**: 95% of capital
- **Kelly**: Dynamic (min 10 trades)

### Stop Loss Guidelines
- **Tight**: 1.5x ATR (scalping)
- **Standard**: 2.0x ATR (recommended)
- **Wide**: 2.5x ATR (trending markets)

## 🚨 Red Flags to Watch

1. **Signal Strength < 3**: Skip trade
2. **Low Volume**: Wait for confirmation
3. **Major News**: Tighten stops
4. **Drawdown > 15%**: Reduce size
5. **Consecutive Losses > 5**: Review settings

## 📱 Alert Template

```
//@version=5
alertcondition(longCondition, "Long Entry", 
"🟢 LONG: {{ticker}} @ {{close}}\nStrength: " + str.tostring(signalStrength))

alertcondition(shortCondition, "Short Entry", 
"🔴 SHORT: {{ticker}} @ {{close}}\nStrength: " + str.tostring(signalStrength))
```

## 💡 Pro Tips

1. **Start with daily timeframe** - Best risk/reward
2. **Paper trade 30 days minimum** - Verify performance
3. **Track slippage** - Adjust if > 0.1%
4. **Review weekly** - Market conditions change
5. **Scale gradually** - 25% → 50% → 75% → Full

## 📊 Performance Benchmarks

### Expected Monthly Metrics
- **Trades**: 8-12
- **Win Rate**: 60-65%
- **Avg Win**: +2.5%
- **Avg Loss**: -1.0%
- **Net Return**: +1.5%

### Warning Thresholds
- Win Rate < 55% ⚠️
- Drawdown > 20% 🚨
- Sharpe < 1.0 ❌

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| No signals | Lower strength to 3 |
| Too many signals | Increase strength to 5 |
| Large drawdowns | Reduce position size |
| Low win rate | Check timeframe match |

## 📞 Quick Links

- **Pine Script**: `/src/strategies/supertrend_ai_optimized.pine`
- **Usage Guide**: `/src/strategies/TRADINGVIEW_USAGE.md`
- **Full Report**: `/reports/SPX_SUPERTREND_AI_MASTER_REPORT.md`
- **Backtest Results**: `/reports/timeframe_analysis/`

---

**Remember**: Past performance ≠ Future results. Always manage risk!

*Last Updated: July 15, 2025 | Version: 1.0.0*