# Immediate Coverage Action Plan - Non-ML Modules

## 🚨 Critical Issues to Fix First

### Import Errors Blocking Tests
These import errors are preventing tests from running and must be fixed immediately:

1. **`SignalFilter` missing from `src.strategies.signals`**
   - Check if class exists with different name
   - May need to create or import from correct module

2. **`BacktestStrategy` missing from `src.backtesting.strategy`**
   - Likely should be `Strategy` or `BaseStrategy`
   - Update test imports

3. **`Rebalancer` missing from `src.portfolio.rebalancer`**
   - Check actual class name in rebalancer.py
   - Fix import statement in tests

4. **`ChartType` missing from `src.reporting.visualization_types`**
   - Module may not exist or enum has different name
   - Update imports or create missing types

## 📋 Immediate Action Items (Day 1)

### 1. Quick Coverage Wins (2-3 hours)

#### A. Complete order.py Coverage (95.7% → 100%)
**File**: `src/backtesting/order.py`
**Missing Lines**: 65, 73, 94

```python
# Create: tests/coverage/test_order_complete.py

def test_order_is_filled():
    """Cover line 65: is_filled() method"""
    from src.backtesting.order import Order, OrderType, OrderSide, OrderStatus
    
    # Test filled order
    order = Order(OrderType.MARKET, OrderSide.BUY, 100)
    order.status = OrderStatus.FILLED
    order.filled_quantity = 100
    assert order.is_filled() is True
    
    # Test unfilled order
    order2 = Order(OrderType.LIMIT, OrderSide.SELL, 50)
    assert order2.is_filled() is False

def test_remaining_quantity():
    """Cover line 73: remaining_quantity calculation"""
    from src.backtesting.order import Order, OrderType, OrderSide
    
    order = Order(OrderType.LIMIT, OrderSide.BUY, 100, price=50.0)
    order.filled_quantity = 60
    assert order.remaining_quantity() == 40
    
    # Test fully filled
    order.filled_quantity = 100
    assert order.remaining_quantity() == 0

def test_partial_fill_status():
    """Cover line 94: partial fill status setting"""
    from src.backtesting.order import Order, OrderType, OrderSide, OrderStatus
    from datetime import datetime
    
    order = Order(OrderType.LIMIT, OrderSide.BUY, 100, price=50.0)
    # Partial fill should set PARTIAL status
    order.fill(60, 50.0, datetime.now())
    assert order.status == OrderStatus.PARTIAL
    assert order.filled_quantity == 60
```

#### B. Complete vwma.py Coverage (67.2% → 100%)
**File**: `src/indicators/vwma.py`
**Missing Lines**: 166-202 (generate_signals), 219-220 (calculate_percent_b), 240-266 (volume_confirmation)

```python
# Create: tests/coverage/test_vwma_complete.py

def test_generate_signals_all_types():
    """Cover lines 166-202: All signal generation logic"""
    import pandas as pd
    import numpy as np
    from src.indicators.vwma import VWMAIndicator
    
    # Create test data with specific patterns
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 5000000, 100),
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
    }, index=dates)
    
    vwma = VWMAIndicator(period=20)
    vwma.calculate(data)
    signals = vwma.generate_signals(data)
    
    # Test all signal types are generated
    expected_signals = [
        'upper_band_touch', 'lower_band_touch',
        'upper_band_break', 'lower_band_break',
        'cross_above_vwma', 'cross_below_vwma',
        'band_squeeze', 'band_expansion'
    ]
    
    for signal in expected_signals:
        assert signal in signals
        assert isinstance(signals[signal], pd.Series)

def test_calculate_percent_b():
    """Cover lines 219-220: %B calculation"""
    import pandas as pd
    import numpy as np
    from src.indicators.vwma import VWMAIndicator
    
    # Create test data
    data = pd.DataFrame({
        'close': [100, 102, 98, 105, 103],
        'volume': [1000000, 1500000, 800000, 2000000, 1200000]
    })
    
    vwma = VWMAIndicator(period=3)
    vwma.calculate(data)
    percent_b = vwma.calculate_percent_b(data)
    
    # Verify %B calculation
    assert len(percent_b) == len(data)
    assert all(-0.5 <= val <= 1.5 for val in percent_b.dropna())

def test_volume_confirmation_signals():
    """Cover lines 240-266: Volume confirmation logic"""
    import pandas as pd
    import numpy as np
    from src.indicators.vwma import VWMAIndicator
    
    # Create data with specific volume patterns
    data = pd.DataFrame({
        'close': [100, 105, 110, 108, 112, 115, 113, 118],
        'volume': [1000000, 2000000, 3000000, 500000, 
                   4000000, 5000000, 600000, 6000000]
    })
    
    vwma = VWMAIndicator(period=3)
    vwma.calculate(data)
    volume_signals = vwma.volume_confirmation(data)
    
    # Test all volume signal types
    assert 'bullish_volume' in volume_signals
    assert 'bearish_volume' in volume_signals
    assert 'low_volume_move' in volume_signals
    
    # Verify signal logic
    # High volume + price up = bullish_volume
    # High volume + price down = bearish_volume
    # Price move + low volume = low_volume_move
```

### 2. Fix Import Errors (1-2 hours)

#### Investigation Script
```python
# Create: tests/fix_imports.py

import os
import importlib
import inspect

# Check actual class names in modules
modules_to_check = [
    ('src.strategies.signals', 'SignalFilter'),
    ('src.backtesting.strategy', 'BacktestStrategy'),
    ('src.portfolio.rebalancer', 'Rebalancer'),
    ('src.reporting.visualization_types', 'ChartType'),
]

for module_name, expected_class in modules_to_check:
    try:
        module = importlib.import_module(module_name)
        available_classes = [name for name, obj in inspect.getmembers(module) 
                           if inspect.isclass(obj)]
        print(f"\n{module_name}:")
        print(f"  Expected: {expected_class}")
        print(f"  Available: {available_classes}")
        
        # Suggest corrections
        if expected_class not in available_classes:
            similar = [c for c in available_classes if expected_class.lower() in c.lower()]
            if similar:
                print(f"  Suggestion: Use {similar[0]} instead of {expected_class}")
    except ImportError as e:
        print(f"\n{module_name}: MODULE NOT FOUND - {e}")
```

### 3. Priority Non-ML Module Tests (Day 1-2)

#### A. Analysis Module Tests (0% → 60%)
```python
# Create: tests/coverage/test_analysis_comprehensive.py

class TestBaselineComparisons:
    """Test baseline_comparisons.py functionality"""
    
    def test_buy_and_hold_comparison(self):
        """Test buy-and-hold baseline calculation"""
        pass
    
    def test_risk_adjusted_returns(self):
        """Test Sharpe, Sortino calculations"""
        pass
    
    def test_drawdown_comparison(self):
        """Test drawdown metrics vs baseline"""
        pass

class TestEnhancedTradeTracker:
    """Test enhanced_trade_tracker.py functionality"""
    
    def test_trade_logging(self):
        """Test trade entry/exit logging"""
        pass
    
    def test_trade_statistics(self):
        """Test win rate, avg profit calculations"""
        pass
    
    def test_trade_duration_analysis(self):
        """Test holding period analytics"""
        pass

class TestPerformanceAttribution:
    """Test performance_attribution.py functionality"""
    
    def test_return_attribution(self):
        """Test return decomposition"""
        pass
    
    def test_factor_analysis(self):
        """Test performance factor identification"""
        pass
```

#### B. Optimization Module Tests (0% → 50%)
```python
# Create: tests/coverage/test_optimization_comprehensive.py

class TestOptimizer:
    """Test optimizer.py core functionality"""
    
    def test_parameter_grid_generation(self):
        """Test parameter space creation"""
        pass
    
    def test_objective_function(self):
        """Test optimization objectives"""
        pass
    
    def test_parallel_optimization(self):
        """Test parallel processing"""
        pass

class TestWalkForward:
    """Test walk_forward.py functionality"""
    
    def test_window_generation(self):
        """Test train/test window creation"""
        pass
    
    def test_out_of_sample_validation(self):
        """Test OOS performance tracking"""
        pass
```

### 4. Core Engine Enhancement Tests (Day 2-3)

#### Backtesting Engine Tests (16% → 70%)
```python
# Create: tests/coverage/test_engine_comprehensive.py

class TestBacktestingEngine:
    """Comprehensive engine testing"""
    
    def test_engine_initialization(self):
        """Test engine setup and configuration"""
        pass
    
    def test_event_processing_pipeline(self):
        """Test market, signal, order, fill events"""
        pass
    
    def test_state_management(self):
        """Test position and portfolio state"""
        pass
    
    def test_error_handling(self):
        """Test graceful error recovery"""
        pass
    
    def test_multi_asset_support(self):
        """Test multiple symbol handling"""
        pass
```

## 📊 Coverage Tracking Dashboard

### Current State (Non-ML Only)
```
Module                  Current    Target    Priority
─────────────────────────────────────────────────────
Order                   95.7%      100%      🟢 Quick Win
VWMA                    67.2%      100%      🟢 Quick Win
Analysis                0%         60%       🔴 Critical
Optimization            0%         50%       🔴 Critical
Backtesting Engine      16%        70%       🔴 Critical
Portfolio Mgmt          26%        70%       🟡 Important
Data Pipeline           23%        60%       🟡 Important
Indicators              17%        50%       🟡 Important
Visualization           0%         40%       🟡 Important
Reporting               15%        50%       🟡 Important
```

### Daily Goals
- **Day 1**: Fix imports, complete order.py and vwma.py (Quick wins)
- **Day 2**: Analysis module to 30%, Optimization to 25%
- **Day 3**: Backtesting engine to 40%, Portfolio to 40%
- **Day 4**: Data pipeline to 40%, Indicators to 30%
- **Day 5**: Comprehensive integration tests

## 🚀 Execution Commands

```bash
# Step 1: Run import fix script
python tests/fix_imports.py

# Step 2: Run quick win tests
pytest tests/coverage/test_order_complete.py -v
pytest tests/coverage/test_vwma_complete.py -v

# Step 3: Run coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing

# Step 4: Focus on specific modules
pytest tests/coverage/test_analysis_comprehensive.py --cov=src.analysis
pytest tests/coverage/test_optimization_comprehensive.py --cov=src.optimization

# Step 5: Generate updated coverage report
python -m coverage html
```

## Success Criteria
1. ✅ All import errors fixed
2. ✅ Order module at 100% coverage
3. ✅ VWMA module at 100% coverage
4. ✅ Analysis module >30% coverage
5. ✅ No failing tests due to import issues
6. ✅ Coverage trend graph showing improvement

---

**Note**: This plan focuses exclusively on non-ML modules. ML coverage will be addressed only after achieving 70%+ coverage on core functionality.