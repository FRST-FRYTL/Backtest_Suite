# Missing Lines Analysis - Coverage Gaps

## src/backtesting/order.py (95.7% coverage)
**Missing Lines: 65, 73, 94**

### Line 65
```python
return self.status == OrderStatus.FILLED
```
- Part of `is_filled()` method
- Simple status check - likely not called in tests

### Line 73
```python
return self.quantity - self.filled_quantity
```
- Part of `remaining_quantity()` method
- Calculates unfilled quantity - needs test coverage

### Line 94
```python
self.status = OrderStatus.PARTIAL
```
- Part of `fill()` method
- Sets partial fill status - edge case not tested

**Fix**: Add tests for:
1. `is_filled()` method with filled orders
2. `remaining_quantity()` for partially filled orders
3. Partial fill scenarios in `fill()` method

## src/indicators/vwma.py (67.2% coverage)
**Missing Lines: 166-202, 219-220, 240-266**

### Missing Methods/Sections:

1. **generate_signals() method (lines 166-202)**
   - Entire signal generation logic untested
   - Includes band touch/break signals
   - Cross above/below signals
   - Band squeeze/expansion signals

2. **calculate_percent_b() method (lines 219-220)**
   - %B indicator calculation untested
   - Measures position within bands

3. **volume_confirmation() method (lines 240-266)**
   - Volume-based signal confirmation untested
   - Bullish/bearish volume signals
   - Low volume move detection

**Fix**: Create comprehensive tests for:
1. Signal generation with various market conditions
2. %B calculation edge cases (when bands are narrow)
3. Volume confirmation signals with different volume patterns

## Priority Test Cases Needed

### For order.py:
```python
def test_order_is_filled():
    # Test filled order status
    
def test_remaining_quantity():
    # Test partial fill quantity calculation
    
def test_partial_fill_status():
    # Test partial fill sets correct status
```

### For vwma.py:
```python
def test_generate_signals_comprehensive():
    # Test all signal types
    
def test_calculate_percent_b():
    # Test %B calculation
    
def test_volume_confirmation_signals():
    # Test volume-based signals
```

## Summary
- **order.py**: Only 3 lines missing, easy to achieve 100%
- **vwma.py**: 21 lines missing, mostly in signal generation and volume analysis
- Both files are critical for backtesting functionality
- Achieving 100% coverage will require ~10-15 additional test cases