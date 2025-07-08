# Strategy Optimization Guide

This guide covers the strategy optimization features of the Backtest Suite.

## Overview

The optimization module provides several methods to find optimal strategy parameters:
- Grid Search
- Random Search
- Differential Evolution
- Walk-Forward Analysis

## Basic Optimization

### Grid Search

Exhaustively tests all parameter combinations:

```python
from src.optimization import StrategyOptimizer

# Define parameter grid
param_grid = {
    'rsi_period': [10, 14, 20, 30],
    'rsi_oversold': [20, 25, 30],
    'rsi_overbought': [70, 75, 80],
    'stop_loss': [0.02, 0.03, 0.05]
}

# Create optimizer
optimizer = StrategyOptimizer(
    data=data,
    strategy_builder=builder,
    optimization_metric="sharpe_ratio"
)

# Run grid search
results = optimizer.grid_search(param_grid, n_jobs=-1)

print(f"Best parameters: {results.best_params}")
print(f"Best score: {results.best_score}")
print(f"Total combinations tested: {results.n_iterations}")
```

### Random Search

More efficient for large parameter spaces:

```python
from scipy.stats import uniform, randint

# Define parameter distributions
param_distributions = {
    'rsi_period': randint(10, 30),
    'rsi_oversold': uniform(20, 15),  # 20-35
    'rsi_overbought': uniform(65, 15), # 65-80
    'stop_loss': uniform(0.02, 0.08)   # 0.02-0.10
}

# Run random search
results = optimizer.random_search(
    param_distributions,
    n_iter=1000,
    n_jobs=-1
)
```

### Differential Evolution

Global optimization using evolutionary algorithms:

```python
# Define parameter bounds
bounds = {
    'rsi_period': (10, 30),
    'rsi_oversold': (20, 35),
    'rsi_overbought': (65, 80),
    'stop_loss': (0.02, 0.10)
}

# Run differential evolution
results = optimizer.differential_evolution(
    bounds,
    population_size=50,
    generations=100
)
```

## Walk-Forward Analysis

Tests strategy robustness by optimizing on historical data and testing on future data:

```python
from src.optimization import WalkForwardAnalysis

# Configure walk-forward analysis
wfa = WalkForwardAnalysis(
    train_days=252,     # 1 year training
    test_days=63,       # 3 months testing
    step_days=21        # 1 month step
)

# Run analysis
wf_results = wfa.run(
    data=data,
    optimizer=optimizer,
    param_grid=param_grid
)

# Analyze results
print("In-Sample Performance:")
for window in wf_results.in_sample_results:
    print(f"  Window {window['window']}: {window['best_score']:.3f}")

print("\nOut-of-Sample Performance:")
for window in wf_results.out_of_sample_results:
    print(f"  Window {window['window']}: {window['score']:.3f}")

print(f"\nParameter Stability: {wf_results.parameter_stability:.3f}")
```

## Optimization Metrics

Available metrics for optimization:

- **sharpe_ratio**: Risk-adjusted returns (default)
- **total_return**: Absolute returns
- **calmar_ratio**: Return / Max Drawdown
- **sortino_ratio**: Downside risk-adjusted returns
- **profit_factor**: Gross profit / Gross loss
- **win_rate**: Percentage of winning trades
- **expectancy**: Average profit per trade

```python
# Use different optimization metric
optimizer = StrategyOptimizer(
    data=data,
    strategy_builder=builder,
    optimization_metric="calmar_ratio"
)
```

## Custom Objective Functions

Create custom optimization objectives:

```python
def custom_objective(backtest_results):
    """Custom objective combining multiple metrics."""
    metrics = backtest_results['performance']
    
    # Weighted combination of metrics
    score = (
        0.4 * metrics['sharpe_ratio'] +
        0.3 * metrics['calmar_ratio'] +
        0.2 * metrics['win_rate'] +
        0.1 * (1 - metrics['max_drawdown'])
    )
    return score

optimizer = StrategyOptimizer(
    data=data,
    strategy_builder=builder,
    optimization_metric=custom_objective
)
```

## Parallel Processing

Leverage multiple cores for faster optimization:

```python
# Use all available cores
results = optimizer.grid_search(param_grid, n_jobs=-1)

# Use specific number of cores
results = optimizer.grid_search(param_grid, n_jobs=4)

# Disable parallel processing
results = optimizer.grid_search(param_grid, n_jobs=1)
```

## Constraints and Filters

Add constraints to parameter optimization:

```python
# Define constraints
def constraint_function(params):
    """Ensure oversold < overbought."""
    return params['rsi_oversold'] < params['rsi_overbought'] - 20

# Apply constraints during optimization
results = optimizer.grid_search(
    param_grid,
    constraints=[constraint_function]
)
```

## Overfitting Prevention

### 1. Out-of-Sample Testing

Always reserve data for final validation:

```python
# Split data
train_size = int(len(data) * 0.7)
train_data = data[:train_size]
test_data = data[train_size:]

# Optimize on training data
optimizer = StrategyOptimizer(data=train_data, ...)
results = optimizer.grid_search(param_grid)

# Validate on test data
engine = BacktestEngine()
test_results = engine.run(test_data, results.best_strategy)
```

### 2. Cross-Validation

Use time series cross-validation:

```python
from src.optimization import TimeSeriesCV

# Configure cross-validation
tscv = TimeSeriesCV(
    n_splits=5,
    test_size=63,  # 3 months
    gap=5          # 5 days gap
)

# Run cross-validated optimization
cv_results = optimizer.cross_validate(
    param_grid,
    cv=tscv
)

print(f"Mean CV Score: {cv_results.mean_score:.3f}")
print(f"Std CV Score: {cv_results.std_score:.3f}")
```

### 3. Parameter Stability

Check if parameters are stable across different periods:

```python
# Analyze parameter stability
stability_report = wf_results.analyze_parameter_stability()

for param, stability in stability_report.items():
    print(f"{param}: {stability:.1%} stable")
```

## Visualization

Visualize optimization results:

```python
from src.visualization import plot_optimization_results

# Plot parameter importance
fig = plot_optimization_results(
    results,
    plot_type='parameter_importance'
)
fig.show()

# Plot optimization surface (2D)
fig = plot_optimization_results(
    results,
    plot_type='surface',
    x_param='rsi_period',
    y_param='stop_loss'
)
fig.show()

# Plot convergence (for evolutionary algorithms)
fig = plot_optimization_results(
    results,
    plot_type='convergence'
)
fig.show()
```

## Best Practices

1. **Start Simple**: Begin with few parameters and expand gradually
2. **Use Domain Knowledge**: Set reasonable parameter bounds
3. **Multiple Metrics**: Don't optimize for a single metric
4. **Robustness Testing**: Always validate on out-of-sample data
5. **Parameter Stability**: Prefer stable parameters over absolute best
6. **Transaction Costs**: Include realistic costs in optimization

## Example: Complete Optimization Workflow

```python
import asyncio
from datetime import datetime, timedelta
from src.data import StockDataFetcher
from src.indicators import RSI, BollingerBands
from src.strategies import StrategyBuilder
from src.optimization import StrategyOptimizer, WalkForwardAnalysis

async def optimize_strategy():
    # 1. Fetch data (2 years)
    fetcher = StockDataFetcher()
    data = await fetcher.fetch(
        "AAPL",
        datetime.now() - timedelta(days=730),
        datetime.now()
    )
    
    # 2. Calculate indicators
    data['rsi'] = RSI(14).calculate(data)
    bb_data = BollingerBands(20).calculate(data)
    data = data.join(bb_data)
    
    # 3. Create strategy builder function
    def create_strategy(params):
        builder = StrategyBuilder("Optimized RSI Strategy")
        
        # Use parameters
        rsi_calc = RSI(params['rsi_period']).calculate(data)
        data['rsi_opt'] = rsi_calc
        
        # Entry rules with parameters
        builder.add_entry_rule(
            f"rsi_opt < {params['rsi_oversold']} and close < bb_lower"
        )
        
        # Exit rules with parameters
        builder.add_exit_rule(
            f"rsi_opt > {params['rsi_overbought']}"
        )
        
        # Risk management with parameters
        builder.set_risk_management(
            stop_loss=params['stop_loss'],
            max_positions=3
        )
        
        return builder.build()
    
    # 4. Define parameter space
    param_grid = {
        'rsi_period': [10, 14, 20],
        'rsi_oversold': [25, 30, 35],
        'rsi_overbought': [65, 70, 75],
        'stop_loss': [0.03, 0.05, 0.07]
    }
    
    # 5. Run walk-forward optimization
    wfa = WalkForwardAnalysis(
        train_days=252,
        test_days=63,
        step_days=21
    )
    
    optimizer = StrategyOptimizer(
        data=data,
        strategy_builder=create_strategy,
        optimization_metric="sharpe_ratio"
    )
    
    results = wfa.run(data, optimizer, param_grid)
    
    # 6. Analyze results
    print("Walk-Forward Analysis Results:")
    print(f"Average In-Sample Sharpe: {results.avg_in_sample:.3f}")
    print(f"Average Out-Sample Sharpe: {results.avg_out_sample:.3f}")
    print(f"Efficiency: {results.efficiency:.1%}")
    
    return results

# Run optimization
results = asyncio.run(optimize_strategy())
```

## Troubleshooting

### Memory Issues

For large parameter spaces:

```python
# Use iterative optimization
optimizer.grid_search(
    param_grid,
    batch_size=1000,  # Process in batches
    cache_results=False  # Don't cache all results
)
```

### Slow Optimization

Speed up optimization:

```python
# 1. Use random search instead of grid search
# 2. Reduce data granularity for initial search
# 3. Use faster indicators
# 4. Implement early stopping

def early_stopping_callback(iteration, current_best):
    """Stop if no improvement for 50 iterations."""
    if iteration - current_best['iteration'] > 50:
        return True
    return False

results = optimizer.differential_evolution(
    bounds,
    callback=early_stopping_callback
)
```