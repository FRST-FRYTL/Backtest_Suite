# Strategy Optimization Guide

Learn how to optimize your trading strategies effectively while avoiding common pitfalls like overfitting.

## Table of Contents

- [Introduction to Optimization](#introduction-to-optimization)
- [Optimization Methods](#optimization-methods)
- [Parameter Selection](#parameter-selection)
- [Avoiding Overfitting](#avoiding-overfitting)
- [Walk-Forward Analysis](#walk-forward-analysis)
- [Performance Metrics](#performance-metrics)
- [Advanced Techniques](#advanced-techniques)
- [Best Practices](#best-practices)
- [Common Pitfalls](#common-pitfalls)

## Introduction to Optimization

### What is Strategy Optimization?

Strategy optimization is the process of finding the best parameters for your trading strategy to maximize performance. However, it's crucial to balance between:

- **Performance**: Maximizing returns and risk-adjusted metrics
- **Robustness**: Ensuring the strategy works in different market conditions
- **Simplicity**: Avoiding over-complex rules that don't generalize

### The Optimization Trade-off

```
Historical Performance ←→ Future Performance
      ↑                        ↑
   Overfitting            Robustness
```

## Optimization Methods

### 1. Grid Search

Exhaustive search through all parameter combinations.

```python
from src.optimization import GridSearchOptimizer

# Define parameter grid
param_grid = {
    'rsi_period': [10, 14, 20, 30],
    'rsi_oversold': [20, 25, 30, 35],
    'rsi_overbought': [65, 70, 75, 80],
    'stop_loss': [0.01, 0.02, 0.03, 0.04]
}

# Run optimization
optimizer = GridSearchOptimizer(objective='sharpe_ratio', n_jobs=-1)
results = optimizer.optimize(
    strategy_class=RSIStrategy,
    parameter_grid=param_grid,
    data=data,
    backtest_kwargs={'initial_capital': 100000}
)

print(f"Best parameters: {results['best_params']}")
print(f"Best Sharpe ratio: {results['best_score']:.2f}")
```

#### Pros and Cons

**Pros:**
- Guaranteed to find the best combination in the grid
- Easy to understand and implement
- Can visualize parameter surfaces

**Cons:**
- Computationally expensive (grows exponentially)
- Requires discrete parameter values
- May miss optimal values between grid points

### 2. Random Search

Randomly sample parameter combinations.

```python
from src.optimization import RandomSearchOptimizer
from scipy.stats import randint, uniform

# Define parameter distributions
param_distributions = {
    'rsi_period': randint(10, 30),
    'rsi_oversold': randint(20, 35),
    'rsi_overbought': randint(65, 80),
    'stop_loss': uniform(0.01, 0.04)
}

# Run optimization
optimizer = RandomSearchOptimizer(
    objective='sharpe_ratio',
    n_iter=1000,  # Number of random samples
    n_jobs=-1
)

results = optimizer.optimize(
    strategy_class=RSIStrategy,
    parameter_distributions=param_distributions,
    data=data
)
```

#### When to Use Random Search

- Large parameter spaces
- Continuous parameters
- Limited computational budget
- Initial exploration phase

### 3. Differential Evolution

Population-based optimization algorithm.

```python
from src.optimization import DifferentialEvolution

# Define parameter bounds
param_bounds = {
    'rsi_period': (10, 30),
    'rsi_oversold': (20, 35),
    'rsi_overbought': (65, 80),
    'stop_loss': (0.01, 0.04),
    'take_profit': (0.02, 0.10)
}

# Run optimization
optimizer = DifferentialEvolution(
    objective='sharpe_ratio',
    population_size=50,
    generations=100,
    mutation_factor=0.8,
    crossover_prob=0.7
)

results = optimizer.optimize(
    strategy_class=RSIStrategy,
    parameter_bounds=param_bounds,
    data=data
)

# Access convergence history
convergence = results['convergence_history']
plt.plot(convergence)
plt.xlabel('Generation')
plt.ylabel('Best Sharpe Ratio')
plt.title('Optimization Convergence')
```

### 4. Bayesian Optimization

Uses probabilistic model to guide search.

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Define search space
search_space = {
    'rsi_period': Integer(10, 30),
    'rsi_oversold': Integer(20, 35),
    'rsi_overbought': Integer(65, 80),
    'stop_loss': Real(0.01, 0.04),
    'take_profit': Real(0.02, 0.10)
}

# Custom objective function
def objective(params):
    strategy = RSIStrategy(**params)
    engine = BacktestEngine()
    results = engine.run(data, strategy)
    return -results['performance']['sharpe_ratio']  # Minimize negative Sharpe

# Run Bayesian optimization
from skopt import gp_minimize

result = gp_minimize(
    func=objective,
    dimensions=list(search_space.values()),
    n_calls=100,
    acq_func='EI'  # Expected Improvement
)
```

## Parameter Selection

### Identifying Key Parameters

Not all parameters are equally important. Focus on:

1. **High Impact Parameters**: Those that significantly affect performance
2. **Stable Parameters**: Those that work across different periods
3. **Logical Parameters**: Those with clear economic reasoning

```python
# Parameter sensitivity analysis
def sensitivity_analysis(strategy_class, base_params, param_name, param_range, data):
    results = []
    
    for value in param_range:
        params = base_params.copy()
        params[param_name] = value
        
        strategy = strategy_class(**params)
        engine = BacktestEngine()
        result = engine.run(data, strategy)
        
        results.append({
            'value': value,
            'sharpe': result['performance']['sharpe_ratio'],
            'return': result['performance']['total_return'],
            'max_dd': result['performance']['max_drawdown']
        })
    
    return pd.DataFrame(results)

# Analyze RSI period sensitivity
sensitivity = sensitivity_analysis(
    RSIStrategy,
    base_params={'rsi_period': 14, 'rsi_oversold': 30},
    param_name='rsi_period',
    param_range=range(5, 50, 5),
    data=data
)

# Plot results
plt.plot(sensitivity['value'], sensitivity['sharpe'])
plt.xlabel('RSI Period')
plt.ylabel('Sharpe Ratio')
plt.title('Parameter Sensitivity Analysis')
```

### Parameter Constraints

Define logical constraints between parameters:

```yaml
# optimization_config.yaml
parameters:
  rsi_oversold:
    min: 20
    max: 40
  
  rsi_overbought:
    min: 60
    max: 80

constraints:
  - "rsi_overbought > rsi_oversold + 20"  # Maintain minimum gap
  - "stop_loss < take_profit"              # Risk/reward constraint
  - "position_size <= 0.25"                # Maximum position constraint
```

## Avoiding Overfitting

### 1. Out-of-Sample Testing

```python
# Split data into train/test sets
train_end = '2022-12-31'
test_start = '2023-01-01'

train_data = data[:train_end]
test_data = data[test_start:]

# Optimize on training data
optimizer = GridSearchOptimizer(objective='sharpe_ratio')
results = optimizer.optimize(
    strategy_class=RSIStrategy,
    parameter_grid=param_grid,
    data=train_data
)

# Test on out-of-sample data
best_strategy = RSIStrategy(**results['best_params'])
engine = BacktestEngine()
test_results = engine.run(test_data, best_strategy)

print(f"Train Sharpe: {results['best_score']:.2f}")
print(f"Test Sharpe: {test_results['performance']['sharpe_ratio']:.2f}")

# Calculate degradation
degradation = (results['best_score'] - test_results['performance']['sharpe_ratio']) / results['best_score']
print(f"Performance degradation: {degradation:.1%}")
```

### 2. Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv(strategy_class, params, data, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    
    for train_idx, test_idx in tscv.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # Run backtest on fold
        strategy = strategy_class(**params)
        engine = BacktestEngine()
        
        fold_result = engine.run(test_data, strategy)
        results.append(fold_result['performance']['sharpe_ratio'])
    
    return {
        'mean_sharpe': np.mean(results),
        'std_sharpe': np.std(results),
        'all_folds': results
    }
```

### 3. Parameter Stability

Check if parameters work across different time periods:

```python
def stability_test(strategy_class, params, data, window_size=252):
    windows = []
    results = []
    
    for i in range(0, len(data) - window_size, window_size // 2):
        window_data = data.iloc[i:i + window_size]
        
        strategy = strategy_class(**params)
        engine = BacktestEngine()
        result = engine.run(window_data, strategy)
        
        windows.append(window_data.index[0])
        results.append(result['performance']['sharpe_ratio'])
    
    # Calculate stability metrics
    stability_score = 1 - (np.std(results) / np.mean(results))
    
    return {
        'windows': windows,
        'sharpe_ratios': results,
        'stability_score': stability_score,
        'consistent': all(r > 0 for r in results)
    }
```

## Walk-Forward Analysis

### Implementation

```python
def walk_forward_optimization(data, strategy_class, param_grid, 
                            in_sample_periods=252, out_sample_periods=63,
                            optimization_metric='sharpe_ratio'):
    results = []
    
    for i in range(in_sample_periods, len(data) - out_sample_periods, out_sample_periods):
        # In-sample period
        in_sample_data = data.iloc[i - in_sample_periods:i]
        
        # Optimize parameters
        optimizer = GridSearchOptimizer(objective=optimization_metric)
        opt_result = optimizer.optimize(
            strategy_class=strategy_class,
            parameter_grid=param_grid,
            data=in_sample_data
        )
        
        # Out-of-sample period
        out_sample_data = data.iloc[i:i + out_sample_periods]
        
        # Test optimized strategy
        strategy = strategy_class(**opt_result['best_params'])
        engine = BacktestEngine()
        oos_result = engine.run(out_sample_data, strategy)
        
        results.append({
            'period_start': out_sample_data.index[0],
            'period_end': out_sample_data.index[-1],
            'best_params': opt_result['best_params'],
            'in_sample_sharpe': opt_result['best_score'],
            'out_sample_sharpe': oos_result['performance']['sharpe_ratio'],
            'out_sample_return': oos_result['performance']['total_return']
        })
    
    return results

# Run walk-forward analysis
wf_results = walk_forward_optimization(
    data=data,
    strategy_class=RSIStrategy,
    param_grid=param_grid,
    in_sample_periods=252,  # 1 year
    out_sample_periods=63   # 3 months
)

# Analyze results
df_results = pd.DataFrame(wf_results)
print(f"Average out-of-sample Sharpe: {df_results['out_sample_sharpe'].mean():.2f}")
print(f"Win rate: {(df_results['out_sample_return'] > 0).mean():.1%}")
```

### Visualizing Walk-Forward Results

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: In-sample vs Out-of-sample Sharpe
axes[0].plot(df_results['period_start'], df_results['in_sample_sharpe'], 
             label='In-sample', marker='o')
axes[0].plot(df_results['period_start'], df_results['out_sample_sharpe'], 
             label='Out-of-sample', marker='s')
axes[0].set_ylabel('Sharpe Ratio')
axes[0].legend()
axes[0].set_title('Walk-Forward Sharpe Ratios')

# Plot 2: Parameter evolution
param_evolution = pd.DataFrame([r['best_params'] for r in wf_results])
for col in param_evolution.columns:
    axes[1].plot(df_results['period_start'], param_evolution[col], 
                 label=col, marker='o')
axes[1].set_ylabel('Parameter Value')
axes[1].legend()
axes[1].set_title('Parameter Evolution')

# Plot 3: Cumulative returns
cumulative_returns = (1 + df_results['out_sample_return'] / 100).cumprod()
axes[2].plot(df_results['period_start'], cumulative_returns)
axes[2].set_ylabel('Cumulative Return')
axes[2].set_title('Walk-Forward Cumulative Performance')

plt.tight_layout()
plt.show()
```

## Performance Metrics

### Choosing the Right Objective

Different objectives lead to different optimal parameters:

```python
# Compare different optimization objectives
objectives = ['sharpe_ratio', 'total_return', 'calmar_ratio', 'sortino_ratio']
objective_results = {}

for objective in objectives:
    optimizer = GridSearchOptimizer(objective=objective)
    result = optimizer.optimize(
        strategy_class=RSIStrategy,
        parameter_grid=param_grid,
        data=data
    )
    objective_results[objective] = result

# Compare results
comparison = pd.DataFrame({
    obj: {
        'best_params': res['best_params'],
        'sharpe': res['best_score'] if obj == 'sharpe_ratio' else 
                  calculate_sharpe(res['best_params'], data),
        'return': calculate_return(res['best_params'], data),
        'max_dd': calculate_max_dd(res['best_params'], data)
    }
    for obj, res in objective_results.items()
})
```

### Custom Objective Functions

```python
def custom_objective(results):
    """
    Custom objective that balances multiple metrics
    """
    sharpe = results['performance']['sharpe_ratio']
    max_dd = results['performance']['max_drawdown']
    win_rate = results['performance']['win_rate']
    
    # Penalize extreme drawdowns
    if max_dd > 0.20:  # 20% drawdown
        return -999
    
    # Weighted combination
    score = (
        0.5 * sharpe +                    # 50% weight on Sharpe
        0.3 * (win_rate - 0.5) * 10 +     # 30% on win rate above 50%
        0.2 * (1 - max_dd) * 5            # 20% on low drawdown
    )
    
    return score

# Use custom objective
optimizer = GridSearchOptimizer(objective=custom_objective)
```

### Multi-Objective Optimization

```python
from scipy.optimize import differential_evolution
import numpy as np

def pareto_optimization(strategy_class, param_bounds, data, n_objectives=2):
    """
    Find Pareto-optimal solutions for multiple objectives
    """
    def multi_objective(params):
        # Convert array to dict
        param_dict = {
            name: params[i] 
            for i, name in enumerate(param_bounds.keys())
        }
        
        strategy = strategy_class(**param_dict)
        engine = BacktestEngine()
        results = engine.run(data, strategy)
        
        # Return multiple objectives (to minimize)
        return [
            -results['performance']['sharpe_ratio'],     # Maximize Sharpe
            results['performance']['max_drawdown'],      # Minimize drawdown
        ]
    
    # Convert bounds to list format
    bounds = [(low, high) for low, high in param_bounds.values()]
    
    # Run multi-objective optimization
    pareto_solutions = []
    for _ in range(100):  # Generate 100 solutions
        result = differential_evolution(
            lambda x: sum(multi_objective(x)),  # Scalarize for DE
            bounds,
            maxiter=50
        )
        
        objectives = multi_objective(result.x)
        pareto_solutions.append({
            'params': result.x,
            'sharpe': -objectives[0],
            'drawdown': objectives[1]
        })
    
    return pareto_solutions

# Plot Pareto frontier
solutions = pareto_optimization(RSIStrategy, param_bounds, data)
sharpes = [s['sharpe'] for s in solutions]
drawdowns = [s['drawdown'] for s in solutions]

plt.scatter(drawdowns, sharpes)
plt.xlabel('Max Drawdown')
plt.ylabel('Sharpe Ratio')
plt.title('Pareto Frontier: Sharpe vs Drawdown')
```

## Advanced Techniques

### 1. Adaptive Parameter Optimization

Parameters that adjust based on market conditions:

```python
class AdaptiveStrategy:
    def __init__(self, base_params, adaptation_rules):
        self.base_params = base_params
        self.adaptation_rules = adaptation_rules
        self.current_params = base_params.copy()
    
    def update_parameters(self, market_data):
        """Update parameters based on current market conditions"""
        
        # Calculate market regime indicators
        volatility = market_data['close'].pct_change().rolling(20).std()
        trend_strength = abs(market_data['close'].rolling(50).mean() - 
                           market_data['close'].rolling(200).mean())
        
        # Apply adaptation rules
        if volatility.iloc[-1] > volatility.rolling(100).mean().iloc[-1]:
            # High volatility: wider stops, smaller positions
            self.current_params['stop_loss'] = self.base_params['stop_loss'] * 1.5
            self.current_params['position_size'] = self.base_params['position_size'] * 0.7
        
        if trend_strength.iloc[-1] > trend_strength.rolling(100).mean().iloc[-1]:
            # Strong trend: relax entry conditions
            self.current_params['rsi_oversold'] = self.base_params['rsi_oversold'] + 5
    
    def get_current_params(self):
        return self.current_params
```

### 2. Ensemble Parameter Sets

Using multiple parameter sets simultaneously:

```python
class EnsembleStrategy:
    def __init__(self, strategy_class, parameter_sets, weights=None):
        self.strategies = [
            strategy_class(**params) for params in parameter_sets
        ]
        self.weights = weights or [1/len(parameter_sets)] * len(parameter_sets)
    
    def generate_signals(self, data):
        """Combine signals from multiple parameter sets"""
        signals = []
        
        for strategy, weight in zip(self.strategies, self.weights):
            signal = strategy.generate_signals(data)
            signals.append(signal * weight)
        
        # Weighted average of signals
        combined_signal = sum(signals)
        
        # Generate trades when consensus is strong
        return combined_signal > 0.6  # 60% agreement threshold

# Example: Top 3 parameter sets from optimization
top_params = [
    {'rsi_period': 14, 'rsi_oversold': 30},
    {'rsi_period': 20, 'rsi_oversold': 25},
    {'rsi_period': 10, 'rsi_oversold': 35}
]

ensemble = EnsembleStrategy(RSIStrategy, top_params)
```

### 3. Meta-Learning Optimization

Learning from optimization history:

```python
class MetaOptimizer:
    def __init__(self):
        self.optimization_history = []
    
    def learn_from_history(self):
        """Analyze past optimizations to guide future searches"""
        if len(self.optimization_history) < 10:
            return None
        
        # Extract patterns
        df = pd.DataFrame(self.optimization_history)
        
        # Find parameter ranges that consistently perform well
        good_results = df[df['out_sample_sharpe'] > df['out_sample_sharpe'].quantile(0.75)]
        
        # Narrow search space based on successful parameters
        refined_bounds = {}
        for param in good_results['params'][0].keys():
            values = [r[param] for r in good_results['params']]
            refined_bounds[param] = (
                np.percentile(values, 25),
                np.percentile(values, 75)
            )
        
        return refined_bounds
    
    def optimize_with_learning(self, strategy_class, initial_bounds, data):
        # Use learned bounds if available
        bounds = self.learn_from_history() or initial_bounds
        
        # Run optimization
        optimizer = DifferentialEvolution(objective='sharpe_ratio')
        result = optimizer.optimize(strategy_class, bounds, data)
        
        # Store result for future learning
        self.optimization_history.append({
            'params': result['best_params'],
            'in_sample_sharpe': result['best_score'],
            'timestamp': pd.Timestamp.now()
        })
        
        return result
```

## Best Practices

### 1. Start Simple, Add Complexity Gradually

```python
# Level 1: Single parameter
simple_grid = {'rsi_period': [10, 14, 20, 30]}

# Level 2: Core parameters
medium_grid = {
    'rsi_period': [10, 14, 20],
    'rsi_oversold': [25, 30, 35]
}

# Level 3: Full parameter set
full_grid = {
    'rsi_period': [10, 14, 20],
    'rsi_oversold': [25, 30, 35],
    'rsi_overbought': [65, 70, 75],
    'stop_loss': [0.02, 0.03, 0.04],
    'position_size': [0.1, 0.25, 0.5]
}
```

### 2. Use Statistical Significance

```python
def bootstrap_confidence_interval(strategy, data, n_bootstrap=1000, confidence=0.95):
    """Calculate confidence intervals using bootstrap"""
    results = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample_indices = np.random.choice(len(data), len(data), replace=True)
        sample_data = data.iloc[sample_indices].sort_index()
        
        # Run backtest on sample
        engine = BacktestEngine()
        result = engine.run(sample_data, strategy)
        results.append(result['performance']['sharpe_ratio'])
    
    # Calculate confidence interval
    lower = np.percentile(results, (1 - confidence) / 2 * 100)
    upper = np.percentile(results, (1 + confidence) / 2 * 100)
    
    return {
        'mean': np.mean(results),
        'std': np.std(results),
        'ci_lower': lower,
        'ci_upper': upper
    }
```

### 3. Document Everything

```python
# optimization_report.py
def generate_optimization_report(optimization_results, output_path):
    """Generate comprehensive optimization report"""
    
    report = {
        'timestamp': pd.Timestamp.now(),
        'data_period': {
            'start': data.index[0],
            'end': data.index[-1],
            'total_days': len(data)
        },
        'optimization_method': optimization_results['method'],
        'objective_function': optimization_results['objective'],
        'parameter_space': optimization_results['parameter_space'],
        'best_parameters': optimization_results['best_params'],
        'performance_metrics': {
            'in_sample': optimization_results['in_sample_performance'],
            'out_sample': optimization_results['out_sample_performance'],
            'degradation': calculate_degradation(optimization_results)
        },
        'robustness_tests': {
            'stability_score': optimization_results['stability_score'],
            'cross_validation': optimization_results['cv_results'],
            'walk_forward': optimization_results['wf_results']
        }
    }
    
    # Save as JSON
    with open(f"{output_path}/optimization_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate visualizations
    create_optimization_plots(optimization_results, output_path)
    
    return report
```

## Common Pitfalls

### 1. Data Snooping

**Problem**: Using future information in optimization

```python
# WRONG: Looking ahead
def bad_indicator(data, period=20):
    # This uses future data!
    return data['close'].rolling(period, center=True).mean()

# CORRECT: Only historical data
def good_indicator(data, period=20):
    return data['close'].rolling(period).mean()
```

### 2. Survivorship Bias

**Problem**: Only testing on stocks that still exist

```python
# Include delisted stocks in your universe
universe = fetch_all_stocks(include_delisted=True)

# Weight results by market cap to avoid small-cap bias
weighted_results = sum(
    result['sharpe'] * market_cap[symbol] 
    for symbol, result in results.items()
) / sum(market_cap.values())
```

### 3. Over-Optimization

**Signs of over-optimization:**
- Parameters work only in specific date ranges
- Small parameter changes cause large performance changes
- Out-of-sample performance is much worse than in-sample

```python
def detect_overfitting(in_sample_results, out_sample_results):
    """Detect potential overfitting"""
    
    degradation = (in_sample_results - out_sample_results) / in_sample_results
    
    if degradation > 0.5:  # 50% degradation
        print("WARNING: Severe overfitting detected!")
    elif degradation > 0.3:  # 30% degradation
        print("WARNING: Moderate overfitting detected")
    
    # Check parameter sensitivity
    param_sensitivity = calculate_parameter_sensitivity(...)
    if param_sensitivity > 0.2:  # 20% change per unit
        print("WARNING: Parameters are too sensitive")
```

### 4. Ignoring Transaction Costs

```python
# Always include realistic costs
realistic_costs = {
    'commission': 0.001,        # 0.1% per trade
    'slippage': 0.0005,        # 0.05% slippage
    'market_impact': 0.0002,   # For large positions
    'borrowing_cost': 0.02/252 # For short positions (2% annual)
}

# Test sensitivity to costs
for cost_multiplier in [0.5, 1.0, 1.5, 2.0]:
    adjusted_costs = {k: v * cost_multiplier for k, v in realistic_costs.items()}
    result = run_backtest_with_costs(strategy, data, adjusted_costs)
    print(f"Cost multiplier {cost_multiplier}: Sharpe = {result['sharpe']:.2f}")
```

## Optimization Workflow Template

```python
# complete_optimization_workflow.py
import asyncio
from datetime import datetime
import json

async def complete_optimization_workflow(symbol, strategy_class):
    """Complete optimization workflow with all best practices"""
    
    print(f"Starting optimization for {symbol} - {datetime.now()}")
    
    # 1. Data Preparation
    print("1. Fetching and preparing data...")
    fetcher = StockDataFetcher()
    full_data = await fetcher.fetch(symbol, '2015-01-01', '2023-12-31')
    
    # Split data
    train_end = '2021-12-31'
    validation_end = '2022-12-31'
    
    train_data = full_data[:train_end]
    validation_data = full_data[train_end:validation_end]
    test_data = full_data[validation_end:]
    
    # 2. Initial Parameter Exploration
    print("2. Initial parameter exploration...")
    param_ranges = {
        'rsi_period': range(5, 50, 5),
        'rsi_oversold': range(20, 40, 5),
        'stop_loss': np.arange(0.01, 0.05, 0.01)
    }
    
    # Quick random search
    random_optimizer = RandomSearchOptimizer(n_iter=100)
    random_results = random_optimizer.optimize(
        strategy_class, param_ranges, train_data
    )
    
    # 3. Refined Grid Search
    print("3. Refined grid search around best parameters...")
    refined_grid = create_refined_grid(random_results['best_params'])
    
    grid_optimizer = GridSearchOptimizer(objective='sharpe_ratio')
    grid_results = grid_optimizer.optimize(
        strategy_class, refined_grid, train_data
    )
    
    # 4. Validation
    print("4. Validating on held-out data...")
    best_strategy = strategy_class(**grid_results['best_params'])
    engine = BacktestEngine()
    
    validation_results = engine.run(validation_data, best_strategy)
    
    # 5. Walk-Forward Analysis
    print("5. Running walk-forward analysis...")
    wf_results = walk_forward_optimization(
        full_data[:validation_end],
        strategy_class,
        refined_grid
    )
    
    # 6. Final Testing
    print("6. Final out-of-sample test...")
    test_results = engine.run(test_data, best_strategy)
    
    # 7. Robustness Checks
    print("7. Running robustness checks...")
    stability = stability_test(strategy_class, grid_results['best_params'], full_data)
    bootstrap = bootstrap_confidence_interval(best_strategy, test_data)
    
    # 8. Generate Report
    print("8. Generating optimization report...")
    report = {
        'symbol': symbol,
        'strategy': strategy_class.__name__,
        'optimization_date': datetime.now().isoformat(),
        'best_parameters': grid_results['best_params'],
        'performance': {
            'train': grid_results['best_score'],
            'validation': validation_results['performance']['sharpe_ratio'],
            'test': test_results['performance']['sharpe_ratio']
        },
        'robustness': {
            'stability_score': stability['stability_score'],
            'bootstrap_ci': bootstrap,
            'walk_forward_consistency': analyze_wf_consistency(wf_results)
        },
        'recommendation': generate_recommendation(
            grid_results, validation_results, test_results, stability
        )
    }
    
    # Save report
    with open(f'optimization_{symbol}_{datetime.now():%Y%m%d}.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("Optimization complete!")
    return report

# Run the workflow
if __name__ == "__main__":
    report = asyncio.run(complete_optimization_workflow('AAPL', RSIStrategy))
```

This comprehensive guide provides all the tools and knowledge needed to optimize trading strategies effectively while maintaining robustness and avoiding common pitfalls.