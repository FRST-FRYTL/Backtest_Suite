# API Documentation

This document provides detailed API documentation for the Backtest Suite modules.

## Data Module

### StockDataFetcher

Fetches stock data from various sources.

```python
from src.data import StockDataFetcher

class StockDataFetcher:
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize data fetcher.
        
        Args:
            cache_dir: Directory for caching data
        """
    
    async def fetch(
        self,
        symbol: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch stock data.
        
        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            interval: Data interval (1m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)
            
        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
    
    async def fetch_multiple(
        self,
        symbols: List[str],
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols concurrently."""
```

## Indicators Module

### Base Indicator Class

All indicators inherit from this base class.

```python
from src.indicators.base import Indicator

class Indicator(ABC):
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate indicator values."""
        pass
```

### Technical Indicators

#### RSI

```python
from src.indicators import RSI

rsi = RSI(
    period: int = 14,
    overbought: float = 70,
    oversold: float = 30
)

# Calculate RSI values
rsi_values = rsi.calculate(data: pd.DataFrame, price_column: str = "close") -> pd.Series

# Get trading signals
signals = rsi.get_signals(rsi_values: pd.Series) -> pd.DataFrame

# Detect divergences
divergences = rsi.divergence(
    price: pd.Series,
    rsi_values: pd.Series,
    window: int = 14
) -> pd.DataFrame
```

#### BollingerBands

```python
from src.indicators import BollingerBands

bb = BollingerBands(
    period: int = 20,
    std_dev: float = 2.0,
    ma_type: str = "sma"  # "sma" or "ema"
)

# Calculate bands
bb_data = bb.calculate(data: pd.DataFrame, price_column: str = "close") -> pd.DataFrame
# Returns: bb_middle, bb_upper, bb_lower, bb_width, bb_percent

# Get signals
signals = bb.get_signals(data: pd.DataFrame, bb_data: pd.DataFrame) -> pd.DataFrame

# Detect patterns
patterns = bb.detect_patterns(
    data: pd.DataFrame,
    bb_data: pd.DataFrame,
    lookback: int = 20
) -> pd.DataFrame
```

## Strategies Module

### StrategyBuilder

```python
from src.strategies import StrategyBuilder, Rule, LogicalOperator

class StrategyBuilder:
    def __init__(self, name: str):
        """Initialize strategy builder."""
    
    def set_description(self, description: str) -> 'StrategyBuilder':
        """Set strategy description."""
    
    def add_entry_rule(
        self,
        rule: Union[Rule, str]
    ) -> 'StrategyBuilder':
        """Add entry rule."""
    
    def add_exit_rule(
        self,
        rule: Union[Rule, str]
    ) -> 'StrategyBuilder':
        """Add exit rule."""
    
    def set_risk_management(
        self,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        max_positions: int = 1
    ) -> 'StrategyBuilder':
        """Configure risk management."""
    
    def set_position_sizing(
        self,
        method: str = "fixed",
        size: float = 1.0,
        max_position: Optional[float] = None
    ) -> 'StrategyBuilder':
        """Configure position sizing."""
    
    def build(self) -> Strategy:
        """Build and return strategy."""
```

### Rule System

```python
from src.strategies import Rule, Condition, LogicalOperator

# Create complex rule
rule = Rule(operator=LogicalOperator.AND)
rule.add_condition('rsi', '<', 30)
rule.add_condition('close', '<', 'bb_lower')

# Nested rules
main_rule = Rule(operator=LogicalOperator.OR)
main_rule.add_rule(entry_rule_1)
main_rule.add_rule(entry_rule_2)
```

## Backtesting Module

### BacktestEngine

```python
from src.backtesting import BacktestEngine

class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 100000,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        event_driven: bool = True
    ):
        """Initialize backtest engine."""
    
    def run(
        self,
        data: pd.DataFrame,
        strategy: Strategy
    ) -> Dict[str, Any]:
        """
        Run backtest.
        
        Returns:
            Dictionary containing:
            - equity_curve: DataFrame with portfolio value over time
            - trades: List of executed trades
            - positions: Current positions
            - stats: Performance statistics
        """
```

### Trade Object

```python
class Trade:
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    side: str  # "long" or "short"
    commission: float
    pnl: Optional[float]
    return_pct: Optional[float]
    exit_reason: Optional[str]
```

## Performance Module

### PerformanceMetrics

```python
from src.utils import PerformanceMetrics

class PerformanceMetrics:
    @staticmethod
    def calculate(
        equity_curve: pd.DataFrame,
        trades: List[Trade],
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary containing:
            - total_return
            - annualized_return
            - sharpe_ratio
            - sortino_ratio
            - max_drawdown
            - calmar_ratio
            - win_rate
            - profit_factor
            - avg_win
            - avg_loss
            - expectancy
        """
    
    @staticmethod
    def generate_report(metrics: Dict[str, float]) -> str:
        """Generate formatted performance report."""
```

## Optimization Module

### StrategyOptimizer

```python
from src.optimization import StrategyOptimizer

class StrategyOptimizer:
    def __init__(
        self,
        data: pd.DataFrame,
        strategy_builder: StrategyBuilder,
        optimization_metric: str = "sharpe_ratio"
    ):
        """Initialize optimizer."""
    
    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        n_jobs: int = -1
    ) -> OptimizationResult:
        """Perform grid search optimization."""
    
    def random_search(
        self,
        param_distributions: Dict[str, Any],
        n_iter: int = 100,
        n_jobs: int = -1
    ) -> OptimizationResult:
        """Perform random search optimization."""
    
    def differential_evolution(
        self,
        bounds: Dict[str, Tuple[float, float]],
        population_size: int = 50,
        generations: int = 100
    ) -> OptimizationResult:
        """Perform differential evolution optimization."""
```

### Walk-Forward Analysis

```python
from src.optimization import WalkForwardAnalysis

wfa = WalkForwardAnalysis(
    train_days=252,  # 1 year
    test_days=63,    # 3 months
    step_days=21     # 1 month
)

results = wfa.run(
    data=data,
    optimizer=optimizer,
    param_grid=param_grid
)
# Returns: in_sample_results, out_of_sample_results, parameter_stability
```

## Visualization Module

### Dashboard

```python
from src.visualization import Dashboard

dashboard = Dashboard(
    theme: str = "plotly_dark",
    include_drawdown: bool = True,
    include_returns_dist: bool = True,
    include_monthly_returns: bool = True
)

# Create interactive dashboard
dashboard_path = dashboard.create_dashboard(
    results: Dict[str, Any],
    output_path: str = "dashboard.html"
) -> str
```

### ChartGenerator

```python
from src.visualization import ChartGenerator

chart_gen = ChartGenerator(style: str = "plotly")

# Equity curve
fig = chart_gen.plot_equity_curve(
    equity_curve: pd.DataFrame,
    benchmark: Optional[pd.Series] = None
)

# Trade visualization
fig = chart_gen.plot_trades(
    data: pd.DataFrame,
    trades: List[Trade],
    symbol: str,
    indicators: Optional[Dict[str, pd.Series]] = None
)

# Returns distribution
fig = chart_gen.plot_returns_distribution(
    returns: pd.Series,
    bins: int = 50
)

# Drawdown analysis
fig = chart_gen.plot_drawdown(
    equity_curve: pd.DataFrame
)
```

## CLI Module

### Commands

```python
# Fetch data
backtest fetch [OPTIONS]
  -s, --symbol TEXT         Stock symbol [required]
  -S, --start TEXT          Start date (YYYY-MM-DD) [required]
  -E, --end TEXT            End date (YYYY-MM-DD) [required]
  -i, --interval TEXT       Data interval [default: 1d]
  -o, --output TEXT         Output file path [required]

# Run backtest
backtest run [OPTIONS]
  -d, --data TEXT           Data file path [required]
  -s, --strategy TEXT       Strategy file or name [required]
  -c, --capital FLOAT       Initial capital [default: 100000]
  -o, --output TEXT         Output directory [required]

# Optimize strategy
backtest optimize [OPTIONS]
  -d, --data TEXT           Data file path [required]
  -s, --strategy TEXT       Strategy file [required]
  -p, --params TEXT         Parameter config file [required]
  -m, --method TEXT         Optimization method [default: grid]
  -o, --output TEXT         Output directory [required]

# Batch testing
backtest batch [OPTIONS]
  -s, --symbols TEXT        Comma-separated symbols [required]
  -S, --strategies TEXT     Strategy directory [required]
  -c, --capital FLOAT       Initial capital [default: 100000]
  -o, --output TEXT         Output directory [required]
```

## Error Handling

All modules implement proper error handling:

```python
from src.exceptions import (
    DataFetchError,
    IndicatorError,
    StrategyError,
    BacktestError,
    OptimizationError
)

try:
    data = await fetcher.fetch("INVALID", start, end)
except DataFetchError as e:
    print(f"Failed to fetch data: {e}")
```

## Logging

The suite uses Python's logging module:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get logger for module
logger = logging.getLogger(__name__)
```

## Configuration

Global configuration can be set via environment variables or config file:

```python
# Environment variables
BACKTEST_CACHE_DIR=/path/to/cache
BACKTEST_LOG_LEVEL=DEBUG
BACKTEST_DEFAULT_COMMISSION=0.001

# Config file (config.yaml)
cache_dir: /path/to/cache
log_level: DEBUG
default_commission: 0.001
```