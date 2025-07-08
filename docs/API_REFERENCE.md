# Backtest Suite API Reference

This document provides comprehensive API documentation for all modules in the Backtest Suite.

## Table of Contents

- [Data Module](#data-module)
- [Indicators Module](#indicators-module)
- [Meta Indicators Module](#meta-indicators-module)
- [Strategies Module](#strategies-module)
- [Backtesting Module](#backtesting-module)
- [Portfolio Module](#portfolio-module)
- [Optimization Module](#optimization-module)
- [Monitoring Module](#monitoring-module)
- [Visualization Module](#visualization-module)
- [CLI Module](#cli-module)
- [Utils Module](#utils-module)

## Data Module

### StockDataFetcher

Fetches stock data from various sources with caching support.

```python
from src.data import StockDataFetcher

class StockDataFetcher:
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize data fetcher with optional cache directory.
        
        Args:
            cache_dir: Directory for caching fetched data
        """
    
    async def fetch(
        self,
        symbol: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch stock data for a single symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            start: Start date (string 'YYYY-MM-DD' or datetime)
            end: End date (string 'YYYY-MM-DD' or datetime)
            interval: Data interval - valid values:
                     '1m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo'
            
        Returns:
            pd.DataFrame with columns: open, high, low, close, volume
            Index: DatetimeIndex
            
        Raises:
            ValueError: If invalid symbol or date range
            ConnectionError: If network issues
            
        Example:
            >>> fetcher = StockDataFetcher()
            >>> data = await fetcher.fetch('AAPL', '2023-01-01', '2023-12-31')
        """
    
    async def fetch_multiple(
        self,
        symbols: List[str],
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols concurrently.
        
        Args:
            symbols: List of stock symbols
            start: Start date
            end: End date
            interval: Data interval
            
        Returns:
            Dictionary mapping symbol to DataFrame
            
        Example:
            >>> data = await fetcher.fetch_multiple(
            ...     ['AAPL', 'GOOGL', 'MSFT'],
            ...     '2023-01-01',
            ...     '2023-12-31'
            ... )
        """
```

### DataLoader

Loads data from CSV files with validation.

```python
from src.data import DataLoader

class DataLoader:
    @staticmethod
    def load_csv(
        filepath: str,
        parse_dates: bool = True,
        index_col: str = 'Date'
    ) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            parse_dates: Whether to parse date columns
            index_col: Column to use as index
            
        Returns:
            pd.DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns missing
        """
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, raises ValueError otherwise
            
        Required columns: open, high, low, close, volume
        """
```

## Indicators Module

### RSI (Relative Strength Index)

```python
from src.indicators import RSI

class RSI:
    def __init__(self, period: int = 14):
        """
        Initialize RSI indicator.
        
        Args:
            period: Lookback period for RSI calculation (default: 14)
        """
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate RSI values.
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            pd.Series with RSI values (0-100)
            
        Example:
            >>> rsi = RSI(14)
            >>> rsi_values = rsi.calculate(data)
        """
    
    def detect_divergence(
        self,
        data: pd.DataFrame,
        rsi_values: pd.Series,
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Detect bullish and bearish divergences.
        
        Args:
            data: Price data with 'close' column
            rsi_values: Calculated RSI values
            lookback: Number of bars to look back for divergence
            
        Returns:
            DataFrame with columns:
            - bullish_divergence: Boolean series
            - bearish_divergence: Boolean series
            
        Example:
            >>> divergences = rsi.detect_divergence(data, rsi_values)
        """
```

### BollingerBands

```python
from src.indicators import BollingerBands

class BollingerBands:
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands indicator.
        
        Args:
            period: Moving average period (default: 20)
            std_dev: Number of standard deviations (default: 2.0)
        """
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            DataFrame with columns:
            - bb_middle: Middle band (SMA)
            - bb_upper: Upper band
            - bb_lower: Lower band
            - bb_width: Band width
            - bb_percent: Percent B
            
        Example:
            >>> bb = BollingerBands(20, 2)
            >>> bands = bb.calculate(data)
        """
    
    def detect_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect W-bottom and M-top patterns.
        
        Args:
            data: DataFrame with Bollinger Bands calculated
            
        Returns:
            DataFrame with columns:
            - w_bottom: Boolean series for W-bottom patterns
            - m_top: Boolean series for M-top patterns
        """
```

### VWMABands (Volume Weighted Moving Average Bands)

```python
from src.indicators import VWMABands

class VWMABands:
    def __init__(self, period: int = 20, multiplier: float = 2.0):
        """
        Initialize VWMA Bands indicator.
        
        Args:
            period: VWMA period (default: 20)
            multiplier: Band multiplier (default: 2.0)
        """
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWMA Bands.
        
        Args:
            data: DataFrame with 'close' and 'volume' columns
            
        Returns:
            DataFrame with columns:
            - vwma: Volume weighted moving average
            - vwma_upper: Upper band
            - vwma_lower: Lower band
            
        Example:
            >>> vwma = VWMABands(20, 2)
            >>> bands = vwma.calculate(data)
        """
```

### TSV (Time Segmented Volume)

```python
from src.indicators import TSV

class TSV:
    def __init__(self, period: int = 13):
        """
        Initialize TSV indicator.
        
        Args:
            period: TSV period (default: 13)
        """
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Time Segmented Volume.
        
        Args:
            data: DataFrame with 'close' and 'volume' columns
            
        Returns:
            pd.Series with TSV values
            
        Example:
            >>> tsv = TSV(13)
            >>> tsv_values = tsv.calculate(data)
        """
    
    def get_signals(self, tsv_values: pd.Series) -> pd.DataFrame:
        """
        Generate buy/sell signals from TSV.
        
        Args:
            tsv_values: Calculated TSV values
            
        Returns:
            DataFrame with columns:
            - tsv_buy: Boolean series for buy signals
            - tsv_sell: Boolean series for sell signals
        """
```

### VWAP (Volume Weighted Average Price)

```python
from src.indicators import VWAP

class VWAP:
    def __init__(self, anchor: str = 'D'):
        """
        Initialize VWAP indicator.
        
        Args:
            anchor: Anchor period ('D' for daily, 'W' for weekly, 'M' for monthly)
        """
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP with standard deviation bands.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with columns:
            - vwap: VWAP line
            - vwap_upper_1: 1 std dev upper band
            - vwap_lower_1: 1 std dev lower band
            - vwap_upper_2: 2 std dev upper band
            - vwap_lower_2: 2 std dev lower band
            
        Example:
            >>> vwap = VWAP('D')
            >>> vwap_data = vwap.calculate(data)
        """
    
    def calculate_rolling(
        self,
        data: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate rolling VWAP.
        
        Args:
            data: DataFrame with OHLCV columns
            window: Rolling window size
            
        Returns:
            DataFrame with rolling VWAP values
        """
```

## Meta Indicators Module

### FearGreedIndex

```python
from src.indicators.meta_indicators import FearGreedIndex

class FearGreedIndex:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Fear & Greed Index fetcher.
        
        Args:
            api_key: Optional API key for premium access
        """
    
    async def fetch_current(self) -> Dict[str, Any]:
        """
        Fetch current Fear & Greed Index value.
        
        Returns:
            Dictionary with:
            - value: Current index value (0-100)
            - value_classification: Text classification
            - timestamp: Data timestamp
            - time_until_update: Seconds until next update
            
        Example:
            >>> fgi = FearGreedIndex()
            >>> current = await fgi.fetch_current()
            >>> print(f"Fear & Greed: {current['value']} ({current['value_classification']})")
        """
    
    async def fetch_historical(
        self,
        limit: int = 30
    ) -> pd.DataFrame:
        """
        Fetch historical Fear & Greed data.
        
        Args:
            limit: Number of days to fetch
            
        Returns:
            DataFrame with columns:
            - value: Index value
            - value_classification: Text classification
            - timestamp: Date
        """
```

### InsiderTrading

```python
from src.indicators.meta_indicators import InsiderTrading

class InsiderTrading:
    def __init__(self):
        """Initialize insider trading data fetcher."""
    
    async def fetch_latest(
        self,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch latest insider trading transactions.
        
        Args:
            limit: Number of transactions to fetch
            
        Returns:
            DataFrame with columns:
            - filing_date: SEC filing date
            - trade_date: Actual trade date
            - ticker: Stock symbol
            - company: Company name
            - insider: Insider name
            - title: Insider's title
            - trade_type: 'Buy' or 'Sell'
            - price: Trade price
            - qty: Quantity traded
            - owned: Shares owned after trade
            - value: Transaction value
            
        Example:
            >>> insider = InsiderTrading()
            >>> trades = await insider.fetch_latest(50)
        """
    
    async def fetch_by_ticker(
        self,
        ticker: str,
        limit: int = 50
    ) -> pd.DataFrame:
        """
        Fetch insider trading for specific ticker.
        
        Args:
            ticker: Stock symbol
            limit: Number of transactions
            
        Returns:
            DataFrame with insider trades for the ticker
        """
    
    def calculate_sentiment(
        self,
        trades: pd.DataFrame,
        window: int = 30
    ) -> pd.Series:
        """
        Calculate insider sentiment score.
        
        Args:
            trades: DataFrame of insider trades
            window: Rolling window in days
            
        Returns:
            Series with sentiment scores (-1 to 1)
            Positive = net buying, Negative = net selling
        """
```

### MaxPain

```python
from src.indicators.meta_indicators import MaxPain

class MaxPain:
    def __init__(self):
        """Initialize max pain calculator."""
    
    async def calculate(
        self,
        ticker: str,
        expiration: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate max pain for options expiration.
        
        Args:
            ticker: Stock symbol
            expiration: Option expiration date (YYYY-MM-DD)
                       If None, uses nearest expiration
            
        Returns:
            Dictionary with:
            - max_pain_price: Max pain strike price
            - current_price: Current stock price
            - expiration: Expiration date used
            - call_oi: Total call open interest
            - put_oi: Total put open interest
            - strikes: List of strike prices
            - pain_values: Pain values for each strike
            
        Example:
            >>> mp = MaxPain()
            >>> result = await mp.calculate('AAPL')
            >>> print(f"Max Pain: ${result['max_pain_price']}")
        """
    
    async def get_support_resistance(
        self,
        ticker: str,
        num_expirations: int = 3
    ) -> Dict[str, List[float]]:
        """
        Get support/resistance levels from options.
        
        Args:
            ticker: Stock symbol
            num_expirations: Number of expirations to analyze
            
        Returns:
            Dictionary with:
            - support_levels: List of support prices
            - resistance_levels: List of resistance prices
            - major_strikes: High open interest strikes
        """
```

## Strategies Module

### StrategyBuilder

```python
from src.strategies import StrategyBuilder

class StrategyBuilder:
    def __init__(self, name: str):
        """
        Initialize strategy builder.
        
        Args:
            name: Strategy name
        """
    
    def add_entry_rule(
        self,
        condition: str,
        logic: str = "AND"
    ) -> 'StrategyBuilder':
        """
        Add entry condition.
        
        Args:
            condition: Condition string (e.g., "rsi < 30")
            logic: How to combine with other rules ("AND" or "OR")
            
        Returns:
            Self for method chaining
            
        Example:
            >>> builder = StrategyBuilder("RSI Strategy")
            >>> builder.add_entry_rule("rsi < 30")
            >>> builder.add_entry_rule("close < bb_lower", "AND")
        """
    
    def add_exit_rule(
        self,
        condition: str,
        logic: str = "OR"
    ) -> 'StrategyBuilder':
        """
        Add exit condition.
        
        Args:
            condition: Condition string
            logic: How to combine with other rules
            
        Returns:
            Self for method chaining
        """
    
    def set_risk_management(
        self,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        position_size: Union[float, str] = 1.0
    ) -> 'StrategyBuilder':
        """
        Set risk management parameters.
        
        Args:
            stop_loss: Stop loss percentage (0.05 = 5%)
            take_profit: Take profit percentage
            trailing_stop: Trailing stop percentage
            position_size: Position size (fraction or "kelly")
            
        Returns:
            Self for method chaining
            
        Example:
            >>> builder.set_risk_management(
            ...     stop_loss=0.05,
            ...     take_profit=0.10,
            ...     position_size=0.25
            ... )
        """
    
    def add_filter(
        self,
        condition: str
    ) -> 'StrategyBuilder':
        """
        Add trade filter (e.g., market regime filter).
        
        Args:
            condition: Filter condition
            
        Returns:
            Self for method chaining
        """
    
    def build(self) -> 'Strategy':
        """
        Build and return the strategy.
        
        Returns:
            Compiled Strategy object
        """
```

### Strategy

```python
from src.strategies import Strategy

class Strategy:
    """Compiled strategy ready for backtesting."""
    
    def evaluate_entry(
        self,
        data: pd.DataFrame,
        index: int
    ) -> bool:
        """
        Evaluate entry conditions at given index.
        
        Args:
            data: Market data with indicators
            index: Current bar index
            
        Returns:
            True if entry conditions met
        """
    
    def evaluate_exit(
        self,
        data: pd.DataFrame,
        index: int,
        position: 'Position'
    ) -> bool:
        """
        Evaluate exit conditions for position.
        
        Args:
            data: Market data with indicators
            index: Current bar index
            position: Current position
            
        Returns:
            True if exit conditions met
        """
    
    def calculate_position_size(
        self,
        capital: float,
        price: float,
        data: pd.DataFrame,
        index: int
    ) -> int:
        """
        Calculate position size.
        
        Args:
            capital: Available capital
            price: Current price
            data: Market data
            index: Current bar index
            
        Returns:
            Number of shares to trade
        """
```

## Backtesting Module

### BacktestEngine

```python
from src.backtesting import BacktestEngine

class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade (0.001 = 0.1%)
            slippage: Slippage factor
        """
    
    def run(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run backtest.
        
        Args:
            data: Market data with indicators
            strategy: Strategy to test
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            Dictionary with:
            - trades: List of executed trades
            - performance: Performance metrics
            - equity_curve: Equity over time
            - drawdown_series: Drawdown series
            - positions: Position history
            
        Example:
            >>> engine = BacktestEngine(initial_capital=100000)
            >>> results = engine.run(data, strategy)
            >>> print(f"Total Return: {results['performance']['total_return']:.2%}")
        """
    
    def run_walk_forward(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        in_sample_periods: int = 252,
        out_sample_periods: int = 63,
        optimization_func: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis.
        
        Args:
            data: Market data
            strategy: Base strategy
            in_sample_periods: In-sample period length
            out_sample_periods: Out-of-sample period length
            optimization_func: Function to optimize parameters
            
        Returns:
            Walk-forward analysis results
        """
```

## Portfolio Module

### Portfolio

```python
from src.portfolio import Portfolio

class Portfolio:
    def __init__(
        self,
        initial_capital: float,
        commission: float = 0.001
    ):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate
        """
    
    def execute_trade(
        self,
        symbol: str,
        quantity: int,
        price: float,
        timestamp: pd.Timestamp,
        trade_type: str = "BUY"
    ) -> 'Trade':
        """
        Execute a trade.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Execution price
            timestamp: Trade timestamp
            trade_type: "BUY" or "SELL"
            
        Returns:
            Trade object with execution details
        """
    
    def get_positions(self) -> Dict[str, 'Position']:
        """Get current open positions."""
    
    def get_equity(self) -> float:
        """Get current total equity."""
    
    def get_cash(self) -> float:
        """Get current cash balance."""
    
    def update_market_prices(
        self,
        prices: Dict[str, float]
    ) -> None:
        """Update market prices for positions."""
```

### Position

```python
from src.portfolio import Position

class Position:
    @property
    def symbol(self) -> str:
        """Stock symbol."""
    
    @property
    def quantity(self) -> int:
        """Number of shares."""
    
    @property
    def entry_price(self) -> float:
        """Average entry price."""
    
    @property
    def current_price(self) -> float:
        """Current market price."""
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L percentage."""
    
    def add_to_position(
        self,
        quantity: int,
        price: float
    ) -> None:
        """Add to existing position."""
    
    def reduce_position(
        self,
        quantity: int,
        price: float
    ) -> float:
        """Reduce position and return realized P&L."""
```

## Optimization Module

### GridSearchOptimizer

```python
from src.optimization import GridSearchOptimizer

class GridSearchOptimizer:
    def __init__(
        self,
        objective: str = "sharpe_ratio",
        n_jobs: int = -1
    ):
        """
        Initialize grid search optimizer.
        
        Args:
            objective: Metric to optimize
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
    
    def optimize(
        self,
        strategy_class: Type[Strategy],
        parameter_grid: Dict[str, List[Any]],
        data: pd.DataFrame,
        backtest_kwargs: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """
        Run grid search optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_grid: Parameter search space
            data: Market data
            backtest_kwargs: Additional backtest arguments
            
        Returns:
            Dictionary with:
            - best_params: Optimal parameters
            - best_score: Best objective value
            - all_results: All parameter combinations tested
            
        Example:
            >>> optimizer = GridSearchOptimizer(objective="sharpe_ratio")
            >>> param_grid = {
            ...     'rsi_period': [10, 14, 20],
            ...     'rsi_oversold': [20, 25, 30],
            ...     'rsi_overbought': [70, 75, 80]
            ... }
            >>> results = optimizer.optimize(RSIStrategy, param_grid, data)
        """
```

### RandomSearchOptimizer

```python
from src.optimization import RandomSearchOptimizer

class RandomSearchOptimizer:
    def __init__(
        self,
        objective: str = "sharpe_ratio",
        n_iter: int = 100,
        n_jobs: int = -1
    ):
        """
        Initialize random search optimizer.
        
        Args:
            objective: Metric to optimize
            n_iter: Number of iterations
            n_jobs: Number of parallel jobs
        """
    
    def optimize(
        self,
        strategy_class: Type[Strategy],
        parameter_distributions: Dict[str, Any],
        data: pd.DataFrame,
        backtest_kwargs: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """
        Run random search optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_distributions: Parameter distributions
            data: Market data
            backtest_kwargs: Additional backtest arguments
            
        Returns:
            Optimization results
        """
```

### DifferentialEvolution

```python
from src.optimization import DifferentialEvolution

class DifferentialEvolution:
    def __init__(
        self,
        objective: str = "sharpe_ratio",
        population_size: int = 50,
        generations: int = 100,
        mutation_factor: float = 0.8,
        crossover_prob: float = 0.7
    ):
        """
        Initialize differential evolution optimizer.
        
        Args:
            objective: Metric to optimize
            population_size: DE population size
            generations: Number of generations
            mutation_factor: Mutation factor (F)
            crossover_prob: Crossover probability (CR)
        """
    
    def optimize(
        self,
        strategy_class: Type[Strategy],
        parameter_bounds: Dict[str, Tuple[float, float]],
        data: pd.DataFrame,
        backtest_kwargs: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """
        Run differential evolution optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_bounds: Parameter bounds (min, max)
            data: Market data
            backtest_kwargs: Additional backtest arguments
            
        Returns:
            Optimization results with convergence history
        """
```

## Monitoring Module

### LiveMonitor

```python
from src.monitoring import LiveMonitor

class LiveMonitor:
    def __init__(
        self,
        update_interval: int = 5,
        port: int = 8050
    ):
        """
        Initialize live monitoring dashboard.
        
        Args:
            update_interval: Update interval in seconds
            port: Dashboard port
        """
    
    def start(
        self,
        backtest_engine: BacktestEngine,
        data_stream: AsyncIterator[pd.DataFrame]
    ) -> None:
        """
        Start live monitoring.
        
        Args:
            backtest_engine: Backtest engine instance
            data_stream: Async iterator of market data
            
        Example:
            >>> monitor = LiveMonitor(update_interval=5)
            >>> monitor.start(engine, data_stream)
        """
    
    def add_metric(
        self,
        name: str,
        calculator: Callable[[Dict], float]
    ) -> None:
        """
        Add custom metric to monitor.
        
        Args:
            name: Metric name
            calculator: Function to calculate metric
        """
    
    def add_chart(
        self,
        name: str,
        chart_type: str,
        data_source: str
    ) -> None:
        """
        Add custom chart to dashboard.
        
        Args:
            name: Chart name
            chart_type: Chart type (line, bar, scatter)
            data_source: Data source for chart
        """
```

### AlertManager

```python
from src.monitoring import AlertManager

class AlertManager:
    def __init__(self):
        """Initialize alert manager."""
    
    def add_alert(
        self,
        name: str,
        condition: Callable[[Dict], bool],
        action: Callable[[Dict], None],
        cooldown: int = 300
    ) -> None:
        """
        Add alert rule.
        
        Args:
            name: Alert name
            condition: Condition function
            action: Action to take when triggered
            cooldown: Cooldown period in seconds
            
        Example:
            >>> alerts = AlertManager()
            >>> alerts.add_alert(
            ...     "High Drawdown",
            ...     lambda metrics: metrics['drawdown'] > 0.10,
            ...     lambda metrics: send_email("Drawdown Alert!")
            ... )
        """
    
    def check_alerts(
        self,
        metrics: Dict[str, Any]
    ) -> List[str]:
        """
        Check all alerts against current metrics.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            List of triggered alert names
        """
```

## Visualization Module

### ChartBuilder

```python
from src.visualization import ChartBuilder

class ChartBuilder:
    def __init__(self, title: str = "Backtest Results"):
        """
        Initialize chart builder.
        
        Args:
            title: Main chart title
        """
    
    def add_price_chart(
        self,
        data: pd.DataFrame,
        trades: List[Trade] = None
    ) -> 'ChartBuilder':
        """
        Add price chart with optional trades.
        
        Args:
            data: OHLC data
            trades: List of trades to plot
            
        Returns:
            Self for method chaining
        """
    
    def add_indicator(
        self,
        name: str,
        data: Union[pd.Series, pd.DataFrame],
        subplot: bool = False,
        color: str = None
    ) -> 'ChartBuilder':
        """
        Add indicator to chart.
        
        Args:
            name: Indicator name
            data: Indicator data
            subplot: Whether to create new subplot
            color: Line color
            
        Returns:
            Self for method chaining
        """
    
    def add_equity_curve(
        self,
        equity: pd.Series,
        benchmark: pd.Series = None
    ) -> 'ChartBuilder':
        """
        Add equity curve chart.
        
        Args:
            equity: Portfolio equity over time
            benchmark: Optional benchmark to compare
            
        Returns:
            Self for method chaining
        """
    
    def add_drawdown(
        self,
        drawdown: pd.Series
    ) -> 'ChartBuilder':
        """
        Add drawdown chart.
        
        Args:
            drawdown: Drawdown series
            
        Returns:
            Self for method chaining
        """
    
    def build(self) -> go.Figure:
        """
        Build and return Plotly figure.
        
        Returns:
            Plotly figure object
        """
    
    def show(self) -> None:
        """Display the chart."""
    
    def save(
        self,
        filename: str,
        format: str = "html"
    ) -> None:
        """
        Save chart to file.
        
        Args:
            filename: Output filename
            format: Output format (html, png, pdf)
        """
```

### DashboardBuilder

```python
from src.visualization import DashboardBuilder

class DashboardBuilder:
    def __init__(
        self,
        title: str = "Backtest Dashboard"
    ):
        """
        Initialize dashboard builder.
        
        Args:
            title: Dashboard title
        """
    
    def add_summary_stats(
        self,
        stats: Dict[str, Any]
    ) -> 'DashboardBuilder':
        """
        Add summary statistics table.
        
        Args:
            stats: Dictionary of statistics
            
        Returns:
            Self for method chaining
        """
    
    def add_monthly_returns(
        self,
        returns: pd.Series
    ) -> 'DashboardBuilder':
        """
        Add monthly returns heatmap.
        
        Args:
            returns: Daily returns series
            
        Returns:
            Self for method chaining
        """
    
    def add_trade_analysis(
        self,
        trades: List[Trade]
    ) -> 'DashboardBuilder':
        """
        Add trade analysis section.
        
        Args:
            trades: List of trades
            
        Returns:
            Self for method chaining
        """
    
    def build(self) -> dash.Dash:
        """
        Build and return Dash app.
        
        Returns:
            Dash application
        """
    
    def run(
        self,
        port: int = 8050,
        debug: bool = False
    ) -> None:
        """
        Run dashboard server.
        
        Args:
            port: Server port
            debug: Debug mode
        """
```

## CLI Module

### CLI Commands

The CLI provides the following commands:

```bash
# Main command
backtest [OPTIONS] COMMAND [ARGS]...

# Available commands:
backtest fetch      # Fetch market data
backtest run        # Run backtest
backtest optimize   # Optimize strategy
backtest monitor    # Start live monitor
backtest indicators # List available indicators
```

### Fetch Command

```bash
backtest fetch [OPTIONS]

Options:
  -s, --symbol TEXT       Stock symbol (required)
  -S, --start TEXT        Start date YYYY-MM-DD (required)
  -E, --end TEXT          End date YYYY-MM-DD (required)
  -i, --interval TEXT     Data interval [default: 1d]
  -o, --output TEXT       Output file path (required)
  --multiple TEXT         Comma-separated symbols
  
Examples:
  # Single symbol
  backtest fetch -s AAPL -S 2023-01-01 -E 2023-12-31 -o data/AAPL.csv
  
  # Multiple symbols
  backtest fetch --multiple AAPL,GOOGL,MSFT -S 2023-01-01 -E 2023-12-31 -o data/
```

### Run Command

```bash
backtest run [OPTIONS]

Options:
  -d, --data TEXT         Data file path (required)
  -s, --strategy TEXT     Strategy YAML file (required)
  -o, --output TEXT       Output directory (required)
  --start TEXT            Backtest start date
  --end TEXT              Backtest end date
  --initial-capital FLOAT Initial capital [default: 100000]
  --commission FLOAT      Commission rate [default: 0.001]
  --html                  Generate HTML report
  --json                  Export results as JSON
  
Examples:
  # Run backtest
  backtest run -d data/AAPL.csv -s strategies/rsi.yaml -o results/
  
  # With custom parameters
  backtest run -d data/AAPL.csv -s strategies/rsi.yaml -o results/ \
    --initial-capital 50000 --commission 0.002 --html
```

### Optimize Command

```bash
backtest optimize [OPTIONS]

Options:
  -d, --data TEXT         Data file path (required)
  -s, --strategy TEXT     Strategy YAML file (required)
  -p, --params TEXT       Parameter config YAML (required)
  -o, --output TEXT       Output directory (required)
  --method TEXT           Optimization method [default: grid]
  --objective TEXT        Objective metric [default: sharpe_ratio]
  --n-jobs INTEGER        Parallel jobs [default: -1]
  
Examples:
  # Grid search
  backtest optimize -d data/AAPL.csv -s strategies/rsi.yaml \
    -p params/rsi_params.yaml -o optimization/
  
  # Random search
  backtest optimize -d data/AAPL.csv -s strategies/rsi.yaml \
    -p params/rsi_params.yaml -o optimization/ --method random
```

## Utils Module

### PerformanceMetrics

```python
from src.utils import PerformanceMetrics

class PerformanceMetrics:
    @staticmethod
    def calculate_returns(
        equity_curve: pd.Series
    ) -> pd.Series:
        """Calculate daily returns from equity curve."""
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Daily returns
            risk_free_rate: Annual risk-free rate
            periods: Periods per year
            
        Returns:
            Annualized Sharpe ratio
        """
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods: int = 252
    ) -> float:
        """Calculate Sortino ratio (downside deviation)."""
    
    @staticmethod
    def calculate_max_drawdown(
        equity_curve: pd.Series
    ) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown.
        
        Returns:
            Tuple of (max_dd, peak_date, trough_date)
        """
    
    @staticmethod
    def calculate_calmar_ratio(
        returns: pd.Series,
        periods: int = 252
    ) -> float:
        """Calculate Calmar ratio (annual return / max DD)."""
    
    @staticmethod
    def calculate_win_rate(
        trades: List[Trade]
    ) -> float:
        """Calculate win rate from trades."""
    
    @staticmethod
    def calculate_profit_factor(
        trades: List[Trade]
    ) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
```

### RiskMetrics

```python
from src.utils import RiskMetrics

class RiskMetrics:
    @staticmethod
    def calculate_var(
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Daily returns
            confidence: Confidence level
            method: "historical" or "parametric"
            
        Returns:
            VaR value
        """
    
    @staticmethod
    def calculate_cvar(
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
    
    @staticmethod
    def calculate_kelly_criterion(
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly criterion for position sizing.
        
        Args:
            win_rate: Probability of winning
            avg_win: Average winning return
            avg_loss: Average losing return (positive)
            
        Returns:
            Optimal position size fraction
        """
```

## Configuration Examples

### Strategy YAML Configuration

```yaml
# strategies/rsi_mean_reversion.yaml
name: "RSI Mean Reversion"
version: "1.0"

indicators:
  - name: rsi
    type: RSI
    params:
      period: 14
  
  - name: bb
    type: BollingerBands
    params:
      period: 20
      std_dev: 2.0

entry_rules:
  - condition: "rsi < 30"
    logic: AND
  - condition: "close < bb_lower"
    logic: AND

exit_rules:
  - condition: "rsi > 70"
    logic: OR
  - condition: "close > bb_upper"
    logic: OR

risk_management:
  stop_loss: 0.05
  take_profit: 0.10
  trailing_stop: 0.03
  position_size: 0.25

filters:
  - condition: "volume > volume.rolling(20).mean()"
```

### Parameter Optimization Configuration

```yaml
# params/rsi_optimization.yaml
optimization:
  method: "grid"  # grid, random, or differential_evolution
  objective: "sharpe_ratio"
  n_jobs: -1

parameters:
  rsi_period:
    type: int
    min: 10
    max: 30
    step: 2
  
  rsi_oversold:
    type: int
    min: 20
    max: 35
    step: 5
  
  rsi_overbought:
    type: int
    min: 65
    max: 80
    step: 5
  
  bb_period:
    type: int
    min: 15
    max: 25
    step: 5
  
  stop_loss:
    type: float
    min: 0.02
    max: 0.10
    step: 0.02

constraints:
  - "rsi_oversold < rsi_overbought - 30"
```

## Error Handling

All modules follow consistent error handling patterns:

```python
from src.exceptions import (
    DataError,
    StrategyError,
    BacktestError,
    OptimizationError
)

try:
    data = await fetcher.fetch("AAPL", "2023-01-01", "2023-12-31")
except DataError as e:
    logger.error(f"Data fetch failed: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
```

## Logging

Configure logging for debugging:

```python
import logging
from src.utils import setup_logging

# Setup with custom config
setup_logging(
    level=logging.INFO,
    log_file="backtest.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Get logger for module
logger = logging.getLogger(__name__)
```

## Performance Tips

1. **Data Caching**: Use the built-in caching in StockDataFetcher
2. **Vectorization**: Indicators use NumPy/Pandas vectorization
3. **Parallel Processing**: Optimization supports parallel execution
4. **Memory Management**: Use data chunking for large datasets
5. **Profiling**: Built-in profiling tools available

```python
from src.utils import profile_backtest

# Profile backtest execution
with profile_backtest("my_backtest"):
    results = engine.run(data, strategy)
```

## Advanced Features

### Custom Indicators

```python
from src.indicators import BaseIndicator

class MyCustomIndicator(BaseIndicator):
    def __init__(self, param1: int = 10):
        self.param1 = param1
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        # Implement calculation
        return result
```

### Custom Strategies

```python
from src.strategies import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def __init__(self, **params):
        super().__init__(**params)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # Implement signal generation
        return signals
```

### Event Handlers

```python
from src.backtesting import EventHandler

class MyEventHandler(EventHandler):
    def on_trade(self, trade: Trade) -> None:
        # Handle trade execution
        pass
    
    def on_bar(self, bar: pd.Series) -> None:
        # Handle new bar
        pass

# Use with backtest engine
engine.add_event_handler(MyEventHandler())
```

This completes the comprehensive API reference documentation for the Backtest Suite.