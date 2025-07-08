"""Backtest Suite - A comprehensive backtesting framework for stock trading strategies."""

__version__ = "0.1.0"
__author__ = "Backtest Suite Team"
__email__ = "team@backtestsuite.com"

from src.data import StockDataFetcher
from src.backtesting import BacktestEngine
from src.strategies import StrategyBuilder

__all__ = ["StockDataFetcher", "BacktestEngine", "StrategyBuilder"]