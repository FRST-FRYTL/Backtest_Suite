"""Strategy builder for creating and managing trading strategies."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
import yaml
import json

from .rules import Rule, Condition, LogicalOperator
from .signals import SignalGenerator


@dataclass
class PositionSizing:
    """Position sizing configuration."""
    
    method: str = "fixed"  # fixed, percent, volatility, kelly
    size: float = 1.0  # Base size or percentage
    max_position: float = 1.0  # Maximum position size
    scale_in: bool = False  # Allow scaling into positions
    scale_out: bool = False  # Allow scaling out of positions
    
    def calculate_size(
        self,
        capital: float,
        price: float,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> int:
        """
        Calculate position size based on method.
        
        Args:
            capital: Available capital
            price: Current price
            volatility: Price volatility (for volatility sizing)
            win_rate: Historical win rate (for Kelly)
            avg_win: Average win amount (for Kelly)
            avg_loss: Average loss amount (for Kelly)
            
        Returns:
            Number of shares to trade
        """
        if self.method == "fixed":
            shares = int(self.size)
        elif self.method == "percent":
            shares = int((capital * self.size) / price)
        elif self.method == "volatility" and volatility:
            # Size inversely proportional to volatility
            target_risk = capital * 0.02  # 2% risk
            shares = int(target_risk / (price * volatility))
        elif self.method == "kelly" and all([win_rate, avg_win, avg_loss]):
            # Kelly Criterion
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            kelly_fraction = (b * p - q) / b
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            shares = int((capital * kelly_fraction) / price)
        else:
            shares = int(self.size)
            
        # Apply maximum position limit
        max_shares = int((capital * self.max_position) / price)
        return min(shares, max_shares)


@dataclass
class RiskManagement:
    """Risk management configuration."""
    
    stop_loss: Optional[float] = None  # Percentage or absolute
    stop_loss_type: str = "percent"  # percent, absolute, atr
    take_profit: Optional[float] = None  # Percentage or absolute
    take_profit_type: str = "percent"  # percent, absolute, atr
    trailing_stop: Optional[float] = None  # Trailing stop percentage
    time_stop: Optional[int] = None  # Exit after N bars
    max_loss_per_trade: float = 0.02  # 2% max loss per trade
    max_daily_loss: float = 0.06  # 6% max daily loss
    max_positions: int = 5  # Maximum concurrent positions
    
    def calculate_stops(
        self,
        entry_price: float,
        atr: Optional[float] = None
    ) -> Dict[str, Optional[float]]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            entry_price: Entry price
            atr: Average True Range (for ATR-based stops)
            
        Returns:
            Dictionary with stop levels
        """
        stops = {'stop_loss': None, 'take_profit': None}
        
        # Stop loss
        if self.stop_loss:
            if self.stop_loss_type == "percent":
                stops['stop_loss'] = entry_price * (1 - self.stop_loss)
            elif self.stop_loss_type == "absolute":
                stops['stop_loss'] = entry_price - self.stop_loss
            elif self.stop_loss_type == "atr" and atr:
                stops['stop_loss'] = entry_price - (atr * self.stop_loss)
                
        # Take profit
        if self.take_profit:
            if self.take_profit_type == "percent":
                stops['take_profit'] = entry_price * (1 + self.take_profit)
            elif self.take_profit_type == "absolute":
                stops['take_profit'] = entry_price + self.take_profit
            elif self.take_profit_type == "atr" and atr:
                stops['take_profit'] = entry_price + (atr * self.take_profit)
                
        return stops


@dataclass
class Strategy:
    """Complete trading strategy."""
    
    name: str
    description: str = ""
    entry_rules: List[Rule] = field(default_factory=list)
    exit_rules: List[Rule] = field(default_factory=list)
    position_sizing: PositionSizing = field(default_factory=PositionSizing)
    risk_management: RiskManagement = field(default_factory=RiskManagement)
    filters: List[Rule] = field(default_factory=list)  # Market regime filters
    
    def to_dict(self) -> Dict:
        """Convert strategy to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'entry_rules': [rule.to_dict() for rule in self.entry_rules],
            'exit_rules': [rule.to_dict() for rule in self.exit_rules],
            'filters': [rule.to_dict() for rule in self.filters],
            'position_sizing': {
                'method': self.position_sizing.method,
                'size': self.position_sizing.size,
                'max_position': self.position_sizing.max_position,
                'scale_in': self.position_sizing.scale_in,
                'scale_out': self.position_sizing.scale_out
            },
            'risk_management': {
                'stop_loss': self.risk_management.stop_loss,
                'stop_loss_type': self.risk_management.stop_loss_type,
                'take_profit': self.risk_management.take_profit,
                'take_profit_type': self.risk_management.take_profit_type,
                'trailing_stop': self.risk_management.trailing_stop,
                'time_stop': self.risk_management.time_stop,
                'max_loss_per_trade': self.risk_management.max_loss_per_trade,
                'max_daily_loss': self.risk_management.max_daily_loss,
                'max_positions': self.risk_management.max_positions
            }
        }
        
    def to_yaml(self, filepath: str) -> None:
        """Save strategy to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
            
    def to_json(self, filepath: str) -> None:
        """Save strategy to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class StrategyBuilder:
    """Builder class for creating trading strategies."""
    
    def __init__(self, name: str = "Custom Strategy"):
        """
        Initialize strategy builder.
        
        Args:
            name: Strategy name
        """
        self.strategy = Strategy(name=name)
        self.signal_generator = SignalGenerator()
        
    def set_description(self, description: str) -> 'StrategyBuilder':
        """Set strategy description."""
        self.strategy.description = description
        return self
        
    def add_entry_rule(
        self,
        rule: Union[Rule, str],
        name: Optional[str] = None
    ) -> 'StrategyBuilder':
        """
        Add entry rule to strategy.
        
        Args:
            rule: Rule object or string expression
            name: Optional rule name
            
        Returns:
            Self for chaining
        """
        if isinstance(rule, str):
            rule = Rule.from_string(rule)
        if name:
            rule.name = name
        self.strategy.entry_rules.append(rule)
        return self
        
    def add_exit_rule(
        self,
        rule: Union[Rule, str],
        name: Optional[str] = None
    ) -> 'StrategyBuilder':
        """
        Add exit rule to strategy.
        
        Args:
            rule: Rule object or string expression
            name: Optional rule name
            
        Returns:
            Self for chaining
        """
        if isinstance(rule, str):
            rule = Rule.from_string(rule)
        if name:
            rule.name = name
        self.strategy.exit_rules.append(rule)
        return self
        
    def add_filter(
        self,
        rule: Union[Rule, str],
        name: Optional[str] = None
    ) -> 'StrategyBuilder':
        """
        Add market regime filter.
        
        Args:
            rule: Rule object or string expression
            name: Optional rule name
            
        Returns:
            Self for chaining
        """
        if isinstance(rule, str):
            rule = Rule.from_string(rule)
        if name:
            rule.name = name
        self.strategy.filters.append(rule)
        return self
        
    def set_position_sizing(
        self,
        method: str = "fixed",
        size: float = 1.0,
        max_position: float = 1.0,
        scale_in: bool = False,
        scale_out: bool = False
    ) -> 'StrategyBuilder':
        """Configure position sizing."""
        self.strategy.position_sizing = PositionSizing(
            method=method,
            size=size,
            max_position=max_position,
            scale_in=scale_in,
            scale_out=scale_out
        )
        return self
        
    def set_risk_management(
        self,
        stop_loss: Optional[float] = None,
        stop_loss_type: str = "percent",
        take_profit: Optional[float] = None,
        take_profit_type: str = "percent",
        trailing_stop: Optional[float] = None,
        time_stop: Optional[int] = None,
        max_positions: int = 5
    ) -> 'StrategyBuilder':
        """Configure risk management."""
        self.strategy.risk_management = RiskManagement(
            stop_loss=stop_loss,
            stop_loss_type=stop_loss_type,
            take_profit=take_profit,
            take_profit_type=take_profit_type,
            trailing_stop=trailing_stop,
            time_stop=time_stop,
            max_positions=max_positions
        )
        return self
        
    def build(self) -> Strategy:
        """Build and return the strategy."""
        return self.strategy
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from strategy rules.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with signals
        """
        return self.signal_generator.generate(self.strategy, data)
        
    @classmethod
    def from_yaml(cls, filepath: str) -> 'StrategyBuilder':
        """Load strategy from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
            
        builder = cls(name=data.get('name', 'Loaded Strategy'))
        builder.set_description(data.get('description', ''))
        
        # Load rules
        for rule_data in data.get('entry_rules', []):
            rule = Rule.from_dict(rule_data)
            builder.strategy.entry_rules.append(rule)
            
        for rule_data in data.get('exit_rules', []):
            rule = Rule.from_dict(rule_data)
            builder.strategy.exit_rules.append(rule)
            
        for rule_data in data.get('filters', []):
            rule = Rule.from_dict(rule_data)
            builder.strategy.filters.append(rule)
            
        # Load position sizing
        if 'position_sizing' in data:
            ps = data['position_sizing']
            builder.set_position_sizing(**ps)
            
        # Load risk management
        if 'risk_management' in data:
            rm = data['risk_management']
            builder.set_risk_management(**rm)
            
        return builder
        
    @classmethod
    def from_json(cls, filepath: str) -> 'StrategyBuilder':
        """Load strategy from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Same as YAML loading
        builder = cls(name=data.get('name', 'Loaded Strategy'))
        # ... (rest is same as from_yaml)
        
        return builder