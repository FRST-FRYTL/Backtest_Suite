"""Rule-based strategy components."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import numpy as np


class LogicalOperator(Enum):
    """Logical operators for combining conditions."""
    AND = "and"
    OR = "or"
    NOT = "not"
    XOR = "xor"


class ComparisonOperator(Enum):
    """Comparison operators for conditions."""
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NEQ = "!="
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"
    BETWEEN = "between"
    OUTSIDE = "outside"


@dataclass
class Condition:
    """Single condition for strategy rules."""
    
    left: str  # Column name or value
    operator: Union[ComparisonOperator, str]
    right: Union[str, float, int]  # Column name or value
    lookback: Optional[int] = None  # For cross conditions
    
    def __post_init__(self):
        """Convert string operator to enum if needed."""
        if isinstance(self.operator, str):
            # Try to convert string to ComparisonOperator
            for op in ComparisonOperator:
                if op.value == self.operator:
                    self.operator = op
                    break
                    
    def evaluate(self, data: pd.DataFrame, index: int) -> bool:
        """
        Evaluate condition at specific index.
        
        Args:
            data: DataFrame with market data
            index: Row index to evaluate
            
        Returns:
            Boolean result of condition
        """
        # Get left value
        left_val = self._get_value(data, index, self.left)
        
        # Handle special operators
        if self.operator == ComparisonOperator.CROSS_ABOVE:
            return self._check_cross_above(data, index, self.left, self.right)
        elif self.operator == ComparisonOperator.CROSS_BELOW:
            return self._check_cross_below(data, index, self.left, self.right)
        elif self.operator == ComparisonOperator.BETWEEN:
            # Right should be a tuple (min, max)
            if isinstance(self.right, tuple) and len(self.right) == 2:
                return self.right[0] <= left_val <= self.right[1]
            return False
        elif self.operator == ComparisonOperator.OUTSIDE:
            # Right should be a tuple (min, max)
            if isinstance(self.right, tuple) and len(self.right) == 2:
                return left_val < self.right[0] or left_val > self.right[1]
            return False
            
        # Get right value
        right_val = self._get_value(data, index, self.right)
        
        # Standard comparisons
        if self.operator == ComparisonOperator.GT:
            return left_val > right_val
        elif self.operator == ComparisonOperator.GTE:
            return left_val >= right_val
        elif self.operator == ComparisonOperator.LT:
            return left_val < right_val
        elif self.operator == ComparisonOperator.LTE:
            return left_val <= right_val
        elif self.operator == ComparisonOperator.EQ:
            return left_val == right_val
        elif self.operator == ComparisonOperator.NEQ:
            return left_val != right_val
            
        return False
        
    def _get_value(
        self,
        data: pd.DataFrame,
        index: int,
        value: Union[str, float, int]
    ) -> Any:
        """Get value from data or return literal value."""
        if isinstance(value, str) and value in data.columns:
            return data.iloc[index][value]
        return value
        
    def _check_cross_above(
        self,
        data: pd.DataFrame,
        index: int,
        left: str,
        right: Union[str, float]
    ) -> bool:
        """Check if left crosses above right."""
        if index == 0:
            return False
            
        curr_left = self._get_value(data, index, left)
        prev_left = self._get_value(data, index - 1, left)
        curr_right = self._get_value(data, index, right)
        prev_right = self._get_value(data, index - 1, right)
        
        return prev_left <= prev_right and curr_left > curr_right
        
    def _check_cross_below(
        self,
        data: pd.DataFrame,
        index: int,
        left: str,
        right: Union[str, float]
    ) -> bool:
        """Check if left crosses below right."""
        if index == 0:
            return False
            
        curr_left = self._get_value(data, index, left)
        prev_left = self._get_value(data, index - 1, left)
        curr_right = self._get_value(data, index, right)
        prev_right = self._get_value(data, index - 1, right)
        
        return prev_left >= prev_right and curr_left < curr_right


@dataclass
class Rule:
    """Rule combining multiple conditions."""
    
    conditions: List[Union[Condition, 'Rule']] = field(default_factory=list)
    operator: LogicalOperator = LogicalOperator.AND
    name: Optional[str] = None
    
    def add_condition(
        self,
        left: str,
        operator: Union[ComparisonOperator, str],
        right: Union[str, float, int, tuple]
    ) -> 'Rule':
        """
        Add a condition to the rule.
        
        Args:
            left: Left side of comparison
            operator: Comparison operator
            right: Right side of comparison
            
        Returns:
            Self for chaining
        """
        condition = Condition(left, operator, right)
        self.conditions.append(condition)
        return self
        
    def add_rule(self, rule: 'Rule') -> 'Rule':
        """
        Add a nested rule.
        
        Args:
            rule: Rule to add
            
        Returns:
            Self for chaining
        """
        self.conditions.append(rule)
        return self
        
    def evaluate(self, data: pd.DataFrame, index: int) -> bool:
        """
        Evaluate rule at specific index.
        
        Args:
            data: DataFrame with market data
            index: Row index to evaluate
            
        Returns:
            Boolean result of rule
        """
        if not self.conditions:
            return True
            
        results = []
        for condition in self.conditions:
            if isinstance(condition, Rule):
                results.append(condition.evaluate(data, index))
            else:
                results.append(condition.evaluate(data, index))
                
        # Apply logical operator
        if self.operator == LogicalOperator.AND:
            return all(results)
        elif self.operator == LogicalOperator.OR:
            return any(results)
        elif self.operator == LogicalOperator.NOT:
            return not all(results)
        elif self.operator == LogicalOperator.XOR:
            # XOR is true if odd number of conditions are true
            return sum(results) % 2 == 1
            
        return False
        
    def evaluate_series(self, data: pd.DataFrame) -> pd.Series:
        """
        Evaluate rule for entire DataFrame.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Series with boolean results
        """
        results = pd.Series(False, index=data.index)
        
        for i in range(len(data)):
            results.iloc[i] = self.evaluate(data, i)
            
        return results
        
    def to_dict(self) -> Dict:
        """Convert rule to dictionary for serialization."""
        return {
            'name': self.name,
            'operator': self.operator.value,
            'conditions': [
                c.to_dict() if hasattr(c, 'to_dict') else {
                    'left': c.left,
                    'operator': c.operator.value if hasattr(c.operator, 'value') else c.operator,
                    'right': c.right,
                    'lookback': c.lookback
                }
                for c in self.conditions
            ]
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'Rule':
        """Create rule from dictionary."""
        rule = cls(
            name=data.get('name'),
            operator=LogicalOperator(data.get('operator', 'and'))
        )
        
        for cond_data in data.get('conditions', []):
            if 'conditions' in cond_data:
                # Nested rule
                rule.add_rule(cls.from_dict(cond_data))
            else:
                # Simple condition
                condition = Condition(
                    left=cond_data['left'],
                    operator=cond_data['operator'],
                    right=cond_data['right'],
                    lookback=cond_data.get('lookback')
                )
                rule.conditions.append(condition)
                
        return rule
        
    @classmethod
    def from_string(cls, expression: str) -> 'Rule':
        """
        Create rule from string expression.
        
        Args:
            expression: String like "rsi < 30 and close < bb_lower"
            
        Returns:
            Rule object
        """
        # This is a simplified parser - in production, use a proper parser
        rule = cls()
        
        # Split by 'and' or 'or'
        if ' and ' in expression:
            rule.operator = LogicalOperator.AND
            parts = expression.split(' and ')
        elif ' or ' in expression:
            rule.operator = LogicalOperator.OR
            parts = expression.split(' or ')
        else:
            parts = [expression]
            
        for part in parts:
            part = part.strip()
            
            # Parse condition
            for op in ['>=', '<=', '!=', '==', '>', '<']:
                if op in part:
                    left, right = part.split(op, 1)
                    left = left.strip()
                    right = right.strip()
                    
                    # Try to convert right to number
                    try:
                        right = float(right)
                    except ValueError:
                        pass
                        
                    rule.add_condition(left, op, right)
                    break
                    
        return rule