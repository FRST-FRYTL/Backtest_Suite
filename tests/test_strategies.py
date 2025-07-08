"""Tests for strategy builder and rules."""

import pytest
import pandas as pd
import numpy as np

from src.strategies import StrategyBuilder, Rule, Condition, LogicalOperator
from src.strategies.rules import ComparisonOperator


@pytest.fixture
def sample_data():
    """Create sample data for strategy testing."""
    dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
    
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(len(dates))),
        'volume': np.random.randint(1000000, 5000000, len(dates)),
        'rsi': 50 + np.random.randn(len(dates)) * 20,
        'bb_upper': 105 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'bb_lower': 95 + np.cumsum(np.random.randn(len(dates)) * 0.5),
    }, index=dates)
    
    # Ensure RSI is in valid range
    data['rsi'] = data['rsi'].clip(0, 100)
    
    return data


class TestCondition:
    """Test Condition class."""
    
    def test_simple_condition(self, sample_data):
        """Test simple comparison conditions."""
        # Test greater than
        cond = Condition('rsi', ComparisonOperator.GT, 70)
        result = cond.evaluate(sample_data, 0)
        assert isinstance(result, bool)
        
        # Test less than with column comparison
        cond = Condition('close', ComparisonOperator.LT, 'bb_lower')
        result = cond.evaluate(sample_data, 0)
        assert isinstance(result, bool)
        
    def test_cross_conditions(self, sample_data):
        """Test crossover conditions."""
        # Add crossing data
        sample_data['ma_fast'] = sample_data['close'].rolling(5).mean()
        sample_data['ma_slow'] = sample_data['close'].rolling(10).mean()
        
        cond = Condition('ma_fast', ComparisonOperator.CROSS_ABOVE, 'ma_slow')
        
        # Test at different indices
        for i in range(1, len(sample_data)):
            result = cond.evaluate(sample_data, i)
            assert isinstance(result, bool)
            
    def test_range_conditions(self, sample_data):
        """Test between and outside conditions."""
        # Test between
        cond = Condition('rsi', ComparisonOperator.BETWEEN, (30, 70))
        result = cond.evaluate(sample_data, 0)
        assert isinstance(result, bool)
        
        # Test outside
        cond = Condition('rsi', ComparisonOperator.OUTSIDE, (30, 70))
        result = cond.evaluate(sample_data, 0)
        assert isinstance(result, bool)


class TestRule:
    """Test Rule class."""
    
    def test_single_condition_rule(self, sample_data):
        """Test rule with single condition."""
        rule = Rule()
        rule.add_condition('rsi', '>', 70)
        
        # Evaluate at single point
        result = rule.evaluate(sample_data, 0)
        assert isinstance(result, bool)
        
        # Evaluate entire series
        series_result = rule.evaluate_series(sample_data)
        assert isinstance(series_result, pd.Series)
        assert len(series_result) == len(sample_data)
        
    def test_multiple_conditions_and(self, sample_data):
        """Test rule with multiple AND conditions."""
        rule = Rule(operator=LogicalOperator.AND)
        rule.add_condition('rsi', '<', 30)
        rule.add_condition('close', '<', 'bb_lower')
        
        series_result = rule.evaluate_series(sample_data)
        
        # Result should be True only when both conditions are True
        expected = (sample_data['rsi'] < 30) & (sample_data['close'] < sample_data['bb_lower'])
        pd.testing.assert_series_equal(series_result, expected, check_names=False)
        
    def test_multiple_conditions_or(self, sample_data):
        """Test rule with multiple OR conditions."""
        rule = Rule(operator=LogicalOperator.OR)
        rule.add_condition('rsi', '>', 70)
        rule.add_condition('close', '>', 'bb_upper')
        
        series_result = rule.evaluate_series(sample_data)
        
        # Result should be True when either condition is True
        expected = (sample_data['rsi'] > 70) | (sample_data['close'] > sample_data['bb_upper'])
        pd.testing.assert_series_equal(series_result, expected, check_names=False)
        
    def test_nested_rules(self, sample_data):
        """Test nested rules."""
        # Create sub-rules
        oversold_rule = Rule(name="Oversold")
        oversold_rule.add_condition('rsi', '<', 30)
        oversold_rule.add_condition('close', '<', 'bb_lower')
        
        volume_rule = Rule(name="High Volume")
        volume_rule.add_condition('volume', '>', 2000000)
        
        # Create main rule
        main_rule = Rule(operator=LogicalOperator.AND)
        main_rule.add_rule(oversold_rule)
        main_rule.add_rule(volume_rule)
        
        result = main_rule.evaluate_series(sample_data)
        assert isinstance(result, pd.Series)
        
    def test_rule_from_string(self):
        """Test creating rule from string expression."""
        rule = Rule.from_string("rsi < 30 and close < 100")
        
        assert len(rule.conditions) == 2
        assert rule.operator == LogicalOperator.AND
        
    def test_rule_serialization(self, sample_data):
        """Test rule to/from dict."""
        rule = Rule(name="Test Rule")
        rule.add_condition('rsi', '>', 70)
        rule.add_condition('volume', '>', 1000000)
        
        # Convert to dict
        rule_dict = rule.to_dict()
        assert rule_dict['name'] == "Test Rule"
        assert len(rule_dict['conditions']) == 2
        
        # Recreate from dict
        new_rule = Rule.from_dict(rule_dict)
        assert new_rule.name == rule.name
        assert len(new_rule.conditions) == len(rule.conditions)


class TestStrategyBuilder:
    """Test StrategyBuilder class."""
    
    def test_strategy_creation(self):
        """Test basic strategy creation."""
        builder = StrategyBuilder("My Strategy")
        builder.set_description("A test strategy")
        
        strategy = builder.build()
        
        assert strategy.name == "My Strategy"
        assert strategy.description == "A test strategy"
        
    def test_add_rules(self):
        """Test adding rules to strategy."""
        builder = StrategyBuilder()
        
        # Add entry rules
        builder.add_entry_rule("rsi < 30", name="RSI Oversold")
        builder.add_entry_rule("close < bb_lower", name="Below BB")
        
        # Add exit rules
        builder.add_exit_rule("rsi > 70", name="RSI Overbought")
        
        strategy = builder.build()
        
        assert len(strategy.entry_rules) == 2
        assert len(strategy.exit_rules) == 1
        assert strategy.entry_rules[0].name == "RSI Oversold"
        
    def test_position_sizing(self):
        """Test position sizing configuration."""
        builder = StrategyBuilder()
        
        builder.set_position_sizing(
            method="percent",
            size=0.1,
            max_position=0.2,
            scale_in=True
        )
        
        strategy = builder.build()
        
        assert strategy.position_sizing.method == "percent"
        assert strategy.position_sizing.size == 0.1
        assert strategy.position_sizing.scale_in is True
        
    def test_risk_management(self):
        """Test risk management configuration."""
        builder = StrategyBuilder()
        
        builder.set_risk_management(
            stop_loss=0.05,
            stop_loss_type="percent",
            take_profit=0.10,
            trailing_stop=0.03,
            max_positions=5
        )
        
        strategy = builder.build()
        
        assert strategy.risk_management.stop_loss == 0.05
        assert strategy.risk_management.take_profit == 0.10
        assert strategy.risk_management.trailing_stop == 0.03
        assert strategy.risk_management.max_positions == 5
        
    def test_strategy_serialization(self):
        """Test saving and loading strategies."""
        import tempfile
        import os
        
        # Create strategy
        builder = StrategyBuilder("Serialization Test")
        builder.add_entry_rule("rsi < 30")
        builder.add_exit_rule("rsi > 70")
        builder.set_risk_management(stop_loss=0.05)
        
        strategy = builder.build()
        
        # Save to YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            strategy.to_yaml(f.name)
            yaml_file = f.name
            
        try:
            # Load from YAML
            loaded_builder = StrategyBuilder.from_yaml(yaml_file)
            loaded_strategy = loaded_builder.build()
            
            assert loaded_strategy.name == strategy.name
            assert len(loaded_strategy.entry_rules) == len(strategy.entry_rules)
            assert loaded_strategy.risk_management.stop_loss == strategy.risk_management.stop_loss
            
        finally:
            os.unlink(yaml_file)
            
    def test_signal_generation(self, sample_data):
        """Test signal generation from strategy."""
        builder = StrategyBuilder()
        builder.add_entry_rule("rsi < 30")
        builder.add_exit_rule("rsi > 70")
        
        # Generate signals
        signals = builder.generate_signals(sample_data)
        
        assert 'entry' in signals.columns
        assert 'exit' in signals.columns
        assert 'position_size' in signals.columns
        assert 'stop_loss' in signals.columns
        assert 'take_profit' in signals.columns
        
        # Signals should be boolean
        assert signals['entry'].dtype == bool
        assert signals['exit'].dtype == bool


class TestPositionSizing:
    """Test position sizing calculations."""
    
    def test_fixed_sizing(self):
        """Test fixed position sizing."""
        from src.strategies.builder import PositionSizing
        
        sizing = PositionSizing(method="fixed", size=100)
        
        shares = sizing.calculate_size(
            capital=100000,
            price=50.0
        )
        
        assert shares == 100
        
    def test_percent_sizing(self):
        """Test percentage-based sizing."""
        from src.strategies.builder import PositionSizing
        
        sizing = PositionSizing(method="percent", size=0.1)
        
        shares = sizing.calculate_size(
            capital=100000,
            price=100.0
        )
        
        assert shares == 100  # 10% of 100k / $100
        
    def test_volatility_sizing(self):
        """Test volatility-based sizing."""
        from src.strategies.builder import PositionSizing
        
        sizing = PositionSizing(method="volatility")
        
        shares = sizing.calculate_size(
            capital=100000,
            price=100.0,
            volatility=0.02  # 2% volatility
        )
        
        # Target risk is 2% of capital = $2000
        # Risk per share = price * volatility = $2
        # Shares = $2000 / $2 = 1000
        assert shares == 1000
        
    def test_kelly_sizing(self):
        """Test Kelly criterion sizing."""
        from src.strategies.builder import PositionSizing
        
        sizing = PositionSizing(method="kelly")
        
        shares = sizing.calculate_size(
            capital=100000,
            price=100.0,
            win_rate=0.6,
            avg_win=100.0,
            avg_loss=50.0
        )
        
        # Kelly fraction = (b*p - q) / b
        # where b = avg_win/avg_loss = 2, p = 0.6, q = 0.4
        # Kelly = (2*0.6 - 0.4) / 2 = 0.4
        # But capped at 25% = 0.25
        # Shares = 100000 * 0.25 / 100 = 250
        assert shares == 250


class TestRiskManagement:
    """Test risk management calculations."""
    
    def test_stop_calculations(self):
        """Test stop loss and take profit calculations."""
        from src.strategies.builder import RiskManagement
        
        risk_mgmt = RiskManagement(
            stop_loss=0.05,
            stop_loss_type="percent",
            take_profit=0.10,
            take_profit_type="percent"
        )
        
        stops = risk_mgmt.calculate_stops(entry_price=100.0)
        
        assert stops['stop_loss'] == 95.0  # 5% below
        assert stops['take_profit'] == 110.0  # 10% above
        
    def test_atr_stops(self):
        """Test ATR-based stops."""
        from src.strategies.builder import RiskManagement
        
        risk_mgmt = RiskManagement(
            stop_loss=2.0,  # 2 ATR
            stop_loss_type="atr",
            take_profit=3.0,  # 3 ATR
            take_profit_type="atr"
        )
        
        stops = risk_mgmt.calculate_stops(entry_price=100.0, atr=2.5)
        
        assert stops['stop_loss'] == 95.0  # 100 - (2 * 2.5)
        assert stops['take_profit'] == 107.5  # 100 + (3 * 2.5)