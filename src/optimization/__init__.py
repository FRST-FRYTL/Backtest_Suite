"""Strategy optimization module."""

try:
    from .optimizer import StrategyOptimizer
    from .walk_forward import WalkForwardAnalysis
    # Export StrategyOptimizer as Optimizer for backward compatibility
    Optimizer = StrategyOptimizer
    __all__ = ["StrategyOptimizer", "Optimizer", "WalkForwardAnalysis"]
except ImportError:
    # Optional imports - may not be available in all contexts
    __all__ = []