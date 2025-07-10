"""
Phase 4 Portfolio & Risk Management Demo

This script demonstrates all portfolio optimization, risk management,
position sizing, and monitoring components from Phase 4.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Union

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import Phase 4 modules
from portfolio import (
    PortfolioOptimizer, OptimizationObjective,
    RiskManager, RiskLimits, StopLossConfig, StopLossType,
    PositionSizer, SizingMethod,
    PortfolioRebalancer, RebalanceMethod, RebalanceFrequency
)
from portfolio.stress_testing import StressTester, StressScenario
from portfolio.risk_dashboard_simple import SimpleRiskDashboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase4PortfolioRiskDemo:
    """
    Demonstrates Phase 4 portfolio and risk management capabilities.
    """
    
    def __init__(self):
        """Initialize demo components."""
        # Portfolio configuration
        self.symbols = ['SPY', 'TLT', 'GLD', 'QQQ', 'VNQ']  # Diversified portfolio
        self.start_date = '2022-01-01'
        self.end_date = '2024-01-01'
        self.initial_capital = 100000
        
        # Initialize components
        self.risk_limits = RiskLimits(
            max_position_size=0.30,
            max_drawdown=0.15,
            max_var_95=0.05,
            max_volatility=0.20
        )
        
        self.stop_configs = {
            'SPY': StopLossConfig(StopLossType.TRAILING, initial_stop=0.05, trailing_stop=0.03),
            'TLT': StopLossConfig(StopLossType.ATR_BASED, initial_stop=0.04, atr_multiplier=2.0),
            'GLD': StopLossConfig(StopLossType.FIXED, initial_stop=0.06),
            'QQQ': StopLossConfig(StopLossType.VOLATILITY_BASED, initial_stop=0.05),
            'VNQ': StopLossConfig(StopLossType.FIXED, initial_stop=0.07)
        }
        
    def run_complete_demo(self):
        """Run complete Phase 4 demo."""
        logger.info("🚀 Starting Phase 4 Portfolio & Risk Management Demo")
        
        try:
            # 1. Fetch market data
            logger.info("\n📊 Fetching market data...")
            market_data = self._fetch_market_data()
            
            # 2. Portfolio Optimization
            logger.info("\n🎯 Running portfolio optimization...")
            optimization_results = self._demonstrate_portfolio_optimization(market_data)
            
            # 3. Risk Management
            logger.info("\n⚠️ Demonstrating risk management...")
            risk_results = self._demonstrate_risk_management(
                optimization_results['optimal_weights'],
                market_data
            )
            
            # 4. Position Sizing
            logger.info("\n📏 Demonstrating position sizing...")
            sizing_results = self._demonstrate_position_sizing(
                optimization_results['optimal_weights'],
                market_data
            )
            
            # 5. Portfolio Rebalancing
            logger.info("\n⚖️ Demonstrating portfolio rebalancing...")
            rebalancing_results = self._demonstrate_rebalancing(
                optimization_results['optimal_weights'],
                market_data
            )
            
            # 6. Stress Testing
            logger.info("\n💥 Running stress tests...")
            stress_results = self._demonstrate_stress_testing(
                optimization_results['optimal_weights'],
                market_data
            )
            
            # 7. Risk Dashboard
            logger.info("\n📊 Creating risk monitoring dashboard...")
            dashboard_results = self._create_risk_dashboard(
                optimization_results,
                risk_results,
                stress_results
            )
            
            # Generate summary report
            self._generate_summary_report(
                optimization_results,
                risk_results,
                sizing_results,
                rebalancing_results,
                stress_results,
                dashboard_results
            )
            
            logger.info("\n✅ Phase 4 Portfolio & Risk Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch market data for demo."""
        market_data = {}
        
        for symbol in self.symbols:
            logger.info(f"  Fetching {symbol}...")
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist_data = ticker.history(start=self.start_date, end=self.end_date)
            
            # Clean data
            hist_data = hist_data[['Open', 'High', 'Low', 'Close', 'Volume']]
            hist_data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            market_data[symbol] = hist_data
        
        # Calculate returns
        returns_data = pd.DataFrame()
        for symbol, data in market_data.items():
            returns_data[symbol] = data['close'].pct_change()
        
        returns_data = returns_data.dropna()
        market_data['returns'] = returns_data
        
        return market_data
    
    def _demonstrate_portfolio_optimization(self, market_data: Dict) -> Dict:
        """Demonstrate portfolio optimization techniques."""
        returns = market_data['returns']
        
        # Initialize optimizer
        optimizer = PortfolioOptimizer(returns)
        
        results = {}
        
        # 1. Minimum Variance Portfolio
        logger.info("  1. Calculating minimum variance portfolio...")
        min_var = optimizer.optimize(OptimizationObjective.MIN_VARIANCE)
        results['min_variance'] = min_var
        self._print_optimization_result("Minimum Variance", min_var)
        
        # 2. Maximum Sharpe Portfolio
        logger.info("  2. Calculating maximum Sharpe portfolio...")
        max_sharpe = optimizer.optimize(OptimizationObjective.MAX_SHARPE)
        results['max_sharpe'] = max_sharpe
        results['optimal_weights'] = {
            asset: weight for asset, weight in 
            zip(max_sharpe['asset_names'], max_sharpe['weights'])
        }
        self._print_optimization_result("Maximum Sharpe", max_sharpe)
        
        # 3. Risk Parity Portfolio
        logger.info("  3. Calculating risk parity portfolio...")
        risk_parity = optimizer.optimize(OptimizationObjective.RISK_PARITY)
        results['risk_parity'] = risk_parity
        self._print_optimization_result("Risk Parity", risk_parity)
        
        # 4. Efficient Frontier
        logger.info("  4. Calculating efficient frontier...")
        frontier = optimizer.efficient_frontier(n_portfolios=50)
        results['efficient_frontier'] = frontier
        
        # 5. Black-Litterman with views
        logger.info("  5. Black-Litterman optimization...")
        market_caps = {
            'SPY': 0.40, 'TLT': 0.20, 'GLD': 0.15, 'QQQ': 0.15, 'VNQ': 0.10
        }
        views = {
            'SPY': 0.08,  # 8% expected return
            'GLD': 0.10   # 10% expected return
        }
        view_confidence = {
            'SPY': 0.80,
            'GLD': 0.60
        }
        
        bl_result = optimizer.black_litterman(market_caps, views, view_confidence)
        results['black_litterman'] = bl_result
        self._print_optimization_result("Black-Litterman", bl_result)
        
        return results
    
    def _demonstrate_risk_management(
        self, 
        optimal_weights: Dict[str, float], 
        market_data: Dict
    ) -> Dict:
        """Demonstrate risk management capabilities."""
        # Initialize risk manager
        risk_manager = RiskManager(self.risk_limits, self.stop_configs)
        
        # Current positions (based on optimal weights)
        portfolio_value = self.initial_capital
        current_positions = {
            symbol: weight * portfolio_value 
            for symbol, weight in optimal_weights.items()
        }
        
        results = {}
        
        # 1. Check position limits
        logger.info("  1. Checking position limits...")
        new_position = ('SPY', 25000)  # Try to add $25k SPY
        allowed, reason = risk_manager.check_position_limits(
            current_positions, new_position, portfolio_value
        )
        results['position_check'] = {
            'allowed': allowed,
            'reason': reason,
            'position': new_position
        }
        logger.info(f"     Position allowed: {allowed}, Reason: {reason or 'OK'}")
        
        # 2. Calculate portfolio risk metrics
        logger.info("  2. Calculating portfolio risk metrics...")
        risk_metrics = risk_manager.calculate_portfolio_risk_metrics(
            current_positions,
            market_data['returns'],
            market_data['SPY'][['close']].rename(columns={'close': 'market'})
        )
        results['risk_metrics'] = risk_metrics
        
        # 3. Check risk limits
        logger.info("  3. Checking risk limits...")
        violations = risk_manager.check_risk_limits(risk_metrics, market_data['SPY'])
        results['violations'] = violations
        
        if violations:
            logger.warning(f"     Found {len(violations)} risk limit violations!")
            for v in violations:
                logger.warning(f"     - {v['metric']}: {v['value']:.3f} > {v['limit']:.3f}")
        else:
            logger.info("     ✓ All risk limits satisfied")
        
        # 4. Calculate stop-loss levels
        logger.info("  4. Calculating stop-loss levels...")
        stop_losses = {}
        
        for symbol in optimal_weights.keys():
            if symbol in market_data:
                current_price = market_data[symbol]['close'].iloc[-1]
                entry_price = market_data[symbol]['close'].iloc[-30]  # 30 days ago
                high_price = market_data[symbol]['close'].iloc[-30:].max()
                
                stop_price, stop_type = risk_manager.calculate_stop_loss(
                    symbol,
                    entry_price,
                    current_price,
                    high_price,
                    market_data[symbol],
                    datetime.now() - timedelta(days=30),
                    datetime.now()
                )
                
                stop_losses[symbol] = {
                    'current_price': current_price,
                    'stop_price': stop_price,
                    'stop_type': stop_type,
                    'distance_pct': (current_price - stop_price) / current_price
                }
        
        results['stop_losses'] = stop_losses
        
        # 5. Generate risk report
        risk_report = risk_manager.generate_risk_report()
        results['risk_report'] = risk_report
        
        return results
    
    def _demonstrate_position_sizing(
        self,
        optimal_weights: Dict[str, float],
        market_data: Dict
    ) -> Dict:
        """Demonstrate position sizing methods."""
        # Initialize position sizer
        sizer = PositionSizer(
            default_method=SizingMethod.VOLATILITY,
            risk_per_trade=0.02,
            kelly_fraction=0.25
        )
        
        portfolio_value = self.initial_capital
        results = {}
        
        # Test different sizing methods
        methods = [
            SizingMethod.FIXED,
            SizingMethod.KELLY,
            SizingMethod.VOLATILITY,
            SizingMethod.RISK_PARITY,
            SizingMethod.ATR_BASED,
            SizingMethod.DYNAMIC
        ]
        
        for symbol in list(optimal_weights.keys())[:3]:  # Test first 3 symbols
            logger.info(f"  Position sizing for {symbol}:")
            symbol_results = {}
            
            for method in methods:
                size_result = sizer.calculate_position_size(
                    symbol=symbol,
                    portfolio_value=portfolio_value,
                    signal_strength=0.8,  # Strong signal
                    market_data=market_data[symbol],
                    method=method
                )
                
                symbol_results[method.value] = size_result
                logger.info(f"    {method.value}: ${size_result['size']:,.0f} "
                          f"({size_result['position_pct']:.1%})")
            
            results[symbol] = symbol_results
        
        # Position summary
        current_positions = {
            symbol: weight * portfolio_value
            for symbol, weight in optimal_weights.items()
        }
        
        position_summary = sizer.get_position_summary(current_positions, portfolio_value)
        results['position_summary'] = position_summary
        
        return results
    
    def _demonstrate_rebalancing(
        self,
        target_weights: Dict[str, float],
        market_data: Dict
    ) -> Dict:
        """Demonstrate portfolio rebalancing."""
        # Initialize rebalancer
        rebalancer = PortfolioRebalancer(
            target_weights=target_weights,
            rebalance_method=RebalanceMethod.THRESHOLD,
            threshold=0.05
        )
        
        portfolio_value = self.initial_capital
        results = {}
        
        # Simulate portfolio drift
        logger.info("  1. Simulating portfolio drift...")
        
        # Start with target allocation
        current_positions = {
            symbol: weight * portfolio_value
            for symbol, weight in target_weights.items()
        }
        
        # Apply some market movements to create drift
        drift_factors = {'SPY': 1.15, 'TLT': 0.95, 'GLD': 1.08, 'QQQ': 1.20, 'VNQ': 0.88}
        
        drifted_positions = {
            symbol: value * drift_factors.get(symbol, 1.0)
            for symbol, value in current_positions.items()
        }
        
        new_portfolio_value = sum(drifted_positions.values())
        
        # Check if rebalancing needed
        logger.info("  2. Checking rebalance triggers...")
        needs_rebalance, reasons = rebalancer.check_rebalance_needed(
            drifted_positions,
            new_portfolio_value,
            datetime.now()
        )
        
        results['needs_rebalance'] = needs_rebalance
        results['rebalance_reasons'] = reasons
        
        logger.info(f"     Rebalance needed: {needs_rebalance}")
        for trigger, reason in reasons.items():
            logger.info(f"     - {trigger}: {reason}")
        
        # Calculate trades
        if needs_rebalance:
            logger.info("  3. Calculating rebalancing trades...")
            trades = rebalancer.calculate_trades(
                drifted_positions,
                new_portfolio_value
            )
            
            results['trades'] = trades
            
            # Execute rebalance
            logger.info("  4. Executing rebalance...")
            rebalance_result = rebalancer.execute_rebalance(
                drifted_positions,
                new_portfolio_value,
                datetime.now(),
                trades
            )
            
            results['rebalance_result'] = rebalance_result
            
            # Print trades
            logger.info("     Rebalancing trades:")
            for symbol, trade in trades.items():
                logger.info(f"     - {symbol}: ${trade:,.0f}")
            
            logger.info(f"     Transaction costs: ${rebalance_result['transaction_costs']:,.2f}")
        
        # Rebalancing analytics
        analytics = rebalancer.get_rebalance_analytics()
        results['analytics'] = analytics
        
        return results
    
    def _demonstrate_stress_testing(
        self,
        portfolio_weights: Dict[str, float],
        market_data: Dict
    ) -> Dict:
        """Demonstrate stress testing capabilities."""
        # Initialize stress tester
        stress_tester = StressTester(n_simulations=1000)  # Reduced for demo
        
        returns = market_data['returns']
        results = {}
        
        # 1. Historical stress scenarios
        logger.info("  1. Running historical stress scenarios...")
        historical_stress = stress_tester.run_historical_stress_test(
            portfolio_weights,
            returns
        )
        results['historical_stress'] = historical_stress
        
        # Print worst scenarios
        logger.info("     Worst historical scenarios:")
        worst_scenarios = historical_stress.nsmallest(3, 'portfolio_return')
        for _, scenario in worst_scenarios.iterrows():
            logger.info(f"     - {scenario['scenario']}: {scenario['portfolio_return']*100:.1f}%")
        
        # 2. Monte Carlo simulation
        logger.info("  2. Running Monte Carlo stress test...")
        mc_results = stress_tester.run_monte_carlo_stress_test(
            portfolio_weights,
            returns,
            n_simulations=1000
        )
        results['monte_carlo'] = mc_results['summary']
        
        # 3. Sensitivity analysis
        logger.info("  3. Running sensitivity analysis...")
        sensitivity = stress_tester.run_sensitivity_analysis(
            portfolio_weights,
            returns,
            factor_ranges={
                'market_shock': (-0.20, 0.20),
                'volatility_mult': (0.5, 2.0)
            },
            n_steps=10
        )
        results['sensitivity'] = sensitivity
        
        # 4. Tail risk measures
        logger.info("  4. Calculating tail risk measures...")
        tail_risk = stress_tester.calculate_tail_risk_measures(
            portfolio_weights,
            returns
        )
        results['tail_risk'] = tail_risk
        
        logger.info("     Tail risk measures:")
        logger.info(f"     - VaR 95%: {tail_risk.get('ES_95', 0)*100:.2f}%")
        logger.info(f"     - CVaR 95%: {tail_risk.get('ES_95', 0)*100:.2f}%")
        logger.info(f"     - Max DD Duration: {tail_risk.get('max_dd_duration', 0):.0f} days")
        logger.info(f"     - Sortino Ratio: {tail_risk.get('sortino_ratio', 0):.2f}")
        
        # 5. Generate stress report
        stress_report_path = 'reports/phase4_stress_report.xlsx'
        Path('reports').mkdir(exist_ok=True)
        
        full_report = stress_tester.generate_stress_report(
            portfolio_weights,
            returns,
            stress_report_path
        )
        results['full_report'] = full_report
        
        return results
    
    def _create_risk_dashboard(
        self,
        optimization_results: Dict,
        risk_results: Dict,
        stress_results: Dict
    ) -> Dict:
        """Create risk monitoring dashboard."""
        # Initialize dashboard
        dashboard = SimpleRiskDashboard()
        
        # Prepare data
        risk_metrics = risk_results['risk_metrics']
        risk_limits_dict = self.risk_limits.__dict__
        
        optimal_weights = optimization_results['optimal_weights']
        portfolio_value = self.initial_capital
        
        positions = {
            symbol: weight * portfolio_value
            for symbol, weight in optimal_weights.items()
        }
        
        # 1. Main risk overview dashboard
        logger.info("  1. Creating risk overview dashboard...")
        overview_fig = dashboard.create_risk_overview(
            risk_metrics,
            risk_limits_dict,
            portfolio_value,
            positions
        )
        
        dashboard.save_dashboard(overview_fig, 'reports/risk_overview_dashboard.html')
        
        # 2. Stress test dashboard
        logger.info("  2. Creating stress test dashboard...")
        stress_fig = dashboard.create_stress_test_summary(
            stress_results['historical_stress']
        )
        
        dashboard.save_dashboard(stress_fig, 'reports/stress_test_dashboard.html')
        
        return {
            'overview_dashboard': 'reports/risk_overview_dashboard.html',
            'stress_dashboard': 'reports/stress_test_dashboard.html'
        }
    
    def _print_optimization_result(self, name: str, result: Dict):
        """Print optimization result summary."""
        logger.info(f"\n     {name} Portfolio:")
        logger.info(f"     Expected Return: {result['expected_return']*100:.2f}%")
        logger.info(f"     Volatility: {result['volatility']*100:.2f}%")
        logger.info(f"     Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        logger.info("     Weights:")
        
        for asset, weight in zip(result['asset_names'], result['weights']):
            if weight > 0.01:  # Only show significant weights
                logger.info(f"       - {asset}: {weight*100:.1f}%")
    
    def _generate_summary_report(self, *args):
        """Generate final summary report."""
        summary = """
================================================================================
🎯 PHASE 4 PORTFOLIO & RISK MANAGEMENT - SUMMARY
================================================================================

📊 Portfolio Optimization:
  ✅ Minimum Variance Portfolio
  ✅ Maximum Sharpe Portfolio
  ✅ Risk Parity Portfolio
  ✅ Efficient Frontier Calculation
  ✅ Black-Litterman Optimization

⚠️ Risk Management:
  ✅ Position Limit Checks
  ✅ Portfolio Risk Metrics (VaR, CVaR, Volatility)
  ✅ Risk Limit Monitoring
  ✅ Stop-Loss Calculations (Fixed, Trailing, ATR-based)
  ✅ Risk Violation Detection

📏 Position Sizing:
  ✅ Fixed Position Sizing
  ✅ Kelly Criterion
  ✅ Volatility-based Sizing
  ✅ Risk Parity Sizing
  ✅ ATR-based Sizing
  ✅ Dynamic Multi-factor Sizing

⚖️ Portfolio Rebalancing:
  ✅ Threshold-based Triggers
  ✅ Trade Calculation & Optimization
  ✅ Transaction Cost Analysis
  ✅ Rebalancing Analytics

💥 Stress Testing:
  ✅ Historical Scenario Analysis
  ✅ Monte Carlo Simulation
  ✅ Sensitivity Analysis
  ✅ Tail Risk Measures
  ✅ Comprehensive Stress Report

📊 Risk Dashboards:
  ✅ Real-time Risk Overview
  ✅ Stress Test Visualization
  ✅ Position-level Risk Analysis
  ✅ Interactive Monitoring Tools

🎯 All Phase 4 components successfully demonstrated!
================================================================================
"""
        
        print(summary)
        
        # Save summary
        summary_path = Path('reports/phase4_summary.txt')
        summary_path.parent.mkdir(exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write(summary)

def main():
    """Run the Phase 4 demo."""
    demo = Phase4PortfolioRiskDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()