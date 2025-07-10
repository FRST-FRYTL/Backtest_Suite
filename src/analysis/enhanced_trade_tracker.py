"""
Enhanced Trade Tracking and Analysis System

This module provides comprehensive trade tracking with detailed signal breakdown,
performance attribution, and market context analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradeType(Enum):
    """Trade type classification"""
    BUY = "BUY"
    SELL = "SELL"

class ExitReason(Enum):
    """Exit reason classification"""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TIME_EXIT = "time_exit"
    SIGNAL_EXIT = "signal_exit"
    CONFLUENCE_EXIT = "confluence_exit"

@dataclass
class TradeEntry:
    """Comprehensive trade entry record"""
    trade_id: str
    timestamp: pd.Timestamp
    symbol: str
    action: TradeType
    price: float
    shares: float
    position_size_usd: float
    position_size_pct: float
    
    # Confluence Analysis
    confluence_score: float
    timeframe_scores: Dict[str, float]
    
    # Indicator Values at Entry
    indicators: Dict[str, float]
    
    # Signal Components
    signal_components: Dict[str, float]
    
    # Risk Management
    stop_loss: float
    take_profit: float
    max_hold_days: int
    
    # Execution Details
    intended_price: float
    execution_price: float
    slippage: float
    commission: float
    spread_cost: float
    total_costs: float
    
    # Market Context
    market_context: Dict[str, Any]

@dataclass
class TradeExit:
    """Comprehensive trade exit record"""
    trade_id: str
    timestamp: pd.Timestamp
    symbol: str
    action: TradeType
    price: float
    shares: float
    proceeds: float
    
    # Performance
    gross_pnl: float
    gross_return_pct: float
    net_pnl: float
    net_return_pct: float
    hold_days: int
    
    # Exit Details
    exit_reason: ExitReason
    exit_trigger: str
    
    # Confluence at Exit
    exit_confluence: float
    confluence_change: float
    
    # Market Context
    market_return: float
    alpha: float
    sector_performance: Dict[str, float]
    vix_change: float

@dataclass
class TradeAnalysis:
    """Complete trade analysis combining entry and exit"""
    entry: TradeEntry
    exit: TradeExit
    
    # Performance Metrics
    total_return: float
    annual_return: float
    risk_adjusted_return: float
    information_ratio: float
    
    # Attribution Analysis
    confluence_attribution: Dict[str, float]
    timeframe_attribution: Dict[str, float]
    component_attribution: Dict[str, float]
    
    # Risk Metrics
    max_adverse_excursion: float
    max_favorable_excursion: float
    drawdown_during_trade: float
    
    # Market Comparison
    vs_buy_hold: float
    vs_benchmark: float
    market_beta: float

class EnhancedTradeTracker:
    """
    Comprehensive trade tracking and analysis system.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the enhanced trade tracker.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.active_trades: Dict[str, TradeEntry] = {}
        self.completed_trades: List[TradeAnalysis] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.portfolio_value_history: List[Dict[str, Any]] = []
        self.daily_pnl: pd.Series = pd.Series(dtype=float)
        
        # Trade ID counter
        self._trade_counter = 0
    
    def generate_trade_id(self, symbol: str) -> str:
        """Generate unique trade ID."""
        self._trade_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{symbol}_{timestamp}_{self._trade_counter:04d}"
    
    def record_trade_entry(
        self,
        symbol: str,
        price: float,
        shares: float,
        confluence_score: float,
        timeframe_scores: Dict[str, float],
        indicators: Dict[str, float],
        signal_components: Dict[str, float],
        stop_loss: float,
        take_profit: float,
        position_size_pct: float = 0.2,
        intended_price: Optional[float] = None,
        commission: float = 0.0,
        spread_cost: float = 0.0,
        market_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a trade entry with comprehensive details.
        
        Args:
            symbol: Symbol being traded
            price: Execution price
            shares: Number of shares
            confluence_score: Overall confluence score
            timeframe_scores: Scores by timeframe
            indicators: Indicator values at entry
            signal_components: Signal component scores
            stop_loss: Stop loss price
            take_profit: Take profit price
            position_size_pct: Position size as percentage of portfolio
            intended_price: Intended entry price
            commission: Commission cost
            spread_cost: Bid-ask spread cost
            market_context: Market context data
            
        Returns:
            Trade ID
        """
        trade_id = self.generate_trade_id(symbol)
        
        # Calculate costs
        position_size_usd = shares * price
        slippage = price - (intended_price or price)
        total_costs = commission + spread_cost + abs(slippage * shares)
        
        # Create trade entry record
        entry = TradeEntry(
            trade_id=trade_id,
            timestamp=pd.Timestamp.now(),
            symbol=symbol,
            action=TradeType.BUY,
            price=price,
            shares=shares,
            position_size_usd=position_size_usd,
            position_size_pct=position_size_pct,
            confluence_score=confluence_score,
            timeframe_scores=timeframe_scores,
            indicators=indicators,
            signal_components=signal_components,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_hold_days=30,  # Default
            intended_price=intended_price or price,
            execution_price=price,
            slippage=slippage,
            commission=commission,
            spread_cost=spread_cost,
            total_costs=total_costs,
            market_context=market_context or {}
        )
        
        # Store active trade
        self.active_trades[trade_id] = entry
        
        # Add to history
        self.trade_history.append({
            'trade_id': trade_id,
            'timestamp': entry.timestamp,
            'type': 'ENTRY',
            'data': asdict(entry)
        })
        
        logger.info(f"Recorded trade entry: {trade_id} - {symbol} @ {price}")
        return trade_id
    
    def record_trade_exit(
        self,
        trade_id: str,
        price: float,
        exit_reason: ExitReason,
        exit_trigger: str,
        exit_confluence: float,
        market_return: float = 0.0,
        sector_performance: Optional[Dict[str, float]] = None,
        vix_change: float = 0.0
    ) -> Optional[TradeAnalysis]:
        """
        Record a trade exit and perform comprehensive analysis.
        
        Args:
            trade_id: Trade ID to exit
            price: Exit price
            exit_reason: Reason for exit
            exit_trigger: Specific exit trigger
            exit_confluence: Confluence score at exit
            market_return: Market return during trade
            sector_performance: Sector performance data
            vix_change: VIX change during trade
            
        Returns:
            Complete trade analysis or None if trade not found
        """
        if trade_id not in self.active_trades:
            logger.warning(f"Trade ID {trade_id} not found in active trades")
            return None
        
        entry = self.active_trades[trade_id]
        
        # Calculate exit metrics
        proceeds = entry.shares * price
        gross_pnl = proceeds - entry.position_size_usd
        gross_return_pct = (gross_pnl / entry.position_size_usd) * 100
        net_pnl = gross_pnl - entry.total_costs
        net_return_pct = (net_pnl / entry.position_size_usd) * 100
        
        hold_days = (pd.Timestamp.now() - entry.timestamp).days
        confluence_change = exit_confluence - entry.confluence_score
        alpha = gross_return_pct - market_return
        
        # Create exit record
        exit_record = TradeExit(
            trade_id=trade_id,
            timestamp=pd.Timestamp.now(),
            symbol=entry.symbol,
            action=TradeType.SELL,
            price=price,
            shares=entry.shares,
            proceeds=proceeds,
            gross_pnl=gross_pnl,
            gross_return_pct=gross_return_pct,
            net_pnl=net_pnl,
            net_return_pct=net_return_pct,
            hold_days=hold_days,
            exit_reason=exit_reason,
            exit_trigger=exit_trigger,
            exit_confluence=exit_confluence,
            confluence_change=confluence_change,
            market_return=market_return,
            alpha=alpha,
            sector_performance=sector_performance or {},
            vix_change=vix_change
        )
        
        # Perform comprehensive trade analysis
        trade_analysis = self._analyze_completed_trade(entry, exit_record)
        
        # Store completed trade
        self.completed_trades.append(trade_analysis)
        del self.active_trades[trade_id]
        
        # Add to history
        self.trade_history.append({
            'trade_id': trade_id,
            'timestamp': exit_record.timestamp,
            'type': 'EXIT',
            'data': asdict(exit_record)
        })
        
        logger.info(f"Recorded trade exit: {trade_id} - {entry.symbol} @ {price} "
                   f"({net_return_pct:.2f}% return)")
        
        return trade_analysis
    
    def _analyze_completed_trade(
        self,
        entry: TradeEntry,
        exit: TradeExit
    ) -> TradeAnalysis:
        """
        Perform comprehensive analysis of completed trade.
        
        Args:
            entry: Trade entry record
            exit: Trade exit record
            
        Returns:
            Complete trade analysis
        """
        # Calculate performance metrics
        total_return = exit.net_return_pct
        hold_years = exit.hold_days / 365.25
        annual_return = (1 + total_return/100) ** (1/hold_years) - 1 if hold_years > 0 else 0
        
        # Risk-adjusted return (simplified)
        risk_adjusted_return = total_return / max(1, abs(entry.confluence_score - 0.5) * 100)
        
        # Information ratio (simplified)
        information_ratio = exit.alpha / max(1, abs(exit.vix_change)) if exit.vix_change != 0 else 0
        
        # Attribution analysis
        confluence_attribution = self._calculate_confluence_attribution(entry, exit)
        timeframe_attribution = self._calculate_timeframe_attribution(entry, exit)
        component_attribution = self._calculate_component_attribution(entry, exit)
        
        # Risk metrics (would need price history for accurate calculation)
        max_adverse_excursion = min(0, total_return * 0.5)  # Simplified
        max_favorable_excursion = max(0, total_return * 1.2)  # Simplified
        drawdown_during_trade = abs(max_adverse_excursion)
        
        # Market comparison
        vs_buy_hold = exit.alpha
        vs_benchmark = exit.alpha  # Simplified
        market_beta = 1.0  # Would need historical correlation
        
        return TradeAnalysis(
            entry=entry,
            exit=exit,
            total_return=total_return,
            annual_return=annual_return * 100,
            risk_adjusted_return=risk_adjusted_return,
            information_ratio=information_ratio,
            confluence_attribution=confluence_attribution,
            timeframe_attribution=timeframe_attribution,
            component_attribution=component_attribution,
            max_adverse_excursion=max_adverse_excursion,
            max_favorable_excursion=max_favorable_excursion,
            drawdown_during_trade=drawdown_during_trade,
            vs_buy_hold=vs_buy_hold,
            vs_benchmark=vs_benchmark,
            market_beta=market_beta
        )
    
    def _calculate_confluence_attribution(
        self,
        entry: TradeEntry,
        exit: TradeExit
    ) -> Dict[str, float]:
        """Calculate how confluence score contributed to performance."""
        base_performance = exit.net_return_pct
        confluence_factor = entry.confluence_score
        
        # Simple attribution model
        confluence_contribution = base_performance * confluence_factor
        other_contribution = base_performance - confluence_contribution
        
        return {
            'confluence_contribution': confluence_contribution,
            'other_factors': other_contribution,
            'confluence_efficiency': confluence_contribution / max(0.01, confluence_factor)
        }
    
    def _calculate_timeframe_attribution(
        self,
        entry: TradeEntry,
        exit: TradeExit
    ) -> Dict[str, float]:
        """Calculate how each timeframe contributed to performance."""
        base_performance = exit.net_return_pct
        attribution = {}
        
        total_weight = sum(entry.timeframe_scores.values())
        if total_weight > 0:
            for tf, score in entry.timeframe_scores.items():
                weight = score / total_weight
                attribution[f'{tf}_contribution'] = base_performance * weight
        
        return attribution
    
    def _calculate_component_attribution(
        self,
        entry: TradeEntry,
        exit: TradeExit
    ) -> Dict[str, float]:
        """Calculate how each signal component contributed to performance."""
        base_performance = exit.net_return_pct
        attribution = {}
        
        total_weight = sum(entry.signal_components.values())
        if total_weight > 0:
            for component, score in entry.signal_components.items():
                weight = score / total_weight
                attribution[f'{component}_contribution'] = base_performance * weight
        
        return attribution
    
    def get_trade_summary_statistics(self) -> Dict[str, Union[float, int]]:
        """
        Get comprehensive trade summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.completed_trades:
            return {}
        
        # Extract returns
        returns = [trade.total_return for trade in self.completed_trades]
        win_returns = [r for r in returns if r > 0]
        loss_returns = [r for r in returns if r <= 0]
        
        # Basic statistics
        total_trades = len(self.completed_trades)
        winning_trades = len(win_returns)
        losing_trades = len(loss_returns)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Return statistics
        avg_return = np.mean(returns) if returns else 0
        avg_win = np.mean(win_returns) if win_returns else 0
        avg_loss = np.mean(loss_returns) if loss_returns else 0
        
        # Risk metrics
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else 0
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
        
        # Hold time analysis
        hold_times = [trade.exit.hold_days for trade in self.completed_trades]
        avg_hold_time = np.mean(hold_times) if hold_times else 0
        
        # Confluence analysis
        confluence_scores = [trade.entry.confluence_score for trade in self.completed_trades]
        avg_confluence = np.mean(confluence_scores) if confluence_scores else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_hold_time': avg_hold_time,
            'avg_confluence_score': avg_confluence,
            'best_trade': max(returns) if returns else 0,
            'worst_trade': min(returns) if returns else 0,
            'total_return': sum(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        }
    
    def analyze_performance_by_confluence(self) -> pd.DataFrame:
        """
        Analyze performance by confluence score ranges.
        
        Returns:
            DataFrame with performance by confluence ranges
        """
        if not self.completed_trades:
            return pd.DataFrame()
        
        # Create confluence ranges
        ranges = [
            (0.0, 0.5, 'Very Low'),
            (0.5, 0.6, 'Low'),
            (0.6, 0.7, 'Medium'),
            (0.7, 0.8, 'High'),
            (0.8, 1.0, 'Very High')
        ]
        
        analysis_data = []
        
        for min_score, max_score, label in ranges:
            trades_in_range = [
                trade for trade in self.completed_trades
                if min_score <= trade.entry.confluence_score < max_score
            ]
            
            if trades_in_range:
                returns = [trade.total_return for trade in trades_in_range]
                win_returns = [r for r in returns if r > 0]
                
                analysis_data.append({
                    'confluence_range': label,
                    'min_score': min_score,
                    'max_score': max_score,
                    'trade_count': len(trades_in_range),
                    'avg_return': np.mean(returns),
                    'win_rate': (len(win_returns) / len(returns)) * 100,
                    'best_trade': max(returns),
                    'worst_trade': min(returns),
                    'std_dev': np.std(returns),
                    'total_return': sum(returns)
                })
        
        return pd.DataFrame(analysis_data)
    
    def analyze_performance_by_timeframe(self) -> pd.DataFrame:
        """
        Analyze which timeframes contribute most to successful trades.
        
        Returns:
            DataFrame with timeframe contribution analysis
        """
        if not self.completed_trades:
            return pd.DataFrame()
        
        # Get all unique timeframes
        all_timeframes = set()
        for trade in self.completed_trades:
            all_timeframes.update(trade.entry.timeframe_scores.keys())
        
        analysis_data = []
        
        for tf in all_timeframes:
            tf_data = []
            
            for trade in self.completed_trades:
                if tf in trade.entry.timeframe_scores:
                    tf_data.append({
                        'score': trade.entry.timeframe_scores[tf],
                        'return': trade.total_return,
                        'winning': trade.total_return > 0
                    })
            
            if tf_data:
                scores = [d['score'] for d in tf_data]
                returns = [d['return'] for d in tf_data]
                win_rate = sum(1 for d in tf_data if d['winning']) / len(tf_data) * 100
                
                # Correlation between score and return
                correlation = np.corrcoef(scores, returns)[0, 1] if len(scores) > 1 else 0
                
                analysis_data.append({
                    'timeframe': tf,
                    'avg_score': np.mean(scores),
                    'avg_return': np.mean(returns),
                    'win_rate': win_rate,
                    'score_return_correlation': correlation,
                    'trade_count': len(tf_data),
                    'contribution_strength': abs(correlation) * win_rate / 100
                })
        
        df = pd.DataFrame(analysis_data)
        return df.sort_values('contribution_strength', ascending=False) if not df.empty else df
    
    def export_trade_details(self, filename: Optional[str] = None) -> str:
        """
        Export detailed trade records to CSV.
        
        Args:
            filename: Optional filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_details_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Prepare detailed trade data
        detailed_data = []
        
        for trade in self.completed_trades:
            entry = trade.entry
            exit = trade.exit
            
            row = {
                'trade_id': entry.trade_id,
                'symbol': entry.symbol,
                'entry_date': entry.timestamp,
                'exit_date': exit.timestamp,
                'entry_price': entry.price,
                'exit_price': exit.price,
                'shares': entry.shares,
                'hold_days': exit.hold_days,
                'gross_return_pct': exit.gross_return_pct,
                'net_return_pct': exit.net_return_pct,
                'confluence_score': entry.confluence_score,
                'exit_reason': exit.exit_reason.value,
                'alpha': exit.alpha,
                'position_size_pct': entry.position_size_pct,
                'stop_loss': entry.stop_loss,
                'take_profit': entry.take_profit,
                'total_costs': entry.total_costs,
                'slippage': entry.slippage
            }
            
            # Add timeframe scores
            for tf, score in entry.timeframe_scores.items():
                row[f'tf_score_{tf}'] = score
            
            # Add signal components
            for component, score in entry.signal_components.items():
                row[f'signal_{component}'] = score
            
            # Add indicator values
            for indicator, value in entry.indicators.items():
                row[f'indicator_{indicator}'] = value
            
            detailed_data.append(row)
        
        # Create DataFrame and export
        df = pd.DataFrame(detailed_data)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Exported {len(detailed_data)} trade records to {filepath}")
        return str(filepath)
    
    def generate_trade_analysis_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive trade analysis report.
        
        Returns:
            Dictionary with complete analysis
        """
        summary_stats = self.get_trade_summary_statistics()
        confluence_analysis = self.analyze_performance_by_confluence()
        timeframe_analysis = self.analyze_performance_by_timeframe()
        
        report = {
            'summary_statistics': summary_stats,
            'confluence_analysis': confluence_analysis.to_dict('records') if not confluence_analysis.empty else [],
            'timeframe_analysis': timeframe_analysis.to_dict('records') if not timeframe_analysis.empty else [],
            'active_trades': len(self.active_trades),
            'completed_trades': len(self.completed_trades),
            'total_trade_history': len(self.trade_history),
            'report_generated': datetime.now().isoformat()
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"trade_analysis_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Generated trade analysis report: {report_file}")
        return report