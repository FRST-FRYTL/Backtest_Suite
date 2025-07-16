"""
Enhanced JSON Export for Trade Reporting

This module provides enhanced JSON export functionality that includes
comprehensive trade price data, stop loss analysis, and risk metrics.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


class EnhancedJSONExporter:
    """Enhanced JSON exporter with comprehensive trade data"""
    
    def __init__(self, trade_reporting_config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced JSON exporter"""
        self.config = trade_reporting_config or {
            'include_detailed_trades': True,
            'include_price_analysis': True,
            'include_stop_loss_analysis': True,
            'include_risk_analysis': True,
            'max_trades_export': 1000,
            'decimal_places': 4
        }
    
    def export_enhanced_report(self, report_data: Dict[str, Any], output_path: Union[str, Path]) -> str:
        """Export enhanced report with comprehensive trade data"""
        
        # Create enhanced report structure
        enhanced_data = {
            'metadata': self._enhance_metadata(report_data.get('metadata', {})),
            'backtest_summary': report_data.get('backtest_summary', {}),
            'sections': report_data.get('sections', {}),
            'trade_analysis': self._create_enhanced_trade_analysis(report_data),
            'price_analysis': self._create_price_analysis(report_data),
            'stop_loss_analysis': self._create_stop_loss_analysis(report_data),
            'risk_analysis': self._create_risk_analysis(report_data),
            'detailed_trades': self._export_detailed_trades(report_data),
            'performance_metrics': self._export_performance_metrics(report_data),
            'visualizations_data': self._export_visualization_data(report_data),
            'export_config': self.config
        }
        
        # Clean and serialize the data
        cleaned_data = self._clean_for_json(enhanced_data)
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(cleaned_data, f, indent=2, default=self._json_serializer)
        
        return str(output_path)
    
    def _enhance_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance metadata with trade reporting information"""
        enhanced = metadata.copy()
        enhanced.update({
            'enhanced_export_version': '1.0.0',
            'export_timestamp': datetime.now().isoformat(),
            'trade_reporting_enabled': True,
            'features_included': [
                'detailed_trades',
                'price_analysis',
                'stop_loss_analysis',
                'risk_analysis',
                'visualization_data'
            ]
        })
        return enhanced
    
    def _create_enhanced_trade_analysis(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced trade analysis section"""
        trade_section = report_data.get('sections', {}).get('tradeanalysis', {})
        
        if not trade_section or 'message' in trade_section:
            return {'message': 'No trades available for analysis'}
        
        # Extract and enhance trade statistics
        enhanced_analysis = {
            'basic_statistics': trade_section.get('trade_statistics', {}),
            'win_loss_analysis': trade_section.get('win_loss_analysis', {}),
            'duration_analysis': trade_section.get('trade_duration_analysis', {}),
            'distribution_analysis': trade_section.get('trade_distribution', {}),
            'profitability_analysis': self._analyze_profitability(trade_section),
            'consistency_analysis': self._analyze_consistency(trade_section),
            'seasonal_analysis': self._analyze_seasonal_patterns(trade_section)
        }
        
        return enhanced_analysis
    
    def _create_price_analysis(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive price analysis"""
        trades_data = self._extract_trades_data(report_data)
        
        if trades_data.empty:
            return {'message': 'No price data available'}
        
        analysis = {
            'entry_price_statistics': self._analyze_entry_prices(trades_data),
            'exit_price_statistics': self._analyze_exit_prices(trades_data),
            'price_movement_analysis': self._analyze_price_movements(trades_data),
            'slippage_analysis': self._analyze_slippage(trades_data),
            'fill_quality_analysis': self._analyze_fill_quality(trades_data),
            'price_improvement_analysis': self._analyze_price_improvement(trades_data)
        }
        
        return analysis
    
    def _create_stop_loss_analysis(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive stop loss analysis"""
        trades_data = self._extract_trades_data(report_data)
        
        if trades_data.empty or 'stop_loss' not in trades_data.columns:
            return {'message': 'No stop loss data available'}
        
        stop_trades = trades_data.dropna(subset=['stop_loss'])
        if stop_trades.empty:
            return {'message': 'No stop loss data available'}
        
        analysis = {
            'stop_loss_usage': self._analyze_stop_usage(stop_trades),
            'stop_loss_effectiveness': self._analyze_stop_effectiveness(stop_trades),
            'stop_distance_analysis': self._analyze_stop_distances(stop_trades),
            'stop_hit_analysis': self._analyze_stop_hits(stop_trades),
            'optimal_stop_analysis': self._analyze_optimal_stops(stop_trades),
            'stop_loss_performance': self._analyze_stop_performance(stop_trades)
        }
        
        return analysis
    
    def _create_risk_analysis(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive risk analysis"""
        trades_data = self._extract_trades_data(report_data)
        
        if trades_data.empty:
            return {'message': 'No risk data available'}
        
        analysis = {
            'risk_per_trade': self._analyze_risk_per_trade(trades_data),
            'position_sizing': self._analyze_position_sizing(trades_data),
            'risk_reward_ratios': self._analyze_risk_reward(trades_data),
            'maximum_risk_analysis': self._analyze_maximum_risk(trades_data),
            'risk_consistency': self._analyze_risk_consistency(trades_data),
            'portfolio_risk_metrics': self._analyze_portfolio_risk(trades_data)
        }
        
        return analysis
    
    def _export_detailed_trades(self, report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Export detailed trade information"""
        trades_data = self._extract_trades_data(report_data)
        
        if trades_data.empty:
            return []
        
        # Limit trades if configured
        max_trades = self.config.get('max_trades_export', 1000)
        if len(trades_data) > max_trades:
            trades_data = trades_data.head(max_trades)
        
        # Convert to list of dictionaries with enhanced information
        detailed_trades = []
        for _, trade in trades_data.iterrows():
            trade_dict = {
                'trade_id': trade.get('trade_id', None),
                'entry_time': self._format_datetime(trade.get('entry_time')),
                'exit_time': self._format_datetime(trade.get('exit_time')),
                'side': trade.get('side', None),
                'size': self._format_number(trade.get('size')),
                'entry_price': self._format_number(trade.get('entry_price')),
                'exit_price': self._format_number(trade.get('exit_price')),
                'stop_loss': self._format_number(trade.get('stop_loss')),
                'take_profit': self._format_number(trade.get('take_profit')),
                'pnl': self._format_number(trade.get('pnl')),
                'duration_hours': self._format_number(trade.get('duration')),
                'exit_reason': trade.get('exit_reason', None),
                'commission': self._format_number(trade.get('commission', 0)),
                'slippage': self._format_number(trade.get('slippage', 0))
            }
            
            # Add calculated fields
            if pd.notna(trade.get('entry_price')) and pd.notna(trade.get('exit_price')):
                entry_price = float(trade['entry_price'])
                exit_price = float(trade['exit_price'])
                trade_dict['price_change_pct'] = self._format_number((exit_price - entry_price) / entry_price * 100)
            
            if pd.notna(trade.get('stop_loss')) and pd.notna(trade.get('entry_price')):
                stop_loss = float(trade['stop_loss'])
                entry_price = float(trade['entry_price'])
                trade_dict['stop_distance_pct'] = self._format_number(abs(stop_loss - entry_price) / entry_price * 100)
            
            if pd.notna(trade.get('size')) and pd.notna(trade.get('entry_price')):
                size = float(trade['size'])
                entry_price = float(trade['entry_price'])
                trade_dict['trade_value'] = self._format_number(size * entry_price)
            
            detailed_trades.append(trade_dict)
        
        return detailed_trades
    
    def _export_performance_metrics(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Export comprehensive performance metrics"""
        metrics = {}
        
        # Extract from backtest summary
        summary = report_data.get('backtest_summary', {})
        if summary:
            metrics.update({
                'performance': summary.get('performance', {}),
                'risk': summary.get('risk', {}),
                'trading': summary.get('trading', {}),
                'evaluation': summary.get('evaluation', {})
            })
        
        # Extract from sections
        sections = report_data.get('sections', {})
        
        # Performance metrics
        if 'performanceanalysis' in sections:
            perf_section = sections['performanceanalysis']
            metrics['detailed_performance'] = {
                'return_analysis': perf_section.get('return_analysis', {}),
                'risk_adjusted_metrics': perf_section.get('risk_adjusted_metrics', {}),
                'rolling_performance': perf_section.get('rolling_performance', {})
            }
        
        # Risk metrics
        if 'riskanalysis' in sections:
            risk_section = sections['riskanalysis']
            metrics['detailed_risk'] = {
                'drawdown_analysis': risk_section.get('drawdown_analysis', {}),
                'volatility_analysis': risk_section.get('volatility_analysis', {}),
                'var_analysis': risk_section.get('var_analysis', {})
            }
        
        return metrics
    
    def _export_visualization_data(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Export data used for visualizations"""
        viz_data = {}
        
        # Extract visualization data from sections
        visualizations = report_data.get('visualizations', {})
        
        for viz_name, viz_info in visualizations.items():
            if isinstance(viz_info, dict) and 'data' in viz_info:
                viz_data[viz_name] = viz_info['data']
        
        return viz_data
    
    def _extract_trades_data(self, report_data: Dict[str, Any]) -> pd.DataFrame:
        """Extract trades data from report"""
        # Try to get trades from various sources
        trades_data = pd.DataFrame()
        
        # From trade analysis section
        trade_section = report_data.get('sections', {}).get('tradeanalysis', {})
        if 'detailed_trades' in trade_section:
            trades_data = pd.DataFrame(trade_section['detailed_trades'])
        
        # From raw backtest results (if available)
        if trades_data.empty:
            raw_results = report_data.get('raw_backtest_results', {})
            if 'trades' in raw_results:
                trades_data = raw_results['trades']
        
        return trades_data
    
    def _analyze_profitability(self, trade_section: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trade profitability patterns"""
        # Implementation would analyze profitability patterns
        return {
            'profit_distribution': 'Analysis pending',
            'profit_streaks': 'Analysis pending',
            'loss_streaks': 'Analysis pending'
        }
    
    def _analyze_consistency(self, trade_section: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trading consistency"""
        return {
            'consistency_score': 'Analysis pending',
            'performance_stability': 'Analysis pending',
            'monthly_consistency': 'Analysis pending'
        }
    
    def _analyze_seasonal_patterns(self, trade_section: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze seasonal trading patterns"""
        return {
            'monthly_patterns': 'Analysis pending',
            'weekly_patterns': 'Analysis pending',
            'daily_patterns': 'Analysis pending'
        }
    
    def _analyze_entry_prices(self, trades_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze entry price statistics"""
        if 'entry_price' not in trades_data.columns:
            return {'message': 'No entry price data available'}
        
        entry_prices = trades_data['entry_price'].dropna()
        if entry_prices.empty:
            return {'message': 'No entry price data available'}
        
        return {
            'mean': self._format_number(entry_prices.mean()),
            'median': self._format_number(entry_prices.median()),
            'std': self._format_number(entry_prices.std()),
            'min': self._format_number(entry_prices.min()),
            'max': self._format_number(entry_prices.max()),
            'quartiles': {
                'q1': self._format_number(entry_prices.quantile(0.25)),
                'q3': self._format_number(entry_prices.quantile(0.75))
            }
        }
    
    def _analyze_exit_prices(self, trades_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze exit price statistics"""
        if 'exit_price' not in trades_data.columns:
            return {'message': 'No exit price data available'}
        
        exit_prices = trades_data['exit_price'].dropna()
        if exit_prices.empty:
            return {'message': 'No exit price data available'}
        
        return {
            'mean': self._format_number(exit_prices.mean()),
            'median': self._format_number(exit_prices.median()),
            'std': self._format_number(exit_prices.std()),
            'min': self._format_number(exit_prices.min()),
            'max': self._format_number(exit_prices.max()),
            'quartiles': {
                'q1': self._format_number(exit_prices.quantile(0.25)),
                'q3': self._format_number(exit_prices.quantile(0.75))
            }
        }
    
    def _analyze_price_movements(self, trades_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price movement patterns"""
        if 'entry_price' not in trades_data.columns or 'exit_price' not in trades_data.columns:
            return {'message': 'Insufficient price data'}
        
        price_data = trades_data.dropna(subset=['entry_price', 'exit_price'])
        if price_data.empty:
            return {'message': 'No complete price data available'}
        
        price_changes = (price_data['exit_price'] - price_data['entry_price']) / price_data['entry_price'] * 100
        
        return {
            'average_price_change_pct': self._format_number(price_changes.mean()),
            'median_price_change_pct': self._format_number(price_changes.median()),
            'price_change_volatility': self._format_number(price_changes.std()),
            'positive_moves': len(price_changes[price_changes > 0]),
            'negative_moves': len(price_changes[price_changes < 0]),
            'largest_positive_move': self._format_number(price_changes.max()),
            'largest_negative_move': self._format_number(price_changes.min())
        }
    
    def _analyze_slippage(self, trades_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze slippage patterns"""
        if 'slippage' not in trades_data.columns:
            return {'message': 'No slippage data available'}
        
        slippage_data = trades_data['slippage'].dropna()
        if slippage_data.empty:
            return {'message': 'No slippage data available'}
        
        return {
            'average_slippage': self._format_number(slippage_data.mean()),
            'median_slippage': self._format_number(slippage_data.median()),
            'max_slippage': self._format_number(slippage_data.max()),
            'total_slippage_cost': self._format_number(slippage_data.sum())
        }
    
    def _analyze_fill_quality(self, trades_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze fill quality metrics"""
        return {
            'fill_rate': 'Analysis pending',
            'partial_fills': 'Analysis pending',
            'rejected_orders': 'Analysis pending'
        }
    
    def _analyze_price_improvement(self, trades_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price improvement patterns"""
        return {
            'improvement_rate': 'Analysis pending',
            'average_improvement': 'Analysis pending',
            'improvement_by_side': 'Analysis pending'
        }
    
    def _analyze_stop_usage(self, stop_trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze stop loss usage patterns"""
        total_trades = len(stop_trades)
        
        return {
            'stop_loss_usage_rate': self._format_number(100.0),  # 100% since we filtered for stop trades
            'trades_with_stops': total_trades,
            'stop_types': 'Analysis pending'
        }
    
    def _analyze_stop_effectiveness(self, stop_trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze stop loss effectiveness"""
        if 'exit_reason' not in stop_trades.columns:
            return {'message': 'No exit reason data available'}
        
        stop_hits = stop_trades[stop_trades['exit_reason'].str.contains('stop', case=False, na=False)]
        hit_rate = len(stop_hits) / len(stop_trades) * 100
        
        return {
            'stop_hit_rate': self._format_number(hit_rate),
            'stop_hits': len(stop_hits),
            'total_with_stops': len(stop_trades),
            'effectiveness_score': 'Analysis pending'
        }
    
    def _analyze_stop_distances(self, stop_trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze stop loss distance patterns"""
        if 'entry_price' not in stop_trades.columns or 'stop_loss' not in stop_trades.columns:
            return {'message': 'Insufficient stop loss data'}
        
        stop_distances = abs(stop_trades['stop_loss'] - stop_trades['entry_price']) / stop_trades['entry_price'] * 100
        
        return {
            'average_stop_distance_pct': self._format_number(stop_distances.mean()),
            'median_stop_distance_pct': self._format_number(stop_distances.median()),
            'std_stop_distance': self._format_number(stop_distances.std()),
            'min_stop_distance': self._format_number(stop_distances.min()),
            'max_stop_distance': self._format_number(stop_distances.max())
        }
    
    def _analyze_stop_hits(self, stop_trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze stop hit patterns"""
        return {
            'stop_hit_distribution': 'Analysis pending',
            'time_to_stop_hit': 'Analysis pending',
            'stop_hit_by_market_conditions': 'Analysis pending'
        }
    
    def _analyze_optimal_stops(self, stop_trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze optimal stop placement"""
        return {
            'optimal_stop_distance': 'Analysis pending',
            'stop_optimization_score': 'Analysis pending',
            'recommended_stops': 'Analysis pending'
        }
    
    def _analyze_stop_performance(self, stop_trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze stop loss performance"""
        return {
            'stop_loss_saved_capital': 'Analysis pending',
            'stop_vs_no_stop_performance': 'Analysis pending',
            'stop_loss_efficiency': 'Analysis pending'
        }
    
    def _analyze_risk_per_trade(self, trades_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze risk per trade metrics"""
        if 'entry_price' not in trades_data.columns or 'stop_loss' not in trades_data.columns:
            return {'message': 'Insufficient risk data'}
        
        risk_data = trades_data.dropna(subset=['entry_price', 'stop_loss'])
        if risk_data.empty:
            return {'message': 'No complete risk data available'}
        
        risk_pct = abs(risk_data['stop_loss'] - risk_data['entry_price']) / risk_data['entry_price'] * 100
        
        return {
            'average_risk_per_trade_pct': self._format_number(risk_pct.mean()),
            'median_risk_per_trade_pct': self._format_number(risk_pct.median()),
            'risk_volatility': self._format_number(risk_pct.std()),
            'max_risk_per_trade': self._format_number(risk_pct.max()),
            'min_risk_per_trade': self._format_number(risk_pct.min())
        }
    
    def _analyze_position_sizing(self, trades_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze position sizing patterns"""
        if 'size' not in trades_data.columns:
            return {'message': 'No position size data available'}
        
        sizes = trades_data['size'].dropna()
        if sizes.empty:
            return {'message': 'No position size data available'}
        
        return {
            'average_position_size': self._format_number(sizes.mean()),
            'median_position_size': self._format_number(sizes.median()),
            'position_size_volatility': self._format_number(sizes.std()),
            'max_position_size': self._format_number(sizes.max()),
            'min_position_size': self._format_number(sizes.min())
        }
    
    def _analyze_risk_reward(self, trades_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze risk/reward ratios"""
        return {
            'average_risk_reward_ratio': 'Analysis pending',
            'risk_reward_distribution': 'Analysis pending',
            'optimal_risk_reward': 'Analysis pending'
        }
    
    def _analyze_maximum_risk(self, trades_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze maximum risk exposure"""
        return {
            'maximum_concurrent_risk': 'Analysis pending',
            'risk_concentration': 'Analysis pending',
            'risk_diversification': 'Analysis pending'
        }
    
    def _analyze_risk_consistency(self, trades_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze risk consistency"""
        return {
            'risk_consistency_score': 'Analysis pending',
            'risk_stability': 'Analysis pending',
            'risk_discipline': 'Analysis pending'
        }
    
    def _analyze_portfolio_risk(self, trades_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze portfolio-level risk metrics"""
        return {
            'portfolio_var': 'Analysis pending',
            'portfolio_beta': 'Analysis pending',
            'correlation_risk': 'Analysis pending'
        }
    
    def _format_datetime(self, dt: Any) -> Optional[str]:
        """Format datetime for JSON export"""
        if pd.isna(dt):
            return None
        if isinstance(dt, (pd.Timestamp, datetime)):
            return dt.isoformat()
        return str(dt)
    
    def _format_number(self, num: Any) -> Optional[float]:
        """Format number for JSON export"""
        if pd.isna(num):
            return None
        try:
            return round(float(num), self.config.get('decimal_places', 4))
        except (ValueError, TypeError):
            return None
    
    def _clean_for_json(self, data: Any) -> Any:
        """Clean data for JSON serialization"""
        if isinstance(data, dict):
            return {k: self._clean_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_for_json(item) for item in data]
        elif isinstance(data, pd.DataFrame):
            return data.to_dict('records')
        elif isinstance(data, pd.Series):
            return data.to_dict()
        elif isinstance(data, (pd.Timestamp, datetime)):
            return data.isoformat()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif pd.isna(data):
            return None
        else:
            return data
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for complex objects"""
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif pd.isna(obj):
            return None
        else:
            return str(obj)


def create_enhanced_json_export(report_data: Dict[str, Any], 
                               output_path: Union[str, Path],
                               config: Optional[Dict[str, Any]] = None) -> str:
    """
    Convenience function to create enhanced JSON export
    
    Args:
        report_data: Complete report data dictionary
        output_path: Path where to save the JSON file
        config: Optional configuration for the export
    
    Returns:
        String path to the created JSON file
    """
    exporter = EnhancedJSONExporter(config)
    return exporter.export_enhanced_report(report_data, output_path)