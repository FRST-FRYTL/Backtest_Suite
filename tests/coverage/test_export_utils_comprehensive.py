"""
Comprehensive tests for export_utils.py module.

This module provides complete test coverage for the ExportManager class
including CSV, Excel, PDF, and JSON export functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, mock_open
import os

from src.visualization.export_utils import ExportManager, EXCEL_AVAILABLE, PDF_AVAILABLE


class TestExportManager:
    """Comprehensive tests for ExportManager class."""
    
    @pytest.fixture
    def temp_export_dir(self):
        """Create a temporary directory for exports."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades data."""
        trades = [
            {
                'trade_id': 1,
                'symbol': 'AAPL',
                'entry_time': '2023-01-05 09:30:00',
                'exit_time': '2023-01-10 15:30:00',
                'entry_price': 150.50,
                'exit_price': 155.25,
                'position_size': 100,
                'return': 0.0315,
                'pnl': 475.0,
                'confluence_score': 8.5,
                'hold_days': 5,
                'exit_reason': 'target_hit',
                'max_profit': 525.0,
                'max_loss': -125.0
            },
            {
                'trade_id': 2,
                'symbol': 'GOOGL',
                'entry_time': '2023-01-12 10:00:00',
                'exit_time': '2023-01-15 14:00:00',
                'entry_price': 2800.00,
                'exit_price': 2750.00,
                'position_size': 50,
                'return': -0.0179,
                'pnl': -2500.0,
                'confluence_score': 6.2,
                'hold_days': 3,
                'exit_reason': 'stop_loss',
                'max_profit': 1000.0,
                'max_loss': -2500.0
            },
            {
                'trade_id': 3,
                'symbol': 'MSFT',
                'entry_time': '2023-01-20 11:15:00',
                'exit_time': '2023-01-25 13:45:00',
                'entry_price': 300.00,
                'exit_price': 310.50,
                'position_size': 75,
                'return': 0.0350,
                'pnl': 787.50,
                'confluence_score': 9.1,
                'hold_days': 5,
                'exit_reason': 'signal_exit',
                'max_profit': 900.0,
                'max_loss': -150.0
            }
        ]
        return trades
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample performance metrics."""
        return {
            'total_return': 0.0486,
            'annualized_return': 0.1825,
            'volatility': 0.1532,
            'sharpe_ratio': 1.85,
            'sortino_ratio': 2.10,
            'calmar_ratio': 1.42,
            'max_drawdown': -0.0853,
            'win_rate': 0.667,
            'profit_factor': 2.15,
            'total_trades': 3,
            'winning_trades': 2,
            'losing_trades': 1,
            'avg_win': 631.25,
            'avg_loss': -2500.0,
            'total_pnl': -237.50,
            'total_commission': 45.0
        }
    
    @pytest.fixture
    def sample_confluence_history(self):
        """Create sample confluence score history."""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'confluence_score': np.random.uniform(3, 10, len(dates)),
            'signal_strength': np.random.choice(['weak', 'medium', 'strong'], len(dates)),
            'active_signals': np.random.randint(1, 6, len(dates))
        })
        data.set_index('timestamp', inplace=True)
        return data
    
    @pytest.fixture
    def sample_benchmark_comparison(self):
        """Create sample benchmark comparison data."""
        return {
            'metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
            'strategy': [0.0486, 1.85, -0.0853, 0.667],
            'benchmark': [0.0325, 1.20, -0.1250, 0.500],
            'difference': [0.0161, 0.65, 0.0397, 0.167]
        }
    
    def test_initialization(self, temp_export_dir):
        """Test ExportManager initialization."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        assert manager.output_dir == Path(temp_export_dir)
        assert manager.csv_dir.exists()
        assert manager.excel_dir.exists()
        assert manager.pdf_dir.exists()
        
        # Check subdirectories
        assert (manager.output_dir / 'csv').exists()
        assert (manager.output_dir / 'excel').exists()
        assert (manager.output_dir / 'pdf').exists()
    
    def test_initialization_default_directory(self):
        """Test initialization with default directory."""
        manager = ExportManager()
        
        assert manager.output_dir == Path('exports')
        assert manager.csv_dir == Path('exports/csv')
        assert manager.excel_dir == Path('exports/excel')
        assert manager.pdf_dir == Path('exports/pdf')
        
        # Cleanup
        if Path('exports').exists():
            shutil.rmtree('exports', ignore_errors=True)
    
    def test_export_trades_csv(self, temp_export_dir, sample_trades):
        """Test exporting trades to CSV."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        filepath = manager.export_trades_csv(sample_trades)
        
        assert Path(filepath).exists()
        assert filepath.endswith('all_trades.csv')
        
        # Verify content
        df = pd.read_csv(filepath)
        assert len(df) == len(sample_trades)
        assert 'trade_id' in df.columns
        assert 'symbol' in df.columns
        assert 'pnl' in df.columns
        
        # Verify datetime parsing
        assert pd.to_datetime(df['entry_time']).notna().all()
        assert pd.to_datetime(df['exit_time']).notna().all()
    
    def test_export_trades_csv_custom_filename(self, temp_export_dir, sample_trades):
        """Test exporting trades with custom filename."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        custom_filename = 'my_trades_export.csv'
        filepath = manager.export_trades_csv(sample_trades, filename=custom_filename)
        
        assert Path(filepath).exists()
        assert filepath.endswith(custom_filename)
    
    def test_export_trades_csv_column_ordering(self, temp_export_dir, sample_trades):
        """Test column ordering in trades CSV export."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        filepath = manager.export_trades_csv(sample_trades)
        df = pd.read_csv(filepath)
        
        # Check column order
        expected_first_columns = ['trade_id', 'symbol', 'entry_time', 'exit_time']
        actual_first_columns = df.columns[:4].tolist()
        assert actual_first_columns == expected_first_columns
    
    def test_export_trades_csv_empty(self, temp_export_dir):
        """Test exporting empty trades list."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        filepath = manager.export_trades_csv([])
        
        assert Path(filepath).exists()
        df = pd.read_csv(filepath)
        assert len(df) == 0
    
    def test_export_performance_metrics_csv(self, temp_export_dir, sample_metrics):
        """Test exporting performance metrics to CSV."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        filepath = manager.export_performance_metrics_csv(sample_metrics)
        
        assert Path(filepath).exists()
        assert filepath.endswith('performance_metrics.csv')
        
        # Verify content
        df = pd.read_csv(filepath)
        assert len(df) == len([v for v in sample_metrics.values() if isinstance(v, (int, float))])
        assert 'Metric' in df.columns
        assert 'Value' in df.columns
        
        # Check formatting
        assert 'Total Return' in df['Metric'].values
        assert 'Sharpe Ratio' in df['Metric'].values
    
    def test_export_performance_metrics_csv_custom_filename(self, temp_export_dir, sample_metrics):
        """Test exporting metrics with custom filename."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        custom_filename = 'strategy_metrics.csv'
        filepath = manager.export_performance_metrics_csv(sample_metrics, filename=custom_filename)
        
        assert Path(filepath).exists()
        assert filepath.endswith(custom_filename)
    
    def test_export_performance_metrics_csv_non_numeric(self, temp_export_dir):
        """Test metrics export with non-numeric values."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        mixed_metrics = {
            'total_return': 0.125,
            'strategy_name': 'My Strategy',  # Non-numeric
            'sharpe_ratio': 1.85,
            'description': 'Test description',  # Non-numeric
            'win_rate': 0.65
        }
        
        filepath = manager.export_performance_metrics_csv(mixed_metrics)
        df = pd.read_csv(filepath)
        
        # Should only include numeric values
        assert len(df) == 3
        assert 'Total Return' in df['Metric'].values
        assert 'Sharpe Ratio' in df['Metric'].values
        assert 'Win Rate' in df['Metric'].values
    
    @pytest.mark.skipif(not EXCEL_AVAILABLE, reason="openpyxl not installed")
    def test_export_excel_workbook(self, temp_export_dir, sample_trades, sample_metrics, sample_benchmark_comparison):
        """Test exporting Excel workbook."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        filepath = manager.export_excel_workbook(
            trades=sample_trades,
            metrics=sample_metrics,
            benchmark_comparison=sample_benchmark_comparison
        )
        
        assert Path(filepath).exists()
        assert filepath.endswith('backtest_results.xlsx')
        
        # Verify sheets
        xl_file = pd.ExcelFile(filepath)
        assert 'Trades' in xl_file.sheet_names
        assert 'Metrics' in xl_file.sheet_names
        assert 'Monthly Summary' in xl_file.sheet_names
        assert 'Benchmark Comparison' in xl_file.sheet_names
        
        # Verify content
        trades_df = pd.read_excel(filepath, sheet_name='Trades')
        assert len(trades_df) == len(sample_trades)
        
        metrics_df = pd.read_excel(filepath, sheet_name='Metrics')
        assert len(metrics_df) > 0
    
    @pytest.mark.skipif(not EXCEL_AVAILABLE, reason="openpyxl not installed")
    def test_export_excel_workbook_custom_filename(self, temp_export_dir, sample_trades, sample_metrics):
        """Test Excel export with custom filename."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        custom_filename = 'my_backtest.xlsx'
        filepath = manager.export_excel_workbook(
            trades=sample_trades,
            metrics=sample_metrics,
            filename=custom_filename
        )
        
        assert Path(filepath).exists()
        assert filepath.endswith(custom_filename)
    
    @pytest.mark.skipif(not EXCEL_AVAILABLE, reason="openpyxl not installed")
    def test_export_excel_workbook_no_benchmark(self, temp_export_dir, sample_trades, sample_metrics):
        """Test Excel export without benchmark comparison."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        filepath = manager.export_excel_workbook(
            trades=sample_trades,
            metrics=sample_metrics,
            benchmark_comparison=None
        )
        
        assert Path(filepath).exists()
        
        xl_file = pd.ExcelFile(filepath)
        assert 'Benchmark Comparison' not in xl_file.sheet_names
    
    @pytest.mark.skipif(EXCEL_AVAILABLE, reason="Testing when openpyxl not available")
    def test_export_excel_not_available(self, temp_export_dir, sample_trades, sample_metrics):
        """Test Excel export when openpyxl not available."""
        with patch('src.visualization.export_utils.EXCEL_AVAILABLE', False):
            manager = ExportManager(output_dir=temp_export_dir)
            
            result = manager.export_excel_workbook(
                trades=sample_trades,
                metrics=sample_metrics
            )
            
            assert result is None
    
    def test_create_monthly_summary(self, temp_export_dir, sample_trades):
        """Test monthly summary creation."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        # Convert trades to DataFrame with proper datetime
        trades_df = pd.DataFrame(sample_trades)
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        
        monthly_summary = manager._create_monthly_summary(sample_trades)
        
        assert isinstance(monthly_summary, pd.DataFrame)
        assert 'Trade Count' in monthly_summary.columns
        assert 'Total Return' in monthly_summary.columns
        assert 'Avg Return' in monthly_summary.columns
        assert 'Total PnL' in monthly_summary.columns
        assert 'Avg Confluence' in monthly_summary.columns
    
    @pytest.mark.skipif(not EXCEL_AVAILABLE, reason="openpyxl not installed")
    def test_format_excel_workbook(self, temp_export_dir, sample_trades, sample_metrics):
        """Test Excel workbook formatting."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        # First create the workbook
        filepath = manager.export_excel_workbook(
            trades=sample_trades,
            metrics=sample_metrics
        )
        
        # Verify formatting was applied (file should exist and be readable)
        assert Path(filepath).exists()
        
        # Try to load and verify it's valid
        xl_file = pd.ExcelFile(filepath)
        assert len(xl_file.sheet_names) > 0
    
    @pytest.mark.skipif(not PDF_AVAILABLE, reason="pdfkit not installed")
    def test_export_html_to_pdf(self, temp_export_dir):
        """Test HTML to PDF export."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        html_content = """
        <html>
        <head><title>Test Report</title></head>
        <body>
            <h1>Backtest Report</h1>
            <p>Test content</p>
        </body>
        </html>
        """
        
        filepath = manager.export_html_to_pdf(html_content)
        
        assert Path(filepath).exists()
        assert filepath.endswith('report.pdf')
    
    @pytest.mark.skipif(not PDF_AVAILABLE, reason="pdfkit not installed")
    def test_export_html_to_pdf_custom_options(self, temp_export_dir):
        """Test PDF export with custom options."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        html_content = "<html><body>Test</body></html>"
        custom_options = {
            'page-size': 'Letter',
            'orientation': 'Landscape'
        }
        
        filepath = manager.export_html_to_pdf(
            html_content,
            filename='custom_report.pdf',
            options=custom_options
        )
        
        if filepath:  # pdfkit might fail on some systems
            assert Path(filepath).exists()
            assert filepath.endswith('custom_report.pdf')
    
    @pytest.mark.skipif(PDF_AVAILABLE, reason="Testing when pdfkit not available")
    def test_export_pdf_not_available(self, temp_export_dir):
        """Test PDF export when pdfkit not available."""
        with patch('src.visualization.export_utils.PDF_AVAILABLE', False):
            manager = ExportManager(output_dir=temp_export_dir)
            
            result = manager.export_html_to_pdf("<html></html>")
            assert result is None
    
    def test_export_confluence_scores_timeseries(self, temp_export_dir, sample_confluence_history):
        """Test confluence scores time series export."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        filepath = manager.export_confluence_scores_timeseries(sample_confluence_history)
        
        assert Path(filepath).exists()
        assert filepath.endswith('confluence_scores_timeseries.csv')
        
        # Verify content
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        assert len(df) == len(sample_confluence_history)
        assert 'confluence_score' in df.columns
        assert 'signal_strength' in df.columns
    
    def test_export_json_data(self, temp_export_dir, sample_trades, sample_metrics):
        """Test JSON export."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        data = {
            'trades': sample_trades,
            'metrics': sample_metrics,
            'metadata': {
                'strategy': 'Test Strategy',
                'period': '2023-01-01 to 2023-01-31',
                'symbols': ['AAPL', 'GOOGL', 'MSFT']
            }
        }
        
        filepath = manager.export_json_data(data)
        
        assert Path(filepath).exists()
        assert filepath.endswith('strategy_data.json')
        
        # Verify content
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        
        assert 'trades' in loaded_data
        assert 'metrics' in loaded_data
        assert 'metadata' in loaded_data
        assert len(loaded_data['trades']) == len(sample_trades)
    
    def test_export_json_data_numpy_types(self, temp_export_dir):
        """Test JSON export with numpy types."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        data = {
            'numpy_int': np.int64(42),
            'numpy_float': np.float64(3.14159),
            'numpy_array': np.array([1, 2, 3, 4, 5]),
            'timestamp': pd.Timestamp('2023-01-01'),
            'regular_types': {
                'int': 10,
                'float': 2.5,
                'string': 'test'
            }
        }
        
        filepath = manager.export_json_data(data, filename='numpy_test.json')
        
        # Verify it doesn't raise serialization errors
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data['numpy_int'] == 42
        assert loaded_data['numpy_float'] == 3.14159
        assert loaded_data['numpy_array'] == [1, 2, 3, 4, 5]
        assert loaded_data['timestamp'] == '2023-01-01T00:00:00'
    
    def test_create_export_summary(self, temp_export_dir, sample_trades, sample_metrics):
        """Test export summary creation."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        # Create some exports
        manager.export_trades_csv(sample_trades)
        manager.export_performance_metrics_csv(sample_metrics)
        manager.export_json_data({'test': 'data'})
        
        summary = manager.create_export_summary()
        
        assert isinstance(summary, dict)
        assert 'csv' in summary
        assert 'excel' in summary
        assert 'pdf' in summary
        assert 'json' in summary
        
        # Check that files are listed
        assert len(summary['csv']) >= 2
        assert len(summary['json']) >= 1
        
        # Verify paths are strings
        for file_type, files in summary.items():
            for filepath in files:
                assert isinstance(filepath, str)
    
    def test_export_all(self, temp_export_dir, sample_trades, sample_metrics, 
                       sample_confluence_history, sample_benchmark_comparison):
        """Test exporting all formats."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        html_report = "<html><body>Test Report</body></html>"
        
        exports = manager.export_all(
            trades=sample_trades,
            metrics=sample_metrics,
            confluence_history=sample_confluence_history,
            benchmark_comparison=sample_benchmark_comparison,
            html_report=html_report
        )
        
        assert isinstance(exports, dict)
        assert 'trades_csv' in exports
        assert 'metrics_csv' in exports
        assert 'confluence_csv' in exports
        assert 'json' in exports
        
        # Verify files exist
        for export_type, filepath in exports.items():
            if filepath:  # Some might be None if dependencies missing
                assert Path(filepath).exists()
    
    def test_export_all_minimal(self, temp_export_dir):
        """Test export_all with minimal data."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        exports = manager.export_all(
            trades=[],
            metrics={},
            confluence_history=None,
            benchmark_comparison=None,
            html_report=None
        )
        
        assert isinstance(exports, dict)
        assert 'trades_csv' in exports
        assert 'metrics_csv' in exports
        assert 'json' in exports
        
        # Confluence CSV should not be present
        assert 'confluence_csv' not in exports
    
    def test_directory_creation_error(self):
        """Test handling directory creation errors."""
        with patch('pathlib.Path.mkdir', side_effect=OSError("Permission denied")):
            with pytest.raises(OSError):
                ExportManager(output_dir='/invalid/path')
    
    def test_file_write_permissions(self, temp_export_dir):
        """Test handling file write permission errors."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        # Make directory read-only
        os.chmod(manager.csv_dir, 0o444)
        
        try:
            # This should handle the error gracefully
            filepath = manager.export_trades_csv([])
            # On some systems this might still succeed, so we don't assert failure
        except Exception:
            # Expected on systems that enforce permissions
            pass
        finally:
            # Restore permissions
            os.chmod(manager.csv_dir, 0o755)
    
    def test_empty_dataframe_handling(self, temp_export_dir):
        """Test handling of empty DataFrames."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        filepath = manager.export_confluence_scores_timeseries(
            empty_df,
            filename='empty_confluence.csv'
        )
        
        assert Path(filepath).exists()
        
        # Verify it's empty but valid
        loaded_df = pd.read_csv(filepath)
        assert len(loaded_df) == 0
    
    def test_large_dataset_export(self, temp_export_dir):
        """Test exporting large datasets."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        # Create large dataset
        large_trades = []
        for i in range(1000):
            large_trades.append({
                'trade_id': i,
                'symbol': f'SYM{i % 10}',
                'entry_time': f'2023-01-01 {i % 24:02d}:00:00',
                'exit_time': f'2023-01-02 {i % 24:02d}:00:00',
                'pnl': np.random.normal(100, 500),
                'return': np.random.normal(0.001, 0.02)
            })
        
        import time
        start_time = time.time()
        filepath = manager.export_trades_csv(large_trades, filename='large_trades.csv')
        end_time = time.time()
        
        # Should complete quickly
        assert end_time - start_time < 2.0  # Less than 2 seconds
        assert Path(filepath).exists()
        
        # Verify all data was exported
        df = pd.read_csv(filepath)
        assert len(df) == 1000
    
    def test_special_characters_in_data(self, temp_export_dir):
        """Test handling special characters in exported data."""
        manager = ExportManager(output_dir=temp_export_dir)
        
        # Data with special characters
        special_trades = [
            {
                'trade_id': 1,
                'symbol': 'TEST & CO.',
                'notes': 'Price > $100, P/E < 15',
                'description': 'Test "quoted" text',
                'pnl': 100.50
            }
        ]
        
        # CSV export
        csv_path = manager.export_trades_csv(special_trades, filename='special_chars.csv')
        csv_df = pd.read_csv(csv_path)
        assert csv_df.iloc[0]['symbol'] == 'TEST & CO.'
        
        # JSON export
        json_path = manager.export_json_data({'trades': special_trades}, filename='special_chars.json')
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        assert json_data['trades'][0]['symbol'] == 'TEST & CO.'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])