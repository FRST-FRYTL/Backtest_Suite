"""
Comprehensive tests for the ExportManager class.

This module provides complete test coverage for export functionality
including CSV, Excel, PDF exports and various data formats.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, mock_open, call
import tempfile
import os
import json
from pathlib import Path
import shutil

from src.visualization.export_utils import ExportManager, EXCEL_AVAILABLE, PDF_AVAILABLE


class TestExportManager:
    """Comprehensive tests for ExportManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def export_manager(self, temp_dir):
        """Create ExportManager instance with temp directory."""
        return ExportManager(output_dir=temp_dir)
    
    @pytest.fixture
    def sample_trades_data(self):
        """Create sample trades dataframe."""
        trades = pd.DataFrame({
            'trade_id': range(1, 21),
            'entry_time': pd.date_range('2023-01-01', periods=20, freq='D'),
            'exit_time': pd.date_range('2023-01-02', periods=20, freq='D'),
            'symbol': ['AAPL', 'GOOGL'] * 10,
            'entry_price': np.random.uniform(100, 200, 20),
            'exit_price': np.random.uniform(100, 200, 20),
            'quantity': np.random.randint(10, 100, 20),
            'profit_loss': np.random.uniform(-500, 1000, 20),
            'return_pct': np.random.uniform(-5, 10, 20)
        })
        return trades
    
    @pytest.fixture
    def sample_performance_data(self):
        """Create sample performance metrics."""
        return {
            'total_return': 0.2543,
            'annualized_return': 0.2856,
            'sharpe_ratio': 1.45,
            'sortino_ratio': 2.13,
            'max_drawdown': -0.1523,
            'win_rate': 0.58,
            'profit_factor': 1.85,
            'total_trades': 50,
            'winning_trades': 29,
            'losing_trades': 21,
            'avg_win': 523.45,
            'avg_loss': -234.67,
            'best_trade': 1523.45,
            'worst_trade': -876.23
        }
    
    @pytest.fixture
    def sample_equity_curve(self):
        """Create sample equity curve data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = 100000 * (1 + np.random.normal(0.001, 0.02, 100)).cumprod()
        
        return pd.DataFrame({
            'date': dates,
            'total_value': values,
            'cash': 50000 * np.ones(100),
            'holdings_value': values - 50000
        })
    
    def test_export_manager_initialization(self, export_manager, temp_dir):
        """Test ExportManager initialization."""
        assert export_manager.output_dir == Path(temp_dir)
        assert export_manager.output_dir.exists()
        
        # Check subdirectories are created
        expected_subdirs = ['csv', 'excel', 'pdf', 'json']
        for subdir in expected_subdirs:
            subdir_path = export_manager.output_dir / subdir
            assert subdir_path.exists()
    
    def test_export_to_csv(self, export_manager, sample_trades_data):
        """Test CSV export functionality."""
        filename = export_manager.export_to_csv(
            sample_trades_data,
            'test_trades',
            include_index=False
        )
        
        assert filename is not None
        assert os.path.exists(filename)
        assert filename.endswith('.csv')
        
        # Verify content
        loaded_df = pd.read_csv(filename)
        assert len(loaded_df) == len(sample_trades_data)
        assert list(loaded_df.columns) == list(sample_trades_data.columns)
    
    def test_export_to_csv_with_metadata(self, export_manager, sample_trades_data):
        """Test CSV export with metadata."""
        metadata = {
            'strategy': 'Test Strategy',
            'period': '2023-01-01 to 2023-12-31',
            'initial_capital': 100000
        }
        
        filename = export_manager.export_to_csv(
            sample_trades_data,
            'test_trades_metadata',
            metadata=metadata
        )
        
        assert os.path.exists(filename)
        
        # Check metadata file
        metadata_file = filename.replace('.csv', '_metadata.json')
        assert os.path.exists(metadata_file)
        
        with open(metadata_file, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata['strategy'] == metadata['strategy']
    
    @pytest.mark.skipif(not EXCEL_AVAILABLE, reason="openpyxl not available")
    def test_export_to_excel(self, export_manager, sample_trades_data, 
                            sample_performance_data, sample_equity_curve):
        """Test Excel export functionality."""
        data_dict = {
            'Trades': sample_trades_data,
            'Performance': pd.DataFrame([sample_performance_data]),
            'Equity Curve': sample_equity_curve
        }
        
        filename = export_manager.export_to_excel(
            data_dict,
            'test_report'
        )
        
        assert filename is not None
        assert os.path.exists(filename)
        assert filename.endswith('.xlsx')
        
        # Verify content
        with pd.ExcelFile(filename) as xls:
            assert set(xls.sheet_names) == set(data_dict.keys())
            
            # Check each sheet
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name)
                assert len(df) > 0
    
    @pytest.mark.skipif(not EXCEL_AVAILABLE, reason="openpyxl not available")
    def test_export_to_excel_with_formatting(self, export_manager, sample_trades_data):
        """Test Excel export with formatting options."""
        filename = export_manager.export_to_excel(
            {'Trades': sample_trades_data},
            'formatted_report',
            apply_formatting=True
        )
        
        assert os.path.exists(filename)
        
        # Load and check formatting was applied
        import openpyxl
        wb = openpyxl.load_workbook(filename)
        ws = wb['Trades']
        
        # Check header formatting
        header_cell = ws['A1']
        assert header_cell.font.bold is True
    
    def test_export_to_json(self, export_manager, sample_performance_data):
        """Test JSON export functionality."""
        filename = export_manager.export_to_json(
            sample_performance_data,
            'test_metrics'
        )
        
        assert filename is not None
        assert os.path.exists(filename)
        assert filename.endswith('.json')
        
        # Verify content
        with open(filename, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data['total_return'] == sample_performance_data['total_return']
        assert loaded_data['sharpe_ratio'] == sample_performance_data['sharpe_ratio']
    
    def test_export_to_json_with_dataframe(self, export_manager, sample_trades_data):
        """Test JSON export with DataFrame."""
        filename = export_manager.export_to_json(
            sample_trades_data,
            'test_trades_json'
        )
        
        assert os.path.exists(filename)
        
        # Verify content
        with open(filename, 'r') as f:
            loaded_data = json.load(f)
        
        # Should be converted to records format
        assert isinstance(loaded_data, list)
        assert len(loaded_data) == len(sample_trades_data)
    
    @pytest.mark.skipif(not PDF_AVAILABLE, reason="pdfkit not available")
    @patch('pdfkit.from_string')
    def test_export_to_pdf(self, mock_pdfkit, export_manager):
        """Test PDF export functionality."""
        html_content = "<html><body><h1>Test Report</h1></body></html>"
        
        filename = export_manager.export_to_pdf(
            html_content,
            'test_report'
        )
        
        mock_pdfkit.assert_called_once()
        assert filename.endswith('.pdf')
    
    def test_export_summary_report(self, export_manager, sample_performance_data,
                                  sample_trades_data, sample_equity_curve):
        """Test comprehensive summary report export."""
        report_data = {
            'performance': sample_performance_data,
            'trades': sample_trades_data,
            'equity_curve': sample_equity_curve,
            'metadata': {
                'strategy': 'Test Strategy',
                'start_date': '2023-01-01',
                'end_date': '2023-12-31'
            }
        }
        
        filenames = export_manager.export_summary_report(
            report_data,
            'comprehensive_report',
            formats=['csv', 'json']
        )
        
        assert isinstance(filenames, dict)
        assert 'csv' in filenames
        assert 'json' in filenames
        
        # Check CSV files
        for csv_file in filenames.get('csv', []):
            assert os.path.exists(csv_file)
        
        # Check JSON file
        assert os.path.exists(filenames['json'])
    
    def test_batch_export(self, export_manager, sample_trades_data):
        """Test batch export functionality."""
        # Create multiple dataframes
        dataframes = {
            f'trades_{i}': sample_trades_data.copy() 
            for i in range(3)
        }
        
        filenames = export_manager.batch_export(
            dataframes,
            format='csv'
        )
        
        assert len(filenames) == 3
        for filename in filenames:
            assert os.path.exists(filename)
            assert filename.endswith('.csv')
    
    def test_export_with_compression(self, export_manager, sample_trades_data):
        """Test export with compression."""
        filename = export_manager.export_to_csv(
            sample_trades_data,
            'compressed_trades',
            compression='gzip'
        )
        
        assert filename.endswith('.csv.gz')
        assert os.path.exists(filename)
        
        # Verify can be read back
        loaded_df = pd.read_csv(filename, compression='gzip')
        assert len(loaded_df) == len(sample_trades_data)
    
    def test_export_large_dataset(self, export_manager):
        """Test export of large dataset."""
        # Create large dataset
        large_df = pd.DataFrame({
            'id': range(10000),
            'value': np.random.randn(10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        filename = export_manager.export_to_csv(
            large_df,
            'large_dataset',
            chunksize=1000
        )
        
        assert os.path.exists(filename)
        
        # Verify all data was exported
        loaded_df = pd.read_csv(filename)
        assert len(loaded_df) == 10000
    
    def test_export_with_custom_path(self, export_manager, sample_trades_data, temp_dir):
        """Test export with custom output path."""
        custom_dir = os.path.join(temp_dir, 'custom')
        os.makedirs(custom_dir, exist_ok=True)
        
        filename = export_manager.export_to_csv(
            sample_trades_data,
            'custom_location',
            output_dir=custom_dir
        )
        
        assert custom_dir in filename
        assert os.path.exists(filename)
    
    def test_export_error_handling(self, export_manager):
        """Test error handling in export operations."""
        # Invalid data type
        with pytest.raises((TypeError, AttributeError)):
            export_manager.export_to_csv("not a dataframe", 'test')
        
        # Invalid path
        with pytest.raises((OSError, IOError)):
            export_manager.export_to_csv(
                pd.DataFrame({'a': [1, 2]}),
                'test',
                output_dir='/invalid/path/that/does/not/exist'
            )
    
    def test_filename_sanitization(self, export_manager, sample_trades_data):
        """Test filename sanitization."""
        # Filename with invalid characters
        filename = export_manager.export_to_csv(
            sample_trades_data,
            'test<>file:name*with?invalid|chars'
        )
        
        assert os.path.exists(filename)
        # Check invalid characters were removed or replaced
        assert not any(char in filename for char in '<>:*?|')
    
    def test_export_with_datetime_handling(self, export_manager):
        """Test export with proper datetime handling."""
        df_with_dates = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'value': range(10)
        })
        
        filename = export_manager.export_to_csv(
            df_with_dates,
            'dates_test',
            date_format='%Y-%m-%d'
        )
        
        assert os.path.exists(filename)
        
        # Verify dates are formatted correctly
        loaded_df = pd.read_csv(filename)
        assert loaded_df['date'][0] == '2023-01-01'
    
    def test_export_metadata_tracking(self, export_manager, sample_trades_data):
        """Test metadata tracking for exports."""
        export_manager.enable_tracking()
        
        # Perform multiple exports
        export_manager.export_to_csv(sample_trades_data, 'test1')
        export_manager.export_to_csv(sample_trades_data, 'test2')
        
        # Get export history
        history = export_manager.get_export_history()
        
        assert len(history) == 2
        assert all('timestamp' in record for record in history)
        assert all('filename' in record for record in history)
    
    @pytest.mark.skipif(not EXCEL_AVAILABLE, reason="openpyxl not available")
    def test_excel_multi_index_export(self, export_manager):
        """Test Excel export with multi-index DataFrame."""
        # Create multi-index DataFrame
        index = pd.MultiIndex.from_product(
            [['A', 'B'], ['X', 'Y', 'Z']],
            names=['Group', 'Subgroup']
        )
        
        df = pd.DataFrame({
            'Value1': np.random.randn(6),
            'Value2': np.random.randn(6)
        }, index=index)
        
        filename = export_manager.export_to_excel(
            {'MultiIndex': df},
            'multiindex_test'
        )
        
        assert os.path.exists(filename)