"""
SPX Multi-Timeframe Data Fetcher

This module fetches S&P 500 index data (using SPY ETF as proxy) across multiple timeframes
for quantitative trading analysis. It handles data downloading, organization, and quality
reporting with comprehensive error handling and progress tracking.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
import pytz
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.fetcher import StockDataFetcher
from src.data.cache import DataCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rich console for beautiful output
console = Console()


class SPXMultiTimeframeFetcher:
    """
    Fetches SPX data (using SPY ETF) across multiple timeframes with coordination hooks.
    """
    
    # Timeframe configurations (adjusted for Yahoo Finance limitations)
    TIMEFRAMES = {
        '1min': {
            'interval': '1m',
            'days_back': 7,  # Yahoo limits 1m data to 7 days
            'description': 'High-frequency trading data'
        },
        '5min': {
            'interval': '5m',
            'days_back': 59,  # Yahoo limits 5m data to 60 days
            'description': 'Intraday momentum tracking'
        },
        '15min': {
            'interval': '15m',
            'days_back': 59,  # Yahoo limits 15m data to 60 days
            'description': 'Short-term trend analysis'
        },
        '30min': {
            'interval': '30m',
            'days_back': 59,  # Yahoo limits 30m data to 60 days
            'description': 'Intraday swing trading'
        },
        '1H': {
            'interval': '1h',
            'days_back': 180,  # 6 months for hourly data
            'description': 'Daily trend confirmation'
        },
        '4H': {
            'interval': '1h',  # Will resample to 4H
            'days_back': 365,  # 1 year for 4-hour data
            'description': 'Multi-day swing trading'
        },
        '1D': {
            'interval': '1d',
            'days_back': 730,  # 2 years for daily data
            'description': 'Position trading and trend following'
        }
    }
    
    def __init__(self, data_dir: str = "data/SPX"):
        """
        Initialize the SPX multi-timeframe fetcher.
        
        Args:
            data_dir: Base directory for storing SPX data
        """
        self.data_dir = Path(data_dir)
        self.symbol = "SPY"  # Using SPY ETF as SPX proxy
        self.fetcher = StockDataFetcher(cache_dir=str(self.data_dir / "cache"))
        self.summary_data = {
            'symbol': self.symbol,
            'fetch_date': datetime.now().isoformat(),
            'timeframes': {}
        }
        
    async def fetch_all_timeframes(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch SPX data for all configured timeframes.
        
        Returns:
            Dictionary mapping timeframe names to DataFrames
        """
        console.print(f"[bold blue]Starting SPX Multi-Timeframe Data Fetch[/bold blue]")
        console.print(f"Symbol: {self.symbol} (SPY ETF as S&P 500 proxy)")
        console.print(f"Target directory: {self.data_dir}")
        
        data_by_timeframe = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                f"Fetching {len(self.TIMEFRAMES)} timeframes...",
                total=len(self.TIMEFRAMES)
            )
            
            for timeframe, config in self.TIMEFRAMES.items():
                progress.update(task, description=f"Fetching {timeframe} data...")
                
                try:
                    # Calculate date range
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=config['days_back'])
                    
                    # Fetch data
                    data = await self._fetch_single_timeframe(
                        timeframe, config, start_date, end_date
                    )
                    
                    if data is not None and not data.empty:
                        data_by_timeframe[timeframe] = data
                        
                        # Run coordination hook for successful download
                        await self._run_post_edit_hook(timeframe, data)
                    
                except Exception as e:
                    logger.error(f"Failed to fetch {timeframe} data: {e}")
                    console.print(f"[red]✗ Failed to fetch {timeframe} data: {e}[/red]")
                
                progress.advance(task)
        
        return data_by_timeframe
    
    async def _fetch_single_timeframe(
        self,
        timeframe: str,
        config: dict,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single timeframe.
        
        Args:
            timeframe: Timeframe identifier
            config: Timeframe configuration
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with fetched data or None if failed
        """
        try:
            # Use the fetcher to get data
            async with self.fetcher as fetcher:
                data = await fetcher.fetch(
                    symbol=self.symbol,
                    start=start_date,
                    end=end_date,
                    interval=config['interval'],
                    prepost=True,  # Include pre/post market data
                    actions=True,   # Include dividends and splits
                    repair=True     # Repair any bad data
                )
            
            if data.empty:
                logger.warning(f"No data returned for {timeframe}")
                return None
            
            # Special handling for 4H timeframe (resample from 1H)
            if timeframe == '4H':
                data = self._resample_to_4h(data)
            
            # Ensure timezone-aware timestamps
            if data.index.tz is None:
                data.index = data.index.tz_localize('America/New_York')
            
            # Save to file
            output_path = self._save_timeframe_data(timeframe, data)
            
            # Update summary
            self.summary_data['timeframes'][timeframe] = {
                'rows': len(data),
                'start_date': data.index.min().isoformat(),
                'end_date': data.index.max().isoformat(),
                'file_path': str(output_path),
                'file_size_mb': output_path.stat().st_size / (1024 * 1024),
                'columns': list(data.columns),
                'description': config['description']
            }
            
            console.print(f"[green]✓ {timeframe}: {len(data)} rows fetched[/green]")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {timeframe} data: {e}")
            raise
    
    def _resample_to_4h(self, hourly_data: pd.DataFrame) -> pd.DataFrame:
        """
        Resample hourly data to 4-hour bars.
        
        Args:
            hourly_data: DataFrame with hourly data
            
        Returns:
            DataFrame with 4-hour bars
        """
        # Resample to 4H bars
        resampled = hourly_data.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        # Remove any rows with all NaN values
        resampled = resampled.dropna(how='all')
        
        return resampled
    
    def _save_timeframe_data(self, timeframe: str, data: pd.DataFrame) -> Path:
        """
        Save timeframe data to organized directory structure.
        
        Args:
            timeframe: Timeframe identifier
            data: DataFrame to save
            
        Returns:
            Path to saved file
        """
        # Create timeframe directory
        timeframe_dir = self.data_dir / timeframe
        timeframe_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with date range
        start_date = data.index.min().strftime('%Y%m%d')
        end_date = data.index.max().strftime('%Y%m%d')
        filename = f"{self.symbol}_{timeframe}_{start_date}_{end_date}.csv"
        
        # Save to CSV
        output_path = timeframe_dir / filename
        data.to_csv(output_path)
        
        # Also save as latest for easy access
        latest_path = timeframe_dir / f"{self.symbol}_{timeframe}_latest.csv"
        data.to_csv(latest_path)
        
        return output_path
    
    async def _run_post_edit_hook(self, timeframe: str, data: pd.DataFrame):
        """
        Run Claude Flow coordination hook after data download.
        
        Args:
            timeframe: Timeframe that was downloaded
            data: Downloaded data
        """
        try:
            memory_key = f"agent/fetcher/{timeframe}"
            file_path = f"data/SPX/{timeframe}/{self.symbol}_{timeframe}_latest.csv"
            
            cmd = f'npx claude-flow@alpha hooks post-edit --file "{file_path}" --memory-key "{memory_key}"'
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                logger.info(f"Successfully ran post-edit hook for {timeframe}")
            else:
                logger.warning(f"Post-edit hook failed for {timeframe}: {stderr.decode()}")
                
        except Exception as e:
            logger.warning(f"Failed to run post-edit hook: {e}")
    
    async def _store_notification(self, message: str):
        """
        Store notification in Claude Flow memory.
        
        Args:
            message: Notification message
        """
        try:
            cmd = f'npx claude-flow@alpha hooks notification --message "{message}"'
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await proc.communicate()
            
        except Exception as e:
            logger.warning(f"Failed to store notification: {e}")
    
    def validate_data_quality(self, data_by_timeframe: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
        """
        Validate the quality of fetched data.
        
        Args:
            data_by_timeframe: Dictionary of timeframe data
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        for timeframe, data in data_by_timeframe.items():
            results = {
                'rows': len(data),
                'missing_values': data.isnull().sum().to_dict(),
                'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
                'price_consistency': {
                    'high_low_valid': (data['High'] >= data['Low']).all(),
                    'ohlc_valid': (
                        (data['High'] >= data['Open']) & 
                        (data['High'] >= data['Close']) & 
                        (data['Low'] <= data['Open']) & 
                        (data['Low'] <= data['Close'])
                    ).all(),
                    'positive_volume': (data['Volume'] >= 0).all()
                },
                'date_gaps': self._check_date_gaps(data, timeframe),
                'data_freshness': self._calculate_data_freshness(data)
            }
            
            validation_results[timeframe] = results
        
        return validation_results
    
    def _calculate_data_freshness(self, data: pd.DataFrame) -> int:
        """
        Calculate how many days old the latest data is.
        
        Args:
            data: DataFrame with datetime index
            
        Returns:
            Number of days since latest data
        """
        latest_timestamp = data.index.max()
        
        # Handle timezone-aware comparison
        if hasattr(latest_timestamp, 'tz') and latest_timestamp.tz is not None:
            # Convert to UTC for comparison
            latest_utc = latest_timestamp.tz_convert('UTC')
            current_utc = pd.Timestamp.now('UTC')
            return (current_utc - latest_utc).days
        else:
            # Naive datetime comparison
            return (datetime.now() - latest_timestamp).days
    
    def _check_date_gaps(self, data: pd.DataFrame, timeframe: str) -> dict:
        """
        Check for gaps in the date sequence.
        
        Args:
            data: DataFrame to check
            timeframe: Timeframe identifier
            
        Returns:
            Dictionary with gap information
        """
        # Expected frequency for each timeframe
        freq_map = {
            '1min': 'T',    # Minutes
            '5min': '5T',   # 5 Minutes
            '15min': '15T', # 15 Minutes
            '30min': '30T', # 30 Minutes
            '1H': 'H',      # Hourly
            '4H': '4H',     # 4 Hours
            '1D': 'B'       # Business days
        }
        
        if timeframe not in freq_map:
            return {'checked': False, 'reason': 'Unknown timeframe'}
        
        try:
            # Create expected date range
            expected_range = pd.date_range(
                start=data.index.min(),
                end=data.index.max(),
                freq=freq_map[timeframe]
            )
            
            # For minute data, filter to market hours only
            if timeframe in ['1min', '5min', '15min', '30min']:
                expected_range = expected_range[
                    (expected_range.hour >= 9) & 
                    (expected_range.hour < 16) |
                    ((expected_range.hour == 16) & (expected_range.minute == 0))
                ]
            
            # Find missing timestamps
            missing = expected_range.difference(data.index)
            
            return {
                'checked': True,
                'missing_count': len(missing),
                'missing_percentage': len(missing) / len(expected_range) * 100,
                'largest_gap': self._find_largest_gap(data.index) if len(data) > 1 else None
            }
            
        except Exception as e:
            return {'checked': False, 'reason': str(e)}
    
    def _find_largest_gap(self, index: pd.DatetimeIndex) -> dict:
        """
        Find the largest gap in a datetime index.
        
        Args:
            index: DatetimeIndex to analyze
            
        Returns:
            Dictionary with largest gap information
        """
        if len(index) < 2:
            return None
        
        # Calculate differences
        diffs = index[1:] - index[:-1]
        max_gap_idx = diffs.argmax()
        
        return {
            'duration': str(diffs[max_gap_idx]),
            'start': index[max_gap_idx].isoformat(),
            'end': index[max_gap_idx + 1].isoformat()
        }
    
    def generate_summary_report(
        self,
        data_by_timeframe: Dict[str, pd.DataFrame],
        validation_results: Dict[str, dict]
    ) -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            data_by_timeframe: Dictionary of timeframe data
            validation_results: Validation results
            
        Returns:
            Path to summary report
        """
        # Create summary table
        table = Table(title="SPX Multi-Timeframe Data Summary", show_header=True)
        table.add_column("Timeframe", style="cyan")
        table.add_column("Rows", justify="right", style="green")
        table.add_column("Date Range", style="yellow")
        table.add_column("Missing %", justify="right", style="red")
        table.add_column("Quality", justify="center")
        
        for timeframe in self.TIMEFRAMES:
            if timeframe in data_by_timeframe:
                data = data_by_timeframe[timeframe]
                validation = validation_results.get(timeframe, {})
                
                # Calculate overall missing percentage
                missing_pct = validation.get('missing_percentage', {})
                avg_missing = sum(missing_pct.values()) / len(missing_pct) if missing_pct else 0
                
                # Determine quality emoji
                if avg_missing < 1:
                    quality = "✅ Excellent"
                elif avg_missing < 5:
                    quality = "⚠️ Good"
                else:
                    quality = "❌ Poor"
                
                table.add_row(
                    timeframe,
                    f"{len(data):,}",
                    f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}",
                    f"{avg_missing:.2f}%",
                    quality
                )
        
        console.print(table)
        
        # Save detailed report
        report_path = self.data_dir / "summary_report.json"
        report_data = {
            'summary': self.summary_data,
            'validation': validation_results,
            'fetch_timestamp': datetime.now().isoformat()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Save markdown report
        md_report_path = self.data_dir / "summary_report.md"
        self._generate_markdown_report(data_by_timeframe, validation_results, md_report_path)
        
        return str(report_path)
    
    def _generate_markdown_report(
        self,
        data_by_timeframe: Dict[str, pd.DataFrame],
        validation_results: Dict[str, dict],
        output_path: Path
    ):
        """
        Generate a markdown summary report.
        
        Args:
            data_by_timeframe: Dictionary of timeframe data
            validation_results: Validation results
            output_path: Path to save markdown report
        """
        with open(output_path, 'w') as f:
            f.write("# SPX Multi-Timeframe Data Summary Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Symbol:** {self.symbol} (SPY ETF as S&P 500 proxy)\n\n")
            
            f.write("## Data Overview\n\n")
            
            for timeframe, config in self.TIMEFRAMES.items():
                if timeframe in data_by_timeframe:
                    data = data_by_timeframe[timeframe]
                    validation = validation_results.get(timeframe, {})
                    
                    f.write(f"### {timeframe} - {config['description']}\n\n")
                    f.write(f"- **Rows:** {len(data):,}\n")
                    f.write(f"- **Date Range:** {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}\n")
                    f.write(f"- **File:** `{timeframe}/{self.symbol}_{timeframe}_latest.csv`\n")
                    
                    # Data quality
                    if 'price_consistency' in validation:
                        pc = validation['price_consistency']
                        f.write(f"- **Data Quality:**\n")
                        f.write(f"  - High/Low Consistency: {'✅' if pc['high_low_valid'] else '❌'}\n")
                        f.write(f"  - OHLC Consistency: {'✅' if pc['ohlc_valid'] else '❌'}\n")
                        f.write(f"  - Volume Valid: {'✅' if pc['positive_volume'] else '❌'}\n")
                    
                    # Date gaps
                    if 'date_gaps' in validation and validation['date_gaps'].get('checked'):
                        gaps = validation['date_gaps']
                        f.write(f"- **Date Gaps:** {gaps['missing_count']} missing ({gaps['missing_percentage']:.2f}%)\n")
                    
                    f.write("\n")
            
            f.write("## Usage Instructions\n\n")
            f.write("```python\n")
            f.write("import pandas as pd\n\n")
            f.write("# Load a specific timeframe\n")
            f.write("df_daily = pd.read_csv('data/SPX/1D/SPY_1D_latest.csv', index_col=0, parse_dates=True)\n\n")
            f.write("# Load multiple timeframes\n")
            f.write("timeframes = ['1min', '5min', '15min', '30min', '1H', '4H', '1D']\n")
            f.write("data = {}\n")
            f.write("for tf in timeframes:\n")
            f.write("    data[tf] = pd.read_csv(f'data/SPX/{tf}/SPY_{tf}_latest.csv', index_col=0, parse_dates=True)\n")
            f.write("```\n")


async def main():
    """
    Main entry point for the SPX multi-timeframe data fetcher.
    """
    fetcher = SPXMultiTimeframeFetcher()
    
    try:
        # Fetch all timeframes
        data_by_timeframe = await fetcher.fetch_all_timeframes()
        
        # Validate data quality
        console.print("\n[bold blue]Validating data quality...[/bold blue]")
        validation_results = fetcher.validate_data_quality(data_by_timeframe)
        
        # Generate summary report
        console.print("\n[bold blue]Generating summary report...[/bold blue]")
        report_path = fetcher.generate_summary_report(data_by_timeframe, validation_results)
        
        console.print(f"\n[bold green]✓ Data fetch complete![/bold green]")
        console.print(f"Summary report: {report_path}")
        
        # Store final notification
        await fetcher._store_notification(
            f"SPX data fetch complete: {len(data_by_timeframe)} timeframes, "
            f"report at {report_path}"
        )
        
        # Run post-task hook
        cmd = 'npx claude-flow@alpha hooks post-task --task-id "spx-data-fetch" --analyze-performance true'
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()
        
    except Exception as e:
        console.print(f"[bold red]Error during data fetch: {e}[/bold red]")
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())