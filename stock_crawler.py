"""
TWSE Stock Data Crawler
=======================
Download historical daily stock prices from Taiwan Stock Exchange (TWSE).

Usage:
    python twstock_getid.py --start 2025-01-01 --end 2025-12-31
    python twstock_getid.py --symbols-file stocks.csv --limit 10
    python twstock_getid.py --symbols 2330,2317 --output-dir ./output
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

# ==================== Configuration ====================
BASE_URL = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
REQUEST_TIMEOUT = 10
DEFAULT_PAUSE = 0.2  # Seconds between requests to avoid rate limiting

# Column mappings from TWSE API response
RAW_COLUMNS = [
    "date", "volume", "turnover", "open", "high", 
    "low", "close", "change", "order_count"
]
OUTPUT_COLUMNS = [
    "stock_id", "date", "open", "high", 
    "low", "close", "volume", "turnover"
]


# ==================== Helper Functions ====================

def print_banner():
    """Display a professional welcome banner."""
    banner = """
╔════════════════════════════════════════════════════════╗
║        TWSE Stock Data Crawler v1.0                    ║
║        Taiwan Stock Exchange Historical Data           ║
╚════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_progress_bar(current: int, total: int, prefix: str = '', length: int = 50):
    """
    Display a progress bar in the terminal.
    
    Args:
        current: Current progress count
        total: Total count
        prefix: Prefix text to display
        length: Length of the progress bar
    """
    if total == 0:
        return
    
    percent = 100 * (current / float(total))
    filled = int(length * current // total)
    bar = '█' * filled + '░' * (length - filled)
    
    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% [{current}/{total}]')
    sys.stdout.flush()
    
    if current == total:
        print()


def load_symbols(csv_path: Path) -> List[str]:
    """
    Load stock symbols from a CSV file.
    
    Args:
        csv_path: Path to CSV file containing stock symbols
        
    Returns:
        List of stock symbol strings
    """
    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    first_column = df.columns[0]
    symbols = df[first_column].astype(str).str.strip()
    symbols = symbols[symbols.ne("")]
    return symbols.tolist()


def parse_symbols_string(symbols_str: str) -> List[str]:
    """
    Parse comma-separated stock symbols string.
    
    Args:
        symbols_str: Comma-separated symbols (e.g., "2330,2317,2454")
        
    Returns:
        List of stock symbols
    """
    return [s.strip() for s in symbols_str.split(',') if s.strip()]


def generate_month_end_dates(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List[pd.Timestamp]:
    """
    Generate a list of month-end dates between start and end dates.
    
    Args:
        start_ts: Start timestamp
        end_ts: End timestamp
        
    Returns:
        List of month-end timestamps
    """
    if start_ts > end_ts:
        return []
    end_of_month = end_ts.to_period("M").to_timestamp("M")
    return list(pd.date_range(start=start_ts, end=end_of_month, freq="M"))


def roc_to_gregorian(value: str) -> Optional[pd.Timestamp]:
    """
    Convert ROC (Taiwan) date format to Gregorian calendar.
    ROC year = Gregorian year - 1911
    
    Args:
        value: Date string in ROC format (e.g., "114/01/15")
        
    Returns:
        Timestamp in Gregorian calendar or None if invalid
    """
    if not isinstance(value, str) or "/" not in value:
        return None
    try:
        year, month, day = value.split("/")
        year = int(year) + 1911
        return pd.Timestamp(year=int(year), month=int(month), day=int(day))
    except (TypeError, ValueError):
        return None


def clean_numeric(series: pd.Series, as_integer: bool = False) -> pd.Series:
    """
    Clean and convert numeric series, removing commas and handling missing values.
    
    Args:
        series: Pandas series to clean
        as_integer: If True, convert to integer type
        
    Returns:
        Cleaned numeric series
    """
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("--", "", regex=False)
    )
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    numeric = pd.to_numeric(cleaned, errors="coerce")
    
    if as_integer:
        return numeric.round().astype("Int64")
    return numeric.astype(float)


# ==================== Data Processing ====================

def transform_month_rows(rows: List[List[str]], symbol: str) -> pd.DataFrame:
    """
    Transform raw API response rows into a structured DataFrame.
    
    Args:
        rows: Raw data rows from TWSE API
        symbol: Stock symbol ID
        
    Returns:
        Cleaned and structured DataFrame
    """
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    
    # Convert ROC dates to Gregorian
    df["date"] = df["date"].apply(roc_to_gregorian)
    df = df.dropna(subset=["date"])
    
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    
    # Clean numeric columns
    for column in ["open", "high", "low", "close"]:
        df[column] = clean_numeric(df[column])
    df["volume"] = clean_numeric(df["volume"], as_integer=True)
    df["turnover"] = clean_numeric(df["turnover"])
    
    # Add stock_id and select output columns
    df["stock_id"] = symbol
    df = df[OUTPUT_COLUMNS]
    
    return df


def fetch_symbol_month(
    session: requests.Session,
    symbol: str,
    query_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Fetch monthly stock data for a single symbol from TWSE API.
    
    Args:
        session: Requests session for connection pooling
        symbol: Stock symbol ID
        query_date: Date to query (will fetch entire month)
        
    Returns:
        DataFrame with stock data for the month
    """
    params = {
        "response": "json",
        "date": query_date.strftime("%Y%m%d"),
        "stockNo": symbol,
    }
    
    try:
        response = session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"\n⚠️  Request failed for {symbol} at {params['date']}: {exc}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    
    try:
        payload = response.json()
    except ValueError as exc:
        print(f"\n⚠️  Invalid JSON for {symbol} at {params['date']}: {exc}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    
    if payload.get("stat") != "OK":
        message = payload.get("stat", "unknown error")
        # Suppress common "no data" messages to reduce noise
        if message not in ["查詢日期小於91年9月2日，請重新查詢!", "很抱歉，沒有符合條件的資料!"]:
            print(f"\n⚠️  TWSE returned '{message}' for {symbol} at {params['date']}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    
    rows = payload.get("data", [])
    return transform_month_rows(rows, symbol)


# ==================== Main Download Logic ====================

def download_price_history(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    sleep_between_requests: float = DEFAULT_PAUSE,
) -> pd.DataFrame:
    """
    Download historical price data for multiple symbols.
    
    Args:
        symbols: Iterable of stock symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        sleep_between_requests: Pause duration between API requests
        
    Returns:
        Combined DataFrame with all stock data
    """
    symbols = list(symbols)
    if not symbols:
        print("⚠️  No symbols provided for download.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    
    # Validate and adjust date range
    start_ts = pd.Timestamp(start_date)
    requested_end_ts = pd.Timestamp(end_date)
    today = pd.Timestamp.today().normalize()
    end_ts = min(requested_end_ts, today)
    
    if requested_end_ts > end_ts:
        print(f"ℹ️  Requested end date {requested_end_ts.date()} is in the future; "
              f"using {end_ts.date()} instead.")
    
    if start_ts > end_ts:
        print(f"⚠️  Start date {start_ts.date()} is after end date {end_ts.date()}.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    
    # Generate query dates (one per month)
    month_end_dates = generate_month_end_dates(start_ts, end_ts)
    if not month_end_dates:
        print("⚠️  No valid month-end dates to query.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    
    total_requests = len(symbols) * len(month_end_dates)
    requests_done = 0
    frames: List[pd.DataFrame] = []
    
    print(f"\n📊 Downloading data for {len(symbols)} symbols across {len(month_end_dates)} months...")
    print(f"📅 Date range: {start_ts.date()} to {end_ts.date()}\n")
    
    with requests.Session() as session:
        for idx, symbol in enumerate(symbols, start=1):
            symbol_frames = []
            
            for month_end in month_end_dates:
                query_date = min(month_end, end_ts)
                month_frame = fetch_symbol_month(session, symbol, query_date)
                requests_done += 1
                
                # Update progress bar
                print_progress_bar(
                    requests_done, 
                    total_requests, 
                    prefix=f'Symbol {idx}/{len(symbols)} ({symbol})'
                )
                
                if not month_frame.empty:
                    # Filter to exact date range
                    month_frame = month_frame.loc[
                        (month_frame["date"] >= start_ts) & 
                        (month_frame["date"] <= end_ts)
                    ]
                    if not month_frame.empty:
                        symbol_frames.append(month_frame)
                
                # Rate limiting
                if sleep_between_requests > 0:
                    time.sleep(sleep_between_requests)
            
            if symbol_frames:
                frames.extend(symbol_frames)
    
    print()  # New line after progress bar
    
    if not frames:
        print("⚠️  No data retrieved.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    
    # Combine and clean data
    print("🔄 Processing downloaded data...")
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["stock_id", "date"])
    data = data.sort_values(["stock_id", "date"]).reset_index(drop=True)
    
    print(f"✅ Downloaded {len(data)} records for {data['stock_id'].nunique()} symbols\n")
    
    return data


# ==================== Output Functions ====================

def save_datasets(df: pd.DataFrame, output_dir: Path, start_date: str, end_date: str) -> None:
    """
    Save downloaded data to CSV and Parquet formats.
    
    Args:
        df: DataFrame to save
        output_dir: Output directory path
        start_date: Start date for filename
        end_date: End date for filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_key = pd.Timestamp(start_date).strftime("%Y%m%d")
    end_key = pd.Timestamp(end_date).strftime("%Y%m%d")
    
    csv_path = output_dir / f"twse_all_{start_key}_{end_key}.csv"
    parquet_path = output_dir / f"twse_all_{start_key}_{end_key}.parquet"
    
    # Save CSV
    df_to_export = df.copy()
    df_to_export["date"] = df_to_export["date"].dt.strftime("%Y-%m-%d")
    df_to_export.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"💾 Saved CSV: {csv_path}")
    
    # Save Parquet (optional, requires pyarrow/fastparquet)
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"💾 Saved Parquet: {parquet_path}")
    except (ImportError, ValueError) as exc:
        print(f"⚠️  Could not save Parquet file (install pyarrow): {exc}")


# ==================== Main Execution ====================

def run(
    symbols_source: Optional[Path] = None,
    symbols_list: Optional[List[str]] = None,
    start_date: str = "2025-04-01",
    end_date: str = "2025-10-03",
    output_dir: Path = Path("data"),
    limit: Optional[int] = None,
    pause: float = DEFAULT_PAUSE,
) -> None:
    """
    Main execution function to orchestrate the download process.
    
    Args:
        symbols_source: CSV file with stock symbols
        symbols_list: Direct list of stock symbols
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        output_dir: Directory for output files
        limit: Optional limit on number of symbols
        pause: Seconds to pause between requests
    """
    print_banner()
    
    # Load symbols from file or use provided list
    if symbols_list:
        symbols = symbols_list
        print(f"📋 Using {len(symbols)} symbols from command-line")
    elif symbols_source and symbols_source.exists():
        symbols = load_symbols(symbols_source)
        print(f"📋 Loaded {len(symbols)} symbols from {symbols_source}")
    else:
        print(f"❌ Error: No valid symbols source provided")
        return
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        symbols = symbols[:limit]
        print(f"🔢 Limited to first {len(symbols)} symbols")
    
    # Download data
    dataset = download_price_history(symbols, start_date, end_date, pause)
    
    if dataset.empty:
        print("❌ No data downloaded. Please check inputs and try again.")
        return
    
    # Save results
    save_datasets(dataset, output_dir, start_date, end_date)
    print("\n✨ Download complete!\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download TWSE daily stock prices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all symbols from CSV for date range
  python twstock_getid.py --start 2025-01-01 --end 2025-12-31
  
  # Download specific symbols
  python twstock_getid.py --symbols 2330,2317,2454 --start 2025-06-01
  
  # Limit to first 10 symbols for testing
  python twstock_getid.py --limit 10
  
  # Custom output directory and slower requests
  python twstock_getid.py --output-dir ./my_data --pause 0.5
        """
    )
    
    # Symbol sources (mutually exclusive)
    symbol_group = parser.add_mutually_exclusive_group()
    symbol_group.add_argument(
        "--symbols-file",
        type=Path,
        default=Path("STOCK_DAY_ALL_20251003.csv"),
        help="CSV file with stock symbols (default: STOCK_DAY_ALL_20251003.csv)"
    )
    symbol_group.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated stock symbols (e.g., 2330,2317,2454)"
    )
    
    # Date range
    parser.add_argument(
        "--start",
        default="2025-04-01",
        help="Start date in YYYY-MM-DD format (default: 2025-04-01)"
    )
    parser.add_argument(
        "--end",
        default="2025-10-03",
        help="End date in YYYY-MM-DD format (default: 2025-10-03)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for datasets (default: ./data)"
    )
    
    # Performance options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of symbols (useful for testing)"
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=DEFAULT_PAUSE,
        help=f"Seconds between requests (default: {DEFAULT_PAUSE})"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Determine symbols source
    symbols_list = None
    symbols_source = None
    
    if args.symbols:
        symbols_list = parse_symbols_string(args.symbols)
    else:
        symbols_source = args.symbols_file
    
    run(
        symbols_source=symbols_source,
        symbols_list=symbols_list,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output_dir,
        limit=args.limit,
        pause=args.pause,
    )