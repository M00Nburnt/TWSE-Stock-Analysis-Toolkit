"""
TWSE Stock Selector - 六脉神剑 Strategy
==================================================
Applies a multi-indicator technical analysis strategy to identify buy signals
in Taiwan Stock Exchange (TWSE) data.

Strategy Components:
    - MACD (Moving Average Convergence Divergence)
    - KD Stochastic Oscillator
    - RSI (Relative Strength Index)
    - LWR (Larry Williams %R)
    - BBI (Bull Bear Index)
    - MMS/MMM (Momentum indicators)

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ==================== Technical Indicator Functions ====================

def ema(series: pd.Series, span: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        series: Price series to calculate EMA on
        span: Number of periods for EMA calculation
        
    Returns:
        EMA series with same index as input
    """
    return series.ewm(span=span, adjust=False).mean()


def tdx_sma(series: pd.Series, period: int, weight: int) -> pd.Series:
    """
    Replicate Tongdaxin SMA(X, N, M) smoothing algorithm.
    
    This is a weighted moving average where:
    - SMA(today) = (weight * price(today) + (period - weight) * SMA(yesterday)) / period
    
    Args:
        series: Input data series
        period: Smoothing period (N in Tongdaxin)
        weight: Weight parameter (M in Tongdaxin)
        
    Returns:
        Smoothed series using Tongdaxin algorithm
        
    Raises:
        ValueError: If period or weight is not positive
    """
    if period <= 0 or weight <= 0:
        raise ValueError("period and weight must be positive integers")

    alpha = weight / period
    result = np.empty(len(series), dtype=float)
    result.fill(np.nan)

    prev: Optional[float] = None
    for idx, value in enumerate(series.to_numpy(dtype=float, copy=False)):
        if np.isnan(value):
            # Preserve previous value if current is NaN
            result[idx] = prev if prev is not None else np.nan
            continue
        if prev is None:
            # Initialize with first valid value
            prev = value
        else:
            # Apply weighted smoothing
            prev = alpha * value + (1.0 - alpha) * prev
        result[idx] = prev
        
    return pd.Series(result, index=series.index)


def llv(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate Lowest Low Value over a rolling window.
    
    Args:
        series: Price series
        window: Rolling window size
        
    Returns:
        Series of lowest values in each window
    """
    return series.rolling(window=window, min_periods=1).min()


def hhv(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate Highest High Value over a rolling window.
    
    Args:
        series: Price series
        window: Rolling window size
        
    Returns:
        Series of highest values in each window
    """
    return series.rolling(window=window, min_periods=1).max()


# ==================== Signal Computation ====================

def compute_macd_signal(close: pd.Series) -> pd.Series:
    """
    Compute MACD buy signal (DIFF > DEA).
    
    Args:
        close: Closing price series
        
    Returns:
        Boolean series indicating MACD buy condition
    """
    diff = ema(close, 8) - ema(close, 13)
    dea = ema(diff, 5)
    return diff > dea


def compute_kd_signal(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Compute KD Stochastic Oscillator buy signal (K > D).
    
    Args:
        high: High price series
        low: Low price series
        close: Closing price series
        
    Returns:
        Boolean series indicating KD buy condition
    """
    llv8 = llv(low, 8)
    hhv8 = hhv(high, 8)
    denom = (hhv8 - llv8).replace(0, np.nan)
    
    # Calculate RSV (Raw Stochastic Value)
    rsv = ((close - llv8) / denom) * 100.0
    
    # Smooth to get K and D lines
    k = tdx_sma(rsv, 3, 1)
    d = tdx_sma(k, 3, 1)
    
    return k > d


def compute_rsi_signal(close: pd.Series) -> pd.Series:
    """
    Compute RSI buy signal (RSI5 > RSI13).
    
    Args:
        close: Closing price series
        
    Returns:
        Boolean series indicating RSI buy condition
    """
    prev_close = close.shift(1)
    momentum_up = np.maximum(close - prev_close, 0.0)
    momentum_abs = np.abs(close - prev_close)

    # RSI with 5-period smoothing
    momentum_up_5 = tdx_sma(momentum_up, 5, 1)
    momentum_abs_5 = tdx_sma(momentum_abs, 5, 1).replace(0, np.nan)
    rsi1 = (momentum_up_5 / momentum_abs_5) * 100.0

    # RSI with 13-period smoothing
    momentum_up_13 = tdx_sma(momentum_up, 13, 1)
    momentum_abs_13 = tdx_sma(momentum_abs, 13, 1).replace(0, np.nan)
    rsi2 = (momentum_up_13 / momentum_abs_13) * 100.0
    
    return rsi1 > rsi2


def compute_lwr_signal(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Compute Larry Williams %R buy signal (LWR1 > LWR2).
    
    Args:
        high: High price series
        low: Low price series
        close: Closing price series
        
    Returns:
        Boolean series indicating LWR buy condition
    """
    llv13 = llv(low, 13)
    hhv13 = hhv(high, 13)
    denom = (hhv13 - llv13).replace(0, np.nan)
    
    # Calculate Williams %R (inverted)
    rsv = -((hhv13 - close) / denom) * 100.0
    
    # Smooth to get LWR lines
    lwr1 = tdx_sma(rsv, 3, 1)
    lwr2 = tdx_sma(lwr1, 3, 1)
    
    return lwr1 > lwr2


def compute_bbi_signal(close: pd.Series) -> pd.Series:
    """
    Compute Bull Bear Index (BBI) buy signal (Close > BBI).
    
    BBI is the average of 4 moving averages.
    
    Args:
        close: Closing price series
        
    Returns:
        Boolean series indicating BBI buy condition
    """
    ma3 = close.rolling(window=3, min_periods=1).mean()
    ma5 = close.rolling(window=5, min_periods=1).mean()
    ma8 = close.rolling(window=8, min_periods=1).mean()
    ma13 = close.rolling(window=13, min_periods=1).mean()
    
    bbi = (ma3 + ma5 + ma8 + ma13) / 4.0
    
    return close > bbi


def compute_momentum_signal(close: pd.Series) -> pd.Series:
    """
    Compute Momentum buy signal (MMS > MMM).
    
    Args:
        close: Closing price series
        
    Returns:
        Boolean series indicating momentum buy condition
    """
    mtm = close - close.shift(1)
    
    # Short-term momentum (MMS)
    ema_mtm_5 = ema(mtm, 5)
    denom_mms = ema(ema(np.abs(mtm), 5), 3).replace(0, np.nan)
    mms = 100.0 * ema(ema_mtm_5, 3) / denom_mms
    
    # Long-term momentum (MMM)
    ema_mtm_13 = ema(mtm, 13)
    denom_mmm = ema(ema(np.abs(mtm), 13), 8).replace(0, np.nan)
    mmm = 100.0 * ema(ema_mtm_13, 8) / denom_mmm
    
    return mms > mmm


def compute_stock_signals(group: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators and signals for a single stock.
    
    Args:
        group: DataFrame containing price data for one stock
        
    Returns:
        DataFrame with original data plus computed indicators and signals
    """
    g = group.copy()
    
    # Extract price data
    close = g["close"].astype(float)
    high = g["high"].astype(float)
    low = g["low"].astype(float)

    # Compute individual indicator signals
    tj1 = compute_macd_signal(close)
    tj2 = compute_kd_signal(high, low, close)
    tj3 = compute_rsi_signal(close)
    tj4 = compute_lwr_signal(high, low, close)
    tj5 = compute_bbi_signal(close)
    tj6 = compute_momentum_signal(close)

    # Combine all signals (resonance occurs when all are True)
    combo = tj1 & tj2 & tj3 & tj4 & tj5 & tj6
    
    # Buy signal: resonance just turned True (wasn't True yesterday)
    buy_signal = combo & (~combo.shift(1).fillna(False))

    # Add computed columns to dataframe
    g = g.assign(
        tj1_macd=tj1,
        tj2_kd=tj2,
        tj3_rsi=tj3,
        tj4_lwr=tj4,
        tj5_bbi=tj5,
        tj6_momentum=tj6,
        resonance=combo,
        buy_signal=buy_signal,
    )
    
    return g


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical analysis signals for all stocks in the dataset.
    
    Args:
        df: DataFrame containing multi-stock price data
        
    Returns:
        Enriched DataFrame with all computed indicators and signals
    """
    # Sort by stock and date for proper time-series calculations
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    
    # Process each stock separately
    groups = []
    for stock_id, group in df.groupby("stock_id", sort=False):
        enriched_group = compute_stock_signals(group)
        groups.append(enriched_group)

    # Combine all stocks
    enriched = pd.concat(groups, ignore_index=True)
    return enriched


# ==================== Stock Selection ====================

def select_stocks(dataset_path: Path, output_path: Optional[Path]) -> pd.DataFrame:
    """
    Load TWSE data, compute signals, and extract buy signals.
    
    Args:
        dataset_path: Path to CSV file with stock price data
        output_path: Optional path to save buy signals CSV
        
    Returns:
        DataFrame containing stocks with buy signals on the latest date
        
    Raises:
        ValueError: If required columns are missing from dataset
    """
    # Load data
    print(f"📂 Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, parse_dates=["date"])
    
    # Validate required columns
    expected_cols = {"stock_id", "date", "open", "high", "low", "close"}
    missing = expected_cols - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"❌ Dataset missing required columns: {missing}")

    # Normalize column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    
    print(f"📊 Processing {len(df['stock_id'].unique())} stocks...")

    # Compute all technical signals
    enriched = compute_signals(df)
    
    if enriched.empty:
        print("⚠️  No data after signal computation.")
        return pd.DataFrame(columns=["stock_id", "date", "close", "resonance", "buy_signal"])

    # Filter for latest date with resonance
    dataset_end = enriched["date"].max()
    print(f"📅 Latest date in dataset: {dataset_end.date()}")
    
    latest = enriched.loc[
        enriched["date"].eq(dataset_end) & enriched["resonance"]
    ].copy()

    if latest.empty:
        print("⚠️  No stocks showing resonance on the latest date.")
        buy_signals = pd.DataFrame(columns=["stock_id", "date", "close", "resonance", "buy_signal"])
    else:
        # Mark buy signals occurring on the latest date
        latest["buy_signal"] = latest["buy_signal"] & latest["date"].eq(dataset_end)
        
        # Select relevant columns
        buy_signals = latest[[
            "stock_id", "date", "close", "resonance", "buy_signal",
            "tj1_macd", "tj2_kd", "tj3_rsi", "tj4_lwr", "tj5_bbi", "tj6_momentum"
        ]]
        buy_signals = buy_signals.sort_values("stock_id").reset_index(drop=True)
        
        print(f"✅ Found {len(buy_signals)} stocks with resonance")
        print(f"🎯 {buy_signals['buy_signal'].sum()} stocks with new buy signals")

    # Save to file if path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        buy_signals.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"💾 Saved signals to {output_path}")

    return buy_signals


# ==================== Command Line Interface ====================

def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TWSE Stock Selector - Technical Analysis Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze dataset and display results
  python stock_selector_v1.py data/twse_all_20250401_20251003.csv
  
  # Save signals to file
  python stock_selector_v1.py data/twse_all_20250401_20251003.csv --output signals.csv

Strategy Indicators:
  TJ1: MACD (EMA8-EMA13 > DEA5)
  TJ2: KD Stochastic (K > D)
  TJ3: RSI (RSI5 > RSI13)
  TJ4: Larry Williams %R (LWR1 > LWR2)
  TJ5: Bull Bear Index (Close > BBI)
  TJ6: Momentum (MMS > MMM)
  
  Resonance: All 6 indicators align
  Buy Signal: Resonance just occurred (wasn't true yesterday)
        """
    )
    
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to TWSE dataset CSV (from twstock_getid.py)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for buy signals CSV (optional)"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for the stock selector."""
    args = parse_args()
    
    # Print header
    print("\n" + "=" * 60)
    print("  TWSE Stock Selector - Multi-Indicator Strategy")
    print("=" * 60 + "\n")
    
    # Run selection
    signals = select_stocks(args.dataset, args.output)
    
    # Display results
    print("\n" + "=" * 60)
    if signals.empty:
        print("❌ No buy signals generated for this dataset.")
    else:
        print(f"📋 Top {min(10, len(signals))} Signals (sorted by stock ID):\n")
        print(signals.head(10).to_string(index=False))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()