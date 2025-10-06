"""
TWSE Stock Selector V2 - Advanced 六脉神剑 Strategy
===========================================================
Implements a comprehensive 10-indicator technical analysis strategy based on
六脉神剑 for Taiwan Stock Exchange (TWSE) stock selection.

Strategy Components (10 Indicators):
    A1: MACD - Moving Average Convergence Divergence
    A2: KD - Stochastic Oscillator
    A3: RSI - Relative Strength Index
    A4: LWR - Larry Williams %R
    A5: BBI - Bull Bear Index
    A6: MMS/MMM - Momentum Indicators
    A7: DBCD - Bias Deviation Indicator
    A8: HOLD - Custom Hold Signal
    A9: ZLGJ - Momentum Strength
    A10: Capital Flow - Volume-based Money Flow

Selection Criteria:
    - Guideline Score: Sum of all 10 indicators (max score: 10)
    - Operation Signal: Triggered when guideline > 9

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ==================== Strategy Parameters ====================
# These parameters control the periods for various technical indicators
N1 = 3   # Short-term period
N2 = 5   # Medium-short period
N3 = 9   # Medium period
N4 = 13  # Medium-long period
N5 = 21  # Long period
N6 = 34  # Extra-long period


# ==================== Basic Technical Functions ====================

def ema(series: pd.Series, span: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        series: Price series
        span: Number of periods for EMA
        
    Returns:
        EMA series
    """
    return series.ewm(span=span, adjust=False).mean()


def ma(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average (MA).
    
    Args:
        series: Price series
        window: Rolling window size
        
    Returns:
        MA series
    """
    return series.rolling(window=window, min_periods=window).mean()


def llv(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate Lowest Low Value over a rolling window.
    
    Args:
        series: Price series
        window: Rolling window size
        
    Returns:
        Series of lowest values
    """
    return series.rolling(window=window, min_periods=1).min()


def hhv(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate Highest High Value over a rolling window.
    
    Args:
        series: Price series
        window: Rolling window size
        
    Returns:
        Series of highest values
    """
    return series.rolling(window=window, min_periods=1).max()


def tdx_sma(series: pd.Series, period: int, weight: int) -> pd.Series:
    """
    Tongdaxin-style SMA (Smoothed Moving Average).
    
    Formula: SMA = (weight * price + (period - weight) * prev_SMA) / period
    
    Args:
        series: Input data series
        period: Smoothing period
        weight: Weight parameter
        
    Returns:
        Smoothed series
        
    Raises:
        ValueError: If period or weight is not positive
    """
    if period <= 0 or weight <= 0:
        raise ValueError("period and weight must be positive integers")

    alpha = weight / period
    values = series.to_numpy(dtype=float, copy=False)
    result = np.empty(len(series), dtype=float)
    result.fill(np.nan)

    prev: Optional[float] = None
    for idx, value in enumerate(values):
        if np.isnan(value):
            result[idx] = prev if prev is not None else np.nan
            continue
        if prev is None or np.isnan(prev):
            prev = value
        else:
            prev = alpha * value + (1 - alpha) * prev
        result[idx] = prev
        
    return pd.Series(result, index=series.index)


def dma(series: pd.Series, alpha: float) -> pd.Series:
    """
    Dynamic Moving Average with fixed smoothing factor.
    
    Args:
        series: Input data series
        alpha: Smoothing factor (0 < alpha <= 1)
        
    Returns:
        Smoothed series
    """
    values = series.to_numpy(dtype=float, copy=False)
    result = np.empty(len(series), dtype=float)
    result.fill(np.nan)

    prev: Optional[float] = None
    for idx, value in enumerate(values):
        if np.isnan(value):
            result[idx] = prev if prev is not None else np.nan
            continue
        if prev is None or np.isnan(prev):
            prev = value
        else:
            prev = alpha * value + (1 - alpha) * prev
        result[idx] = prev
        
    return pd.Series(result, index=series.index)


# ==================== Individual Indicator Signals ====================

def compute_macd_signal(close: pd.Series) -> pd.Series:
    """
    A1: MACD Buy Signal (DIFF > DEA).
    
    Args:
        close: Closing price series
        
    Returns:
        Boolean series for MACD buy condition
    """
    diff = ema(close, N3) - ema(close, N4)
    dea = ema(diff, N2)
    return diff > dea


def compute_kd_signal(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    A2: KD Stochastic Oscillator Buy Signal (K > D).
    
    Args:
        high: High price series
        low: Low price series
        close: Closing price series
        
    Returns:
        Boolean series for KD buy condition
    """
    llv_n3 = llv(low, N3)
    hhv_n3 = hhv(high, N3)
    denom = (hhv_n3 - llv_n3).replace(0, np.nan)
    
    # Calculate RSV (Raw Stochastic Value)
    rsv = ((close - llv_n3) / denom) * 100
    
    # Smooth to get K and D lines
    k = tdx_sma(rsv, N1, 1)
    d = tdx_sma(k, N1, 1)
    
    return k > d


def compute_rsi_signal(close: pd.Series) -> pd.Series:
    """
    A3: RSI Buy Signal (RSI_N2 > RSI_N4).
    
    Args:
        close: Closing price series
        
    Returns:
        Boolean series for RSI buy condition
    """
    prev_close = close.shift(1)
    gain = np.maximum(close - prev_close, 0)
    loss = np.abs(close - prev_close)
    
    # Short-term RSI
    up_n2 = tdx_sma(gain, N2, 1)
    down_n2 = tdx_sma(loss, N2, 1).replace(0, np.nan)
    rsi1 = (up_n2 / down_n2) * 100
    
    # Long-term RSI
    up_n4 = tdx_sma(gain, N4, 1)
    down_n4 = tdx_sma(loss, N4, 1).replace(0, np.nan)
    rsi2 = (up_n4 / down_n4) * 100
    
    return rsi1 > rsi2


def compute_lwr_signal(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    A4: Larry Williams %R Buy Signal (LWR1 > LWR2).
    
    Args:
        high: High price series
        low: Low price series
        close: Closing price series
        
    Returns:
        Boolean series for LWR buy condition
    """
    llv_n4 = llv(low, N4)
    hhv_n4 = hhv(high, N4)
    denom = (hhv_n4 - llv_n4).replace(0, np.nan)
    
    # Inverted Williams %R
    rsv = -((hhv_n4 - close) / denom) * 100
    
    lwr1 = tdx_sma(rsv, N1, 1)
    lwr2 = tdx_sma(lwr1, N1, 1)
    
    return lwr1 > lwr2


def compute_bbi_signal(close: pd.Series) -> pd.Series:
    """
    A5: Bull Bear Index Buy Signal (Close > BBI).
    
    BBI is the average of 4 different period MAs.
    
    Args:
        close: Closing price series
        
    Returns:
        Boolean series for BBI buy condition
    """
    ma1 = ma(close, N1)
    ma2 = ma(close, N2)
    ma3 = ma(close, N3)
    ma4 = ma(close, N4)
    bbi = (ma1 + ma2 + ma3 + ma4) / 4
    
    return close > bbi


def compute_momentum_signal(close: pd.Series) -> pd.Series:
    """
    A6: Momentum Buy Signal (MMS > MMM).
    
    Args:
        close: Closing price series
        
    Returns:
        Boolean series for momentum buy condition
    """
    mtm = close - close.shift(1)
    
    # Short-term momentum (MMS)
    ema_mtm_n2 = ema(mtm, N2)
    denom_mms = ema(ema(np.abs(mtm), N2), N1).replace(0, np.nan)
    mms = 100 * ema(ema_mtm_n2, N1) / denom_mms
    
    # Long-term momentum (MMM)
    ema_mtm_n4 = ema(mtm, N4)
    denom_mmm = ema(ema(np.abs(mtm), N4), N3).replace(0, np.nan)
    mmm = 100 * ema(ema_mtm_n4, N3) / denom_mmm
    
    return mms > mmm


def compute_dbcd_signal(close: pd.Series) -> pd.Series:
    """
    A7: DBCD (Bias Deviation) Buy Signal.
    
    Args:
        close: Closing price series
        
    Returns:
        Boolean series for DBCD buy condition
    """
    ma_close_n2 = ma(close, N2).replace(0, np.nan)
    bias = (close - ma_close_n2) / ma_close_n2 * 100
    
    # Calculate deviation
    dif = bias - bias.shift(16)
    dbcd = tdx_sma(dif, 76, 1)
    mm = ma(dbcd, 5)
    
    return dbcd > mm


def compute_hold_signal(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    A8: Custom HOLD indicator buy signal.
    
    Args:
        high: High price series
        low: Low price series
        close: Closing price series
        
    Returns:
        Boolean series for HOLD buy condition
    """
    llv27 = llv(low, 27)
    hhv27 = hhv(high, 27)
    denom = (hhv27 - llv27).replace(0, np.nan)
    
    rsv27 = ((close - llv27) / denom) * 100
    sma5 = tdx_sma(rsv27, 5, 1)
    sma5_again = tdx_sma(sma5, 3, 1)
    
    # Custom formula
    hold = 3 * sma5 - 2 * sma5_again
    downtrend = ma(hold, 12)
    
    return hold > downtrend


def compute_zlgj_signal(close: pd.Series) -> pd.Series:
    """
    A9: ZLGJ (Momentum Strength) Buy Signal.
    
    Args:
        close: Closing price series
        
    Returns:
        Boolean series for ZLGJ buy condition
    """
    mt = close - close.shift(1)
    ema_mt_n3 = ema(mt, N3)
    denom = ema(ema(np.abs(mt), N3), N3).replace(0, np.nan)
    
    zlgj = 100 * ema(ema_mt_n3, N3) / denom
    mazl = ma(zlgj, 5)
    
    return zlgj > mazl


def compute_capital_flow_signal(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    open_: pd.Series,
    volume: pd.Series
) -> pd.Series:
    """
    A10: Capital Flow Buy Signal (positive inflow).
    
    Uses volume-weighted price analysis to detect capital flow.
    
    Args:
        high: High price series
        low: Low price series
        close: Closing price series
        open_: Opening price series
        volume: Volume series
        
    Returns:
        Boolean series for capital flow buy condition
    """
    # Calculate typical price
    typical_price = (high + low + close * 2) / 4
    pjj = dma(typical_price, 0.9)
    
    # Volume distribution calculation
    denominator = ((high - low) * 2) - np.abs(close - open_)
    denominator = denominator.replace(0, np.nan)
    qjj = volume / denominator
    
    # Bullish volume
    term1 = np.where(
        close > open_,
        qjj * (high - low),
        np.where(close < open_, qjj * (high - open_ + close - low), volume / 2),
    )
    term1 = pd.Series(term1, index=close.index)
    
    # Bearish volume
    term2 = np.where(
        close > open_,
        -qjj * (high - close + open_ - low),
        np.where(close < open_, -qjj * (high - low), -volume / 2),
    )
    term2 = pd.Series(term2, index=close.index)
    
    # Net volume flow
    xvl = term1 + term2
    hsl = (xvl / 20) / 1.15
    
    # Weighted attack flow
    attack_flow = (hsl * 0.55) + (hsl.shift(1) * 0.33) + (hsl.shift(2) * 0.22)
    capital_flow = ema(attack_flow, 3)
    
    return capital_flow > 0


# ==================== Main Signal Computation ====================

def compute_stock_signals(group: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 10 technical indicators for a single stock.
    
    Args:
        group: DataFrame containing price data for one stock
        
    Returns:
        DataFrame enriched with all indicators and signals
        
    Raises:
        ValueError: If volume column is missing
    """
    g = group.sort_values("date").reset_index(drop=True)

    # Extract price data
    close = g["close"].astype(float)
    open_ = g["open"].astype(float)
    high = g["high"].astype(float)
    low = g["low"].astype(float)
    
    volume = g.get("volume")
    if volume is None:
        raise ValueError("Dataset must include a 'volume' column for the strategy.")
    volume = volume.astype(float)

    # Compute all 10 indicator signals
    a1 = compute_macd_signal(close)
    a2 = compute_kd_signal(high, low, close)
    a3 = compute_rsi_signal(close)
    a4 = compute_lwr_signal(high, low, close)
    a5 = compute_bbi_signal(close)
    a6 = compute_momentum_signal(close)
    a7 = compute_dbcd_signal(close)
    a8 = compute_hold_signal(high, low, close)
    a9 = compute_zlgj_signal(close)
    a10 = compute_capital_flow_signal(high, low, close, open_, volume)

    # Calculate guideline score (sum of all signals)
    guideline = (
        a1.astype(int) + a2.astype(int) + a3.astype(int) + a4.astype(int) + a5.astype(int) +
        a6.astype(int) + a7.astype(int) + a8.astype(int) + a9.astype(int) + a10.astype(int)
    )
    
    # Operation signal: triggered when guideline > 9
    operation_signal = guideline > 9

    # Store intermediate calculations for reference
    diff = ema(close, N3) - ema(close, N4)
    dea = ema(diff, N2)
    
    llv_n3 = llv(low, N3)
    hhv_n3 = hhv(high, N3)
    rsv1 = ((close - llv_n3) / (hhv_n3 - llv_n3).replace(0, np.nan)) * 100
    k = tdx_sma(rsv1, N1, 1)
    d = tdx_sma(k, N1, 1)
    
    prev_close = close.shift(1)
    gain = np.maximum(close - prev_close, 0)
    loss = np.abs(close - prev_close)
    up_n2 = tdx_sma(gain, N2, 1)
    down_n2 = tdx_sma(loss, N2, 1).replace(0, np.nan)
    rsi1 = (up_n2 / down_n2) * 100
    up_n4 = tdx_sma(gain, N4, 1)
    down_n4 = tdx_sma(loss, N4, 1).replace(0, np.nan)
    rsi2 = (up_n4 / down_n4) * 100
    
    llv_n4 = llv(low, N4)
    hhv_n4 = hhv(high, N4)
    rsv2 = -((hhv_n4 - close) / (hhv_n4 - llv_n4).replace(0, np.nan)) * 100
    lwr1 = tdx_sma(rsv2, N1, 1)
    lwr2 = tdx_sma(lwr1, N1, 1)
    
    ma1 = ma(close, N1)
    ma2 = ma(close, N2)
    ma3 = ma(close, N3)
    ma4 = ma(close, N4)
    bbi = (ma1 + ma2 + ma3 + ma4) / 4
    
    mtm = close - close.shift(1)
    ema_mtm_n2 = ema(mtm, N2)
    mms = 100 * ema(ema_mtm_n2, N1) / ema(ema(np.abs(mtm), N2), N1).replace(0, np.nan)
    ema_mtm_n4 = ema(mtm, N4)
    mmm = 100 * ema(ema_mtm_n4, N3) / ema(ema(np.abs(mtm), N4), N3).replace(0, np.nan)
    
    ma_close_n2 = ma(close, N2).replace(0, np.nan)
    bias = (close - ma_close_n2) / ma_close_n2 * 100
    dif_bias = bias - bias.shift(16)
    dbcd = tdx_sma(dif_bias, 76, 1)
    mm = ma(dbcd, 5)
    
    llv27 = llv(low, 27)
    hhv27 = hhv(high, 27)
    rsv27 = ((close - llv27) / (hhv27 - llv27).replace(0, np.nan)) * 100
    sma5 = tdx_sma(rsv27, 5, 1)
    sma5_again = tdx_sma(sma5, 3, 1)
    hold = 3 * sma5 - 2 * sma5_again
    downtrend = ma(hold, 12)
    
    mt = close - close.shift(1)
    ema_mt_n3 = ema(mt, N3)
    zlgj = 100 * ema(ema_mt_n3, N3) / ema(ema(np.abs(mt), N3), N3).replace(0, np.nan)
    mazl = ma(zlgj, 5)
    
    typical_price = (high + low + close * 2) / 4
    pjj = dma(typical_price, 0.9)
    denominator = ((high - low) * 2) - np.abs(close - open_)
    qjj = volume / denominator.replace(0, np.nan)
    term1 = np.where(close > open_, qjj * (high - low),
                     np.where(close < open_, qjj * (high - open_ + close - low), volume / 2))
    term2 = np.where(close > open_, -qjj * (high - close + open_ - low),
                     np.where(close < open_, -qjj * (high - low), -volume / 2))
    xvl = pd.Series(term1, index=g.index) + pd.Series(term2, index=g.index)
    hsl = (xvl / 20) / 1.15
    attack_flow = (hsl * 0.55) + (hsl.shift(1) * 0.33) + (hsl.shift(2) * 0.22)
    capital_flow = ema(attack_flow, 3)

    # Add all computed values to dataframe
    g = g.assign(
        diff=diff,
        dea=dea,
        k=k,
        d=d,
        rsi1=rsi1,
        rsi2=rsi2,
        lwr1=lwr1,
        lwr2=lwr2,
        bbi=bbi,
        mms=mms,
        mmm=mmm,
        dbcd=dbcd,
        mm=mm,
        hold=hold,
        downtrend=downtrend,
        zlgj=zlgj,
        mazl=mazl,
        capital_flow=capital_flow,
        guideline=guideline,
        operation_signal=operation_signal,
        a1=a1,
        a2=a2,
        a3=a3,
        a4=a4,
        a5=a5,
        a6=a6,
        a7=a7,
        a8=a8,
        a9=a9,
        a10=a10,
    )

    return g


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute signals for all stocks in the dataset.
    
    Args:
        df: Multi-stock price dataset
        
    Returns:
        Enriched dataset with all indicators
    """
    groups = []
    for stock_id, group in df.groupby("stock_id", sort=False):
        enriched = compute_stock_signals(group)
        
        # Ensure stock_id column exists
        if "stock_id" not in enriched.columns:
            enriched.insert(0, "stock_id", stock_id)
        else:
            enriched = enriched.copy()
            enriched["stock_id"] = stock_id
            
        groups.append(enriched)
        
    if not groups:
        return pd.DataFrame(columns=df.columns)
        
    return pd.concat(groups, ignore_index=True)


# ==================== Stock Selection ====================

def select_stocks(dataset_path: Path, output_path: Optional[Path]) -> pd.DataFrame:
    """
    Load data, compute signals, and select stocks meeting criteria.
    
    Args:
        dataset_path: Path to TWSE dataset CSV
        output_path: Optional path to save selected stocks
        
    Returns:
        DataFrame of stocks meeting selection criteria
        
    Raises:
        ValueError: If required columns are missing
    """
    print(f"📂 Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, parse_dates=["date"])
    df.columns = [col.lower() for col in df.columns]

    # Validate required columns
    required = {"stock_id", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"❌ Dataset missing required columns: {missing}")

    print(f"📊 Processing {len(df['stock_id'].unique())} stocks...")
    
    # Compute all signals
    enriched = compute_signals(df)
    
    if enriched.empty:
        print("⚠️  No data after signal computation.")
        return pd.DataFrame(columns=["stock_id", "date", "close", "guideline", "operation_signal"])

    # Filter for latest date with operation signal
    cutoff = enriched["date"].max()
    print(f"📅 Latest date in dataset: {cutoff.date()}")
    
    latest = enriched.loc[
        (enriched["date"] == cutoff) & enriched["operation_signal"],
        [
            "stock_id", "date", "close", "guideline", "operation_signal",
            "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10",
        ]
    ]
    latest = latest.sort_values("stock_id").reset_index(drop=True)

    print(f"✅ Found {len(latest)} stocks with operation signal (guideline > 9)")

    # Save to file if requested
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        latest.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"💾 Saved selections to {output_path}")

    return latest


# ==================== Command Line Interface ====================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TWSE Stock Selector V2 - Advanced Multi-Indicator Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze dataset and display results
  python stock_selector_v2.py data/twse_all_20250401_20251003.csv
  
  # Save selected stocks to file
  python stock_selector_v2.py data/twse_all_20251003.csv --output picks.csv

Indicator Details:
  A1:  MACD - Trend following momentum
  A2:  KD   - Stochastic oscillator
  A3:  RSI  - Relative strength
  A4:  LWR  - Williams %R
  A5:  BBI  - Bull/Bear balance
  A6:  MOM  - Momentum comparison
  A7:  DBCD - Bias deviation
  A8:  HOLD - Custom hold signal
  A9:  ZLGJ - Momentum strength
  A10: FLOW - Capital flow direction
  
  Guideline: Sum of all 10 indicators (0-10)
  Signal: Triggered when guideline > 9
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
        help="Output CSV path for selected stocks (optional)"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Print header
    print("\n" + "=" * 70)
    print("  TWSE Stock Selector V2 - 10-Indicator Strategy (小鑫课程)")
    print("=" * 70 + "\n")
    
    # Run selection
    selections = select_stocks(args.dataset, args.output)
    
    # Display results
    print("\n" + "=" * 70)
    if selections.empty:
        print("❌ No stocks met the guideline > 9 threshold on the latest date.")
    else:
        print(f"📋 Top {min(20, len(selections))} Selected Stocks:\n")
        print(selections.head(20).to_string(index=False))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()