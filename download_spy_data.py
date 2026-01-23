#!/usr/bin/env python3
"""
Download SPY data from Yahoo Finance
Supports both yfinance and direct CSV download (fallback)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

SPY_PATH = DATA_DIR / "SPY.csv"
START_DATE = "1993-01-29"  # SPY inception date

print("=" * 80)
print("DOWNLOADING SPY DATA")
print("=" * 80)
print(f"Date range: {START_DATE} to present")
print(f"Output: {SPY_PATH}\n")

# Try Method 1: yfinance (preferred - includes adjusted close)
try:
    import yfinance as yf

    print("⬇ Downloading via yfinance...")
    # Use auto_adjust=True to get simpler column structure
    df = yf.download('SPY', start=START_DATE, progress=False, auto_adjust=True)

    if len(df) > 0:
        # Flatten column names if they're multi-level (newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Reset index to make Date a column
        df.index.name = 'Date'
        df.reset_index(inplace=True)

        # Ensure Adj Close column exists (auto_adjust=True removes it)
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']

        # Save to CSV
        df.to_csv(SPY_PATH, index=False)

        print(f"✓ Downloaded {len(df):,} rows")
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  File size: {SPY_PATH.stat().st_size / 1024:.1f} KB")
        print(f"\n✓ Saved to {SPY_PATH}")
        sys.exit(0)
    else:
        print("⚠ yfinance returned no data, trying fallback...")

except Exception as e:
    print(f"⚠ yfinance failed: {e}")
    print("  Trying direct CSV download...")

# Method 2: Direct CSV download from Yahoo (fallback)
try:
    # Convert date to Unix timestamp
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.now()

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    # Yahoo Finance direct CSV URL
    url = f"https://query1.finance.yahoo.com/v7/finance/download/SPY?period1={start_ts}&period2={end_ts}&interval=1d&events=history"

    print(f"⬇ Downloading via direct CSV: {url[:80]}...")
    df = pd.read_csv(url)

    if len(df) > 0:
        df.to_csv(SPY_PATH, index=False)

        print(f"✓ Downloaded {len(df):,} rows")
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"  File size: {SPY_PATH.stat().st_size / 1024:.1f} KB")
        print(f"\n✓ Saved to {SPY_PATH}")
        sys.exit(0)
    else:
        print("✗ Direct CSV returned no data")

except Exception as e:
    print(f"✗ Direct CSV failed: {e}")

# If both methods failed
print("\n" + "=" * 80)
print("❌ DOWNLOAD FAILED")
print("=" * 80)
print("\nTroubleshooting:")
print("1. Check internet connection")
print("2. Upgrade dependencies: pip3 install -r requirements.txt --upgrade")
print("3. Try manual download from: https://finance.yahoo.com/quote/SPY/history")
print("4. Save as data/SPY.csv with columns: Date,Open,High,Low,Close,Volume")
sys.exit(1)
from __future__ import annotations

import os

import pandas as pd

from utils import load_SPY_data  # canonical SPY loader (data/SPY.csv)

PREDICTIONS_PATH = "logs/daily_predictions_cleaned.csv"
OUTPUT_PATH = "logs/daily_predictions_enriched.csv"


def main():
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(f"{PREDICTIONS_PATH} not found.")

    pred_df = pd.read_csv(PREDICTIONS_PATH)
    if "Date" not in pred_df.columns:
        raise KeyError(f"'Date' column missing in {PREDICTIONS_PATH}")

    # Robust date parsing with fast-path ISO and a flexible fallback
    date_series = pred_df["Date"].astype(str).str.strip()
    try:
        pred_df["Date"] = pd.to_datetime(date_series, format="%Y-%m-%d", errors="coerce")
        if pred_df["Date"].isna().all():
            raise ValueError("all-na after strict ISO parse; falling back")
    except Exception:
        pred_df["Date"] = pd.to_datetime(date_series, errors="coerce", utc=False)

    # Canonical SPY → reset index to merge on "Date"
    spy = load_SPY_data().reset_index()
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in spy.columns]
    spy = spy[keep]

    merged = pred_df.merge(spy, on="Date", how="left")
    merged = merged[merged["Date"].notna()]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"[✅] Final enriched predictions written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
