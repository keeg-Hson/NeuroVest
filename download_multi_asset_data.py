#!/usr/bin/env python3
"""
Download multi-asset data for expanded training set

This script downloads historical data for major ETFs to expand our training dataset
from 5,201 samples (SPY only) to 20,000+ samples (multiple assets).

Assets:
- SPY: S&P 500 (already have)
- QQQ: Nasdaq 100
- IWM: Russell 2000 (small caps)
- DIA: Dow Jones Industrial Average
- EEM: Emerging Markets
- TLT: 20+ Year Treasury Bonds
- GLD: Gold
- USO: Oil
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

# Assets to download
ASSETS = {
    'QQQ': 'Nasdaq 100',
    'IWM': 'Russell 2000',
    'DIA': 'Dow Jones',
    'EEM': 'Emerging Markets',
    'TLT': '20+ Year Treasuries',
    'GLD': 'Gold',
    'USO': 'Oil',
    'XLF': 'Financials',
    'XLK': 'Technology',
    'XLE': 'Energy',
}

# Download from 2000 onwards to maximize data
START_DATE = '2000-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("DOWNLOADING MULTI-ASSET DATA")
print("=" * 80)
print(f"\nDate range: {START_DATE} to {END_DATE}")
print(f"Assets: {len(ASSETS)}")
print()

successful = []
failed = []

for symbol, name in ASSETS.items():
    filepath = DATA_DIR / f"{symbol}.csv"

    # Skip if already exists and is recent
    if filepath.exists():
        existing = pd.read_csv(filepath)
        if len(existing) > 1000:  # Has substantial data
            print(f"✓ {symbol:6s} ({name:20s}) - Using existing data ({len(existing)} rows)")
            successful.append(symbol)
            continue

    try:
        print(f"⬇ {symbol:6s} ({name:20s}) - Downloading...", end='', flush=True)

        df = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False)

        if len(df) > 0:
            # Reset index to make Date a column
            df = df.reset_index()

            # Save to CSV
            df.to_csv(filepath, index=False)

            print(f" ✓ {len(df)} rows")
            successful.append(symbol)
        else:
            print(f" ✗ No data")
            failed.append(symbol)

    except Exception as e:
        print(f" ✗ Error: {str(e)[:50]}")
        failed.append(symbol)

print("\n" + "=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)
print(f"✓ Successful: {len(successful)}/{len(ASSETS)}")
print(f"✗ Failed: {len(failed)}/{len(ASSETS)}")

if successful:
    print(f"\nSuccessful downloads:")
    for symbol in successful:
        filepath = DATA_DIR / f"{symbol}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f"  {symbol:6s}: {len(df):5d} rows - {df['Date'].min()} to {df['Date'].max()}")

if failed:
    print(f"\nFailed downloads: {', '.join(failed)}")

# Calculate total available training samples
total_samples = 0
if successful:
    for symbol in successful:
        filepath = DATA_DIR / f"{symbol}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            total_samples += len(df)

    # Add SPY
    spy_path = DATA_DIR / "SPY.csv"
    if spy_path.exists():
        spy_df = pd.read_csv(spy_path)
        total_samples += len(spy_df)
        successful.append('SPY')

print(f"\n" + "=" * 80)
print(f"TOTAL TRAINING SAMPLES AVAILABLE: {total_samples:,}")
print(f"Assets: {len(successful)}")
if len(successful) > 0:
    print(f"Average per asset: {total_samples // len(successful):,}")
else:
    print("Average per asset: N/A (no successful downloads)")
print("=" * 80)
