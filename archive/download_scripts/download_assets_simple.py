#!/usr/bin/env python3
"""
Download multi-asset data using simple pandas CSV download
(bypasses yfinance dependency issues)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time

# Yahoo Finance CSV download URLs
def get_yahoo_csv_url(symbol, start_date, end_date):
    """Generate Yahoo Finance CSV download URL"""
    # Convert dates to Unix timestamps
    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    return f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={start_ts}&period2={end_ts}&interval=1d&events=history"

# Assets to download
ASSETS = {
    'QQQ': 'Nasdaq 100',
    'IWM': 'Russell 2000',
    'DIA': 'Dow Jones',
    'EEM': 'Emerging Markets',
    'TLT': '20+ Year Treasuries',
    'GLD': 'Gold',
    'VTI': 'Total Stock Market',
    'VEA': 'Developed Markets',
}

# Download from 2000 onwards
START_DATE = datetime(2000, 1, 1)
END_DATE = datetime.now()

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("DOWNLOADING MULTI-ASSET DATA (Simple Method)")
print("=" * 80)
print(f"\nDate range: {START_DATE.date()} to {END_DATE.date()}")
print(f"Assets: {len(ASSETS)}\n")

successful = []
failed = []

for symbol, name in ASSETS.items():
    filepath = DATA_DIR / f"{symbol}.csv"

    # Skip if already exists with recent data
    if filepath.exists():
        try:
            existing = pd.read_csv(filepath)
            if len(existing) > 1000:
                last_date = pd.to_datetime(existing['Date'].iloc[-1])
                if (datetime.now() - last_date).days < 7:  # Updated within last week
                    print(f"✓ {symbol:6s} ({name:25s}) - Using cached ({len(existing)} rows)")
                    successful.append(symbol)
                    continue
        except Exception:
            pass

    try:
        print(f"⬇ {symbol:6s} ({name:25s}) - Downloading...", end='', flush=True)

        url = get_yahoo_csv_url(symbol, START_DATE, END_DATE)
        df = pd.read_csv(url)

        if len(df) > 0:
            df.to_csv(filepath, index=False)
            print(f" ✓ {len(df)} rows")
            successful.append(symbol)
        else:
            print(f" ✗ No data")
            failed.append(symbol)

        time.sleep(0.5)  # Rate limiting

    except Exception as e:
        print(f" ✗ Error: {str(e)[:40]}")
        failed.append(symbol)

print("\n" + "=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)
print(f"✓ Successful: {len(successful)}/{len(ASSETS)}")
print(f"✗ Failed: {len(failed)}/{len(ASSETS)}")

if successful:
    print(f"\nDownloaded assets:")
    for symbol in successful:
        filepath = DATA_DIR / f"{symbol}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f"  {symbol:6s}: {len(df):5d} rows ({df['Date'].min()} to {df['Date'].max()})")

# Calculate total samples
total_samples = 0
all_assets = successful.copy()

# Add SPY
spy_path = DATA_DIR / "SPY.csv"
if spy_path.exists():
    spy_df = pd.read_csv(spy_path)
    total_samples += len(spy_df)
    all_assets.append('SPY')
    print(f"  SPY   : {len(spy_df):5d} rows (existing)")

for symbol in successful:
    filepath = DATA_DIR / f"{symbol}.csv"
    if filepath.exists():
        df = pd.read_csv(filepath)
        total_samples += len(df)

print(f"\n" + "=" * 80)
print(f"TOTAL SAMPLES AVAILABLE: {total_samples:,}")
print(f"Total assets: {len(all_assets)}")
print(f"Average per asset: {total_samples // len(all_assets):,}" if all_assets else "N/A")
print("=" * 80)
print(f"\n✓ Ready for multi-asset training!")
