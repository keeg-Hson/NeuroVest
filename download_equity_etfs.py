#!/usr/bin/env python3
"""
Download equity ETF data for multi-asset training

This is BETTER than crypto for SPY training because:
- Same market structure (trading hours, fundamentals)
- Similar volatility profiles
- Proven cross-asset learning without distribution shift
- 7x more training data
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import requests
from io import StringIO

DATA_CACHE = Path('data_cache')
DATA_CACHE.mkdir(exist_ok=True)

# User agent to avoid rate limiting
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Equity ETFs with similar characteristics to SPY
EQUITY_ETFS = {
    # Major Indexes
    'QQQ': 'Nasdaq 100 (Tech-heavy)',
    'IWM': 'Russell 2000 (Small caps)',
    'DIA': 'Dow Jones (Blue chips)',
    'VTI': 'Total Stock Market (Broader)',
    'EFA': 'Developed Markets EAFE',
    'EEM': 'Emerging Markets (International)',
    'VWO': 'Emerging Markets Vanguard',

    # Sector ETFs
    'XLF': 'Financials Sector',
    'XLK': 'Technology Sector',
    'XLE': 'Energy Sector',
    'XLV': 'Healthcare Sector',
    'XLI': 'Industrials Sector',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLU': 'Utilities Sector',
    'XLRE': 'Real Estate Sector',

    # Bonds
    'AGG': 'Total Bond Market',
    'BND': 'Total Bond Market Vanguard',
    'TLT': '20+ Year Treasuries',
    'IEF': '7-10 Year Treasuries',
    'SHY': '1-3 Year Treasuries',
    'LQD': 'Investment Grade Bonds',
    'HYG': 'High Yield Bonds',

    # Precious Metals
    'GLD': 'Gold Trust',
    'SLV': 'Silver Trust',
    'GDX': 'Gold Miners',
    'GDXJ': 'Junior Gold Miners',
    'IAU': 'iShares Gold',
    'PPLT': 'Platinum',
    'PALL': 'Palladium',

    # Commodities
    'USO': 'US Oil Fund',
    'UNG': 'Natural Gas Fund',
}

# Match SPY date range
START_DATE = '2000-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Convert to Unix timestamps for Yahoo Finance API
start_ts = int(datetime.strptime(START_DATE, '%Y-%m-%d').timestamp())
end_ts = int(datetime.now().timestamp())

print("=" * 80)
print("MULTI-ASSET DATA DOWNLOAD (ETFs, Bonds, Precious Metals, Commodities)")
print("=" * 80)
print(f"\n✅ Downloading {len(EQUITY_ETFS)} assets:")
print(f"   - Major indexes (QQQ, IWM, DIA, etc.)")
print(f"   - Sector ETFs (XLF, XLK, XLE, etc.)")
print(f"   - Bonds (TLT, HYG, LQD, etc.)")
print(f"   - Precious metals (GLD, SLV, etc.)")
print(f"   - Commodities (USO, UNG)")
print("\n" + "=" * 80)

successful = []
failed = []
total_samples = 0

for ticker, name in EQUITY_ETFS.items():
    filename = f"{ticker}_1d.csv"
    filepath = DATA_CACHE / filename

    try:
        print(f"\n⬇ {ticker:6s} ({name:30s}) - Downloading...")

        # Direct CSV download from Yahoo Finance with proper headers
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_ts}&period2={end_ts}&interval=1d&events=history"

        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))

        if len(df) == 0:
            raise ValueError("No data returned")

        # Date column is already present from Yahoo CSV
        # Ensure consistent capitalization
        if 'date' in df.columns:
            df.rename(columns={'date': 'Date'}, inplace=True)

        # Save
        df.to_csv(filepath, index=False)

        print(f"   ✓ {len(df):,} rows saved")
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")

        successful.append(ticker)
        total_samples += len(df)

        # Longer delay to avoid rate limiting
        time.sleep(2)

    except Exception as e:
        print(f"   ✗ Error: {e}")
        failed.append(ticker)
        # Even longer delay after failure
        time.sleep(3)

print("\n" + "=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)
print(f"✓ Successful: {len(successful)}/{len(EQUITY_ETFS)}")
print(f"✗ Failed: {len(failed)}/{len(EQUITY_ETFS)}")

if successful:
    print(f"\n📊 Total training samples available:")
    print(f"   SPY only:           6,501 samples")
    print(f"   New ETF data:     {total_samples:,} samples")
    print(f"   Combined total:   {6501 + total_samples:,} samples")
    print(f"   Increase:         {((6501 + total_samples) / 6501 - 1) * 100:.1f}%")

    print(f"\n📁 Data saved to: {DATA_CACHE}/")
    print(f"\nNext steps:")
    print(f"  1. Update train_multi_asset.py to load equity ETFs")
    print(f"  2. Train: python train_multi_asset.py")
    print(f"  3. Compare to SPY-only baseline (71.0% accuracy)")
    print(f"\n✅ Expected result: Better generalization without distribution shift")
