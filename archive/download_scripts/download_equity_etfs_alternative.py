#!/usr/bin/env python3
"""
Alternative equity ETF downloader using multiple data sources

This script tries multiple methods to download equity ETF data:
1. pandas_datareader (uses Yahoo Finance, FRED, etc.)
2. Alpha Vantage API (requires free API key)
3. Direct CSV download with cookies

Fallback order ensures we get data even if one source fails.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import os

DATA_CACHE = Path('data_cache')
DATA_CACHE.mkdir(exist_ok=True)

# Equity ETFs with similar characteristics to SPY
EQUITY_ETFS = {
    'QQQ': 'Nasdaq 100 (Tech-heavy)',
    'IWM': 'Russell 2000 (Small caps)',
    'DIA': 'Dow Jones (Blue chips)',
    'VTI': 'Total Stock Market (Broader)',
    'EEM': 'Emerging Markets (International)',
    'XLF': 'Financials Sector',
    'XLK': 'Technology Sector',
    'XLE': 'Energy Sector',
}

START_DATE = '2000-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

print("=" * 80)
print("EQUITY ETF DATA DOWNLOAD - ALTERNATIVE SOURCES")
print("=" * 80)
print(f"\nTrying multiple data sources in fallback order:")
print(f"  1. pandas_datareader (Yahoo Finance)")
print(f"  2. Alpha Vantage API (if API key available)")
print(f"  3. Manual CSV download instructions")
print("\n" + "=" * 80)

successful = []
failed = []
total_samples = 0


def try_pandas_datareader(ticker):
    """Method 1: Use pandas_datareader"""
    try:
        import pandas_datareader as pdr
        df = pdr.get_data_yahoo(ticker, start=START_DATE, end=END_DATE)
        df.reset_index(inplace=True)
        return df
    except ImportError:
        print(f"      ℹ️  pandas_datareader not installed")
        return None
    except Exception as e:
        print(f"      ⚠️  pandas_datareader failed: {e}")
        return None


def try_alpha_vantage(ticker):
    """Method 2: Use Alpha Vantage API"""
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print(f"      ℹ️  No ALPHA_VANTAGE_API_KEY in environment")
        return None

    try:
        from alpha_vantage.timeseries import TimeSeries
        ts = TimeSeries(key=api_key, output_format='pandas')
        df, _ = ts.get_daily(symbol=ticker, outputsize='full')

        # Rename columns to match our format
        df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }, inplace=True)

        df.reset_index(inplace=True)
        df.rename(columns={'date': 'Date'}, inplace=True)

        # Filter to our date range
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)]

        return df
    except ImportError:
        print(f"      ℹ️  alpha_vantage not installed")
        return None
    except Exception as e:
        print(f"      ⚠️  Alpha Vantage failed: {e}")
        return None


def provide_manual_instructions(ticker):
    """Method 3: Provide manual download instructions"""
    print(f"\n      📋 Manual Download Instructions for {ticker}:")
    print(f"         1. Visit: https://finance.yahoo.com/quote/{ticker}/history")
    print(f"         2. Set date range: {START_DATE} to {END_DATE}")
    print(f"         3. Click 'Download' button")
    print(f"         4. Move downloaded file to: data_cache/{ticker}_1d.csv")
    return None


for ticker, name in EQUITY_ETFS.items():
    filename = f"{ticker}_1d.csv"
    filepath = DATA_CACHE / filename

    # Skip if already downloaded
    if filepath.exists():
        try:
            df = pd.read_csv(filepath)
            print(f"\n✓ {ticker:6s} ({name:30s}) - Already downloaded")
            print(f"   {len(df):,} rows | {df['Date'].min()} to {df['Date'].max()}")
            successful.append(ticker)
            total_samples += len(df)
            continue
        except Exception as e:
            print(f"\n⚠ {ticker:6s} ({name:30s}) - Existing file corrupted, re-downloading...")

    print(f"\n⬇ {ticker:6s} ({name:30s}) - Downloading...")

    df = None

    # Try Method 1: pandas_datareader
    print(f"   [1] Trying pandas_datareader...")
    df = try_pandas_datareader(ticker)

    # Try Method 2: Alpha Vantage
    if df is None:
        print(f"   [2] Trying Alpha Vantage API...")
        df = try_alpha_vantage(ticker)
        if df is not None:
            time.sleep(12)  # Alpha Vantage free tier: 5 calls/min

    # Method 3: Manual instructions
    if df is None:
        print(f"   [3] Automatic download failed")
        provide_manual_instructions(ticker)
        failed.append(ticker)
        continue

    # Save successful download
    try:
        if len(df) == 0:
            raise ValueError("No data returned")

        df.to_csv(filepath, index=False)
        print(f"   ✓ {len(df):,} rows saved")
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")

        successful.append(ticker)
        total_samples += len(df)

        time.sleep(1)  # Brief delay between downloads

    except Exception as e:
        print(f"   ✗ Error saving: {e}")
        failed.append(ticker)

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
    print(f"  python3 train_multi_asset.py")
    print(f"  python3 compare_approaches.py")

if failed:
    print(f"\n⚠️  Failed downloads: {', '.join(failed)}")
    print(f"\nTo get Alpha Vantage API key (FREE):")
    print(f"  1. Visit: https://www.alphavantage.co/support/#api-key")
    print(f"  2. Get free API key")
    print(f"  3. export ALPHA_VANTAGE_API_KEY='your-key-here'")
    print(f"  4. Re-run this script")

    print(f"\nAlternatively, download manually and place in data_cache/")
