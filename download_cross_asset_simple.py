#!/usr/bin/env python3
"""
Simple Cross-Asset Data Downloader

Downloads cross-asset data directly from Yahoo Finance using simple HTTP requests.
No external dependencies beyond requests and pandas (already installed).
"""

import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import time

# Create cache directory
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download_yahoo_data(ticker, start_date="2000-01-01"):
    """
    Download data from Yahoo Finance using simple HTTP request

    Args:
        ticker: Yahoo Finance ticker symbol
        start_date: Start date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data
    """
    # Convert dates to Unix timestamps
    start_ts = int(pd.Timestamp(start_date).timestamp())
    end_ts = int(pd.Timestamp.now().timestamp())

    # Yahoo Finance download URL
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
    params = {
        'period1': start_ts,
        'period2': end_ts,
        'interval': '1d',
        'events': 'history'
    }

    try:
        print(f"   Downloading {ticker}...", end=" ")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        # Parse CSV data
        from io import StringIO
        df = pd.read_csv(StringIO(response.text), parse_dates=['Date'], index_col='Date')

        print(f"✅ {len(df)} rows")
        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Parse error: {e}")
        return pd.DataFrame()

def download_all_cross_assets():
    """Download all required cross-asset instruments"""

    print("=" * 80)
    print("DOWNLOADING CROSS-ASSET DATA (Simple HTTP Method)")
    print("=" * 80)
    print("\nThis may take 1-2 minutes...")

    tickers = {
        'SPY': 'S&P 500 ETF',
        'HYG': 'High Yield Corporate Bonds',
        'LQD': 'Investment Grade Corporate Bonds',
        'TLT': '20+ Year Treasury Bonds',
        'GLD': 'Gold ETF',
        '^VIX': 'CBOE Volatility Index',
        '^TNX': '10-Year Treasury Yield',
        '^FVX': '5-Year Treasury Yield',
    }

    data = {}

    for ticker, name in tickers.items():
        print(f"\n{ticker} ({name})")

        # Check cache first
        cache_file = CACHE_DIR / f"{ticker.replace('^', '')}_daily.csv"

        if cache_file.exists():
            # Load from cache
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                print(f"   Loaded from cache: {len(df)} rows")
                data[ticker] = df
                continue
            except Exception as e:
                print(f"   Cache load failed: {e}")

        # Download
        df = download_yahoo_data(ticker)

        if not df.empty:
            # Save to cache
            try:
                df.to_csv(cache_file)
                data[ticker] = df
            except Exception as e:
                print(f"   ⚠️  Cache save failed: {e}")

        # Rate limiting
        time.sleep(0.5)

    print(f"\n✅ Downloaded {len(data)} instruments")
    print(f"💾 Cached in: {CACHE_DIR}/")

    return data

if __name__ == "__main__":
    # Download all data
    data = download_all_cross_assets()

    if data:
        print("\n" + "=" * 80)
        print("DOWNLOAD SUMMARY")
        print("=" * 80)

        for ticker, df in data.items():
            print(f"   {ticker:<8s} {len(df):>6,} rows  ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")

        print("\n✅ Cross-asset data download complete!")
        print("\n🎯 Next step: Run integrate_cross_asset_features.py")
    else:
        print("\n❌ No data downloaded")
