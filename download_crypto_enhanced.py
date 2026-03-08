#!/usr/bin/env python3
"""
Crypto data downloader.

Source: Binance via ccxt (free, no API key, paginated from 2017+).
CoinGecko is intentionally excluded — its free tier returns weekly
granularity for long date ranges, producing only ~330 records.

Saves to data_cache/ as {BASE}_{QUOTE}_1d.csv (e.g. BTC_USDT_1d.csv).

Usage:
    python3 download_crypto_enhanced.py
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import time

DATA_CACHE = Path('data_cache')
DATA_CACHE.mkdir(exist_ok=True)

# Must match config/assets.yaml crypto section
CRYPTO_ASSETS = {
    'BTC/USDT':  'Bitcoin',
    'ETH/USDT':  'Ethereum',
    'BNB/USDT':  'Binance Coin',
    'SOL/USDT':  'Solana',
    'XRP/USDT':  'Ripple',
    'ADA/USDT':  'Cardano',
    'AVAX/USDT': 'Avalanche',
    'DOT/USDT':  'Polkadot',
    'MATIC/USDT':'Polygon',
    'LINK/USDT': 'Chainlink',
}

# Binance started serving most pairs from 2017-08-17.
# Use 2017-01-01 so we catch any earlier pairs without error.
BINANCE_START = '2017-01-01T00:00:00Z'


def download_from_binance(symbol: str) -> pd.DataFrame | None:
    """
    Paginated daily OHLCV fetch from Binance via ccxt.
    Returns a DataFrame with columns [Date, Open, High, Low, Close, Volume],
    or None on failure.
    """
    try:
        import ccxt
    except ImportError:
        print("   ccxt not installed — run: pip install ccxt")
        return None

    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        since = exchange.parse8601(BINANCE_START)

        all_ohlcv = []
        while True:
            batch = exchange.fetch_ohlcv(symbol, '1d', since=since, limit=1000)
            if not batch:
                break
            all_ohlcv.extend(batch)
            last_ts = batch[-1][0]
            since = last_ts + 86_400_000  # advance by 1 day (ms)
            if since > exchange.milliseconds():
                break
            if len(batch) < 1000:
                break
            time.sleep(0.3)

        if not all_ohlcv:
            return None

        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = (
            df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            .drop_duplicates('Date')
            .sort_values('Date')
            .reset_index(drop=True)
        )
        return df

    except Exception as e:
        print(f"   Binance error: {e}")
        return None


def main():
    print("=" * 80)
    print("CRYPTO DATA DOWNLOAD")
    print(f"Source: Binance (paginated daily OHLCV, 2017+)")
    print(f"Assets: {len(CRYPTO_ASSETS)}")
    print("=" * 80)

    successful = []
    failed = []

    for symbol, name in CRYPTO_ASSETS.items():
        filename = symbol.replace('/', '_') + '_1d.csv'
        filepath = DATA_CACHE / filename

        print(f"\n⬇ {symbol:12s} ({name})")

        df = download_from_binance(symbol)

        if df is not None and len(df) > 0:
            df.to_csv(filepath, index=False)
            years = (df['Date'].max() - df['Date'].min()).days / 365.25
            print(f"   ✓ {len(df):,} rows ({years:.1f} years)")
            print(f"   {df['Date'].min().date()} → {df['Date'].max().date()}")
            successful.append(symbol)
        else:
            print(f"   ✗ No data retrieved")
            failed.append(symbol)

    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"✓ Successful: {len(successful)}/{len(CRYPTO_ASSETS)}")
    if failed:
        print(f"✗ Failed:     {', '.join(failed)}")
    print(f"\nData saved to: {DATA_CACHE}/")


if __name__ == "__main__":
    main()
