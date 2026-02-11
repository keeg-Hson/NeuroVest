#!/usr/bin/env python3
"""
Download crypto data for multi-asset training

WARNING: Adding crypto to SPY training may hurt performance due to:
- Different market dynamics (24/7 vs market hours)
- Much higher volatility (5-10x)
- Different fundamental drivers

Only use if you plan to trade crypto, not just for SPY training.
"""

import ccxt
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DATA_CACHE = Path('data_cache')
DATA_CACHE.mkdir(exist_ok=True)

# Crypto assets to download
CRYPTO_ASSETS = [
    'BTC/USDT',  # Bitcoin - most liquid
    'ETH/USDT',  # Ethereum - 2nd most liquid
    'SOL/USDT',  # Solana - high volatility alt
]

# Date range (match SPY data: 2000-2025)
# But crypto only exists from 2017 onwards on most exchanges
START_DATE = '2017-01-01'  # Binance launch date
END_DATE = datetime.now().strftime('%Y-%m-%d')

print("=" * 80)
print("CRYPTO DATA DOWNLOAD")
print("=" * 80)
print(f"\n⚠️  WARNING: Crypto has very different characteristics from SPY")
print(f"   - 24/7 trading vs market hours")
print(f"   - 5-10x higher volatility")
print(f"   - Different fundamental drivers")
print(f"\n   This may hurt SPY predictions. Consider equity ETFs instead.")
print("\n" + "=" * 80)

# Initialize Binance (free, no API key needed for public data)
exchange = ccxt.binance({
    'enableRateLimit': True,
})

successful = []
failed = []

for symbol in CRYPTO_ASSETS:
    filename = symbol.replace('/', '_') + '_1d.csv'
    filepath = DATA_CACHE / filename

    try:
        print(f"\n⬇ {symbol:12s} - Downloading daily data...")

        # Fetch OHLCV data
        since = exchange.parse8601(f'{START_DATE}T00:00:00Z')
        ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=since)

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Save
        df.to_csv(filepath, index=False)

        print(f"   ✓ {len(df)} rows saved to {filepath}")
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")

        successful.append(symbol)

    except Exception as e:
        print(f"   ✗ Error: {e}")
        failed.append(symbol)

print("\n" + "=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)
print(f"✓ Successful: {len(successful)}/{len(CRYPTO_ASSETS)}")
print(f"✗ Failed: {len(failed)}/{len(CRYPTO_ASSETS)}")

if successful:
    print(f"\n📁 Crypto data saved to: {DATA_CACHE}/")
    print(f"\nNext steps:")
    print(f"  1. Review downloaded data quality")
    print(f"  2. Run: python train_multi_asset.py")
    print(f"  3. Compare performance to SPY-only model")
    print(f"\n⚠️  Recommendation: Test both approaches and compare backtest results")
