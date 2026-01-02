#!/usr/bin/env python3
"""
Reload crypto data with 3000 days by fetching in chunks

Coinbase limits responses to 300 candles, so we need to fetch
multiple batches going backwards in time.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_manager_postgres import DataManager
import ccxt
import pandas as pd
from sqlalchemy import text
from datetime import datetime, timedelta
import time

def fetch_crypto_history(symbol, exchange_name='coinbase', days=3000):
    """Fetch crypto data in chunks to overcome API limits"""

    exchange = getattr(ccxt, exchange_name)()
    all_data = []

    # Calculate how many chunks we need (300 candles per chunk)
    chunk_size = 300
    num_chunks = (days // chunk_size) + 1

    print(f"   Fetching {days} days in {num_chunks} chunks of {chunk_size}...", end=' ', flush=True)

    # Start from now and go backwards
    since = None

    for i in range(num_chunks):
        try:
            # Fetch chunk
            if since:
                ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=since, limit=chunk_size)
            else:
                ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=chunk_size)

            if not ohlcv:
                break

            # Convert to DataFrame
            df_chunk = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
            df_chunk = df_chunk.set_index('timestamp')

            all_data.append(df_chunk)

            # Get timestamp for next chunk (go backwards)
            oldest_timestamp = int(df_chunk.index.min().timestamp() * 1000)
            since = oldest_timestamp - (chunk_size * 24 * 60 * 60 * 1000)  # Go back chunk_size days

            # Don't hammer the API
            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:
            print(f"chunk {i+1} failed: {e}", end=' ')
            break

    if not all_data:
        return None

    # Combine all chunks
    combined_df = pd.concat(all_data)
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]  # Remove duplicates
    combined_df = combined_df.sort_index()

    # Add Adj_Close column (same as close for crypto)
    combined_df['Adj_Close'] = combined_df['close']

    return combined_df

def main():
    print("\n" + "="*70)
    print("₿ RELOADING CRYPTO WITH MAX HISTORY (3000 DAYS IN CHUNKS)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    dm = DataManager()

    if dm.backend != 'postgresql':
        print("❌ This script only works with PostgreSQL")
        return 1

    # Crypto assets to reload
    crypto_configs = [
        ('BTC/USDT', 'BTC_USDT', 'coinbase'),
        ('ETH/USDT', 'ETH_USDT', 'coinbase'),
        ('SOL/USDT', 'SOL_USDT', 'coinbase'),
        ('BNB/USDT', 'BNB_USDT', 'binance'),  # Not on Coinbase
        ('XRP/USDT', 'XRP_USDT', 'coinbase'),
        ('ADA/USDT', 'ADA_USDT', 'coinbase'),
        ('DOGE/USDT', 'DOGE_USDT', 'coinbase'),
        ('AVAX/USDT', 'AVAX_USDT', 'coinbase'),
        ('MATIC/USDT', 'MATIC_USDT', 'binance'),  # Not on Coinbase
        ('LINK/USDT', 'LINK_USDT', 'coinbase')
    ]

    print("🗑️  Clearing old crypto data...")

    # Delete old crypto data
    with dm.engine.begin() as conn:
        for _, ticker, _ in crypto_configs:
            conn.execute(text("DELETE FROM price_data WHERE ticker = :ticker"), {"ticker": ticker})
            conn.execute(text("""
                UPDATE asset_metadata
                SET last_update = NULL, last_timestamp = NULL, total_records = 0
                WHERE ticker = :ticker
            """), {"ticker": ticker})

    print("✅ Old crypto data cleared\n")

    print(f"₿ Loading Crypto Data (3000 days in chunks)...")
    print(f"Assets: {len(crypto_configs)}\n")

    crypto_success = 0
    crypto_records = 0

    for symbol, ticker, exchange in crypto_configs:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {ticker}...", end=' ', flush=True)

            # Register asset
            dm.register_asset(ticker, 'crypto', 'daily')

            # Fetch data in chunks
            data = fetch_crypto_history(symbol, exchange, days=3000)

            if data is not None and not data.empty:
                dm.update_from_source(ticker, 'crypto', data)
                print(f"✅ {len(data)} records ({exchange})")
                crypto_success += 1
                crypto_records += len(data)
            else:
                print(f"⚠️  No data")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "="*70)
    print("📊 CRYPTO RELOAD SUMMARY")
    print("="*70)
    print(f"  Success: {crypto_success}/{len(crypto_configs)} assets")
    print(f"  Records: {crypto_records:,} total")
    print(f"  Average: {crypto_records//crypto_success if crypto_success > 0 else 0:,} rows per asset")
    print("="*70 + "\n")

    dm.close()

    if crypto_success >= 8:
        print("✅ CRYPTO RELOAD SUCCESSFUL!")
        return 0
    else:
        print(f"⚠️  PARTIAL SUCCESS: Only {crypto_success}/{len(crypto_configs)} loaded")
        return 1


if __name__ == '__main__':
    sys.exit(main())
