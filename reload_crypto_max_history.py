#!/usr/bin/env python3
"""
Reload ONLY crypto data with 3000 days (8 years) of history

This script clears old crypto data and reloads with maximum history.
Run this when crypto is stuck at 300 rows.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_manager_postgres import DataManager
from core.scheduler import create_ccxt_callback
from sqlalchemy import text
from datetime import datetime

def main():
    print("\n" + "="*70)
    print("₿ RELOADING CRYPTO WITH MAX HISTORY (3000 DAYS)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    dm = DataManager()

    if dm.backend != 'postgresql':
        print("❌ This script only works with PostgreSQL")
        return 1

    # Crypto assets to reload
    crypto_symbols = [
        ('BTC/USDT', 'BTC_USDT'),
        ('ETH/USDT', 'ETH_USDT'),
        ('SOL/USDT', 'SOL_USDT'),
        ('BNB/USDT', 'BNB_USDT'),
        ('XRP/USDT', 'XRP_USDT'),
        ('ADA/USDT', 'ADA_USDT'),
        ('DOGE/USDT', 'DOGE_USDT'),
        ('AVAX/USDT', 'AVAX_USDT'),
        ('MATIC/USDT', 'MATIC_USDT'),
        ('LINK/USDT', 'LINK_USDT')
    ]

    print("🗑️  Clearing old crypto data...")

    # Delete old crypto data
    with dm.engine.begin() as conn:
        for _, ticker in crypto_symbols:
            conn.execute(text("DELETE FROM price_data WHERE ticker = :ticker"), {"ticker": ticker})
            conn.execute(text("""
                UPDATE asset_metadata
                SET last_update = NULL, last_timestamp = NULL, total_records = 0
                WHERE ticker = :ticker
            """), {"ticker": ticker})

    print("✅ Old crypto data cleared\n")

    print(f"₿ Loading Crypto Data (3000 days = ~8 years)...")
    print(f"Assets: {len(crypto_symbols)}\n")

    crypto_success = 0
    crypto_records = 0

    for symbol, ticker in crypto_symbols:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {ticker}...", end=' ', flush=True)

            # Register asset
            dm.register_asset(ticker, 'crypto', 'daily')

            # Try Coinbase first, fallback to Binance for BNB/MATIC
            data = None
            exchange = 'coinbase'

            try:
                callback = create_ccxt_callback(symbol, exchange, '1d', limit=3000)
                data = callback()
            except Exception as e:
                # If Coinbase fails (BNB/MATIC not available), try Binance
                if ticker in ['BNB_USDT', 'MATIC_USDT']:
                    print(f"Coinbase failed, trying Binance...", end=' ', flush=True)
                    exchange = 'binance'
                    try:
                        callback = create_ccxt_callback(symbol, exchange, '1d', limit=3000)
                        data = callback()
                    except Exception as e2:
                        print(f"❌ Binance also failed: {e2}")
                        continue
                else:
                    print(f"❌ Error: {e}")
                    continue

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
    print(f"  Success: {crypto_success}/{len(crypto_symbols)} assets")
    print(f"  Records: {crypto_records:,} total")
    print(f"  Average: {crypto_records//crypto_success if crypto_success > 0 else 0:,} rows per asset")
    print("="*70 + "\n")

    dm.close()

    if crypto_success >= 8:  # At least 80% success
        print("✅ CRYPTO RELOAD SUCCESSFUL!")
        return 0
    else:
        print(f"⚠️  PARTIAL SUCCESS: Only {crypto_success}/{len(crypto_symbols)} loaded")
        return 1


if __name__ == '__main__':
    sys.exit(main())
