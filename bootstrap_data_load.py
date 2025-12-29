#!/usr/bin/env python3
"""
Bootstrap Data Loader - Load all historical data into database

This runs ONCE to populate the database with all historical data.
After this, the worker handles incremental updates.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_manager import DataManager
from core.scheduler import create_yfinance_callback, create_ccxt_callback
from datetime import datetime

def main():
    print("\n" + "="*70)
    print("📊 LOADING ALL HISTORICAL DATA")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    # Initialize data manager
    dm = DataManager('data/market_data.db')

    # Stock/ETF assets
    stock_tickers = [
        'SPY', 'QQQ', 'IWM', 'DIA',  # Major indices
        'TLT', 'IEF', 'SHY',  # Bonds
        'GLD', 'SLV', 'GDX',  # Precious metals
        'PPLT', 'PALL',  # Platinum/Palladium
        'USO', 'UNG',  # Energy
        'DBA', 'CORN', 'WEAT'  # Agriculture
    ]

    print("📈 Loading Stock/ETF Data (3 years)...")
    print(f"Assets: {len(stock_tickers)}")
    print()

    stock_success = 0
    stock_records = 0

    for ticker in stock_tickers:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {ticker}...", end=' ', flush=True)

            # Register asset
            dm.register_asset(ticker, 'stock', 'daily')

            # Fetch data
            callback = create_yfinance_callback(ticker, period='3y')
            data = callback()

            if data is not None and not data.empty:
                dm.update_from_source(ticker, 'stock', data)
                print(f"✅ {len(data)} records")
                stock_success += 1
                stock_records += len(data)
            else:
                print(f"⚠️  No data")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Crypto assets (Coinbase only - BNB and MATIC not available)
    crypto_symbols = [
        ('BTC/USDT', 'BTC_USDT'),
        ('ETH/USDT', 'ETH_USDT'),
        ('SOL/USDT', 'SOL_USDT'),
        ('XRP/USDT', 'XRP_USDT'),
        ('ADA/USDT', 'ADA_USDT'),
        ('DOGE/USDT', 'DOGE_USDT'),
        ('DOT/USDT', 'DOT_USDT'),
        ('AVAX/USDT', 'AVAX_USDT')
    ]

    print(f"\n₿ Loading Crypto Data (daily, max available)...")
    print(f"Assets: {len(crypto_symbols)}")
    print()

    crypto_success = 0
    crypto_records = 0

    for symbol, ticker in crypto_symbols:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {ticker}...", end=' ', flush=True)

            # Register asset
            dm.register_asset(ticker, 'crypto', 'daily')

            # Fetch data - Coinbase limits to 300, so fetch multiple times
            callback = create_ccxt_callback(symbol, 'coinbase', '1d', limit=300)
            data = callback()

            if data is not None and not data.empty:
                dm.update_from_source(ticker, 'crypto', data)
                print(f"✅ {len(data)} records")
                crypto_success += 1
                crypto_records += len(data)
            else:
                print(f"⚠️  No data")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Summary
    total_success = stock_success + crypto_success
    total_records = stock_records + crypto_records
    total_assets = len(stock_tickers) + len(crypto_symbols)

    stats = dm.get_stats()

    print("\n" + "="*70)
    print("📊 DATA LOAD SUMMARY")
    print("="*70)
    print(f"  Stocks:  {stock_success}/{len(stock_tickers)} assets, {stock_records:,} records")
    print(f"  Crypto:  {crypto_success}/{len(crypto_symbols)} assets, {crypto_records:,} records")
    print(f"  Total:   {total_success}/{total_assets} assets, {total_records:,} records")
    print(f"\n  Database: {stats['db_size_mb']:.1f} MB")
    print("="*70)

    dm.close()

    # Accept partial success if 90%+ of assets loaded
    success_rate = (total_success / total_assets * 100) if total_assets > 0 else 0

    if total_success == total_assets:
        print("\n✅ ALL DATA LOADED SUCCESSFULLY!")
        return 0
    elif success_rate >= 90:
        print(f"\n✅ PARTIAL SUCCESS: {total_success}/{total_assets} assets ({success_rate:.1f}%)")
        print("   Continuing - this is acceptable for production")
        return 0
    else:
        print(f"\n⚠️  INSUFFICIENT DATA: Only {total_success}/{total_assets} assets ({success_rate:.1f}%)")
        return 1


if __name__ == '__main__':
    sys.exit(main())
