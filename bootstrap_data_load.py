#!/usr/bin/env python3
"""
Bootstrap Data Loader - Load all historical data into database

This runs ONCE to populate the database with all historical data.
After this, the worker handles incremental updates.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_manager_postgres import DataManager
from core.scheduler import create_yfinance_callback, create_ccxt_callback
from datetime import datetime

def main():
    print("\n" + "="*70)
    print("📊 LOADING ALL HISTORICAL DATA")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    # Initialize data manager (auto-detects DATABASE_URL for PostgreSQL)
    dm = DataManager()

    # Stock/ETF assets - ALL 31 stock/ETF/commodity assets
    stock_tickers = [
        # Major indices (6)
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'EEM',
        # Sector ETFs (3)
        'XLF', 'XLK', 'XLE',
        # Bonds & Treasury (6)
        'TLT', 'IEF', 'SHY', 'HYG', 'LQD', 'TNX',
        # Dollar (2)
        'DXY', 'UUP',
    ]

    # Precious metals (7)
    precious_metals = [
        'GLD', 'SLV', 'GDX', 'GDXJ', 'IAU', 'PPLT', 'PALL'
    ]

    # Commodities - Energy & Agriculture (5)
    commodities = [
        'USO', 'UNG',  # Energy
        'DBA', 'CORN', 'WEAT'  # Agriculture
    ]

    # Combine for loading
    all_stock_assets = stock_tickers + precious_metals + commodities

    print("📈 Loading Stock/ETF/Metals/Commodities Data (MAX HISTORY)...")
    print(f"Stocks & ETFs: {len(stock_tickers)}")
    print(f"Precious Metals: {len(precious_metals)}")
    print(f"Commodities: {len(commodities)}")
    print(f"Total: {len(all_stock_assets)}")
    print()

    stock_success = 0
    stock_records = 0

    for ticker in all_stock_assets:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {ticker}...", end=' ', flush=True)

            # Register asset
            dm.register_asset(ticker, 'stock', 'daily')

            # Fetch data - use 'max' for all available history
            callback = create_yfinance_callback(ticker, period='max')
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

    # Dashboard's CRYPTO_ASSETS (10 assets)
    # Note: Using Coinbase - BNB not available (Binance-specific), try others
    crypto_symbols = [
        ('BTC/USDT', 'BTC_USDT'),
        ('ETH/USDT', 'ETH_USDT'),
        ('SOL/USDT', 'SOL_USDT'),
        ('BNB/USDT', 'BNB_USDT'),      # Try Binance exchange instead
        ('XRP/USDT', 'XRP_USDT'),
        ('ADA/USDT', 'ADA_USDT'),
        ('DOGE/USDT', 'DOGE_USDT'),
        ('AVAX/USDT', 'AVAX_USDT'),
        ('MATIC/USDT', 'MATIC_USDT'),
        ('LINK/USDT', 'LINK_USDT')
    ]

    print(f"\n₿ Loading Crypto Data (3000 days = ~8 years)...")
    print(f"Assets: {len(crypto_symbols)}")
    print()

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
            except Exception:
                # If Coinbase fails (BNB/MATIC not available), try Binance
                if ticker in ['BNB_USDT', 'MATIC_USDT']:
                    exchange = 'binance'
                    callback = create_ccxt_callback(symbol, exchange, '1d', limit=3000)
                    data = callback()

            if data is not None and not data.empty:
                dm.update_from_source(ticker, 'crypto', data)
                print(f"✅ {len(data)} records ({exchange})")
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
