"""
Data Update Script - Manual data update interface
Use this script to update data on-demand or set up automated updates
"""

import argparse
from datetime import datetime
from core.data_manager_postgres import DataManager
from core.scheduler import (
    DataScheduler,
    create_yfinance_callback,
    create_ccxt_callback,
    create_fallback_callback
)


def update_all_assets(use_live_sources: bool = True):
    """
    Update all registered assets

    Args:
        use_live_sources: Use real API sources (yfinance, ccxt) if True
    """
    print("\n" + "="*70)
    print("UPDATING ALL ASSETS")
    print("="*70)
    print(f"Mode: {'LIVE APIs' if use_live_sources else 'SYNTHETIC DATA'}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Initialize data manager
    dm = DataManager('data/market_data.db')

    # Stock assets
    stock_tickers = [
        'SPY',   # S&P 500
        'QQQ',   # Nasdaq 100
        'IWM',   # Russell 2000
        'TLT',   # 20+ Year Treasury
        'GLD',   # Gold
        'SLV',   # Silver
        'GDX',   # Gold Miners
        'PPLT',  # Platinum
        'PALL',  # Palladium
        'USO',   # Oil
        'DBA'    # Agriculture
    ]

    print(f"📊 Updating {len(stock_tickers)} stock/commodity assets...")

    for ticker in stock_tickers:
        # Register if not already registered
        dm.register_asset(ticker, 'stock', 'daily')

        try:
            if use_live_sources:
                # Try yfinance
                import yfinance as yf
                stock = yf.Ticker(ticker)
                df = stock.history(period='max')  # Get all historical data

                if not df.empty:
                    df = df.rename(columns={'Open': 'Open', 'High': 'High',
                                           'Low': 'Low', 'Close': 'Close',
                                           'Volume': 'Volume'})
                    df['Adj_Close'] = df.get('Adj Close', df['Close'])
                    dm.update_from_source(ticker, 'stock', df)
                else:
                    print(f"  ⚠️  No data for {ticker}")
            else:
                # Use synthetic data
                callback = create_fallback_callback(ticker)
                df = callback()
                dm.update_from_source(ticker, 'stock', df)

        except Exception as e:
            print(f"  ✗ Error updating {ticker}: {e}")

    # Crypto assets
    crypto_symbols = [
        ('BTC/USDT', 'BTC_USDT'),
        ('ETH/USDT', 'ETH_USDT'),
        ('BNB/USDT', 'BNB_USDT'),
        ('SOL/USDT', 'SOL_USDT'),
        ('XRP/USDT', 'XRP_USDT')
    ]

    print(f"\n₿ Updating {len(crypto_symbols)} crypto assets...")

    for symbol, ticker in crypto_symbols:
        # Register if not already registered
        dm.register_asset(ticker, 'crypto', 'hourly')

        try:
            if use_live_sources:
                # Try CCXT
                import ccxt
                exchange = ccxt.binance()
                ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=1000)

                if ohlcv:
                    import pandas as pd
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    df['Adj_Close'] = df['Close']
                    dm.update_from_source(ticker, 'crypto', df)
                else:
                    print(f"  ⚠️  No data for {symbol}")
            else:
                # Use synthetic data
                callback = create_fallback_callback(ticker, base_price=50000)
                df = callback()
                dm.update_from_source(ticker, 'crypto', df)

        except Exception as e:
            print(f"  ✗ Error updating {symbol}: {e}")

    # Print final statistics
    print("\n" + "="*70)
    print("UPDATE COMPLETE")
    print("="*70)

    stats = dm.get_stats()
    print(f"\n  Total assets: {stats['total_assets']}")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Database size: {stats['db_size_mb']} MB")
    print(f"  Cache hit rate: {stats['cache_hit_rate']}%")

    dm.close()


def start_scheduler(interval_minutes: int = 60):
    """
    Start the automated data update scheduler

    Args:
        interval_minutes: Update check interval in minutes
    """
    from core.scheduler import run_scheduler_daemon
    run_scheduler_daemon(update_interval=interval_minutes)


def query_data(ticker: str, start_date: str = None, end_date: str = None):
    """
    Query data for a specific asset

    Args:
        ticker: Asset ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    print(f"\n📊 Querying data for {ticker}")
    if start_date:
        print(f"  Start date: {start_date}")
    if end_date:
        print(f"  End date: {end_date}")

    dm = DataManager('data/market_data.db')
    df = dm.get_data(ticker, start_date, end_date)

    if df.empty:
        print(f"\n  ⚠️  No data found for {ticker}")
    else:
        print(f"\n  Found {len(df)} records")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nLast 5 rows:")
        print(df.tail())

        # Statistics
        print(f"\nStatistics:")
        print(f"  Latest close: ${df['Close'].iloc[-1]:.2f}")
        print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        print(f"  Avg volume: {df['Volume'].mean():,.0f}")

    dm.close()


def export_data(ticker: str, output_file: str):
    """
    Export data to CSV

    Args:
        ticker: Asset ticker symbol
        output_file: Output CSV file path
    """
    print(f"\n📤 Exporting {ticker} to {output_file}")

    dm = DataManager('data/market_data.db')
    dm.export_to_csv(ticker, output_file)
    dm.close()


def show_stats():
    """Show database statistics"""
    print("\n" + "="*70)
    print("DATABASE STATISTICS")
    print("="*70)

    dm = DataManager('data/market_data.db')
    stats = dm.get_stats()

    print(f"\nAssets: {stats['total_assets']}")
    print(f"Records: {stats['total_records']}")
    print(f"Database size: {stats['db_size_mb']} MB")
    print(f"\nCache performance:")
    print(f"  Hits: {stats['cache_hits']}")
    print(f"  Misses: {stats['cache_misses']}")
    print(f"  Hit rate: {stats['cache_hit_rate']}%")
    print(f"  Cached items: {stats['cache_size']}")

    dm.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='NeuroVest Data Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Update all data once
  python update_data.py update

  # Start automated scheduler (updates every hour)
  python update_data.py schedule

  # Start scheduler with custom interval
  python update_data.py schedule --interval 30

  # Query specific asset
  python update_data.py query SPY --start 2024-01-01

  # Export to CSV
  python update_data.py export SPY spy_data.csv

  # Show statistics
  python update_data.py stats
        '''
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Update command
    update_parser = subparsers.add_parser('update', help='Update all assets once')
    update_parser.add_argument('--synthetic', action='store_true',
                              help='Use synthetic data instead of live APIs')

    # Schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Start automated scheduler')
    schedule_parser.add_argument('--interval', type=int, default=60,
                                help='Update interval in minutes (default: 60)')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query data for an asset')
    query_parser.add_argument('ticker', help='Asset ticker symbol')
    query_parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    query_parser.add_argument('--end', help='End date (YYYY-MM-DD)')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export data to CSV')
    export_parser.add_argument('ticker', help='Asset ticker symbol')
    export_parser.add_argument('output', help='Output CSV file path')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')

    args = parser.parse_args()

    if args.command == 'update':
        update_all_assets(use_live_sources=not args.synthetic)
    elif args.command == 'schedule':
        start_scheduler(args.interval)
    elif args.command == 'query':
        query_data(args.ticker, args.start, args.end)
    elif args.command == 'export':
        export_data(args.ticker, args.output)
    elif args.command == 'stats':
        show_stats()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
