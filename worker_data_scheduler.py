#!/usr/bin/env python3
"""
Render Background Worker - Continuous Data Scheduler
Runs 24/7 to keep market data updated
"""

import os
import sys
import time
import signal
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_manager import DataManager
from core.scheduler import (
    DataScheduler,
    create_yfinance_callback,
    create_ccxt_callback,
    create_fallback_callback
)


class WorkerScheduler:
    """Production-ready worker for Render"""

    def __init__(self):
        self.running = False
        self.dm = None
        self.scheduler = None

    def setup(self):
        """Initialize data manager and scheduler"""
        print("\n" + "="*70)
        print("🚀 NEUROVEST DATA WORKER - STARTING")
        print("="*70)
        print(f"  Platform: Render")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Python: {sys.version.split()[0]}")
        print("="*70 + "\n")

        # Initialize data manager
        db_path = os.getenv('DATABASE_PATH', 'data/market_data.db')
        self.dm = DataManager(db_path)

        # Initialize scheduler
        self.scheduler = DataScheduler(self.dm)

        # Register stock assets
        stock_tickers = [
            'SPY', 'QQQ', 'IWM', 'DIA',  # Major indices
            'TLT', 'IEF', 'SHY',  # Bonds
            'GLD', 'SLV', 'GDX',  # Precious metals
            'PPLT', 'PALL',  # Platinum/Palladium
            'USO', 'UNG',  # Energy
            'DBA', 'CORN', 'WEAT'  # Agriculture
        ]

        print("📊 Registering stock/commodity assets...")
        for ticker in stock_tickers:
            self.dm.register_asset(ticker, 'stock', 'daily')

            try:
                callback = create_yfinance_callback(ticker, period='5d')
                self.scheduler.register_update_callback(
                    ticker, callback, interval_minutes=60
                )
                print(f"  ✓ {ticker} (yfinance, 60min)")
            except Exception as e:
                print(f"  ⚠️  {ticker} - Using fallback: {e}")
                callback = create_fallback_callback(ticker)
                self.scheduler.register_update_callback(
                    ticker, callback, interval_minutes=60
                )

        # Register crypto assets
        crypto_symbols = [
            ('BTC/USDT', 'BTC_USDT'),
            ('ETH/USDT', 'ETH_USDT'),
            ('BNB/USDT', 'BNB_USDT'),
            ('SOL/USDT', 'SOL_USDT'),
            ('XRP/USDT', 'XRP_USDT'),
            ('ADA/USDT', 'ADA_USDT'),
            ('DOGE/USDT', 'DOGE_USDT'),
            ('MATIC/USDT', 'MATIC_USDT'),
            ('DOT/USDT', 'DOT_USDT'),
            ('AVAX/USDT', 'AVAX_USDT')
        ]

        print("\n₿ Registering crypto assets...")
        for symbol, ticker in crypto_symbols:
            self.dm.register_asset(ticker, 'crypto', 'hourly')

            try:
                callback = create_ccxt_callback(
                    symbol, 'binance', '1h', limit=100
                )
                self.scheduler.register_update_callback(
                    ticker, callback, interval_minutes=15
                )
                print(f"  ✓ {ticker} (CCXT, 15min)")
            except Exception as e:
                print(f"  ⚠️  {ticker} - Using fallback: {e}")
                callback = create_fallback_callback(ticker, base_price=50000)
                self.scheduler.register_update_callback(
                    ticker, callback, interval_minutes=15
                )

        print("\n✅ Worker setup complete!\n")

    def run(self):
        """Main worker loop"""
        self.setup()
        self.running = True

        # Run initial update
        print("🔄 Running initial data update...")
        self.scheduler.run_once()

        # Start scheduler (checks every 60 minutes)
        update_interval = int(os.getenv('UPDATE_INTERVAL', '60'))
        self.scheduler.start(update_interval_minutes=update_interval)

        print(f"\n{'='*70}")
        print(f"✅ WORKER RUNNING - Updates every {update_interval} minutes")
        print(f"{'='*70}")
        print("  Press Ctrl+C to stop\n")

        # Keep running and print status
        try:
            while self.running:
                time.sleep(30)  # Update status every 30 seconds

                stats = self.dm.get_stats()
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Assets: {stats['total_assets']:>3} | "
                      f"Records: {stats['total_records']:>8,} | "
                      f"Cache: {stats['cache_hit_rate']:>3}% | "
                      f"DB: {stats['db_size_mb']:.1f}MB",
                      end='', flush=True)

        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown"""
        print("\n\n" + "="*70)
        print("🛑 SHUTTING DOWN WORKER")
        print("="*70)

        self.running = False

        if self.scheduler:
            self.scheduler.stop()
            print("  ✓ Scheduler stopped")

        if self.dm:
            self.dm.close()
            print("  ✓ Database closed")

        print("\n✅ Worker shutdown complete")
        sys.exit(0)


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\n\n⚠️  Received signal {signum}")
    worker.shutdown()


if __name__ == '__main__':
    # Create worker instance
    worker = WorkerScheduler()

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Run worker
    worker.run()
