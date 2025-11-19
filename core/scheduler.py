"""
Data Update Scheduler - Automated real-time data updates
Runs in background and updates data at specified intervals
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
import schedule
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_manager import DataManager


class DataScheduler:
    """
    Automated data update scheduler

    Features:
    - Background thread for continuous operation
    - Configurable update intervals per asset
    - Market hours awareness
    - Error handling and retry logic
    - Update callbacks for custom data sources
    """

    def __init__(self, data_manager: DataManager):
        """
        Initialize scheduler

        Args:
            data_manager: DataManager instance for data storage
        """
        self.dm = data_manager
        self.running = False
        self.thread = None
        self.update_callbacks = {}  # ticker -> update function
        self.last_updates = {}  # ticker -> last update time
        self.update_intervals = {}  # ticker -> interval in minutes

        print("✓ Data Scheduler initialized")

    def register_update_callback(self, ticker: str, callback: Callable,
                                 interval_minutes: int = 60):
        """
        Register a data update function for an asset

        Args:
            ticker: Asset ticker symbol
            callback: Function that returns updated DataFrame
            interval_minutes: Update interval in minutes
        """
        self.update_callbacks[ticker] = callback
        self.update_intervals[ticker] = interval_minutes
        print(f"✓ Registered update callback for {ticker} (every {interval_minutes}min)")

    def _should_update(self, ticker: str) -> bool:
        """Check if asset should be updated based on interval"""
        if ticker not in self.last_updates:
            return True

        last_update = self.last_updates[ticker]
        interval = self.update_intervals.get(ticker, 60)
        elapsed = (datetime.now() - last_update).total_seconds() / 60

        return elapsed >= interval

    def _is_market_hours(self, asset_type: str = 'stock') -> bool:
        """Check if market is open (US stock market hours)"""
        now = datetime.now()
        weekday = now.weekday()

        # Weekend check
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Market hours: 9:30 AM - 4:00 PM ET (approximate)
        # For simplicity, using 9:00 AM - 5:00 PM local time
        hour = now.hour

        if asset_type == 'stock':
            return 9 <= hour < 17
        elif asset_type == 'crypto':
            return True  # Crypto markets are 24/7
        else:
            return True

    def update_asset(self, ticker: str, asset_type: str):
        """Update a single asset"""
        try:
            # Check if should update
            if not self._should_update(ticker):
                return

            # Check market hours for stocks
            if asset_type == 'stock' and not self._is_market_hours('stock'):
                return

            # Call update callback
            if ticker in self.update_callbacks:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Updating {ticker}...")

                # Get new data from callback
                new_data = self.update_callbacks[ticker]()

                if new_data is not None and not new_data.empty:
                    # Incremental update
                    self.dm.update_from_source(ticker, asset_type, new_data)
                    self.last_updates[ticker] = datetime.now()
                    print(f"  ✓ {ticker} updated successfully")
                else:
                    print(f"  ⚠️  No new data for {ticker}")

        except Exception as e:
            print(f"  ✗ Error updating {ticker}: {e}")

    def update_all(self):
        """Update all registered assets"""
        print(f"\n{'='*70}")
        print(f"SCHEDULED UPDATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        # Get assets from database that are enabled
        assets = self.dm.get_assets_needing_update(max_age_hours=1)

        if not assets:
            print("  All assets up to date")
            return

        print(f"  Updating {len(assets)} assets...")

        for ticker, asset_type in assets:
            self.update_asset(ticker, asset_type)

        # Print stats after update
        stats = self.dm.get_stats()
        print(f"\n  Database: {stats['total_records']} records, {stats['db_size_mb']} MB")
        print(f"  Cache hit rate: {stats['cache_hit_rate']}%")

    def _run_schedule(self):
        """Background thread function"""
        while self.running:
            schedule.run_pending()
            time.sleep(1)

    def start(self, update_interval_minutes: int = 60):
        """
        Start the scheduler

        Args:
            update_interval_minutes: How often to check for updates
        """
        if self.running:
            print("Scheduler already running")
            return

        # Schedule periodic updates
        schedule.every(update_interval_minutes).minutes.do(self.update_all)

        # Start background thread
        self.running = True
        self.thread = threading.Thread(target=self._run_schedule, daemon=True)
        self.thread.start()

        print(f"✓ Scheduler started (updates every {update_interval_minutes} minutes)")
        print(f"  Background thread: {self.thread.name}")

    def stop(self):
        """Stop the scheduler"""
        if not self.running:
            return

        self.running = False
        schedule.clear()

        if self.thread:
            self.thread.join(timeout=5)

        print("✓ Scheduler stopped")

    def run_once(self):
        """Manually trigger an update"""
        self.update_all()


# ==============================================================================
# Example Update Callbacks
# ==============================================================================

def create_yfinance_callback(ticker: str, period: str = "5d"):
    """
    Create a yfinance-based update callback

    Args:
        ticker: Stock ticker symbol
        period: Data period (1d, 5d, 1mo, etc.)

    Returns:
        Callback function that fetches data
    """
    def fetch_data():
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            if df.empty:
                return None

            # Rename columns to match expected format
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            df['Adj_Close'] = df['Close']

            return df

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None

    return fetch_data


def create_ccxt_callback(symbol: str, exchange_name: str = 'binance',
                        timeframe: str = '1d', limit: int = 100):
    """
    Create a CCXT-based crypto update callback

    Args:
        symbol: Crypto symbol (e.g., BTC/USDT)
        exchange_name: Exchange name (binance, coinbase, etc.)
        timeframe: Timeframe (1m, 5m, 1h, 1d)
        limit: Number of candles to fetch

    Returns:
        Callback function that fetches crypto data
    """
    def fetch_data():
        try:
            import ccxt

            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class()

            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['Adj_Close'] = df['Close']

            return df

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None

    return fetch_data


def create_fallback_callback(ticker: str, base_price: float = 100.0):
    """
    Create a synthetic data callback for testing

    Args:
        ticker: Asset ticker
        base_price: Starting price

    Returns:
        Callback that generates synthetic data
    """
    def fetch_data():
        import pandas as pd
        import numpy as np

        # Generate last 5 days of synthetic data
        dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
        returns = np.random.randn(5) * 0.02

        prices = base_price * (1 + returns).cumprod()

        df = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(5) * 0.002),
            'High': prices * (1 + np.abs(np.random.randn(5)) * 0.005),
            'Low': prices * (1 - np.abs(np.random.randn(5)) * 0.005),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 5),
            'Adj_Close': prices
        }, index=dates)

        return df

    return fetch_data


# ==============================================================================
# Main Script
# ==============================================================================

def setup_default_scheduler():
    """Setup scheduler with default stock assets"""
    print("\n" + "="*70)
    print("SETTING UP DATA SCHEDULER")
    print("="*70)

    # Initialize data manager
    dm = DataManager('data/market_data.db')

    # Initialize scheduler
    scheduler = DataScheduler(dm)

    # Register stock assets
    stock_tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'SLV', 'GDX', 'PPLT', 'PALL', 'USO', 'DBA']

    for ticker in stock_tickers:
        # Register asset in database
        dm.register_asset(ticker, 'stock', 'daily')

        # Try yfinance first, fallback to synthetic
        try:
            import yfinance
            callback = create_yfinance_callback(ticker, period='5d')
            print(f"  Using yfinance for {ticker}")
        except ImportError:
            callback = create_fallback_callback(ticker)
            print(f"  Using synthetic data for {ticker}")

        # Register update callback (every 60 minutes)
        scheduler.register_update_callback(ticker, callback, interval_minutes=60)

    # Register crypto assets
    crypto_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']

    for symbol in crypto_symbols:
        ticker = symbol.replace('/', '_')

        # Register asset in database
        dm.register_asset(ticker, 'crypto', 'hourly')

        # Try CCXT first, fallback to synthetic
        try:
            import ccxt
            callback = create_ccxt_callback(symbol, 'binance', '1h', limit=100)
            print(f"  Using CCXT for {symbol}")
        except ImportError:
            callback = create_fallback_callback(ticker, base_price=50000)
            print(f"  Using synthetic data for {symbol}")

        # Register update callback (every 15 minutes for crypto)
        scheduler.register_update_callback(ticker, callback, interval_minutes=15)

    return scheduler, dm


def run_scheduler_daemon(update_interval: int = 60):
    """
    Run scheduler as a daemon

    Args:
        update_interval: Update check interval in minutes
    """
    scheduler, dm = setup_default_scheduler()

    print(f"\n{'='*70}")
    print("STARTING DATA SCHEDULER DAEMON")
    print(f"{'='*70}")
    print(f"  Update interval: {update_interval} minutes")
    print(f"  Press Ctrl+C to stop\n")

    try:
        # Run initial update
        scheduler.run_once()

        # Start scheduler
        scheduler.start(update_interval_minutes=update_interval)

        # Keep running
        while True:
            time.sleep(10)
            # Print status every 10 seconds
            stats = dm.get_stats()
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Assets: {stats['total_assets']} | "
                  f"Records: {stats['total_records']} | "
                  f"Cache: {stats['cache_hit_rate']}% hit rate",
                  end='', flush=True)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        scheduler.stop()
        dm.close()
        print("✓ Scheduler stopped gracefully")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Data Update Scheduler')
    parser.add_argument('--interval', type=int, default=60,
                       help='Update interval in minutes (default: 60)')
    parser.add_argument('--run-once', action='store_true',
                       help='Run update once and exit')

    args = parser.parse_args()

    if args.run_once:
        scheduler, dm = setup_default_scheduler()
        scheduler.run_once()
        dm.close()
    else:
        run_scheduler_daemon(args.interval)
