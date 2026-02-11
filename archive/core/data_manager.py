"""
Data Manager - Smart data storage and real-time updates
Handles database storage, incremental updates, and caching
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
import threading
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages market data with SQLite backend and intelligent caching

    Features:
    - SQLite database for persistent storage
    - Incremental updates (only fetch new data)
    - Automatic data refresh
    - Multi-asset support (stocks, crypto, commodities)
    - Caching for performance
    """

    def __init__(self, db_path: str = 'data/market_data.db'):
        """Initialize data manager with database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections (fixes thread-safety issue)
        self._local = threading.local()
        self._lock = threading.Lock()

        # Initialize database schema using a temporary connection
        with self._get_connection() as conn:
            self._create_tables(conn)

        # Cache for frequently accessed data
        self._cache = {}
        self._cache_timestamps = {}
        self.cache_ttl = 300  # 5 minutes

        logger.info(f"Data Manager initialized: {self.db_path}")

    def _get_connection(self):
        """Get thread-local database connection (thread-safe)"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=True,  # Enforce thread safety
                timeout=30.0  # Prevent indefinite blocking
            )
        return self._local.conn

    def _create_tables(self, conn):
        """Create database schema for time-series data"""
        cursor = conn.cursor()

        try:
            # Main price data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    adjusted_close REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, timestamp)
                )
            ''')

            # Index for fast queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_ticker_timestamp
                ON price_data(ticker, timestamp DESC)
            ''')

            # Metadata table for tracking updates
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS asset_metadata (
                    ticker TEXT PRIMARY KEY,
                    asset_type TEXT NOT NULL,
                    last_update DATETIME,
                    last_timestamp DATETIME,
                    total_records INTEGER DEFAULT 0,
                    update_frequency TEXT DEFAULT 'daily',
                    enabled INTEGER DEFAULT 1
                )
            ''')

            # Cache statistics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_stats (
                    ticker TEXT PRIMARY KEY,
                    hit_count INTEGER DEFAULT 0,
                    miss_count INTEGER DEFAULT 0,
                    last_access DATETIME
                )
            ''')

            conn.commit()
            logger.info("Database schema initialized")
        except sqlite3.Error as e:
            logger.error(f"Error creating database schema: {e}")
            conn.rollback()
            raise

    def register_asset(self, ticker: str, asset_type: str,
                      update_frequency: str = 'daily'):
        """Register an asset for tracking"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO asset_metadata
                (ticker, asset_type, update_frequency)
                VALUES (?, ?, ?)
            ''', (ticker, asset_type, update_frequency))
            conn.commit()
            logger.info(f"Registered {ticker} ({asset_type})")
        except sqlite3.Error as e:
            logger.error(f"Error registering {ticker}: {e}")
            conn.rollback()
            raise

    def get_last_timestamp(self, ticker: str) -> Optional[datetime]:
        """Get the latest timestamp for an asset (with null check)"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT MAX(timestamp) FROM price_data WHERE ticker = ?
            ''', (ticker,))
            result = cursor.fetchone()

            # FIX: Null check before accessing result[0]
            if result and result[0]:
                return pd.to_datetime(result[0])
            return None
        except sqlite3.Error as e:
            logger.error(f"Error getting last timestamp for {ticker}: {e}")
            return None

    @staticmethod
    def _safe_float(value, default=0.0):
        """Safely convert value to float"""
        try:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    def insert_data(self, ticker: str, asset_type: str, df: pd.DataFrame):
        """
        Insert price data for an asset with transaction management

        Args:
            ticker: Asset ticker symbol
            asset_type: Type (stock, crypto, commodity)
            df: DataFrame with columns [Open, High, Low, Close, Volume, Adj_Close]
        """
        # FIX: Check if DataFrame is empty
        if df.empty:
            logger.warning(f"No data to insert for {ticker}")
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Prepare data for insertion with safe float conversion
            records = []
            for timestamp, row in df.iterrows():
                # Convert timestamp to ISO string for SQLite compatibility
                timestamp_str = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)

                records.append((
                    ticker,
                    asset_type,
                    timestamp_str,
                    self._safe_float(row.get('Open', row.get('open', 0))),
                    self._safe_float(row.get('High', row.get('high', 0))),
                    self._safe_float(row.get('Low', row.get('low', 0))),
                    self._safe_float(row.get('Close', row.get('close', 0))),
                    self._safe_float(row.get('Volume', row.get('volume', 0))),
                    self._safe_float(row.get('Adj_Close', row.get('Close', row.get('close', 0))))
                ))

            # Insert with conflict resolution (update on duplicate)
            cursor.executemany('''
                INSERT OR REPLACE INTO price_data
                (ticker, asset_type, timestamp, open, high, low, close, volume, adjusted_close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)

            # FIX: Safe index access for last timestamp
            last_timestamp = df.index[-1] if len(df) > 0 else None

            # Update metadata
            if last_timestamp:
                # Convert timestamps to ISO strings for SQLite
                last_update_str = datetime.now().isoformat()
                last_timestamp_str = last_timestamp.isoformat() if hasattr(last_timestamp, 'isoformat') else str(last_timestamp)

                cursor.execute('''
                    UPDATE asset_metadata
                    SET last_update = ?,
                        last_timestamp = ?,
                        total_records = (SELECT COUNT(*) FROM price_data WHERE ticker = ?)
                    WHERE ticker = ?
                ''', (last_update_str, last_timestamp_str, ticker, ticker))

            # FIX: Transaction management - commit only if all succeed
            conn.commit()

            # FIX: Invalidate all cache entries for this ticker
            keys_to_delete = [k for k in self._cache if k.startswith(f"{ticker}_")]
            for key in keys_to_delete:
                del self._cache[key]
                if key in self._cache_timestamps:
                    del self._cache_timestamps[key]

            logger.info(f"Inserted {len(records)} records for {ticker}")

        except sqlite3.Error as e:
            logger.error(f"Database error inserting data for {ticker}: {e}")
            conn.rollback()
            raise
        except Exception as e:
            logger.error(f"Unexpected error inserting data for {ticker}: {e}")
            conn.rollback()
            raise

    def get_data(self, ticker: str, start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 use_cache: bool = True) -> pd.DataFrame:
        """
        Retrieve data for an asset with caching

        Args:
            ticker: Asset ticker symbol
            start_date: Start date (YYYY-MM-DD) or None for all
            end_date: End date (YYYY-MM-DD) or None for all
            use_cache: Use cached data if available

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{ticker}_{start_date}_{end_date}"

        # Check cache
        if use_cache and cache_key in self._cache:
            if time.time() - self._cache_timestamps[cache_key] < self.cache_ttl:
                self._update_cache_stats(ticker, hit=True)
                return self._cache[cache_key].copy()

        self._update_cache_stats(ticker, hit=False)

        # Build query
        query = '''
            SELECT timestamp, open, high, low, close, volume, adjusted_close
            FROM price_data
            WHERE ticker = ?
        '''
        params = [ticker]

        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)

        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)

        query += ' ORDER BY timestamp ASC'

        # Execute query with thread-safe connection
        conn = self._get_connection()
        df = pd.read_sql_query(query, conn, params=params,
                               parse_dates=['timestamp'])

        if df.empty:
            return pd.DataFrame()

        # Format dataframe
        df.set_index('timestamp', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']

        # Cache result
        self._cache[cache_key] = df.copy()
        self._cache_timestamps[cache_key] = time.time()

        return df

    def get_multi_asset(self, tickers: List[str],
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Retrieve data for multiple assets"""
        result = {}
        for ticker in tickers:
            df = self.get_data(ticker, start_date, end_date)
            if not df.empty:
                result[ticker] = df
        return result

    def needs_update(self, ticker: str, max_age_hours: int = 24) -> bool:
        """Check if asset data needs updating"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT last_update FROM asset_metadata WHERE ticker = ?
            ''', (ticker,))
            result = cursor.fetchone()

            if not result or not result[0]:
                return True

            last_update = pd.to_datetime(result[0])
            age = datetime.now() - last_update
            return age.total_seconds() / 3600 > max_age_hours
        except sqlite3.Error as e:
            logger.error(f"Error checking update status for {ticker}: {e}")
            return True  # Assume needs update if error occurs

    def get_assets_needing_update(self, max_age_hours: int = 24) -> List[Tuple[str, str]]:
        """Get list of assets that need updating"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT ticker, asset_type FROM asset_metadata
                WHERE enabled = 1
                AND (last_update IS NULL
                     OR datetime(last_update) < datetime('now', '-' || ? || ' hours'))
            ''', (max_age_hours,))
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error getting assets needing update: {e}")
            return []

    def update_from_source(self, ticker: str, asset_type: str,
                          source_data: pd.DataFrame):
        """
        Incremental update - only adds new data

        Args:
            ticker: Asset ticker
            asset_type: Type (stock, crypto, commodity)
            source_data: New data from external source
        """
        # Get last timestamp in database
        last_ts = self.get_last_timestamp(ticker)

        if last_ts is not None:
            # Only insert new data
            new_data = source_data[source_data.index > last_ts]
            if new_data.empty:
                print(f"  {ticker}: Already up to date")
                return
            print(f"  {ticker}: Adding {len(new_data)} new records")
        else:
            new_data = source_data
            print(f"  {ticker}: Initial load - {len(new_data)} records")

        self.insert_data(ticker, asset_type, new_data)

    def _update_cache_stats(self, ticker: str, hit: bool):
        """Update cache statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            now_str = datetime.now().isoformat()

            if hit:
                cursor.execute('''
                    INSERT INTO cache_stats (ticker, hit_count, last_access)
                    VALUES (?, 1, ?)
                    ON CONFLICT(ticker) DO UPDATE SET
                        hit_count = hit_count + 1,
                        last_access = ?
                ''', (ticker, now_str, now_str))
            else:
                cursor.execute('''
                    INSERT INTO cache_stats (ticker, miss_count, last_access)
                    VALUES (?, 1, ?)
                    ON CONFLICT(ticker) DO UPDATE SET
                        miss_count = miss_count + 1,
                        last_access = ?
                ''', (ticker, now_str, now_str))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error updating cache stats for {ticker}: {e}")
            conn.rollback()

    def get_stats(self) -> Dict:
        """Get database and cache statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Asset counts
            cursor.execute('SELECT COUNT(*) FROM asset_metadata WHERE enabled = 1')
            result = cursor.fetchone()
            total_assets = result[0] if result else 0

            # Total records
            cursor.execute('SELECT COUNT(*) FROM price_data')
            result = cursor.fetchone()
            total_records = result[0] if result else 0

            # Cache stats
            cursor.execute('''
                SELECT SUM(hit_count), SUM(miss_count) FROM cache_stats
            ''')
            result = cursor.fetchone()
            cache_hits = result[0] if result and result[0] else 0
            cache_misses = result[1] if result and result[1] else 0

            hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0

            # Database size
            db_size_mb = self.db_path.stat().st_size / (1024 * 1024)

            return {
                'total_assets': total_assets,
                'total_records': total_records,
                'db_size_mb': round(db_size_mb, 2),
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
                'cache_hit_rate': round(hit_rate * 100, 2),
                'cache_size': len(self._cache)
            }
        except sqlite3.Error as e:
            logger.error(f"Error getting database stats: {e}")
            return {
                'total_assets': 0,
                'total_records': 0,
                'db_size_mb': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'cache_hit_rate': 0,
                'cache_size': len(self._cache)
            }

    def clear_cache(self):
        """Clear in-memory cache"""
        self._cache.clear()
        self._cache_timestamps.clear()
        print("✓ Cache cleared")

    def vacuum(self):
        """Optimize database (reclaim space)"""
        print("Optimizing database...")
        conn = self._get_connection()

        try:
            conn.execute('VACUUM')
            conn.commit()
            print("✓ Database optimized")
        except sqlite3.Error as e:
            logger.error(f"Error optimizing database: {e}")
            print(f"✗ Database optimization failed: {e}")

    def export_to_csv(self, ticker: str, output_path: str):
        """Export asset data to CSV"""
        df = self.get_data(ticker)
        if not df.empty:
            df.to_csv(output_path)
            print(f"✓ Exported {ticker} to {output_path}")

    def close(self):
        """Close database connection for current thread"""
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            try:
                self._local.conn.close()
                self._local.conn = None
                logger.info("Database connection closed for current thread")
            except sqlite3.Error as e:
                logger.error(f"Error closing database connection: {e}")
        print("✓ Data Manager closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def test_data_manager():
    """Test the data manager"""
    print("\n" + "="*70)
    print("TESTING DATA MANAGER")
    print("="*70)

    with DataManager('data/test_market_data.db') as dm:
        # Register assets
        dm.register_asset('SPY', 'stock', 'daily')
        dm.register_asset('BTC/USD', 'crypto', 'hourly')

        # Create sample data
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.randn(len(dates)).cumsum() + 100,
            'High': np.random.randn(len(dates)).cumsum() + 102,
            'Low': np.random.randn(len(dates)).cumsum() + 98,
            'Close': np.random.randn(len(dates)).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, len(dates)),
        }, index=dates)
        sample_data['Adj_Close'] = sample_data['Close']

        # Insert data
        dm.insert_data('SPY', 'stock', sample_data)

        # Retrieve data
        spy_data = dm.get_data('SPY', start_date='2024-06-01')
        print(f"\n✓ Retrieved {len(spy_data)} records")
        print(f"  Date range: {spy_data.index[0]} to {spy_data.index[-1]}")

        # Test incremental update
        new_dates = pd.date_range('2025-01-01', '2025-01-10', freq='D')
        new_data = pd.DataFrame({
            'Open': np.random.randn(len(new_dates)).cumsum() + 100,
            'High': np.random.randn(len(new_dates)).cumsum() + 102,
            'Low': np.random.randn(len(new_dates)).cumsum() + 98,
            'Close': np.random.randn(len(new_dates)).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, len(new_dates)),
            'Adj_Close': np.random.randn(len(new_dates)).cumsum() + 100,
        }, index=new_dates)

        dm.update_from_source('SPY', 'stock', new_data)

        # Get stats
        stats = dm.get_stats()
        print("\n" + "="*70)
        print("DATA MANAGER STATISTICS")
        print("="*70)
        for key, value in stats.items():
            print(f"  {key}: {value}")

    print("\n✓ Data Manager test complete")


if __name__ == '__main__':
    test_data_manager()
