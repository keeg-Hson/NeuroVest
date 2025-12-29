"""
Data Manager - PostgreSQL/SQLite dual-mode with diagnostic logging
Automatically uses PostgreSQL if DATABASE_URL is set, otherwise SQLite
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import json
import time
import threading
import logging

# Import database libraries
import sqlite3
try:
    from sqlalchemy import create_engine, text, MetaData, Table, Column
    from sqlalchemy import Integer, String, Float, DateTime, Index
    from sqlalchemy.exc import SQLAlchemyError
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Dual-mode data manager supporting both PostgreSQL and SQLite

    Features:
    - Auto-detects DATABASE_URL for PostgreSQL
    - Falls back to SQLite if no DATABASE_URL
    - Diagnostic logging for debugging
    - Thread-safe connections
    - Identical interface regardless of backend
    """

    def __init__(self, db_path: str = 'data/market_data.db'):
        """Initialize data manager - auto-detects Postgres vs SQLite"""

        # Check for PostgreSQL DATABASE_URL first
        database_url = os.environ.get("DATABASE_URL", "")

        if database_url and database_url.startswith("postgres"):
            try:
                self._init_postgresql(database_url)
            except Exception as e:
                logger.error(f"PostgreSQL initialization failed: {e}")
                logger.info("Falling back to SQLite")
                self._init_sqlite(db_path)
        else:
            self._init_sqlite(db_path)

        # Cache for frequently accessed data
        self._cache = {}
        self._cache_timestamps = {}
        self.cache_ttl = 300  # 5 minutes

    def _init_postgresql(self, database_url: str):
        """Initialize PostgreSQL backend"""
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy required for PostgreSQL but not installed")

        # Log DB fingerprint for diagnostics
        u = urlparse(database_url)
        logger.info("="*70)
        logger.info("🔍 DATABASE CONNECTION DIAGNOSTICS")
        logger.info("="*70)
        logger.info(f"[DB] Backend: PostgreSQL")
        logger.info(f"[DB] Scheme: {u.scheme}")
        logger.info(f"[DB] Host: {u.hostname}")
        logger.info(f"[DB] Port: {u.port}")
        logger.info(f"[DB] Database: {u.path.lstrip('/')}")
        logger.info(f"[DB] User: {u.username}")
        logger.info("="*70)

        self.backend = 'postgresql'
        self.engine = create_engine(database_url, pool_pre_ping=True, connect_args={'connect_timeout': 10})
        self._create_tables_postgres()

        logger.info("Data Manager initialized: PostgreSQL")

    def _init_sqlite(self, db_path: str):
        """Initialize SQLite backend"""
        logger.info("="*70)
        logger.info("🔍 DATABASE CONNECTION DIAGNOSTICS")
        logger.info("="*70)
        logger.info(f"[DB] Backend: SQLite")
        logger.info(f"[DB] Path: {db_path}")
        logger.info("="*70)

        self.backend = 'sqlite'
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections
        self._local = threading.local()
        self._lock = threading.Lock()

        # Initialize schema
        with self._get_connection() as conn:
            self._create_tables_sqlite(conn)

        logger.info(f"Data Manager initialized: SQLite - {self.db_path}")

    def _get_connection(self):
        """Get database connection (SQLite only)"""
        if self.backend != 'sqlite':
            raise RuntimeError("_get_connection() only for SQLite backend")

        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=True,
                timeout=30.0
            )
        return self._local.conn

    def _create_tables_postgres(self):
        """Create PostgreSQL schema"""
        with self.engine.begin() as conn:
            # Main price data table
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS price_data (
                    id SERIAL PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    adjusted_close REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, timestamp)
                )
            '''))

            # Index for fast queries
            conn.execute(text('''
                CREATE INDEX IF NOT EXISTS idx_ticker_timestamp
                ON price_data(ticker, timestamp DESC)
            '''))

            # Metadata table
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS asset_metadata (
                    ticker TEXT PRIMARY KEY,
                    asset_type TEXT NOT NULL,
                    last_update TIMESTAMP,
                    last_timestamp TIMESTAMP,
                    total_records INTEGER DEFAULT 0,
                    update_frequency TEXT
                )
            '''))

            # Log row count for diagnostics
            result = conn.execute(text("SELECT COUNT(*) FROM price_data"))
            count = result.scalar()
            logger.info(f"[DB] Current price_data rows: {count:,}")

    def _create_tables_sqlite(self, conn):
        """Create SQLite schema"""
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

            # Metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS asset_metadata (
                    ticker TEXT PRIMARY KEY,
                    asset_type TEXT NOT NULL,
                    last_update DATETIME,
                    last_timestamp DATETIME,
                    total_records INTEGER DEFAULT 0,
                    update_frequency TEXT
                )
            ''')

            conn.commit()

            # Log row count for diagnostics
            cursor.execute("SELECT COUNT(*) FROM price_data")
            count = cursor.fetchone()[0]
            logger.info(f"[DB] Current price_data rows: {count:,}")

        except sqlite3.Error as e:
            logger.error(f"Error creating SQLite tables: {e}")
            conn.rollback()

    def register_asset(self, ticker: str, asset_type: str, frequency: str = 'daily'):
        """Register a new asset for tracking"""
        if self.backend == 'postgresql':
            with self.engine.begin() as conn:
                conn.execute(text('''
                    INSERT INTO asset_metadata (ticker, asset_type, update_frequency)
                    VALUES (:ticker, :asset_type, :frequency)
                    ON CONFLICT (ticker) DO UPDATE SET
                        asset_type = :asset_type,
                        update_frequency = :frequency
                '''), {"ticker": ticker, "asset_type": asset_type, "frequency": frequency})
        else:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO asset_metadata
                    (ticker, asset_type, update_frequency)
                    VALUES (?, ?, ?)
                ''', (ticker, asset_type, frequency))
                conn.commit()

    def _normalize_price_df(self, df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
        """Normalize column names and data types from different sources (yfinance, CCXT)"""
        df = df.copy()

        # If timestamp is the index (common with yfinance), move it into a column
        if "timestamp" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                # Handle various index names
                if df.columns[0] in ['Date', 'Datetime', 'date', 'datetime']:
                    df = df.rename(columns={df.columns[0]: "timestamp"})
            elif "Date" in df.columns:
                df = df.rename(columns={"Date": "timestamp"})
            elif "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "timestamp"})
            elif "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})

        # Standardize OHLCV column names (handle yfinance capitalization)
        rename_map = {
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
            "Adj Close": "adjusted_close", "AdjClose": "adjusted_close", "Adjusted Close": "adjusted_close",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # CCXT sometimes returns timestamp as milliseconds
        if "timestamp" in df.columns:
            # If it's numeric milliseconds, convert
            if pd.api.types.is_numeric_dtype(df["timestamp"]):
                # Heuristic: big numbers are ms
                if df["timestamp"].dropna().astype("int64").max() > 10_000_000_000:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
                else:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(None)
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

        # Ensure required columns exist (fill missing with NaN)
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        for c in required:
            if c not in df.columns:
                df[c] = pd.NA

        # If adjusted_close missing, use close
        if "adjusted_close" not in df.columns:
            df["adjusted_close"] = df["close"]

        # Keep only what we insert
        keep = ["timestamp", "open", "high", "low", "close", "volume", "adjusted_close"]
        return df[keep].dropna(subset=["timestamp"])

    def save_data(self, ticker: str, df: pd.DataFrame, asset_type: str = 'stock'):
        """Save price data with diagnostic logging"""
        if df is None or len(df) == 0:
            logger.warning(f"No data to save for {ticker}")
            return

        print(f"[DB] Attempting to insert {len(df)} rows for {ticker}")
        print(f"[DB] Columns before normalize: {list(df.columns)}")

        # Normalize column names and data types
        df_clean = self._normalize_price_df(df, asset_type)

        print(f"[DB] Columns after normalize: {list(df_clean.columns)}")
        if len(df_clean) > 0:
            print(f"[DB] Sample row: {df_clean.head(1).to_dict('records')[0]}")
        else:
            print(f"[DB] WARNING: DataFrame empty after normalization!")
            return

        # Add ticker and asset_type columns
        df_clean['ticker'] = ticker
        df_clean['asset_type'] = asset_type

        if self.backend == 'postgresql':
            self._save_data_postgres(ticker, df_clean)
        else:
            self._save_data_sqlite(ticker, df_clean)

    def _save_data_postgres(self, ticker: str, df: pd.DataFrame):
        """Save data to PostgreSQL with diagnostics"""
        try:
            # Get count before insert
            with self.engine.begin() as conn:
                before_count = conn.execute(text("SELECT COUNT(*) FROM price_data")).scalar()

                # Insert data (ON CONFLICT DO NOTHING for duplicates)
                for _, row in df.iterrows():
                    conn.execute(text('''
                        INSERT INTO price_data
                        (ticker, asset_type, timestamp, open, high, low, close, volume, adjusted_close)
                        VALUES (:ticker, :asset_type, :timestamp, :open, :high, :low, :close, :volume, :adj_close)
                        ON CONFLICT (ticker, timestamp) DO NOTHING
                    '''), {
                        "ticker": row['ticker'],
                        "asset_type": row['asset_type'],
                        "timestamp": row['timestamp'],
                        "open": row.get('open'),
                        "high": row.get('high'),
                        "low": row.get('low'),
                        "close": row.get('close'),
                        "volume": row.get('volume'),
                        "adj_close": row.get('adjusted_close')
                    })

                # Get count after insert
                after_count = conn.execute(text("SELECT COUNT(*) FROM price_data")).scalar()
                inserted = after_count - before_count

                print(f"[DB] ✓ Inserted {inserted} new rows for {ticker}")
                print(f"[DB] Total price_data rows: {after_count:,}")

                # Update metadata
                conn.execute(text('''
                    UPDATE asset_metadata
                    SET last_update = :now,
                        last_timestamp = :last_ts,
                        total_records = (SELECT COUNT(*) FROM price_data WHERE ticker = :ticker)
                    WHERE ticker = :ticker
                '''), {
                    "now": datetime.now(),
                    "last_ts": df['timestamp'].max(),
                    "ticker": ticker
                })

        except Exception as e:
            logger.error(f"Error saving to PostgreSQL for {ticker}: {e}")
            raise

    def _save_data_sqlite(self, ticker: str, df: pd.DataFrame):
        """Save data to SQLite with diagnostics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            try:
                # Get count before insert
                cursor.execute("SELECT COUNT(*) FROM price_data")
                before_count = cursor.fetchone()[0]

                # Convert timestamps to ISO strings for SQLite
                df_to_save = df.copy()
                df_to_save['timestamp'] = df_to_save['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

                # Insert data
                inserted = 0
                for _, row in df_to_save.iterrows():
                    try:
                        cursor.execute('''
                            INSERT OR IGNORE INTO price_data
                            (ticker, asset_type, timestamp, open, high, low, close, volume, adjusted_close)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            row['ticker'], row['asset_type'], row['timestamp'],
                            row.get('open'), row.get('high'), row.get('low'),
                            row.get('close'), row.get('volume'), row.get('adjusted_close')
                        ))
                        if cursor.rowcount > 0:
                            inserted += 1
                    except sqlite3.IntegrityError:
                        continue

                # Get count after insert
                cursor.execute("SELECT COUNT(*) FROM price_data")
                after_count = cursor.fetchone()[0]

                print(f"[DB] ✓ Inserted {inserted} new rows for {ticker}")
                print(f"[DB] Total price_data rows: {after_count:,}")

                # Update metadata
                cursor.execute('''
                    UPDATE asset_metadata
                    SET last_update = ?,
                        last_timestamp = ?,
                        total_records = (SELECT COUNT(*) FROM price_data WHERE ticker = ?)
                    WHERE ticker = ?
                ''', (datetime.now(), df['timestamp'].max(), ticker, ticker))

                conn.commit()

            except sqlite3.Error as e:
                logger.error(f"Error saving to SQLite for {ticker}: {e}")
                conn.rollback()
                raise

    def get_data(self, ticker: str, start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Get price data for a ticker"""

        # Check cache first
        cache_key = f"{ticker}_{start_date}_{end_date}"
        if cache_key in self._cache:
            cache_time = self._cache_timestamps.get(cache_key, 0)
            if time.time() - cache_time < self.cache_ttl:
                return self._cache[cache_key].copy()

        if self.backend == 'postgresql':
            df = self._get_data_postgres(ticker, start_date, end_date)
        else:
            df = self._get_data_sqlite(ticker, start_date, end_date)

        # Update cache
        if df is not None and len(df) > 0:
            self._cache[cache_key] = df.copy()
            self._cache_timestamps[cache_key] = time.time()

        return df

    def _get_data_postgres(self, ticker: str, start_date: Optional[datetime],
                           end_date: Optional[datetime]) -> Optional[pd.DataFrame]:
        """Get data from PostgreSQL"""
        try:
            query = "SELECT * FROM price_data WHERE ticker = :ticker"
            params = {"ticker": ticker}

            if start_date:
                query += " AND timestamp >= :start_date"
                params["start_date"] = start_date
            if end_date:
                query += " AND timestamp <= :end_date"
                params["end_date"] = end_date

            query += " ORDER BY timestamp ASC"

            with self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)

            if len(df) == 0:
                return None

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            return df

        except Exception as e:
            logger.error(f"Error reading from PostgreSQL for {ticker}: {e}")
            return None

    def _get_data_sqlite(self, ticker: str, start_date: Optional[datetime],
                         end_date: Optional[datetime]) -> Optional[pd.DataFrame]:
        """Get data from SQLite"""
        try:
            with self._get_connection() as conn:
                query = "SELECT * FROM price_data WHERE ticker = ?"
                params = [ticker]

                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date.strftime('%Y-%m-%d %H:%M:%S'))
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date.strftime('%Y-%m-%d %H:%M:%S'))

                query += " ORDER BY timestamp ASC"

                df = pd.read_sql_query(query, conn, params=params)

            if len(df) == 0:
                return None

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            return df

        except sqlite3.Error as e:
            logger.error(f"Error reading from SQLite for {ticker}: {e}")
            return None

    def get_stats(self) -> Dict:
        """Get database statistics"""
        if self.backend == 'postgresql':
            return self._get_stats_postgres()
        else:
            return self._get_stats_sqlite()

    def _get_stats_postgres(self) -> Dict:
        """Get PostgreSQL statistics"""
        try:
            with self.engine.connect() as conn:
                total_assets = conn.execute(text("SELECT COUNT(*) FROM asset_metadata")).scalar()
                total_records = conn.execute(text("SELECT COUNT(*) FROM price_data")).scalar()

                return {
                    'total_assets': total_assets or 0,
                    'total_records': total_records or 0,
                    'cache_hit_rate': 0,  # TODO: implement
                    'db_size_mb': 0  # TODO: implement
                }
        except Exception as e:
            logger.error(f"Error getting PostgreSQL stats: {e}")
            return {'total_assets': 0, 'total_records': 0, 'cache_hit_rate': 0, 'db_size_mb': 0}

    def _get_stats_sqlite(self) -> Dict:
        """Get SQLite statistics"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM asset_metadata")
                total_assets = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM price_data")
                total_records = cursor.fetchone()[0]

                db_size_mb = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0

                return {
                    'total_assets': total_assets,
                    'total_records': total_records,
                    'cache_hit_rate': 0,  # TODO: implement
                    'db_size_mb': db_size_mb
                }
        except sqlite3.Error as e:
            logger.error(f"Error getting SQLite stats: {e}")
            return {'total_assets': 0, 'total_records': 0, 'cache_hit_rate': 0, 'db_size_mb': 0}

    def get_last_timestamp(self, ticker: str) -> Optional[datetime]:
        """Get the latest timestamp for an asset"""
        if self.backend == 'postgresql':
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text('''
                        SELECT MAX(timestamp) FROM price_data WHERE ticker = :ticker
                    '''), {"ticker": ticker})
                    max_ts = result.scalar()
                    return pd.to_datetime(max_ts) if max_ts else None
            except Exception as e:
                logger.error(f"Error getting last timestamp for {ticker}: {e}")
                return None
        else:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    SELECT MAX(timestamp) FROM price_data WHERE ticker = ?
                ''', (ticker,))
                result = cursor.fetchone()
                if result and result[0]:
                    return pd.to_datetime(result[0])
                return None
            except sqlite3.Error as e:
                logger.error(f"Error getting last timestamp for {ticker}: {e}")
                return None

    def update_from_source(self, ticker: str, asset_type: str, source_data: pd.DataFrame):
        """
        Incremental update - only adds new data that's newer than last timestamp

        Args:
            ticker: Asset ticker
            asset_type: Type (stock, crypto, commodity)
            source_data: New data from external source (DataFrame with timestamp index)
        """
        # Get last timestamp in database
        last_ts = self.get_last_timestamp(ticker)

        if last_ts is not None:
            # Only insert new data
            new_data = source_data[source_data.index > last_ts]
            if new_data.empty:
                print(f"  {ticker}: Already up to date")
                return
            print(f"  {ticker}: Adding {len(new_data)} new records (after {last_ts.strftime('%Y-%m-%d')})")
        else:
            # No data yet, insert everything
            new_data = source_data
            print(f"  {ticker}: First load, adding {len(new_data)} records")

        # Use existing save_data method
        self.save_data(ticker, new_data, asset_type)

    def close(self):
        """Close database connections"""
        if self.backend == 'postgresql':
            if hasattr(self, 'engine'):
                self.engine.dispose()
        else:
            if hasattr(self, '_local') and hasattr(self._local, 'conn'):
                if self._local.conn:
                    self._local.conn.close()

        logger.info("Data Manager closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
