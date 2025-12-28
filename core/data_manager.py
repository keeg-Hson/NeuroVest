"""
Data Manager V2 - PostgreSQL + SQLite support
Handles database storage with SQLAlchemy for both PostgreSQL and SQLite
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
import logging
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Index, UniqueConstraint
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy.exc import SQLAlchemyError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages market data with PostgreSQL/SQLite backend and intelligent caching

    Features:
    - PostgreSQL for production (Railway)
    - SQLite for local development
    - Automatic database detection via DATABASE_URL
    - Incremental updates (only fetch new data)
    - Multi-asset support (stocks, crypto, commodities)
    - Caching for performance
    """

    def __init__(self, db_url: Optional[str] = None):
        """Initialize data manager with database connection"""
        # Determine database URL
        if db_url is None:
            db_url = os.getenv('DATABASE_URL')

        if db_url:
            # PostgreSQL (production)
            # Fix Railway's postgres:// to postgresql://
            if db_url.startswith('postgres://'):
                db_url = db_url.replace('postgres://', 'postgresql://', 1)
            self.engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True  # Verify connections before using
            )
            self.db_type = 'postgresql'
            logger.info(f"Data Manager initialized: PostgreSQL")
        else:
            # SQLite (local development)
            db_path = os.getenv('DATABASE_PATH', 'data/market_data.db')
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(
                f'sqlite:///{db_path}',
                poolclass=NullPool,
                connect_args={'check_same_thread': False}
            )
            self.db_type = 'sqlite'
            logger.info(f"Data Manager initialized: SQLite ({db_path})")

        # Initialize database schema
        self._create_tables()

        # Cache for frequently accessed data
        self._cache = {}
        self._cache_timestamps = {}
        self.cache_ttl = 300  # 5 minutes

    def _create_tables(self):
        """Create database schema"""
        with self.engine.connect() as conn:
            try:
                # Main price data table
                conn.execute(text('''
                    CREATE TABLE IF NOT EXISTS price_data (
                        id SERIAL PRIMARY KEY,
                        ticker VARCHAR(50) NOT NULL,
                        asset_type VARCHAR(50) NOT NULL,
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
                        ticker VARCHAR(50) PRIMARY KEY,
                        asset_type VARCHAR(50) NOT NULL,
                        last_update TIMESTAMP,
                        last_timestamp TIMESTAMP,
                        total_records INTEGER DEFAULT 0,
                        update_frequency VARCHAR(50) DEFAULT 'daily',
                        enabled INTEGER DEFAULT 1
                    )
                '''))

                # Cache statistics
                conn.execute(text('''
                    CREATE TABLE IF NOT EXISTS cache_stats (
                        ticker VARCHAR(50) PRIMARY KEY,
                        hit_count INTEGER DEFAULT 0,
                        miss_count INTEGER DEFAULT 0,
                        last_access TIMESTAMP
                    )
                '''))

                conn.commit()
                logger.info("Database schema initialized")
            except SQLAlchemyError as e:
                logger.error(f"Error creating database schema: {e}")
                conn.rollback()
                raise

    def register_asset(self, ticker: str, asset_type: str,
                      update_frequency: str = 'daily'):
        """Register an asset for tracking"""
        with self.engine.connect() as conn:
            try:
                # Use INSERT ... ON CONFLICT for PostgreSQL compatibility
                if self.db_type == 'postgresql':
                    conn.execute(text('''
                        INSERT INTO asset_metadata (ticker, asset_type, update_frequency)
                        VALUES (:ticker, :asset_type, :freq)
                        ON CONFLICT (ticker) DO UPDATE
                        SET asset_type = :asset_type, update_frequency = :freq
                    '''), {'ticker': ticker, 'asset_type': asset_type, 'freq': update_frequency})
                else:
                    conn.execute(text('''
                        INSERT OR REPLACE INTO asset_metadata
                        (ticker, asset_type, update_frequency)
                        VALUES (:ticker, :asset_type, :freq)
                    '''), {'ticker': ticker, 'asset_type': asset_type, 'freq': update_frequency})

                conn.commit()
                logger.info(f"Registered {ticker} ({asset_type})")
            except SQLAlchemyError as e:
                logger.error(f"Error registering {ticker}: {e}")
                conn.rollback()
                raise

    def get_last_timestamp(self, ticker: str) -> Optional[datetime]:
        """Get the latest timestamp for an asset"""
        with self.engine.connect() as conn:
            try:
                result = conn.execute(text('''
                    SELECT MAX(timestamp) FROM price_data WHERE ticker = :ticker
                '''), {'ticker': ticker})

                row = result.fetchone()
                if row and row[0]:
                    return pd.to_datetime(row[0])
                return None
            except SQLAlchemyError as e:
                logger.error(f"Error getting last timestamp for {ticker}: {e}")
                return None

    def add_data(self, ticker: str, asset_type: str, df: pd.DataFrame):
        """Add market data for an asset"""
        if df is None or len(df) == 0:
            logger.warning(f"{ticker}: No data to add")
            return

        # Prepare data
        df = df.copy()
        df['ticker'] = ticker
        df['asset_type'] = asset_type

        # Ensure timestamp column
        if 'Date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date'])
        elif 'timestamp' not in df.columns:
            df['timestamp'] = df.index

        # Select required columns
        required_cols = ['ticker', 'asset_type', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        df_to_insert = df[required_cols].copy()

        # Add adjusted_close if exists
        if 'Adj Close' in df.columns:
            df_to_insert['adjusted_close'] = df['Adj Close']
        elif 'adjusted_close' in df.columns:
            pass  # Already there
        else:
            df_to_insert['adjusted_close'] = df_to_insert['close']

        # Convert timestamp to string for SQL
        df_to_insert['timestamp'] = df_to_insert['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Insert data
        with self.engine.connect() as conn:
            try:
                inserted = 0
                for _, row in df_to_insert.iterrows():
                    try:
                        if self.db_type == 'postgresql':
                            conn.execute(text('''
                                INSERT INTO price_data
                                (ticker, asset_type, timestamp, open, high, low, close, volume, adjusted_close)
                                VALUES (:ticker, :asset_type, :timestamp, :open, :high, :low, :close, :volume, :adj_close)
                                ON CONFLICT (ticker, timestamp) DO NOTHING
                            '''), {
                                'ticker': row['ticker'],
                                'asset_type': row['asset_type'],
                                'timestamp': row['timestamp'],
                                'open': float(row['open']) if pd.notna(row['open']) else None,
                                'high': float(row['high']) if pd.notna(row['high']) else None,
                                'low': float(row['low']) if pd.notna(row['low']) else None,
                                'close': float(row['close']) if pd.notna(row['close']) else None,
                                'volume': float(row['volume']) if pd.notna(row['volume']) else None,
                                'adj_close': float(row['adjusted_close']) if pd.notna(row['adjusted_close']) else None
                            })
                        else:
                            conn.execute(text('''
                                INSERT OR IGNORE INTO price_data
                                (ticker, asset_type, timestamp, open, high, low, close, volume, adjusted_close)
                                VALUES (:ticker, :asset_type, :timestamp, :open, :high, :low, :close, :volume, :adj_close)
                            '''), {
                                'ticker': row['ticker'],
                                'asset_type': row['asset_type'],
                                'timestamp': row['timestamp'],
                                'open': float(row['open']) if pd.notna(row['open']) else None,
                                'high': float(row['high']) if pd.notna(row['high']) else None,
                                'low': float(row['low']) if pd.notna(row['low']) else None,
                                'close': float(row['close']) if pd.notna(row['close']) else None,
                                'volume': float(row['volume']) if pd.notna(row['volume']) else None,
                                'adj_close': float(row['adjusted_close']) if pd.notna(row['adjusted_close']) else None
                            })
                        inserted += 1
                    except SQLAlchemyError:
                        pass  # Skip duplicates

                # Update metadata
                last_ts = df_to_insert['timestamp'].max()
                conn.execute(text('''
                    UPDATE asset_metadata
                    SET last_update = CURRENT_TIMESTAMP,
                        last_timestamp = :last_ts,
                        total_records = (SELECT COUNT(*) FROM price_data WHERE ticker = :ticker)
                    WHERE ticker = :ticker
                '''), {'ticker': ticker, 'last_ts': last_ts})

                conn.commit()
                logger.info(f"{ticker}: Added {inserted} records")

            except SQLAlchemyError as e:
                logger.error(f"Error adding data for {ticker}: {e}")
                conn.rollback()
                raise

    def get_data(self, ticker: str, start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 use_cache: bool = True) -> pd.DataFrame:
        """Retrieve data for an asset with caching"""
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
            WHERE ticker = :ticker
        '''
        params = {'ticker': ticker}

        if start_date:
            query += ' AND timestamp >= :start_date'
            params['start_date'] = start_date

        if end_date:
            query += ' AND timestamp <= :end_date'
            params['end_date'] = end_date

        query += ' ORDER BY timestamp ASC'

        # Execute query
        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=params)

        if df.empty:
            return pd.DataFrame()

        # Format dataframe
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']

        # Cache result
        self._cache[cache_key] = df.copy()
        self._cache_timestamps[cache_key] = time.time()

        return df

    def get_stats(self) -> Dict:
        """Get database statistics"""
        with self.engine.connect() as conn:
            try:
                # Asset counts
                result = conn.execute(text('SELECT COUNT(*) FROM asset_metadata WHERE enabled = 1'))
                total_assets = result.fetchone()[0]

                # Total records
                result = conn.execute(text('SELECT COUNT(*) FROM price_data'))
                total_records = result.fetchone()[0]

                return {
                    'total_assets': total_assets,
                    'total_records': total_records,
                    'db_type': self.db_type,
                    'cache_size': len(self._cache)
                }
            except SQLAlchemyError as e:
                logger.error(f"Error getting stats: {e}")
                return {'total_assets': 0, 'total_records': 0, 'db_type': self.db_type}

    def _update_cache_stats(self, ticker: str, hit: bool):
        """Update cache hit/miss statistics"""
        with self.engine.connect() as conn:
            try:
                if self.db_type == 'postgresql':
                    conn.execute(text('''
                        INSERT INTO cache_stats (ticker, hit_count, miss_count, last_access)
                        VALUES (:ticker, :hit, :miss, CURRENT_TIMESTAMP)
                        ON CONFLICT (ticker) DO UPDATE
                        SET hit_count = cache_stats.hit_count + :hit,
                            miss_count = cache_stats.miss_count + :miss,
                            last_access = CURRENT_TIMESTAMP
                    '''), {'ticker': ticker, 'hit': 1 if hit else 0, 'miss': 0 if hit else 1})
                else:
                    conn.execute(text('''
                        INSERT OR REPLACE INTO cache_stats
                        (ticker, hit_count, miss_count, last_access)
                        VALUES (
                            :ticker,
                            COALESCE((SELECT hit_count FROM cache_stats WHERE ticker = :ticker), 0) + :hit,
                            COALESCE((SELECT miss_count FROM cache_stats WHERE ticker = :ticker), 0) + :miss,
                            CURRENT_TIMESTAMP
                        )
                    '''), {'ticker': ticker, 'hit': 1 if hit else 0, 'miss': 0 if hit else 1})

                conn.commit()
            except SQLAlchemyError:
                pass  # Non-critical

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.engine.dispose()
