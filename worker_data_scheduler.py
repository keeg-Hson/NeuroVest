#!/usr/bin/env python3
"""
Production Worker - Full Automation Pipeline
- Continuous data collection
- Automated model training (weekly)
- Automated predictions (daily)
- Runs 24/7
"""

import os
import sys
import time
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Timezone for scheduling (US Eastern)
EST = ZoneInfo('America/New_York')

from core.data_manager_postgres import DataManager
from core.scheduler import (
    DataScheduler,
    create_yfinance_callback,
    create_ccxt_callback,
    create_fallback_callback
)

# Asset configuration source
from framework.asset_manager import AssetManager


def create_paginated_ccxt_callback(symbol: str, exchange_name: str = 'binance',
                                   timeframe: str = '1d'):
    """
    CCXT callback with full-history pagination.

    Unlike the base create_ccxt_callback (single fetch, exchange-capped at
    300-1000 candles), this loops until all available history is collected.
    Uses Binance by default (1000 candles/request, data from 2017+).
    """
    def fetch_data():
        try:
            import ccxt
            import pandas as pd

            exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})

            # Start from 2017-01-01 (Binance launch date)
            since = exchange.parse8601('2017-01-01T00:00:00Z')

            all_ohlcv = []
            while True:
                batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                if not batch:
                    break
                all_ohlcv.extend(batch)
                last_ts = batch[-1][0]
                since = last_ts + 86_400_000   # advance 1 day in ms
                if since > exchange.milliseconds():
                    break
                if len(batch) < 1000:
                    break
                time.sleep(0.3)

            if not all_ohlcv:
                return None

            df = pd.DataFrame(
                all_ohlcv,
                columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop_duplicates('timestamp').set_index('timestamp')
            df['Adj_Close'] = df['Close']
            return df

        except Exception as e:
            print(f"  Error fetching {symbol} from {exchange_name}: {e}")
            return None

    return fetch_data


class WorkerScheduler:
    """Production-ready worker with full ML pipeline automation"""

    def __init__(self):
        self.running = False
        self.dm = None
        self.scheduler = None
        self.ml_scheduler = None  # APScheduler for training/predictions

    def setup(self):
        """Initialize data manager and scheduler"""
        print("\n" + "="*70)
        print("🚀 NEUROVEST DATA WORKER - STARTING")
        print("="*70)
        print(f"  Platform: Railway")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Python: {sys.version.split()[0]}")
        print("="*70 + "\n")

        # Initialize data manager (auto-detects DATABASE_URL for PostgreSQL)
        self.dm = DataManager()

        # Initialize scheduler
        self.scheduler = DataScheduler(self.dm)

        # Load all configured assets from config/assets.yaml (currently 63)
        asset_manager = AssetManager()
        all_assets = asset_manager.get_all_assets()

        # Split into equity/commodity (yfinance) vs crypto (ccxt/Binance)
        non_crypto = [a for a in all_assets if a.asset_type != 'crypto']
        crypto_assets = [a for a in all_assets if a.asset_type == 'crypto']

        print(f"📊 Registering {len(non_crypto)} stock/bond/commodity assets...")
        for asset in non_crypto:
            ticker = asset.ticker
            self.dm.register_asset(ticker, asset.asset_type, 'daily')

            try:
                # period='max' → full history from inception on first load;
                # update_from_source() keeps it incremental thereafter.
                callback = create_yfinance_callback(ticker, period='max')
                self.scheduler.register_update_callback(
                    ticker, callback, interval_minutes=60
                )
                print(f"  ✓ {ticker} (yfinance max history, 60min)")
            except Exception as e:
                print(f"  ⚠️  {ticker} - Using fallback: {e}")
                callback = create_fallback_callback(ticker)
                self.scheduler.register_update_callback(
                    ticker, callback, interval_minutes=60
                )

        # Polygon rebranded MATIC → POL in Sept 2024.
        # Binance still lists the asset; map config ticker to the correct Binance symbol.
        BINANCE_SYMBOL_MAP = {
            'MATIC/USDT': 'POL/USDT',   # Binance migrated MATIC to POL Sept 2024
        }

        print(f"\n₿ Registering {len(crypto_assets)} crypto assets via Binance (paginated)...")
        for asset in crypto_assets:
            symbol = BINANCE_SYMBOL_MAP.get(asset.ticker, asset.ticker)
            ticker = asset.ticker.replace('/', '_')
            self.dm.register_asset(ticker, 'crypto', 'daily')

            try:
                # Paginated fetch — overcomes Binance/exchange per-request candle caps.
                # Fetches all history from 2017-01-01 forward.
                callback = create_paginated_ccxt_callback(symbol, 'binance', '1d')
                self.scheduler.register_update_callback(
                    ticker, callback, interval_minutes=60
                )
                print(f"  ✓ {ticker} (binance paginated, 60min)")
            except Exception as e:
                print(f"  ⚠️  {ticker} - Using fallback: {e}")
                callback = create_fallback_callback(ticker, base_price=50000)
                self.scheduler.register_update_callback(
                    ticker, callback, interval_minutes=60
                )

        print("\n✅ Worker setup complete!\n")

    def train_models(self):
        """Automated model training - runs weekly"""
        print(f"\n{'='*70}")
        print(f"🤖 AUTOMATED MODEL TRAINING STARTED")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n", flush=True)

        try:
            # Check if training script exists
            train_script = Path("train_unified.py")
            if train_script.exists():
                print(f"🏃 Running {train_script} --model ensemble --output-prefix multi_asset...\n", flush=True)
                result = subprocess.run(
                    [sys.executable, "train_unified.py", "--model", "ensemble", "--output-prefix", "multi_asset"],
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                # Always print stdout
                if result.stdout:
                    print(result.stdout)
                # Always print stderr if present
                if result.stderr:
                    print(f"STDERR:\n{result.stderr}")

                if result.returncode == 0:
                    # List generated model files
                    models_dir = Path("models")
                    if models_dir.exists():
                        model_files = list(models_dir.glob("multi_asset*.pkl"))
                        print(f"📁 Generated {len(model_files)} model files:")
                        for mf in model_files:
                            print(f"   - {mf.name}")
                    print("✅ Model training completed successfully\n")
                else:
                    print(f"⚠️  Model training exit code: {result.returncode}\n")
            else:
                print("⚠️  train_unified.py not found, skipping training\n")
        except subprocess.TimeoutExpired:
            print("⚠️  Model training timed out after 1 hour\n")
        except Exception as e:
            import traceback
            print(f"⚠️  Model training failed: {e}")
            traceback.print_exc()
            print()

    def generate_predictions(self):
        """Automated prediction generation - runs daily"""
        print(f"\n{'='*70}")
        print(f"🔮 AUTOMATED PREDICTION GENERATION STARTED")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n", flush=True)

        # Check if models exist before running predictions
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pkl"))
            print(f"📁 Models directory: {len(model_files)} .pkl files found")
            for mf in model_files[:5]:
                print(f"   - {mf.name}")
            if len(model_files) > 5:
                print(f"   ... and {len(model_files) - 5} more")
        else:
            print("⚠️  Models directory does not exist!")
            print("   Training required before predictions can run.\n")
            return

        try:
            # Check if prediction script exists (use predict_all_assets.py for all 40 assets)
            pred_script = Path("predict_all_assets.py")
            if pred_script.exists():
                print(f"🏃 Running {pred_script}...\n", flush=True)
                result = subprocess.run(
                    [sys.executable, "predict_all_assets.py"],
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minute timeout
                )
                # Always print stdout
                if result.stdout:
                    print(result.stdout)
                # Always print stderr if present
                if result.stderr:
                    print(f"STDERR:\n{result.stderr}")

                if result.returncode == 0:
                    print("✅ Prediction generation completed successfully\n")
                else:
                    print(f"⚠️  Prediction generation exit code: {result.returncode}\n")
            else:
                # Fallback to SPY-only predictions
                fallback_script = Path("predict.py")
                if fallback_script.exists():
                    print("⚠️  predict_all_assets.py not found, using SPY-only fallback\n")
                    result = subprocess.run(
                        [sys.executable, "predict.py"],
                        capture_output=True,
                        text=True,
                        timeout=1800
                    )
                    if result.stdout:
                        print(result.stdout)
                    if result.stderr:
                        print(f"STDERR:\n{result.stderr}")
                else:
                    print("⚠️  No prediction scripts found, skipping predictions\n")
        except subprocess.TimeoutExpired:
            print("⚠️  Prediction generation timed out after 30 minutes\n")
        except Exception as e:
            import traceback
            print(f"⚠️  Prediction generation failed: {e}")
            traceback.print_exc()
            print()

    def _heartbeat(self):
        """Periodic heartbeat to prove scheduler is alive"""
        now_est = datetime.now(EST).strftime('%Y-%m-%d %H:%M:%S %Z')
        jobs = self.ml_scheduler.get_jobs() if self.ml_scheduler else []
        print(f"\n💓 HEARTBEAT [{now_est}] - Scheduler alive, {len(jobs)} jobs registered", flush=True)
        for job in jobs:
            if job.id != 'heartbeat' and job.next_run_time:
                print(f"   Next {job.name}: {job.next_run_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    def _check_staleness(self):
        """Check for stale predictions/models and regenerate if needed"""
        now_est = datetime.now(EST)
        print(f"\n🔍 STALENESS CHECK [{now_est.strftime('%Y-%m-%d %H:%M:%S %Z')}]", flush=True)

        try:
            # Check prediction staleness
            preds_df = self.dm.get_latest_predictions(limit=1)
            if preds_df.empty:
                print("   ⚠️  No predictions found - generating now...")
                self.generate_predictions()
            else:
                # Check if latest prediction is older than 24 hours
                latest_date = preds_df.iloc[0].get('prediction_date')
                if latest_date:
                    from datetime import date
                    if isinstance(latest_date, str):
                        latest_date = datetime.strptime(latest_date[:10], '%Y-%m-%d').date()
                    elif hasattr(latest_date, 'date'):
                        latest_date = latest_date.date()

                    days_old = (date.today() - latest_date).days
                    if days_old >= 1:
                        print(f"   ⚠️  Predictions are {days_old} day(s) old - regenerating...")
                        self.generate_predictions()
                    else:
                        print(f"   ✓ Predictions are current (from {latest_date})")

            # Check model staleness (weekly check)
            models_dir = Path("models")
            if models_dir.exists():
                model_files = list(models_dir.glob("multi_asset_*.pkl"))
                if model_files:
                    oldest_mtime = min(f.stat().st_mtime for f in model_files)
                    age_days = (datetime.now().timestamp() - oldest_mtime) / 86400
                    if age_days > 7:
                        print(f"   ⚠️  Models are {age_days:.1f} days old - retraining...")
                        self.train_models()
                    else:
                        print(f"   ✓ Models are {age_days:.1f} days old (OK)")
                else:
                    print("   ⚠️  No models found - training now...")
                    self.train_models()

        except Exception as e:
            print(f"   ✗ Staleness check failed: {e}")

    def _job_listener(self, event):
        """Log job execution events for visibility"""
        job_id = event.job_id
        now = datetime.now(EST).strftime('%Y-%m-%d %H:%M:%S %Z')

        if event.exception:
            print(f"\n{'!'*70}")
            print(f"❌ JOB FAILED: {job_id} at {now}")
            print(f"   Error: {event.exception}")
            print(f"{'!'*70}\n", flush=True)
        else:
            print(f"\n{'='*70}")
            print(f"✅ JOB COMPLETED: {job_id} at {now}")
            print(f"{'='*70}\n", flush=True)

    def _job_missed_listener(self, event):
        """Log missed jobs"""
        print(f"\n⚠️  JOB MISSED: {event.job_id} - scheduled run was missed", flush=True)

    def setup_ml_automation(self):
        """Set up automated ML pipeline schedules"""
        self.ml_scheduler = BackgroundScheduler(timezone=EST)

        # Add job listeners for visibility
        self.ml_scheduler.add_listener(self._job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
        self.ml_scheduler.add_listener(self._job_missed_listener, EVENT_JOB_MISSED)

        # Weekly model training - Sundays at 2 AM EST
        self.ml_scheduler.add_job(
            self.train_models,
            CronTrigger(day_of_week='sun', hour=2, minute=0, timezone=EST),
            id='weekly_training',
            name='Weekly Model Training',
            misfire_grace_time=3600  # Allow 1 hour grace period
        )

        # Daily predictions - Every day at 4:30 PM EST (after market close)
        self.ml_scheduler.add_job(
            self.generate_predictions,
            CronTrigger(hour=16, minute=30, timezone=EST),
            id='daily_predictions',
            name='Daily Prediction Generation',
            misfire_grace_time=3600
        )

        # Heartbeat - every 30 minutes to prove scheduler is alive
        self.ml_scheduler.add_job(
            self._heartbeat,
            IntervalTrigger(minutes=30),
            id='heartbeat',
            name='Scheduler Heartbeat'
        )

        # Staleness check - every 6 hours to catch up if jobs missed
        self.ml_scheduler.add_job(
            self._check_staleness,
            IntervalTrigger(hours=6),
            id='staleness_check',
            name='Staleness Check'
        )

        self.ml_scheduler.start()

        now_est = datetime.now(EST).strftime('%Y-%m-%d %H:%M:%S %Z')
        print(f"\n📅 ML Automation Schedule (current time: {now_est}):")
        print("   • Model Training: Every Sunday at 2:00 AM EST")
        print("   • Predictions: Every day at 4:30 PM EST")
        print("   • Staleness Check: Every 6 hours (catches missed jobs)")
        print("   • Heartbeat: Every 30 minutes")

        # Show next scheduled runs
        jobs = self.ml_scheduler.get_jobs()
        print("\n   Next scheduled runs:")
        for job in jobs:
            if job.next_run_time:
                print(f"   • {job.name}: {job.next_run_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print()

    def run(self):
        """Main worker loop"""
        self.setup()
        self.running = True

        # Set up ML automation
        self.setup_ml_automation()

        # Diagnostic: Show model and database state
        print(f"\n{'='*70}")
        print("📊 STARTUP DIAGNOSTICS")
        print(f"{'='*70}")

        # Check models directory
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pkl"))
            print(f"📁 Models directory: {len(model_files)} .pkl files")
            for mf in sorted(model_files)[:10]:
                import os
                mtime = datetime.fromtimestamp(os.path.getmtime(mf))
                age_hours = (datetime.now() - mtime).total_seconds() / 3600
                print(f"   - {mf.name} (modified {age_hours:.1f}h ago)")
        else:
            print("⚠️  Models directory missing - training required!")

        # Check database state
        try:
            models_df = self.dm.get_latest_models()
            if not models_df.empty:
                print(f"\n📊 Database model_metadata: {len(models_df)} models")
                for _, row in models_df.iterrows():
                    trained_at = row.get('trained_at', 'unknown')
                    print(f"   - {row.get('model_type', 'unknown')} trained at {trained_at}")
            else:
                print("\n⚠️  Database model_metadata is EMPTY")

            preds_df = self.dm.get_latest_predictions(limit=5)
            if not preds_df.empty:
                print(f"\n📊 Database predictions: latest entries")
                for _, row in preds_df.head(3).iterrows():
                    print(f"   - {row.get('ticker', '?')}: {row.get('prediction_label', '?')} ({row.get('prediction_date', '?')})")
            else:
                print("\n⚠️  Database predictions is EMPTY")
        except Exception as e:
            print(f"\n⚠️  Could not query database: {e}")

        print(f"{'='*70}\n")

        # Check if data is actually fresh (not just metadata)
        print("🔍 Checking data freshness...", flush=True)
        try:
            # Get the actual latest date in price_data
            with self.dm.engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text('''
                    SELECT MAX(date) as latest_date FROM price_data
                '''))
                row = result.fetchone()
                if row and row[0]:
                    from datetime import date
                    latest_data_date = row[0]
                    if hasattr(latest_data_date, 'date'):
                        latest_data_date = latest_data_date.date()
                    elif isinstance(latest_data_date, str):
                        latest_data_date = datetime.strptime(latest_data_date[:10], '%Y-%m-%d').date()

                    days_stale = (date.today() - latest_data_date).days
                    print(f"   Latest data: {latest_data_date} ({days_stale} days old)")

                    if days_stale > 1:
                        print(f"   ⚠️  Data is STALE - forcing full refresh...")
                        # Force update by clearing last_update timestamps
                        conn.execute(text('UPDATE asset_metadata SET last_update = NULL'))
                        conn.commit()
                else:
                    print("   ⚠️  No data in database - will bootstrap")
        except Exception as e:
            print(f"   ⚠️  Could not check data freshness: {e}")

        # Run initial update
        print("\n🔄 Running initial data update...", flush=True)
        self.scheduler.run_once()

        # Check if models exist - if not, train first
        models_dir = Path("models")
        model_files = list(models_dir.glob("multi_asset_*.pkl")) if models_dir.exists() else []

        if len(model_files) < 3:
            print("⚠️  Missing models detected - running training first...")
            print("   (This is required before predictions can run)\n")
            self.train_models()

        # Generate predictions after fresh data
        print("🔮 Running initial predictions...", flush=True)
        self.generate_predictions()

        # Start scheduler (checks every 60 minutes)
        update_interval = int(os.getenv('UPDATE_INTERVAL', '60'))
        self.scheduler.start(update_interval_minutes=update_interval)

        print(f"\n{'='*70}")
        print(f"✅ WORKER RUNNING - Updates every {update_interval} minutes")
        print(f"{'='*70}")
        print("  Press Ctrl+C to stop\n")

        # Keep running and print status
        status_counter = 0
        try:
            while self.running:
                time.sleep(60)  # Check every minute
                status_counter += 1

                now_est = datetime.now(EST)
                stats = self.dm.get_stats()

                # Every 10 minutes, print full status with scheduler info
                if status_counter % 10 == 0:
                    print(f"\n[{now_est.strftime('%Y-%m-%d %H:%M:%S %Z')}] STATUS UPDATE")
                    print(f"  Data: {stats['total_assets']} assets, {stats['total_records']:,} records")
                    if self.ml_scheduler:
                        jobs = self.ml_scheduler.get_jobs()
                        for job in jobs:
                            if job.id != 'heartbeat' and job.next_run_time:
                                print(f"  Next {job.id}: {job.next_run_time.strftime('%Y-%m-%d %H:%M %Z')}")
                    print("", flush=True)
                else:
                    # Brief status line
                    print(f"\r[{now_est.strftime('%H:%M:%S')}] "
                          f"Assets: {stats['total_assets']:>3} | "
                          f"Records: {stats['total_records']:>8,} | "
                          f"Scheduler: {'RUNNING' if self.ml_scheduler and self.ml_scheduler.running else 'STOPPED'}",
                          end='', flush=True)

        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown"""
        print("\n\n" + "="*70)
        print("🛑 SHUTTING DOWN WORKER")
        print("="*70)

        self.running = False

        if self.ml_scheduler:
            self.ml_scheduler.shutdown()
            print("  ✓ ML automation stopped")

        if self.scheduler:
            self.scheduler.stop()
            print("  ✓ Data scheduler stopped")

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
