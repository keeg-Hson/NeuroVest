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
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

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
                # Fetch 3 years of historical data for ML training
                callback = create_yfinance_callback(ticker, period='3y')
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
            self.dm.register_asset(ticker, 'crypto', 'daily')

            try:
                # Use Coinbase instead of Binance (Binance blocks Railway's region)
                # Fetch 3 years of daily data for ML training (matches stock data)
                callback = create_ccxt_callback(
                    symbol, 'coinbase', '1d', limit=1095  # 3 years of daily data
                )
                self.scheduler.register_update_callback(
                    ticker, callback, interval_minutes=60  # Update daily, not every 15min
                )
                print(f"  ✓ {ticker} (CCXT, 60min)")
            except Exception as e:
                print(f"  ⚠️  {ticker} - Using fallback: {e}")
                callback = create_fallback_callback(ticker, base_price=50000)
                self.scheduler.register_update_callback(
                    ticker, callback, interval_minutes=15
                )

        print("\n✅ Worker setup complete!\n")

    def train_models(self):
        """Automated model training - runs weekly"""
        print(f"\n{'='*70}")
        print(f"🤖 AUTOMATED MODEL TRAINING STARTED")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        try:
            # Check if training script exists
            train_script = Path("train_multi_asset.py")
            if train_script.exists():
                result = subprocess.run(
                    [sys.executable, "train_multi_asset.py"],
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                print(result.stdout)
                if result.returncode == 0:
                    print("✅ Model training completed successfully\n")
                else:
                    print(f"⚠️  Model training had errors:\n{result.stderr}\n")
            else:
                print("⚠️  train_multi_asset.py not found, skipping training\n")
        except subprocess.TimeoutExpired:
            print("⚠️  Model training timed out after 1 hour\n")
        except Exception as e:
            print(f"⚠️  Model training failed: {e}\n")

    def generate_predictions(self):
        """Automated prediction generation - runs daily"""
        print(f"\n{'='*70}")
        print(f"🔮 AUTOMATED PREDICTION GENERATION STARTED")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        try:
            # Check if prediction script exists
            pred_script = Path("predict_multi_asset_ensemble.py")
            if pred_script.exists():
                result = subprocess.run(
                    [sys.executable, "predict_multi_asset_ensemble.py"],
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minute timeout
                )
                print(result.stdout)
                if result.returncode == 0:
                    print("✅ Prediction generation completed successfully\n")
                else:
                    print(f"⚠️  Prediction generation had errors:\n{result.stderr}\n")
            else:
                print("⚠️  predict_multi_asset_ensemble.py not found, skipping predictions\n")
        except subprocess.TimeoutExpired:
            print("⚠️  Prediction generation timed out after 30 minutes\n")
        except Exception as e:
            print(f"⚠️  Prediction generation failed: {e}\n")

    def setup_ml_automation(self):
        """Set up automated ML pipeline schedules"""
        self.ml_scheduler = BackgroundScheduler()

        # Weekly model training - Sundays at 2 AM
        self.ml_scheduler.add_job(
            self.train_models,
            CronTrigger(day_of_week='sun', hour=2, minute=0),
            id='weekly_training',
            name='Weekly Model Training'
        )

        # Daily predictions - Every day at 4:30 PM EST (after market close)
        self.ml_scheduler.add_job(
            self.generate_predictions,
            CronTrigger(hour=16, minute=30),
            id='daily_predictions',
            name='Daily Prediction Generation'
        )

        self.ml_scheduler.start()

        print("📅 ML Automation Schedule:")
        print("   • Model Training: Every Sunday at 2:00 AM")
        print("   • Predictions: Every day at 4:30 PM EST")
        print()

    def run(self):
        """Main worker loop"""
        self.setup()
        self.running = True

        # Set up ML automation
        self.setup_ml_automation()

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
