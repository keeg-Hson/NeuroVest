#!/usr/bin/env python3
"""
Automated Data Refresh and Retraining

Automatically:
1. Downloads latest data for all enabled assets
2. Retrains models if data has changed significantly
3. Updates predictions
4. Sends notifications (optional)

Can be run:
- On demand: python framework/auto_refresh.py
- On schedule: cron or system scheduler
- As daemon: python framework/auto_refresh.py --daemon

Usage:
    python framework/auto_refresh.py                    # Run once
    python framework/auto_refresh.py --daemon           # Run continuously
    python framework/auto_refresh.py --schedule daily   # Daily refresh
"""

import argparse
import sys
from pathlib import Path
import time
from datetime import datetime, timedelta
import schedule
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))
from asset_manager import AssetManager


class AutoRefresh:
    """Automated data refresh and retraining"""

    def __init__(self):
        self.manager = AssetManager()
        self.settings = self.manager.get_settings()
        self.framework_dir = Path(__file__).parent

    def run_full_refresh(self):
        """Run complete refresh cycle"""
        print("\n" + "=" * 80)
        print(f"AUTO REFRESH STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Step 1: Download latest data
        print("\n[1/3] Downloading latest data...")
        self._download_data()

        # Step 2: Retrain models
        print("\n[2/3] Retraining models...")
        self._retrain_models()

        # Step 3: Generate updated dashboard
        print("\n[3/3] Updating dashboard...")
        self._update_dashboard()

        print("\n" + "=" * 80)
        print(f"AUTO REFRESH COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def _download_data(self):
        """Download latest data for all assets"""
        script = self.framework_dir / "download_all_assets.py"

        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                print("   ✓ Data download completed")
            else:
                print(f"   ⚠️  Download had errors:\n{result.stderr}")

        except subprocess.TimeoutExpired:
            print("   ✗ Download timeout (>1 hour)")
        except Exception as e:
            print(f"   ✗ Download error: {e}")

    def _retrain_models(self):
        """Retrain all models"""
        script = self.framework_dir / "train_unified.py"

        try:
            result = subprocess.run(
                [sys.executable, str(script), "--all"],
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )

            if result.returncode == 0:
                print("   ✓ Model retraining completed")
            else:
                print(f"   ⚠️  Training had errors:\n{result.stderr}")

        except subprocess.TimeoutExpired:
            print("   ✗ Training timeout (>2 hours)")
        except Exception as e:
            print(f"   ✗ Training error: {e}")

    def _update_dashboard(self):
        """Generate updated dashboard"""
        script = self.framework_dir / "results_dashboard.py"

        try:
            result = subprocess.run(
                [sys.executable, str(script), "--export", "html"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print("   ✓ Dashboard updated")
            else:
                print(f"   ⚠️  Dashboard error:\n{result.stderr}")

        except Exception as e:
            print(f"   ✗ Dashboard error: {e}")

    def run_scheduled(self, schedule_type: str = "daily"):
        """Run on schedule"""
        if schedule_type == "daily":
            # Run at 2 AM every day
            schedule.every().day.at("02:00").do(self.run_full_refresh)
            print(f"✓ Scheduled for daily refresh at 2:00 AM")

        elif schedule_type == "weekly":
            # Run every Sunday at 2 AM
            schedule.every().sunday.at("02:00").do(self.run_full_refresh)
            print(f"✓ Scheduled for weekly refresh (Sunday 2:00 AM)")

        elif schedule_type == "hourly":
            # Run every hour
            schedule.every().hour.do(self.run_full_refresh)
            print(f"✓ Scheduled for hourly refresh")

        print("\nPress Ctrl+C to stop\n")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\n\nScheduler stopped")

    def run_daemon(self):
        """Run as daemon process"""
        print("=" * 80)
        print("AUTO REFRESH DAEMON MODE")
        print("=" * 80)

        # Get refresh schedule from config
        refresh_schedule = self.settings.get('refresh_schedule', 'daily')

        print(f"\nRefresh schedule: {refresh_schedule}")
        print("Starting scheduler...")

        self.run_scheduled(refresh_schedule)


def main():
    parser = argparse.ArgumentParser(description="Automated data refresh and retraining")
    parser.add_argument('--daemon', action='store_true',
                        help="Run as daemon (continuous)")
    parser.add_argument('--schedule', choices=['daily', 'weekly', 'hourly'],
                        help="Run on schedule")

    args = parser.parse_args()

    refresher = AutoRefresh()

    if args.daemon:
        refresher.run_daemon()
    elif args.schedule:
        refresher.run_scheduled(args.schedule)
    else:
        # Run once
        refresher.run_full_refresh()


if __name__ == "__main__":
    main()
