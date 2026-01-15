#!/usr/bin/env python3
"""
Render Cron Job - Daily Market Predictions
Runs once per day after market close to generate predictions
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def log(message: str):
    """Print with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


def run_command(cmd: list, description: str):
    """Run a command and handle errors"""
    log(f"▶️  {description}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        if result.stdout:
            print(result.stdout)

        log(f"✅ {description} - SUCCESS")
        return True

    except subprocess.CalledProcessError as e:
        log(f"❌ {description} - FAILED")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def main():
    """Main cron job execution"""
    print("\n" + "="*70)
    print("🔮 NEUROVEST DAILY PREDICTIONS - CRON JOB")
    print("="*70)
    log("Starting daily prediction pipeline")
    print("="*70 + "\n")

    # Step 1: Update SPY data (market indicator)
    log("STEP 1/3: Updating SPY market data")
    if not run_command(
        ["python3", "download_spy_data.py"],
        "Download latest SPY data"
    ):
        log("⚠️  SPY data update failed, continuing anyway...")

    # Step 2: Update all asset data
    log("\nSTEP 2/3: Updating all asset data")
    if not run_command(
        ["python3", "update_data.py", "update"],
        "Update all market data"
    ):
        log("⚠️  Data update had errors, continuing anyway...")

    # Step 3: Generate predictions for ALL assets
    log("\nSTEP 3/3: Generating predictions for all 40 assets")

    # Use predict_all_assets to generate predictions for every asset
    all_assets_script = Path("predict_all_assets.py")
    if all_assets_script.exists():
        if not run_command(
            ["python3", "predict_all_assets.py"],
            "Generate predictions for all assets"
        ):
            log("❌ All-assets prediction failed")
    else:
        log("⚠️  predict_all_assets.py not found, falling back to SPY only")
        if not run_command(
            ["python3", "predict_multi_asset_ensemble.py"],
            "Generate SPY prediction"
        ):
            log("❌ Prediction generation failed")

    # Summary
    print("\n" + "="*70)
    log("✅ DAILY PREDICTION PIPELINE COMPLETE")
    print("="*70)
    log(f"Next run: Tomorrow at {os.getenv('CRON_SCHEDULE', '16:30 EST')}")
    print()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
