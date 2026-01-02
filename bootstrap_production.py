#!/usr/bin/env python3
"""
Production Bootstrap - One-time setup script

This script:
1. Downloads ALL historical data for all assets (3+ years)
2. Trains all models
3. Generates initial predictions
4. Prepares the system for production use

Run this ONCE when deploying, then let the worker handle incremental updates.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*70}")
    print(f"📌 {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode == 0:
            print(f"\n✅ {description} - SUCCESS")
            return True
        else:
            print(f"\n⚠️  {description} - FAILED (exit code {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n⚠️  {description} - TIMEOUT (> 1 hour)")
        return False
    except Exception as e:
        print(f"\n⚠️  {description} - ERROR: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("🚀 NEUROVEST PRODUCTION BOOTSTRAP")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    results = {}

    # Step 1: Download all data
    print("\n\n")
    print("STEP 1: DOWNLOADING HISTORICAL DATA")
    print("This will populate the database with 3+ years of data for all assets")
    print("Expected time: 5-10 minutes")

    # Use the worker's data download logic
    results['data_download'] = run_command(
        [sys.executable, "worker_data_scheduler.py"],
        "Download All Historical Data"
    )

    # Wait for data download to complete, then stop it
    # Actually, let's use a different approach - run the data download directly

    # Step 2: Train models
    if Path("train_multi_asset.py").exists():
        print("\n\n")
        print("STEP 2: TRAINING MODELS")
        print("This will train XGBoost, LightGBM, and CatBoost models for all assets")
        print("Expected time: 10-30 minutes")

        results['training'] = run_command(
            [sys.executable, "train_multi_asset.py"],
            "Train All Models"
        )
    else:
        print("\n⚠️  STEP 2 SKIPPED: train_multi_asset.py not found")
        results['training'] = False

    # Step 3: Generate predictions
    if Path("predict_multi_asset_ensemble.py").exists():
        print("\n\n")
        print("STEP 3: GENERATING PREDICTIONS")
        print("This will generate predictions for all assets")
        print("Expected time: 5-10 minutes")

        results['predictions'] = run_command(
            [sys.executable, "predict_multi_asset_ensemble.py"],
            "Generate Initial Predictions"
        )
    else:
        print("\n⚠️  STEP 3 SKIPPED: predict_multi_asset_ensemble.py not found")
        results['predictions'] = False

    # Summary
    print("\n\n" + "="*70)
    print("📊 BOOTSTRAP SUMMARY")
    print("="*70)

    for step, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {step:20s}: {status}")

    all_success = all(results.values())

    print("\n" + "="*70)
    if all_success:
        print("🎉 BOOTSTRAP COMPLETE - PRODUCTION READY!")
        print("="*70)
        print("\nYour system is now ready:")
        print("  • All historical data loaded")
        print("  • Models trained")
        print("  • Predictions generated")
        print("\nThe worker will now handle incremental updates automatically.")
        return 0
    else:
        print("⚠️  BOOTSTRAP INCOMPLETE - SOME STEPS FAILED")
        print("="*70)
        print("\nCheck the logs above for errors.")
        print("You may need to run individual steps manually.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
