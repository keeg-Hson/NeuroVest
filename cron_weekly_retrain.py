#!/usr/bin/env python3
"""
Render Cron Job - Weekly Model Retraining
Runs once per week to retrain ML models with latest data
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
            text=True,
            timeout=3600  # 1 hour timeout for training
        )

        if result.stdout:
            print(result.stdout)

        log(f"✅ {description} - SUCCESS")
        return True

    except subprocess.TimeoutExpired:
        log(f"⏱️  {description} - TIMEOUT (>1 hour)")
        return False
    except subprocess.CalledProcessError as e:
        log(f"❌ {description} - FAILED")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def main():
    """Main weekly retraining job"""
    print("\n" + "="*70)
    print("🧠 NEUROVEST WEEKLY MODEL RETRAINING - CRON JOB")
    print("="*70)
    log("Starting weekly model retraining")
    print("="*70 + "\n")

    # Step 1: Update all data first
    log("STEP 1/3: Ensuring data is current")
    run_command(
        ["python3", "update_data.py", "update"],
        "Update all market data"
    )

    # Step 2: Check which training script to use
    log("\nSTEP 2/3: Selecting training script")

    train_scripts = [
        ("train_multi_asset.py", "Multi-asset 3-class training"),
        ("framework/train_unified.py", "Unified training framework"),
        ("train_improved.py", "Improved training script"),
        ("train.py", "Standard training script")
    ]

    train_script = None
    for script_path, description in train_scripts:
        if Path(script_path).exists():
            train_script = (script_path, description)
            log(f"  ✓ Found: {script_path}")
            break

    if not train_script:
        log("❌ No training script found!")
        sys.exit(1)

    # Step 3: Train models
    log(f"\nSTEP 3/3: Training models with {train_script[1]}")

    success = run_command(
        ["python3", train_script[0]],
        f"Train models using {train_script[0]}"
    )

    if not success:
        log("❌ Model training failed - using existing models")

    # Step 4: Generate fresh predictions for all assets with new models
    log("\nSTEP 4/3 (BONUS): Generating predictions for all assets with retrained models")

    predict_script = Path("predict_all_assets.py")
    if predict_script.exists():
        run_command(
            ["python3", "predict_all_assets.py"],
            "Generate predictions for all 40 assets"
        )
    else:
        # Fallback to SPY-only predictions
        ensemble_script = Path("predict_multi_asset_ensemble.py")
        if ensemble_script.exists():
            log("  ⚠️  Using fallback SPY-only predictions")
            run_command(
                ["python3", "predict_multi_asset_ensemble.py"],
                "Generate SPY predictions (fallback)"
            )

    # Summary
    print("\n" + "="*70)
    log("✅ WEEKLY MODEL RETRAINING COMPLETE")
    print("="*70)
    log(f"Next run: Next {os.getenv('RETRAIN_DAY', 'Sunday')} at {os.getenv('RETRAIN_TIME', '02:00 EST')}")
    print()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
