#!/usr/bin/env python3
"""
Data Diagnostics Script
Checks for common data issues that cause prediction/backtest failures
"""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("DATA DIAGNOSTICS")
print("=" * 80)

# Check SPY.csv
print("\n📊 Checking data/SPY.csv...")
spy_path = Path("data/SPY.csv")

if not spy_path.exists():
    print("   ❌ File does not exist!")
else:
    print(f"   ✓ File exists ({spy_path.stat().st_size:,} bytes)")

    try:
        df = pd.read_csv(spy_path, low_memory=False)
        print(f"   ✓ Loaded {len(df):,} rows")

        if len(df) == 0:
            print("   ❌ File is empty!")
        else:
            print(f"   ✓ Columns: {list(df.columns)}")

            # Check Date column
            if 'Date' not in df.columns:
                print("   ❌ No 'Date' column found!")
            else:
                print(f"\n   📅 Date column analysis:")
                print(f"      First date: {df['Date'].iloc[0]}")
                print(f"      Last date: {df['Date'].iloc[-1]}")

                # Try to parse dates
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                valid_dates = df['Date'].notna().sum()
                invalid_dates = df['Date'].isna().sum()

                if invalid_dates > 0:
                    print(f"      ❌ Invalid dates: {invalid_dates:,}")
                    print(f"      Rows with invalid dates:")
                    bad_rows = df[df['Date'].isna()]
                    print(bad_rows.head())
                else:
                    print(f"      ✓ All {valid_dates:,} dates are valid")
                    print(f"      Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    except Exception as e:
        print(f"   ❌ Error reading file: {e}")

# Check prediction files
print("\n📊 Checking prediction files...")
logs_dir = Path("logs")

if not logs_dir.exists():
    print("   ⚠️  logs/ directory does not exist")
else:
    pred_files = [
        "labeled_predictions.csv",
        "daily_predictions.csv",
        "ensemble_analysis.csv"
    ]

    for filename in pred_files:
        filepath = logs_dir / filename
        if not filepath.exists():
            print(f"   ⚠️  {filename} does not exist")
        else:
            try:
                df = pd.read_csv(filepath)
                print(f"   ✓ {filename}: {len(df):,} rows")

                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    invalid = df['Date'].isna().sum()
                    if invalid > 0:
                        print(f"      ❌ {invalid:,} invalid dates!")
                        print(f"      Rows with invalid dates:")
                        print(df[df['Date'].isna()].head())

            except Exception as e:
                print(f"   ❌ Error reading {filename}: {e}")

# Check model files
print("\n📊 Checking model files...")
models_dir = Path("models")

if not models_dir.exists():
    print("   ⚠️  models/ directory does not exist")
else:
    model_files = [
        "xgboost_multi_asset.pkl",
        "lightgbm_multi_asset.pkl",
        "catboost_multi_asset.pkl",
        "multi_asset_features.txt"
    ]

    for filename in model_files:
        filepath = models_dir / filename
        if not filepath.exists():
            print(f"   ⚠️  {filename} does not exist")
        else:
            size = filepath.stat().st_size
            print(f"   ✓ {filename}: {size:,} bytes")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)

print("\n💡 Recommendations:")
print("   1. If SPY.csv is empty or has invalid dates:")
print("      → Run: python3 update_spy_data.py")
print("   2. If model files are missing:")
print("      → Run: python3 train_multi_asset.py")
print("   3. If prediction files have invalid dates:")
print("      → Delete them and re-run: python3 predict_multi_asset_ensemble.py")
print("   4. If prediction files don't exist:")
print("      → Run: python3 predict_multi_asset_ensemble.py")
