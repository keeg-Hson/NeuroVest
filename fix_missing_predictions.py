#!/usr/bin/env python3
"""
Generate predictions for assets that are missing them
"""
import sys
sys.path.insert(0, '.')

print("=" * 80)
print("GENERATING PREDICTIONS FOR MISSING ASSETS")
print("=" * 80)

from core.data_manager_postgres import DataManager
from sqlalchemy import text
import subprocess

dm = DataManager()

# Find assets without predictions
print("\n1. Finding assets without predictions...")

with dm.engine.connect() as conn:
    result = conn.execute(text("""
        SELECT a.ticker, a.asset_type,
               COALESCE((SELECT COUNT(*) FROM price_data p WHERE p.ticker = a.ticker), 0) as row_count
        FROM asset_metadata a
        WHERE a.ticker NOT IN (SELECT DISTINCT ticker FROM predictions)
        ORDER BY a.ticker
    """))

    missing = list(result)

if not missing:
    print("✅ All assets have predictions!")
    dm.close()
    sys.exit(0)

print(f"\n❌ Found {len(missing)} assets without predictions:")
print(f"{'Ticker':<15} {'Type':<10} {'Rows':<10}")
print("-" * 80)

assets_with_data = []
assets_without_data = []

for ticker, asset_type, row_count in missing:
    print(f"{ticker:<15} {asset_type or 'unknown':<10} {row_count:<10,}")

    if row_count > 0:
        assets_with_data.append(ticker)
    else:
        assets_without_data.append(ticker)

print(f"\n📊 Summary:")
print(f"   Assets with data: {len(assets_with_data)}")
print(f"   Assets without data: {len(assets_without_data)}")

if assets_without_data:
    print(f"\n⚠️  Assets without data (can't generate predictions):")
    for ticker in assets_without_data:
        print(f"   - {ticker}")

if assets_with_data:
    print(f"\n✅ Assets ready for predictions ({len(assets_with_data)}):")
    for ticker in assets_with_data:
        print(f"   - {ticker}")

    print(f"\n2. Generating predictions for all 40 assets...")
    print("-" * 80)

    # Run predict_all_assets.py which will generate predictions for ALL assets
    result = subprocess.run(
        [sys.executable, "predict_all_assets.py"],
        capture_output=True,
        text=True,
        timeout=600  # 10 minute timeout
    )

    print(result.stdout)

    if result.returncode != 0:
        print(f"\n❌ Prediction script failed with exit code {result.returncode}")
        if result.stderr:
            print(f"Error output:\n{result.stderr}")
        dm.close()
        sys.exit(1)

    # Check results
    print("\n3. Verifying predictions...")
    with dm.engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(DISTINCT ticker) FROM predictions"))
        pred_count = result.scalar()

        result = conn.execute(text("SELECT COUNT(*) FROM asset_metadata"))
        total_assets = result.scalar()

        print(f"   Total assets: {total_assets}")
        print(f"   Assets with predictions: {pred_count}")

        if pred_count == total_assets:
            print("   ✅ All assets now have predictions!")
        else:
            print(f"   ⚠️  Still missing predictions for {total_assets - pred_count} assets")

dm.close()

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
