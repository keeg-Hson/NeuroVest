#!/usr/bin/env python3
"""
Check which assets are missing predictions and why
"""
import sys
sys.path.insert(0, '.')
from core.data_manager_postgres import DataManager

dm = DataManager()

print("=" * 80)
print("MISSING PREDICTIONS ANALYSIS")
print("=" * 80)

# Get all assets
all_assets = dm.get_all_assets()
print(f"\nTotal assets in database: {len(all_assets)}")

# Get assets with predictions based on backend type
if dm.backend == 'postgresql':
    from sqlalchemy import text
    with dm.engine.connect() as conn:
        result = conn.execute(text("""
            SELECT DISTINCT ticker
            FROM predictions
            ORDER BY ticker
        """))
        assets_with_predictions = set(row[0] for row in result)
else:
    # SQLite backend
    conn = dm._get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT DISTINCT ticker
            FROM predictions
            ORDER BY ticker
        """)
        assets_with_predictions = set(row[0] for row in cursor.fetchall())
    except Exception:
        # Predictions table may not exist in SQLite
        print("⚠️  Predictions table not found in SQLite database")
        assets_with_predictions = set()

print(f"Assets with predictions: {len(assets_with_predictions)}")

# Find missing
missing_predictions = []
for ticker, asset_type in all_assets:
    if ticker not in assets_with_predictions:
        # Check if they have data
        data = dm.get_data(ticker)
        row_count = len(data) if data is not None else 0
        missing_predictions.append({
            'ticker': ticker,
            'asset_type': asset_type,
            'rows': row_count,
            'status': 'HAS DATA' if row_count > 0 else 'NO DATA'
        })

print(f"\n{'='*80}")
print(f"ASSETS MISSING PREDICTIONS: {len(missing_predictions)}")
print(f"{'='*80}\n")

if missing_predictions:
    print(f"{'Ticker':<15} {'Type':<10} {'Rows':<10} {'Status':<15}")
    print("-" * 80)
    for asset in missing_predictions:
        print(f"{asset['ticker']:<15} {asset['asset_type'] or 'stock':<10} {asset['rows']:<10} {asset['status']:<15}")

    # Categorize
    no_data = [a for a in missing_predictions if a['rows'] == 0]
    has_data = [a for a in missing_predictions if a['rows'] > 0]

    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  No data: {len(no_data)} assets")
    print(f"  Has data but no prediction: {len(has_data)} assets")
    print(f"{'='*80}")

    if has_data:
        print("\n⚠️  CRITICAL: These assets have data but no predictions!")
        print("   Run: python3 predict_all_assets.py --assets", ' '.join(a['ticker'] for a in has_data))
else:
    print("✅ All assets have predictions!")

dm.close()
