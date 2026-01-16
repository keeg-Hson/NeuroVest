#!/usr/bin/env python3
"""
Find assets missing predictions without heavy dependencies
"""
import os
import sys

# Database URL from environment
db_url = os.environ.get('DATABASE_URL', os.environ.get('DATABASE_PRIVATE_URL'))

if not db_url:
    print("❌ No DATABASE_URL found in environment")
    sys.exit(1)

# Use psycopg2 directly
try:
    import psycopg2
except ImportError:
    print("❌ psycopg2 not available")
    sys.exit(1)

print("="*80)
print("FINDING ASSETS WITHOUT PREDICTIONS")
print("="*80)

conn = psycopg2.connect(db_url)
cur = conn.cursor()

# Get all assets
cur.execute("""
    SELECT ticker, asset_type
    FROM asset_metadata
    ORDER BY ticker
""")
all_assets = cur.fetchall()

print(f"\nTotal assets in database: {len(all_assets)}")

# Get assets with predictions
cur.execute("""
    SELECT DISTINCT ticker
    FROM predictions
""")
assets_with_preds = set(row[0] for row in cur.fetchall())

print(f"Assets with predictions: {len(assets_with_preds)}")

# Find missing
missing = []
for ticker, asset_type in all_assets:
    if ticker not in assets_with_preds:
        # Check if has data
        cur.execute("""
            SELECT COUNT(*) FROM price_data WHERE ticker = %s
        """, (ticker,))
        row_count = cur.fetchone()[0]
        missing.append({
            'ticker': ticker,
            'type': asset_type,
            'rows': row_count
        })

print(f"\n❌ MISSING PREDICTIONS: {len(missing)}")
print(f"\n{'Ticker':<15} {'Type':<10} {'Rows':<10} {'Status':<20}")
print("-"*80)

for asset in missing:
    status = "HAS DATA" if asset['rows'] > 0 else "NO DATA"
    print(f"{asset['ticker']:<15} {asset['type'] or 'unknown':<10} {asset['rows']:<10,} {status:<20}")

cur.close()
conn.close()

print("\n" + "="*80)
