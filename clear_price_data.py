#!/usr/bin/env python3
"""
Clear all price data from database to allow full reload with maximum history

This script deletes all rows from price_data table while preserving
the table structure and predictions/models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_manager_postgres import DataManager
from sqlalchemy import text

def main():
    print("\n" + "="*70)
    print("⚠️  CLEARING ALL PRICE DATA FROM DATABASE")
    print("="*70)
    print("This will delete all historical price data but keep predictions/models")
    print("="*70 + "\n")

    dm = DataManager()

    if dm.backend != 'postgresql':
        print("❌ This script only works with PostgreSQL")
        return 1

    try:
        with dm.engine.begin() as conn:
            # Get current row count
            result = conn.execute(text("SELECT COUNT(*) FROM price_data"))
            before_count = result.fetchone()[0]
            print(f"📊 Current price_data rows: {before_count:,}")

            # Delete all price data
            print("\n🗑️  Deleting all price data...")
            conn.execute(text("DELETE FROM price_data"))

            # Reset asset metadata (last_update, last_timestamp, total_records)
            print("🔄 Resetting asset metadata...")
            conn.execute(text("""
                UPDATE asset_metadata
                SET last_update = NULL,
                    last_timestamp = NULL,
                    total_records = 0
            """))

            # Verify deletion
            result = conn.execute(text("SELECT COUNT(*) FROM price_data"))
            after_count = result.fetchone()[0]

            print(f"\n✅ Deleted {before_count:,} rows")
            print(f"📊 Remaining price_data rows: {after_count:,}")

            # Check predictions and models (should still be there)
            result = conn.execute(text("SELECT COUNT(*) FROM predictions"))
            pred_count = result.fetchone()[0]
            result = conn.execute(text("SELECT COUNT(*) FROM model_metadata"))
            model_count = result.fetchone()[0]

            print(f"\n📈 Predictions preserved: {pred_count:,} rows")
            print(f"🤖 Model metadata preserved: {model_count:,} rows")

        print("\n" + "="*70)
        print("✅ DATABASE CLEARED - Ready for full data reload")
        print("="*70)
        print("\nNext steps:")
        print("1. Railway will detect this push and redeploy DataWorker2")
        print("2. Bootstrap will run automatically and load MAX history")
        print("3. This time it will INSERT all 6000-8000 rows per asset")
        print("="*70 + "\n")

        dm.close()
        return 0

    except Exception as e:
        print(f"\n❌ Error clearing database: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
