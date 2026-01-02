#!/usr/bin/env python3
"""
Run database migration to create predictions tables

Usage:
    python migrations/run_migration.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_manager_postgres import DataManager
from sqlalchemy import text

def main():
    print("\n" + "="*70)
    print("🔄 RUNNING DATABASE MIGRATION")
    print("="*70)

    dm = DataManager()

    if dm.backend != 'postgresql':
        print("❌ Error: PostgreSQL required for this migration")
        print(f"   Current backend: {dm.backend}")
        sys.exit(1)

    # Read migration SQL
    migration_file = Path(__file__).parent / "001_create_predictions_tables.sql"
    print(f"\n📄 Reading: {migration_file.name}")

    with open(migration_file, 'r') as f:
        sql = f.read()

    # Execute migration
    print("\n⚙️  Executing migration...")
    try:
        with dm.engine.begin() as conn:
            # Split by semicolon and execute each statement
            statements = [s.strip() for s in sql.split(';') if s.strip()]

            for i, statement in enumerate(statements, 1):
                if statement:
                    print(f"   [{i}/{len(statements)}] Executing statement...")
                    conn.execute(text(statement))

        print("\n✅ Migration completed successfully!")

        # Verify tables created
        print("\n🔍 Verifying tables...")
        with dm.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('predictions', 'model_metadata')
                ORDER BY table_name
            """))
            tables = [row[0] for row in result]

            for table in ['predictions', 'model_metadata']:
                if table in tables:
                    print(f"   ✅ {table}")
                else:
                    print(f"   ❌ {table} NOT FOUND")

            # Check views
            result = conn.execute(text("""
                SELECT table_name FROM information_schema.views
                WHERE table_schema = 'public'
                AND table_name IN ('latest_predictions', 'latest_models')
                ORDER BY table_name
            """))
            views = [row[0] for row in result]

            for view in ['latest_predictions', 'latest_models']:
                if view in views:
                    print(f"   ✅ {view} (view)")
                else:
                    print(f"   ❌ {view} NOT FOUND")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*70)
    print("✅ DATABASE READY FOR PREDICTIONS")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
