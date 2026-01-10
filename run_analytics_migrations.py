#!/usr/bin/env python3
"""
Run analytics migrations on Railway PostgreSQL database
Creates users table (if needed) and request_logs table
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_manager_postgres import DataManager
from sqlalchemy import text
import os

def main():
    print("\n" + "="*70)
    print("🔄 RUNNING ANALYTICS MIGRATIONS")
    print("="*70)

    # Check DATABASE_URL
    db_url = os.getenv('DATABASE_URL', '')
    if not db_url:
        print("❌ Error: DATABASE_URL not set")
        sys.exit(1)

    print(f"\n✅ Connected to: {db_url.split('@')[1] if '@' in db_url else 'database'}")

    dm = DataManager()

    if dm.backend != 'postgresql':
        print("❌ Error: PostgreSQL required for this migration")
        print(f"   Current backend: {dm.backend}")
        sys.exit(1)

    print("\n📊 Database backend: PostgreSQL")

    try:
        with dm.engine.begin() as conn:
            # Step 1: Check if users table exists, create if not
            print("\n[1/3] Checking users table...")
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'users'
                )
            """))
            users_exists = result.scalar()

            if not users_exists:
                print("   Creating users table...")
                conn.execute(text("""
                    CREATE TABLE users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(100) UNIQUE NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        api_key VARCHAR(255) UNIQUE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                print("   ✅ Users table created")
            else:
                print("   ✅ Users table exists")

            # Step 2: Run add_user_tiers migration
            print("\n[2/3] Adding tier column to users...")
            tier_sql = Path(__file__).parent / "migrations" / "add_user_tiers.sql"

            with open(tier_sql, 'r') as f:
                sql = f.read()
                # Remove the SELECT statement (causes issues in transaction)
                sql = sql.split('-- Display current tier distribution')[0]
                statements = [s.strip() for s in sql.split(';') if s.strip() and not s.strip().startswith('--')]

                for statement in statements:
                    if statement and not statement.startswith('DO $$'):
                        conn.execute(text(statement))

            print("   ✅ User tiers migration complete")

            # Step 3: Run add_request_logging migration
            print("\n[3/3] Creating request_logs table...")
            logging_sql = Path(__file__).parent / "migrations" / "add_request_logging.sql"

            with open(logging_sql, 'r') as f:
                sql = f.read()
                statements = [s.strip() for s in sql.split(';') if s.strip() and not s.strip().startswith('--')]

                for statement in statements:
                    if statement and not statement.startswith('DO $$'):
                        conn.execute(text(statement))

            print("   ✅ Request logging migration complete")

        # Verify tables
        print("\n🔍 Verifying tables...")
        with dm.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('users', 'request_logs')
                ORDER BY table_name
            """))
            tables = [row[0] for row in result]

            for table in ['users', 'request_logs']:
                if table in tables:
                    # Get row count
                    count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = count_result.scalar()
                    print(f"   ✅ {table} ({count} rows)")
                else:
                    print(f"   ❌ {table} NOT FOUND")

        print("\n" + "="*70)
        print("✅ ANALYTICS MIGRATIONS COMPLETE")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
